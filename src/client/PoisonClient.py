import copy
import time

import torch
from torch.utils.data import DataLoader

from client.Client import Client
from loss.LossFactory import LossFactory
from utils import ModuleFindTool
from utils.DatasetUtils import FLDataset
from utils.Tools import to_cpu
from client.NormalClient import NormalClient
from utils.DatasetUtils import PoisonFLDataset, SemanticPoisonFLDataset
from utils import ModuleFindTool

import numpy as np

class PoisonClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, poison_index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.group_id = 0
        self.trigger_config = config["trigger"]
        self.weight_scale = config["weight_scale"]
        self.projection_norm = config["projection_norm"] / self.weight_scale
        self.warm_up = config["warm_up"]
        
        self.poison_index_list = poison_index_list
        
    def init_client(self):
        config = self.config
        self.train_ds = self.message_queue.get_train_dataset()
        self.transform, self.target_transform = self._get_transform(config)
        if len(self.poison_index_list) > 0:
            self.fl_train_ds_poison = SemanticPoisonFLDataset(self.trigger_config, self.train_ds, list(self.index_list), 
                                                       list(self.poison_index_list), self.transform, self.target_transform)
            Sampler = ModuleFindTool.find_class_by_path(self.config["sampler"]["path"])
            sampler = Sampler(self.poison_index_list, self.index_list, self.config["sampler"]["sem_size"], self.batch_size)
            batch_size = None
            shuffle = False
        else:
            self.fl_train_ds_poison = PoisonFLDataset(self.trigger_config, self.train_ds, list(self.index_list),
                                            self.transform, self.target_transform)
            sampler = None
            batch_size = self.batch_size
            shuffle = True
        
        if self.warm_up > 0:
            self.fl_train_ds_clean = FLDataset(self.train_ds, list(self.index_list), self.transform, self.target_transform)
            self.train_dl_clean = DataLoader(self.fl_train_ds_clean, batch_size=batch_size, shuffle=True)
            

        self.model = self._get_model(config)
        self.model = self.model.to(self.dev)

        # optimizer
        opti_class = ModuleFindTool.find_class_by_path(self.optimizer_config["path"])
        self.opti = opti_class(self.model.parameters(), **self.optimizer_config["params"])

        # loss function
        self.loss_func = LossFactory(config["loss"], self).create_loss()
        self.train_dl_poision = DataLoader(self.fl_train_ds_poison, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
        
        
    def test_poison(self):
        self.poison_test_data = self.message_queue.get_poison_test_dataset()
        self.test_data = self.message_queue.get_test_dataset()
        trigger_config = copy.copy(self.trigger_config)
        trigger_config['poisoned_rate'] = 1.
        if self.poison_test_data is not None:
            test_ds = SemanticPoisonFLDataset(trigger_config, self.poison_test_data, [], np.arange(len(self.poison_test_data)).tolist())
        else:
            test_ds = PoisonFLDataset(trigger_config, self.test_data, np.arange(len(self.test_data)))
        self.poison_test_dl = DataLoader(test_ds, batch_size=100, shuffle=False, drop_last=False)
        correct = 0
        data_sum = 0
        loss = 0
        with torch.no_grad():
            for data, label in self.poison_test_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.model(data)
                correct += preds.argmax(1).eq(label).sum()
                loss += self.loss_func(preds, label).detach().item()
                data_sum += label.size(0)
            accuracy = correct / len(test_ds)
            loss = loss / len(test_ds)
        print(f"Client {self.client_id}, Poison Test Accuracy: {accuracy}, Poison Test Loss: {loss}")
        
    
class PoisonClientConstrainScale(PoisonClient):
    """
    Implementation of 'How To Backdoor Federated Learning'
    """
        
    def train_one_epoch(self):
        global_model = copy.deepcopy(self.model.state_dict())
        if self.time_stamp < self.warm_up:
            self.train_dl  = self.train_dl_clean
        else: 
            self.train_dl  = self.train_dl_poision
        
        data_sum, weights = super().train_one_epoch()
        # self.test_poison()
        
        for k in weights:
            weights[k] = (weights[k] - global_model[k]) * self.weight_scale + global_model[k]
        torch.cuda.empty_cache()
        return data_sum, weights
    
    
class PoisonClientPGD(PoisonClient):
    """
    Implementation of PGD attack.
    """
    def train_one_epoch(self):
        if self.time_stamp < self.warm_up:
            self.train_dl  = self.train_dl_clean
        else: 
            self.train_dl  = self.train_dl_poision
            
        global_model = copy.deepcopy(self.model)
        target_params_variables = dict()
        for name, param in global_model.state_dict().items():
            target_params_variables[name] = param.clone()
        
        data_sum = 0
        for epoch in range(self.epoch):
            for data, label in self.train_dl:
                self.opti.zero_grad()
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.model(data)
                # Calculate the loss function
                loss = self.loss_func(preds, label)
                data_sum += label.size(0)
                # proximal term
                proximal_term = self._model_dist_norm(global_model)
                if self.mu != 0:
                    loss = loss + (self.mu / 2) * proximal_term
                
                loss.backward()
                self.opti.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                
                # Weight projection
                self._projection(global_model)
        # Return the updated model parameters obtained by training on the client's own data.
        weights = self.model.state_dict()
        for k in weights:
            weights[k] = (weights[k] - target_params_variables[k]) * self.weight_scale + target_params_variables[k]
        # torch.save(weights, 'model.pth')
        self.test_poison()
        torch.cuda.empty_cache()
        return data_sum, weights
    

        
    