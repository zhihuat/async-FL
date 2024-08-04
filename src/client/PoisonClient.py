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
from utils.DatasetUtils import PoisonFLDataset

import numpy as np

class PoisonClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.group_id = 0
        self.trigger_config = config["trigger"]
        self.weight_scale = config["weight_scale"]
        
    def init_client(self):
        config = self.config
        self.train_ds = self.message_queue.get_train_dataset()

        self.transform, self.target_transform = self._get_transform(config)
        self.fl_train_ds = PoisonFLDataset(self.trigger_config, self.train_ds, list(self.index_list),
                                           self.transform, self.target_transform)

        self.model = self._get_model(config)
        self.model = self.model.to(self.dev)

        # optimizer
        opti_class = ModuleFindTool.find_class_by_path(self.optimizer_config["path"])
        self.opti = opti_class(self.model.parameters(), **self.optimizer_config["params"])

        # loss function
        self.loss_func = LossFactory(config["loss"], self).create_loss()

        self.train_dl = DataLoader(self.fl_train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
    
class PoisonClientConstrainScale(PoisonClient):
    """
    Implementation of 'How To Backdoor Federated Learning'
    """
        
    def train_one_epoch(self):
        global_model = copy.deepcopy(self.model.state_dict())
        
        data_sum, weights = super().train_one_epoch()
        for k in weights:
            weights[k] = (weights[k] - global_model[k]) * self.weight_scale + global_model[k]
        torch.cuda.empty_cache()
        return data_sum, weights
    
    
class PoisonClientPGD(PoisonClient):
    """
    Implementation of PGD attack.
    """
    def train_one_epoch(self):
        if self.mu != 0:
            global_model = copy.deepcopy(self.model)
        data_sum = 0
        l2norm = []
        for epoch in range(self.epoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.model(data)
                # Calculate the loss function
                loss = self.loss_func(preds, label)
                data_sum += label.size(0)
                # proximal term
                if self.mu != 0:
                    proximal_term = 0.0
                    for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    loss = loss + (self.mu / 2) * proximal_term
                
                # weights_vector = torch.cat([param.view(-1) for param in self.model.parameters()]).detach()
                # global_weights_vector = torch.cat([param.view(-1) for param in global_model.parameters()]).detach()
                # l2_norm = torch.norm(weights_vector-global_weights_vector, p=2)
                # l2norm.append(l2_norm.item())

                # backpropagate
                loss.backward()
                # Update the gradient
                self.opti.step()  
                
                # Weight projection
                for w in self.model.parameters():
                    w = w / self.weight_scale
                
                # Zero out the gradient and initialize the gradient.
                self.opti.zero_grad()
        # Return the updated model parameters obtained by training on the client's own data.
        weights = self.model.state_dict()
        for k in weights:
            weights[k] = (weights[k] - global_model[k]) * self.weight_scale + global_model[k]
        torch.cuda.empty_cache()
        return data_sum, weights