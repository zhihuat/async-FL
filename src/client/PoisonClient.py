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


class PoisonClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.group_id = 0
        self.trigger_config = config["trigger"]
        
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
        # # clip_rate = (helper.params['scale_weights'] / current_number_of_adversaries)
        # global_model = copy.deepcopy(self.model.state_dict())
        # upload_weight = copy.deepcopy(global_model)
        # data_sum, weights = super().train_one_epoch()
        # for key, value in weights:

        #     upload_weight[key] = global_model[key] + (value - global_model[key]) * clip_rate
        # torch.cuda.empty_cache()
        global_model = copy.deepcopy(self.model.state_dict())
        
        data_sum, weights = super().train_one_epoch()
        for k in weights:
            weights[k] = (weights[k] - global_model[k]) * 2 + global_model[k]
        torch.cuda.empty_cache()
        return data_sum, weights