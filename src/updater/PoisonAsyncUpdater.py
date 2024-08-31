import torch
import numpy as np
from numgenerator.NumGeneratorFactory import NumGeneratorFactory
from updater.AsyncUpdater import AsyncUpdater
from torch.utils.data import DataLoader
import wandb
from utils.DatasetUtils import PoisonFLDataset, SemanticPoisonFLDataset

class PoisonAsyncUpdater(AsyncUpdater):
    """
    Add methods "run_poison_server_test" and "get_poison_accuracy_and_loss_list"
    """
    def __init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem):
        AsyncUpdater.__init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem)
        self.trigger_config = self.config["trigger"]
        self.poison_accuracy_list = []
        self.poison_loss_list = []
        
        self.poison_test_data = self.message_queue.get_poison_test_dataset()
        
    def server_update(self, epoch, update_list):
        self.update_server_weights(epoch, update_list)
        self.run_server_test(epoch)
        self.run_poison_server_test(epoch)
    
    def run_poison_server_test(self, epoch):
        if self.poison_test_data is not None:
            test_ds = SemanticPoisonFLDataset(self.trigger_config, self.poison_test_data, [], np.arange(len(self.poison_test_data)).tolist())
        else:
            test_ds = PoisonFLDataset(self.trigger_config, self.test_data, np.arange(len(self.test_data)))
        dl = DataLoader(test_ds, batch_size=100, shuffle=False, drop_last=False)
        
        correct = 0
        data_num = 0
        loss = 0
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad():
            for data, label in dl:
                data, label = data.to(dev), label.to(dev)
                preds = self.server_network(data)
                correct += preds.argmax(1).eq(label).sum().item()
                loss += self.loss_func(preds, label).detach().item()
                data_num += label.size(0)
            accuracy = correct / data_num
            loss = loss / data_num
            self.poison_loss_list.append(loss)
            self.poison_accuracy_list.append(accuracy)
            print('Epoch(t):', epoch, 'poison accuracy:', accuracy, 'poison loss', loss)
            if self.config['enabled']:
                wandb.log({'poison accuracy': accuracy, 'poison loss': loss})
        return accuracy, loss

    def get_poison_accuracy_and_loss_list(self):
        return self.poison_accuracy_list, self.poison_loss_list