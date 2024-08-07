from torchvision import datasets, transforms
import yaml
from dataset.BaseDataset import BaseDataset
import numpy as np
import pickle

class CIFAR10(BaseDataset):
    def __init__(self, clients, iid_config, params):
        BaseDataset.__init__(self, iid_config)
        transformer = transforms.Compose([
            transforms.ToTensor(),
        ])
        # 获取数据集
        self.train_dataset = datasets.CIFAR10(root=self.path, train=True,
                                              transform=transformer, download=True)
        self.test_dataset = datasets.CIFAR10(root=self.path, train=False,
                                             transform=transformer, download=True)
        self.init(clients, self.train_dataset, self.test_dataset)


class CIFAR10_Semantic(BaseDataset):
    def __init__(self, clients, iid_config, params):
        BaseDataset.__init__(self, iid_config)
        transformer = transforms.Compose([
            transforms.ToTensor(),
        ])
        # 获取数据集
        self.train_dataset = datasets.CIFAR10(root=self.path, train=True,
                                              transform=transformer, download=True)
        self.test_dataset = datasets.CIFAR10(root=self.path, train=False,
                                             transform=transformer, download=True)
          
        # Indices of (samantically) poison data
        with open('../dataset/cifar10.yaml') as f:
            filter_indices = yaml.load(f, Loader=yaml.FullLoader)
        
        self.filter_indices = filter_indices["poison_images"]
        self.filter_indices_test = filter_indices["poison_images_test"]
        
        self.poison_train_index_list = np.array_split(self.filter_indices, clients["poison_client_num"])
        self.poison_test_index_list = self.filter_indices_test
        
        self.init(clients, self.train_dataset, self.test_dataset)
    

class CIFAR10_EdgeCase(BaseDataset):
    def __init__(self, clients, iid_config, params):
        BaseDataset.__init__(self, iid_config)
        transformer = transforms.Compose([
            transforms.ToTensor(),
        ])
        # 获取数据集
        self.train_dataset = datasets.CIFAR10(root=self.path, train=True,
                                              transform=transformer, download=True)
        self.test_dataset = datasets.CIFAR10(root=self.path, train=False,
                                             transform=transformer, download=True)
        self.init(clients, self.train_dataset, self.test_dataset)
        
        num_train = len(self.train_dataset.data)
        
        train_path = "../data/southwest_images_new_train.pkl"
        test_path = "../data/southwest_images_new_test.pkl"

        with open(train_path, 'rb') as f:
            poison_train_data = pickle.load(f)
        num_poison_train = len(poison_train_data)
            
        with open(test_path, 'rb') as f:
            poison_test_data = pickle.load(f)  
        num_poison_test = len(poison_test_data)
        
        poison_train_targets = 2 * np.ones((num_poison_train,), dtype =int) # southwest airplane -> label as bird
        poison_test_targets = 2 * np.ones((num_poison_test,), dtype =int) # southwest airplane -> label as bird

        self.train_dataset.data = np.concatenate((self.train_dataset.data, poison_train_data, poison_test_data), axis=0)
        self.train_dataset.targets = np.concatenate((self.train_dataset.targets, poison_train_targets, poison_test_targets), axis=0)
        
        self.filter_indices = np.arange(num_train, num_train+num_poison_train)
        self.filter_indices_test = np.arange(num_train+num_poison_train, num_train+num_poison_train+num_poison_test)
        
        self.poison_train_index_list = np.array_split(self.filter_indices, clients["poison_client_num"])
        self.poison_test_index_list = self.filter_indices_test
    
        self.init(clients, self.train_dataset, self.test_dataset)
        
        
    
    


    

        