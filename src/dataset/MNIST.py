import torch
from torchvision import datasets, transforms

from dataset.BaseDataset import BaseDataset
import numpy as np

class MNIST(BaseDataset):
    def __init__(self, clients, iid_config, params):
        BaseDataset.__init__(self, iid_config)
        transformer = transforms.Compose([
            # 将图片转化为Tensor格式
            transforms.ToTensor()
        ])
        # 获取数据集
        self.train_dataset = datasets.MNIST(root=self.path, train=True,
                                            transform=transformer, download=True)
        self.test_dataset = datasets.MNIST(root=self.path, train=False,
                                           transform=transformer, download=True)
        self.init(clients, self.train_dataset, self.test_dataset)



class MNIST_EdgeCase(BaseDataset):
    def __init__(self, clients, iid_config, params):
        BaseDataset.__init__(self, iid_config)
        transformer = transforms.Compose([
            transforms.ToTensor(),
        ])
        # 获取数据集
        self.train_dataset = datasets.MNIST(root=self.path, train=True,
                                              transform=transformer, download=True)
        self.test_dataset = datasets.MNIST(root=self.path, train=False,
                                             transform=transformer, download=True)
        self.init(clients, self.train_dataset, self.test_dataset)
        
        num_train = len(self.train_dataset.data)
        
        
        
        x_train=np.loadtxt('../data/ARDIS/ARDIS_train_2828.csv', dtype='int8')
        x_test=np.loadtxt('../data/ARDIS/ARDIS_test_2828.csv', dtype='int8')
        y_train=np.loadtxt('../data/ARDIS/ARDIS_train_labels.csv', dtype='int8')
        y_test=np.loadtxt('../data/ARDIS/ARDIS_test_labels.csv', dtype='int8')


        #### reshape to be [samples][pixels][width][height]
        x_train = torch.from_numpy(x_train).type(torch.uint8).reshape(x_train.shape[0], 28, 28)
        x_test = torch.from_numpy(x_test).type(torch.uint8).reshape(x_test.shape[0], 28, 28)
        
        poison_train_data = x_train[y_train.argmax(1)==7] # y_train and y_test are stored as one-hot
        poison_test_data = x_test[y_test.argmax(1)==7]
        
        num_poison_train = len(poison_train_data)
        num_poison_test = len(poison_test_data)
        
        poison_train_targets = 1 * torch.ones((num_poison_train,), dtype =int) # number 7 -> number 1
        poison_test_targets = 1 * torch.ones((num_poison_test,), dtype =int) # number 7 -> number 1

        self.train_dataset.data = torch.cat((self.train_dataset.data, poison_train_data, poison_test_data), axis=0)
        self.train_dataset.targets = torch.cat((self.train_dataset.targets, poison_train_targets, poison_test_targets), axis=0)
        
        self.filter_indices = np.arange(num_train, num_train+num_poison_train)
        self.filter_indices_test = np.arange(num_train+num_poison_train, num_train+num_poison_train+num_poison_test)
        
        self.poison_train_index_list = np.array_split(self.filter_indices, clients["poison_client_num"])
        self.poison_test_index_list = self.filter_indices_test
    
        self.init(clients, self.train_dataset, self.test_dataset)