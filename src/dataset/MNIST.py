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
        
        
        
        x_train=np.loadtxt('../data/ARDIS/ARDIS_train_2828.csv', dtype='float')
        x_test=np.loadtxt('.../data/ARDIS/ARDIS_test_2828.csv', dtype='float')
        y_train=np.loadtxt('../data/ARDIS/ARDIS_train_labels.csv', dtype='float')
        y_test=np.loadtxt('../data/ARDIS/ARDIS_test_labels.csv', dtype='float')


        #### reshape to be [samples][pixels][width][height]
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')
        
        poison_train_data = x_train[y_train.argmax(1)==7] # y_train and y_test are stored as one-hot
        poison_test_data = x_test[y_test.argmax(1)==7]
        
        num_poison_train = len(poison_train_data)
        num_poison_test = len(poison_test_data)
        
        poison_train_targets = 1 * np.ones((num_poison_train,), dtype =int) # number 7 -> number 1
        poison_test_targets = 1 * np.ones((num_poison_test,), dtype =int) # number 7 -> number 1

        self.train_dataset.data = np.concatenate((self.train_dataset.data, poison_train_data, poison_test_data), axis=0)
        self.train_dataset.targets = np.concatenate((self.train_dataset.targets, poison_train_targets, poison_test_targets), axis=0)
        
        self.filter_indices = np.arange(num_train, num_train+num_poison_train)
        self.filter_indices_test = np.arange(num_train+num_poison_train, num_train+num_poison_train+num_poison_test)
        
        self.poison_train_index_list = np.array_split(self.filter_indices, clients["poison_client_num"])
        self.poison_test_index_list = self.filter_indices_test
    
        self.init(clients, self.train_dataset, self.test_dataset)