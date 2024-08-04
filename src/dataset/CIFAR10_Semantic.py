from torchvision import datasets, transforms

from dataset.BaseDataset import BaseDataset
import yaml

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
        self.init(clients, self.train_dataset, self.test_dataset)\
          
        all_images = set(range(len(self.train_dataset)))
        # Indices of (samantically) poison data
        with open('./cifar10.yaml') as f:
            self.exclude_indices = yaml.load(f, Loader=yaml.FullLoader)
            
        self.clean_indices = list(all_images.difference(set(self.exclude_indices)))
        
        