
import torch
from torch.utils.data import DataLoader, Sampler
import numpy as np
import random

class SemanticSampler(Sampler):
    def __init__(self, sem_indices, normal_indices, sem_batch_size, batch_size):
        self.sem_indices = sem_indices
        self.sem_batch_size = sem_batch_size
        
        self.normal_indices = normal_indices
        # self.batch_size = batch_size
        self.rem_batch_size = batch_size - sem_batch_size
        self.normal_num_samples = len(normal_indices)

    def __iter__(self):
        batches = []
        for _ in range(self.normal_num_samples // self.rem_batch_size):
            batch_A = np.random.choice(self.sem_indices, self.sem_batch_size)
            batch_B = np.random.choice(self.normal_indices, self.rem_batch_size)
            batch = np.append(batch_A, batch_B).tolist()
            # batches.append(batch)
            yield batch

    def __len__(self):
        return len(self.normal_indices) + len(self.sem_indices)