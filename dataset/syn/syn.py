import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch.distributions as dists
from torch.distributions.transformed_distribution import TransformedDistribution

from dataset.syn.transforms import trans_dict

class SynDataset(Dataset):
    def __init__(self, name, splits = [25000, 2500, 2500]):
        self.name = name
        self.splits = splits
        self.trans = trans_dict[self.name](sum(splits))
        self.num = self.trans.num
        self.dim = self.trans.dim
        self.dag = self.trans.dag
        self.dist = TransformedDistribution(self.trans.prior(), self.trans)

    def generate_data(self):
        return self.dist.sample((self.num,)).cpu()

    def set_data(self, data):
        self.data = data

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self.data[index]

    def log_prob(self, x):
        return self.dist.log_prob(x)


    '''
        if self.name == "four_node_chain":
            self.trans = FourNodeChainTransform(sum(splits))
        elif self.name == "nlin_simpson":
            self.trans = NlinSimpson(sum(splits))

    self.scaler = preprocessing.StandardScaler()
        print(data)
        if scaler:
            self.scaler.fit(self.data)
            self.transform(self.data)


    def transform(self, data):
        print(self.scaler.transform(data.numpy()))

    def inverse_transform(self, data):
         self.scaler.inverse_transform(data.numpy())

    '''
