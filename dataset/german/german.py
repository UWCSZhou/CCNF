import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
import torch.distributions as dists
import lightning as pl
from torch.utils.data import Dataset, DataLoader, Subset, random_split

from scm.dag import DAG
from utils.combined_dists import CombinedDistributions

class GermanModule(pl.LightningDataModule):
    def __init__(self, noise = False, path_dir = "dataset/german/data", batch_size = 1000,
                 splits = [800, 100, 100], device = "cuda"):
        super().__init__()
        self.batch_size = batch_size
        self.splits = splits
        self.folder = path_dir
        self.noise = noise
        self.column_names = [
            "Age",
            "Sex",
            "Job",
            "Housing",
            "Saving accounts",
            "Checking account",
            "Credit amount",
            "Duration",
            "Risk"
        ]
        self.category = [1, 3, 4, 5, 8]
        self.dim = len(self.column_names)
        self.dist = None
        self.dag = DAG(np.array([[0, 0, 1, 0, 0, 0, 1, 1, 0], #age
                                 [0, 0, 1, 0, 0, 0, 1, 1, 0], #sex
                                 [0, 0, 0, 1, 1, 1, 0, 0, 1], #job
                                 [0, 0, 0, 0, 1, 1, 0, 0, 1], #housing
                                 [0, 0, 0, 0, 0, 1, 0, 0, 1], #saving
                                 [0, 0, 0, 0, 0, 0, 0, 0, 1], #checking
                                 [0, 0, 0, 0, 0, 0, 0, 1, 1], #credit
                                 [0, 0, 0, 0, 0, 0, 0, 0, 1], #duration
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0]]))#risk
        self.prepare_data()
        self.mu, self.std, self.limits = torch.load(self.folder + "/predata.tensor")

    def _split_dataset(self, data, splits):
        train, valid, test = random_split(data, splits)
        torch.save(train.indices, self.folder + '/train.indices')
        torch.save(valid.indices, self.folder + '/valid.indices')
        torch.save(test.indices, self.folder + '/test.indices')
        return train.indices

    def prepare_data(self):
        if not os.path.exists(self.folder + "/data.pt"):
            data = pd.read_csv(self.folder + '/german_credit_data.csv')
            data.drop(columns = ['Purpose'], axis = 1, inplace = True)
            self.data = torch.zeros(len(data), len(self.column_names))
            le = LabelEncoder()
            for i in range(9):
                name = self.column_names[i]
                if i in self.category:
                    self.data[:, i] = torch.from_numpy(le.fit_transform(data[name]))
                else:
                    self.data[:, i] = torch.from_numpy(data[name].values)
            torch.save(self.data, self.folder + "/data.pt")
            indices = self._split_dataset(self.data, self.splits)
            mu = self.data[indices].mean(dim = 0)
            std = self.data[indices].std(dim = 0)
            limits = torch.hstack((self.data[indices].min(dim = 0)[0].unsqueeze(1),
                                   self.data[indices].max(dim = 0)[0].unsqueeze(1)))
            torch.save([mu, std, limits], self.folder + "/predata.tensor")

    def add_noise(self, data):
        #add random number for category data
        #the method of flow++ maybe better, but we will not take it for now
        data[:, self.category] += \
            dists.Normal(0, 0.05).sample((len(data), len(self.category)))
        #torch.rand(len(data), len(self.category))
        return data

    def setup(self, stage):
        data = torch.load(self.folder + "/data.pt")
        if self.noise:
            data = self.add_noise(data)
        self.data = data
        if stage == "fit":
            indice = torch.load(self.folder + '/train.indices')
            self.train = Subset(self.data, indice)
            indice = torch.load(self.folder + '/valid.indices')
            self.valid = Subset(self.data, indice)
        if stage == "test":
            indice = torch.load(self.folder + '/test.indices')
            self.test = Subset(self.data, indice)

    def train_dataloader(self):
        # do not believe the warning bullshit about increasing number_worker
        # It is really really slooooooooooooooooow
        return DataLoader(self.train, batch_size = self.batch_size,
                          shuffle = False, pin_memory = True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size = 10000,
                          shuffle = False, pin_memory = True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size = 10000,
                          shuffle = False, pin_memory = True)
