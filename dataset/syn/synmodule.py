import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import lightning as pl
from torch.utils.data import Dataset, DataLoader, Subset, random_split

from dataset.syn.syn import SynDataset

class SynModule(pl.LightningDataModule):
    def __init__(self, dataset, path_dir = "dataset/syn/",
                 batch_size = 10000):
        super().__init__()
        self.batch_size = batch_size
        self.folder = path_dir + dataset
        self.dataset = SynDataset(dataset)
        self.prepare_data()
        self.mu, self.std, self.limits = torch.load(self.folder + "/predata.tensor")
        #print(self.mu, self.std, self.limits)

    def _split_dataset(self, data, splits):
        train, valid, test = random_split(data, splits)
        torch.save(train.indices, self.folder + '/train.indices')
        torch.save(valid.indices, self.folder + '/valid.indices')
        torch.save(test.indices, self.folder + '/test.indices')
        return train.indices

    def prepare_data(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            data = self.dataset.generate_data()
            #sns.kdeplot(data)
            #plt.show()
            torch.save(data, self.folder + "/data.pt")
            indices = self._split_dataset(data, self.dataset.splits)
            mu = data[indices].mean(dim = 0)
            std = data[indices].std(dim = 0)
            limits = torch.hstack((data[indices].min(dim = 0)[0].unsqueeze(1),
                                   data[indices].max(dim = 0)[0].unsqueeze(1)))
            torch.save([mu, std, limits], self.folder + "/predata.tensor")

    def setup(self, stage):
        data = torch.load(self.folder + "/data.pt")
        self.dataset.set_data(data)
        if stage == "fit":
            indice = torch.load(self.folder + '/train.indices')
            self.train = Subset(self.dataset, indice)
            indice = torch.load(self.folder + '/valid.indices')
            self.valid = Subset(self.dataset, indice)
        if stage == "test":
            indice = torch.load(self.folder + '/test.indices')
            self.test = Subset(self.dataset, indice)

    def train_dataloader(self):
        # do not believe the warning bullshit about increasing number_worker
        # It is really really slooooooooooooooooow
        return DataLoader(self.train, batch_size = self.batch_size,
                          shuffle =  False, pin_memory = True)#, num_workers = 15)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size = 10000,
                          shuffle = False, pin_memory = True)#, num_workers = 15)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size = 10000,
                          shuffle = False, pin_memory = True)#, num_workers = 15)
