import time
import numpy as np
import torch
import lightning as pl
import torch.nn.functional as F
from torch import nn
import torch.distributions as dists
from matplotlib import pyplot as plt
import seaborn as sns

from utils.metrics import kl_distance

class NormalizingFlowModule(pl.LightningModule):
    def __init__(self, priors, data_priors, model, args):
        super().__init__()
        self.save_hyperparameters()
        self.priors = priors
        self.data_priors = data_priors
        self.record = {
            "train_time" : [],
            "valid_time" : [],
            "kl" : 0
        }
        self.flows = model(*args)
        print(self.flows)

    def sample(self, size):
        u = self.priors.sample((size,))
        x, _ = self.flows.forward(u.clone())
        return u, x

    def training_step(self, batch, batch_idx):
        u, logd = self.flows.reward(batch) # or batch[0]? I don't know..
        log_prob = self.priors.log_prob(u)
        loss_sum = log_prob + logd
        loss = -loss_sum.mean()
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_start(self):
        self.sclk = time.time()

    def on_train_epoch_end(self):
        clk = time.time() - self.sclk
        self.record["train_time"].append(clk)
        self.log('train_time', clk)

    def validation_step(self, batch, batch_idx):
        u, logd = self.flows.reward(batch)
        #sns.kdeplot(u[0].cpu().detach().numpy())
        #plt.show()
        log_prob = self.priors.log_prob(u)
        loss_sum = log_prob + logd
        loss = -loss_sum.mean()
        print("\nval loss:", loss)
        self.log("val_loss", loss)

    def on_validation_epoch_start(self):
        self.sclk = time.time()

    def on_validation_epoch_end(self):
        clk = time.time() - self.sclk
        self.record["valid_time"].append(clk)
        self.log('valid_time', clk)

    def test_step(self, batch, batch_idx):
        u, logd1 = self.flows.reward(batch)
        gfd = sns.kdeplot(u.cpu())
        #gfd.legend(fontsize = 20)
        plt.show()
        if self.data_priors is not None:
            kl = kl_distance(self, batch)
            self.record["kl"] = kl.item()
            print(kl)
            return kl

    def configure_optimizers(self):
        # german 1e-4, other 1e-3
        return torch.optim.Adam(self.flows.parameters(), lr = 1e-3,
                                weight_decay = 1e-6)
