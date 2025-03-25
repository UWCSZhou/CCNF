import time
import numpy as np
import torch
import lightning as pl
import torch.nn.functional as F
import torch.distributions as dists
from torch import nn
from matplotlib import pyplot as plt
import seaborn as sns

from layers.VACA import VACALayer

class VACAModule(pl.LightningModule):
    def __init__(self, model, dim, edge_index, num_enc_layers, num_dec_layers,
                 hidden_dim_of_z, hidden_enc_channels, hidden_dec_channels,
                 dropout = 0.0, pre_layers = 1, post_layers = 1,
                 mu = 0.0, std = 1.0, device = "cuda"):
        super().__init__()
        self.record = {
            "train_time" : [],
            "valid_time" : [],
            "kl" : 0
        }
        self.priors = dists.Normal(torch.zeros(hidden_dim_of_z).to(device),
                                   torch.ones(hidden_dim_of_z).to(device))
        self.model = VACALayer(model, self.priors, dim, edge_index,
                               num_enc_layers, num_dec_layers,
                               hidden_dim_of_z, hidden_enc_channels, hidden_dec_channels,
                               dropout, pre_layers, post_layers, mu, std)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        #print("\ntrain loss:", loss)
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_start(self):
        self.sclk = time.time()

    def on_train_epoch_end(self):
        clk = time.time() - self.sclk
        self.record["train_time"].append(clk)
        self.log('train_time', clk)

    def validation_step(self, batch, batch_idx):
        loss = self.model(batch)
        print("\nval loss:", loss)
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_start(self):
        self.sclk = time.time()

    def on_validation_epoch_end(self):
        clk = time.time() - self.sclk
        self.record["valid_time"].append(clk)
        self.log('valid_time', clk)

    def test_step(self, batch, batch_idx):
        qz_x = self.model(batch, get_qz_x = True)
        x = self.model.forward_with_priors(self.priors, len(batch))[1]
        sns.kdeplot(x.cpu().detach().numpy())
        sns.kdeplot(batch.cpu().detach().numpy())
        plt.show()
        kl = dists.kl.kl_divergence(qz_x, self.priors).flatten(1).sum(1).mean()
        self.record["kl"] = kl.item()
        print("kl:", kl)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr = 1e-4,
                                weight_decay = 1e-6)
