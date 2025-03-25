import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from layers.made.made import MaskedMLP

class MAFLayer(nn.Module):
    '''
    A single MAF layer with MADE
    '''
    def __init__(self, dim, hidden_layer, order = None, adj = None):
        super().__init__()
        self.layers = [dim] + hidden_layer + [dim * 2]
        self.dim = dim
        self.order = [i for i in range(self.dim)] if order == None else order
        #self.add_module("mades", MaskedMLP(layers, self.order))
        self.mades = MaskedMLP(self.layers, self.order, adj)

    def forward(self, u):
        """
        reward can be one step
        but forward must be 1d by 1d
        """
        x = torch.zeros_like(u)
        logd = torch.zeros(u.shape[0], device = u.device)
        for i in range(self.dim):
            t = self.order[i]
            #outs = self.mades(x)
            mu, log_sigma = self.mades.get_mu_logsigma(self.mades(x))
            x[:, t] = u[:, t] * torch.exp(log_sigma[:, t]) + mu[:, t]
            logd += log_sigma[:, t]
        return x, logd

    def reward(self, x):
        mu, log_sigma = self.mades.get_mu_logsigma(self.mades(x))
        u = (x - mu) * torch.exp(-log_sigma)
        logd = torch.sum(-log_sigma, dim = 1)
        return u, logd
