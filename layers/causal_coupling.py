import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import lightning as pl

from layers.made.causal_made import CausalMaskedMLP

'''
The direction of our layer is always u --> x
'''

class CausalCouplingLayer(nn.Module):
    '''
    a single causal coupling layer, with a single causal layer
    '''
    def __init__(self, clayer, hidden_layers):
        super().__init__()
        self.made = CausalMaskedMLP(clayer, hidden_layers)
        self.clayer = clayer

    def forward(self, coupling, u):
        # u is [[1, 2, 3, ...], [4, 5, 6, ...]]
        x, logd = u.clone(), 0.0
        mu, log_sigma = self.made.get_mu_logsigma(self.made(coupling))
        x[:, self.clayer.ends] = torch.exp(log_sigma) * u[:, self.clayer.ends] + mu
        logd = torch.sum(log_sigma, dim = 1)
        return x, logd

    def reward(self, coupling, x):
        u, logd = x.clone(), 0.0
        #coupling = x[:, self.causal_layer.starts]
        mu, log_sigma = self.made.get_mu_logsigma(self.made(coupling))
        u[:, self.clayer.ends] = (x[:, self.clayer.ends] - mu) * torch.exp(-log_sigma)
        logd = torch.sum(-log_sigma, dim = 1)
        return u, logd