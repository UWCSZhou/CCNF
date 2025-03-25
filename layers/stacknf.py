import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class StackNF(nn.Module):
    def __init__(self, mu, std, layers):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad = False)
        self.std = nn.Parameter(std, requires_grad = False)
        # the direction of layers is u-->x
        self.layers = nn.ModuleList(layers)

    def forward(self, u):
        x, logd_sum = u, 0.0
        for layer in self.layers:
            x, logd = layer.forward(x)
            logd_sum += logd
        x = x * self.std + self.mu
        return x, logd_sum + torch.log(self.std).sum()

    def reward(self, x):
        u, logd_sum = x.clone(), 0.0
        # remembter, type of logd_sum is torch.zeros(u.shape[0])
        u = (u - self.mu) / self.std
        for layer in reversed(self.layers):
            u, logd = layer.reward(u)
            logd_sum += logd
        return u, logd_sum - torch.log(self.std).sum()
