import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import lightning as pl

class CouplingLayer(nn.Module):
    '''
    single coupling layer of real NVP flow.
    '''
    def __init__(self, mask, hidden_dim = 256):
        super().__init__()
        self.dim = len(mask)
        self.mask = nn.Parameter(torch.Tensor(mask), requires_grad = False)
        self.unmask = nn.Parameter(1 - self.mask, requires_grad = False)
        self.s = nn.Sequential(nn.Linear(self.dim, hidden_dim), nn.LeakyReLU(),
                               nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
                               nn.Linear(hidden_dim, self.dim), nn.Tanh())
        self.t = nn.Sequential(nn.Linear(self.dim, hidden_dim), nn.LeakyReLU(),
                               nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
                               nn.Linear(hidden_dim, self.dim))

        u_m, u_um = self.mask * u, self.unmask * u
        logs, t = self.unmask * self.s(u_m), self.unmask * self.t(u_m)
        x = u_m + (u_um * torch.exp(logs) + t)
        logd = torch.sum(logs, dim = 1)
        return x, logd

    def reward(self, x):
        '''
        u <-- x
        '''
        x_m, x_um = self.mask * x, self.unmask * x
        logs, t = self.unmask * self.s(x_m), self.unmask * self.t(x_m)
        u = x_m + (x_um - t) * torch.exp(-logs)
        logd = torch.sum(-logs, dim = 1)
        return u, logd
