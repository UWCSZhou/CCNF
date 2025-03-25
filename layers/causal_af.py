import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class CausalAFLayer(nn.Module):
    '''
    a single causal affine constant flow
    '''
    def __init__(self, nodes, slope = 1e-3):
        super().__init__()
        self.dim = len(nodes)
        self.nodes = nodes
        self.slope = slope
        self.un_s = nn.Parameter(torch.empty(self.dim))
        torch.nn.init.normal_(self.un_s, 0., 0.1)
        self.t = nn.Parameter(torch.empty(self.dim))
        torch.nn.init.normal_(self.t, 0., 0.1)

    def forward(self, u):
        x = u.clone()
        # this comes from zuko. I love them!
        logs = self.un_s / (1 + abs(self.un_s / np.log(self.slope)))
        x[:, self.nodes] = u[:, self.nodes] * torch.exp(logs) + self.t
        logd = torch.sum(logs).repeat(u.shape[0])
        return x, logd

    def reward(self, x):
        u = x.clone()
        # this comes from zuko. I love them!
        logs = self.un_s / (1 + abs(self.un_s / np.log(self.slope)))
        u[:, self.nodes] = (x[:, self.nodes] - self.t) * torch.exp(-logs)
        logd = torch.sum(-logs).repeat(x.shape[0])
        return u, logd
