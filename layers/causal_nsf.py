import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from layers.rqs.rqs import RQS

class CausalNSFLayer(nn.Module):
    '''
    causal neural spine flow
    '''
    def __init__(self, nodes, limits):
        super().__init__()
        self.nodes = nodes
        bins = 2 * torch.ceil(limits[nodes].abs().max()).int()
        knots = bins * 4
        dim = len(self.nodes)
        self.w = nn.Parameter(torch.Tensor(dim, knots).uniform_(-0.5, 0.5))
        self.h = nn.Parameter(torch.Tensor(dim, knots).uniform_(-0.5, 0.5))
        self.d = nn.Parameter(torch.Tensor(dim, knots - 1).uniform_(-0.5, 0.5))
        self.rqs = RQS(dim, bins, knots, self.w, self.h, self.d)

    def forward(self, u):
        x = u.clone()
        x[:, self.nodes], logd = self.rqs.forward(u[:, self.nodes])
        return x, logd

    def reward(self, x):
        u = x.clone()
        u[:, self.nodes], logd = self.rqs.reward(x[:, self.nodes])
        return u, logd
