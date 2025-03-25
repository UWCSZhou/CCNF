import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from layers.rqs.rqs import RQS

class NSFLayer(nn.Module):
    '''
    neural spine flow
    '''
    def __init__(self, dim, bins = 3, knots = 5):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(dim, knots).uniform_(-0.5, 0.5))
        self.h = nn.Parameter(torch.Tensor(dim, knots).uniform_(-0.5, 0.5))
        self.d = nn.Parameter(torch.Tensor(dim, knots - 1).uniform_(-0.5, 0.5))
        self.rqs = RQS(dim, bins, knots, self.w, self.h, self.d)

    def forward(self, u):
        x, logd = self.rqs.forward(u)
        return x, logd

    def reward(self, x):
        u, logd = self.rqs.reward(x)
        return u, logd
