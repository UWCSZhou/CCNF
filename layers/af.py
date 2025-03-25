import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class AFLayer(nn.Module):
    '''
    a single affine constant flow
    '''
    def __init__(self, dim, slope = 1e-3):
        super().__init__()
        un_s = torch.empty(self.dim)
        torch.nn.init.normal_(un_s)
        # this comes from zuko. I love them!
        self.logs = nn.Parameter(un_s / (1 + abs(un_s / np.log(slope))))
        self.t = nn.Parameter(torch.randn(dim, requires_grad = True))

    def forward(self, u):
        x = u * torch.exp(self.logs) + self.t
        logd = torch.sum(self.logs).repeat(u.shape[0])
        return x, logd

    def reward(self, x):
        u = (x - self.t) * torch.exp(-self.logs)
        logd = torch.sum(-self.logs).repeat(x.shape[0])
        return u, logd
