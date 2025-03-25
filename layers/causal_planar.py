import numpy as np
import torch
import torch.nn as nn

# most code are from nomflow github. modified a little
"""Planar flow as introduced in [arXiv: 1505.05770](https://arxiv.org/abs/1505.05770)

    ```
        f(z) = z + u * h(w * z + b)
    ```
"""
class CausalPlanarLayer(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.nodes = nodes
        self.dim = len(self.nodes)
        lim_w = np.sqrt(2 / self.dim)
        lim_v = np.sqrt(2)
        self.w = nn.Parameter(torch.empty(self.dim).normal_(0, 0.1))
        self.v = nn.Parameter(torch.empty(self.dim).normal_(0, 0.1))
        self.b = nn.Parameter(torch.empty(1).normal_(0, 0.01))
        #nn.init.uniform_(self.w, -lim_w, lim_w)
        #nn.init.uniform_(self.v, -lim_v, lim_v)
        #nn.init.uniform_(self.b, -lim_b, lim_b)
        self.h = torch.nn.LeakyReLU(negative_slope = 0.1)

    def forward(self, u):
        x = u.clone()
        lin = torch.sum(self.w * u[:, self.nodes], dim = 1, keepdim = True) + self.b
        a = (lin < 0) * (self.h.negative_slope - 1.0) + 1.0 # absorb leakyReLU slope into u
        inner = torch.sum(self.w * self.v)
        v = self.v + (torch.log(1 + torch.exp(inner)) - 1 - inner) \
            * self.w / torch.sum(self.w ** 2)
        v = a * v
        inner_ = torch.sum(self.w * v, dim = 1, keepdim = True)
        x[:, self.nodes] = u[:, self.nodes] - v * (lin / (1 + inner_))
        logd = -torch.log(torch.abs(1 + inner_)).squeeze(1)
        return x, logd

    def reward(self, x):
        u = x.clone()
        lin = torch.sum(self.w * x[:, self.nodes], dim = 1, keepdim = True) + self.b
        inner = torch.sum(self.w * self.v)
        v = self.v + (torch.log(1 + torch.exp(inner)) - 1 - inner) \
            * self.w / torch.sum(self.w ** 2)  # constraint w.T * v > -1
        h_ = lambda x: (x < 0) * (self.h.negative_slope - 1.0) + 1.0
        u[:, self.nodes] = x[:, self.nodes] + v * self.h(lin)
        logd = torch.log(torch.abs(1 + torch.sum(self.w * v) * h_(lin.reshape(-1))))
        return u, logd
