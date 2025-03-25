import numpy as np
import torch
import torch.nn as nn

# most code are from nomflow github. modified a little
class PlanarLayer(nn.Module):
    """Planar flow as introduced in [arXiv: 1505.05770](https://arxiv.org/abs/1505.05770)

    ```
        f(z) = z + u * h(w * z + b)
    ```
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        lim_w = np.sqrt(2.0 / dim)
        lim_u = np.sqrt(2)
        self.u = nn.Parameter(torch.empty(dim))
        nn.init.uniform_(self.u, -lim_u, lim_u)
        self.w = nn.Parameter(torch.empty(dim))
        nn.init.uniform_(self.w, -lim_w, lim_w)
        self.b = nn.Parameter(torch.zeros(1))
        self.h = torch.nn.LeakyReLU(negative_slope = 0.2)

    def forward(self, z):
        lin = torch.sum(self.w * z, dim = 1, keepdim = True) + self.b
        inner = torch.sum(self.w * self.u)
        u = self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) \
            * self.w / torch.sum(self.w ** 2)  # constraint w.T * u > -1
        h_ = lambda x: (x < 0) * (self.h.negative_slope - 1.0) + 1.0
        x = z + u * self.h(lin)
        logd = torch.log(torch.abs(1 + torch.sum(self.w * u) * h_(lin.reshape(-1))))
        return x, logd

    def reward(self, x):
        lin = torch.sum(self.w * x, dim = 1, keepdim = True) + self.b
        a = (lin < 0) * (self.h.negative_slope - 1.0) + 1.0 # absorb leakyReLU slope into u
        inner = torch.sum(self.w * self.u)
        u = self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) \
            * self.w / torch.sum(self.w ** 2)
        u = a * u
        inner_ = torch.sum(self.w * u, dim = 1, keepdim = True)
        z = x - u * (lin / (1 + inner_))
        logd = -torch.log(torch.abs(1 + inner_)).squeeze(1)
        return z, logd
