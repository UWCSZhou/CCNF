import numpy as np
import torch
import torch.nn as nn

class BNLayer(nn.Module):
    def __init__(self, dim, eps = 1e-5, momentum = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.running_m = nn.Parameter(torch.zeros(dim), requires_grad = False)
        self.running_v = nn.Parameter(torch.zeros(dim), requires_grad = False)

    def forward(self, u):
        if self.training:
            m = u.mean(dim = 0).detach()
            v = u.var(dim = 0).detach() + self.eps
            self.running_m.mul_(1 - self.momentum)
            self.running_m.add_(self.momentum * m)
            self.running_v.mul_(1 - self.momentum)
            self.running_v.add_(self.momentum * v)
        else:
            m = self.running_m
            v = self.running_v

        x_hat = (u - self.beta) * torch.exp(-self.gamma) * torch.sqrt(v) + m
        logd = torch.sum(-self.gamma + 0.5 * torch.sqrt(v)).repeat(u.shape[0])
        return x_hat, logd

    def reward(self, x):
        if self.training:
            m = x.mean(dim = 0).detach()
            v = x.var(dim = 0).detach() + self.eps
            self.running_m.mul_(1 - self.momentum)
            self.running_m.add_(self.momentum * m)
            self.running_v.mul_(1 - self.momentum)
            self.running_v.add_(self.momentum * v)
        else:
            m = self.running_m
            v = self.running_v

        u_hat = (x - m) / torch.sqrt(v) * torch.exp(self.gamma) + self.beta
        logd = torch.sum(self.gamma - 0.5 * torch.log(v)).repeat(x.shape[0])
        # torch.sum(self.gamma - 0.5 * torch.log(v)) works either;
        # Since 1 + tensor[1 2 3] equals tensor[2, 3, 4]
        return u_hat, logd