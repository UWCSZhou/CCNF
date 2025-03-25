import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class MaskedLinear(nn.Linear):
    def __init__(self, nin, nout, mask):
        super().__init__(nin, nout)
        self.register_parameter("mask", nn.Parameter(mask, requires_grad = False))
        #self.mask = nn.Parameter(mask)
        #self.weight.data = self.weight.data * self.mask

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)

class MaskedMLP(nn.Module):
    def __init__(self, layers, order, adj = None, slope = 1e-3):
        super().__init__()
        self.layers = layers
        self.order = order
        self.outlen = self.layers[-1] // self.layers[0]
        self.slope = slope
        if adj is None:
            m = self._generate_random_state()
            self.masks = self._generate_masks(m)
        else:
            adj = torch.tensor(adj)
            self.masks = self._generate_masks_with_adj(adj)
        self.nets = nn.Sequential(*self._generate_nets())

    def _generate_random_state(self):
        m = [np.array([self.order.index(i) for i in range(len(self.order))])]
        r = np.random.RandomState()
        for i in range(1, len(self.layers) - 1):
            m.append(r.randint(min(m[i - 1]), self.layers[0], self.layers[i]))
        m.append(np.tile(m[0], 2))
        #print(m)
        return m

    def _generate_masks(self, m):
        masks = []
        #masks.append(torch.tensor([[1 if self.order.index(i) <= j else 0\
        #                            for i in range(len(self.order))] for j in self.m[1]]))
        #print(masks)
        for l0, l1 in zip(m[:-2], m[1:]):
            masks.append(torch.tensor([[1 if i <= j else 0 for i in l0] for j in l1]))
        masks.append(torch.tensor([[1 if i < j else 0 for i in m[-2]] \
                                   for j in m[-1]]))
        #print(masks)
        return masks

    # to make fair comparation, this code borrows from causal-nf
    # this code is further borrowed from zuko
    # actually I am not well understood this code
    # But fuck it, it works
    # Take a look at: https://github.com/probabilists/zuko/discussions/30
    def _generate_masks_with_adj(self, adj):
        adjacency, inverse = torch.unique(adj, dim = 0, return_inverse = True)
        # P_ij = 1 if A_ik = 1 for all k such that A_jk = 1
        precedence = adjacency @ adjacency.T == adjacency.sum(dim = -1)
        masks = []
        for i, features in enumerate(self.layers[1:]):
            if i > 0:
                mask = precedence[:, indices]
            else:
                mask = adjacency
            if i < len(self.layers) - 2:
                reachable = mask.sum(dim = -1).nonzero().squeeze(dim = -1)
                indices = reachable[torch.arange(features) % len(reachable)]
                mask = mask[indices]
            else:
                mask = mask[inverse.repeat(self.outlen)]
            masks.append(mask)
        return masks

    def _generate_nets(self):
        nets = []
        for i in range(len(self.layers) - 1):
            nets.append(MaskedLinear(self.layers[i], self.layers[i + 1],
                                     self.masks[i]))
            nets.append(nn.ReLU())
        nets.pop()
        return nets

    def forward(self, x):
        return self.nets(x)

    def get_mu_logsigma(self, outs):
        mu, log_sigma = torch.split(outs, self.layers[0], dim = 1)
        # this comes from zuko. I love them!
        log_sigma = log_sigma / (1 + abs(log_sigma / np.log(self.slope)))
        return mu, log_sigma

    def get_flow_plus_plus(self, outs, k, loga_slope = None):
        e = self.layers[0]
        log_a, b, un_pi, mu, log_sigma = torch.split(outs,
                                                     [e, e, k * e, k * e, k * e], dim = 1)
        log_pi = torch.zeros_like(un_pi)
        for i in range(e):
            indices = [i + j * e for j in range(k)]
            log_pi[:, indices] = torch.log_softmax(un_pi[:, indices], dim = 1)
        # self.slope is also too large for log_a
        if loga_slope is None:
            loga_slope = self.slope
        log_a = log_a / (1 + abs(log_a / np.log(loga_slope)))
        log_sigma = abs(log_sigma) / (1 + abs(log_sigma / np.log(loga_slope)))
        return log_a, b, log_pi, mu, log_sigma