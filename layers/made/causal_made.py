import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from layers.made.made import MaskedLinear

class CausalMaskedMLP(nn.Module):
    def __init__(self, clayer, hidden_layers, slope = 1e-3, outlen = 2):
        super().__init__()
        self.clayer = clayer
        self.slope = slope
        self.outlen = outlen
        self.layers = [len(clayer.starts)] + hidden_layers +  [outlen * len(clayer.ends)]
        self.masks = self._generate_masks()
        self.nets = nn.Sequential(*self._generate_nets())

    def _generate_masks(self):
        masks = []
        clayer, layers = self.clayer, self.layers
        s, n, e = len(clayer.starts), len(clayer.edges), len(clayer.ends)
        c = math.ceil(self.layers[1] / n)
        # starts ->hidden
        masks.append([[1 if clayer.starts[i] in clayer.edges[clayer.ends[j // c]] else 0
                       for i in range(layers[0])] for j in range(layers[1])])
        # hidden -> hidden
        for i in range(1, len(layers) - 2):
            masks.append([[1 if (i // c) == (j // c) else 0
                           for i in range(layers[i])] for j in range(layers[i + 1])])
        # hidden -> ends
        masks.append([[1 if (i // c) == (j % e) else 0
                       for i in range(layers[-2])] for j in range(layers[-1])])
        return masks

    def _generate_nets(self):
        nets = []
        for i in range(len(self.layers) - 1):
            nets.append(MaskedLinear(self.layers[i], self.layers[i + 1],
                                     torch.Tensor(self.masks[i])))
            nets.append(nn.ReLU())
        nets.pop()
        return nets

    def forward(self, x):
        return self.nets(x)

    def get_mu_logsigma(self, outs):
        mu, log_sigma = torch.split(outs, len(self.clayer.ends), dim = 1)
        # this comes from zuko. I love them!
        log_sigma = log_sigma / (1 + abs(log_sigma / np.log(self.slope)))
        return mu, log_sigma

    def get_flow_plus_plus(self, outs, k, loga_slope = None):
        e = len(self.clayer.ends)
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
