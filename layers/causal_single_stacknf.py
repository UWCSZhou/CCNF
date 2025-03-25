import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import lightning as pl

from layers.causal_af import CausalAFLayer
from layers.causal_coupling import CausalCouplingLayer

class CausalSingleStackNF(nn.Module):
    def __init__(self, flows, clayer = None):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.clayer = clayer
        if clayer:
            self.coupling_s = \
                nn.Parameter(torch.ones((len(flows), len(self.clayer.starts)),
                                        requires_grad = True))
            self.coupling_t = \
                nn.Parameter(torch.zeros((len(flows), len(self.clayer.starts)),
                                         requires_grad = True))

    def create_couplings(self, x):
        coupling = torch.zeros((len(self.flows), x.shape[0], len(self.clayer.starts)),
                               device = x.device)
        for i in range(len(self.flows)):
            coupling[i] = x[:, self.clayer.starts]#self.coupling_s[i] * x[:, self.clayer.starts] + self.coupling_t[i]
        #print(coupling)
        return coupling

    def forward(self, u):
        # the direction of layers is u-->x
        x, logd_sum = u.clone(), 0.0
        if self.clayer == None:
            for layer in self.flows:
                x, logd = layer.forward(x)
                logd_sum += logd
        else:
            couplings = self.create_couplings(u)
            for coupling, layer in zip(couplings, self.flows):
                x, logd = layer.forward(coupling, x)
                logd_sum += logd
        return x, logd_sum

    def reward(self, x):
        u, logd_sum = x.clone(), 0.0
        if self.clayer == None:
            for layer in reversed(self.flows):
                u, logd = layer.reward(u)
                logd_sum += logd
        else:
            couplings = self.create_couplings(x)
            for coupling, layer in zip(reversed(couplings), reversed(self.flows)):
                u, logd = layer.reward(coupling, u)
                logd_sum += logd
        return u, logd_sum
