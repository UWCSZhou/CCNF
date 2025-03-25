import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import lightning as pl

from layers.causal_af import CausalAFLayer
from layers.causal_nsf import CausalNSFLayer
from layers.causal_coupling import CausalCouplingLayer
from layers.causal_single_stacknf import CausalSingleStackNF
from layers.causal_flow_plusplus import CausalFlowPlusPlusLayer
from layers.causal_planar import CausalPlanarLayer
from layers.causal_gf import CausalGaussianizationLayer

class CausalStackNF(nn.Module):
    def __init__(self, dag, nums, hidden_layers, limits, mu = 0.0, std = 1.0,
                 init_layer = None):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.limits = limits
        self.mu = nn.Parameter(mu, requires_grad = False)
        self.std = nn.Parameter(std, requires_grad = False)
        self.dag = dag
        if len(nums) == 1:
            self.nums = nums * (self.dag.diam + 1)
        else:
            self.nums = nums
        if init_layer == None:
            init_layer = ["af"] + ["maf"] * self.dag.diam
        self.causal_levels = nn.ModuleList(self._create_level(init_layer))

    def _create_level(self, init_layer):
        if init_layer[0] == "af":
            levels = [CausalSingleStackNF([CausalAFLayer(self.dag.start_nodes)
                                           for _ in range(self.nums[0])])]
        elif init_layer[0] == "nsf":
            levels = [CausalSingleStackNF([CausalNSFLayer(self.dag.start_nodes, self.limits)
                                           for _ in range(self.nums[0])])]
        elif init_layer[0] == "planar":
            levels = [CausalSingleStackNF([CausalPlanarLayer(self.dag.start_nodes)
                                           for _ in range(self.nums[0])])]
        elif init_layer[0] == "gaussian":
            levels = [CausalSingleStackNF(
                [i for item in [[CausalGaussianizationLayer(self.dag.start_nodes, 100),
                                 CausalAFLayer(self.dag.start_nodes)]
                                for _ in range(self.nums[0])] for i in item])]
        for t, clayer, num in zip(init_layer[1:], self.dag.causal_layers, self.nums[1:]):
            if t == "maf":
                levels.append(CausalSingleStackNF(
                    [CausalCouplingLayer(clayer, self.hidden_layers)
                     for _ in range(num)], clayer))
            elif t == "flow++":
                levels.append(CausalSingleStackNF(
                    [CausalFlowPlusPlusLayer(clayer, self.hidden_layers)
                     for _ in range(num)], clayer))
        return levels

    def forward(self, u):
        # the direction of layers is u-->x
        x, logd_sum = u.clone(), 0.0
        for level in self.causal_levels:
            x, logd = level.forward(x)
            logd_sum += logd
        x = x * self.std + self.mu
        return x, logd_sum + torch.log(self.std).sum()

    def reward(self, x):
        u, logd_sum = x.clone(), 0.0 # remembter, logd_sum is torch.zeros(u.shape[0])
        u = (u - self.mu) / self.std
        for i, level in enumerate(reversed(self.causal_levels)):
            u, logd = level.reward(u)
            logd_sum += logd
        return u, logd_sum - torch.log(self.std).sum()
