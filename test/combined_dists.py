import numpy as np
import unittest
import torch
import torch.distributions as dists

from scm.dag import DAG
from dataset.syn import SynModule
from layers import MAFLayer, VACALayer, StackNF, CausalStackNF
from layers.causal_flow_plusplus import CausalFlowPlusPlusLayer
from plmodule import NormalizingFlowModule, VACAModule
from utils.combined_dists import CombinedDistributions

class CombinedDistributionsTest(unittest.TestCase):
    def setUp(self):
        self.dists = CombinedDistributions([dists.Normal(0, 1),
                                            dists.Normal(0, 1),
                                            dists.Normal(0, 1)])
        self.target_dists = dists.MultivariateNormal(torch.zeros(3), torch.eye(3))

    def test_sample(self):
        self.assertTrue(self.dists.sample(100).shape ==
                        self.target_dists.sample((100,)).shape)

    def test_log_prob(self):
        x = torch.Tensor([[1, 2, 3], [2, 3, 4], [4, 5, 6], [1, 2, 3]])
        self.assertTrue(torch.allclose(self.dists.log_prob(x),
                                       self.target_dists.log_prob(x),
                                       atol = 1e-6))


if __name__ == "__main__":
    unittest.main()
