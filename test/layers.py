import numpy as np
import unittest
import torch
import torch.distributions as dists

from scm.dag import DAG
from dataset.syn import SynModule
from layers import MAFLayer, VACALayer, StackNF, CausalStackNF
from layers.causal_flow_plusplus import CausalFlowPlusPlusLayer
from layers.causal_planar import CausalPlanarLayer
from plmodule import NormalizingFlowModule, VACAModule

class LayerTest(unittest.TestCase):
    def setUp(self):
        #self.datamodule = SynModule("nlin_simpson")
        #self.dataset = self.datamodule.dataset
        #self.priors = dists.MultivariateNormal(torch.zeros(self.dataset.dim,
        #                                                   device = "cuda:0"),
         #                                      torch.eye(self.dataset.dim,
         #                                                device = "cuda:0"))
        dag = DAG(np.array([[0, 0, 1, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0]]))
        #self.module = NormalizingFlowModule(self.priors, self.dataset.dist,
        #                                    CausalStackNF,
        #                                    [self.dataset.dag, 4,
        #                                     [16, 16], "af"])
        self.layer = CausalPlanarLayer(dag.start_nodes)

    def test_forward(self):
        x = torch.Tensor([[1, 2, -3, -4], [-5, -6, 7, 8], [-1, -2, -3, 5]])
        u, logd1 = self.layer.reward(x)
        x2, logd2 = self.layer.forward(u)
        #x, logd1 = self.module.flows(u)
        #u2, logd2 = self.module.flows.reward(x)
        print(x, "\n", x2, "\n====================\n")
        print(logd1, "\n", logd2, "\n====================\n")


if __name__ == "__main__":
    unittest.main()
