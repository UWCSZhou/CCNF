import numpy as np
import unittest
import torch
import torch.distributions as dists
import seaborn as sns
from matplotlib import pyplot as plt

from scm.dag import DAG
from dataset.syn import SynModule
from dataset.syn.syn import SynDataset
from layers import MAFLayer, VACALayer, StackNF, CausalStackNF
from layers.causal_flow_plusplus import CausalFlowPlusPlusLayer
from layers.causal_planar import CausalPlanarLayer
from plmodule import NormalizingFlowModule, VACAModule

class GMMTest(unittest.TestCase):
    def setUp(self):
        normal = dists.Normal(torch.tensor(0), torch.tensor(1))
        print(normal.event_shape)
        self.dataset = SynDataset("gmm")

    def test_data(self):
        x = self.dataset.generate_data()
        sns.kdeplot(x)
        plt.show()


if __name__ == "__main__":
    unittest.main()
