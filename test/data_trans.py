import unittest
import torch
import torch.distributions as dists
import seaborn as sns
import matplotlib.pyplot as plt

from dataset.syn.transforms.nlin_simpson import NlinSimpsonTransform
from dataset.syn.transforms.nlin_triangle import NlinTriangleTransform
from dataset.syn.transforms.nlin_triangle import NlinTriangleTransform
from dataset.syn.transforms.m_graph import MGraphTransform
from dataset.syn.transforms.network import NetworkTransform
from dataset.syn.transforms.backdoor import BackdoorTransform
from dataset.syn.transforms.eight_node_chain import EightNodeChainTransform
from dataset.syn.transforms.weak_arrow import WeakArrowTransform

class DataTransformTest(unittest.TestCase):
    def setUp(self):
        self.trans = WeakArrowTransform()

    def test_bidirectional(self):
        #u = self.trans.prior().sample((1000,))
        u = torch.Tensor([[5, 5, 5, 5, 5, 5, 5, 5, 5],
                          [-5, -5, -5, -5, -5, -5, -5, -5, -5],
                          [5, -5, 5, -5, 5, -5, 5, -5, 5],
                          [-5, 5, -5, 5, -5, 5, -5, 5, -5],
                          [5, 5, 5, -5, -5, -5, 5, -5, 5]])
        x = self.trans._call(u)
        u2 = self.trans._inverse(x)
        print(u, x, u2)
        #sns.kdeplot(x.cpu())
        #plt.show()
        return self.assertTrue(torch.allclose(u, u2, atol = 1e-6))

    def test_logd(self):
        torch.autograd.set_detect_anomaly(True)
        u = self.trans.prior().sample((1,))
        x = self.trans._call(u)
        logd1 = self.trans.log_abs_det_jacobian(u, x)
        jaco = torch.autograd.functional.jacobian(self.trans._call, u).view(8, 8)
        logd2 = torch.log(torch.det(jaco).abs())
        print(jaco)
        print(logd1, logd2)
        return self.assertTrue(torch.allclose(logd1[0][0], logd2, atol = 1e-4))


if __name__ == "__main__":
    unittest.main()
