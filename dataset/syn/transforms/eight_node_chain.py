import numpy as np
import torch
from torch.distributions.transforms import Transform, constraints
from torch.utils.data import Dataset
import torch.distributions as dists

from scm.dag import DAG

class EightNodeChainTransform(Transform):
    def __init__(self, num = 30000):
        super().__init__()
        self.dag = DAG(np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0, 0, 0]]))
        self.num = num
        self.dim = 8
        self.codomain = constraints.real
        self.domain = constraints.real
        self.bijective = True
        self.softsign = torch.nn.functional.softsign

    def inverse_softsign(self, y):
        return torch.where(y > 0, y / (1 - y), y / (1 + y))

    def logd_softsign(self, x):
        return -torch.log(torch.pow(1 + x.abs(), 2))

    def prior(self, device = "cuda"):
        return dists.MultivariateNormal(torch.zeros(self.dim, device = device),
                                        torch.eye(self.dim, device = device))

    def _call(self, u):
        x0 = u[:, 0]
        x1 = self.softsign(x0 + u[:, 1]).exp()
        x2 = x1 + self.softsign(x1.exp() * u[:, 2])
        x3 = x2 + self.softsign(x2.exp() * u[:, 3])
        x4 = x3 + self.softsign(x3.exp() * u[:, 4])
        x5 = x4 + self.softsign(x4.exp() * u[:, 5])
        x6 = x5 + self.softsign(x5.exp() * u[:, 6])
        x7 = x6 + self.softsign(x6.exp() * u[:, 7])
        return torch.stack((x0, x1, x2, x3, x4, x5, x6, x7), dim = 1)

    def _inverse(self, x):
        u = torch.zeros_like(x)
        for i in range(2, 8):
            u[:, i] = self.inverse_softsign(x[:, i] - x[:, i - 1]) / x[:, i - 1].exp()
        u[:, 1] = self.inverse_softsign(x[:, 1].log()) - x[:, 0]
        u[:, 0] = x[:, 0]
        return u

    # according to real_nvp: https://arxiv.org/abs/1605.08803
    # since dx_i / du_j = 0 (i < j), it equals with abs(1 * -1 * -1.5 * 1)
    def log_abs_det_jacobian(self, u, x):
        res = torch.log(x[:, 1]) + self.logd_softsign(x[:, 0] + u[:, 1])
        for i in range(2, 8):
            tmp = self.logd_softsign(x[:, i - 1].exp() * u[:, i]) + x[:, i - 1]
            res += tmp
        return res.unsqueeze(1)
