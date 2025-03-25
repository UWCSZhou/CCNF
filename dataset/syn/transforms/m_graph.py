import numpy as np
import torch
from torch.distributions.transforms import Transform, constraints
from torch.utils.data import Dataset
import torch.distributions as dists

from scm.dag import DAG

class MGraphTransform(Transform):
    def __init__(self, num = 30000):
        super().__init__()
        self.dag = DAG(np.array([[0, 0, 1, 1, 0],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0]]))
        self.num = num
        self.dim = 5
        self.codomain = constraints.real
        self.domain = constraints.real
        self.bijective = True
        self.softsign = torch.nn.functional.softsign

    def inverse_softsign(self, y):
        return torch.where(y > 0, y / (1 - y), y / (1 + y))

    def prior(self, device = "cuda"):
        return dists.MultivariateNormal(torch.zeros(self.dim, device = device),
                                        torch.eye(self.dim, device = device))

    def _call(self, u):
        x0 = u[:, 0]
        x1 = u[:, 1]
        x2 = self.softsign(torch.exp(x0) + u[:, 2])
        x3 = (torch.pow(x1, 2) + 0.5 * torch.pow(x0, 2) + 1) * u[:, 3]
        x4 = (-1.5 * torch.pow(x1, 2) - 1) * u[:, 4]
        return torch.stack((x0, x1, x2, x3, x4), dim = 1)

    def _inverse(self, x):
        u = torch.zeros_like(x)
        u[:, 4] = x[:, 4] / (-1.5 * torch.pow(x[:, 1], 2) - 1)
        u[:, 3] = x[:, 3] / (torch.pow(x[:, 1], 2) + 0.5 * torch.pow(x[:, 0], 2) + 1)
        u[:, 2] = self.inverse_softsign(x[:, 2]) - torch.exp(x[:, 0])
        u[:, 1] = x[:, 1]
        u[:, 0] = x[:, 0]
        return u

    # according to real_nvp: https://arxiv.org/abs/1605.08803
    # since dx_i / du_j = 0 (i < j)
    def log_abs_det_jacobian(self, u, x):
        return (-torch.log(torch.pow(1 + (torch.exp(x[:, 0]) + u[:, 2]).abs(), 2)) +
                torch.log(torch.pow(x[:, 1], 2) + \
                          0.5 * torch.pow(x[:, 0], 2) + 1) +
                torch.log(1.5 * torch.pow(x[:, 1], 2) + 1)).unsqueeze(1)
