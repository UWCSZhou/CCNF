import numpy as np
import torch
from torch.distributions.transforms import Transform, constraints
from torch.utils.data import Dataset
import torch.distributions as dists

from scm.dag import DAG

class NlinTriangleTransform(Transform):
    def __init__(self, num = 30000):
        super().__init__()
        self.dag = DAG(np.array([[0, 1, 1],
                                 [0, 0, 1],
                                 [0, 0, 0]]))
        self.num = num
        self.dim = 3
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
        x = torch.zeros_like(u)
        '''
        x[:, 0] = u[:, 0] + 1
        x[:, 1] = 2 * torch.pow(x[:, 0], 2) - \
            (torch.pow(x[:, 0], 2) + 1) * torch.pow(u[:, 1], 2)
        x[:, 2] = 20 / (1 + torch.exp(-torch.pow(x[:, 1], 2) + x[:, 0])) + u[:, 2]
        '''
        x0 = u[:, 0] + 1
        x1 = 2 * torch.pow(x0, 2) - self.softsign(x0 + u[:, 1])
        x2 = 20 / (1 + torch.exp(-torch.pow(x1, 2) + x0)) + u[:, 2]
        x[:, 0] = x0
        x[:, 1] = x1
        x[:, 2] = x2
        return x

    def _inverse(self, x):
        u = torch.zeros_like(x)
        u[:, 2] = x[:, 2] - 20 / (1 + torch.exp(-torch.pow(x[:, 1], 2) + x[:, 0]))
        u[:, 1] = self.inverse_softsign((2 * torch.pow(x[:, 0], 2) - x[:, 1])) - x[:, 0]
        u[:, 0] = x[:, 0] - 1
        return u

    # according to real_nvp: https://arxiv.org/abs/1605.08803
    # since dx_i / du_j = 0 (i < j)
    def log_abs_det_jacobian(self, u, x):
        return torch.log(abs(1 / torch.pow((1 + (x[:, 0] + u[:, 1]).abs()), 2)))\
                    .unsqueeze(1).to(u.device)
