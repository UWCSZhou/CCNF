import numpy as np
import torch
from torch.distributions.transforms import Transform, constraints
from torch.utils.data import Dataset
import torch.distributions as dists

from scm.dag import DAG

class FourNodeChainTransform(Transform):
    def __init__(self, num = 30000):
        super().__init__()
        self.dag = DAG(np.array([[0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 0, 0]]))
        self.num = num
        self.dim = 4
        self.codomain = constraints.real
        self.domain = constraints.real
        self.bijective = True

    def prior(self, device = "cuda"):
        return dists.MultivariateNormal(torch.zeros(self.dim, device = device),
                                        torch.eye(self.dim, device = device))

    def _call(self, u):
        x = torch.zeros_like(u)
        x[:, 0] = u[:, 0]
        x[:, 1] = 5 * x[:, 0] - u[:, 1]
        x[:, 2] = -0.5 * x[:, 1] - 1.5 * u[:, 2]
        x[:, 3] = x[:, 2] + u[:,  3]
        return x

    def _inverse(self, x):
        u = torch.zeros_like(x)
        u[:, 3] = x[:, 3] - x[:, 2]
        u[:, 2] = (0.5 * x[:, 1] + x[:, 2]) / -1.5
        u[:, 1] = 5 * x[:, 0] - x[:, 1]
        u[:, 0] = x[:, 0]
        return u

    # according to real_nvp: https://arxiv.org/abs/1605.08803
    # since dx_i / du_j = 0 (i < j), it equals with abs(1 * -1 * -1.5 * 1)
    def log_abs_det_jacobian(self, u, x):
        return torch.log(torch.tensor(1.5)).repeat(u.shape[0], 1).to(u.device)
