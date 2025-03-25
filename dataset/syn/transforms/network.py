import numpy as np
import torch
from torch.distributions.transforms import Transform, constraints
from torch.utils.data import Dataset
import torch.distributions as dists

from scm.dag import DAG

class NetworkTransform(Transform):
    def __init__(self, num = 30000):
        super().__init__()
        self.dag = DAG(np.array([[0, 1, 1, 0, 0, 0],
                                 [0, 0, 0, 1, 1, 0],
                                 [0, 0, 0, 1, 1, 0],
                                 [0, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0]]))
        self.num = num
        self.dim = 6
        self.codomain = constraints.real
        self.domain = constraints.real
        self.bijective = True
        self.sigmoid = dists.transforms.SigmoidTransform()
        self.softplus = dists.transforms.SoftplusTransform()

    def prior(self, device = "cuda"):
        return dists.MultivariateNormal(torch.zeros(self.dim, device = device),
                                        torch.eye(self.dim, device = device))

    def _call(self, u):
        x0 = u[:, 0]
        x1 = self.softplus(2 * x0 + u[:, 1] + 1)
        x2 = self.softplus(-1.5 * x0 + u[:, 2] - 1)
        x3 = self.softplus(2 * x1 - 2.5 * x2 + u[:, 3])
        x4 = self.softplus(0.5 * x1 - 3.5 * x2 + u[:, 4])
        x5 = 10 * (self.sigmoid(0.5 * x3 - 1.5 * x4 + u[:, 5]) - 0.5)
        return torch.stack((x0, x1, x2, x3, x4, x5), dim = 1)

    def _inverse(self, x):
        u = torch.zeros_like(x)
        u[:, 5] = self.sigmoid.inv(x[:, 5] / 10 + 0.5) - 0.5 * x[:, 3] + 1.5 * x[:, 4]
        u[:, 4] = self.softplus.inv(x[:, 4]) - 0.5 * x[:, 1] + 3.5 * x[:, 2]
        u[:, 3] = self.softplus.inv(x[:, 3]) - 2 * x[:, 1] + 2.5 * x[:, 2]
        u[:, 2] = self.softplus.inv(x[:, 2]) + 1.5 * x[:, 0] + 1
        u[:, 1] = self.softplus.inv(x[:, 1]) - 2 * x[:, 0] - 1
        u[:, 0] = x[:, 0]
        return u

    # according to real_nvp: https://arxiv.org/abs/1605.08803
    # since dx_i / du_j = 0 (i < j)
    def log_abs_det_jacobian(self, u, x):
        return (self.softplus.log_abs_det_jacobian(2 * x[:, 0] + u[:, 1] + 1,
                                                   x[:, 1]) +
                self.softplus.log_abs_det_jacobian(-1.5 * x[:, 0] + u[:, 2] - 1,
                                                   x[:, 2]) +
                self.softplus.log_abs_det_jacobian(2 * x[:, 1] - 2.5 * x[:, 2] + u[:, 3],
                                                   x[:, 3]) +
                self.softplus.log_abs_det_jacobian(0.5 * x[:, 1] - 3.5 * x[:, 2] + u[:, 4],
                                                   x[:, 4]) +
                np.log(10) +
                self.sigmoid.log_abs_det_jacobian(0.5 * x[:, 3] - 1.5 * x[:, 4] + u[:, 5],
                                                  x[:, 5] / 10 + 0.5)).unsqueeze(1)
