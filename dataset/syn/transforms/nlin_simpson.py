import numpy as np
import torch
from torch.distributions.transforms import Transform, constraints
from torch.utils.data import Dataset
import torch.distributions as dists

from scm.dag import DAG
from utils.combined_dists import CombinedDistributions

class NlinSimpsonTransform(Transform):
    def __init__(self, num = 30000):
        super().__init__()
        self.dag = DAG(np.array([[0, 1, 1, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 0, 0]]))
        self.num = num
        self.dim = 4
        self.codomain = constraints.real
        self.domain = constraints.real
        self.bijective = True
        self.s = torch.nn.Softplus()
        # The origin paper use tanh, However tanh is hard to reverse:
        # tanh x != atanh x in some case
        # therefore we choose softsign instead
        self.softsign = torch.nn.functional.softsign

    def prior(self, device = "cuda"):
        return dists.MultivariateNormal(torch.zeros(self.dim, device = device),
                                        torch.eye(self.dim, device = device))
    '''
        mix = dists.Categorical(torch.ones(2, device = device))
        comp = dists.Normal(torch.tensor([-2.5, 2.5], device = device),
                            torch.tensor([1.0, 1.0], device = device))
        gmm = dists.MixtureSameFamily(mix, comp)
        return CombinedDistributions([gmm] +
                                      [dists.Normal(torch.tensor(0.0, device = device),
                                                    torch.tensor(1.0, device = device))
                                       for _ in range(self.dim - 1)],
                                     device = device)
    '''

    def inverse_softsign(self, y):
        return torch.where(y > 0, y / (1 - y), y / (1 + y))

    def _call(self, u):
        x = torch.zeros_like(u)
        x[:, 0] = u[:, 0]
        x[:, 1] = self.s(x[:, 0]) + np.sqrt(3 / 20) * u[:, 1]
        x[:, 2] = self.softsign(2 * x[:, 1]) + 1.5 * x[:, 0] - 1 + self.softsign(u[:, 2])
        x[:, 3] = (x[:, 2] - 4) / 5 + 3 + 1 / np.sqrt(10) * u[:, 3]
        return x

    def _inverse(self, x):
        u = torch.zeros_like(x)
        u[:, 3] = (x[:, 3] - (x[:, 2] - 4) / 5 - 3) * np.sqrt(10)
        u[:, 2] = self.inverse_softsign(x[:, 2] - self.softsign(2 * x[:, 1]) -
                                        1.5 * x[:, 0] + 1)
        u[:, 1] = (x[:, 1] - self.s(x[:, 0])) / np.sqrt(3 / 20)
        u[:, 0] = x[:, 0]
        return u

    # according to real_nvp: https://arxiv.org/abs/1605.08803
    # dx_i / du_j = 0 (i < j) ->logdx equals with product of trace
    # type of this function should be like the shape of u
    # Here we sum them together, so do not forgget .unsqueeze(1)
    def log_abs_det_jacobian(self, u, x):
        return torch.log(np.sqrt(3 / 200) / torch.pow((1 + u[:, 2].abs()), 2))\
                    .unsqueeze(1).to(u.device)
