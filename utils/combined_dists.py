import torch
from torch.distributions.distribution import Distribution
# it is interesting that it is not in offical api
# Maybe it exists but I dont know?

class CombinedDistributions(Distribution):
    arg_constraints = {}

    def __init__(self, dists, device = "cuda"):
        # dist is like [Normal, Normal ...]
        super().__init__()
        self.dim = len(dists)
        self.device = device
        self.dists = dists

    def sample(self, num):
        # we each time only same 1 value
        num = num[0]
        x = torch.zeros(num, self.dim, device = self.device)
        for i in range(self.dim):
            x[:, i] = self.dists[i].sample((num,))
        return x

    def log_prob(self, x):
        logp = torch.zeros_like(x)
        for i in range(self.dim):
            logp[:, i] = self.dists[i].log_prob(x[:, i])
        return logp.sum(axis = 1)
