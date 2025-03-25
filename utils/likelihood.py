import torch
import torch.nn as nn
import torch.distributions as dists

### Delta codes from VACA: https://github.com/psanch21/VACA/blob/main/utils/likelihoods.py
### it seems using centered normal distributions to approximate dirac

class Delta(dists.Distribution):
    def __init__(self, center = None, lambda_ = 1.0, validate_args = False):
        self.center = center
        self.lambda_ = lambda_
        super(Delta, self).__init__(self.center.size(), validate_args = validate_args)

    def mean(self):
        return self.center

    def sample(self, sample_shape = torch.Size()):
        return self.center

    def rsample(self, sample_shape = torch.Size()):
        raise NotImplementedError()

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return - (1 / self.lambda_) * (value - self.center) ** 2

class Normalikelihood(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.params_size = 2 * hidden_dim

    def forward(self, logit):
        size = logit.size(1) // 2 #logit size must be even
        mu, log_sigma = logit.split(size, dim = -1)
        # vaca has no clamp but causal-nf has.
        # I believe clamp is in need since there are too many inf
        # All magic number are borrowed from causal-nf
        #log_sigma = log_sigma.clamp(min = -70, max = 70)
        sigma = torch.exp(log_sigma / 2)
        sigma = sigma.clamp(min = 0.001, max = 10)
        return dists.Normal(mu, sigma)

class Deltalikelihood(nn.Module):
    def __init__(self, hidden_dim, lambda_ = 0.01, params_size = 1):
        super().__init__()
        self.params_size = params_size
        self.lambda_ = lambda_

    def forward(self, logit):
        return Delta(logit, self.lambda_)
