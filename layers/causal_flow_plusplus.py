import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from layers.made.causal_made import CausalMaskedMLP

class CausalFlowPlusPlusLayer(nn.Module):
    '''
    a single causal flow++ layer with K mixed logistc prob
    '''
    def __init__(self, clayer, hidden_layers, slope = 1e-2, k = 4):
        super().__init__()
        self.clayer = clayer
        self.slope = slope
        self.k = k
        self.made = CausalMaskedMLP(self.clayer, hidden_layers,
                                    slope = slope, outlen = 2 + 3 * k)
        self.dim = len(self.clayer.ends)

    def _log_pdf(self, x, mu, log_sigma):
        z = torch.exp(-log_sigma) * (x - mu)
        return z - log_sigma - 2 * F.softplus(z)

    def _log_cdf(self, x, mu, log_sigma):
        z = torch.exp(-log_sigma) * (x - mu)
        return F.logsigmoid(z)

    def _mixed_logistic_pdf(self, x, log_pi, mu, log_sigma):
        #logd = -log_sigma
        outs = x.repeat(1, self.k)
        outs = self._log_pdf(outs, mu, log_sigma) + log_pi
        log_d = torch.zeros_like(x)
        for i in range(self.dim):
            indices = [i + j * self.dim for j in range(self.k)]
            log_d[:, i] = torch.logsumexp(outs[:, indices], dim = 1)
        return log_d

    def _mixed_logistic_cdf(self, x, log_pi, mu, log_sigma):
        outs = x.repeat(1, self.k)
        outs = self._log_cdf(outs, mu, log_sigma) + log_pi
        log_p = torch.zeros_like(x)
        for i in range(self.dim):
            indices = [i + j * self.dim for j in range(self.k)]
            log_p[:, i] = torch.logsumexp(outs[:, indices], dim = 1)
        return log_p

    def _inverse_logsitic_cdf(self, y, log_pi, mu, log_sigma, it = 1000):
        low = torch.full_like(y, -1e3)
        high = torch.full_like(y, 1e3)
        for i in range(it):
            mid = (low + high) / 2
            val = torch.exp(self._mixed_logistic_cdf(mid, log_pi, mu, log_sigma))
            low = torch.where(val < y, mid, low)
            high = torch.where(val > y, mid, high)
            if torch.all(torch.abs(high - low) < 1e-6):
                break
        x = (low + high) / 2
        return x

    def forward(self, coupling, u):
        log_a, b, log_pi, mu, log_sigma = \
            self.made.get_flow_plus_plus(self.made(coupling), self.k, loga_slope = 1e-1)
        x = (u[:, self.clayer.ends] - b) * torch.exp(-log_a)
        logd = -log_a
        x = torch.sigmoid(x)
        logd += (torch.log(x) + torch.log(1 - x))
        x = self._inverse_logsitic_cdf(x, log_pi, mu, log_sigma)
        logd -= self._mixed_logistic_pdf(x, log_pi, mu, log_sigma)
        logd = torch.sum(logd, dim = 1)
        x_t = u.clone()
        x_t[:, self.clayer.ends] = x
        return x_t, logd

    def reward(self, coupling, x):
        log_a, b, log_pi, mu, log_sigma = \
            self.made.get_flow_plus_plus(self.made(coupling), self.k, loga_slope = 1e-1)
        u = self._mixed_logistic_cdf(x[:, self.clayer.ends], log_pi, mu, log_sigma)
        u = u.clamp(max = -1e-5)
        u = torch.exp(u)
        logd = self._mixed_logistic_pdf(x[:, self.clayer.ends], log_pi, mu, log_sigma)
        logd -= (torch.log(u) + torch.log(1 - u))
        u = torch.logit(u)
        u = u * torch.exp(log_a) + b
        logd += log_a
        logd = torch.sum(logd, dim = 1)
        u_t = x.clone()
        u_t[:, self.clayer.ends] = u
        return u_t, logd