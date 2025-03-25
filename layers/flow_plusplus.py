import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from layers.made.made import MaskedMLP

class FlowPlusPlusLayer(nn.Module):
    '''
    a single causal flow++ layer with K mixed logistc prob
    '''
    def __init__(self, dim, hidden_layers, order, adj, slope = 1e-2, k = 4):
        super().__init__()
        self.slope = slope
        self.k = k
        self.dim = dim
        self.order = order
        self.layers = [dim] + hidden_layers + [dim * (3 * k + 2)]
        self.made = MaskedMLP(self.layers, order, adj, slope = slope)

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
            outs = self._log_cdf(mid.unsqueeze(1).repeat(1, self.k), mu, log_sigma) + log_pi
            val = torch.exp(torch.logsumexp(outs, dim = 1))
            low = torch.where(val < y, mid, low)
            high = torch.where(val > y, mid, high)
            if torch.all(torch.abs(high - low) < 1e-6):
                break
        x = (low + high) / 2
        return x

    def forward(self, u):
        x = torch.zeros_like(u)
        logd = torch.zeros(u.shape[0], device = u.device)
        for i in range(self.dim):
            t = self.order[i]
            log_a, b, log_pi, mu, log_sigma = \
                self.made.get_flow_plus_plus(self.made(x), self.k, loga_slope = 1e-1)
            log_a, b = log_a[:, t], b[:, t]
            index = [t + l * self.layers[0] for l in range(self.k)]
            log_pi, mu, log_sigma = log_pi[:, index], mu[:, index], log_sigma[:, index]
            x_t = (u[:, t] - b) * torch.exp(-log_a)
            logd_t = -log_a
            x_t = torch.sigmoid(x_t)
            logd_t += (torch.log(x_t) + torch.log(1 - x_t))
            x_t = self._inverse_logsitic_cdf(x_t, log_pi, mu, log_sigma)
            outs = self._log_pdf(x_t.unsqueeze(1).repeat(1, self.k), mu, log_sigma) + log_pi
            logd_t -= torch.logsumexp(outs, dim = 1)
            x[:, t] = x_t
            logd += logd_t
        return x, logd

    def reward(self, x):
        log_a, b, log_pi, mu, log_sigma = \
            self.made.get_flow_plus_plus(self.made(x), self.k, loga_slope = 1e-1)
        u = self._mixed_logistic_cdf(x, log_pi, mu, log_sigma)
        u = u.clamp(max = -1e-5)
        u = torch.exp(u)
        logd = self._mixed_logistic_pdf(x, log_pi, mu, log_sigma)
        logd -= (torch.log(u) + torch.log(1 - u))
        u = torch.logit(u)
        u = u * torch.exp(log_a) + b
        logd += log_a
        logd = torch.sum(logd, dim = 1)
        return u, logd