import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEFAULT_MIN_KNOTS_WIDTH = 1e-3
DEFAULT_MIN_KNOTS_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

class RQS():
    def __init__(self, dim, bins, knots, w, h, d):
        self.dim = dim
        self.bins = bins
        self.knots = knots
        self.weights = w#torch.Tensor(self.dim, self.knots).uniform_(-0.5, 0.5)
        self.heights = h#torch.Tensor(self.dim, self.knots).uniform_(-0.5, 0.5)
        self.derivatives = d#torch.Tensor(self.dim, self.knots - 1).uniform_(-0.1, 1)
        #self.weights = torch.Tensor([[10, 20, 30, 40, 50]])
        #self.heights = torch.Tensor([[15, 20, 25, 30, 35]])
        #self.derivatives = torch.Tensor([[10, 20, 30, 40, 50]])

    def generate_rqs(self):
        w, h, d = torch.softmax(self.weights, dim = 1), \
            torch.softmax(self.heights, dim = 1), F.softplus(self.derivatives)
        #w
        w = DEFAULT_MIN_KNOTS_WIDTH + (1 - DEFAULT_MIN_KNOTS_WIDTH * self.knots) * w
        cum_w = F.pad(torch.cumsum(w, dim = 1), pad = (1, 0))
        cum_w[:, 0], cum_w[:, -1] = 0., 1.
        cum_w = -self.bins + 2 * self.bins * cum_w
        cum_w = cum_w.contiguous()
        w = cum_w[:, 1:] - cum_w[:, :-1]
        #d
        d += DEFAULT_MIN_DERIVATIVE
        d = F.pad(d, pad = (1, 1))
        d[:, 0] = d[:, -1] = 1.
        #h
        h = DEFAULT_MIN_KNOTS_HEIGHT + (1 - DEFAULT_MIN_KNOTS_HEIGHT * self.knots) * h
        cum_h = F.pad(torch.cumsum(h, dim = 1), pad = (1, 0))
        cum_h[:, 0], cum_h[:, -1] = 0., 1.
        cum_h = -self.bins + 2 * self.bins * cum_h
        cum_h = cum_h.contiguous()
        h = cum_h[:, 1:] - cum_h[:, :-1]
        return w, cum_w, h, cum_h, d, h / w

    def collect_loc(self, cum, col, output):
        loc = torch.searchsorted(cum, col)
        outside = (loc == self.knots + 1) | (loc == 0)
        loc -= 1
        loc[outside] = 0
        inside = ~outside
        output[outside] = col[outside]
        return inside, loc, output

    def gather_whd(self, inside, loc, w, cum_w, h, cum_h, d, delta):
        knot_w = w.gather(1, loc)[inside]
        knot_cum_w = cum_w.gather(1, loc)[inside]
        knot_h = h.gather(1, loc)[inside]
        knot_cum_h = cum_h.gather(1, loc)[inside]
        knot_d = d.gather(1, loc)[inside]
        knot_d_n = d.gather(1, loc + 1)[inside]
        knot_delta = delta.gather(1, loc)[inside]
        return knot_w, knot_cum_w, knot_h, knot_cum_h, knot_d, knot_d_n, knot_delta

    def forward(self, u):
        x, col = torch.zeros_like(u.T), u.T.contiguous()
        w, cum_w, h, cum_h, d, delta = self.generate_rqs()
        inside, loc, x = self.collect_loc(cum_w, col, x)
        knot_w, knot_cum_w, knot_h, knot_cum_h, knot_d, knot_d_n, knot_delta = \
            self.gather_whd(inside, loc, w, cum_w, h, cum_h, d, delta)
        # forward start
        x_1 = (col[inside] - knot_cum_w) / knot_w
        x_1, x_2, x_3 = x_1.pow(2), x_1 * (1 - x_1), (1 - x_1).pow(2)
        a = knot_h * (x_1 * knot_delta + knot_d * x_2)
        b = knot_delta + ((knot_d_n + knot_d - 2 * knot_delta) * x_2)
        x[inside] = knot_cum_h + a / b
        c = knot_delta.pow(2) * (knot_d_n * x_1 + 2 * knot_delta * x_2 + knot_d * x_3)
        y = torch.zeros_like(x)
        y[inside] = torch.log(c) - 2 * torch.log(b)
        return x.T, y.sum(dim = 0)

    def reward(self, x):
        u, col = torch.zeros_like(x.T), x.T.contiguous()
        w, cum_w, h, cum_h, d, delta = self.generate_rqs()
        inside, loc, u = self.collect_loc(cum_h, col, u)
        knot_w, knot_cum_w, knot_h, knot_cum_h, knot_d, knot_d_n, knot_delta = \
            self.gather_whd(inside, loc, w, cum_w, h, cum_h, d, delta)
        #reward start
        a = knot_h * (knot_delta - knot_d) + \
            (col[inside] - knot_cum_h) * (knot_d + knot_d_n - 2 * knot_delta)
        b = knot_h * knot_d - \
            (col[inside] - knot_cum_h) * (knot_d + knot_d_n - 2 * knot_delta)
        c = -knot_delta * (col[inside] - knot_cum_h)
        d = b.pow(2) - 4 * a * c
        r = 2 * c / (-b - torch.sqrt(d))
        u[inside] = r * knot_w + knot_cum_w
        x_1, x_2, x_3 = r.pow(2), r * (1 - r), (1 - r).pow(2)
        db = knot_delta + ((knot_d_n + knot_d - 2 * knot_delta) * x_2)
        dc = knot_delta.pow(2) * (knot_d_n * x_1 + 2 * knot_delta * x_2 + knot_d * x_3)
        y = torch.zeros_like(u)
        y[inside] = torch.log(dc) - 2 * torch.log(db)
        return u.T, -y.sum(dim = 0)