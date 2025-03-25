import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists

"""
A little different from Gaussianization Flows: we do not rotate.
"""

class GaussianizationLayer(nn.Module):
    def __init__(self, dim, datapoint_num, device = "cpu"):
        super().__init__()
        self.dim = dim
        self.datapoint_num = datapoint_num
        self.normal = dists.Normal(torch.tensor(0, device = device),
                                   torch.tensor(1, device = device))
        self.datapoints = nn.Parameter(torch.ones(self.datapoint_num, self.dim))
        # the one bandwidth init is used for gaussian kernels with plug-in method?
        # I don't know, it confuses me a lot
        bandwidth = (4. * np.sqrt(np.pi) / ((np.pi ** 4) * self.datapoint_num)) ** 0.2
        self.log_hs = nn.Parameter(torch.ones(self.datapoint_num, self.dim)
                                   * np.log(bandwidth))
        self.hs_min = 1e-7
        self.mask_bound = 0.5e-7

    def logistic_kernel_log_cdf(self, x):
        hs = torch.clamp(torch.exp(self.log_hs), min = self.hs_min)
        x_norm = (x[None, ...] - self.datapoints[:, None, :]) / hs[:, None, :]
        log_cdfs = -F.softplus(-x_norm)
        return torch.logsumexp(log_cdfs, dim = 0) - np.log(self.datapoint_num)

    def logistic_kernel_log_complement_cdf(self, x):
        hs = torch.clamp(torch.exp(self.log_hs), min = self.hs_min)
        x_norm = (x[None, ...] - self.datapoints[:, None, :]) / hs[:, None, :]
        log_ccdfs = -x_norm - F.softplus(-x_norm)
        return torch.logsumexp(log_ccdfs, dim = 0) - np.log(self.datapoint_num)

    def logistic_kernel_log_pdf(self, x):
        hs = torch.clamp(torch.exp(self.log_hs), min = self.hs_min)
        log_hs = torch.clamp(self.log_hs, min = np.log(self.hs_min))
        x_norm = (x[None, ...] - self.datapoints[:, None, :]) / hs[:, None, :]
        log_pdfs = -x_norm - log_hs[:, None, :] - 2 * F.softplus(-x_norm)
        return torch.logsumexp(log_pdfs, dim = 0) - np.log(self.datapoint_num)

    def normal_cdf(self, x, it = 1000):
        low = torch.full_like(x, -1e4)
        high = torch.full_like(x, 1e4)
        for i in range(it):
            mid = (low + high) / 2
            val, _ = self.inverse_normal_cdf(mid)
            low = torch.where(val < x, mid, low)
            high = torch.where(val > x, mid, high)
            if torch.all(torch.abs(high - low) < 1e-6):
                break
        u = (low + high) / 2
        _, logd = self.inverse_normal_cdf(u)
        return (low + high) / 2, -logd

    def inverse_normal_cdf(self, x):
        # most code copy from Gaussianization_Flows

        log_cdf = self.logistic_kernel_log_cdf(x) # log(CDF)
        cdf = torch.exp(log_cdf)
        log_ccdf = self.logistic_kernel_log_complement_cdf(x) # log(1-CDF)
        # Approximate Gaussian CDF
        # inv(CDF) ~ np.sqrt(-2 * np.log(1-x)) #right, lim x = 1
        # inv(CDF) ~ -np.sqrt(-2 * np.log(x)) #left, lim x = 0
        # 1) Step1: invert good CDF
        cdf_mask = ((cdf > self.mask_bound) & (cdf < 1 - (self.mask_bound))).float()
        # Keep good CDF, mask the bad CDF values to 0.5(inverse(0.5)=0.)
        cdf_good = cdf * cdf_mask + 0.5 * (1. - cdf_mask)
        inverse = self.normal.icdf(cdf_good)

        # 2) Step2: invert BAD large CDF
        cdf_mask_right = (cdf >= 1. - (self.mask_bound)).float()
        # Keep large bad CDF, mask the good and small bad CDF values to 0.
        cdf_bad_right_log = log_ccdf * cdf_mask_right
        inverse += torch.sqrt(-2. * cdf_bad_right_log)

        # 3) Step3: invert BAD small CDF
        cdf_mask_left = (cdf <= self.mask_bound).float()
        # Keep small bad CDF, mask the good and large bad CDF values to 0.
        cdf_bad_left_log = log_cdf * cdf_mask_left
        inverse += (-torch.sqrt(-2 * cdf_bad_left_log))

        # Compute PDF
        log_pdf = self.logistic_kernel_log_pdf(x) # log(PDF)
        logd_good = self.normal.log_prob(inverse) * cdf_mask
        # y = sqrt(-2 * log(x))
        cdf_bad_left_log += -1. * (1 - cdf_mask_left) # avoid torch.log(0)
        logd_left = (torch.log(torch.sqrt(-2 * cdf_bad_left_log)) - log_cdf) \
            * cdf_mask_left
        cdf_bad_right_log += -1. * (1 - cdf_mask_right)
        logd_right = (torch.log(torch.sqrt(-2. * cdf_bad_right_log)) - log_ccdf) \
            * cdf_mask_right
        log_g = logd_good + logd_left + logd_right
        logd = (log_pdf - log_g).sum(dim = -1)
        return inverse, logd

    def forward(self, u):
        x = self.normal_cdf(u)
        return x

    def reward(self, x):
        return self.inverse_normal_cdf(x)