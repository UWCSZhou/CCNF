import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn.models import PNA, GIN, GAT
from torch_geometric.utils import degree
import torch
import torch.nn as nn
import torch.distributions as dists

from utils.likelihood import Deltalikelihood, Normalikelihood

class VACALayer(nn.Module):
    def __init__(self, model, priors, dim, edge_index,
                 num_enc_layers, num_dec_layers, hidden_dim_of_z = 4,
                 hidden_enc_channels = 16, hidden_dec_channels = 16,
                 dropout = 0.0, pre_layers = 1, post_layers = 1,
                 mu = 0.0, std = 1.0):
        super().__init__()
        self.dim = dim
        self.edge_index = nn.Parameter(torch.tensor(edge_index, dtype = torch.long),
                                       requires_grad = False)
        self.mu = nn.Parameter(mu, requires_grad = False)
        self.std = nn.Parameter(std, requires_grad = False)
        self.priors_z = priors
        # Note: this is deg, not dag
        deg = self._compute_deg()
        scalers = ['identity', 'amplification', 'attenuation', 'linear', 'inverse_linear']
        aggregators = ['sum', 'min', 'max', 'std', 'var']
        self.enlikelihood = Normalikelihood(hidden_dim_of_z)
        self.delikelihood = Deltalikelihood(hidden_dim_of_z, 0.05)
        if model == "pna":
            self.encoder = PNA(in_channels = 1,
                               hidden_channels = hidden_enc_channels,
                               num_layers = num_enc_layers,
                               out_channels = self.enlikelihood.params_size,
                               aggregators = aggregators, scalers = scalers, deg = deg,
                               pre_layers = pre_layers, post_layers = post_layers)
            # the hidden channel, dropout of decoder is fixed, ... for now
            self.decoder = PNA(in_channels = hidden_dim_of_z,
                               hidden_channels = hidden_dec_channels,
                               num_layers = num_dec_layers, dropout = dropout,
                               out_channels = self.delikelihood.params_size,
                               aggregators = aggregators, scalers = scalers, deg = deg)
        elif model == "gin":
            self.encoder = GIN(in_channels = 1,
                               hidden_channels = hidden_enc_channels,
                               num_layers = num_enc_layers,
                               out_channels = self.enlikelihood.params_size)
            self.decoder = GIN(in_channels = hidden_dim_of_z,
                               hidden_channels = hidden_dec_channels,
                               num_layers = num_dec_layers,
                               out_channels = self.delikelihood.params_size)
        elif model == "gat":
            self.encoder = GAT(in_channels = 1,
                               hidden_channels = hidden_enc_channels,
                               num_layers = num_enc_layers,
                               out_channels = self.enlikelihood.params_size)
            self.decoder = GAT(in_channels = hidden_dim_of_z,
                               hidden_channels = hidden_dec_channels,
                               num_layers = num_dec_layers,
                               out_channels = self.delikelihood.params_size)
        print(self.encoder.convs, self.decoder.convs)

    def _compute_deg(self):
        d = degree(self.edge_index[1], num_nodes = self.dim, dtype = torch.long)
        return torch.bincount(d)

    def extend_edge_index(self, n):
        # this code may seem strange
        # actucally it is extending graph to batch
        # print edge_index and self.edge_index before return you can see why
        edge_num = self.edge_index.shape[1]
        edge_index = self.edge_index.repeat(1, n)
        for i in range(1, n):
            edge_index[:, i * edge_num:] += self.dim
        #print(self.edge_index, edge_index)
        return edge_index

    def _elbo(self, qz_x, px_z, x, beta = 1.0):
        logp = px_z.log_prob(x).flatten(1).sum(1).mean()
        kl = dists.kl.kl_divergence(qz_x, self.priors_z).flatten(1).sum(1).mean()
        return logp - beta * kl

    def encode(self, x, edge_index):
        logit = self.encoder(x, edge_index)
        qz_x = self.enlikelihood(logit)
        # WARNING!!!!!!!!! rsample()!!!!!!!! not sample()!!!!!!!!
        # Pytorch should merge sample and rsample together and use arg to distinguish
        return qz_x, qz_x.rsample()

    def decode(self, z, edge_index):
        logit = self.decoder(z, edge_index)
        return self.delikelihood(logit)

    def forward(self, x, get_qz_x = False):
        x = (x - self.mu) / self.std
        n = x.shape[0] # // self.dim
        edge_index = self.extend_edge_index(n)
        x = x.view(-1, 1)
        qz_x, z = self.encode(x, edge_index)
        if get_qz_x:
            return qz_x
        px_z = self.decode(z, edge_index)
        return -self._elbo(qz_x, px_z, x)

    def forward_with_priors(self, priors, n, u = None):
        if u is None:
            u = priors.sample((n * self.dim,))
        edge_index = self.extend_edge_index(n)
        x = self.decode(u, edge_index).sample().view(n, self.dim)
        x = x * self.std + self.mu
        return u, x
