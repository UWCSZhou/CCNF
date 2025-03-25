import numpy as np
import torch
from torch.autograd.functional import jacobian

def compute_jacobian(test_dl, dag, func):
    cg = torch.tensor(dag.ends_adj + np.eye(dag.dim))
    loss = []
    for batch in test_dl:
        for item in batch:
            jac = jacobian(func, item.view(1, len(item))).view(dag.dim, dag.dim)
            jac[cg != 0] = 0
            loss.append(torch.norm(jac, p = 2).item())
    loss = np.array(loss)
    return loss

def kl_distance(module, x):
    log_px = module.data_priors.log_prob(x)
    u, logd = module.flows.reward(x)
    log_pu = module.priors.log_prob(u)
    kl_sum = log_px - log_pu - logd
    return kl_sum.mean()
