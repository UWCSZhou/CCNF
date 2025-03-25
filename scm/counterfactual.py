import torch

def counter(x):
    return 0

def counterfactual(x, dim, forward, reward):
    u = reward(x)
    x2 = x.clone()
    x2[:, dim] = counter(x[:, dim])
    u2 = reward(x2)
    u[:, dim] = u2[:, dim]
    return forward(u)

def counterfactual_with_nf(module, x, dim):
    u, _ = module.flows.reward(x)
    x2 = x.clone()
    x2[:, dim] = counter(x[:, dim])
    u2, _ = module.flows.reward(x2)
    u[:, dim] = u2[:, dim]
    return module.flows.forward(u)[0]

def counterfactual_with_vaca(module, x, dim):
    num = len(x)
    u = module.model(x, get_qz_x = True).sample()
    x2 = x.clone()
    x2[:, dim] = counter(x[:, dim])
    u2 = module.model(x2, get_qz_x = True).sample()
    index = [dim + i * module.model.dim for i in range(num)]
    u[index] = u2[index]
    return module.model.forward_with_priors(None, num, u)[1]
