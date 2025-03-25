import torch

def intervene(x, dim, value, forward, reward):
    u = reward(x)
    x2 = x.clone()
    x2[:, dim] = value
    u2 = reward(x2)
    u[:, dim] = u2[:, dim]
    return forward(u)

def intervene_with_nf(module, num, dim, value):
    u = module.priors.sample((num,))
    x, _ = module.flows.forward(u)
    x[:, dim] = value
    u2, _ = module.flows.reward(x)
    u[:, dim] = u2[:, dim]
    return module.flows.forward(u)[0]

def intervene_with_vaca(module, num, dim, value):
    u, x = module.model.forward_with_priors(module.priors, num)
    x[:, dim] = value
    u2 = module.model(x, get_qz_x = True).sample()
    index = [dim + i * module.model.dim for i in range(num)]
    u[index] = u2[index]
    return module.model.forward_with_priors(None, num, u)[1]
