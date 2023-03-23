import torch
from einops import repeat

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def scatter_mean(src, t, index, dim, eps = 1e-5):
    index = repeat(index, '... -> ... d', d = t.shape[-1])
    numer = src.scatter_add(dim, index, t)
    denom = src.scatter_add(dim, index, torch.ones_like(t))
    return numer / denom.clamp(min = eps)

def sample_gumbel(shape, device, dtype):
    u = torch.empty(shape, device = device, dtype = dtype).uniform_(0, 1)
    return -log(-log(u))

def sinkhorn(r, n_iters = 8):
    n = r.shape[1]
    for _ in range(n_iters):
        r = r - torch.logsumexp(r, dim = 2, keepdim = True)
        r = r - torch.logsumexp(r, dim = 1, keepdim = True)
    return torch.exp(r)

def gumbel_sinkhorn(r, n_iters = 8, temperature = 0.7):
    r = log(r)
    gumbel = sample_gumbel(r.shape, r.device, r.dtype)
    r = (r + gumbel) / temperature
    return sinkhorn(r, n_iters)
