import torch
import torch.nn.functional as F

def exists(val):
    return val is not None

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def coor_descent(
    s,
    *,
    n_iters,
    k,
    eps = 1e-1,
    clamp_fn = F.relu,
    mask = None,
):
    mask_value = -torch.finfo(s.dtype).max
    constant = eps * log(k)

    b = -clamp_fn(s)

    for _ in range(n_iters):
        if exists(mask):
            s = s.masked_fill(~mask, mask_value)

        a = constant - eps * ((s + b) / eps).logsumexp(dim = -1, keepdim = True)
        b = -clamp_fn(s + a)

    if exists(mask):
        s = s.masked_fill(~mask, mask_value)

    scores = ((s + a + b) / eps).exp()
    return scores