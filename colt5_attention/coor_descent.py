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
    mask = None
):
    mask_value = -torch.finfo(s.dtype).max

    if not isinstance(k, torch.Tensor):
        k = torch.Tensor([k]).to(s)

    constant = eps * log(k)

    b = -F.relu(s)

    for _ in range(n_iters):
        if exists(mask):
            s = s.masked_fill(~mask, mask_value)

        a = constant - eps * ((s + b) / eps).logsumexp(dim = -1, keepdim = True)
        b = -F.relu(s + a)

    if exists(mask):
        s = s.masked_fill(~mask, mask_value)

    scores = ((s + a + b) / eps).exp()
    return scores
