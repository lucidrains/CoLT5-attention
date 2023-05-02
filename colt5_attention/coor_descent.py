import torch
import torch.nn.functional as F

from einops import rearrange

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
    else:
        k = rearrange(k, '... -> ... 1')

    constant = eps * log(k)

    if exists(mask):
        s = s.masked_fill(~mask, mask_value)

    a = 0
    b = -s

    for _ in range(n_iters):
        sb = ((s + b) / eps)

        if exists(mask):
            sb = sb.masked_fill(~mask, mask_value)

        a = constant - eps * sb.logsumexp(dim = -1, keepdim = True)
        b = -F.relu(s + a)

    scores = ((s + a + b) / eps).exp()

    if exists(mask):
        scores = scores.masked_fill(~mask, 0.)

    return scores
