import torch
import torch.nn.functional as F

from einops import rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def coor_descent(
    s,
    *,
    n_iters,
    k,
    eps = 1e-1,
    eps_init = None,
    eps_decay = 1.,
    mask = None
):
    """
    coordinate descent  - https://arxiv.org/abs/1502.04759, utilized in https://arxiv.org/abs/2303.09752
    ε-scaling           - https://arxiv.org/abs/1610.06519, utilized in https://arxiv.org/abs/2304.04947

    in a follow up paper applying coordinate descent routing to efficient fine tuning
    they were able to cut n_iters from 50 -> 20 by setting eps_init = 4 and eps_decay = 0.7
    eps was dependent on the task, and ranged from 0.02 to 1
    """

    assert n_iters > 0

    mask_value = -torch.finfo(s.dtype).max

    if not isinstance(k, torch.Tensor):
        k = torch.Tensor([k]).to(s)
    else:
        k = rearrange(k, '... -> ... 1')

    logk = log(k)

    if exists(mask):
        s = s.masked_fill(~mask, mask_value)

    a = 0
    b = -s

    current_eps = max(default(eps_init, eps), eps)

    for _ in range(n_iters):
        sb = ((s + b) / current_eps)

        if exists(mask):
            sb = sb.masked_fill(~mask, mask_value)

        a = current_eps * (logk - sb.logsumexp(dim = -1, keepdim = True))
        b = -F.relu(s + a)

        current_eps = max(current_eps * eps_decay, eps)

    scores = ((s + a + b) / current_eps).exp()

    if exists(mask):
        scores = scores.masked_fill(~mask, 0.)

    return scores
