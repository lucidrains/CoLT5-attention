import torch
from collections import namedtuple
from colt5_attention.coor_descent import coor_descent

TopkReturn = namedtuple('TopkReturn', ['values', 'indices', 'coor_descent_values', 'gates'])

def topk(
    x,
    k,
    coor_descent_k_ratio = 9 / 8,
    n_iters = 20,
    eps = 1e-1,
    eps_init = None,
    eps_decay = 1.,
    mask = None,
    fused = False
):
    """
    differentiable top-k on last dimension
    """
    assert coor_descent_k_ratio >= 1.
    assert k > 0

    # whether to used fused kernel or not

    fn = coor_descent

    if fused and x.is_cuda:
        from colt5_attention.triton_coor_descent import triton_coor_descent
        fn = triton_coor_descent

    # do coordinate descent for gradients

    coor_descent_out = fn(
        x,
        k = min(k * coor_descent_k_ratio, x.shape[-1]),   # fetch a bit more for better learning, as in CoLT5 paper (they fetched 9 / 8 times more)
        mask = mask,
        n_iters = n_iters,
        eps = eps,
        eps_init = eps_init,
        eps_decay = eps_decay
    )

    # do straight through

    gates = coor_descent_out + (1 - coor_descent_out).detach()

    x = x * gates

    # hard topk

    values, indices = torch.topk(x, k, dim = -1)

    # return something that looks like a usual topk, but now differentiable

    coor_descent_values = coor_descent_out.gather(-1, indices)
    gates = gates.gather(-1, indices)

    return TopkReturn(values, indices, coor_descent_values, gates)
