from math import log

import torch
from torch import Tensor
from torch import autograd
import torch.nn.functional as F

from colt5_attention.coor_descent import coor_descent
from einops import pack, unpack, repeat

try:
    import triton
    import triton.language as tl
except ImportError as e:
    print('triton is not installed, please install by running `pip install triton -U --pre`')
    exit()

# make sure it is latest triton

from packaging import version
assert version.parse(triton.__version__) >= version.parse('2.0')

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_num_warps(block_size):
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 4096:
        num_warps = 16
    return num_warps

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]    

def num_to_groups(num, groups):
    assert 0 < groups <= num
    floor = num // groups
    remainder = num % groups
    out = []
    for ind in range(groups):
        out.append(floor + int(ind < remainder))
    assert sum(out) == num
    return out

# forwards

@triton.jit
def coor_descent_kernel_forward(
    a_ptr,
    b_ptr,
    input_ptr,
    mask_ptr,
    k_ptr,
    a_iter_stride,
    b_row_stride,
    b_iter_stride,
    input_row_stride,
    mask_row_stride,
    n_iters,
    current_eps,
    eps_decay,
    eps,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    # load mask as ints (for some reason as boolean breaks triton)

    mask_start_ptr = mask_ptr + row_idx * mask_row_stride
    mask_ptrs = mask_start_ptr + col_offsets

    mask_ints = tl.load(mask_ptrs, mask = col_mask, other = 0)
    mask = mask_ints == 1

    # load a and b

    a_ptr = a_ptr + row_idx
    a = tl.load(a_ptr)

    b_start_ptr = b_ptr + row_idx * b_row_stride
    b_ptrs = b_start_ptr + col_offsets
    b = tl.load(b_ptrs, mask = col_mask, other = 0)

    # load the scores s

    row_start_ptr = input_ptr + row_idx * input_row_stride
    input_ptrs = row_start_ptr + col_offsets
    s = tl.load(input_ptrs, mask = mask, other = -float('inf'))

    # load k - controls the sparsity of output

    k_ptr = k_ptr + row_idx
    k = tl.load(k_ptr)

    # initialize some constants

    logk = tl.log(k)

    for _ in range(n_iters):        

        a = (s + b) / current_eps
        a = tl.where(mask, a, -float('inf'))

        # stable log sum exp

        a_max = tl.max(a, axis = 0)
        a_minus_max = tl.where(mask, a - a_max, -float('inf'))
        exp = tl.exp(a_minus_max)
        sum_exp = tl.sum(exp, axis = 0)
        log_sum_exp = tl.log(sum_exp) + a_max

        a = current_eps * (logk - log_sum_exp)

        # update b

        b = s + a
        b = tl.where(b >= 0., -b, 0.)

        # decay epsilon, from epsilon-scaling

        current_eps *= eps_decay

        if current_eps < eps:
            current_eps = eps

    # store a and b for next round

    next_a_ptrs = a_ptr + a_iter_stride
    next_b_ptrs = b_ptrs + b_iter_stride

    tl.store(next_a_ptrs, a)
    tl.store(next_b_ptrs, b, mask = col_mask)

# backwards

@triton.jit
def coor_descent_kernel_backward(
    dk_ptr,
    input_ptr,
    a_ptr,
    b_ptr,
    mask_ptr,
    ds_ptr,
    db_ptr,
    k_ptr,
    input_row_stride,
    b_row_stride,
    mask_row_stride,
    ds_row_stride,
    db_row_stride,
    n_iters,
    eps_init,
    eps_decay,
    eps,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # load and generate mask

    col_mask = col_offsets < n_cols

    # load mask as ints (for some reason as boolean breaks triton)

    mask_start_ptr = mask_ptr + row_idx * mask_row_stride
    mask_ptrs = mask_start_ptr + col_offsets

    mask_ints = tl.load(mask_ptrs, mask = col_mask, other = 0)
    mask = mask_ints == 1

     # load a and b

    a_ptr = a_ptr + row_idx
    init_a = tl.load(a_ptr)

    b_start_ptr = b_ptr + row_idx * b_row_stride
    b_ptrs = b_start_ptr + col_offsets
    init_b = tl.load(b_ptrs, mask = mask, other = 0)

    # load input

    row_start_ptr = input_ptr + row_idx * input_row_stride
    input_ptrs = row_start_ptr + col_offsets

    s = tl.load(input_ptrs, mask = mask, other = -float('inf'))

    # load k - controls the sparsity of output

    k_ptr = k_ptr + row_idx

    k = tl.load(k_ptr)
    logk = tl.log(k)

    # load initial ds

    ds_row_start_ptr = ds_ptr + row_idx * ds_row_stride
    ds_ptrs = ds_row_start_ptr + col_offsets

    ds = tl.load(ds_ptrs, mask = mask, other = 0.)

    # load initial db

    db_row_start_ptr = db_ptr + row_idx * db_row_stride
    db_ptrs = db_row_start_ptr + col_offsets

    db = tl.load(db_ptrs, mask = mask, other = 0.)

    # load initial dk

    dk_ptr = dk_ptr + row_idx
    dk = tl.load(dk_ptr)

    # temp variables

    last_da = tl.sum(ds, axis = 0)

    # backwards

    for ind in range(n_iters):
        a = init_a
        b = init_b

        sa = s * 0
        softmax = s * 0

        # calculate epsilon

        current_eps = eps_init / eps_decay

        # recompute

        for _ in range(n_iters - ind):
            # update epsilon

            current_eps *= eps_decay

            if current_eps < eps:
                current_eps = eps

            # updating a

            sb = (s + b) / current_eps
            sb = tl.where(mask, sb, -float('inf'))

            # stable log sum exp

            sb_max = tl.max(sb, axis = 0)
            sb_minus_max = tl.where(mask, sb - sb_max, -float('inf'))
            exp = tl.exp(sb_minus_max)
            sum_exp = tl.sum(exp, axis = 0)
            softmax = exp / sum_exp
            log_sum_exp = tl.log(sum_exp) + sb_max

            a = current_eps * (logk - log_sum_exp)

            # update b

            sa = s + a
            b = tl.where(sa > 0., -sa, 0.)

        # go backwards

        dsa = db * tl.where(sa > 0, -1., 0.)

        ds += dsa

        da = tl.sum(dsa, axis = 0) + last_da

        dk += da * current_eps

        dsb = da * -softmax

        ds += dsb
        db = dsb

        last_da = 0.

    # store dk

    tl.store(dk_ptr, dk)

    # store ds

    tl.store(ds_ptrs, ds, mask = col_mask)

    # store db

    tl.store(db_ptrs, db, mask = col_mask)

# function forwards and backwards

class _coor_descent(autograd.Function):
    @classmethod
    def forward(
        self,
        ctx,
        x,
        n_iters,
        k,
        eps,
        eps_init,
        eps_decay,
        mask,
        checkpoint_segments
    ):
        assert n_iters > 0
        assert x.is_cuda, 'triton coordinate descent must be on cuda'

        batch, requires_grad, device, dtype = x.shape[0], x.requires_grad, x.device, x.dtype

        if not exists(mask):
            mask = torch.ones_like(x, dtype = torch.bool, device = x.device)

        x, shape = pack_one(x, '* n')
        mask, _ = pack_one(mask, '* n')

        x = x.masked_fill(~mask, -torch.finfo(x.dtype).max)
        mask_ints = mask.int()

        epsilons = []
        eps_init = default(eps_init, eps)
        current_eps = float(max(eps_init, eps))

        n_rows, n_cols = x.shape

        if isinstance(k, (int, float)):
            k = torch.full((n_rows,), k)

        assert k.numel() == n_rows

        k = k.to(x)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        assert BLOCK_SIZE <= 131072, 'the maximum block size allowed is 131072 for triton cuda kernel - set the `route_block_size` for the CoordinateDescentRouter to be this value or less in order to uniformly route to get around this limitation'

        num_warps = calc_num_warps(BLOCK_SIZE)

        checkpointed_a = torch.empty((checkpoint_segments + 1, n_rows), device = device, dtype = dtype)
        checkpointed_b = torch.empty((checkpoint_segments + 1, n_rows, n_cols), device = device, dtype = dtype)

        checkpointed_a[0] = torch.zeros_like(k)
        checkpointed_b[0] = -x

        for ind, segment_iters in enumerate(num_to_groups(n_iters, checkpoint_segments)):
            is_last = ind == (checkpoint_segments - 1)

            epsilons.append(current_eps)

            coor_descent_kernel_forward[(n_rows,)](
                checkpointed_a[ind],
                checkpointed_b[ind],
                x,
                mask_ints,
                k,
                checkpointed_a.stride(0),
                n_cols,
                checkpointed_b.stride(0),
                x.stride(0),
                mask_ints.stride(0),
                segment_iters,
                current_eps,
                eps_decay,
                eps,
                n_cols,
                num_warps = num_warps,
                BLOCK_SIZE = BLOCK_SIZE,
            )

            current_eps *= (eps_decay ** segment_iters)
            current_eps = max(current_eps, eps)

        last_a, last_b = map(lambda t: t[-1], (checkpointed_a, checkpointed_b))
        y = torch.exp((last_a[..., None] + last_b + x) / current_eps)

        epsilons.append(current_eps)

        if requires_grad:
            checkpointed_a = checkpointed_a[:-1]
            checkpointed_b = checkpointed_b[:-1]

            ctx.args = (n_iters, checkpoint_segments, epsilons, eps_decay, eps)
            ctx.save_for_backward(x, y, k, mask, checkpointed_a, checkpointed_b)

        y = unpack_one(y, shape, '* n')

        return y

    @classmethod
    def backward(
        self,
        ctx,
        grad_probs
    ):
        assert grad_probs.is_cuda

        batch = grad_probs.shape[0]

        n_iters, checkpoint_segments, epsilons, eps_decay, eps = ctx.args
        x, y, k, mask, checkpointed_a, checkpointed_b = ctx.saved_tensors

        grad_probs, shape = pack_one(grad_probs, '* n')

        if exists(mask):
            grad_probs = grad_probs.masked_fill(~mask, 0.)

        n_rows, n_cols = grad_probs.shape

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        *epsilons, last_eps = epsilons

        ds = grad_probs * y / last_eps
        db = ds.clone()
        dk = torch.zeros_like(k)

        mask_int = mask.int()

        items = zip(
            reversed(checkpointed_a.unbind(dim = 0)),
            reversed(checkpointed_b.unbind(dim = 0)),
            reversed(num_to_groups(n_iters, checkpoint_segments)),
            reversed(epsilons)
        )

        for init_a, init_b, segment_iters, eps_init, in items:
            coor_descent_kernel_backward[(n_rows,)](
                dk,
                x,
                init_a,
                init_b,
                mask_int,
                ds,
                db,
                k,
                x.stride(0),
                init_b.stride(0),
                mask_int.stride(0),
                ds.stride(0),
                db.stride(0),
                segment_iters,
                eps_init,
                eps_decay,
                eps,
                n_cols,
                num_warps = num_warps,
                BLOCK_SIZE = BLOCK_SIZE
            )

        ds += -db
        ds = unpack_one(ds, shape, '* n')

        if not k.requires_grad:
            dk = None
        else:
            dk /= k

        return ds, None, dk, None, None, None, None, None

def triton_coor_descent(
    s,
    *,
    n_iters,
    k,
    eps = 1e-1,
    eps_init = None,
    eps_decay = 1.,
    mask = None,
    checkpoint_segments = 1
):
    if not s.is_cuda:
        return coor_descent(s, n_iters = n_iters, k = k, eps = eps, eps_init = eps_init, eps_decay = eps_decay, mask = mask)

    return _coor_descent.apply(s, n_iters, k, eps, eps_init, eps_decay, mask, checkpoint_segments)
