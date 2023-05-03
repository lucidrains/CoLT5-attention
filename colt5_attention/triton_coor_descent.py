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

# forwards

@triton.jit
def coor_descent_kernel_forward(
    output_ptr,
    input_ptr,
    mask_ptr,
    k_ptr,
    output_row_stride,
    input_row_stride,
    mask_row_stride,
    k_row_stride,
    n_iters,
    eps,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)

    row_start_ptr = input_ptr + row_idx * input_row_stride
    k_ptr = k_ptr + row_idx * k_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    col_mask = col_offsets < n_cols

    # load mask as ints (for some reason as boolean breaks triton)

    mask_start_ptr = mask_ptr + row_idx * mask_row_stride
    mask_ptrs = mask_start_ptr + col_offsets

    mask_ints = tl.load(mask_ptrs, mask = col_mask, other = 0)
    mask = mask_ints == 1

    # load the scores s

    s = tl.load(input_ptrs, mask = mask, other = -float('inf'))

    # load k - controls the sparsity of output

    k = tl.load(k_ptr)

    constant = tl.log(k) * eps

    inv_eps = 1. / eps

    # initialize a and b for coordinate descent

    a = k * 0 # init a to 0, triton needs to know shape and type outside loop

    b = -s

    for _ in range(n_iters):        

        a = (s + b) * inv_eps
        a = tl.where(mask, a, -float('inf'))

        # stable log sum exp

        a_max = tl.max(a, axis = 0)
        a_minus_max = a - a_max
        exp = tl.exp(a_minus_max)
        sum_exp = tl.sum(exp, axis = 0)
        log_sum_exp = tl.log(sum_exp) + a_max

        a = constant - eps * log_sum_exp

        # update b

        b = s + a
        b = tl.where(b >= 0., -b, 0.)

    s = tl.exp((s + a + b) * inv_eps)

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets

    s = tl.where(mask, s, 0.)
    tl.store(output_ptrs, s)

# backwards

@triton.jit
def coor_descent_kernel_backward(
    output_ptr,
    input_ptr,
    mask_ptr,
    grad_ptr,
    k_ptr,
    output_row_stride,
    input_row_stride,
    mask_row_stride,
    grad_row_stride,
    k_row_stride,
    n_iters,
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

    # load input

    row_start_ptr = input_ptr + row_idx * input_row_stride
    input_ptrs = row_start_ptr + col_offsets

    s = tl.load(input_ptrs, mask = mask, other = -float('inf'))

    # load k - controls the sparsity of output

    k_ptr = k_ptr + row_idx * k_row_stride

    k = tl.load(k_ptr)
    constant = tl.log(k) * eps

    # load grads

    grad_row_start_ptr = grad_ptr + row_idx * grad_row_stride
    grad_ptrs = grad_row_start_ptr + col_offsets

    grad_row = tl.load(grad_ptrs, mask = mask, other = 0.)

    # recompute

    init_a = k * 0
    init_b = -s

    ds = s * 0
    db = s * 0
    last_da = k * 0

    inv_eps = 1. / eps

    # backwards

    for ind in range(n_iters):
        is_first = ind == 0

        a = init_a
        b = init_b

        sa = s * 0
        softmax = s * 0

        for _ in range(n_iters - ind):

            sb = (s + b) * inv_eps
            sb = tl.where(mask, sb, -float('inf'))

            # stable log sum exp

            sb_max = tl.max(sb, axis = 0)
            sb_minus_max = sb - sb_max
            exp = tl.exp(sb_minus_max)
            sum_exp = tl.sum(exp, axis = 0)
            softmax = exp / sum_exp
            log_sum_exp = tl.log(sum_exp) + sb_max

            a = constant - eps * log_sum_exp

            # update b

            sa = s + a
            b = tl.where(sa >= 0., -sa, 0.)

        if is_first:
            o = tl.exp((s + a + b) * inv_eps)

            o = tl.where(mask, o, 0.)

            ds = grad_row * o
            ds *= inv_eps

            ds = tl.where(mask, ds, 0.)

            last_da = tl.sum(ds, axis = 0)
            db = ds

        # go backwards

        if n_iters > 0:

            dsa = db * tl.where(sa >= 0, -1., 0.)
            ds += dsa

            da = tl.sum(dsa, axis = 0) + last_da
            da *= -eps

            dsb = da * softmax * inv_eps

            ds += dsb
            db = dsb

            last_da = 0.

    ds += -db

    # store output

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets

    ds = tl.where(mask, ds, 0.)
    tl.store(output_ptrs, ds)

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
        mask
    ):
        assert x.is_cuda, 'triton coordinate descent must be on cuda'

        batch, requires_grad = x.shape[0], x.requires_grad

        if not exists(mask):
            mask = torch.ones_like(x, dtype = torch.bool, device = x.device)

        x, shape = pack_one(x, '* n')
        mask, _ = pack_one(mask, '* n')

        x = x.masked_fill(~mask, -torch.finfo(x.dtype).max)
        mask_ints = mask.int()

        n_rows, n_cols = x.shape

        if isinstance(k, (int, float)):
            k = torch.full((n_rows,), k)
        elif k.ndim == 1:
            k = repeat(k, 'n -> (b n)', b = x.shape[0] // k.shape[0])

        k = k.to(x)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        y = torch.empty_like(x)

        coor_descent_kernel_forward[(n_rows,)](
            y,
            x,
            mask_ints,
            k,
            y.stride(0),
            x.stride(0),
            mask_ints.stride(0),
            k.stride(0),
            n_iters,
            eps,
            n_cols,
            num_warps = num_warps,
            BLOCK_SIZE = BLOCK_SIZE,
        )

        if requires_grad:
            ctx.args = (n_iters, eps)
            ctx.save_for_backward(x, k, mask)

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

        n_iters, eps = ctx.args
        x, k, mask = ctx.saved_tensors

        if exists(mask):
            grad_probs = grad_probs.masked_fill(~mask, 0.)

        grad_probs, shape = pack_one(grad_probs, '* n')
        n_rows, n_cols = grad_probs.shape

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        dx = torch.empty_like(grad_probs)

        mask_int = mask.int()

        coor_descent_kernel_backward[(n_rows,)](
            dx,
            x,
            mask_int,
            grad_probs,
            k,
            dx.stride(0),
            x.stride(0),
            mask_int.stride(0),
            grad_probs.stride(0),
            k.stride(0),
            n_iters,
            eps,
            n_cols,
            num_warps = num_warps,
            BLOCK_SIZE = BLOCK_SIZE
        )

        dx = unpack_one(dx, shape, '* n')

        return dx, None, None, None, None

def triton_coor_descent(
    s,
    *,
    n_iters,
    k,
    eps = 1e-1,
    mask = None
):
    if not s.is_cuda:
        return coor_descent(s, n_iters = n_iters, k = k, eps = eps, mask = mask)

    return _coor_descent.apply(s, n_iters, k, eps, mask)
