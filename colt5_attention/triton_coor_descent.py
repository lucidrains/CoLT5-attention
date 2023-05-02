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
    k_ptr,
    input_row_stride,
    output_row_stride,
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

    mask = col_offsets < n_cols

    # load the scores s

    s = tl.load(input_ptrs, mask = mask, other = -float('inf'))

    # load k - controls the sparsity of output

    k = tl.load(k_ptr)

    constant = tl.log(k) * eps

    inv_eps = 1. / eps

    # initialize a and b for coordinate descent

    a = k * 0                           # init a to 0, triton needs to know shape and type outside loop
    b = -s

    for _ in range(n_iters):        

        a = (s + b) * inv_eps

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

    tl.store(output_ptrs, s, mask = mask)

# backwards

@triton.jit
def coor_descent_kernel_backward(
    output_ptr,
    input_ptr,
    grad_ptr,
    k_ptr,
    k_row_stride,
    grad_row_stride,
    input_row_stride,
    output_row_stride,
    n_iters,
    eps,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

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
        exp = s * 0
        sum_exp = k * 0

        for _ in range(n_iters - ind):

            sb = (s + b) * inv_eps

            # stable log sum exp

            sb_max = tl.max(sb, axis = 0)
            sb_minus_max = sb - sb_max
            exp = tl.exp(sb_minus_max)
            sum_exp = tl.sum(exp, axis = 0)
            log_sum_exp = tl.log(sum_exp) + sb_max

            a = constant - eps * log_sum_exp

            # update b

            sa = s + a
            b = tl.where(sa >= 0., -sa, 0.)

        if is_first:
            o = tl.exp((s + a + b) * inv_eps)

            ds = grad_row * o
            ds *= inv_eps

            last_da = tl.sum(ds, axis = 0)
            db = ds

        # go backwards

        if n_iters > 0:

            dsa = db * tl.where(sa >= 0, -1., 0.)
            ds += dsa

            da = tl.sum(dsa, axis = 0) + last_da
            da *= -eps

            dsb = exp * da / sum_exp
            dsb *= inv_eps

            ds += dsb
            db = dsb

            last_da = 0.

    ds += -db

    # store output

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets

    tl.store(output_ptrs, ds, mask = mask)

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

        if exists(mask):
            mask_value = -torch.finfo(x.dtype).max
            x = x.masked_fill(~mask, mask_value)

        x, shape = pack_one(x, '* n')

        n_rows, n_cols = x.shape

        if isinstance(k, (int, float)):
            k = torch.full((n_rows,), k)

        k = k.to(x)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        y = torch.empty_like(x)

        coor_descent_kernel_forward[(n_rows,)](
            y,
            x,
            k,
            x.stride(0),
            y.stride(0),
            k.stride(0),
            n_iters,
            eps,
            n_cols,
            num_warps = num_warps,
            BLOCK_SIZE = BLOCK_SIZE,
        )

        if requires_grad:
            ctx.args = (n_iters, eps)
            ctx.save_for_backward(x, k)

        y = unpack_one(y, shape, '* n')

        if exists(mask):
            y = y.masked_fill(~mask, 0.)

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
        x, k = ctx.saved_tensors

        grad_probs, shape = pack_one(grad_probs, '* n')
        n_rows, n_cols = grad_probs.shape

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        dx = torch.empty_like(grad_probs)

        coor_descent_kernel_backward[(n_rows,)](
            dx,
            x,
            grad_probs,
            k,
            k.stride(0),
            grad_probs.stride(0),
            x.stride(0),
            dx.stride(0),
            n_iters,
            eps,
            n_cols,
            num_warps = num_warps,
            BLOCK_SIZE = BLOCK_SIZE
        )

        return unpack_one(dx, shape, '* n'), None, None, None, None

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
