from math import log

import torch
from torch import Tensor
from torch import autograd
import torch.nn.functional as F

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

    # initialize a and b for coordinate descent

    a = k * 0                           # init a to 0, triton needs to know shape and type outside loop
    b = tl.where(s >= 0., -s, 0.)

    for _ in range(n_iters):        

        a = (s + b) / eps

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

    s = tl.exp((s + a + b) / eps)

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets

    tl.store(output_ptrs, s, mask = mask)

# backwards

@triton.jit
def coor_descent_kernel_backward(
    output_ptr,
    input_ptr,
    grad_ptr,
    grad_row_stride,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)

    row_start_ptr = input_ptr + row_idx * input_row_stride
    grad_row_start_ptr = grad_ptr + row_idx * grad_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    grad_ptrs = grad_row_start_ptr + col_offsets

    mask = col_offsets < n_cols

    probs_row = tl.load(input_ptrs, mask = mask, other = 0.)
    grad_row = tl.load(grad_ptrs, mask = mask, other = 0.)

    dxhat = probs_row * grad_row
    softmax_grad_output = dxhat - probs_row * tl.sum(dxhat, axis = 0)

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_grad_output, mask = mask)

# function forwards and backwards

class _coor_descent(autograd.Function):
    @classmethod
    def forward(
        self,
        ctx,
        x,
        n_iters,
        k,
        eps
    ):
        assert x.is_cuda

        batch = x.shape[0]

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

        return unpack_one(y, shape, '* n')

    @classmethod
    def backward(
        self,
        ctx,
        grad_probs
    ):
        assert grad_probs.is_cuda

        batch = grad_probs.shape[0]

        probs, = ctx.saved_tensors

        grad_probs, shape = pack_one(grad_probs, '* n')
        n_rows, n_cols = grad_probs.shape

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        dx = torch.empty_like(probs)

        coor_descent_kernel_backward[(n_rows,)](
            dx,
            probs,
            grad_probs,
            grad_probs.stride(0),
            probs.stride(0),
            dx.stride(0),
            n_cols,
            num_warps = num_warps,
            BLOCK_SIZE = BLOCK_SIZE
        )

        return unpack_one(dx, shape, '* n'), None, None, None

triton_coor_descent = _coor_descent.apply
