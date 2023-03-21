import torch
import torch.nn.functional as F
from torch import nn, einsum

from local_attention import LocalMHA
from einops import rearrange

# helper functions

def exists(val):
    return val is not None

# normalization

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma

# modules

def FeedForward(dim, mult = 4):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim_hidden),
        nn.GELU(),
        nn.Linear(dim_hidden, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_hidden = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_hidden * 3, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim, bias = False)

    def forward(self, x, mask = None):
        h = self.heads

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# routing related logic

# main classes

class ConditionalRoutedFeedForward(nn.Module):
    def __init__(
        self,
        dim,
        light_ff_mult = 0.5,
        heavy_ff_mult = 4
    ):
        super().__init__()
        self.light_ff = FeedForward(dim, light_ff_mult)
        self.heavy_ff = FeedForward(dim, heavy_ff_mult)

    def forward(self, x):
        light_out = self.light_ff(x)
        heavy_out = self.heavy_ff(x)

        return light_out + heavy_out

class ConditionalRoutedAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        light_window_size = 128
    ):
        super().__init__()

        self.light_attn = LocalMHA(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            window_size = light_window_size,
            prenorm = True
        )

        self.heavy_attn = Attention(
            dim = dim,
            dim_head = dim_head,
            heads = heads
        )

    def forward(self, x, mask = None):
        light_out = self.light_attn(x, mask = mask)
        heavy_out = self.heavy_attn(x, mask = mask)
        return light_out + heavy_out

# block

class ConditionalRoutedTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        light_ff_mult = 0.5,
        heavy_ff_mult = 4
    ):
        super().__init__()
        self.conditional_ff = ConditionalRoutedFeedForward(
            dim,
            light_ff_mult = light_ff_mult,
            heavy_ff_mult = heavy_ff_mult
        )

        self.conditional_attn = ConditionalRoutedAttention(
            dim,
            dim_head = dim_head,
            heads = heads
        )

    def forward(self, x, mask = None):
        x = self.conditional_attn(x, mask = mask) + x
        x = self.conditional_ff(x) + x
        return x
