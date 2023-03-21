import torch
import torch.nn.functional as F
from torch import nn, einsum

from local_attention import LocalMHA
from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

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
        self.to_q = nn.Linear(dim, dim_hidden, bias = False)
        self.to_kv = nn.Linear(dim, dim_hidden * 2, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        normalized_scores_kv = None
    ):
        h = self.heads

        x = self.norm(x)

        if exists(context):
            context = self.norm(context)

        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        if exists(normalized_scores_kv):
            # in paper, not sure how they passed back the signal from heavy attention to normalized scores for key/values. just multiply the values by the normalized kv scores for now
            v = v * rearrange(normalized_scores_kv, 'b n -> b 1 n 1')

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

class DifferentiableTopKRouter(nn.Module):
    """ differentiable topk using cumulative softmax """

    def __init__(
        self,
        dim,
        straight_through = True
    ):
        super().__init__()
        self.routing_token = nn.Parameter(torch.randn(dim))
        self.straight_through = straight_through

    def forward(
        self,
        x,
        *,
        num_tokens,
        temperature = 1.
    ):
        assert temperature > 0

        scores = einsum('b n d, d -> b n', x, self.routing_token)

        scores = scores / temperature

        scores, indices = scores.sort(dim = -1, descending = True)

        scores = scores - scores.amax(dim = -1, keepdim = True).detach()

        exp_scores = scores.exp()

        cum_softmax = exp_scores / exp_scores.cumsum(dim = -1).clamp(min = 1e-6)

        selected_scores, selected_indices = map(lambda t: t[:, -num_tokens:], (cum_softmax, indices))

        if self.straight_through:
            # this would make sure all normalized scores returned are 1., but still differentiable using straight-through trick
            selected_scores = selected_scores + (1. - selected_scores).detach()

        return selected_scores, selected_indices

# main classes

class ConditionalRoutedFeedForward(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_heavy_tokens,
        light_ff_mult = 0.5,
        heavy_ff_mult = 4,
        router_straight_through = True # would make sure all normalized scores are 1., still differentiable
    ):
        super().__init__()
        self.num_heavy_tokens = num_heavy_tokens

        self.router = DifferentiableTopKRouter(
            dim = dim,
            straight_through = router_straight_through
        )

        self.light_ff = FeedForward(dim, light_ff_mult)
        self.heavy_ff = FeedForward(dim, heavy_ff_mult)

    def forward(
        self,
        x,
        num_heavy_tokens = None
    ):
        batch, device, num_heavy_tokens = x.shape[0], x.device, default(num_heavy_tokens, self.num_heavy_tokens)

        batch_range = torch.arange(batch, device = device)
        batch_range = rearrange(batch_range, 'b -> b 1')

        # light feedforward sees all the tokens (hidden dimension is only 1/2 of model dimensions)

        light_out = self.light_ff(x)

        # route tokens appropriately for heavy branch

        normalized_scores, indices = self.router(x, num_tokens = num_heavy_tokens)

        # select the tokens to be routed to heavier feedforward (hidden dimension is 4 times model dimensions)

        routed_tokens = x[batch_range, indices]

        # do the heavier branch with only routed tokens

        routed_tokens_out = self.heavy_ff(routed_tokens) * rearrange(normalized_scores, '... -> ... 1')

        # scatter back the output of the heavy feedforward branch

        heavy_out = torch.zeros_like(x)
        heavy_out[batch_range, indices] = routed_tokens_out

        # sum light and heavy branches

        return light_out + heavy_out

class ConditionalRoutedAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_heavy_tokens_q,
        num_heavy_tokens_kv,
        dim_head = 64,
        heads = 8,
        light_window_size = 128,
        router_straight_through = True # would make sure all normalized scores are 1., still differentiable
    ):
        super().__init__()
        self.num_heavy_tokens_q = num_heavy_tokens_q
        self.num_heavy_tokens_kv = num_heavy_tokens_kv

        self.light_attn = LocalMHA(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            window_size = light_window_size,
            prenorm = True
        )

        # for now, just do qkv for starters, need to separate to q and kv

        self.q_router = DifferentiableTopKRouter(
            dim = dim,
            straight_through = router_straight_through
        )

        self.kv_router = DifferentiableTopKRouter(
            dim = dim,
            straight_through = router_straight_through
        )

        self.heavy_attn = Attention(
            dim = dim,
            dim_head = dim_head,
            heads = heads
        )

    def forward(
        self,
        x,
        *,
        num_heavy_tokens_q = None,
        num_heavy_tokens_kv = None,
        mask = None
    ):
        batch, device = x.shape[0], x.device

        num_heavy_tokens_q = default(num_heavy_tokens_q, self.num_heavy_tokens_q)
        num_heavy_tokens_kv = default(num_heavy_tokens_kv, self.num_heavy_tokens_kv)

        batch_range = torch.arange(batch, device = device)
        batch_range = rearrange(batch_range, 'b -> b 1')

        # light local attention sees all tokens

        light_out = self.light_attn(x, mask = mask)

        # route tokens appropriately for heavy branch

        normalized_scores_q, indices_q = self.q_router(x, num_tokens = num_heavy_tokens_q)
        normalized_scores_kv, indices_kv = self.kv_router(x, num_tokens = num_heavy_tokens_kv)

        # select the tokens to be routed to heavier feedforward (hidden dimension is 4 times model dimensions)

        routed_tokens_q = x[batch_range, indices_q]
        routed_tokens_kv = x[batch_range, indices_q]

        # do the heavier branch with only routed tokens

        routed_tokens_out = self.heavy_attn(
            routed_tokens_q,
            context = routed_tokens_kv,
            normalized_scores_kv = normalized_scores_kv
        )

        routed_tokens_out = routed_tokens_out * rearrange(normalized_scores_q, '... -> ... 1')

        # scatter back the output of the heavy feedforward branch

        heavy_out = torch.zeros_like(x)
        heavy_out[batch_range, indices_q] = routed_tokens_out

        # sum light and heavy branches

        return light_out + heavy_out

# block

class ConditionalRoutedTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_heavy_attn_tokens_q,
        num_heavy_attn_tokens_kv,
        num_heavy_ff_tokens,
        dim_head = 64,
        heads = 8,
        light_ff_mult = 0.5,
        heavy_ff_mult = 4,
    ):
        super().__init__()
        self.conditional_ff = ConditionalRoutedFeedForward(
            dim,
            num_heavy_tokens = num_heavy_ff_tokens,
            light_ff_mult = light_ff_mult,
            heavy_ff_mult = heavy_ff_mult
        )

        self.conditional_attn = ConditionalRoutedAttention(
            dim,
            num_heavy_tokens_q = num_heavy_attn_tokens_q,
            num_heavy_tokens_kv = num_heavy_attn_tokens_kv,
            dim_head = dim_head,
            heads = heads
        )

    def forward(
        self,
        x,
        mask = None,
        num_heavy_attn_tokens_q = None,
        num_heavy_attn_tokens_kv = None,
        num_heavy_ff_tokens = None
    ):
        x = self.conditional_attn(x, mask = mask, num_heavy_tokens_q = num_heavy_attn_tokens_q, num_heavy_tokens_kv = num_heavy_attn_tokens_kv) + x
        x = self.conditional_ff(x, num_heavy_tokens = num_heavy_ff_tokens) + x
        return x
