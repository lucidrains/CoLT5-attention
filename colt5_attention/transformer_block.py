from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum

from local_attention import LocalMHA
from einops import rearrange, repeat

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# tensor helpers

def create_batch_range(t):
    b, device = t.shape[0], t.device
    batch_range = torch.arange(b, device = device)
    return rearrange(batch_range, 'b -> b 1')

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
        straight_through = True,
        temperature = 1.
    ):
        super().__init__()
        self.routing_token = nn.Parameter(torch.randn(dim))
        self.straight_through = straight_through
        self.temperature = temperature

    def forward(
        self,
        x,
        *,
        num_tokens,
        mask = None
    ):
        scores = einsum('b n d, d -> b n', x, self.routing_token)

        scores = scores / self.temperature

        if exists(mask):
            mask_value = -torch.finfo(scores.dtype).max
            scores = scores.masked_fill(~mask, mask_value)

        scores, indices = scores.sort(dim = -1)

        scores = scores - scores.amax(dim = -1, keepdim = True).detach()

        exp_scores = scores.exp()

        cum_softmax = exp_scores / exp_scores.cumsum(dim = -1).clamp(min = 1e-6)

        selected_scores, selected_indices = map(lambda t: t[:, -num_tokens:], (cum_softmax, indices))

        if self.straight_through:
            # this would make sure all normalized scores returned are 1., but still differentiable using straight-through trick
            selected_scores = selected_scores + (1. - selected_scores).detach()

            if exists(mask):
                batch_range = create_batch_range(x)
                selected_mask = mask[batch_range, selected_indices]
                selected_scores = selected_scores.masked_fill(~selected_mask, 0.)

        return selected_scores, selected_indices

# sinkhorn type routing, with ties to optimal transport

from colt5_attention.sinkhorn import gumbel_sinkhorn, scatter_mean, log

class SinkhornRouter(nn.Module):
    """ gumbel sinkhorn router """

    def __init__(
        self,
        dim,
        straight_through = True,
        n_iters = 8,
        temperature = 0.7
    ):
        super().__init__()
        self.routing_token = nn.Parameter(torch.randn(dim))

        self.straight_through = straight_through
        self.gumbel_sinkhorn_fn = partial(gumbel_sinkhorn, temperature = temperature, n_iters = n_iters)

    def forward(
        self,
        x,
        *,
        num_tokens,
        mask = None
    ):
        n = x.shape[-2]
        num_tokens = min(n, num_tokens)

        scores = einsum('b n d, d -> b n', x, self.routing_token)

        scores = repeat(scores, '... j -> ... i j', i = num_tokens)

        if exists(mask):
            mask_value = -torch.finfo(scores.dtype).max
            sinkhorn_mask = rearrange(mask, 'b j -> b 1 j')
            scores = scores.masked_fill(~sinkhorn_mask, mask_value)

        # sinkhorn

        scores = self.gumbel_sinkhorn_fn(scores)

        # mask again just in case

        if exists(mask):
            scores = scores.masked_fill(~sinkhorn_mask, mask_value)

        selected_scores, selected_indices = scores.topk(1, dim = -1)
        selected_scores, selected_indices = map(lambda t: rearrange(t, '... 1 -> ...'), (selected_scores, selected_indices))

        if self.straight_through:
            # this would make sure all normalized scores returned are 1., but still differentiable using straight-through trick
            selected_scores = selected_scores + (1. - selected_scores).detach()

            if exists(mask):
                batch_range = create_batch_range(x)
                selected_mask = mask[batch_range, selected_indices]
                selected_scores = selected_scores.masked_fill(~selected_mask, 0.)

        return selected_scores, selected_indices

class CoordinateDescent(nn.Module):
    """
    from Wright et al. https://arxiv.org/abs/1502.04759
    then adopted by https://arxiv.org/abs/2211.01267 for multi-vector document retrieval by Qian et al
    finally, used successfully by this paper for routing to heavy branch attention / feedforward
    """

    def __init__(
        self,
        dim,
        straight_through = True,
        n_iters = 50,           # 50 iterations in the paper
        k = 8,                  # k value in coordinate descent, in paper this value was 1, 2, 4, 6, or 8 - controls the sparsity
        fetch_k_ratio = 9 / 9,  # in the paper, they do a bit slightly higher k (times this ratio) for better learning
        eps = 1e-1              # the epsilon for coordinate descent, values in the paper range from 1e-3 to 1e-2
    ):
        super().__init__()
        assert fetch_k_ratio >= 1.
        self.eps = eps

        self.n_iters = n_iters
        self.k = k
        self.fetch_k_ratio = fetch_k_ratio

        self.routing_token = nn.Parameter(torch.randn(dim))
        self.straight_through = straight_through

    def forward(
        self,
        x,
        *,
        num_tokens,
        mask = None
    ):
        n, device, mask_value, eps = x.shape[-2], x.device, -torch.finfo(x.dtype).max, self.eps
        num_tokens = min(num_tokens, n)

        # s stands for eventual normalized score

        s = einsum('b n d, d -> b n', x, self.routing_token)

        # k, which controls the sparsity of the outputted scores from iterative coordinate descent

        k = torch.tensor([self.k], device = device)

        # coordinate descent

        constant = eps * log(k)
        b = -s.relu()

        for _ in range(self.n_iters):
            if exists(mask):
                s = s.masked_fill(~mask, mask_value)

            a = constant - eps * ((s + b) / eps).logsumexp(dim = -1, keepdim = True)
            b = -(s + a).relu()

        # calculate final score

        if exists(mask):
            # mask again just in case
            s = s.masked_fill(~mask, mask_value)

        scores = ((s + a + b) / eps).exp()

        # get the topk scores and indices from the sparse matrix

        selected_scores, selected_indices = scores.topk(num_tokens, dim = -1)

        if self.straight_through:
            # this would make sure all normalized scores returned are 1., but still differentiable using straight-through trick
            selected_scores = selected_scores + (1. - selected_scores).detach()

            if exists(mask):
                batch_range = create_batch_range(x)
                selected_mask = mask[batch_range, selected_indices]
                selected_scores = selected_scores.masked_fill(~selected_mask, 0.)

        return selected_scores, selected_indices

# all router types

ROUTERS = dict(
    cum_softmax = DifferentiableTopKRouter,
    sinkhorn = SinkhornRouter,
    coor_descent = CoordinateDescent
)

# main classes

class ConditionalRoutedFeedForward(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_heavy_tokens,
        light_ff_mult = 0.5,
        heavy_ff_mult = 4,
        router_straight_through = True, # would make sure all normalized scores are 1., still differentiable
        router_type = 'coor_descent',
        router_kwargs: dict = {}
    ):
        super().__init__()
        assert router_type in ROUTERS.keys()

        self.num_heavy_tokens = num_heavy_tokens

        self.router_type = router_type

        router_klass = ROUTERS.get(router_type)

        self.router = router_klass(
            dim = dim,
            straight_through = router_straight_through,
            **router_kwargs
        )

        self.light_ff = FeedForward(dim, light_ff_mult)
        self.heavy_ff = FeedForward(dim, heavy_ff_mult)

    def forward(
        self,
        x,
        mask = None,
        num_heavy_tokens = None
    ):
        device, num_heavy_tokens = x.device, default(num_heavy_tokens, self.num_heavy_tokens)

        batch_range = create_batch_range(x)

        # light feedforward sees all the tokens (hidden dimension is only 1/2 of model dimensions)

        light_out = self.light_ff(x)

        # route tokens appropriately for heavy branch

        normalized_scores, indices = self.router(x, num_tokens = num_heavy_tokens, mask = mask)

        # select the tokens to be routed to heavier feedforward (hidden dimension is 4 times model dimensions)

        routed_tokens = x[batch_range, indices]

        # do the heavier branch with only routed tokens

        routed_tokens_out = self.heavy_ff(routed_tokens) * rearrange(normalized_scores, '... -> ... 1')

        # scatter back the output of the heavy feedforward branch

        heavy_out = torch.zeros_like(x)

        if self.router_type == 'sinkhorn':
            heavy_out = scatter_mean(heavy_out, routed_tokens_out, indices, dim = 1)
        else:
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
        light_dim_head = 64,
        light_heads = 8,
        light_window_size = 128,        # each token would see ~ 64 tokens either way to left or right
        heavy_dim_head = 64,
        heavy_heads = 8,
        router_straight_through = True, # would make sure all normalized scores are 1., still differentiable
        router_type = 'coor_descent',
        router_kwargs: dict = {}
    ):
        super().__init__()
        assert router_type in ROUTERS.keys()

        self.router_type = router_type

        router_klass = ROUTERS.get(router_type)

        self.num_heavy_tokens_q = num_heavy_tokens_q
        self.num_heavy_tokens_kv = num_heavy_tokens_kv

        self.light_attn = LocalMHA(
            dim = dim,
            dim_head = light_dim_head,
            heads = light_heads,
            window_size = light_window_size // 2,
            prenorm = True,
            causal = False,
            use_rotary_pos_emb = False,
            look_backward = 1,
            look_forward = 1
        )

        # for now, just do qkv for starters, need to separate to q and kv

        self.q_router = router_klass(
            dim = dim,
            straight_through = router_straight_through,
            **router_kwargs
        )

        self.kv_router = router_klass(
            dim = dim,
            straight_through = router_straight_through,
            **router_kwargs
        )

        self.heavy_attn = Attention(
            dim = dim,
            dim_head = heavy_dim_head,
            heads = heavy_heads
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

        normalized_scores_q, indices_q = self.q_router(x, num_tokens = num_heavy_tokens_q, mask = mask)
        normalized_scores_kv, indices_kv = self.kv_router(x, num_tokens = num_heavy_tokens_kv, mask = mask)

        # select the tokens to be routed to heavier feedforward (hidden dimension is 4 times model dimensions)

        routed_tokens_q = x[batch_range, indices_q]
        routed_tokens_kv = x[batch_range, indices_kv]

        # calculate key padding mask

        routed_tokens_kv_mask = None
        if exists(mask):
            routed_tokens_kv_mask = mask[batch_range, indices_kv]

        # do the heavier branch with only routed tokens

        routed_tokens_out = self.heavy_attn(
            routed_tokens_q,
            mask = routed_tokens_kv_mask,
            context = routed_tokens_kv,
            normalized_scores_kv = normalized_scores_kv
        )

        routed_tokens_out = routed_tokens_out * rearrange(normalized_scores_q, '... -> ... 1')

        # scatter back the output of the heavy feedforward branch

        heavy_out = torch.zeros_like(x)

        if self.router_type == 'sinkhorn':
            heavy_out = scatter_mean(heavy_out, routed_tokens_out, indices_q, dim = 1)
        else:
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
        light_dim_head = 64,
        light_heads = 8,
        light_window_size = 128,
        heavy_dim_head = 64,
        heavy_heads = 8,
        light_ff_mult = 0.5,
        heavy_ff_mult = 4,
        router_straight_through = True,
        router_type = 'coor_descent',
        router_kwargs: dict = {}
    ):
        super().__init__()
        self.conditional_ff = ConditionalRoutedFeedForward(
            dim,
            num_heavy_tokens = num_heavy_ff_tokens,
            light_ff_mult = light_ff_mult,
            heavy_ff_mult = heavy_ff_mult,
            router_straight_through = router_straight_through,
            router_type = router_type,
            router_kwargs = router_kwargs
        )

        self.conditional_attn = ConditionalRoutedAttention(
            dim,
            light_dim_head = light_dim_head,
            light_heads = light_heads,
            light_window_size = light_window_size,
            heavy_dim_head = heavy_dim_head,
            heavy_heads = heavy_heads,
            num_heavy_tokens_q = num_heavy_attn_tokens_q,
            num_heavy_tokens_kv = num_heavy_attn_tokens_kv,
            router_straight_through = router_straight_through,
            router_type = router_type,
            router_kwargs = router_kwargs
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
        x = self.conditional_ff(x, mask = mask, num_heavy_tokens = num_heavy_ff_tokens) + x
        return x
