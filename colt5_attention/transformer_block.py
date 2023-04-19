from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum

from local_attention import LocalMHA
from einops import rearrange, repeat, pack, unpack

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

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

        # if routing dimension is not there, unsqueeze for 1 routing dimension

        if context.ndim == 3:
            context = rearrange(context, 'b n d -> b 1 n d')

        if exists(normalized_scores_kv):
            if normalized_scores_kv.ndim == 2:
                normalized_scores_kv = rearrange(normalized_scores_kv, 'b n -> b 1 n')

            normalized_scores_kv = rearrange(normalized_scores_kv, 'b r n -> b r 1 n 1')

        num_kv_routes = context.shape[1]

        # get queries

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        # handle key / values, with the routing dimension, dividing the number of heads in between the routes

        assert divisible_by(h, num_kv_routes), 'number of heads must be divisible by the number of key / value routes'
        heads_per_route = h // num_kv_routes

        kv_weight = rearrange(self.to_kv.weight, '(r h d) i -> r h d i', h = heads_per_route, r = num_kv_routes)

        kv = einsum('r h d i, b r n i -> b r h n d', kv_weight, context)
        k, v = kv.chunk(2, dim = -1)

        if exists(normalized_scores_kv):
            # in paper, not sure how they passed back the signal from heavy attention to normalized scores for key/values. just multiply the values by the normalized kv scores for now
            v = v * normalized_scores_kv

        k, v = map(lambda t: rearrange(t, 'b r h n d -> b (r h) n d'), (k, v))

        # scale and get similarity

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # masking

        if exists(mask):
            if mask.ndim == 3:
                mask = repeat(mask, 'b r j -> b (r h) 1 j', h = heads_per_route)
            else:
                mask = rearrange(mask, 'b j -> b 1 1 j')

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim = -1)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# routing related logic

class DifferentiableTopKRouter(nn.Module):
    """ differentiable topk using cumulative softmax """

    def __init__(
        self,
        dim,
        straight_through = True,
        temperature = 1.,
        num_routing_tokens = 1
    ):
        super().__init__()
        self.is_one_routing_token = num_routing_tokens == 1
        self.num_routing_tokens = num_routing_tokens
        self.routing_token = nn.Parameter(torch.randn(num_routing_tokens, dim))

        self.straight_through = straight_through
        self.temperature = temperature

    def forward(
        self,
        x,
        *,
        num_tokens,
        mask = None
    ):
        num_routes = self.num_routing_tokens

        # eventual normalized score

        scores = einsum('b n d, r d -> b r n', x, self.routing_token)

        # merge routing dimension into batch

        x = repeat(x, 'b ... -> (b r) ...', r = num_routes)
        scores, ps = pack_one(scores, '* n')

        if exists(mask):
            mask = repeat(mask, 'b ... -> (b r) ...', r = num_routes)

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

        # split out routing dimension again if need be

        if not self.is_one_routing_token:
            selected_scores = unpack_one(selected_scores, ps, '* n')
            selected_indices = unpack_one(selected_indices, ps, '* n')

        return selected_scores, selected_indices

# sinkhorn type routing, with ties to optimal transport

from colt5_attention.sinkhorn import gumbel_sinkhorn, scatter_mean, log

class SinkhornRouter(nn.Module):
    """ gumbel sinkhorn router """
    """ ex. https://arxiv.org/abs/1910.09036 """

    def __init__(
        self,
        dim,
        straight_through = True,
        n_iters = 8,
        temperature = 0.7,
        num_routing_tokens = 1
    ):
        super().__init__()
        self.is_one_routing_token = num_routing_tokens == 1
        self.num_routing_tokens = num_routing_tokens
        self.routing_token = nn.Parameter(torch.randn(num_routing_tokens, dim))

        self.straight_through = straight_through
        self.gumbel_sinkhorn_fn = partial(gumbel_sinkhorn, temperature = temperature, n_iters = n_iters)

    def forward(
        self,
        x,
        *,
        num_tokens,
        mask = None
    ):
        n, num_routes = x.shape[-2], self.num_routing_tokens
        num_tokens = min(n, num_tokens)

        # eventual normalized score

        scores = einsum('b n d, r d -> b r n', x, self.routing_token)

        # merge routing dimension into batch

        x = repeat(x, 'b ... -> (b r) ...', r = num_routes)
        scores, ps = pack_one(scores, '* n')

        if exists(mask):
            mask = repeat(mask, 'b ... -> (b r) ...', r = num_routes)

        # calculate scores

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

        # split out routing dimension again if need be

        if not self.is_one_routing_token:
            selected_scores = unpack_one(selected_scores, ps, '* n')
            selected_indices = unpack_one(selected_indices, ps, '* n')

        return selected_scores, selected_indices

from colt5_attention.coor_descent import coor_descent

class CoordinateDescentRouter(nn.Module):
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
        fetch_k_ratio = 9 / 8,  # in the paper, they do a bit slightly higher k (times this ratio) for better learning
        eps = 1.,               # the epsilon for coordinate descent. in CoLT5 paper they used 1. apparently
        num_routing_tokens = 1
    ):
        super().__init__()
        assert fetch_k_ratio >= 1.
        self.eps = eps

        self.n_iters = n_iters
        self.fetch_k_ratio = fetch_k_ratio

        self.is_one_routing_token = num_routing_tokens == 1
        self.num_routing_tokens = num_routing_tokens
        self.routing_token = nn.Parameter(torch.randn(num_routing_tokens, dim))
        self.straight_through = straight_through

    def forward(
        self,
        x,
        *,
        num_tokens,
        mask = None
    ):
        n, device, eps, num_routes = x.shape[-2], x.device, self.eps, self.num_routing_tokens

        # s stands for eventual normalized score

        s = einsum('b n d, r d -> b r n', x, self.routing_token)

        # merge routing dimension into batch

        x = repeat(x, 'b ... -> (b r) ...', r = num_routes)
        s, ps = pack_one(s, '* n')

        if exists(mask):
            mask = repeat(mask, 'b ... -> (b r) ...', r = num_routes)

        # k, which controls the sparsity of the outputted scores from iterative coordinate descent

        effective_k = min(num_tokens * self.fetch_k_ratio, n)

        k = torch.tensor([effective_k], device = device)

        # coordinate descent

        scores = coor_descent(s, n_iters = self.n_iters, mask = mask, k = k, eps = eps)

        # get the topk scores and indices from the sparse matrix

        selected_scores, selected_indices = scores.topk(num_tokens, dim = -1)

        if self.straight_through:
            # this would make sure all normalized scores returned are 1., but still differentiable using straight-through trick
            selected_scores = selected_scores + (1. - selected_scores).detach()

            if exists(mask):
                batch_range = create_batch_range(x)
                selected_mask = mask[batch_range, selected_indices]
                selected_scores = selected_scores.masked_fill(~selected_mask, 0.)

        # split out routing dimension again if need be

        if not self.is_one_routing_token:
            selected_scores = unpack_one(selected_scores, ps, '* n')
            selected_indices = unpack_one(selected_indices, ps, '* n')

        return selected_scores, selected_indices

# all router types

ROUTERS = dict(
    cum_softmax = DifferentiableTopKRouter,
    sinkhorn = SinkhornRouter,
    coor_descent = CoordinateDescentRouter
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

        # light local attention sees all tokens in a limited context

        light_out = self.light_attn(x, mask = mask)

        # route tokens appropriately for heavy branch

        normalized_scores_q, indices_q = self.q_router(x, num_tokens = num_heavy_tokens_q, mask = mask)
        normalized_scores_kv, indices_kv = self.kv_router(x, num_tokens = num_heavy_tokens_kv, mask = mask)

        # select the tokens to be routed to full attention

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

        # scatter back the output of the heavy branch

        heavy_out = torch.zeros_like(x)

        if self.router_type == 'sinkhorn':
            heavy_out = scatter_mean(heavy_out, routed_tokens_out, indices_q, dim = 1)
        else:
            heavy_out[batch_range, indices_q] = routed_tokens_out

        # sum light and heavy branches

        return light_out + heavy_out

# adapting the conditional routed self attention to cross attention

class ConditionalRoutedCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_tokens_q,
        num_tokens_kv,
        num_sets_kv = 1,                # setting this greater than 1 would route multiple sets of key / values, each of size num_tokens_kv, using this many routing tokens
        dim_head = 64,
        heads = 8,
        router_straight_through = True, # would make sure all normalized scores are 1., still differentiable
        router_type = 'coor_descent',
        router_kwargs: dict = {},
        kv_routing_tokens = 1
    ):
        super().__init__()
        assert router_type in ROUTERS.keys()

        self.router_type = router_type

        router_klass = ROUTERS.get(router_type)

        self.num_tokens_q = num_tokens_q
        self.num_tokens_kv = num_tokens_kv

        self.q_router = router_klass(
            dim = dim,
            straight_through = router_straight_through,
            **router_kwargs
        )

        self.kv_router = router_klass(
            dim = dim,
            straight_through = router_straight_through,
            num_routing_tokens = kv_routing_tokens,
            **router_kwargs
        )

        self.heavy_attn = Attention(
            dim = dim,
            dim_head = dim_head,
            heads = heads
        )

    def forward(
        self,
        x,
        context,
        *,
        num_tokens_q = None,
        num_tokens_kv = None,
        mask = None,
        context_mask = None
    ):
        batch, device = x.shape[0], x.device

        batch_range = torch.arange(batch, device = device)
        batch_range = rearrange(batch_range, 'b -> b 1')

        # route the queries

        query_length = x.shape[-2]
        num_tokens_q = default(num_tokens_q, self.num_tokens_q)

        routed_tokens_q = x
        should_route_queries = query_length > num_tokens_q

        if should_route_queries:
            normalized_scores_q, indices_q = self.q_router(x, num_tokens = num_tokens_q, mask = mask)

            routed_tokens_q = x[batch_range, indices_q]

        # route the long contexts

        key_value_length = context.shape[-2]
        num_tokens_kv = default(num_tokens_kv, self.num_tokens_kv)

        routed_tokens_kv = context
        routed_tokens_kv_mask = context_mask
        normalized_scores_kv = None

        should_route_kv = key_value_length > num_tokens_kv

        if should_route_kv:
            normalized_scores_kv, indices_kv = self.kv_router(context, num_tokens = num_tokens_kv, mask = context_mask)

            routed_tokens_kv = x[batch_range, indices_kv]

            routed_tokens_kv_mask = None
            if exists(context_mask):
                routed_tokens_kv_mask = context_mask[batch_range, indices_kv]

        # do the heavier branch with only routed tokens

        routed_tokens_out = self.heavy_attn(
            routed_tokens_q,
            mask = routed_tokens_kv_mask,
            context = routed_tokens_kv,
            normalized_scores_kv = normalized_scores_kv
        )

        if should_route_queries:
            routed_tokens_out = routed_tokens_out * rearrange(normalized_scores_q, '... -> ... 1')

        # early return if queries did not undergo routing

        if not should_route_queries:
            return routed_tokens_out

        # otherwise, scatter back the query outputs

        out = torch.zeros_like(x)

        if self.router_type == 'sinkhorn':
            out = scatter_mean(heavy_out, routed_tokens_out, indices_q, dim = 1)
        else:
            out[batch_range, indices_q] = routed_tokens_out

        return out

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
