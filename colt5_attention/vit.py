import torch
from torch import nn

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange, Reduce

from colt5_attention.transformer_block import (
    ConditionalRoutedImageAttention,
    ConditionalRoutedFeedForward
)

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    pe = pe.type(dtype)
    return rearrange(pe, '(h w) d -> h w d', h = h, w = w)

# classes

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        attn_num_heavy_tokens_q,
        attn_num_heavy_tokens_kv,
        attn_light_dim_head,
        attn_light_heads,
        attn_light_window_size,
        attn_heavy_dim_head,
        attn_heavy_heads,
        ff_num_heavy_tokens,
        ff_light_mult,
        ff_heavy_mult,
        router_straight_through = True,
        router_kwargs: dict = {},
        router_use_triton = False,
        flash_attn = True,
        attn_num_routed_kv = 1
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):

            ff = ConditionalRoutedFeedForward(
                dim,
                num_heavy_tokens = ff_num_heavy_tokens,
                light_ff_mult = ff_light_mult,
                heavy_ff_mult = ff_heavy_mult,
                router_straight_through = router_straight_through,
                router_kwargs = router_kwargs,
                use_triton = router_use_triton
            )

            attn = ConditionalRoutedImageAttention(
                dim,
                num_heavy_tokens_q = attn_num_heavy_tokens_q,
                num_heavy_tokens_kv = attn_num_heavy_tokens_kv,
                num_routed_kv = attn_num_routed_kv,
                light_dim_head = attn_light_dim_head,
                light_heads = attn_light_heads,
                light_window_size = attn_light_window_size,
                heavy_dim_head = attn_heavy_dim_head,
                heavy_heads = attn_heavy_heads,
                router_straight_through = router_straight_through,
                router_kwargs = router_kwargs,
                use_triton = router_use_triton,
                use_flash_attn = flash_attn,
                channel_first = False,
                use_null_q_tokens = True
            )

            self.layers.append(nn.ModuleList([attn, ff]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x

            x, ps = pack([x], 'b * d')
            x = ff(x) + x            
            x, = unpack(x, ps, 'b * d')

        return x

class ConditionalRoutedViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        attn_num_heavy_tokens_q,
        attn_num_heavy_tokens_kv,
        attn_heavy_dim_head,
        attn_heavy_heads,
        attn_light_dim_head,
        attn_light_heads,
        attn_light_window_size,
        ff_num_heavy_tokens,
        ff_heavy_mult,
        ff_light_mult,
        channels = 3,
        router_straight_through = True,
        router_kwargs: dict = {},
        router_use_triton = False,
        flash_attn = True,
        attn_num_routed_kv = 1
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(
            dim,
            depth,
            attn_num_heavy_tokens_q,
            attn_num_heavy_tokens_kv,
            attn_light_dim_head,
            attn_light_heads,
            attn_light_window_size,
            attn_heavy_dim_head,
            attn_heavy_heads,
            ff_num_heavy_tokens,
            ff_light_mult,
            ff_heavy_mult,
            router_straight_through,
            router_kwargs,
            router_use_triton,
            flash_attn,
            attn_num_routed_kv
        )

        self.linear_head = nn.Sequential(
            Reduce('b h w c -> b c', 'mean'),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        x = x + posemb_sincos_2d(x)        

        x = self.transformer(x)

        return self.linear_head(x)
