<img src="./colt5.png" width="400px"></img>

## CoLT5 Attention - Pytorch

Implementation of the conditionally routed efficient attention in the proposed <a href="https://arxiv.org/abs/2303.09752">CoLT5</a> architecture, in Pytorch.

Besides their use of coordinate descent from <a href="https://arxiv.org/abs/2211.01267">this paper</a> (main algorithm originally from <a href="https://arxiv.org/abs/1502.04759">Wright et al</a>), will also add two other approaches, one based on cumulative softmax, the other gumbel sinkhorn (optimal transport).

Update: unsure of how the routing normalized scores for the key-values are used. Did some improvising there, multiplying it into the projected values, but if you think you know the answer, please open an issue

## Appreciation

- <a href="https://stability.ai/">Stability.ai</a> for the generous sponsorship to work on cutting edge artificial intelligence research

## Install

```bash
$ pip install colt5-attention
```

## Usage

```python
import torch

from colt5_attention import (
    ConditionalRoutedFeedForward,
    ConditionalRoutedAttention,
    ConditionalRoutedTransformerBlock
)

# mock input, say it is 32768 length

tokens = torch.randn(2, 32768, 512)
mask = torch.ones(2, 32768).bool()  # can handle variable lengthed sequences

# feedforward

ff = ConditionalRoutedFeedForward(
    dim = 512,
    light_ff_mult = 0.5,      # hidden dimension ratio of light branch
    heavy_ff_mult = 4,        # hidden dimension ratio of heavy branch
    num_heavy_tokens = 1024   # heavy branch receives only 1024 routed tokens of 32768
)

ff_out = ff(tokens, mask = mask)  # (2, 32768, 512) - light and heavy branch summed

# attention

attn = ConditionalRoutedAttention(
    dim = 512,
    light_dim_head = 64,       # attention head dimension of light branch
    light_heads = 8,           # number of attention heads for light branch
    light_window_size = 128,   # local attention receptive field for light
    heavy_dim_head = 64,       # attention head dimension of heavy branch
    heavy_heads = 8,           # number of attention heads for heavy branch
    num_heavy_tokens_q = 1024, # heavy branch receives only 1024 routed tokens of 32768
    num_heavy_tokens_kv = 1024 # heavy branch receives only 1024 routed tokens of 32768
)

attn_out = attn(tokens, mask = mask) # (2, 32768, 512) - light and heavy branch summed

# both attention and feedforward with residual
# the complete transformer block
# a stack of these would constitute the encoder of CoLT5

block = ConditionalRoutedTransformerBlock(
    dim = 512,
    light_dim_head = 64,
    light_heads = 8,
    light_window_size = 128,
    heavy_dim_head = 64,
    heavy_heads = 8,
    light_ff_mult = 0.5,
    heavy_ff_mult = 4,
    num_heavy_ff_tokens = 1024,
    num_heavy_attn_tokens_q = 1024,
    num_heavy_attn_tokens_kv = 1024,
    router_type = 'coor_descent'  # you have your choice of coordinate descent, as in paper - or 'sinkhorn' or 'cum_softmax'
)

block_out = block(tokens, mask = mask) # (2, 32768, 512)
```

## Todo

- [x] add the coordinate descent method as another router
- [ ] for variable sequence lengths, allow for setting k as a function of sequence lengths per sample in batch

## Citations

```bibtex
@inproceedings{Ainslie2023CoLT5FL,
    title   = {CoLT5: Faster Long-Range Transformers with Conditional Computation},
    author  = {Joshua Ainslie and Tao Lei and Michiel de Jong and Santiago Ontan'on and Siddhartha Brahma and Yury Zemlyanskiy and David Uthus and Mandy Guo and James Lee-Thorp and Yi Tay and Yun-Hsuan Sung and Sumit Sanghai},
    year    = {2023}
}
```
