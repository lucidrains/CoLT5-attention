<img src="./colt5.png" width="400px"></img>

## CoLT5 Attention - Pytorch (wip)

Implementation of the conditionally routed efficient attention in the proposed <a href="https://arxiv.org/abs/2303.09752">CoLT5</a> architecture, in Pytorch. Besides their routing, which is based on normalizing the scores and weighting the outputs of the "heavy" modules based on <a href="https://arxiv.org/abs/2211.01267">this paper</a>, will also try using sinkhorn for differentible topk, as I've seen in some mixture of experts papers.


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

# mock input, say it is 8192 length

tokens = torch.randn(2, 8192, 512)
mask = torch.ones(2, 32768).bool()  # can handle variable lengthed sequences

# feedforward

ff = ConditionalRoutedFeedForward(
    dim = 512,
    num_heavy_tokens = 1024   # heavy branch receives only 1024 routed tokens of 8192
)

ff_out = ff(tokens, mask = mask)  # (2, 8192, 512) - light and heavy branch summed

# attention

attn = ConditionalRoutedAttention(
    dim = 512,
    num_heavy_tokens = 1024   # heavy branch receives only 1024 routed tokens of 8192
)

attn_out = attn(tokens, mask = mask) # (2, 8192, 512) - light and heavy branch summed

# both attention and feedforward with residual
# the complete transformer block
# a stack of these would constitute the encoder of CoLT5

block = ConditionalRoutedTransformerBlock(
    dim = 512,
    num_heavy_ff_tokens = 1024,
    num_heavy_attn_tokens = 1024
)

block_out = block(tokens, mask = mask) # (2, 8192, 512)
```


## Citations

```bibtex
@inproceedings{Ainslie2023CoLT5FL,
    title   = {CoLT5: Faster Long-Range Transformers with Conditional Computation},
    author  = {Joshua Ainslie and Tao Lei and Michiel de Jong and Santiago Ontan'on and Siddhartha Brahma and Yury Zemlyanskiy and David Uthus and Mandy Guo and James Lee-Thorp and Yi Tay and Yun-Hsuan Sung and Sumit Sanghai},
    year    = {2023}
}
```
