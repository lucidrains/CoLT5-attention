<img src="./colt5.png" width="450px"></img>

## CoLT5 Attention - Pytorch (wip)

Implementation of the conditionally routed efficient attention in the proposed <a href="https://arxiv.org/abs/2303.09752">CoLT5</a> architecture, in Pytorch. Besides their routing, which is based on normalizing the scores and weighting the outputs of the "heavy" modules based on <a href="https://arxiv.org/abs/2211.01267">this paper</a>, will also try using sinkhorn for differentible topk, as I've seen in some mixture of experts papers.

## Citations

```bibtex
@inproceedings{Ainslie2023CoLT5FL,
    title   = {CoLT5: Faster Long-Range Transformers with Conditional Computation},
    author  = {Joshua Ainslie and Tao Lei and Michiel de Jong and Santiago Ontan'on and Siddhartha Brahma and Yury Zemlyanskiy and David Uthus and Mandy Guo and James Lee-Thorp and Yi Tay and Yun-Hsuan Sung and Sumit Sanghai},
    year    = {2023}
}
```
