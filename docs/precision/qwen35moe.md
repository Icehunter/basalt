# Qwen3.8-35B-A3B precision

Where the `qwen35moe` model computes in what, where it rounds, and what that costs against the BF16 model. The method:
[how quality is measured](quality.md).

## Weights

| Tensors | Stored as |
| --- | --- |
| Routed experts (256 a layer, 8 used) | NVFP4 blobs: gate, up and down as 64 values in 36 bytes (four UE4M3 scales, 32 code bytes), then a 16-byte FP32 tail {s_gate, s_up, s_down, 0} |
| Attention, GDN and shared-expert matrices | Q8_0 |
| Router, shared-expert gate | BF16 |
| Output head | F8_E4M3 rows with an FP32 scale per row |
| Token embedding, norms | BF16, FP32 |
| MTP layer | the base model's (drafting only: cannot change output) |

## Where the experts round

- **Decode windows and prompts under 64 rows:** each token's normed row is rounded to INT8 with a scale per 16 values, and each
  (token, expert) pair runs its own gate/up and down products on INT8 dot products.
- **Prompt chunks of 64 rows or more:** the experts run on the FP4 block-scaled tensor cores, read in place from the blob.
  - A token's row is rounded once to FP4 as two terms: its FP4 rounding and the FP4 rounding of what that leaves (E2M1 codes,
    UE4M3 scale per 16 values, one FP32 scale per token), about 8 bits, and the product runs over both terms.
  - The gate/up product ends in the tail-scaled SwiGLU, `silu(g s_gate) (u s_up) s_down`, written as FP16.
  - Those rows are rounded to two FP4 terms the same way for the down product, whose rows are written as FP16.
  - The layer's output is the BF16 residual plus the 8 routed rows weighted by the router's probabilities plus the
    sigmoid-gated shared expert, added in FP32 and rounded once.
- A token's output rows do not depend on the other tokens of the chunk: a batch of 4,096 tokens and one of 2,048 give the same
  bits for the same tokens.

## Measured costs

Real experts of the first layer, random tokens routed as in a chunk, against a double-precision reference (dequantize the
blob, accumulate in double), relative RMS of the layer's routed output (`qwen35_moe_parity_test`):

| Path | Against the double reference | Against the per-pair INT8 path |
| --- | --- | --- |
| Per-pair INT8 (decode) | 0.8% | |
| FP4 two terms (prompt chunks) | 1.5% | 1.7% |
| MMQ INT8 (the prompt path before the FP4 products) | not measured | 1.0% |

The FP4 products are noisier than the INT8 ones, and the model does not show it: the quality gate's prompt halves run these
products (256-token prefill, 64 or more rows a layer).

One FP4 term in place of two is not used: it was no faster and costs quality (mean KLD 0.01404, top-1 94.36% against 0.01303
and 94.63% with two terms, int8 KV).

## Quality

The reference is the BF16 GGUF of the same model through `llama-perplexity` (llama.cpp `b1b770d90`), wikitext-2, 100 chunks of
512, `qwen35moe-bf16.kld`, BF16 perplexity 6.6994. `basalt-bench --quality 100`, T 1:

| KV cache | PPL | Mean KLD | Median KLD | p99 KLD | Top-1 |
| --- | --- | --- | --- | --- | --- |
| fp16 | 6.7335 | 0.01289 | 0.00691 | 0.1075 | 94.73% |
| int8 | 6.7336 | 0.01284 | 0.00691 | 0.1072 | 94.73% |
| nvfp4 | 6.7538 | 0.01630 | 0.00916 | 0.1250 | 93.99% |

- int8 KV is the same as fp16 within the gate's noise (about 2% of the mean KLD).
- nvfp4 KV costs 27% in mean KLD and 0.74 points of top-1 against int8; the rest of the 0.0128 comes from the weights and the
  decode and prompt products.
- The prompt products' change (MMQ INT8 to FP4 two terms, with FP16 expert rows) moved int8 KV from KLD 0.01302 and top-1 94.54%
  (PPL 6.7336) to 0.01284 and 94.73% (PPL 6.7336), and nvfp4 KV from 0.01637 and 94.15% (PPL 6.7477) to 0.01630 and 93.99%
  (PPL 6.7538): no change beyond the gate's noise.

Reproduce, with `qwen35moe-bf16.kld` from the pack's Hugging Face repository:

```sh
basalt-bench --model PACK.basalt --quality 100 --kld-base qwen35moe-bf16.kld -- --kv int8
```
