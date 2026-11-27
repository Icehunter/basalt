# Qwen3.8-35B-A3B precision

What the `qwen35moe` model computes in, where it rounds, and how far that is from the BF16 model. The method:
[how quality is measured](quality.md).

## Weights

| Tensors | Stored as |
| --- | --- |
| Routed experts (256 a layer, 8 used) | NVFP4 blobs: gate, up and down as 64 values in 36 bytes (four UE4M3 scales, 32 code bytes), then a 16-byte FP32 tail {s_gate, s_up, s_down, 0} |
| Attention, GDN and shared-expert matrices | Q8_0 |
| Router, shared-expert gate | BF16 |
| Output head | F8_E4M3 rows with an FP32 scale per row |
| Token embedding, norms | BF16, FP32 |
| MTP layer | the base model's (drafting only: it cannot change output) |

## Where the experts round

- **Decode windows, and prompts under 64 rows:** each token's normed row is rounded to INT8 with a scale per 16 values. Each
  (token, expert) pair runs its gate/up and down products on INT8 dot products.
- **Prompt chunks of 64 rows or more:** the experts run on the FP4 block-scaled tensor cores, read in place from the blob.
  - A token's row is rounded to FP4 in two steps: the row, then the error that step leaves. Both terms share one FP32 scale for
    the token (E2M1 codes, UE4M3 scale per 16 values). That is about 8 bits, and the product runs over both terms.
  - The gate/up product ends in the SwiGLU with the blob's tail scales, `silu(g s_gate) (u s_up) s_down`, written as FP16.
  - Those rows are rounded to two FP4 terms the same way for the down product, which writes FP16 rows.
  - The layer adds the 8 routed rows, weighted by the router's probabilities, and the shared expert times its sigmoid gate to the
    BF16 residual. The sum is in FP32 and rounds once to BF16.
- A token's output rows do not depend on the other tokens in the chunk: for the same tokens, a batch of 4,096 and a batch of 2,048
  give the same bits.

## Measured error

Real experts from the first layer, random tokens routed as in a chunk, against a double-precision reference (dequantize the blob,
add in double). Relative RMS of the routed output (`qwen35_moe_parity_test`):

| Path | Against the double reference | Against the per-pair INT8 path |
| --- | --- | --- |
| Per-pair INT8 (decode) | 0.8% | |
| FP4, two terms (prompt chunks) | 1.5% | 1.7% |

The FP4 path is noisier per product than the INT8 path. The quality gate's prompt halves (256 tokens, 64 or more rows in a layer)
run it, so the gate's scores include it.

Rounding to one FP4 term instead of two is not used: it is no faster here and it loses quality (mean KLD 0.01404 and top-1 94.36%,
against 0.01303 and 94.63% with two terms, int8 KV).

The gate reads 256 tokens of each chunk as a prompt. Quality against BF16 after a long prompt is not measured.

## Quality

The reference is the BF16 GGUF of the same model through `llama-perplexity` (llama.cpp `b1b770d90`), wikitext-2, 100 chunks of
512, `qwen35moe-bf16.kld`, BF16 perplexity 6.6994. `basalt-bench --quality 100`, T 1:

| KV cache | PPL | Mean KLD | Median KLD | p99 KLD | Top-1 |
| --- | --- | --- | --- | --- | --- |
| fp16 | 6.7335 | 0.01289 | 0.00691 | 0.1075 | 94.73% |
| int8 | 6.7336 | 0.01284 | 0.00691 | 0.1072 | 94.73% |
| nvfp4 | 6.7538 | 0.01630 | 0.00916 | 0.1250 | 93.99% |

- int8 KV scores the same as fp16 within the gate's noise (about 2% of the mean KLD).
- nvfp4 KV has 27% more mean KLD than int8 and 0.74 points less top-1. The rest of the 0.0128 comes from the weights and the
  products.

Reproduce, with `qwen35moe-bf16.kld` from the pack's Hugging Face repository:

```sh
basalt-bench --model PACK.basalt --quality 100 --kld-base qwen35moe-bf16.kld -- --kv int8
```
