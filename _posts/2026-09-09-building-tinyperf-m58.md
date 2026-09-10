---
layout: post
math: true
title: "Building tinyperf M58: Compressed attention: a million tokens, and the indexer that pays for them"
date: 2026-09-09 18:54:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "DeepSeek-V4's compressed sparse and heavily compressed attention, modeled from the public report and configs: the cache shrinks to 8.7% of a latent cache, decode barely notices a million tokens, prefill at a million is 78% indexer — and pricing it exposed two mistakes in the milestone-41 indexer."
---

*Milestone 58 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `compress_ratios` in `nets/transformer.py`, presets `deepseek_v4_flash` / `deepseek_v4_pro` · Example: `examples/37_compressed_attention.py`.*

DeepSeek-V4 (technical report arXiv 2606.19348; configs on Hugging Face)
serves a million tokens of context by compressing the KV cache *along the
sequence*. Two mechanisms, mixed by layer: compressed sparse attention
(CSA) pools every 4 tokens into one latent entry, runs an indexer over the
compressed keys and attends to the top 512 of them; heavily compressed
attention (HCA) pools 128 tokens into one entry and attends densely over
the short cache that remains. Every layer also keeps a 128-token window
of raw latents. This milestone adds both, the pieces around them, and
presets for the two released sizes — and finds two mistakes in the
milestone-41 indexer along the way.

## What the model prices

Per layer, a compression ratio: 1 (window only), 4 (CSA) or 128 (HCA).
The attention is multi-query on a 512-dim latent — queries of 512+64 per
head against one shared key/value stream, so there are no k/v
up-projections. A compressed layer adds a compressor (a projection of the
latent to pooling weights and values, softmax over the group, a weighted
reduce), and a CSA layer adds the indexer: 64 heads of 128 scoring every
*compressed* key, with its own compressor. Around the attention: a
grouped low-rank output projection (8 groups down to rank 1024, one
projection up), manifold hyper-connections that widen the residual
stream 4× and mix it per sub-block, and a MoE (256 routed experts, top-6
plus one shared, folded into `top_k=7`) whose first three layers route by
hash and carry no router. Capacity counts the window latents for every
layer, `context/ratio` pooled latents for the compressed ones, and one
index key per pooled entry in CSA layers.

The presets come out at 284B and 1570B parameters against the public
284B and 1.6T, 14.2B and 51.2B active against 13B and 49B.

## The cache

```
  context     KV/seq GB   61-layer latent cache   ratio   max seqs, 8xB200 (FP4)
  32768           0.25                    2.8    0.090                    1123
  131072          0.99                   11.3    0.088                     286
  1048576         7.88                   90.1    0.087                      36
```

Against a 61-layer latent cache with a per-token index key — the previous
generation's design — V4-Flash holds 8.7% of the bytes at every context
length; the report quotes 7% for Flash at a million tokens (it counts an
FP8 cache, this table counts bf16). Eight B200s hold 36 million-token
conversations at once. That is the number the architecture exists for.

## The step

```
  context   TTFT s   indexer share   TPOT b=8 ms   HCA share
  32768       0.15             15%          5.8          4%
  131072      0.79             35%          6.0          6%
  1048576    20.52             78%          7.9         21%
```

Decode barely notices a million tokens: 7.9 ms against 5.8 at 32k, and
the growth is the dense HCA attention over a cache of `context/128`
entries. Prefill is another matter. At a million tokens the indexer is
78% of TTFT: every one of a million queries scores ~131k compressed keys
with 64 heads of 128, and under tensor parallelism that work is
replicated on every GPU, because splitting the heads would need the
per-key score sums reduced across ranks — a matrix the size of the
scores themselves. Splitting the *queries* is context parallelism, which
this path does not yet support; that is the exercise, and it is what a
production deployment would do.

## Two corrections to milestone 41

Pricing V4's indexer at a million tokens produced a number that could
not be true — an implied 100 PFLOPS per GPU — and the reason was in the
milestone-41 indexer that the new path inherited.

- **The key is shared across heads.** The indexer has one 128-dim key per
  token and 64 per-head queries. Batching the score matmul per head, as
  M41 did, charged the key cache once per head: 64× the traffic. The
  scores are now one matmul per sequence with the heads stacked as rows.
- **The scores are never written.** As an unfused matmul the indexer
  wrote an fp32 score matrix of queries × keys per layer — at a million
  tokens, terabytes. Real indexers stream the top-k selection and never
  materialize it, the same fact milestone 23 established for QK^T. The
  indexer is now a fused streaming op priced on its math and one read of
  the keys.

The milestone-41 headline — attention flat under the cap while the
indexer grows with context — stands. Its number does not: from 4k to 64k
the indexer grows 4.9×, not 14.7×, and its absolute cost falls by more.
The post carries an erratum.

## A third, from the serving engine

The engine's per-token KV accounting used per-head bytes for every
model. For latent-cache models that is several times the latent (3.7×
for GLM-5.3 at tp=4); the pool it computed for GLM-5.3 and Kimi-K3 was
correspondingly small. It now
derives the per-token figure from the exact per-sequence cache at a
representative context, for latent and compressed caches alike.

## Not modeled, and unvalidated

The compressor's pooling is priced as an elementwise pass over the
latents; the hyper-connection mixing as traffic over the widened stream
plus a small projection; the indexer's top-k selection cost is folded
into the fused op. The shared expert's single extra weight set (0.4% of
parameters) is not counted so expert parallelism divides the 256 routed
experts. Context parallelism and cascade attention are refused for
compressed models rather than approximated. And nothing here has met
silicon: no V4 checkpoint fits this box, so the path inherits the MLA
and DSA error bars by construction until a cross-model or datacenter
check is made.

## Exercises

1. Add context parallelism to the compressed path: shard the queries,
   replicate the compressed cache, and see how TTFT at a million tokens
   scales on 8, 16 and 32 GPUs.
2. Price the indexer in FP8 with the FP4 serving recipe (it is) and then
   in FP4 — does the report's "10% of V3.2 FLOPs at 1M" follow?
3. The three hash-routed layers have no router but also no load balancer.
   Add an imbalance factor to them alone and see whether it matters.
