---
layout: post
math: true
title: "Building tinyperf M7+M8: Fusion and feasibility — making the model honest"
date: 2026-08-18 16:49:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Two lies fixed: fused attention removes score-matrix traffic that never existed, and a capacity model stops pricing configs that don't fit."
---

*Milestones 7+8 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tinyperf/passes.py`, `tinyperf/capacity.py`, FMHA model in `gemm_model.py` · Demo: `examples/06_fmha_capacity.py`.*

Milestones 1–6 built a model that answers "how fast?". This pair fixes the
two ways it lied to you.

## Lie #1: unfused attention

Our builder writes attention the mathematical way — `QK^T -> softmax -> PV`
— which materializes the S x S score matrix in DRAM four times (two matmul
sides, softmax read, softmax write). No modern runtime does that:
FlashAttention-style kernels keep scores on-chip and stream K/V through
shared memory. A model that prices the unfused graph charges for traffic
that does not exist, and the error grows **quadratically** with context.

The fix has two halves, mirroring production tools exactly:

**A graph pass** (`passes.py`, ~40 lines): find `BatchedMatMul -> Softmax ->
BatchedMatMul` chains *by dataflow* (each op must consume the previous op's
output tensor) and replace them with one `FusedAttention` op. This is the
smallest real instance of what production fusion-pass frameworks do at
scale — simulate the kernel library's fusion decisions, not the model
author's math.

**A fused kernel model** (`estimate_fmha`): two matmuls' worth of FLOPs, but
DRAM traffic touching only Q, K, V, O:

```python
flops      = 2 * batch * m * kv * (k + out_dim)          # QK^T + PV
dram_bytes = bytes * batch * (m*k + kv*(k+out_dim) + m*out_dim)  # no S x S!
time       = max(flops / (peak * 0.65), dram_bytes / bw) + launch
```

The 0.65 is an honest constant: fused kernels interleave MMAs with softmax
bookkeeping and hit ~60–70% of tensor-core peak (public FlashAttention-2
numbers). One efficiency scalar is the poor man's version of what production tools
do — model FMHA with an actual event-driven kernel simulation because
closed form gets it wrong.

Results (LLaMA2-7B prefill, b=1, A100):

```
seq_len   unfused ms   fused ms   saved    scores matrix/layer
  2048        108.7      104.6     3.7%      0.1 GB
  8192        521.6      466.6    10.5%      2.1 GB
 32768       3729.4     2876.4    22.9%     34.4 GB
```

At 2k context, fusion is a rounding error — at 32k it's a quarter of the
runtime. The test suite pins the *shape* of this curve (saving must grow
monotonically with context), not a point value; that's the right way to
test a model whose constants you expect to calibrate later.

## Lie #2: pricing impossible configs

Milestone 6 happily "ran" 141 GB of LLaMA3-70B weights on an 80 GB H100.
Capacity is often the *binding* constraint, and it's closed-form
(`capacity.py`):

```
weights/GPU = sharded params / tp + unsharded embed & LM head
KV/GPU      = 2 * layers * local_kv_heads * head_dim * bytes * batch * context
activations = 2 * tokens * widest intermediate    (small for decode)
fits        = total <= 0.90 * HBM                 (headroom stated, not hidden)
```

The TP table from milestone 6 now carries the truth column:

```
tp   ms/token   tok/s   mem/GPU GB   fits?
 1      67.78     472        184.1     NO
 2      36.33     881         94.1     NO    <- weights fit, KV doesn't!
 4      20.98    1525         49.2    yes
 8      14.35    2230         26.7    yes
```

The tp=2 row is the interesting one: 72.7 GB of weights looks like it fits
in 80 GB — but against the 90%-headroom budget of 72 GB, the weights
*alone* are already over, before a single KV byte. `max_batch` returns 0:
TP-4 is the floor for this model on 80 GB parts, and no batch tuning
rescues TP-2. And the derived table every serving engineer actually wants,
`max_batch(model, device, context)`:

```
context   7b@A100 tp1   70b@H100 tp4   70b@H100 tp8
   2048            54            199            600
   8192            13             49            150
  32768             3             12             37
```

Halve the context, double the batch — KV cache *is* the capacity story.
Cross-reference milestone 5's throughput curve (still rising at batch 128
for 2k context, which capacity confirms is reachable) and you have the
complete speed-vs-fit picture from two closed-form models.

## In production-scale models

Production allocators don't estimate — they *walk the graph* and simulate
alloc/free against tensor lifetimes to find true peak memory, including
activation checkpointing and weight offloading, and emit per-component
memory-requirement reports that gate which batch-sweep configs even launch
(OOM filtering). Fusion-wise, production pass frameworks carry dozens of
pattern passes (MHA, MLP, pointwise chains, wgrad/dgrad...), each paired
with kernel models for the fused result.

## Exercises

1. Fuse decode attention too and measure what it saves (hint: mostly
   launches — decode scores are tiny; flash-*decode*'s real win is split-KV
   parallelism, which maps to milestone 2's split-K exercise).
2. Add paged KV (block size 16/128) and re-derive max_batch with
   fragmentation.
3. Add weight quantization (INT8/FP4 storage, fp16 math) to both models —
   watch max_batch and decode latency improve together.

*That closes the core series: ~700 lines of stdlib Python that price LLMs on GPUs from first principles. The exercises across these posts (split-K, MoE, calibration, an event-driven serving layer) are the map for what comes next.*
