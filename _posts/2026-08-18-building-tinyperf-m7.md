---
layout: post
math: true
title: "Building tinyperf M7: FMHA fusion — price what the kernel library runs"
date: 2026-08-18 16:41:50 -0700
categories: [tinyperf, perf-modeling]
excerpt: "A dataflow rewrite pass plus a fused kernel model remove score-matrix traffic that never existed — a saving that grows quadratically with context."
---

*Milestone 7 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tinyperf/passes.py`,
FMHA model in `gemm_model.py` · Demo: `examples/06_fmha_capacity.py`.*

Milestones 1–6 built a model that answers "how fast?". It lies to you in
two ways; this milestone fixes the first (the second falls in milestone 8).

## The lie: unfused attention

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
numbers). One efficiency scalar is the poor man's version of what production
tools do — model FMHA with an actual event-driven kernel simulation because
closed form gets it wrong.

## Results

LLaMA2-7B prefill, b=1, A100:

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

## In production-scale models

Production pass frameworks carry dozens of pattern passes (MHA, MLP,
pointwise chains, wgrad/dgrad...), each paired with a kernel model for the
fused result — the pass and the pricing always ship together, because a
rewrite without a cost model is just wishful thinking.

## Exercises

1. Fuse decode attention too and measure what it saves (hint: mostly
   launches — decode scores are tiny; flash-*decode*'s real win is split-KV
   parallelism, which maps to milestone 2's split-K exercise).
2. Fuse the pointwise SwiGLU into the gate/up GEMM epilogue and measure the
   saved read+write of the activation.
3. Calibrate the 0.65 efficiency scalar against published FlashAttention-2
   benchmarks at several sequence lengths — does one constant suffice?

*Next: milestone 8 adds the other honesty fix — a memory capacity model, so
the simulator stops pricing configurations that don't physically fit.*
