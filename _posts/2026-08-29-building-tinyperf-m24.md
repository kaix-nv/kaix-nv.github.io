---
layout: post
math: true
title: "Building tinyperf M24: Hybrid linear attention — when the cache stops growing"
date: 2026-08-29 16:30:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Current open models run three-in-four gated-delta layers with a fixed-size state. Pricing them shows the trade exactly: nothing at 2k context, 3.5x and 4.7x more sequences at 256k."
---

*Milestone 24 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `estimate_linear_attention` in `gemm_model.py`, hybrid layers in `nets/transformer.py` · Demo: `examples/20_hybrid_linear_attention.py`.*

Everything in this series so far assumed a transformer is a stack of
identical softmax-attention layers. The current generation of open models
has quietly stopped being that. Point the model at a released 2026-era
checkpoint and the config says:

```
layer_types: [linear_attention, linear_attention, linear_attention,
              full_attention, linear_attention, ...]     # 1 in 4
```

Three of every four layers are **gated-delta linear attention** — a
recurrent block with a fixed-size state — and only the fourth is the
softmax attention this series has been modeling. A model that can't price
those 48 layers can't price the model at all.

## What a linear-attention layer is, to a performance model

A fused QKV projection, a short depthwise convolution, a recurrent core,
an output gate, and an out projection. Everything except the core is
already in our vocabulary (GEMMs and bandwidth-bound ops). The core has
two regimes, and the gap between them is the whole architecture:

**Prefill** runs the chunked delta rule: quadratic *inside* a chunk of
64 tokens, a fixed-size state update *across* chunks. Cost is therefore
**linear in sequence length**, where softmax attention is quadratic.

```python
per_head = 2*n*chunk*(k_dim + v_dim) + 4*n*k_dim*v_dim     # linear in n
```

**Decode** is a pure recurrent step: read the state, accumulate one outer
product, query it, write it back. The math is trivial (megaflops); the
cost is state traffic. And that state does not grow with context — where
a KV cache would.

## The consequence, priced

A 27B hybrid (16 full-attention + 48 linear layers) against a
counterfactual with the same dimensions where every layer is full
attention, tp=8, batch 32:

```
  context  hybrid ms  all-attn ms  speedup  hybrid KV+state  all-attn KV
     2048       8.38         8.45     1.01           1.7 GB       4.3 GB
     8192       9.34        12.30     1.32           4.9 GB      17.2 GB
    32768      13.19        27.67     2.10          17.8 GB      68.7 GB
   131072      28.56        89.18     3.12          69.3 GB     274.9 GB
   262144      49.06       171.18     3.49         138.1 GB     549.8 GB
```

And in the currency an operator actually spends — concurrent sequences
per GPU:

```
  context    hybrid   all-attn   ratio
     8192       393        113     3.5
    32768       109         28     3.9
   262144        14          3     4.7
```

Read the top row before the bottom one. At 2k context the hybrid wins
**nothing** (1.01x): a 154 MB constant state costs about what a short KV
cache costs, and the recurrent core has to be read every step regardless.
The architecture is not free — it is a *trade*, and it pays only as
context grows. By 256k it is 3.5x faster per step and fits 4.7x more
sequences. Everything in the modern long-context race is in those two
numbers.

Note also what the 1-in-4 full-attention layers do to the curve: they
keep the KV cache growing (65.5 KB/token instead of 262 KB/token — a
quarter, not zero). The architecture deliberately keeps them, and the
model shows exactly what they cost.

## Modeling notes, and what is approximate

The state is fp32 (the ssm dtype in the config) and sized
`v_heads x k_dim x v_dim` per layer; the conv adds a small ring buffer.
Chunk length is fixed at 64. The delta-rule core carries its own math
efficiency (0.55) because it is a mix of small matmuls, gating and
normalization rather than one clean GEMM — the same style of honest
constant as milestone 7's FMHA efficiency, and the same caveat: it wants
measuring against a real kernel, not asserting.

Two things this milestone does *not* model: the vision tower of a
multimodal checkpoint, and the MTP (multi-token prediction) head. Both
are separate stacks; the parameter count lands at 26.9B against the
checkpoint's 27.25B, and the 0.37B difference is precisely the MTP layer
we skipped. That the residual is exactly a component we know we omitted
is the check that the rest is right.

## Exercises

1. Measure a real gated-delta kernel and calibrate the 0.55 efficiency —
   does it hold across prefill chunk sizes and decode batch sizes?
2. Sweep the full-attention interval (1-in-2, 1-in-4, 1-in-8, all-linear)
   and plot quality-agnostic cost vs context. Where does the marginal
   full-attention layer stop being worth its KV?
3. Model MTP: one extra layer that proposes a token, verified by the main
   stack — milestone 17's speculative decoding with a draft that shares
   the trunk. What acceptance rate makes it pay?
4. Add the vision tower as a prefill-side encoder and price an image-heavy
   multimodal request end to end.
