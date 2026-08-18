---
layout: post
math: true
title: "Building tinyperf M6: Tensor parallelism and the price of communication"
date: 2026-08-18 16:41:40 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Shard the shapes, insert the all-reduces, price them in closed form — and watch Amdahl emerge without being programmed."
---

*Milestone 6 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tinyperf/comm_model.py`, TP logic in `nets/transformer.py` · Demo: `examples/05_tensor_parallel.py`.*

## Parallelism is a property of the graph, not the simulator

The single most transferable design decision in this whole series: we do
not simulate N GPUs. We build the graph **one GPU sees**, with sharded
shapes and explicit communication ops:

```python
nh_l  = n_heads // tp          # local query heads
ffn_l = ffn_hidden // tp       # local FFN width
...
attn_out = g.Linear("attn_out", ctx, out_features=hidden)
if tp > 1:
    attn_out = g.AllReduce("attn_allreduce", attn_out, group_size=tp)
```

Megatron-style TP needs exactly two all-reduces per decoder layer (after
attention output projection, after FFN down projection); the builder inserts
them where the math requires. Since all TP ranks are symmetric, one GPU's
timeline is the system's timeline. Production graph IRs work identically —
sharded shapes plus training-aware collective ops baked in at build time;
only pipeline parallelism needs scheduling machinery beyond this.

## Pricing a collective in closed form

Ring all-reduce on `n` GPUs: reduce-scatter then all-gather, each byte
crossing each link twice, plus a latency term per hop:

```
t = 2(n-1)/n * bytes / link_bw  +  2(n-1) * hop_latency  +  launch
```

Two regimes, one formula. Big messages pay the bandwidth term (1 GiB across
8 H100s ≈ 4.2 ms); small messages pay latency — decode all-reduces move only
`batch x hidden` elements (~0.5 MB), so their cost is mostly the ~17 µs of
hops and launch. Latency terms are why TP doesn't scale to small work.

## The result: sublinear, and honestly so

LLaMA3-70B decode, batch 32, kv 4096, H100 (`examples/05_tensor_parallel.py`):

```
tp   ms/token   tok/s   speedup   comm %   weights/GPU
 1      69.86     458      1.00      0.0        141 GB   (doesn't fit!)
 2      37.61     851      1.86      2.6         71 GB
 4      21.86    1464      3.20      7.9         35 GB
 8      15.03    2129      4.65     20.3         18 GB
```

TP-8 yields 4.65x, not 8x, and the report's `comm %` column says exactly
why: by tp=8, a fifth of every token's time is all-reduce, and each GPU's
GEMMs have shrunk toward the launch-overhead floor. Also note the tp=1 row:
141 GB doesn't fit an 80 GB GPU — people run 70B at TP≥2 for *capacity*
first, speed second. (Our model happily prices impossible configs; a
capacity check is milestone 5's exercise 1, implemented in milestone 8.)

Amdahl shows up without being programmed: the unsharded LM head and the
per-op launch overheads are the serial fraction. Analytical models are at
their best exactly here — the *shape* of a scaling curve from first
principles, minutes instead of a cluster-day per point.

## In production-scale models

Production communication models add: topology awareness (NVLink/NVSwitch
domains, IB rails across nodes — an 8-GPU all-reduce and a 1024-GPU one
are different algorithms), multiple algorithms per collective (ring/tree/multishot) with
best-of selection like GEMM tiles, comm/compute overlap models, and
EP/CP/hybrid schemes whose dispatch (AllToAllV) costs depend on expert
routing. Plus disaggregated-inference KV-transfer models. All of it is this
milestone's pattern — closed-form cost per comm op, inserted in the graph —
with better constants and topology.

## Where the series lands

~700 lines of stdlib Python now answer, in milliseconds: what does it cost
to serve a 70B model at TP-8? What batch saturates an A100? Is this GEMM
worth quantizing? The mechanism — *describe the machine as rates, the
workload as shapes, and search over kernel configurations analytically* —
is the entire core of the industrial tools; everything else there is
constants, coverage, and calibration. Which is not a small thing: it is
thousands of engineer-years of making these 700 lines *right*, per
architecture, per kernel library, per model family. But now you know where
every one of those lines would go.

## Exercises

1. Add pipeline parallelism analytically: `(microbatches + stages - 1) /
   microbatches` bubble factor over per-stage times.
2. Add a 2-node (16 GPU) topology where inter-node links are 50 GB/s IB:
   make `AllReduce` hierarchical and find where TP should stop at the node
   boundary.
3. Overlap: hide all-reduce under the next GEMM (exercise 3 of milestone 4)
   and re-derive the TP-8 speedup.

*Next: milestones 7 and 8 make the model honest — fused attention, and a capacity model that refuses configs that don't fit.*
