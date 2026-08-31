---
layout: post
math: true
title: "Building tinyperf M35: What survives of the FP4 promise"
date: 2026-08-31 14:05:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Naive uniform-FP4 arithmetic claims 3.7x at long context; the deployable recipe (fp4 weights, fp8 KV, fp16 head) delivers 2.0x. Both get called FP4 speedup — only one can be scheduled against."
---

*Milestone 35 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `apply_recipe` + `RECIPE_FP4_SERVING` in `passes.py` · Demo: `examples/25_precision_recipes.py`.*

Every estimate so far priced one dtype per run. No deployed model works
that way: a Blackwell-era serving config runs FP4 GEMM weights, FP8
attention and KV cache, and a full-precision head — precision is
per-layer-class, not global. A uniform-dtype estimate misses exactly the
tensors that refuse to shrink, and those are the ones that grow with
context.

## The mechanism: a recipe is a pass

```python
g = build_llm_graph(model, "decode", batch=64, seq_len=32768)
fuse_attention(g)
apply_recipe(g, RECIPE_FP4_SERVING)   # name-prefix -> dtype, first match wins
```

Like fusion, it operates on the built graph, so it composes with MoE,
hybrid linear attention, speculation and the serving engine without
touching any of them — the milestone-4 dispatch bet paying out one more
time. Math-family ops (GEMM, fused attention, delta rule) take the tag;
normalization and residual traffic keep the model dtype; a device without
tensor rates for a requested precision refuses the graph outright.

## Three ways to say "FP4"

Qwen3-8B decode on a B200:

```
 batch      kv  fp16 ms  recipe ms  naive ms  recipe win  naive win
     1    4096     3.17       1.83      1.69       1.73x      1.88x
     8    4096     3.72       2.11      1.83       1.76x      2.03x
    64    8192    12.86       6.72      4.11       1.92x      3.13x
    64   32768    41.85      21.21     11.36       1.97x      3.68x
   128   32768    80.86      40.69     21.11       1.99x      3.83x
```

At small batch the naive uniform-FP4 projection and the deployable recipe
agree, because decode is streaming weights and the weights really are
fp4. As KV grows they part ways: the cache is fp8 in any recipe you can
actually serve, the head stays high-precision, and by 64x32k the naive
number claims **3.7x** where the deployable answer is **2.0x**. Both get
called "FP4 speedup" in slideware. Only one can be scheduled against —
and the gap between them *widens* with exactly the workloads (long
context, big batch) where you most wanted the speedup.

## The general point

Datasheet arithmetic scales every byte and every FLOP by the format
ratio. Deployment scales the tensors the accuracy budget allows. An
estimator that cannot express the difference will bless capacity plans
off by 2x on the workloads that matter most; the fix was ~30 lines,
because per-op pricing already existed and precision only needed to
become a per-op property.

## Exercises

1. Add an FP8-everything recipe and find the workloads where fp4-weights
   vs fp8-weights actually changes a deployment decision.
2. KV-cache-only quantization (fp8 KV under fp16 weights) is a popular
   middle ground — price it across the context sweep.
3. Recipes change weight *memory*, not just time; extend the capacity
   model to price a recipe's weight bytes and re-derive max_batch.
