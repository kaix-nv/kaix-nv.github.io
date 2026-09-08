---
layout: post
math: true
title: "Building tinyperf M47: Context parallelism: splitting the sequence, not the model"
date: 2026-09-08 17:15:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Ring attention for prefill, a sharded cache for decode: context parallelism cuts a 128k prompt from eleven seconds to a second and a half, and buys decode nothing until the cache, not the weights, is the traffic."
---

*Milestone 47 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `cp` in `nets/transformer.py`, `capacity.py`, `serving.py`, `sweep.py` · Example: `examples/30_context_parallel.py`.*

Three ways to spread a model over more GPUs are in the model already:
slice the weights (tensor parallelism, milestone 5), stack the layers
(pipeline, milestone 44), replicate the attention (data parallelism,
milestone 46). None of them makes one long prompt faster past the NVLink
domain. Tensor parallelism stops paying at the domain edge, a pipeline
stage does not shorten a single request, and a replica serves *other*
requests. The fourth axis splits the sequence itself.

## Two different mechanisms under one flag

**Prefill: ring attention.** Each of `cp` ranks holds `s/cp` tokens of
every sequence and every projection, norm and FFN runs on that shard —
GEMM rows fall by `cp`. Attention needs the whole context, so the KV
blocks circulate around the ring while each rank computes its queries
against them. With the zigzag balancing every implementation uses, each
rank ends up with exactly `1/cp` of the causal work, which the model
expresses by keeping the average causal KV length and shrinking the query
rows. The circulation is one all-gather of the layer's KV per rank per
layer, priced serially like every other collective here (real rings
overlap it with compute, so the term is an upper bound).

**Decode: shard the cache.** A decode step has one query per sequence and
a long cache. Splitting the query is pointless, so the queries are
replicated on all `cp` ranks, each rank attends to its `1/cp` of the
cache, and the partial outputs with their log-sum-exp are combined — a
small all-reduce of `heads × (head_dim + 2)` per token. The GEMMs run in
full on every rank. That last sentence is the whole story of what CP does
and does not buy for decode: **weights are replicated across cp**, so
weight streaming does not shrink; only cache traffic does.

Capacity follows: the cache per rank is `1/cp`, the weights are not, and
the recurrent state of a hybrid's linear layers stays whole.

## LLaMA3-70B on H100s, tp8 × cp

```
  layout    GPUs  TTFT 32k s  TTFT 128k s  TPOT@4k ms  TPOT@128k ms  KV/GPU GB
  tp8 cp1      8        1.71        11.20       14.02         29.54       42.9
  tp8 cp2     16        0.86         5.61       14.22         21.97       21.5
  tp8 cp4     32        0.45         2.84       14.41         18.29       10.7
  tp8 cp8     64        0.24         1.49       14.99         16.93        5.4
```

Prefill scales almost perfectly — 7.5× at cp8 — because both terms that
make up a long prefill, the quadratic attention and the linear GEMM rows,
shard. The 128k prompt that takes eleven seconds on one node takes a
second and a half on eight. Decode at 4k context gets *slower* with cp:
every rank still streams all 70B weights, and the combine is pure
overhead. At 128k the cache is the traffic and cp8 takes 43% off the
step. Whether the fourth axis pays is a question about the workload's
context length, and the table gives the crossover.

## What this composes with

`cp` sits alongside `tp`, `dp` and `pp` in the step model, the capacity
planner and the sweep pricer; the world is `tp × cp × dp × pp`, and a
layout that leaves the NVLink domain prices its collectives over the
fabric. Hybrid models shard too — the linear-attention layers carry no
cache to gather, so their prefill simply runs on `s/cp` tokens. Experts
are replicated across cp ranks; the cp rank set dispatches its own token
shard over its expert-parallel group.

## Not modeled

Ring-attention overlap of the KV circulation with compute (the serial
price is the upper bound); the load imbalance of a naive, non-zigzag
split; sequence parallelism proper (sharding the norms and dropout under
tensor parallelism, a training-side memory optimization); and any silicon
— context parallelism inherits the tensor-parallel error bars by
construction until a multi-GPU long-context run is measured.

## Exercises

1. Add an `overlap` fraction to the KV all-gather and find the context
   length below which the ring's communication becomes visible.
2. Price a 1M-token prompt: at what `cp` does the all-gather over the
   fabric start to dominate the attention it enables?
3. Decode with `cp` gains only cache traffic. Combine it with the paged
   allocator of milestone 20 and see how many more concurrent 128k
   sequences a node holds.
