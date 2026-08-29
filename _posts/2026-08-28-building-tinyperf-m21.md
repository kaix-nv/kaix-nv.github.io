---
layout: post
math: true
title: "Building tinyperf M21: DP and ZeRO — sixteen bytes, and where gradient sync binds"
date: 2026-08-28 17:00:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Mixed-precision Adam stores 16 bytes per parameter; ZeRO shards them away stage by stage — and gradient sync binds at exactly the microbatches pipeline bubbles ask for."
---

*Milestone 21 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — DP/ZeRO additions in `tinyperf/training.py` · Demo: `examples/19_dp_zero.py`.*

Milestone 12 built training as a graph pass and left two exercises open:
data parallelism with its gradient synchronization, and training memory.
They turn out to be one story — the story of what mixed-precision Adam
actually stores, and who pays to move it.

## Sixteen bytes per parameter

Mixed-precision training keeps fp16 weights (2 B) and gradients (2 B),
plus fp32 master weights and two Adam moments (12 B): **16 bytes per
parameter**, of which the weights you think of as "the model" are 12%.
ZeRO shards the rest across the DP group, stage by stage:

```
== gpt3-175b, pure DP=64, microbatch 2048 tokens, recompute on ==
  ZeRO  states GB/GPU  acts GB    total  fits 80GB?
     0         2802.9      4.8   2807.7          NO
     1          733.6      4.8    738.4          NO
     2          388.7      4.8    393.5          NO
     3           43.8      4.8     48.6         yes
```

Replicated model states for 175B are **2.8 TB** — the famous "needs
hundreds of GPUs just to fit," reproduced from arithmetic. Optimizer
state alone (stage 1's target) is 8x the weights; only ZeRO-3 brings a
pure-DP fleet of 80 GB parts under budget. (tp x pp sharding attacks the
same term from the other side; activations get their own model, with
recompute shrinking them 12x for one extra forward pass in time.)

## Where gradient sync binds

DP all-reduces the local gradients every step (ZeRO stages 0–2 move the
same first-order traffic — an all-reduce *is* a reduce-scatter plus an
all-gather; stage 3 pays ~1.5x to re-gather parameters). Production
stacks overlap the sync under backward, so the step is
`max(backward, sync)` — and the interesting question is when the max
flips:

```
== llama2-7b DP=8, overlapped ==
        device ubatch tok  bwd ms  sync ms  step ms  DP eff
     RTX_A6000        512     111      421      421    0.26
     RTX_A6000       2048     429      421      429    1.00
      H100_SXM        512      20       52       52    0.38
      H100_SXM       2048      70       52       70    1.00
```

Gradient bytes are fixed per step, so sync is a horizontal line; backward
shrinks with the microbatch. Two lessons the table forced on the author:

- **A slow GPU is not automatically comm-bound.** My first draft blamed
  the A6000's thin 56 GB/s link — but its backward is slow too, and at
  2048 tokens the sync hides completely. What binds is the *ratio* of
  compute to communication, not either number alone.
- **The M12 tension.** Pipeline bubbles want many small microbatches;
  DP sync punishes exactly those (efficiency 0.26–0.38 at 512 tokens).
  Fleet design is picking the microbatch where both taxes are
  affordable — and now both taxes come out of the same model.

## In production-scale models

Production training stacks schedule bucketed overlapping all-reduces
(per-layer buckets fire as their gradients finish), hierarchical
collectives across the NVLink/IB boundary, ZeRO-3 prefetch to hide the
parameter gathers, and gradient compression when the interconnect truly
binds. The accounting in this milestone — sixteen bytes, sharded by
stage, moved once per step — is the invariant underneath all of it.

## Exercises

1. Hierarchical DP: all-reduce within a node at NVLink speed, across
   nodes at IB speed. Re-derive the crossover microbatch for a 16-node
   fleet.
2. ZeRO-3 prefetch: overlap the parameter all-gathers with compute the
   same way sync overlaps backward. How much of the 1.5x tax vanishes?
3. Compose with M12's pipeline model: total step = bubbles + exposed
   sync + optimizer. Find the microbatch that minimizes real step time
   for GPT-3 on 64 GPUs, and compare against the M12 table's optimum.
4. Gradient compression: at what compression ratio does the A6000's
   512-token row reach DP eff 1.0, and what accuracy literature says
   that ratio costs?
