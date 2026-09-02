---
layout: post
math: true
title: "Building tinyperf M43: Routing imbalance: where MoE skew actually bites"
date: 2026-09-02 12:50:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "A hot expert-parallel rank makes prefill 32% slower at 2x skew and leaves small-batch decode exactly where it was: routing imbalance is a prefill and large-batch tax, not a latency tax."
---

*Milestone 43 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `moe_imbalance` on the graph builder, step model and sweep configs.*

Milestone 11 assumed perfectly balanced routing. Real routers are
skewed — hot experts, cold experts — and with expert parallelism the
step is gated by the most-loaded rank. `moe_imbalance` is that rank's
load over the mean (1.0 balanced; 1.5 = the hottest rank carries 50% more
than average), a quantity engines report and load balancers exist to
drive toward 1.0. It is a workload input, like speculation acceptance,
not something the model can know.

gpt-oss-120B, 4 GPUs (tp=4, ep=8), B200:

```
  imb  prefill 8x2048  decode b8  decode b64
  1.0          34.2ms     3.91ms      6.86ms
  1.5          39.6ms     3.91ms      6.86ms
  2.0          45.0ms     3.91ms      6.87ms
```

The asymmetry is the finding. Prefill's expert GEMMs are math-bound, so
the hot rank's extra rows cost time: +32% at 2x skew. Small-batch decode
is weight-streaming-bound — every resident expert is read regardless of
how many rows it serves — so doubling the rows per expert changes
nothing. Imbalance is a prefill and large-batch tax, not a latency tax;
plans that fear it at low batch are fearing the wrong regime.

Also fixed here: the builder had been ignoring `first_k_dense` (added to
the accounting in milestone 41 but never to the graph), so GLM-5.3's and
Kimi-K3's dense first layers were being priced as MoE. They now carry
their real (much wider) dense FFN, and the MoE ops carry the right layer
count — pinned.

## Exercises

1. Imbalance changes which experts are *touched* at small batch too
   (concentrated routing touches fewer) — model the distinct-expert count
   under a skewed distribution and see if decode gets *faster*.
2. Capacity-factor implementations pad every expert to C x mean load;
   add the padding variant and compare with the variable-group model.
3. Measure real routing statistics from a served MoE and calibrate the
   knob instead of assuming it.
