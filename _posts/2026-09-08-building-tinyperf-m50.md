---
layout: post
math: true
title: "Building tinyperf M50: Pipeline schedules: buying back the bubble"
date: 2026-09-08 16:44:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "GPipe, 1F1B, interleaved, zero-bubble and DualPipe as first-order bubble algebra in the model's own forward/backward chunk times: at eight microbatches the schedule is the difference between 33% and 60% MFU, and each one pays in a different currency."
---

*Milestone 50 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `pipeline_bubble_us`, `split_fwd_bwd`, `train_step_us` in `training.py` · Example: `examples/33_training_schedules.py`.*

Milestone 12 priced pipeline parallelism as `(pp − 1)` idle slots per step
and moved on. That is GPipe's bubble, and 1F1B's too; the schedules that
came after exist to shrink it, and each one pays for the bubble it
removes in a different currency. This milestone adds them as first-order
algebra — the table the zero-bubble and DualPipe papers publish — priced
in the model's own chunk times rather than in abstract units.

## Chunks the model already has

A pipeline schedule is arithmetic over three chunk times per stage per
microbatch: the forward `F`, the input-gradient backward `D`, and the
weight-gradient backward `W`. Milestone 12's backward pass writes exactly
those ops (`_dgrad`, `_wgrad`, `_bwd`), so `split_fwd_bwd` reads them off
a priced graph. For GPT-3 175B on a tensor-parallel rank of eight with a
2048-token microbatch: `F` 164 ms, `D` 169 ms, `W` 101 ms. The W chunk is
smaller than D — it has no attention backward in it — and that asymmetry
is what zero-bubble schedules exploit.

```
  gpipe / 1f1b    (pp − 1)(F + D + W)
  interleaved     (pp − 1)(F + D + W) / v       v virtual stages per rank
  zb_h1           (pp − 1)(F + D − W)           W fills the holes
  zb_h2           0                             at ~2x the activations
  dualpipe        (pp/2 − 1)(F + 2D + W)        two directions, 2x params
```

With equal chunks zero-bubble H1 has a third of 1F1B's bubble, H2 none,
interleaving divides by `v`, DualPipe halves the effective depth. The
old `pipeline_efficiency(pp, mb)` is unchanged and is the `1f1b` row.

## GPT-3 175B on 64 H100s, tp8 × pp8

Step time and MFU by microbatch count; memory columns at 32 microbatches:

```
  schedule            mb=8         mb=16         mb=32         mb=64   act ubatches  params
  1f1b            835ms 33%    1270ms 43%    2138ms 51%    3875ms 57%           8       1
  interleaved v=2 645ms 42%    1080ms 51%    1948ms 56%    3685ms 59%           8       1
  zb_h1           658ms 42%    1093ms 50%    1961ms 56%    3698ms 59%           8       1
  zb_h2           455ms 60%     890ms 62%    1758ms 62%    3495ms 63%          15       1
  dualpipe        606ms 45%    1040ms 53%    1908ms 57%    3645ms 60%           9       2
```

Three things the table says that the bubble formula alone did not:

- **Schedules matter most where you can least afford microbatches.** At
  eight microbatches the spread is 33% to 60% MFU — nearly 2×. At 64 it
  is 57% to 63%. Long-context training, where a microbatch is one
  sequence and eight of them is all the memory allows, is where the
  schedule is the design decision.
- **Zero-bubble H1 is free.** Same resident activations as 1F1B, same
  parameter copies, and the bubble drops by the `W` chunk per stage —
  worth nine points of MFU at `mb=8`. H2 removes the rest for roughly
  double the activation memory: 42 GB instead of 22 GB per GPU here.
- **DualPipe's price is a second parameter copy.** 5.5 GB per GPU on
  this model, for a bubble between H1 and H2 — and the expert all-to-all
  hidden, which this dense model cannot show and a MoE would.

## Sequence parallelism, while we are counting activations

Under tensor parallelism the GEMM activations shard `1/tp` but the norm
and dropout inputs and the residual stream do not — Megatron's
24-of-34 split. Sequence parallelism shards those too. `activation_bytes`
now takes `tp` and `sequence_parallel`: at tp8 the replicated share
leaves activations at 38% of a single rank's, and SP takes them to 12.5%.
On the 1F1B row above that is 22 GB → 7 GB per GPU, the difference
between fitting and recomputing.

## Not modeled

The extra point-to-point traffic of interleaving (`v×` the hand-offs);
DualPipe's actual compute overlap (`F&B` is priced as `F + B`); the
schedule-dependent warm-up when microbatches are fewer than stages;
selective recompute policies (recompute stays all-or-nothing); and any
measurement — the training tier remains first-order arithmetic anchored
to the 6N rule.

## Exercises

1. Add the interleaved schedule's `v×` hand-off traffic (milestone 44's
   `p2p_us`) and find the `v` past which it stops paying on IB.
2. Price DualPipe on a MoE (milestone 11) with the expert all-to-all
   hidden — set the a2a ops' time to zero under `dualpipe` — and see
   whether the second parameter copy is worth it at `ep=8`.
3. Zero-bubble H2's memory can be traded for recompute. Combine
   `recompute=True` with `zb_h2` and compare against `zb_h1` at the same
   memory budget.
