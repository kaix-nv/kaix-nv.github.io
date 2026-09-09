---
layout: post
math: true
title: "Building tinyperf M56: The mixed step, measured directly (and the step nobody planned for)"
date: 2026-09-08 23:12:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Inject one prefill into a steadily decoding batch and the spike in everyone's inter-token latency is the mixed step. Sixteen cells confirm the sum-minus-one-weight-pass shape, expose a small-chunk residual, and reveal the one extra step that async scheduling adds to every request's TTFT."
---

*Milestone 56 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tools/measure_mixed_step.py`, `async_scheduling` in `serving.py`.*

Milestone 54 changed how the model prices a step that carries both a
running decode batch and a prefill chunk — from the maximum of the two to
their sum less one pass over the weights — on the strength of a load
sweep. That is indirect evidence: a sweep mixes scheduling, queueing and
step costs, and a wrong step price can hide behind a right queue. This
milestone measures the step itself.

## How to see one step inside a serving engine

Start `B` requests decoding steadily on a live server (128-token prompts,
400 tokens out). Their inter-token latency is the decode step. Now inject
one request with a `C`-token prompt and `max_tokens = 1`. The step that
carries its prefill runs longer, and every background request's next
token arrives late by the same amount: the spike in their ITL *is* the
mixed step, measured on the GPU that ran it, with no queueing in the way.
Three injections per cell, medians; 16 cells over `B ∈ {1, 8, 32, 64}`
and `C ∈ {256, 512, 1024, 1920}` (the largest chunk the 2048-token budget
allows). Predictions frozen first, for three candidate prices: the M54
formula, the plain sum, and the M19 max.

## The mixed step

```
   B     C   decode ITL  meas   mixed meas   model    r    sum    r    max    r
   1   256        23.8              45.8     42.5  0.93   66.2 1.45   42.3 0.92
   1  1920        23.9             283.1    295.1  1.04  318.8 1.13  294.9 1.04
   8   256        24.5              50.0     44.2  0.88   67.8 1.36   42.3 0.85
   8  1920        24.5             287.1    296.8  1.03  320.4 1.12  294.9 1.03
  32   256        27.7              60.1     49.1  0.82   72.7 1.21   42.3 0.70
  32  1920        27.8             298.3    301.7  1.01  325.3 1.09  294.9 0.99
  64   256        32.1              69.7     55.3  0.79   78.9 1.13   42.3 0.61
  64  1024        32.2             180.5    170.6  0.95  194.2 1.08  157.6 0.87
  64  1920        32.1             313.4    307.9  0.98  331.5 1.06  294.9 0.94
```

Over the sixteen cells the M54 price sits at a geometric mean of 0.95
against 1.18 for the plain sum and 0.90 for the old maximum, and it is
the only one of the three that stays inside ±10% for most cells. The
shape is right: a mixed step is the prefill's math plus the decode's
attention and bookkeeping, with the weights streamed once. That was the
claim, and the step confirms it.

The shape is right and the small corner is not. With a 256-token chunk
the under-prediction grows with the batch — 0.93, 0.88, 0.82, 0.79 for 1,
8, 32, 64 decodes. Something in a mixed step costs a few hundred
microseconds per running sequence *beyond* what a pure decode step pays,
and it matters when the chunk is small enough that it is not hidden.
vLLM runs pure-decode batches as full CUDA graphs and mixed batches
piecewise, with attention and input preparation issued eagerly, and that
is the size of the gap. It is not fitted here: a fit would need cells that
vary it independently of the chunk size, and these do not.

## Two things the same runs said about earlier milestones

**The pure decode step.** Between injections the background requests
measure the decode step alone, at batch 64 as cleanly as it can be
measured: 32.1 ms. The graph says 32.6. Milestone 55's per-sequence
constant, fitted on a decode-heavy load sweep, adds 4 ms and lands at
36.6 — 14% high. The sweep it was fitted on had arrivals mixing small
prefills into its steps, and this measurement says part of that
constant was the small-chunk residual above wearing a different name.
Both numbers are in the envelope; neither is refitted on this data.

**The step nobody planned for.** The injected requests' own TTFT — from
send to first token — came back about one full decode step above the
model at every one of the sixteen cells, on top of the half step already
charged for the step in progress. That is what async scheduling looks
like from outside: the engine plans step `k+1` while step `k` runs, so a
request that arrives during step `k` is first eligible for step `k+2`.
`simulate` now defaults to that admission rule (`async_scheduling=False`
is the milestone-13 engine). With it the injected TTFTs land at 1.03
(0.92–1.12), and — the check that matters — the milestone-54 sweep's
low-load TTFT medians, which had sat at 0.86–0.89 for two milestones,
move to 1.00–1.01 with nothing else changed. The mechanism was found on
one measurement and confirmed on another.

## What is left

The knee. On the load sweep the server tips over at 4 requests per second
and the model, with every correction so far, tips over between 4 and 5;
at the knee itself TTFT is still 0.5–0.6 of measured. Everything below
the knee is now within a few percent and everything at saturation within
10%, so what remains is the transition: how fast a real scheduler falls
behind once prefill demand exceeds what its budget clears per step. The
small-chunk mixed cost is part of that story, and so, probably, is the
per-sequence constant's overreach. A finer sweep between 3 and 5 requests
per second is the next measurement.

## Exercises

1. Vary the chunk at fixed batch 64 in finer steps (128, 192, 256, 384)
   and separate the per-mixed-step cost from the per-token one.
2. Run the same protocol with `--cudagraph-mode FULL` (or none) and see
   whether the small-chunk gap moves with the graph mode.
3. Refit the per-sequence constant on pure-decode ITLs from this data
   and re-check the M55 sweep: does the mixed-step residual then explain
   the rest?
