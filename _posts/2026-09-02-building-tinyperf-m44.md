---
layout: post
math: true
title: "Building tinyperf M44: Pipeline parallelism as a serving layout"
date: 2026-09-02 13:45:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "A pipeline stage is a memory lever first: weight-bound decode gains nothing per token, math-bound work shrinks with the micro-batch, and a lone prefill request still pays the whole traverse."
---

*Milestone 44 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `pp` in `serving.py`, `capacity.py`, `comm_model.py` · Example: `examples/28_pipeline_parallel.py`.*

Tensor parallelism stops paying past the NVLink domain (milestone 34), and
expert parallelism stops at the expert count. Models that still do not
fit, or fleets that want more requests in flight per replica, reach for
the third axis: cut the layer stack into stages and hand activations down
the line. This milestone adds `pp` to the step model, the capacity
planner and the sweep pricer, and asks what a stage buys and what it
costs.

## The model

A pipeline of `pp` stages holds `L/pp` layers each — as even as integers
allow, so 93 layers over 4 stages is 24/23/23/23 — with the embedding on
the first stage and the final norm and LM head on the last. The graph
builder gained three switches (`include_embed`, `include_head`,
`n_layers_override`) that build one stage as a small model; layer classes
(full attention, linear attention, windowed, the dense-first layers of a
MoE) scale with the override in proportion.

A step is priced the way a 1F1B engine runs it. The engine keeps `pp`
micro-batches in flight, one per stage, so a step over a batch of `B`
costs one traverse of a `B/pp` micro-batch — every stage's time, summed —
plus `pp − 1` hand-offs of `rows × hidden × dtype` bytes over NVLink when
`tp × pp` fits the domain and over the fabric when it does not. Memory is
planned for the heaviest stage: layer weights and KV take that stage's
share, the embedding and head are charged at whichever end is heavier.

Three consequences fall out before running anything:

- **Weight-bound decode gains nothing per token.** Each stage still
  streams its weights once per micro-batch; a smaller micro-batch does not
  make that faster.
- **Math-bound work shrinks with the micro-batch.** Rows per stage drop by
  `pp`, so a big-batch decode step approaches `1/pp` of its single-stage time.
- **A lone prefill request cannot be split.** Its TTFT is the full
  traverse, and the other stages idle meanwhile: the pipeline bubble.

## What a stage buys — LLaMA3-70B, H100, tp8

Decode step time relative to `tp8 pp1`, KV 4096:

```
  layout     GPUs    hops      b=1      b=8    b=256  GB/GPU
  tp8 pp1       8  NVLink     1.00     1.00     1.00    21.3
  tp8 pp2      16      IB     1.00     0.98     0.68    10.7
  tp8 pp4      32      IB     1.00     0.97     0.53     6.4
  tp8 pp8      64      IB     1.01     0.97     0.46     4.2
```

`b=1` pays only the hand-offs — under one percent even across the fabric.
`b=8` is weight-bound and barely moves. `b=256` is math-bound and tracks
the micro-batch. And the weights column halves each row: a stage is first
and foremost a memory lever.

## Same 64 GPUs, three layouts — Kimi-K3 NVFP4, B200

At a fixed budget the pipeline axis trades against expert-parallel width.
Decode at `b=32`, 16k context; TTFT for one 8k request:

```
  layout          GPUs  GB/GPU  max b  TPOT ms  TTFT 8k ms
  tp8 ep64 pp1      64      58    410     39.1         137
  tp8 ep32 pp2      64      51    832     24.8         183
  tp8 ep16 pp4      64      49   1620     17.9         242
```

Two things move in opposite directions. Halving the expert-parallel width
doubles the experts each rank hosts, so one request's layer runs its
expert work on 16 GPUs instead of 64 while the other stages wait — TTFT
grows 77% at `pp4`. Decode goes the other way: the micro-batch shrinks
per stage while each rank's expert streaming stays put, and the KV a
stage holds shrinks with its layer share, so the batch the fleet can hold
quadruples. Which layout wins is a question about the traffic — a
latency-sensitive prefill-heavy service and a throughput-bound decode
service pick different columns.

One semantic to keep in view: tinyperf's `batch` is the tokens in flight
in one pipeline, attention runs on one TP group, and experts spread over
the EP ranks. Layouts that rely on data-parallel attention across EP
groups are a different deployment and priced differently.

## What is not modeled

The steady-state 1F1B schedule only. Ramp-up and drain bubbles, stage
imbalance when a hybrid net's layer classes do not split evenly, and
interleaved (virtual-stage) schedules are all absent. Activation memory of
the micro-batches resident on a stage is charged at the whole batch, which
is conservative. And the hand-off is one message; engines that overlap it
with the next micro-batch's compute will land a little under our number.

## Exercises

1. Add the `pp − 1` warm-up bubble to a request's first decode step and see
   how much it matters at `b=8` versus `b=256`.
2. Build the stages of a hybrid net (`qwen3_8_27b`) and print each stage's
   time — how uneven are they, and would an uneven layer split fix it?
3. Price the same 64-GPU Kimi layouts under the serving SLO sweep
   (`examples/14`) and find the request rate where the ranking flips.
