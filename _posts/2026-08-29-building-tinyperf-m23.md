---
layout: post
math: true
title: "Building tinyperf M23: Fixing the attention model — accumulator precision and pipeline fill"
date: 2026-08-29 13:45:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "The scores matrix is fp32 and enormous, and a two-iteration K loop cannot fill a pipeline. Two physics fixes cut the long-context error — and doubled the modeled value of attention fusion."
---

*Milestone 23 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `nets/transformer.py`, `gemm_model.py`, `operators.py`, `scheduler.py` · Validated across three calibrated devices.*

Cross-model validation kept pointing at one weakness: the model's error
grew with sequence length. Short prompts matched a reference model well;
long ones drifted optimistic, and the drift tracked context length. Since
prefill attention is the only term that grows quadratically with context,
the suspect was obvious. Two physics errors turned out to be hiding
there.

## Error 1: the scores matrix was stored at half its real precision

`QK^T` is a tensor-core matmul, and tensor cores accumulate in **fp32**.
The scores matrix that lands in memory before softmax is therefore fp32,
not fp16 — and it is, by a wide margin, the largest tensor in a
long-context prefill. Our operators inherited their output dtype from
their input, so we were pricing it at half its true size.

How big is that? For LLaMA2-7B at 32k context, per layer:

```
32 heads x 32768 queries x 16384 keys (causal) x 4 B = 68.7 GB
```

Sixty-nine gigabytes, per layer, of traffic the model was billing at
half rate. `BatchedMatMul` and `Softmax` now take an explicit
`out_dtype`, the builder marks scores as fp32 and probabilities back as
fp16 (what the PV matmul needs to hit tensor cores), and — a second
latent bug in the same family as milestone 22's — the scheduler now
passes the op's actual output dtype into the GEMM model, which had
supported mixed input/output precision all along and never been told.

## Error 2: a two-iteration K loop was assumed to run at full rate

Attention's `QK^T` contracts over `head_dim = 128`. At `BK = 64` that is
a main loop of **two iterations**. A multistage GEMM kernel needs
`SMEM_STAGES - 1` iterations just to fill its software pipeline; a loop
that short never reaches steady state. The old model applied only a
*tile-size* efficiency term, so it happily predicted 207 TFLOPS for that
GEMM on an A100 — 66% of peak, for a kernel that spends half its life in
prologue. The fix is one line of physics:

```python
split_iters = ceil(k_split / BK)
pipe_eff    = split_iters / (split_iters + SMEM_STAGES - 1)
```

K=128 gives 0.67; K=4096 gives 0.985. It bites attention and leaves every
dense layer untouched — which is exactly the shape of the error.

## A bug I caught on the way

The first version of that patch named its variable `k_iters`, shadowing
the outer loop bound that split-K uses to size each split. Every
subsequent split halved `k_split` again, and dense GEMMs came out **15x
too fast**. The sanity check that caught it was mechanical: after any
change to the kernel model, re-check the ops you did *not* mean to touch.
`qkv_proj` moving from 0.81 to 0.05 of its reference is not a result, it
is a stack trace.

## Results

The fix has to bite where the error was — long context — and nowhere else.
Predicted prefill time, LLaMA2-7B on A100, b=1:

```
  s=2048     108.7 ms  ->   113.4 ms   (+4.3%)
  s=8192     521.6 ms  ->   593.6 ms  (+13.8%)
  s=32768   3729.4 ms  ->  4826.7 ms  (+29.4%)
```

The correction scales with context exactly as a quadratic term should,
and short prompts barely move. That shape is the tell that this targeted
a real mechanism rather than applying a global fudge: it moved the
predictions that were wrong and left the ones that were right alone.
Measured against a reference model across three calibrated devices, the
error's growth with sequence length — the thing that made long context
untrustworthy — dropped by about 40% on every one of them.

And a free consistency check: **the value of attention fusion doubled**.

```
              fusion win, b=1
  s=2048       3.7%  ->   7.2%
  s=8192      10.5%  ->  20.6%
  s=32768     22.9%  ->  40.0%
```

Of course it did. FlashAttention exists precisely to keep that fp32
scores matrix off the memory bus; once you price the matrix correctly,
avoiding it is worth twice as much. Milestone 7's number was
understated for the same reason milestone 23 existed.

## What is still off

The PV matmul remains meaningfully faster than the reference. Here we are
deliberately not chasing it: we model the performance-optimal path where
softmax writes fp16 probabilities so PV can use tensor cores, and the
first-principles arithmetic (134 MB of probabilities at 2 TB/s ≈ 78 µs)
backs our number. A reference modeling an fp32-probability path would
land roughly twice as slow. When two models disagree and both are
self-consistent, the
honest move is to name the assumption, not to tune until the numbers
match.

## Exercises

1. Add an `accum_dtype` to the GEMM model and let split-K's fp32 partials
   flow from it rather than a hardcoded 4 bytes.
2. Model softmax as two passes when a score row exceeds shared memory
   (long context) and one pass when it fits — does the remaining softmax
   gap close?
3. Sweep `SMEM_STAGES` (3, 4 — Hopper-class pipelines are deeper) and see
   how the pipeline-fill penalty on attention changes.
