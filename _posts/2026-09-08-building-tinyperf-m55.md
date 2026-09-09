---
layout: post
math: true
title: "Building tinyperf M55: Sixty-three microseconds: fitting the engine's per-sequence cost, and holding it out"
date: 2026-09-08 19:58:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "The residual milestone 54 refused to fit, fitted on a workload chosen to isolate it and checked on the one it was found in: 63 microseconds per running sequence per step, and half of the saturation gap closes with nothing else changed."
---

*Milestone 55 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `per_seq_step_overhead_us` in `methodology.py`, applied per step in `serving.py`.*

Milestone 54 ended with a residual it refused to fit. Under saturation the
live server's decode steps ran 10 ms longer than the model's at sixty-four
sequences, 6 ms at twenty, 1 ms at four — a cost that scales with the
running batch, which is the signature of the engine's per-sequence work:
scheduling each request, sampling its token, detokenizing it, pushing it
down its stream. The number was not fitted because the only data that
would fit it was the data it would then be checked against. This
milestone gets the other data.

## A workload chosen to isolate one term

The M54 sweep was prefill-heavy — 1024-token prompts, 128 tokens out — so
its saturation mixed two effects, the per-sequence cost and the mixed
prefill-plus-decode step. Flip the shape: 128-token prompts, 512 tokens
out. Now the steps are almost pure decode, the batch runs from fourteen
sequences at one request per second to the cap of sixty-four at four and
above, and whatever the model misses per step is the per-sequence term
and nothing else. Predictions frozen first, as always.

```
  rate   in-flight  running   TPOT meas   model   delta ms   delta/seq
     1        14.4     14.4       28.04   26.94       1.10       76 us
     2        33.8     33.8       32.82   29.94       2.88       85 us
     4        94.0     64.0       38.19   34.76       3.43       54 us
     8       238.6     64.0       38.28   34.10       4.18       65 us
```

One detail matters for the denominator. Above the knee, Little's law puts
94 and 239 requests "in flight" — but most of them are in the queue, and
the step never sees a queued request. The per-step cost scales with the
*running* batch, capped at 64. With that, the deltas line up at 54–85 µs
per sequence per step, and a least-squares line through the origin gives
**63 µs**.

Refitting the same workload with the constant is not evidence — it has to
land — and it lands: TPOT 0.98–1.00, throughput 1.03–1.05 at every rate.

## The held-out check

Then the M54 sweep, untouched since it was measured, with the constant
applied and nothing else changed:

```
  rate   TPOT meas   before      after     tok/s meas   before   after
     1       30.56   0.97        0.98            126     1.01    1.01
     2       39.52   0.97        1.00            248     1.02    1.02
     3       59.45   0.89        0.94            364     1.02    1.02
     4      106.36   0.79        0.86            456     1.06    1.05
     5      112.86   0.86        0.90            468     1.11    1.07
     6      113.19   0.89        0.92            478     1.11    1.07
     8      111.42   0.91        0.94            493     1.10    1.06
```

Everything moves the right way and nothing moves too far. Below the knee
TPOT is now within 2%. At saturation the constant closes about half of
the gap — 0.86–0.91 becomes 0.90–0.94, throughput 1.10 becomes 1.06 — and
the half it leaves is exactly what the workload was chosen to exclude:
the mixed prefill-plus-decode step, which on the 1024-token workload
carries a 2048-token budget of prefill chunks in nearly every saturated
step. That residual has a name now and a place to be measured, and it is
not this constant's job.

## What the constant is, and is not

Sixty-three microseconds per running sequence per step is the online
engine's bookkeeping: on this stack, a step with sixty-four sequences
spends 4 ms of its 38 on work the GPU graph does not contain. It lives in
the calibration next to the 25 ms online overhead and the 2.2 ms pipeline
step cost, and like them it is a property of the software — vLLM 0.15's
OpenAI server — not of the GPU. It is applied by the serving engine to
each step, scaled by the sequences in it, and never to an offline step:
the single-GPU, pipeline and prefix-cache envelopes are unchanged to the
byte, and the test checks that.

The TTFT side of both sweeps is not changed by this and still runs 10–30%
optimistic approaching the knee. That is the next residual, and it is a
different mechanism — step alignment of arriving requests under load —
that a different measurement will have to isolate.

## Exercises

1. Fit the constant on a *third* shape (prompt 2048 / gen 32) and see
   whether 63 µs survives; if not, the term is not purely per-sequence.
2. Turn off streaming in the load generator and refit. How much of the
   63 µs is the SSE chunk per token?
3. The remaining saturation gap on the 1024/128 sweep is the mixed step.
   Measure a mixed step directly — a fixed decode batch plus one prefill
   chunk, offline — and compare with `mixed_step_us`.
