---
layout: post
math: true
title: "Building tinyperf M79: Error bars at the knee: the model's step error, run through the queue"
date: 2026-09-27 10:48:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Near saturation a 1-2% step error becomes a 40% error on the tail. Measured on every held-out cell since milestone 70, the model's step error, run back through the simulator, gives intervals that are narrow where the model is sure and wide where the queue amplifies. On two new knee-dense sweeps, frozen first, 35 of 36 measurements landed inside them."
---

*Milestone 79 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `prediction_interval` and `simulate(step_scale=)` in `serving.py` · Data: `data/validation/serving_error_retrospective_m79.json`, `comparison_qwen3_8b_rtx_a6000_serving_intervals.txt`.*

Four of the last nine milestones lost a cell at the same place: the knee.
M75's FlashInfer sweep read 0.72 on TTFT p95 at 3 req/s. M76 read 0.69
at its knee, and M78's long prompts read 1.20. Each time the engine's own
clock said the model's steps were right to within one or two percent.
Near saturation that is enough: the queue turns a 2% step error into 40%
on the tail. A point prediction there can't be trusted to that 40%. It
should come with the uncertainty it carries.

## The step error, measured

Every held-out online run since milestone 70 recorded the engine's step
clock. There are thirteen runs and 52 cells, across Qwen3-8B at one and
two GPUs and Qwen3-30B-A3B at two. For each cell, re-priced by today's
model, the step bias is the model's step times over the engine's, summed
over the cell's steps at their recorded composition. It lies between
0.981 and 1.026.

Applied to the cell it came from, it explains most of the misses. Every
TPOT lands within 0.98–1.02, and the knee cells come in:

```
  cell                                  model        with the cell's own step bias
  M75 FlashInfer, 3 req/s, TTFT p95      0.72                 1.02
  M78 long prompts, 2 req/s, TTFT p50    1.20                 1.00
  M70 long prompts, 2.5 req/s, p50/p95   1.09 / 1.17          0.95 / 1.04
```

So the error that matters at the knee is mostly a small, steady step
error, which the queue magnifies.

## An interval

`prediction_interval` runs the simulator with every step scaled across
the step-error spread, `simulate(step_scale=s)` for s within ±1.7%. That
is the 90th percentile of the 52 biases. It then widens the range by what
remains once each cell's own bias is applied: 5.2% on TTFT p50, 7.2% on
p95 and 1.3% on TPOT. Away from a knee the step scale barely moves the
queue and the interval is about ±10%. At a knee it opens up, because
that is where the queue amplifies.

Left one run out at a time, with the band estimated from the other
twelve, the intervals covered the 52 cells:

```
                          TTFT p50   TTFT p95   TPOT
  interval coverage          96%        94%      94%
  median width (hi/lo)      1.21x      1.27x    1.08x   (up to 2.2-2.8x at knees)
  a flat +-15% / +-5%        96%        92%      92%
```

A flat band covers as much. The difference is where each is wrong. The
flat band is ±15% everywhere, too wide away from the knee and too narrow
at it. The interval is narrow where the model is sure and wide where it
isn't. The cells it misses are M78's long-prompt run, whose step bias
lies outside the other runs' spread. That is the routing mechanism M78
named, a model gap rather than noise.

## Frozen, then measured

Two new knee-dense sweeps ran on GPU 1 alone, with Qwen3-8B, six rates
each around its knee. S1 was decode-heavy, S2 prefill-heavy. Here is S1,
each measured value with its interval as ratios to it:

```
  rate   TTFT p50 [low  point  high]    TTFT p95 [low  point  high]
   2        185 ms [0.92 0.99 1.08]         283 ms [0.87 0.96 1.07]
   2.75     224    [0.89 0.96 1.03]         523    [0.62 0.72 1.11]
   3        292    [0.81 0.91 1.02]        3161    [0.52 0.76 1.06]
   3.25     330    [0.81 0.91 0.99]*       5608    [0.74 0.88 1.09]
   3.5     1538    [0.46 0.79 1.14]        8900    [0.74 0.92 1.08]
   4.5     6775    [0.84 0.94 1.07]       18518    [0.84 0.95 1.07]
```

35 of the 36 cell-metric pairs, across both sweeps, landed inside their
intervals. The criterion was 80%. The point prediction with a flat band
holds 33, and it misses exactly the knee cells above, where the point
reads 0.72, 0.76 and 0.79. Fifteen of the 24 TTFT intervals are narrower
than the flat band, and the widest is 2.5×, at the knee. The one value
outside is S1's TTFT p50 at 3.25 req/s, 1% above its interval.

GPU 1 carried nothing else, and GPU 0 nothing at all. At S2's higher
rates GPU 1 ran at about 1,500 MHz rather than 1,650–1,700. The cells'
step bias there, 1.003–1.007, stayed inside the band.

## Exercises

1. The band is one number for all load levels. Is the step bias larger
   at saturation, where steps carry prefill? Fit it by step composition
   and see whether the knee intervals tighten.
2. M78's long-prompt cells fall outside because of a missing mechanism.
   Add composition-aware routing and re-run the retrospective. Do they
   come inside at the same band?
3. A slower GPU shifts every step the same way. Price TTFT against the
   SM clock the telemetry records, and see how much of the step-bias
   spread it accounts for.
