---
layout: post
math: true
title: "Building tinyperf M57: Closing the knee: three constants become two mechanisms, and the knee becomes a probability"
date: 2026-09-09 12:13:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "The pure decode step carries no engine cost; the mixed step carries a fixed and a per-sequence one; the graph runs at the next captured batch size. With those, the model's knee lands on the measured one — and a fine sweep shows the real server is bistable there, so the knee is a probability over traces."
---

*Milestone 57 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `mixed_step_*` in `methodology.py`, `padded_batch` in `serving.py`, `sweep.tip_fraction`.*

Three milestones of serving-dynamics work left one residual: the knee.
Below it the engine was within a few percent, past it within ten, and at
the transition the real server tipped over while the model did not. This
milestone set out to close that with data already in hand and one fine
sweep, and closed it in a way I did not expect — by finding that the knee
on this hardware is not a rate.

## First, the constants: re-partitioned, not re-fitted

Milestone 56 measured decode steps and mixed steps directly. Its pure
decode steps, between injections, matched the graph alone within 0.7 ms
at every batch size from one to sixty-four. So milestone 55's constant —
63 µs per running sequence per step, applied to every step — was wrong
about *where* the cost lives. It goes to zero. Its cost moves to the
steps that carry prefill: on the sixteen mixed cells the residual is a
clean `a + b·B`, 3.3 ms per mixed step plus 213 µs per running sequence,
fitted on the small-chunk cells and checked on the large ones. That is
the price of leaving the full CUDA graph: piecewise replay, eager
attention, eager input preparation, all scaling with the batch.

Held out on the milestone-54 sweep, with nothing else changed, the
model's knee moved from 5 requests per second to 4 — the measured one —
and TTFT at the knee improved from 0.51 to 0.62.

## Then the fine sweep, and a surprise

Nine rates from 3.0 to 5.0 requests per second in quarter steps, 240
requests each, the same server. Predictions frozen with the pre-refit
model, so the sweep is held out for the constants above.

```
  rate   TTFT p50 meas   TTFT p95 meas   TPOT meas
  3.00           368.1           784.2       61.76
  3.25           414.7          1513.1       84.56
  3.50          4602.4          6895.7      133.37
  3.75          2654.2          5226.8      121.11
  4.00          1144.5          2820.7      108.34
  4.25          2637.0          5010.0      109.71
  4.50          3988.6          7728.0      110.82
  5.00          5955.4         12263.8      112.89
```

Read the middle rows twice. At 3.5 requests per second the server's
median TTFT is 4.6 seconds and its TPOT is 133 ms — *worse* than at 5
requests per second. At 4.0 it recovers to 1.1 seconds. These are not
noise around a curve; they are two branches. Near saturation a
continuous-batching engine is bistable: a burst of arrivals fills the
running set, the steps lengthen because every one of them now carries
prefill for the queue, the longer steps hold more sequences in flight,
and the queue never drains within the run. Requests admitted together
also finish together — same output length — and free their slots in
cohorts, so the prefill arrives in cohorts of two or three prompts per
step, which is why the tipped branch runs *slower* than steady
saturation. Whether an eighty-second run at 3.5 requests per second
tips is decided by its first few seconds of arrivals.

A deterministic simulation with one seed lands on one branch. The
model's single trace stayed on the quiet branch until 4.0; the measured
runs tipped at 3.25 and 3.5 and recovered at 4.0. Comparing those point
by point is comparing two draws from a distribution.

## The batch the graph actually runs

Before treating the transition as pure chance, one more mechanism had to
be found, because the model was still 7% low on the step at 3.0 requests
per second — and on the decode-heavy sweep of milestone 55 at every rate
— while milestone 56's direct step measurements were exact. The
difference between those measurements is the batch sizes they used.
Milestone 56 measured at 1, 8, 32 and 64. vLLM captures CUDA graphs at
1, 2, 4, 8, 16, 24, 32, 40, 48, 56 and 64 sequences and replays the next
size up: a 34-sequence decode step runs the 40-sequence graph and pays
for 40 rows of attention, sampling and bookkeeping. Continuous serving
lives at sizes like 34 and 60; the step measurement sat exactly on
capture sizes and could not see it. `simulate` now pads the running
batch to the engine's capture sizes (a mechanism, no constant, off with
`cudagraph_sizes=None`), and the 7% closes: TPOT at 1–3 requests per
second on the milestone-54 sweep is 0.98–1.03, the decode-heavy sweep
0.93–1.00, the fine sweep's stable point 0.97.

## The knee as a probability

With the model's step costs now right on both branches, the transition
is what is left, and the right question is not "where is the knee" but
"how likely is a run at this rate to tip". `sweep.tip_fraction` runs the
engine over many Poisson traces and reports the fraction whose TTFT p95
exceeds a threshold (here 3× the 1 req/s value):

```
  rate   measured run tipped?   model tipping fraction (12 traces)
  3.00        no                      0.00
  3.25        yes                     0.33
  3.50        yes                     0.33
  3.75        yes                     0.58
  4.00        yes                     0.75
  4.25        yes                     0.75
  4.50        yes                     0.83
  4.75        yes                     1.00
  5.00        yes                     1.00
```

The model's transition band, 3.25 to 4.5, brackets the measured one; the
stable branch at 3.0 is priced at 1.04 on TTFT and 0.97 on TPOT, the
saturated branch above 4.25 at 0.93–0.94 on p95 and 0.93 on TPOT. What
the model does not reproduce is the *depth* of the 3.5 run — 6.9 s
against the ensemble's worst trace of 3.2 s — and it is not obvious that
any model should: that is one draw of a heavy tail. A deployment
engineer reading this table would provision for the band, not the rate,
which is the useful answer and the one the single-number knee of
milestones 15 and 16 could not give.

## What this closes, and what it opens

Closed: the serving-dynamics thread's residuals are now either priced
mechanisms (mixed-step cost, graph padding, async admission, online
overhead) or a described distribution (the transition). The three
serving sweeps and the mixed-step cells share one calibration with no
constant fitted on the data it is checked against.

Open: the depth of the tipped branch; the decode-heavy sweep's 5%
residual at saturation; all of it on one model, one GPU, one engine
version. And an erratum owed to milestone 16: its knee finder returns a
rate. It should return a rate and a width, and the ensemble is how.

## Exercises

1. Run `tip_fraction` with 48 traces and plot the fraction against rate:
   is the transition a logistic in rate, and what sets its width?
2. Break the cohorts: give requests output lengths drawn from a
   distribution (milestone 20's `max_gen`) and see whether the tipped
   branch's TPOT still exceeds steady saturation.
3. Change `cudagraph_sizes` to a finer list (every 4) and re-price the
   3.0 req/s point. How much of the step is padding at batch 34?
