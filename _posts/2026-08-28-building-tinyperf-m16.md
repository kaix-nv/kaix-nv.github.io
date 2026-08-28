---
layout: post
math: true
title: "Building tinyperf M16: The SLO price curve — what interactivity costs"
date: 2026-08-28 12:40:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Sweep the SLO budget itself: every device shows a wall, a steep segment, and a flat one — and the last 30 ms of latency cost 13x."
---

*Milestone 16 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `find_knee` in `tinyperf/sweep.py` · Demo: `examples/14_slo_price_curve.py`.*

Milestone 15 found each deployment's knee at one SLO. The obvious next
question is the one product and infra teams actually argue about: *what
does the SLO itself cost?* Tighten TPOT from 80 ms to 30 ms — what happens
to $/Mtok? This milestone sweeps the SLO axis and turns that argument into
a table.

## Two small pieces of machinery

**A shared step-price memo.** Sweeps re-simulate the same deployment
hundreds of times; `simulate()` now accepts a shared `StepLatencyModel`,
and the sweep layer caches one per deployment. Result: ~315 serving
simulations in 1.8 s — every simulation after the first prices its steps
from the memo.

**A knee finder.** `find_knee(cfg)` sweeps offered load and returns the
highest-goodput operating point that keeps 99% of tokens inside the SLO —
or `None`, which turns out to be the most interesting return value in the
milestone.

## The curve

LLaMA2-7B, prompt 512 / gen 128, TTFT <= 1 s. Each cell is $/Mtok at that
device's SLO knee (A6000 CALIBRATED, datacenter parts PROJ):

```
 TPOT SLO       RTX_A6000   A100_SXM_80GB        H100_SXM
     20ms        — wall —           0.35$           0.22$
     30ms           3.87$           0.25$           0.22$
     40ms           1.00$           0.22$           0.22$
     60ms           0.36$           0.22$           0.22$
     80ms           0.29$           0.22$           0.22$
    200ms           0.29$           0.22$           0.22$
```

Every column has the same three-segment shape, and each segment carries an
operator lesson:

- **The wall.** Below a deployment's single-request decode time, no load
  level qualifies — `find_knee` returns None. The A6000's calibrated
  b=1 decode is ~26 ms, so a 20 ms TPOT SLO is not expensive there; it is
  *impossible*. Walls move only with better hardware, quantization, or
  speculative decoding — not with money.
- **The steep segment.** Just above the wall, interactivity is bought by
  running tiny batches: at 30 ms the A6000 serves 35 tok/s of goodput and
  every million tokens costs **$3.87 — 13x its own flat price**. The last
  30 milliseconds of latency are the expensive ones.
- **The flat segment.** Once the SLO clears the knee's natural TPOT, the
  constraint stops binding and only capacity economics remain: the H100
  is flat at $0.22 from 20 ms on — for this workload it is never
  SLO-bound, so paying for it *because of* a latency requirement below
  ~40 ms is the only regime where that argument holds.

That last point is the negotiation in one table: an SLO discussion is a
walk along a known curve, and the curve says precisely when tightening is
free, when it costs multiples, and when it is physics.

## In production-scale models

Fleet planners maintain exactly these surfaces — cost at knee over (model,
hardware, SLO, context mix) — regenerated as models, kernels, and prices
move. The two-tier trick applies here too: sweep the SLO axis under PROJ,
re-price the operating points you would actually sign under CALIBRATED.

## Exercises

1. Sweep the TTFT SLO instead: prefill-bound walls behave differently —
   find a workload where TTFT, not TPOT, is the binding axis everywhere.
2. Price speculative decoding as a wall-mover: if a draft model halves
   effective TPOT at b=1, how far left does each wall shift, and what
   does the steep segment do?
3. Add prompt-length mix: rerun the curve at prompt 4096. Which segment
   moves most?
4. Two-tier it: PROJ for the sweep, CALIBRATED re-pricing at the chosen
   point — how often does the *chosen config* change?

*Still queued: calibrating a datacenter GPU, so the PROJ columns of these tables carry measured constants too.*
