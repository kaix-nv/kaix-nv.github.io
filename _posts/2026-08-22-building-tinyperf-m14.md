---
layout: post
math: true
title: "Building tinyperf M14: The sweep harness — from one answer to the answer space"
date: 2026-08-22 14:10:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Three functions turn millisecond pricing into Pareto frontiers: a 70B deployment designed in 0.1 seconds — and the series retrospective."
---

*Milestone 14 — the finale — of [building an analytical GPU performance
model from scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tinyperf/sweep.py` ·
Demo: `examples/12_sweep.py`.*

Thirteen milestones built a model that prices one configuration in
milliseconds. This last one explains why that speed was worth fighting
for: no design question is ever "how fast is this config?" It is "which
of these ten thousand configs should I run?"

## Three functions, no framework

```python
configs = expand({"model": [...], "tp": [1,2,4,8], "batch": [...], "kv": [...]})
results = run_sweep(configs, price_decode_config)   # capacity gate, then price
frontier = pareto(results, "tpot_ms", "tok_s_per_gpu")
```

`expand` is a cartesian product; `run_sweep` prices every point (capacity
check *first* — milestone 8's rule: never price what cannot exist, and
keep the reason when it can't); `pareto` reduces the cloud to the only
part an operator reads — the configs where you cannot improve one metric
without paying in another. Sequential, single-process, on purpose: at
~1 ms per config this is laptop work.

## Designing a 70B deployment, in a tenth of a second

LLaMA3-70B decode on H100: 72 configs (tp x batch x context), 31 feasible
— 41 fail capacity before pricing. The kv=2048 frontier:

```
  tp  batch  TPOT ms  tok/s/GPU  mem/GPU GB
   8      1    11.38         11        21.4   <- lowest latency money can buy
   8     16    12.01        166        22.7
   8     64    14.03        570        26.7
   8    256    23.11       1384        42.9   <- max efficiency, SLO permitting
```

Two readings, both the kind of thing you only see from a frontier:

- **tp=8 dominates everywhere** — not just at the low-latency end. Every
  tp<8 point is beaten on both axes, because a 70B at tp<8 either fails
  capacity outright or spends its memory budget on weights instead of
  batch. The frontier discovered milestone 8's conclusion on its own.
- **The latency-throughput exchange rate is visible**: from batch 1 to
  256, you pay 2x TPOT for 125x tok/s/GPU. The whole business of serving
  SLOs is choosing a point on that curve — and now it's a table you can
  read instead of a cluster-week you must schedule.

Put two architectures on one frontier (dense 70B vs Mixtral, kv=4096) and
the M11 story returns at deployment scale: every frontier point is
Mixtral, topping out at 3139 tok/s/GPU — active-param FLOPs win per-GPU
efficiency once EP capacity is paid for.

144 configurations, priced end to end, in **0.1 seconds** — that is the
entire reason the analytical layer had to be fast.

## In production-scale models

The loop does not change at scale; everything around it does. Production
sweep fleets run this exact expand-price-reduce cycle at six more orders
of magnitude: schedulers fanning millions of configs across clusters,
two-tier pricing (a fast methodology for the sweep, an accurate one to
re-price the frontier — the M9 ladder as infrastructure), adaptive
samplers that spend evaluations near the frontier instead of uniformly,
result lakes and dashboards, and nightly regression sweeps that catch a
model change by the drift it leaves in ten thousand answers.

## Where the series lands, for real

Fourteen milestones, ~1500 lines of stdlib Python, zero dependencies:

- a **device** is a bag of priceable rates (M1);
- a **kernel** is a searched config space over quantized resources
  (M2, M10);
- a **workload** is shapes and bytes, never execution (M3–M5);
- **parallelism** is sharded shapes plus explicit, priceable
  communication (M6, M11);
- the model must price **what the runtime runs**, not what the math says
  (M7), refuse what cannot exist (M8), and carry its own error bars (M9);
- **training** is a graph transformation plus schedule algebra (M12);
- **dynamics** belong to an event layer that consumes the analytical
  prices (M13);
- and the payoff is the **answer space**, not the answer (M14).

Everything left out — energy, DP/ZeRO, kernel-level event simulation,
imbalanced routing, paged KV — is an exercise on a skeleton you now know
how to extend. That was the point all along.

## Exercises

1. Two-tier sweep: price the full space under SOL, re-price only the
   frontier's neighborhood under PROJ/CALIBRATED. How much accuracy do
   you keep for how much speedup?
2. Add cost: $/GPU-hour per device, frontier of $/Mtok vs TPOT. Where
   does an A100 beat an H100 on price alone?
3. Sweep the *serving* simulator (M13) instead of the step model: the
   frontier becomes SLO-goodput vs cost — the actual capacity-planning
   artifact.
4. Adaptive sampling: replace `expand` with a sampler that refines near
   the current frontier. How few evaluations find the same frontier?

*That's the core series. Thanks for building along — the exercises are
yours now.*
