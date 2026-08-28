---
layout: post
math: true
title: "Building tinyperf M17: Speculative decoding — the thing that moves walls"
date: 2026-08-28 14:45:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "A draft model changes the physics: the 20 ms SLO wall becomes serveable, the steep segment collapses 11x — and the crossover at alpha 0.5 falls out of the cost ratio."
---

*Milestone 17 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tinyperf/spec_decode.py` · Demo: `examples/15_spec_decode.py`.*

Milestone 16 ended on a hard fact: below a deployment's single-request
decode time, an SLO is a **wall** — unserveable at any price, because
decode emits one token per weight-streaming pass and money does not buy
memory bandwidth. This milestone prices the standard algorithmic answer.

## The trick, priced by our own model

Speculative decoding: a small *draft* model proposes k tokens
autoregressively, the big *target* verifies all k+1 in one forward pass,
rejection sampling keeps the output distribution exactly the target's.
The whole trick rests on one claim — **verification is nearly free** —
and our model can check it rather than assert it: a decode-shaped GEMM at
m=1 and at m=5 streams the same weights and takes the same time while
memory-bound (63.6 vs 63.7 µs on the calibrated A6000), and FMHA reads
the same KV either way. One verify pass ≈ one decode step. The M2 tile
model, built nine milestones ago, contains the reason speculative
decoding works.

With per-token acceptance `alpha` (i.i.d., the standard first-order
model), a k-draft cycle emits `E = (1 - alpha^(k+1)) / (1 - alpha)`
tokens, so

```
effective TPOT = (k x draft_decode + target_verify) / E
```

`SpecStepLatencyModel` implements the same interface as the milestone-13
`StepLatencyModel` — so it plugs into the serving engine, the knee
finder, and the SLO sweep *unchanged*. Composition, again, is the whole
architecture. (It is a mean-field model — expected tokens, no acceptance
variance — and `alpha` is a measured property of the model pair and
traffic, not something a simulator can invent. Draft weights and KV are
charged in time but not against the KV budget; a documented
approximation.)

## The wall moves

LLaMA2-7B + TinyLlama-1.1B draft (k=4), calibrated A6000. M16's table,
re-run ($/Mtok at the SLO knee; "wall" = unserveable):

```
 TPOT SLO        plain   spec a=0.6   spec a=0.8   spec a=0.9
     12ms         wall         wall         wall         wall
     16ms         wall         wall         wall        1.95$
     20ms         wall         wall        1.94$        0.66$
     30ms        3.87$        1.95$        0.50$        0.34$
     60ms        0.36$        0.28$        0.26$        0.18$
```

Three readings:

- **The wall retreats with alpha.** 20 ms — impossible at any price in
  milestone 16 — costs $1.94 at alpha=0.8 and $0.66 at alpha=0.9. The
  wall didn't vanish; it moved to ~cycle_time / E (12 ms is still
  physics, even at alpha=0.9).
- **The steep segment collapses.** The 30 ms row falls from $3.87 to
  $0.34 — 11x — because the expensive thing about tight SLOs was tiny
  batches, and speculation multiplies tokens-per-pass instead.
- **The flat segment barely cares** (0.36 -> 0.18 at 60 ms): where
  batching already amortizes weights, speculation only halves the
  per-token cost at best — and spends draft compute to do it.

## No free lunch: the crossover

```
plain: 25.7 ms      alpha=0.5: 26.0 ms   <- break-even
alpha=0.0: 50.3 ms  alpha=0.8: 15.0 ms
alpha=0.4: 30.5 ms  alpha=0.9: 12.3 ms
```

The break-even sits at alpha ~ 0.5 for this pair, and the model says why:
the 1.1B draft costs ~1/5 of the 7B target per step, so k=4 drafts add
about one target-step of overhead — which pays for itself only when the
cycle emits ~2 tokens, i.e. `E >= 2`, i.e. alpha >= 0.5. Below that,
speculation is pure overhead. Every term in that sentence came out of the
calibrated model rather than a benchmark run.

## In production-scale models

Production stacks model speculative decoding with measured
per-position acceptance curves (not i.i.d. alpha), draft/target pairs as
a searchable axis (draft size, k, tree-speculation fan-out), and the
batch-dependence this milestone's approximation flags: at large
batch x (k+1), verification stops being free and the win shrinks —
which is why speculation shines at interactive batch sizes and fades at
throughput-oriented ones.

## Exercises

1. Batch-dependence: price verify as a chunked forward honestly (m = k+1
   rows per sequence) and find the batch where verification stops being
   free on each device.
2. Sweep k at fixed alpha: the optimum shifts with the draft/target cost
   ratio. Derive the closed-form k* and check it against the sweep.
3. Tree speculation: n parallel draft chains raise E per verify at the
   cost of a wider verify pass. Where does the tree beat the chain?
4. Self-speculation (layer-skip drafts): draft cost becomes a fraction of
   the target's — redo the crossover with draft = 0.3x target.

*Still queued: a datacenter-GPU calibration, so these tables carry measured constants on every column.*
