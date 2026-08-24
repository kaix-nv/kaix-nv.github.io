---
layout: post
math: true
title: "Building tinyperf M15: Capacity planning — sweeping the serving simulator"
date: 2026-08-23 23:20:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "The event engine swept by the harness: SLO knees found automatically, the U-shaped cost curve, and \$/Mtok tables from 33 simulations in 0.7 seconds."
---

*Milestone 15 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `price_serving_config` in `tinyperf/sweep.py` · Demo: `examples/13_capacity_planning.py`.*

Milestone 14 closed the core series by sweeping the *step* model. The
first thing anyone asks next: sweep the *serving* simulator instead. Steps
have latencies; deployments have SLOs, knees, and unit economics. This
milestone composes M13 (the event engine) with M14 (the harness) into the
artifact capacity planning actually runs on: **$/Mtok vs goodput, at an
SLO**.

## A serving point is a config too

`price_serving_config` runs one (deployment x offered-load) point through
the event simulator and reduces it to what procurement reads:

- **goodput** — tokens from requests that met the SLO, per second;
- **$/Mtok** — `world x $/GPU-hr / goodput` (device prices are
  illustrative public-cloud numbers, marked as such in the specs);
- the p95s the SLO is written against.

Two lessons surfaced *while writing the tests* — both worth more than the
code:

**The SLO must be dual.** Our first pricer only checked TPOT, and on an
A100 with a batch cap it never fired: under overload, decode steps stay
fast — the damage all lands in TTFT as the queue grows. A TPOT-only SLO
sleeps through the incident. Real SLOs bound both, so does ours
(TPOT p95 <= 60 ms AND TTFT <= 1 s).

**Cost is U-shaped in load.** Our first test asserted "more load, more
$/Mtok" — and failed, because it's economically backwards. An idle GPU
makes *expensive* tokens ($2.25/Mtok at 1 req/s on A100); utilization
cheapens them toward the knee ($0.26 at 12 req/s); only past SLO collapse
does waste drive cost back up ($1.02 at 200 req/s, with 78% of tokens
missing the SLO). The test now pins the U-curve. Every serving fleet
lives on the left wall of that U, as close to the knee as it dares.

## The artifact

LLaMA2-7B, prompt 512 / gen 128, SLO: TPOT p95 <= 60 ms, TTFT p95 <= 1 s.
33 serving simulations across three devices x eleven load points, 0.7 s:

```
device           $/hr  knee req/s  goodput tok/s  TPOT p95  TTFT p95   $/Mtok
RTX_A6000        0.49         3.0            377     57.3m      176m     0.36
A100_SXM_80GB    1.29        16.0           1665     33.0m      681m     0.22
H100_SXM         2.49        32.0           3089     17.0m      366m     0.22
```

Readings an operator would pay for:

- **The knee is found, not guessed**: for each device, the highest load
  where 99%+ of tokens still meet the SLO. The A6000's knee is TPOT-bound
  (57.3 of 60 ms); the datacenter parts are TTFT-bound — different walls
  of the same U.
- **A100 and H100 tie at $0.22/Mtok** for this workload — but the tie is
  not a wash: at equal unit cost, the H100 buys 1.9x the goodput per
  device *and* halves TPOT. Same price per token, very different fleet
  size and user experience.
- **One honest asterisk**: the A6000 row runs CALIBRATED (real cuBLAS
  constants); the A100/H100 rows run PROJ, the optimistic tier. A real
  cross-device decision wants calibrations everywhere — which is exactly
  why the next measurement campaign (H100/B200) is queued.

## In production-scale models

This two-layer sweep — an event simulator per point, an analytical model
per step, a harness over everything — is how fleet capacity actually gets
planned: SLO-goodput surfaces over (model, hardware, parallelism, load),
re-priced nightly, with procurement reading the $/Mtok column and SRE
reading the knee margins. The whole tinyperf stack participates in every
cell of that table.

## Exercises

1. Add TP to the space: does llama2-7b at tp=2 ever beat tp=1 on $/Mtok,
   given the extra GPU?
2. Sweep the SLO itself: plot $/Mtok at knee vs TPOT budget from 20 to
   200 ms — the price curve of interactivity.
3. Mixed fleets: given a traffic forecast with a daily cycle, choose a
   two-device fleet that minimizes cost. (The frontier is the input.)
4. Calibrate H100, re-run this table, and measure how far PROJ's
   optimism moved the $/Mtok column.

*Next in the queue: calibrating a datacenter GPU, so every row of that table carries measured constants.*
