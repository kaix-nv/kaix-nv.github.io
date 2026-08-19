---
layout: post
math: true
title: "Building tinyperf M9: Methodologies and calibration — the model meets a real GPU"
date: 2026-08-18 17:50:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "SOL / Proj / Calibrated tiers, a five-knob fit against real cuBLAS on an RTX A6000 — and the model bug the residuals caught."
---

*Milestone 9 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) —
`tinyperf/methodology.py`, `tools/measure_gemm.py`,
`tools/fit_calibration.py` · Demo: `examples/07_methodologies.py`.*

Eight milestones built a model. This one asks the uncomfortable question:
*how wrong is it?* — and makes the answer a feature instead of a footnote.

## Trust as a parameter: SOL / PROJ / CALIBRATED

Production performance models expose fidelity as named *methodologies*, and
we adopt the same three-tier ladder:

- **SOL** ("speed of light") — pure roofline per op. No tile search, no
  launch overheads, no latency terms, ideal traffic. Nothing real ever
  beats it; the gap to SOL is the room kernels leave on the table.
- **PROJ** — the projected model: everything milestones 1–8 built.
  Physics plus modeled software effects, uncalibrated constants.
- **CALIBRATED** — PROJ running on a device whose constants were *fitted
  to measurements* of real kernel libraries. Same mechanism, honest
  numbers. Asking for CALIBRATED without a calibration file is an error,
  not a silent fallback — the tiers only mean something if they're honest.

The calibration itself is five scalars with clear physical meaning
(`math_efficiency`, `dram_efficiency`, `l2_bw_gbps`, `kernel_launch_us`,
`fmha_math_efficiency`), stored per device with provenance. Five knobs
against dozens of measurements spanning four shape regimes: enough freedom
to absorb honest hardware constants, not enough to memorize the benchmark.

## The pipeline

```
tools/measure_gemm.py     (torch, runs on the GPU)   -> measurements JSON
tools/fit_calibration.py  (stdlib, runs anywhere)    -> data/calibration/<device>.json
execute(graph, dev, methodology=CALIBRATED)          -> honest numbers
```

The measurement grid spans the regimes the model distinguishes: square
math-bound GEMMs, skinny memory-bound decode GEMMs, batched attention
shapes, launch-bound crumbs. The fit is coordinate descent on the mean
absolute log-error — symmetric in over/under-prediction, scale-free from
microseconds to milliseconds.

## Real numbers: calibrating an RTX A6000

We ran the pipeline against cuBLAS (torch 2.9, CUDA 12.8) on an RTX A6000 —
deliberately *not* a datacenter GPU: GA102 has a 6 MB L2 and GDDR6, about
as far from an A100's 40 MB and HBM as Ampere gets. Fitted constants:

```
math_efficiency   0.75    (sustained tensor clocks sit well below boost)
dram_efficiency   0.85    (achievable GDDR6 fraction)
l2_bw_scale       1.5x    (our milestone-1 estimate was low)
kernel_launch_us  12      (includes framework dispatch, not just launch)
mean |log-error|: 0.405 uncalibrated  ->  0.110 calibrated  (~11% typical)
```

Every constant is physically interpretable — that's the point of fitting
five knobs instead of five hundred. The residual that won't calibrate away
(tiny GEMMs land at ~0.4x) is framework dispatch overhead we don't model,
and now we know its size.

## Calibration found a model bug

The best part wasn't the fit — it was the residual *before* the fit. The
uncalibrated model over-predicted 8192³ by **1.73x** on this card while
matching A100 anchors. That localized the error precisely: milestone 2's
L2-reuse model judged whether a wave's **full-K working set** (66 MB here)
fits in L2, predicting 14x thrashing. But real kernels stream K in
BK-sized slices — only a ~500 KB *slice* needs to be resident at a time.
The A100's 40 MB L2 masked the wrong granularity; the A6000's 6 MB exposed
it. One-line fix in `gemm_model.py`, A100 anchors unchanged, A6000 squares
now land at 0.93–1.00x.

That is the real lesson of this milestone: **calibration is not just
fitting constants — structured residuals are a debugger for the model
itself.** A shape-dependent error points at a mechanism; a uniform error
points at a constant. And calibrate on hardware *unlike* the hardware you
developed against, or your bugs will hide in the corners you never moved.

## The ladder in action

LLaMA2-7B on the calibrated A6000 (`examples/07_methodologies.py`):

```
workload                 SOL ms  PROJ ms   CAL ms  PROJ/SOL  CAL/PROJ
prefill b=1 s=2048       197.89   221.20   276.79      1.12      1.25
decode  b=1 kv=2048       18.62    20.18    26.75      1.08      1.33
decode  b=32 kv=2048      62.40    64.30    78.66      1.03      1.22
```

Read the two ratio columns separately: PROJ/SOL is *modeled* software
inefficiency (tiles, waves, launches); CAL/PROJ is *measured* reality the
uncalibrated constants missed (sustained clocks, achievable bandwidth).
Production tools report these tiers side by side for exactly this reason —
the spread between them is the error bar the single-number tools hide.

## In production-scale models

The same ladder, industrialized: named methodologies selectable per run
(speed-of-light, max-optimized, projected, calibrated), per-architecture
calibration overlays maintained against silicon as it arrives, and
regression fleets that re-fit and re-validate the constants continuously.
Bring-up of a new chip is largely this loop: measure, fit, chase the
residuals that won't fit — each one is either a kernel bug or a model bug.

## Exercises

1. Run `tools/measure_gemm.py` on an H100 or B200 and compare the fitted
   constants across architectures — which ones are stable, which move?
2. Add per-regime calibration: separate `math_efficiency` for math-bound vs
   memory-bound shapes. Does the tiny-GEMM residual shrink or just move?
3. Calibrate `fmha_math_efficiency` against published FlashAttention-2/3
   benchmarks and re-run the milestone-7 saving curve.

*Next: milestone 10 puts the A6000 measurements to further use — split-K,
the missing kernel trick that decode-shaped GEMMs have been waiting for.*
