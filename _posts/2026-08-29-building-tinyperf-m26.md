---
layout: post
math: true
title: "Building tinyperf M26: The software stack is a calibration axis"
date: 2026-08-29 23:30:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Per-kernel overhead is 13.7 us under eager PyTorch and 3.5 us under CUDA graphs. On a step that issues 835 kernels, that one constant decides whether a prediction is useful."
---

*Milestone 26 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `Calibration.for_stack` in `methodology.py`, `tools/measure_launch.py`.*

Milestone 9 fitted five constants per device and called them hardware
properties. One of them never was. `kernel_launch_us` came out at 12 us,
fitted from PyTorch eager calls, and eager calls bundle Python and
framework dispatch with the actual launch. Milestone 22 flagged this in
a caveat. This milestone measures it and makes it a first-class axis.

## Two stacks, one number

Launch a long chain of deliberately tiny kernels — small enough that
launch dominates the work — and divide. On a B200:

```
  kernel     n   eager us/k   graph us/k   speedup
      64   100       13.864        2.780      4.99
     128   400       13.222        3.475      3.81
     256   400       16.556        3.707      4.47
     median          13.69         3.50        3.9x
```

Two things worth noticing. The eager number **independently confirms the
12 us that milestone 9 fitted from GEMM timings** — a constant fitted
against one workload reproducing under a completely different measurement
is the kind of agreement that makes a model trustworthy. And CUDA-graph
replay costs about a quarter as much, which is the whole reason serving
engines capture their decode steps.

So `Calibration.for_stack("graph")` swaps that one constant and leaves the
hardware ones alone. Math efficiency and achievable bandwidth are
properties of silicon; per-kernel overhead is a property of how the
kernels get issued.

## Why it matters more than it sounds

A decode step of a 64-layer hybrid issues **835 kernels**. At 12 us that
is 10 ms of pure overhead; at 3.5 us it is 2.9 ms. On a step whose
weight-streaming floor is 6.7 ms, that single constant is the difference
between a plausible prediction and a useless one.

Against a reference model, calibrated decode moves:

```
                       eager      graph
  dense (15 cells)      1.70       1.22
  hybrid 27B             1.80       1.15
```

Both regimes tighten by a third or more, and the improvement is largest
exactly where kernel counts are highest. The remaining ~20% is a
different question — the reference itself models per-op overheads we do
not — but the dominant term is now measured rather than assumed.

## The general lesson

An analytical model's constants are not all the same kind of thing. Some
are hardware (peak rates, achievable bandwidth, cache sizes). Some are
*software stack* (launch overhead, kernel selection, whether the graph is
captured). Milestone 9's ladder made fidelity explicit; this milestone
makes the second axis explicit too. **Calibrate against the stack you
intend to model, and say which stack a constant came from** — a number
fitted under eager PyTorch is simply not the same number as one fitted
under a graph-capturing engine, and quoting one for the other is a 4x
error hiding in a footnote.

## Exercises

1. Measure launch overhead on a second GPU generation — is 3.5 us a
   Blackwell number or a CUDA-runtime number?
2. Add a third stack: graph replay with a persistent-kernel engine, where
   the per-step overhead approaches zero. What does that do to the
   small-batch decode floor?
3. Sweep kernel count per layer (fusion aggressiveness) against the two
   launch constants — how much fusion is worth how much overhead?
