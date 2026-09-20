---
layout: post
math: true
title: "Building tinyperf M64: The isolation gap: the kernel is not slower in company, the step reads more bytes"
date: 2026-09-20 15:15:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "An investigation, not a mechanism. Nsight Systems, Nsight Compute and 20 kHz GPU-metrics sampling agree: alone, the fused-MoE kernel reads exactly its weights at the DRAM roof; inside the engine's step it stays at the roof but the step reads six gigabytes more than its kernels need. Fourteen experiments say what that traffic is not — and why a kernel benchmark is not a step benchmark."
---

*Milestone 64 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — no model change · Data: `data/validation/isolation_gap_gpt_oss_20b_rtx_a6000.json`.*

Milestone 63 ended on a finding it could not explain: vLLM's Triton
fused-MoE decode kernel, replayed alone on the same weights with the same
routing, ran 1.4–1.6× faster than inside the engine's decode step at
batch 8 and above, and identically at batch 1. The model prices the
in-step cost, so nothing was wrong with the predictions — but a
calibration constant whose cause is unknown is a debt. This milestone is
the investigation. It does not find the cause. It finds what the gap *is*,
which turns out to be a different thing from what it looked like, and it
eliminates most of the places it could be.

## First question: slower per launch, or more launches?

`nsys` with graph-node tracing on the engine's decode step at batch 8,
and on the isolated replay of the 24 expert layers:

```
                              gate-up launch     down launch      launches/step   streams   overlaps
  engine, CUDA graphs           608.6 us           306.3 us          48             1        0
  engine, eager                 608.8 us           307.3 us          48             1        0
  replay, CUDA graphs           435.0 us           221.2 us          48             1        0
```

Same grid (5760 and 2880 blocks of 128 threads), same count, one stream,
no concurrent kernel, memcpy or memset. Slower per launch, then. And not
transient: with each expert layer executed twice inside the engine's
graph, the second copy — running immediately after an identical first —
costs the same 649 / 327 µs.

## Second question: is it the kernel?

Nsight Compute, which replays each profiled kernel in isolation, says
no. A whole decode step's 48 fused-MoE launches, profiled in situ:

```
                          kernels   total      DRAM read   DRAM throughput
  engine (in situ)          48      17.05 ms   11.71 GB     93.7 % of peak
  replay                    48      16.01 ms   10.96 GB     93.3 % of peak
```

Per launch the bytes are exactly the touched experts' weights (a layer
that routes to 9 experts reads 298.9 MB of gate-up weights, 9 × 33.2 MB),
and both run at 94% of the DRAM roof. In application-replay mode — live
clocks, no cache flush — the in-step launches read 431 and 398 MB in 619
and 568 µs at 96% of peak. Whenever this kernel runs alone, with any
tool, it is a well-behaved streaming kernel.

## Third question: what does the GPU do during the live kernel?

`nsys --gpu-metrics-devices` samples hardware counters at 20 kHz next to
the kernel timeline. Inside the fused-MoE windows of the live run:

```
                        DRAM read     GPC clock   SMs active
  engine step            90.6 %       1841 MHz      95.8 %
  isolated replay        93.7 %       1861 MHz      92.5 %
```

The live kernel is at the roof too. Same clock. It just runs 1.4× longer.
A bandwidth-bound kernel that runs longer at the same bandwidth is moving
more bytes. Integrating the sampled bandwidth over whole steps (the
memory clock is 7.6 GHz, so the roof is 730 GB/s, not the datasheet's
768) makes it exact — the replay integrates to 11.00 GB per pass against
Nsight Compute's 10.96:

```
              live DRAM read    kernels need    excess      excess per touched expert-block
  batch  1       7.79 GB           7.21 GB      +0.6 GB          6 MB   (12% of the weights)
  batch  8      19.84 GB          13.96 GB      +5.9 GB         25 MB   (51%)
  batch 32      27.79 GB          21.46 GB      +6.3 GB         17 MB   (33%)
```

That is the gap. The live decode step reads about six gigabytes per step
that its kernels, run alone, do not need — at batch 8 and 32 alike, and
almost none at batch 1. Per touched expert per layer it is half an
expert's weights at batch 8, a third at batch 32, an eighth at batch 1;
`1 / (1 + excess)` reproduces milestone 63's step-derived efficiencies
(0.67 against 0.65–0.74 at batch 8, 0.75 against 0.72–0.74 at batch 32).
The kernel is not inefficient in company. Something in the live pipeline
costs DRAM traffic in proportion to the number of expert blocks the step
touches.

## What it is not

Each of these was tried in the isolated graph replay, and each left the
expert layers at their 16.08 ms baseline (ratio 1.00 unless noted):

- **Cache and memory neighbours** between layers: an 8 MB L2 flush, a
  200 MB device copy, a skinny bandwidth GEMM, a 4096³ tensor-core GEMM
  (0.90), ten tiny elementwise kernels, GPU idle gaps of 50 µs, 500 µs
  and 2 ms; the engine's own norms, router and attention projections,
  singly and together.
- **Thermal**: a 150 s soak at the power cap drifts 16.10 → 16.17 ms.
- **Host activity**: threads spinning on event queries, stream queries,
  small device-to-host copies, memory queries — the things the engine's
  CPU thread does during a step.
- **Process history**: the engine's own decode run for 12 s in the same
  process leaves a subsequent replay at 16.09 ms.
- **L2 policy**: the persisting set-aside is the 1.125 MB default in
  both contexts with no access-policy windows; clearing it in the engine
  changes the step by 1%; reserving up to 4.1 MB of the 6 MB L2 in the
  replay changes nothing — the streaming kernel does not care about L2
  capacity.
- **The kernel's inputs**: every launch argument logged side by side in
  one process — shapes, strides, 256-byte alignment, config
  `{16, 32, 64}`, flags — identical; producing the activation tile freshly
  before each call (a clone, an in-place rewrite, the engine's own norm),
  and fresh router logits, changes nothing.
- **Clocks and power**: 1.87–1.89 GHz SM, 7.6 GHz memory, the 300 W cap
  active, in both.

And two things that looked like evidence and were not. The torch
profiler inflates this kernel by 22% when it is replayed alone (16.1 →
19.7 ms per pass) and by about nothing in-step, so profile-derived
efficiency curves were wrong by the difference. And timing an eager
24-layer replay with CUDA events measures the CPU: the launches take 24
ms per pass on this host, the kernels 16.

The one quantitative regularity — an excess of roughly 25 MB per
touched 16-row expert block at batch 8 — is the size of the activation
tile a block's 270 programs re-read across their 45 K-steps. If those
re-reads missed L2 in the live step and hit alone, the numbers would
work. But the tile is 46 KB, re-touched every microsecond, and making it
freshly written did not slow the replay. So it is a candidate with the
right size and no mechanism, and the artifact says so.

## What this means for the model

Nothing in the model changes, and one thing about it is now understood.
The milestone-63 curves are *pipeline* efficiencies: bytes-time at the
DRAM roof including whatever the live step reads beyond the weights. They
were derived from unprofiled step times, which is why they are right.
Had they been derived from the kernel — a microbenchmark, or Nsight
Compute in any of its modes, which by design drains the GPU and measures
the kernel alone — they would have priced this kernel class 1.4× too
cheap at batch 8 and above. That is a methodological result worth more
than the cause: **for this kernel on this stack, a kernel benchmark is
not a step benchmark**, and the envelope now says which one the model's
constants describe.

The two-GPU batch-32 rank (milestone 62's open item, still at 0.71–0.73)
is the same phenomenon with more company: an all-gather and a
reduce-scatter between the expert kernels, and 16 local sequences.

## Exercises

1. Count the bytes at the source: Nsight Compute cannot see the live
   pipeline, but `nsys` GPU metrics can be sampled per L2 slice on newer
   parts. Find a counter that separates weight-stream misses from
   re-read misses and run it on the live step.
2. Change the block's activation traffic without changing anything
   else: a Triton config with `BLOCK_SIZE_N = 64` halves the number of
   programs that re-read each tile. If the live excess halves, the tile
   was the source; if it does not, it was not.
3. Reproduce the excess in the replay by running the *whole* layer
   (attention with real metadata, norms, router, experts) from captured
   inputs. Bisect from there.
