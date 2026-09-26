---
layout: post
math: true
title: "Building tinyperf M73: Tensor parallelism under load: the link's real curve, and a kernel that never finishes"
date: 2026-09-25 16:00:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Tensor parallelism had only been checked offline, with a named residual. Measured densely, the two-GPU link does not follow one latency-plus-bandwidth line, and decode traffic sits exactly where NCCL changes protocol; priced from the measured curve, the tp=2 decode step lands within 4% and seven of eight held-out online cells within the frozen criteria. And vLLM's default all-reduce on this PCIe pair takes seven seconds a step."
---

*Milestone 73 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `collective_curve_2gpu` and `_curve_us` in `methodology.py` / `scheduler.py`, `tools/measure_nccl.py`, `tools/measure_custom_ar.py` · Data: `data/validation/comparison_qwen3_8b_rtx_a6000_tp2_serving.txt`, `engine_steps_m73_qwen3_8b_rtx_a6000.json`.*

Tensor parallelism had only ever been checked offline here, and milestone
68 left it with a named residual: tp=2 decode read 0.90–0.93. The
diagnosis then was the link. NCCL's all-reduce at 64–256 KB, and the
all-gather that assembles the sharded logits, ran slower than the ring
model. This milestone measures the link properly and prices what it
finds, then takes tp=2 online.

## The link, measured densely

`tools/measure_nccl.py --mode graph` times each collective inside a CUDA
graph, the way a serving engine calls it. It now takes 21 sizes from
8 KB to 32 MB. Two runs agree to a median of 1.000:

```
  per rank    all-reduce   ring model    all-gather   ring model
     8 KB        17.2 µs      1.22          15.9 µs      0.82
    64 KB        49.6         0.71          44.8         0.61
   128 KB        69.9         0.74          65.8         0.67
   256 KB       101.4         0.83         100.8         0.76
     1 MB       298.3         0.94         335.7         0.81
    16 MB      4375.8         0.96        5100.3         0.82
```

NCCL does not follow one latency-plus-bandwidth line. It changes protocol
and channel count as messages grow. A decode step's tensor-parallel
all-reduce is 4096 × 2 bytes per sequence, so 8–512 KB, which is exactly
where those switches happen. The all-gather levels off at 3.3 GB/s, where
the ring model had assumed the all-reduce's 4.0.

So a 2-GPU collective on this pair is now priced from the measured curve,
interpolated between points, as milestone 69 did for cuBLAS's row curve.
Any other link, or a larger group, keeps the ring model.

## The kernel that never finishes

vLLM's default at tensor parallel 2 is its own custom all-reduce, not
NCCL, for anything under 8 MB. It enables it on this box even though
the two GPUs meet only through the PCIe host bridge. My first tp=2 run
died after 5 minutes on an RPC timeout, before its first request had a
decode token. The second recorded its steps: a 1024-token prefill took
256 ms (8 MB all-reduces, which go to NCCL), and each decode step took
about 7 seconds. A standalone probe, `tools/measure_custom_ar.py`,
did not finish 8 calls of 8 KB in four minutes.

Milestone 45 had turned the kernel off (`disable_custom_all_reduce` sits
in its data), apparently for the same reason. Every tp=2 run here does
the same, and the model prices NCCL.

## The decode step, on the engine's clock

With NCCL, the tp=2 decode step reads 13.9 ms at batch 1, matching
milestone 45. Forward time on the engine's clock, from 1 to 64 sequences:

```
  batch   forward     ring model   measured curve
     1    13.93 ms      1.03          1.01
     8    17.73         0.94          1.00
    16    20.83         0.93          1.00
    32    26.60         0.92          0.98
    64    38.58         0.92          0.96
```

The curve closes the mid-batch gap. At batch 64, 4% remains, about the
size of the cuBLAS row curve's held-out error on tp=2's per-rank shapes.

## tp=2 online

On M54's shape, tp=2 saturates below a single GPU on this box: about 400
tokens a second against 455–493. Every layer of a prefill step sends a
16 MB all-reduce across the host bridge, and the model has that. The
model's one bias shows up here too. Inside the engine those 16 MB
all-reduces run slightly faster than alone, so prefill-carrying steps
price about 1.5% high. The frozen file said so in advance.

The predictions were pushed before either held-out sweep ran, with the
ring model beside them:

- **H1:** M54's shape on seed 1, at 1–4 req/s.
- **H2:** a decode-heavy trace on seed 2 (prompts 129–384, outputs
  257–758, 1–4 req/s), whose batches keep the all-reduces in the
  mid-size regime.

```
  predicted / measured     TTFT p50   TTFT p95   TPOT    TPOT, ring model
  H1  1 req/s                1.04       1.07     1.02        0.98
  H1  2                      1.01       1.03     1.01        0.91
  H1  3                      1.03       1.05     1.05        0.91
  H1  4                      1.05       1.05     1.02        1.00
  H2  1                      1.09       1.06     0.99        0.91
  H2  2                      1.06       1.10     0.99        0.89
  H2  3                      1.03       0.91     0.99        0.93
  H2  4                      0.94       0.95     0.99        0.94
```

Seven of eight cells meet the criteria. The miss is H1 at 3 req/s, where
TPOT reads 1.054 against the 1.05 bar, in the direction the frozen file
named. Throughput is within 0.99–1.02 throughout. At the composition the
engine recorded, decode steps read 0.97–1.02 and prefill-carrying steps
1.01–1.02. The ring model's TPOT is below 0.95 in six of the eight cells,
and its TTFT p95 falls to 0.61. The GPU process log shows only the pair's
own two workers in every window.

## Errata

- **M68:** the tp=2 residual it named is priced; its three frozen cells
  read 0.94–0.99.
- **M45:** with the measured curve its tp=2 prefill cells read 1.00–1.07
  (they had read 0.97–1.05). At 4–8 MB the link runs about 5% below the
  ring model's 4.0 GB/s.

## Exercises

1. On an NVLink pair, run `tools/measure_custom_ar.py` and give vLLM's
   custom all-reduce its own curve. Where does it beat NCCL?
2. The engine's 16 MB all-reduces run about 1.5% faster than the same
   call alone. Time them in place with the step clock. Is it NCCL's
   channel count, or overlap with the next kernel?
3. Give the ring model NCCL's protocols instead: LL, LL128 and Simple,
   each a latency and a bandwidth, taking the fastest at each size. How
   close do six constants get to the 21-point curve?
