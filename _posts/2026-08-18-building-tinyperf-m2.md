---
layout: post
math: true
title: "Building tinyperf M2: The analytical GEMM model"
date: 2026-08-18 16:41:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Price a matmul without running it: tile-candidate search, tile and wave quantization, and an L2 reuse model that decides what binds."
---

*Milestone 2 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tinyperf/gemm_model.py` · Demo: `examples/02_gemm.py`.*

This is the heart of the simulator. Nearly every expensive op in deep
learning is a GEMM of some shape, so if you can price
`C[M,N] = A[M,K] @ B[K,N]` accurately, you can price a network.

## Why the roofline isn't enough

The roofline needs `flops` and `bytes`. FLOPs are trivial (`2*M*N*K`), but
*bytes are not a property of the problem — they're a property of the
kernel*. A GEMM reads `(M*K + K*N)` unique input bytes, but a kernel with
poor cache reuse fetches many times that from DRAM, and a kernel with small
tiles wastes math on padding. The model must therefore simulate, in
closed form, the *shape of the kernel*: its tiling.

## The mechanism: price every tile, keep the best

A GEMM kernel tiles C into `BM x BN` blocks; each CTA computes one tile.
We enumerate ~12 candidate tile shapes (256x128 down to 16x64) and for each
compute three independent times, taking their max (a perfectly pipelined
kernel hides the smaller two under the largest):

```
time(tile) = max(math, dram, l2) + launch_overhead
best       = min over feasible tiles
```

Taking `min` over tiles models what a tuned kernel library (cuBLAS,
CUTLASS) achieves by autotuning. Production models do exactly this with
per-architecture curated tile databases and a far richer config space
(split-K, cluster/CGA shapes, sparsity, mixed input precisions).

### Math time: two quantization effects

```python
ctas  = ceil(M/BM) * ceil(N/BN) * batch
waves = ceil(ctas / sm_count)
math  = waves * (BM * BN * K_padded) / (macs_per_sm_clk * eff * clock)
```

- **Tile quantization**: a 8200x8192 GEMM pads M to 8448 with 256-wide
  tiles. The demo shows it: 94.8% efficiency vs 99.7% for the aligned case.
- **Wave quantization**: CTAs launch in waves of `sm_count`; 109 CTAs on
  108 SMs take two full waves. The `ceil` *is* the model.
- **Small-tile efficiency** (`eff`): a CTA computing a 16x64 tile can't keep
  an SM's tensor cores fed; we scale throughput by `min(1, BM*BN/4096)`.

### DRAM time: modeling L2 reuse

The subtle part. Each CTA reads a `BM x K` slice of A and `K x BN` slice of
B; naively that's `ctas * (BM+BN) * K` bytes — for 8192^3 with 128x128
tiles, 17 GB against a 134 MB ideal. Reality sits between, thanks to L2.

Real kernels rasterize CTAs in *swizzled, roughly square super-tiles* so one
wave of concurrent CTAs touches a compact set of A rows and B columns. We
model exactly that:

```python
rows, cols  = ~sqrt(ctas_per_wave)            # square super-tile
wave_ws     = (rows*BM + cols*BN) * K * bytes  # unique bytes per wave
wave_dram   = wave_ws * max(1, wave_ws / l2_capacity)   # graceful thrashing
dram_bytes  = clamp(waves * wave_dram, ideal, no_reuse_bound)
```

If the per-wave working set fits in L2, DRAM sees each byte once per wave;
past capacity, reuse degrades proportionally. This one formula is the
difference between predicting 8192^3 fp16 on A100 as memory-bound at 6.4 ms
(wrong) and math-bound at 3.54 ms (right — cuBLAS achieves ~3.9 ms).

### L2 time

SM-side bandwidth also binds: every CTA pulls its tiles from L2, so
`l2_time = ctas * (BM+BN) * K * bytes / l2_bw`. The batched-attention case in
the demo (32x [2048 x 2048 x 128]) is the rare shape that is L2-bound — K is
too small to amortize tile traffic.

## Results across regimes

From `examples/02_gemm.py` on A100 fp16 (peak 312 TFLOPS):

```
case                          m     n     k   b   time_us  TFLOPS   eff%  tile     bound
square (training-like)     8192  8192  8192   1    3535.4   311.0   99.7  256x128  math
tile quantization          8200  8192  8192   1    3721.4   295.7   94.8  256x128  math
skinny (decode, b=1)          1  4096  4096   1      20.4     1.6    0.5  16x256   dram
decode, b=64                 64  4096  4096   1      21.1   102.0   32.7  64x64    dram
attention (batched)        2048  2048   128  32     152.1   225.9   72.4  128x256  l2
tiny (launch-bound)         128   128   128   1       3.4     1.2    0.4  64x64    math
```

Every row teaches something: the +8 rows of M cost 5% (quantization); the
b=1 and b=64 decode GEMMs take the *same time* (both stream the same weight
bytes — this is why batching inference is nearly free); the tiny GEMM is
dominated by the 3 µs launch overhead.

The skinny case lands at 20.4 µs vs a 16.4 µs pure weight-streaming bound —
about right for a kernel *without* split-K. We don't model split-K
(parallelizing over K when there are too few CTAs); it's exercise 1.

## In production-scale models

Production GEMM models implement this mechanism with: per-arch tile
databases, split-K search, cluster quantization on newer architectures, an
L1 model, a prefetch model, a launch/prolog latency model per architecture,
sparsity and weight-compression traffic, and calibration overlays that
correct residual error against silicon. And for kernels where closed form
fails (FMHA), they fall back to event-driven kernel simulation. The
skeleton, however, is the ~100 lines you just read.

## Exercises

1. Add split-K: when `ctas < sm_count`, split K across `sm_count // ctas`
   CTAs and charge a `split_k * M * N` fp32 reduction pass.
2. Plot achieved TFLOPS vs M for N=K=8192, M in 1..1024. Explain the staircase.
3. Calibrate: benchmark cuBLAS on any GPU you have, and fit `l2_bw_gbps`
   and `kernel_launch_us` to minimize error. (This is what a "calibrated"
   methodology means in production tools.)

*Next up, milestone 3: a graph IR — describing whole networks as shapes, so the GEMM model has something to price.*
