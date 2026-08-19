---
layout: post
math: true
title: "Building tinyperf M10: Split-K — rescuing launch-starved GEMMs"
date: 2026-08-18 20:30:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "When a GEMM has too few output tiles to fill the GPU, split K across CTAs — a second search axis that trades math starvation for cheap partials, validated against real cuBLAS."
---

*Milestone 10 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — split-K search in
`tinyperf/gemm_model.py` · Demo: `examples/08_split_k.py`.*

## The starvation problem

Milestone 2's model launches one CTA per output tile. That's fine when C
is big — but consider `[128, 4096] = [128, K] @ [K, 4096]`: about 32
output tiles for 108 SMs. Two thirds of the GPU sits idle while each CTA
grinds a serial K loop that grows without bound. The longer K gets, the
worse the crime: work scales with K, parallelism doesn't.

Real kernel libraries fix this with **split-K**: divide the K range across
`split_k` CTAs per tile, each computing a partial sum, then run a small
reduction kernel over the fp32 partials. Parallelism multiplies by
`split_k`; the price is `split_k x M x N` partials of traffic, plus one
more kernel launch.

## The model

Split-K is one more axis in the tile search — the search space becomes
tiles x splits, and `min` over all of it models what the library's
heuristic picks:

```python
for bm, bn in TILE_CANDIDATES:
    for split_k in (1, 2, 4, 8, 16):
        k_split = ceil(k_iters / split_k) * BK       # shorter K loop
        ctas   *= split_k                             # more parallelism
        ...same math/dram/l2 model...
        if split_k > 1:                               # the price:
            total += launch + (M*N*(4*split_k + out_bytes)) / dram_bw
```

Because split_k=1 is always searched, enabling split-K can never make a
prediction slower — the test suite pins that superset property, along with
the engagement pattern: chosen for launch-starved long-K shapes, declined
for big squares (plenty of CTAs) and for memory-bound decode (partial
traffic would add cost, not remove it).

## Where it wins — and where it stops winning

`examples/08_split_k.py`, m=128 n=4096 on A100:

```
      k  no-splitK us  splitK us chosen sk  speedup  bound
   2048          14.6       14.6         1     1.00   math
   8192          49.5       49.5         1     1.00   math
  16384          96.0       90.9         8     1.06   dram
  32768         188.9      158.7         8     1.19   dram
  65536         374.8      294.5         8     1.27   dram
```

Note the `bound` column flip: split-K fixes the *math starvation*, and the
reward for fixing it is a new binder — DRAM. That's the recurring shape of
kernel optimization: you don't eliminate bottlenecks, you trade them for
cheaper ones. At k=8192 the model declines to split — wave quantization
makes every split land on the same math time — which is exactly the kind
of cusp a heuristic search handles and a closed-form rule would botch.

## Validated against real cuBLAS

The milestone-9 A6000 measurements double as a validation set. The
calibrated model with split-K lands within **0.90–1.06x** of measured
cuBLAS across every shape with k >= 4096 — including the two shapes the
pre-M9 model missed by 1.5–1.9x:

```
   b     m      n      k   meas us  model us  sk  ratio
   1   128   4096  11008     156.7     156.1   1   1.00
   1  2048   4096  32768    4779.5    5007.0   2   1.05
```

(Re-fitting the calibration after adding split-K left the constants
unchanged — the mechanism, not the constants, was what those shapes
needed. That distinction is the M9 lesson paying rent.)

## In production-scale models

Production GEMM models search split-K jointly with tiles, cluster shapes,
and stage counts, with per-architecture bounds on the split factor. The
modern generalization is **stream-K** — decompose the whole GEMM into a
work list of K-slices and hand SMs a continuous stream, eliminating both
wave quantization and the split-factor choice. And the same idea under a
different name powers attention: flash-*decode* splits the KV dimension
across CTAs exactly like split-K splits K — the milestone-7 exercise you
can now price.

## Exercises

1. Model stream-K: perfect work balance (no wave quantization at all) plus
   one fixup pass — how much of the split-K table's remaining quantization
   loss does it recover?
2. Model the atomic-reduction variant: no second kernel, but partial writes
   become read-modify-writes. Where does it beat the two-kernel form?
3. Apply split-KV to decode FMHA (flash-decode) and measure what it does to
   batch-1 decode latency at 32k context.

*Next: milestone 11 — MoE and expert parallelism, where routing turns one
big GEMM into many small ones and the dispatch becomes the workload.*
