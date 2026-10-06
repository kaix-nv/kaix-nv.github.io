---
layout: post
title: "Building tinyperf, chapter 3: Pricing a GEMM"
date: 2026-09-29 12:03:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/03-pricing-a-gemm/
excerpt: "Nearly every expensive operation in a transformer is a matrix multiply: the attention projections, the feed-forward layers, the experts of a mixture-of-experts model and the output head. If you can say how long C[M,N] = A[M,K] · B[K,N] takes on a given GPU, for any shape, you can price most of a network. Chapter 5 turns a model into a list of such operations and adds them up. This chapter builds the piece that prices each one."
redirect_from:
  - /tinyperf/perf-modeling/2026/08/18/building-tinyperf-m2.html
  - /tinyperf/perf-modeling/2026/08/18/building-tinyperf-m9.html
  - /tinyperf/perf-modeling/2026/08/18/building-tinyperf-m10.html
  - /tinyperf/perf-modeling/2026/08/29/building-tinyperf-m23.html
  - /tinyperf/perf-modeling/2026/09/24/building-tinyperf-m69.html
---

*[Building tinyperf](/series/tinyperf/) · Part I: Pricing one kernel · Code: [`tinyperf/gemm_model.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/gemm_model.py), `estimate_gemm` · Every table and the plot in this chapter come from `python3 book/scripts/ch03_gemm.py`.*

Nearly every expensive operation in a transformer is a matrix multiply:
the attention projections, the feed-forward layers, the experts of a
mixture-of-experts model and the output head. If you can say how long
`C[M,N] = A[M,K] · B[K,N]` takes on a given GPU, for any shape, you can
price most of a network. Chapter 5 turns a model into a list of such
operations and adds them up. This chapter builds the piece that prices
each one.

By the end of it you will know:

- why the bytes a GEMM moves depend on the kernel, not just on the
  matrices;
- the effects the model prices: work padded to whole tiles (the output
  blocks a kernel computes) and to whole rounds of tiles across the
  GPU, reuse through the L2 cache, a kernel's start-up before it runs at
  rate, and a fixed cost per kernel;
- how split-K, which divides the inner dimension of the multiply among
  several thread blocks, rescues a GEMM that has too few tiles to fill
  the GPU;
- how close the result gets to cuBLAS on three GPUs, and where it still
  reads as low as 0.55 of the measured time.

## The roofline is not enough

Chapter 2 described a GPU by a few rates and gave the simplest possible
price for any operation, the roofline:

```
time = max(flops / peak_flops, bytes / memory_bandwidth)
```

For a GEMM the FLOPs are `2·M·N·K`, and the bytes, at their minimum,
are reading A and B once and writing C once. To see how far that gets
us, take one GEMM from a transformer layer, a 4096 × 4096 weight, and
time it on an RTX A6000 at the row counts a server feeds it: one row per
sequence in a decode batch, one row per token in a prefill. We give the
roofline every advantage: this GPU's best measured math and memory rates
instead of its datasheet peaks, plus the 3.5 µs a kernel costs to launch.
(Chapter 4 shows how those rates are measured.)

```
Table 3.1  A 4096 x 4096 weight GEMM by rows, RTX A6000: the roofline at its best
   rows  measured us  roofline us  roofline/measured
      1         52.0         52.1               1.00
     64         57.4         53.6               0.93
    128         56.4         55.1               0.98
    129         97.5         55.1               0.56
    192         89.9         59.0               0.66
    256        132.5         77.5               0.58
    257        173.9         77.8               0.45
    320        180.8         96.0               0.53
    512        177.5        151.4               0.85
    768        270.6        225.4               0.83
   1024        338.7        299.4               0.88
   1280        371.2        373.4               1.01
```

Throughout the book a ratio like the last column is **predicted ÷
measured**. Below 1 the model reads low, too optimistic; above 1 it
reads high, too pessimistic.

At both ends the roofline is right. Up to 128 rows the GEMM's time is
streaming its 33.5 MB of weights, and the roofline knows that. At 1,280
rows it is firmly limited by math, and the roofline knows that too. In
between it reads as low as 0.45. One extra row, from 128 to 129, takes
the measured time from 56 to 98 µs while the roofline doesn't move.

The roofline assumes that every byte of A and B crosses the memory bus
exactly once, and that every tensor core is busy on every cycle. Between
the two regimes neither is true, and how far from true depends on how
the kernel divides the work. The FLOPs are a property of the problem.
The bytes actually moved and the cycles actually lost are properties of
the kernel that runs it. To price the middle, we have to model the
kernel. This chapter builds that model, then comes back to this GEMM:
the model closes about a third of the gap, and the rest says something
about kernel libraries.

## How a GEMM kernel works

A GPU is a set of streaming multiprocessors (SMs): 108 on an A100, 84 on
an RTX A6000. A GEMM kernel splits the output C into tiles of `BM × BN`
elements. Each tile is computed by one thread block, which the CUDA
literature calls a CTA (cooperative thread array), running on one SM.

![A GEMM divided into tiles. One output tile of C needs a panel of rows
of A and a panel of columns of B; the kernel walks the K dimension in
slices of BK.](/assets/tinyperf-book/ch03-tiling.svg)

*Figure 3.1. One CTA computes one `BM × BN` tile of C. It needs the `BM`
rows of A and the `BN` columns of B that cross its tile, each `K` long.
It walks K in steps of `BK`: at each step it loads a `BM × BK` slice of
A and a `BK × BN` slice of B into the SM's shared memory, multiplies
them on the tensor cores, and adds the result into its tile. While it
computes on one slice it loads the next, so loading and computing
overlap.*

A kernel library such as cuBLAS or CUTLASS ships many kernels, with
different tile shapes, and picks one for each problem. Big tiles reuse
each loaded byte more; small tiles make more CTAs, which keeps more SMs
busy. The best choice depends on the shape and the GPU.

That gives the model its structure. We enumerate a menu of candidate
tiles, price each one, and keep the fastest:

```
time(tile) = max(math, dram, l2) + launch
best       = min over tiles that fit in shared memory
```

The `max` models the overlap in Figure 3.1. A pipelined kernel loads and
computes at the same time, so the slowest of the three streams sets the
pace: tensor-core math, traffic from DRAM (the GPU's main memory,
high-bandwidth HBM or graphics GDDR), and traffic from the L2 cache into
the SMs. The `min` models the library's choice, assuming it picks well.
That assumption holds most of the time; the end of the chapter shows
where it doesn't.

The menu has twelve tiles, from 256×128 down to 16×64, with `BK = 64`.
A tile is feasible only if its two slices, double-buffered, fit in one
SM's shared memory: `(BM + BN) · BK · bytes · 2` must not exceed it.

The next three sections price the terms: math, then DRAM, then L2 and
launch.

## Math time: the work the tensor cores actually do

Each CTA does `BM · BN · K` multiply-accumulates (MACs). An SM does a
fixed number per clock: 1,024 fp16 MACs per clock on an A100. The CTAs
don't all run at once, though. The model assumes one CTA per SM at a
time (a real SM can hold several small ones), so the GPU runs them in
**waves**, and the math time is:

```
ctas  = ceil(M / BM) · ceil(N / BN) · batch
waves = ceil(ctas / sm_count)
math  = waves · (BM · BN · K) / (macs_per_sm_per_clock · efficiency · clock)
```

Three effects hide in that formula.

**Tile quantization.** A tile computes all `BM × BN` outputs even when
the matrix edge cuts through it. The padding is wasted work. On the A100
the model runs an 8,200-row GEMM on 256-row tiles, which pads it to
8,448 rows: 3% extra math for 8 extra rows.

**Wave quantization.** A wave is as slow as its busiest SM, whether the
other SMs have work or not (Figure 3.2).

![Eighteen tiles on eight SMs run in three waves, the last one with only
two tiles.](/assets/tinyperf-book/ch03-waves.svg)

*Figure 3.2. Wave quantization. The third wave runs two tiles and leaves
six SMs idle, but it takes as long as a full wave. The `ceil` in the
formula is the whole effect.*

Here it is in the model, on an A100 with M = 4,096 and K = 8,192. The
first two lines allow only 256×128 tiles; the last two let the model
choose:

```
256x128 tiles, N=3456: 432 tiles on 108 SMs = 4 waves, 752.5 us, math-bound
256x128 tiles, N=3584: 448 tiles on 108 SMs = 5 waves, 939.9 us, math-bound
best tile, N=3456: 256x128, 4 waves, 752.5 us
best tile, N=3584: 128x128, 9 waves, 846.2 us
```

432 tiles fill exactly four waves. 448 need a fifth, which runs 16 tiles
on 108 SMs, so 3.7% more work costs 25% more time. Given the whole menu,
the model switches to 128×128 tiles at N = 3,584, 896 of them in nine
waves, and pays 12% instead. A library does the same kind of thing: a
different tile can soften a bad wave count, but not always avoid it.

**Efficiency: small tiles and short loops.** Two things keep a CTA below
its SM's peak rate, and the model multiplies them:

```
tile_eff = min(1, BM · BN / 4096)
pipe_eff = iters / (iters + stages - 1),   iters = K / BK
```

A tile smaller than 64×64 outputs doesn't keep enough matrix
instructions in flight to feed the tensor cores. And the load-compute
pipeline needs `stages - 1` iterations to fill before it runs at rate,
where `stages` is the number of slices the kernel buffers in shared
memory: 2 here, the double buffering of Figure 3.1. With two stages, a
K loop of 64 iterations (K = 4,096) runs at 64/65 of its rate. A loop of
two iterations runs at 2/3. That second case is common: attention's
score matrix, `Q · Kᵀ`, contracts over the head dimension, typically
128, which is just two steps of 64.

## Memory time: where L2 reuse comes from

How many bytes does a GEMM pull from DRAM? There are two extremes. At
best, each element of A and B is read once and C is written once. At
worst, every CTA fetches its own panels of A and B (the strips of rows
and columns that cross its tile, Figure 3.1) from DRAM with no sharing
at all. For an 8192³ fp16 GEMM:

```
unique bytes (the best case):                       0.40 GB
every CTA fetches its own panels (256x128 tiles):  13.0 GB
```

They are 32 times apart. Which end a kernel is near depends on the L2
cache, and on which tiles run at the same time.

![One wave of twelve tiles, arranged as a 3 by 4 block, shares three
panels of A and four panels of B.](/assets/tinyperf-book/ch03-l2-wave.svg)

*Figure 3.3. Tiles that run in the same wave can share their inputs
through L2. Libraries launch CTAs in compact, roughly square blocks for
this reason. Twelve tiles in a 3 × 4 block need 7 panels of A and B;
twelve tiles in a single row would need 13. Because the kernel walks K
in steps, the L2 only has to hold one `BK`-wide slice of those panels at
a time.*

The model follows Figure 3.3. It arranges one wave's tiles as a block of
`rows × cols` tiles, as close to square as the grid allows, and computes
the width of A and B that block reads:

```
window  = (rows · BM + cols · BN) · bytes         # bytes of A and B per unit of K
slice   = problems_per_wave · window · BK         # what L2 must hold at once
visits  = ctas / (rows · cols)                    # how many such blocks the GEMM runs
dram    = visits · window · K · max(1, slice / L2_capacity) + C_bytes
dram    = clamp(dram, unique_bytes, no_reuse_bytes)
```

`problems_per_wave` is 1 for a single GEMM; the batched case is below.

If the slice fits in L2, each block reads its panels from DRAM once. If
it doesn't, reuse degrades in proportion. Each block streams all of K,
so by the time a later wave needs a panel an earlier wave used, it has
long been evicted. The model charges every block its own panels: the
reuse is within a wave, not between waves.

Worked through for 8192³ fp16 on the RTX A6000, whose L2 is 6 MB (the
model counts 80% of it as usable, since other traffic shares it):

```
tile 256x128, 84 tiles per wave arranged about 9 x 10
one K-slice of the wave's A rows and B columns: 0.46 MB (usable L2 4.8 MB)
DRAM bytes: unique 0.40 GB, model 1.48 GB, no reuse at all 13.0 GB
model: math 7.34 ms, DRAM 1.93 ms -> math-bound, 7.34 ms
```

The slice fits easily, so this GEMM reads about 3.7 times its unique
bytes. That is 1.9 ms of DRAM time, well under its 7.3 ms of math, so
the GEMM is math-bound.

One refinement matters for batched GEMMs, like attention computed for
many heads at once. When the batch supplies the parallelism, one wave
holds tiles from several independent problems. Each has its own A and B,
so the L2 has to hold a slice for every one of them.

> **Field note: the wrong working set.** An earlier version of this
> model asked whether the wave's panels fit in L2 over the *whole* K
> range. For this GEMM that is 58.7 MB, 12 times the usable L2, so it
> predicted heavy thrashing: 13 GB of DRAM traffic and 17 ms, where
> cuBLAS takes 10.3 ms. On an A100, whose L2 is 40 MB, the mistake
> rarely showed. It took a GPU with a small L2 to expose it. A kernel
> streams K a slice at a time, so a slice is what has to stay resident.
> The corrected model predicts 7.3 ms, 0.71 of the measurement; that gap
> has another cause, the clocks this GPU sustains, which chapter 4 fits.
> But the model no longer invents 13 GB of traffic. The general lesson:
> test your model on hardware unlike the hardware you built it on, or its
> errors will hide where you never looked.

## L2 time and launch cost

Two terms remain.

**L2 time.** Every CTA pulls its own panels from L2 into shared memory,
whether or not DRAM supplied them to L2 only once. That traffic is
`ctas · (BM + BN) · K · bytes`, plus the output, at the L2's bandwidth.
It rarely binds, but it can when a GEMM has many tiles and K is too
short to amortize loading them.

**Launch cost.** Every kernel pays a fixed cost to launch and to run its
prologue and epilogue. On the GPU itself that is a few microseconds; the
model's default is 3 µs. Measured from the host, it depends on how the
kernels are issued: about 25 µs per call when PyTorch issues them one at
a time on the A6000, and 3.5 µs when a CUDA graph, a sequence of
launches recorded once, replays them. Chapter 4 fits this constant for
each case. For a tiny GEMM it is the whole price.

## The code

Here is the core of `estimate_gemm`, lightly trimmed. The trimmed parts
are the sustained-clock factor, the byte scaling for quantized and
sparse weights (chapter 10), per-kernel-class scale factors (chapter 9)
and the bookkeeping of the result.

```python
def estimate_gemm(device, m, n, k, dtype, batch=1, out_dtype=None, enable_split_k=True):
    in_b, out_b = dtype.nbytes, (out_dtype or dtype).nbytes
    macs_per_sm_clk = device.tensor_macs_per_sm_clk[dtype.key]
    clk_hz = device.boost_clock_ghz * 1e9
    k_iters = math.ceil(k / BK)
    l2_cap = L2_USABLE_FRACTION * device.l2_size_mb * 1e6

    best = None
    for bm, bn in TILE_CANDIDATES:
        if (bm + bn) * BK * in_b * SMEM_STAGES > device.smem_kb_per_sm * 1024:
            continue                                        # doesn't fit in shared memory
        ctas_m, ctas_n = math.ceil(m / bm), math.ceil(n / bn)
        tile_eff = min(1.0, (bm * bn) / FULL_EFFICIENCY_TILE_ELEMS)

        for split_k in SPLIT_K_CANDIDATES:                  # 1, 2, 4, 8, 16: see below
            if split_k > k_iters or (split_k > 1 and not enable_split_k):
                break
            k_split = math.ceil(k_iters / split_k) * BK
            part_b = 4 if split_k > 1 else out_b            # split partials are fp32
            ctas = ctas_m * ctas_n * batch * split_k
            waves = math.ceil(ctas / device.sm_count)

            # math: one CTA's K loop, once per wave
            split_iters = max(1, math.ceil(k_split / BK))
            pipe_eff = split_iters / (split_iters + SMEM_STAGES - 1)
            cta_math_s = (bm * bn * k_split) / (macs_per_sm_clk * tile_eff * pipe_eff * clk_hz)
            math_s = waves * cta_math_s

            # L2 -> SMs: every CTA pulls its own panels
            out_bytes = batch * m * n * part_b * split_k
            percta_bytes = ctas * (bm + bn) * k_split * in_b + out_bytes
            l2_s = device.l2_time_s(percta_bytes)

            # DRAM: one wave's tiles, as a near-square block, share panels through L2
            ctas_per_wave = min(ctas, device.sm_count)
            tiles_per_slice = ctas_m * ctas_n
            slices_pw = max(1, math.ceil(ctas_per_wave / tiles_per_slice))
            within = min(ctas_per_wave, tiles_per_slice)
            rows = min(ctas_m, max(1, round(math.sqrt(within))))
            cols = min(ctas_n, math.ceil(within / rows))
            window = (rows * bm + cols * bn) * in_b
            slice_ws = slices_pw * window * BK
            visits = math.ceil(ctas / (rows * cols))
            ideal_bytes = batch * (m * k + k * n) * in_b + out_bytes
            dram_bytes = visits * window * k_split * max(1.0, slice_ws / l2_cap) + out_bytes
            dram_bytes = min(max(dram_bytes, ideal_bytes), percta_bytes)
            dram_s = device.dram_time_s(dram_bytes)

            total_us = max(math_s, dram_s, l2_s) * 1e6 + device.kernel_launch_us
            if split_k > 1:                                 # a separate reduction kernel
                red_bytes = batch * m * n * (4 * split_k + out_b)
                total_us += device.dram_time_s(red_bytes) * 1e6 + device.kernel_launch_us
            if best is None or total_us < best.time_us:
                best = ...                                  # keep tile, split, times, bound
    return best
```

This loop is the heart of everything that follows. `slices_pw` is the
batched refinement above: how many independent problems share one wave.

## One model, six regimes

Table 3.2 runs the model on an A100 across the shapes you meet in
practice. The model has no measured inputs here, only datasheet rates.
Its bound column names the term that set each time: math, DRAM or L2.

```
Table 3.2  One model, six regimes: A100 SXM, fp16, datasheet rates
  case                           batch x M x N x K       us  % of peak      tile  waves  bound
  square                    1 x 8192 x 8192 x 8192   3563.0       98.9   256x128     19  math
  8 rows past a tile edge   1 x 8200 x 8192 x 8192   3750.4       94.1   256x128     20  math
  decode, 1 row                1 x 1 x 4096 x 4096     19.5        0.6    16x256      1  dram
  decode, 64 rows             1 x 64 x 4096 x 4096     20.0       34.5     64x64      1  dram
  attention scores          32 x 2048 x 2048 x 128    168.6       65.4   256x128     38  math
  tiny                         1 x 128 x 128 x 128      3.5        0.4     64x64      1  math
```

Each row shows a different mechanism.

- **Square.** A large GEMM reaches 98.9% of peak. Its 2,048 tiles fill
  19 waves almost exactly (18.96), and its K loop is long.
- **8 rows past a tile edge.** The padding adds a 33rd row of tiles, 64
  more CTAs, which pushes the GEMM into a 20th wave. Eight rows cost
  5.3% more time.
- **Decode, 1 row.** One token through a 4096 × 4096 layer does almost
  no math. Its time is streaming the 33.5 MB of weights from DRAM
  (16.5 µs at 2,039 GB/s) plus the launch. That is why decoding a single
  sequence uses under 1% of the GPU's math.
- **Decode, 64 rows.** 64 times the work costs 3% more time, because
  the weights are still read once. This is why serving engines batch
  decodes: until the math catches up with the weight traffic, extra
  sequences are nearly free. Chapter 6 finds where it catches up.
- **Attention scores.** K is 128, a two-step loop, so pipeline fill
  alone caps each CTA at 2/3 of its rate.
- **Tiny.** 3.5 µs, of which 3 µs is the launch.

## Split-K: when there aren't enough tiles

Take `M = 128, N = 4096` with a long K. On 128×128 tiles that is 32
tiles for 108 SMs: most of the GPU idles while 32 CTAs grind through a
long K loop. Adding K adds work but no parallelism.

Libraries fix this with **split-K**. They divide the K range among `s`
CTAs per tile. Each computes a partial sum in fp32, and a second, small
kernel adds the partials. Parallelism multiplies by `s`. The price is
writing and reading `s` fp32 partials of C, plus a second launch.

In the model, split-K is one more axis of the search. The loop in the
code above tries `s = 1, 2, 4, 8, 16` for every tile. Because `s = 1`
is always a candidate, allowing split-K can never make a prediction
slower.

```
Table 3.3  Split-K on a tile-starved shape: A100, M=128, N=4096
       K  no split us   best us  split  speedup  bound
    2048         15.0      15.0      1     1.00  math  (math 12.0, DRAM 9.0)
    8192         49.8      49.8      1     1.00  math  (math 46.8, DRAM 34.5)
   16384         96.3      90.9      8     1.06  dram  (math 71.9, DRAM 76.1)
   32768        189.3     158.7      8     1.19  dram  (math 141.6, DRAM 144.0)
   65536        375.2     295.8      8     1.27  math  (math 281.1, DRAM 279.8)
```

The longer the K loop, the more the split pays for its partials. Once
split, math and DRAM time are nearly equal, and the bound column flips
between them on a few percent. Most of that DRAM traffic is streaming
the weights, 78–91% of the bytes; the partials add the rest. Removing
one bottleneck usually exposes the next.

None of the cuBLAS shapes we measured is starved enough for split-K to
matter much, so this part of the model has no silicon evidence behind
it yet. Exercise 2 asks you to supply some.

## How close is it?

To compare against measurements, the model needs a few constants fitted
to each GPU: how close to peak its tensor cores and memory actually
run, its L2 bandwidth, and its per-kernel overhead. Chapter 4 is about
fitting them. As in Table 3.1, we give the roofline and the tile model
the same fitted constants, so the only difference between them is the
mechanism this chapter built.

We summarize a set of ratios by their **typical error**: the geometric
mean of how far each ratio is from 1, in either direction
(`exp(mean |ln ratio|) − 1`).

```
Table 3.4  Typical error against measured cuBLAS, both models on the same fitted constants
  GPU        shapes  roofline + launch  tile model
  rtx_a6000      27               5.2%        4.9%
  h100_sxm       27               8.8%        8.5%
  b200_sxm       27               9.4%        9.1%
```

That is a slightly deflating result. On 27 standard shapes per GPU, the
tile model barely beats a roofline with a launch cost. The reason is
where those shapes sit. Big squares are firmly math-bound and
decode shapes firmly memory-bound. Deep inside a regime, the roofline's
`max` already picks the right term, and the fitted constants absorb the
rest.

The mechanism earns its place between the regimes. Figure 3.4 and Table
3.5 use a different measurement: the four weight GEMMs of one
transformer layer (Qwen3-8B), timed at 39 row counts from 1 to 1,280,
the way a serving engine calls them, inside a CUDA graph.

![Time of a 4096 by 4096 weight GEMM on the RTX A6000 from 1 to 1280
rows: measured, tile model and roofline.](/assets/tinyperf-book/ch03-row-curve.svg)

*Figure 3.4. A 4096 × 4096 weight GEMM by rows, the same GEMM as Table
3.1. Up to 128 rows all three agree: the time is streaming the weights.
Beyond that the roofline rises in a straight line, the tile model
follows the measured trend more closely, and the measurement jumps just
past 128, 256, 512 and 1,024 rows.*

```
Table 3.5  Typical error by row range: four GEMMs, 39 row counts, RTX A6000 in a CUDA graph
  rows           roofline + launch  tile model  worst tile/measured
  1-128 rows                  7.2%        6.0%                 0.87
  129-512 rows               52.4%       33.1%                 0.55
  513-1280 rows              18.0%       10.8%                 0.78
  all                        26.1%       17.2%                 0.55
```

Between 129 and 512 rows the measurement is typically 1.5 times the
roofline's price, and the tile model cuts that error by about a third.
That middle is where serving often lives: a decode batch of more than
128 sequences, or a prefill chunk (a slice of a long prompt, run in one
forward pass) of a few hundred tokens. It is also where the tile model
is still worst.

## Where it breaks

Even the tile model reads 0.55 at its worst. Figure 3.4 shows where:
the measured time jumps at 129, 257, 513 and 1,025 rows, one row past a
multiple of 128. The first jump is the largest:

```
qkv_proj     128 rows    97.7 us   129 rows   124.4 us   x1.27
attn_out     128 rows    56.4 us   129 rows    97.5 us   x1.73
ffn_gate_up  128 rows   370.7 us   129 rows   413.2 us   x1.11
ffn_down     128 rows   175.0 us   129 rows   301.9 us   x1.73
```

For the two GEMMs with 4,096 output columns, one extra row costs 73%
more time. That is close to what reading the weight matrix twice would
cost. `attn_out`'s weights are 33.5 MB, 48.5 µs of streaming at the
fitted bandwidth of 691 GB/s; two passes plus the launch come to about
100 µs, and 129 rows take 97.5 µs. The model prices one pass: at 129
rows it picks 32×128 tiles, and all five of their row blocks run in the
same wave and share each weight column through L2. This explanation
fits the numbers; we have not profiled which kernel cuBLAS picks at 129
rows. For wider GEMMs the jump is smaller (×1.27 at 6,144 columns, ×1.11
at 24,576), where more column tiles give the library other options.

The general point is about the `min`. The model assumes the library
picks the best tile from a generous menu and launches its tiles in the
ideal order. A real library has a finite menu and a heuristic, and its
choices leave edges like these. They can't be derived from first
principles; they have to be measured. The calibrated tier, the model
run on constants fitted to one GPU (chapter 4), stores a measured
correction curve for exactly this, and chapter 15 uses it to price
decode steps.

Other things this model leaves out:

- **Newer kernel designs.** Stream-K and persistent kernels balance work
  across SMs and remove most wave quantization. Hopper- and
  Blackwell-class features (tensor memory accelerators, warp
  specialization, thread-block clusters) change how tiles load and share
  data.
- **Epilogue fusion.** Libraries often fuse a bias or activation into
  the GEMM's epilogue. The model prices those as separate operations.
- **Clocks.** A long GEMM can run the GPU into its power limit and below
  its boost clock. The A6000 does at 8192³. The model has no power
  model here; the fitted constants in chapter 4 absorb the effect.

## What you built

- A GEMM priced as `max(math, dram, l2) + launch`, searched over tiles
  and split factors, the way a library autotunes.
- Math time with tile and wave quantization, small-tile efficiency and
  pipeline fill.
- DRAM traffic from L2 reuse among the tiles of one wave, judged one K
  slice at a time.
- Evidence: 5–9% typical error on standard shapes on three GPUs, and
  17% across a layer's GEMMs from 1 to 1,280 rows, where a roofline
  reads 26%.
- A known gap: the library's tile edges, which have to be measured.

## Exercises

1. Plot achieved TFLOPS against M for N = K = 8192 on the A100, M from 1
   to 1,024. Explain each step in the curve with tile and wave counts.
2. Test split-K on silicon. Find a shape where the model chooses a split
   of 4 or more with a speedup above 1.2, time it with `torch.matmul` on
   a GPU you have, and compare it with the model with split-K disabled.
3. Reproduce the 129-row edge. Restrict `TILE_CANDIDATES` to 128-row
   tiles for M above 128, and count DRAM traffic as if different row
   blocks did not share weights. Does the model now jump at 129 rows?
   By how much?
4. Model stream-K: perfect balance across SMs (no wave quantization)
   plus one fix-up pass for the partial tiles. How much of Table 3.2's
   second row does it recover?

---

*[← Chapter 2: A GPU as a handful of rates]({% post_url 2026-09-29-tinyperf-02-a-gpu-as-a-handful-of-rates %}) · [Contents](/series/tinyperf/) · [Chapter 4: Meeting a real GPU: calibration and its tiers →]({% post_url 2026-09-29-tinyperf-04-meeting-a-real-gpu %})*
