---
layout: post
math: true
title: "Building tinyperf M63: The decode kernel's efficiency is a curve in rows per expert: a batch sweep, the router as a table, and a kernel that is slower in company"
date: 2026-09-20 11:30:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "A batch sweep against predictions frozen from milestone 62's two constants: the bf16 kernel got slower per byte at batch 64, the Marlin kernel faster. The router becomes a measured table, the kernels get efficiency curves in rows per expert derived from step times, and two findings on the way: the profiler reads these kernels long, and the engine's own expert layer runs 1.5× faster alone than inside its own step."
---

*Milestone 63 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — efficiency curves in `methodology.py`, `routing_distinct` in `nets/transformer.py` · Data: `data/validation/comparison_gpt_oss_20b_rtx_a6000_batchsweep.txt`, `kernel_profile_gpt_oss_20b_rtx_a6000_decode.json`.*

Milestone 62 left a number on the table. The fused-MoE decode kernel
streamed expert weights at 0.70 of the calibrated DRAM rate once an
expert had more than one row, 0.91 with one — and on the two-GPU rank
that held sixteen sequences, 0.50. Three points, called a constant with
an exception. This milestone measures the shape: the batches the
validation grid had always skipped, both checkpoints, predictions frozen
first from the constant, so that the curve has to show itself against a
committed number.

## The sweep

Batch 2, 4, 16 and 64 at prompts 512 and 2048, one RTX A6000, the same
engine, the same protocol as milestone 32. The frozen model said the
constant would hold: batch 64 (about thirteen rows per touched expert)
would land where it landed at batch 8. Decode against silicon:

```
   bf16 (Triton fused MoE)          MXFP4 (Marlin MoE)
   b  prompt  meas   frozen    r    meas   frozen    r     rows/expert
   2    512   17.11   17.26  1.01    9.07    9.30  1.03       1.4
   2   2048   16.46   17.37  1.06    8.99    9.41  1.05       1.4
   4    512   22.58   22.31  0.99   11.07   11.16  1.01       1.9
   4   2048   23.10   22.53  0.98   10.98   11.38  1.04       1.9
  16    512   33.77   37.70  1.12   15.15   17.01  1.12       5.1
  16   2048   38.26   38.58  1.01   16.85   17.88  1.06       5.1
  64    512   68.31   59.72  0.87   21.64   26.33  1.22      13.7
  64   2048   69.17   63.21  0.91   25.10   29.83  1.19      13.7
```

Two kernels, two opposite failures at batch 64. The bf16 Triton kernel
got *slower* per byte than the constant said: 0.87–0.91. The Marlin
kernel got *faster*: 1.19–1.22. Prefill was unremarkable, 0.89–1.06
everywhere except the tiny 2×512 cells (0.69 / 0.79, the milestone-53
miss). So the constant is wrong in a direction that depends on the
kernel, and it is wrong exactly where rows per expert leave the range
the constant was read from.

## The router, as a table

Milestone 62's Zipf exponent of 2.15 reproduced the router's
distinct-expert counts up to batch 32. Extending the histogram to batch
64 and 128 (shorter sequences to fit memory) gave 18.7 and 20.6 of 32,
where the Zipf curve says 21.2 and 25.7. The router saturates lower
than any Zipf law — per layer, two or three experts are always on and
the rest fill in slowly. Rather than fit a third parameter, the preset
now carries the measured table:

```
  batch     1     2     4     8    16    32    64   128
  experts   4.0   5.5   8.4   9.7  12.5  15.9  18.7  20.6   of 32
```

interpolated in log2 inside its range, handing over to the Zipf rule
beyond it (and to all 32 experts at prefill sizes, where the histogram
also says 29–31). A model that has been measured gets its table; one that
has not keeps the Zipf rule with skew 0 — uniform, the conservative
default.

## What the kernel costs per row per expert

With the router's counts known at every batch, each measured step gives
one efficiency number: take the step time, subtract the model's
non-expert part (attention, dense GEMMs, head, glue — all within a few
percent of the engine's own kernel profile), and divide the touched
experts' weight bytes by what remains. Sixteen cells per checkpoint:

```
  rows/expert     1.0    1.4    1.9    3.3    5.1    8.0   13.7
  bf16 (512)      1.00   0.77   0.81   0.65   0.75   0.74   0.53
  bf16 (2048)     0.99   0.82   0.80   0.67   0.67   0.72   0.55
  MXFP4 (512)     0.77   0.58   0.61   0.57   0.57   0.59   0.58
  MXFP4 (2048)    0.73   0.61   0.64   0.52   0.53   0.57   0.58
```

The bf16 kernel falls in steps: full rate with one row per expert,
about 0.8 at one to two rows, about 0.7 from three to eight, and 0.53 at
fourteen — the point where the Triton default configuration switches to
64-row tiles that are mostly padding, and the padded tile's math stops
hiding under the weight stream. The Marlin kernel drops once, from 0.77
to about 0.58, and stays there; it was built for exactly this regime.
The calibration file now holds each as a curve — points in rows per
expert, interpolated in log2 — fitted on the prompt-512 column and
checked on the prompt-2048 and 8192 cells it never saw:

```
                        fit (prompt 512)    held out (2048 / 8192)   two GPUs (held out)
  bf16   decode         1.02  (0.96–1.06)    1.02  (0.93–1.11)       0.93–0.96 / 0.84–0.92 / 0.71–0.73
  MXFP4  decode         1.04  (1.00–1.07)    1.02  (0.95–1.09)       —
```

Both single-GPU grids now sit within about 10% at every batch from 1 to
64. One segment of each curve is declared rather than measured: between
25 and 256 rows per expert nothing was stepped, and the curves return to
1.0 by 128 rows because the prefill cells at 256 rows and up are priced
by the math-efficiency constants alone and match. The batch-1×512 prefill
cell moved from 0.77 to 0.94 on that declared segment, which is a
coincidence to be aware of, not a result.

## A kernel that is slower in company

The obvious way to measure a kernel's efficiency curve is to call the
kernel. That was tried first, three ways, and it gave the wrong answer
every time, which is why the curve above comes from step times.

vLLM's own expert layers — real weights, real layout, real biases, the
real kernel — replayed alone on the hidden states and router logits
captured from a decode step, all 24 layers in a CUDA graph, run
**1.4–1.6× faster** than the same layers inside the engine's step at
batch 8 and above, and identically at batch 1. A synthetic benchmark of
the kernel with the engine's full 38 GB weight footprint agrees with the
isolated replay, not with the step. The suspects were checked one by
one: SM clock and power are the same in both (1.87–1.89 GHz, pinned at
the 300 W cap); the engine in eager mode shows the same in-step time as
under CUDA graphs; the profiler's prefill contamination was removed with
a clean decode-only cut. Whatever the fused-MoE kernel loses when
attention, dense GEMMs and norms run between its calls — cache state,
TLB reach, something in the memory system — it loses it only in company,
and the difference is not small. The model prices the in-step behaviour,
because a step is what a user pays for; the isolated numbers are in the
kernel-profile file for whoever finds the cause.

The torch profiler, meanwhile, reads these kernels 7–15% longer than
the unprofiled steps they sit in — the profiled kernel sums exceed the
measured step time at batch 16 and above. Kernel durations under CUPTI
are usually trusted; here they are not, and the milestone-62 curve drawn
from traces would have put the model 16–28% high at large batch.

> **Note (milestone 64).** "Slower in company" was the wrong picture.
> Sampled at 20 kHz, the kernel runs at 91% of the DRAM roof inside the
> step and 94% alone; it is not slower, the step *reads more bytes* — about
> 6 GB per decode step beyond what its kernels need at batch 8 and 32,
> roughly half an expert's weights per touched expert per layer, and
> almost nothing at batch 1. Where those bytes come from is not
> established; what they are not is listed in milestone 64. The curves
> above are unchanged: they measure the pipeline, which is what a step
> costs.

## Errata and what stays open

Milestone 62's two constants (0.70 / 0.53) are superseded by the curves;
its post carries a note. The two-GPU batch-32 decode is unchanged at
0.71–0.73: the rank holds sixteen sequences at eight rows per touched
expert, where the single-GPU curve says 0.71 and the rank runs at about
0.50. That is the same "slower in company" effect with more company —
the all-gather and reduce-scatter between the expert kernels — and it
will move when the cause does.

## Exercises

1. Reproduce the isolation gap with `nsys` instead of the torch profiler:
   are the in-step fused-MoE kernels slower per launch, or are there
   more of them?
2. Put the expert weights on 2 MB pages explicitly (or check the driver
   already does) and re-run the isolated replay against the step. If the
   gap closes, it was the TLB.
3. Set `VLLM_MOE_DP_CHUNK_SIZE` or a hand-tuned Triton config for
   `E=32, N=2880` on this GPU and re-run batch 64: how much of the 0.53
   comes back, and does the curve's step at fourteen rows disappear?
