---
layout: post
math: true
title: "Building tinyperf M61: Like for like: gpt-oss-20B in bf16, a fused-MoE kernel at 44%, and a router that is not uniform"
date: 2026-09-14 12:30:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "The same eight cells with every weight in bf16. Batch-1 decode lands within 2% with nothing fitted; prefill was 1.8× fast because the engine's fused-MoE kernel runs at 44% of dense on this GPU; batch-8 decode was 40% high because the router is not uniform — a Zipf skew of 1.4 touches 13 of 32 experts, not 21. Decode is settled; prefill is a kernel question."
---

*Milestone 61 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `routing_skew` and the per-token routing rule in `nets/transformer.py`, `moe_math_efficiency` in `methodology.py` · Data: `data/validation/comparison_gpt_oss_20b_bf16_rtx_a6000.txt`.*

Milestone 60 ended with an exercise: run the same grid with a bf16 copy
of the experts and compare like-for-like with the reference model's bf16
numbers. The MXFP4 run had settled what it could — decode at batch 1 sat
within 9% of the grouped-GEMM price — but every comparison to the
reference still crossed a precision boundary, and every prefill cell
carried a fitted constant whose cause was a guess. A community release
of the checkpoint dequantized to bf16 weighs 40 GB and fits the A6000
with room for a 112k-token cache. Same eight cells, same engine,
predictions frozen and committed before the run was opened.

One line in the engine log matters for everything below. With bf16
experts vLLM takes its Triton fused-MoE path, and on this GPU it prints
`Using default MoE config` — the kernel has a per-shape tuning table
and there is no entry for an RTX A6000, so it runs the default tile
configuration.

## The frozen predictions, and what silicon said

```
   b  prompt   TTFT meas   frozen    r      TPOT meas   frozen    r
   1     512        92.6     71.0  0.77         11.53    11.76  1.02
   1    2048       245.6    134.9  0.55         11.66    11.81  1.01
   1    8192      1119.2    592.1  0.53         12.16    12.03  0.99
   8     512       489.8    257.3  0.53         30.65    39.83  1.30
   8    2048      1901.4   1049.9  0.55         30.09    40.26  1.34
   8    8192      9062.6   4712.2  0.52         29.68    42.01  1.42
  32     512      1842.3   1017.1  0.55         42.95    59.98  1.40
  32    2048      7607.5   4187.7  0.55         45.68    61.73  1.35
```

Batch-1 decode landed at 0.99–1.02 with nothing fitted. That is the
cleanest result the series has produced for MoE: four experts' bf16
weights plus attention, priced from datasheet bandwidth and the
calibrated launch constant, within 2% of an engine. The other two
patterns are the same two shapes milestone 60 saw, both larger.

**Prefill was 1.8× fast at every cell above 512 tokens.** Not 25–40% low
as in MXFP4 — a flat 0.52–0.55, in proportion to the expert work. At
batch 8 × 2048 the expert GEMMs are 73% of the modeled prefill, so the
miss on the total means those GEMMs ran 2.2–2.6× below their
dense-bf16 price. Their shapes are not the problem: 2048 rows per
expert against 5760 × 2880 and 2880 × 2880 weights are exactly what
cuBLAS runs near peak. The kernel is. A Triton grouped GEMM on the
default tile configuration, on a GPU its tuning table has never seen,
is a different kernel class from cuBLAS, and its efficiency is a
property of that class on this stack. It gets a constant of its own:
`moe_math_efficiency = 0.44`, applied to expert GEMMs that are not
weight-only, fitted on the batch-8 and batch-32 prefill cells and held
out on batch 1, which lands at 1.00 (2k) and 0.92 (8k). The 512-token
cell stays at 0.77 — the tiny-prefill engine cost of milestone 53. The
weight-only constant from milestone 60 (0.66, Marlin) is a separate
number for a separate kernel; the two do not stack.

**Decode at batch 8 and 32 was 30–42% high.** Milestone 60 had already
replaced balanced routing with random routing, and the MXFP4 run had
still come in 11–17% high there, which that post called the open item.
The bf16 run makes the miss bigger and its shape clearer. The steps at
batch 8 ran at the price of about 13 experts' weights, not the 21 that
uniform random routing predicts for 8 tokens picking 4 distinct experts
each. The router is not uniform. Some experts are popular, and a
concentrated router touches fewer distinct experts per step than a fair
one — which is less weight traffic, which is what a small-batch decode
step pays for.

## A router with a popularity distribution

The rule now takes a measured skew. Expert `i` has a popularity
`p_i ∝ i^−z`, a Zipf law; a token picks `top_k` *distinct* experts, so
each expert's inclusion probability is `q_i = min(1, c·p_i)` with `c`
chosen so the `q_i` sum to `top_k`. Over `t` tokens on a rank the
expected number of distinct experts touched is

```
touched = Σ_i  1 − (1 − q_i)^t
```

and `z` is `routing_skew`. Two properties of the form are why it is
written this way. At `t = 1` it is exactly `top_k` for any `z` — one
token touches four experts whether the router is fair or concentrated —
so the batch-1 cells that were already right stay right. (The first
attempt used a with-replacement form, which sends batch-1 decode 13%
low under any skew above zero; the silicon caught it in one run.) And
at `z = 0` it reduces to the uniform rule, `e(1 − (1 − k/e)^t)`, which
puts 8 tokens × top-4 at 21 of 32 experts. Milestone 60 quoted 20 from
the coarser with-replacement approximation; the post carries a note.

`z = 1.4` was fitted on the bf16 batch-8 and batch-32 decode cells. The
distinct-expert curve it produces, against uniform:

```
  batch        1    2    4    8   16   32   64  128
  skew 1.4     4    6    9   13   18   23   28   31
  uniform      4    8   13   21   28   32   32   32
```

The skew is a property of the model's router and of the traffic — these
are random-token prompts, and real text may be more or less concentrated
— so it lives on the preset (`gpt_oss_20b` carries 1.4), can be
overridden per run, and defaults to 0 for any model that has not been
measured. Zero is uniform, which is the conservative assumption: the
most distinct experts and the most weight traffic a step can have.
`moe_imbalance` is unchanged and still sets the hot expert's rows; the
two knobs answer different questions.

## After

```
   b  prompt   TTFT meas    live    r      TPOT meas    live    r
   1     512        92.6     71.0  0.77         11.53    11.81  1.02
   1    2048       245.6    245.2  1.00         11.66    11.87  1.02
   1    8192      1119.2   1031.2  0.92         12.16    12.08  0.99
   8     512       489.8    477.4  0.97         30.65    27.68  0.90
   8    2048      1901.4   1927.5  1.01         30.09    28.12  0.93
   8    8192      9062.6   8220.7  0.91         29.68    29.87  1.01
  32     512      1842.3   1894.8  1.03         42.95    46.09  1.07
  32    2048      7607.5   7696.3  1.01         45.68    47.83  1.05
```

TTFT 0.95 geometric (0.77–1.03), TPOT 1.00 (0.90–1.07). Two constants,
each fitted on four cells and checked on four it had not seen.

> **Erratum (milestone 62).** The skew of 1.4 was an *effective*
> constant, not the router's. Milestone 62 read the router directly: a
> decode step touches 4 / 5.5 / 8.4 / 9.7 / 12.5 / 15.9 of 32 experts at
> batch 1 / 2 / 4 / 8 / 16 / 32 — a Zipf exponent of 2.15, so batch 8
> touches 10, not 13. The step times fitted here matched because the
> fused-MoE decode kernel streams the touched experts' weights at only
> 0.70 of the calibrated DRAM rate once an expert has more than one row
> (the engine's own kernel profile), and the timing fit had folded that
> kernel inefficiency into a larger expert count. The two are now
> separate, measured constants (`routing_skew = 2.15`,
> `moe_dram_efficiency = 0.70`); the batch-1 result, within 2% with
> nothing fitted, stands. The MXFP4 held-out numbers above move too
> (the Marlin kernel streams at 0.53; the MXFP4 grid re-lands at
> 0.91–1.18).
>
> **Erratum (milestone 65).** The milestone-62 correction above was
> itself wrong, and this post's fitted skew was closer to the truth. The
> router count 9.7 was taken at the prompt's last position; a decode step
> routes generated tokens, and routing spreads as the decode proceeds.
> Averaged over the 128 decode steps a TPOT covers, the engine's own
> routers touch 13.1 experts at batch 8 and 21.3 at batch 32 on these
> prompts — the 1.4 skew said 13 and 23. The fused-MoE kernel, paired
> launch by launch with each step's routing, streams at the calibrated
> DRAM rate (1.0, not 0.70) up to 32 tokens per launch. This grid lands at
> 0.90–1.02 on decode with the measured routing and no fitted term.

The skew was fitted on bf16 and the MXFP4 run of milestone 60 never saw
it, which makes that run a held-out check of the routing rule alone:

```
  MXFP4 decode     b  prompt   meas   M60 rule    r   M61 rule    r
                   8     512  12.68     14.79  1.17     11.41  0.90
                   8    2048  13.77     15.22  1.11     11.84  0.86
                   8    8192  14.90     16.97  1.14     13.59  0.91
                  32     512  18.07     21.17  1.17     17.29  0.96
                  32    2048  20.22     22.92  1.13     19.04  0.94
```

The open item of milestone 60 was routing skew. The rule now runs the
Marlin path 4–14% low at these batches rather than 11–17% high; what
remains is the residual per-step cost of that kernel at 13–23 touched
experts, and it is the new open item, smaller than the old one.

## What this settles about the reference disagreement

Milestone 60 could say that silicon favored the grouped-GEMM accounting
for decode at batch 1, across a precision boundary. With every weight
in bf16 on all three sides the statement is clean. **Decode is settled.**
The GPU runs a batch-1 step within 2% of the grouped-GEMM price with
nothing fitted, and the reference model's per-expert operator chains
put the same step well above where the GPU ran it, at every batch and
context in the grid. For the weight-streaming regime the cost of an
expert layer is its weight traffic, and building it as a chain of
per-expert operators overcharges it.

**Prefill is not settled in either model's favor.** tinyperf priced the
expert GEMMs at dense efficiency and was 1.8× fast; the reference sat on
the other side of the silicon, above it by a smaller margin. The truth
was between them, and what decides it is a kernel-efficiency question
that neither model could answer from first principles: which grouped
GEMM the engine picks for this GPU and whether it has been tuned for it.
Here it had not been, and 0.44 is that fact expressed as a number. It is
a per-stack constant. On a GPU with an entry in the tuning table, or an
engine that routes bf16 experts to a CUTLASS grouped GEMM, it should be
re-measured and is expected to move toward 1 — the envelope says so, and
says which stack the 0.44 belongs to.

## Erratum for milestone 60

Two corrections, both in the milestone 60 post as notes. The 0.66
weight-only constant was attributed to dequantization inside the GEMM
mainloop. The bf16 run shows a fused-MoE kernel with no dequantization
at all running at 0.44 of dense efficiency on the same GPU, so most of
what 0.66 measures is the fused-MoE kernel class, not the dequantize.
The number stands; the story behind it does not. And the 11–17% decode
overshoot at batch 8–32 that the post left as an open item was routing
skew, not a calibrated-tier effect.

## What this does not model

The skew of a real router on real text — 1.4 is one model on random
tokens, and a chat workload with a hot domain will concentrate further.
Skew that varies by layer (the number is one Zipf exponent for all 24).
The fused-MoE kernel on a GPU it has been tuned for, where 0.44 should
not be reused. And still: gpt-oss-120B, any datacenter GPU, and expert
parallelism across ranks, where the routing rule's per-rank
distinct-expert count is exactly what the all-to-all skew of milestone
43 depends on and has not been measured.

## Exercises

1. Run vLLM's fused-MoE tuning script for this GPU and re-measure the
   batch-8 prefill cell with the generated config. How much of the 0.44
   comes back, and does the constant belong to the kernel or to the
   missing table entry?
2. Log the router's expert histogram on a real text corpus and on random
   tokens; fit `z` to each. Does 1.4 survive real text, and does the
   histogram look like a Zipf law at all or like a few hot experts and a
   flat tail?
3. From the touched-expert curve, find the batch at which skew stops
   mattering for this model (distinct experts within 10% of all 32) and
   compare with where the milestone 43 imbalance table said routing
   stops being the decode bottleneck.
