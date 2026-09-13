---
layout: post
math: true
title: "Building tinyperf M60: The first MoE silicon anchor: gpt-oss-20B, random routing, and what a weight-only GEMM costs"
date: 2026-09-12 22:22:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Every mixture-of-experts number in the series had met another model and never a GPU. gpt-oss-20B on one A6000 changes that: decode at batch 1 within 9% out of the box, and two mechanisms — random routing touches 20 of 32 experts, not 32; a weight-only GEMM runs at two thirds of dense efficiency — for the rest."
---

*Milestone 60 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `gpt_oss_20b`, `apply_weight_only`, the routing rule in `nets/transformer.py` · Data: `data/validation/comparison_gpt_oss_20b_rtx_a6000.txt`.*

Every mixture-of-experts number in this series had been checked against
another model and never against a GPU. The largest open disagreement
with the independent reference model was on MoE — a factor of two on
both phases of gpt-oss-120B, traced to how the two price expert
computation — and it could not be settled model-versus-model. gpt-oss-20B
has the same layer structure at 24 layers and 32 experts and fits one of
this box's A6000s in its shipped MXFP4 form. This milestone downloads it,
freezes predictions, runs vLLM's eight-cell grid, and lets the silicon
speak.

## Pricing a checkpoint that ships in MXFP4

The experts are stored at 4.25 bits per weight; attention, router and
embeddings are bf16. On a GPU with no native FP4 tensor rate the engine
runs its weight-only path: stream the packed weights, dequantize in the
kernel, multiply in bf16. That is a distinct thing from milestone 35's
precision recipes, which change both the bytes and the math. `apply_weight_only`
tags weight GEMMs with bytes per weight; the GEMM model scales the B
operand's traffic by it and leaves the FLOPs at the activation dtype. It
composes with the precision recipes for GPUs that do have the rate, and
refuses attention matmuls like its siblings.

One more thing had to be built before anything ran: the box's NVIDIA
userland had been upgraded under a running kernel module, and vLLM gates
its CUDA platform on the NVML library, which no longer initialized while
CUDA itself worked fine. The measurement tool gained an opt-in flag that
opens that gate, applied in every interpreter because the engine core is
a spawned child. Check the driver stack before the model.

## The frozen predictions, and what silicon said

```
   b  prompt   TTFT meas   frozen    r      TPOT meas   frozen    r
   1     512        59.4     35.9  0.60          7.05     6.74  0.96
   1    2048       183.7    134.9  0.73          7.21     6.80  0.94
   1    8192       850.0    592.1  0.70          7.72     7.02  0.91
   8     512       344.0    257.3  0.75         12.68    20.13  1.59
   8    2048      1374.7   1049.9  0.76         13.77    20.56  1.49
   8    8192      6925.8   4712.2  0.68         14.90    22.31  1.50
  32     512      1299.9   1017.1  0.78         18.07    21.18  1.17
  32    2048      5494.6   4187.7  0.76         20.22    22.93  1.13
```

Decode at batch 1 landed within 9%: the weight-only bytes are right, and
so is everything the model prices for a small-batch MoE step. Two things
were wrong, and they have the two shapes a modeler wants to see.

**Decode at batch 8 was 50% high, and batch 32 was not.** Milestone 11
assumed balanced routing: a step with `a` expert assignments touches
`min(e, a)` of a rank's `e` experts. At batch 8 with top-4 over 32
experts that says all 32, and every step streams every expert's weights.
Real routers scatter assignments at random, and 32 draws from 32 bins
land in about 20 of them: `e(1 − (1 − 1/e)^a)`. At batch 1 the two rules
agree (4 draws, 4 experts); at batch 32 they agree again (128 draws fill
the bins); at batch 8 they differ by the 1.5× the silicon showed. The
builder now uses the expectation, and the reference model, it turns out,
always had — its batch-8 step ran 22 expert GEMMs, not 32. Milestone 43's
exercise 1 was this question; the answer is now a mechanism, with the
imbalance knob still skewing the rows per touched expert.

**Prefill was 25–40% low at every cell, in proportion to the expert
work.** With the routing rule in place the expert GEMMs still needed 1.4
to 1.8× their dense-bf16 price to close the prefill cells — and the
kernel says why. The weight-only Marlin path dequantizes inside the
mainloop and was built for small M; at prefill's thousands of rows it
runs well below cuBLAS. That is a property of the kernel class on this
GPU, so it is a calibration constant: `weight_only_math_efficiency = 0.66`,
fitted on the batch-8 and batch-32 prefill cells and checked on batch 1,
which it had not seen: 0.98 at 2k tokens and 0.91 at 8k. The 512-token
batch-1 cell stays at 0.80, the same tiny-prefill engine cost milestone
53 pinned on a 128-token prompt.

## After

```
   b  prompt   TTFT meas    live    r      TPOT meas    live    r
   1     512        59.4    47.3  0.80          7.05    6.76  0.96
   1    2048       183.7   179.6  0.98          7.21    6.81  0.94
   1    8192       850.0   769.9  0.91          7.72    7.03  0.91
   8     512       344.0   346.4  1.01         12.68   14.79  1.17
   8    2048      1374.7  1405.1  1.02         13.77   15.22  1.11
   8    8192      6925.8  6132.3  0.89         14.90   16.97  1.14
  32     512      1299.9  1372.3  1.06         18.07   21.17  1.17
  32    2048      5494.6  5607.9  1.02         20.22   22.92  1.13
```

TTFT 0.96 geometric, TPOT 1.06. The decode steps at batch 8 and 32 sit
11–17% high, which is the same size as the calibrated tier's overshoot on
dense decode at higher batch, and is the open item.

## What this settles about the reference disagreement

The two models were 2× apart on MoE because one builds a chain of
per-expert operators and the other prices grouped GEMMs, and neither had
met a GPU. Now one has. At batch 1, where the disagreement is purest,
the silicon sits within 9% of tinyperf's decode; the reference's
per-expert accounting would put the same step well above where the GPU
ran it. That does not make the reference wrong about everything — its
distinct-expert count was right where tinyperf's was not — but on the
question of what an expert layer costs to execute, the grouped-GEMM
assumption is the one the hardware bears out on this stack. The 120B
model, the datacenter GPUs and a bf16 checkpoint remain unmeasured, and
the envelope says so.

## Exercises

1. Run the same grid with a bf16 copy of the experts (dequantize once,
   save, serve) and compare like-for-like with the reference model's
   bf16 numbers. Does the 2× survive contact?
2. The weight-only efficiency was fitted at M ≥ 4096 rows. Sweep a Marlin
   GEMM directly from 8 to 8192 rows and find where the kernel's
   dequantization stops being free.
3. Add the routing rule to the M43 imbalance table and re-derive where
   imbalance stops mattering for decode.
