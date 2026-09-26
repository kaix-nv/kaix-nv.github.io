---
layout: post
math: true
title: "Building tinyperf M75: The attention backend: FlashInfer packs what FlashAttention-2 re-reads, and the knee moves"
date: 2026-09-26 00:05:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Every vLLM run so far used FlashAttention-2, which re-reads each decoding sequence's KV once per query head in any step that also carries a prefill chunk. FlashInfer does not. Made a model parameter and frozen before the runs, it held on the engine's mixed-step grid in all 36 cells, and online it moved the saturation point where the model said it would."
---

*Milestone 75 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `attn_backend` in `serving.py`, `attention_backends` in `methodology.py`, `--backend flashinfer` on `tools/measure_decode_attention.py` and `tools/measure_mixed_attention.py` · Data: `data/validation/comparison_qwen3_8b_rtx_a6000_flashinfer.txt`, `engine_steps_m75_qwen3_8b_rtx_a6000.json`.*

Every vLLM run in this series used the engine's default attention
backend on this GPU, FlashAttention-2. Milestone 69 found a cost in it.
In any step that carries a prefill chunk beside decoding sequences, the
kernel stops packing a KV head's query heads together. Each decode row
then reads its KV cache once per query head: four times, for Qwen3-8B's
32 query heads over 8 KV heads. That milestone ended with an exercise.
Does FlashInfer, vLLM's other backend here, do the same? This milestone
answers it and makes the backend a model parameter.

## How FlashInfer runs a mixed step

vLLM 0.15.1's FlashInfer backend reorders every step so its decodes come
first. It then makes two calls per layer:

- **The decodes** go through FlashInfer's decode kernel with tensor cores
  on, which packs a KV head's query heads as rows. It does this whether
  or not the step carries a chunk.
- **The chunk** goes through a separate prefill call.

Both calls are planned on the host before the forward. So a mixed step
under FlashInfer should cost its two parts, with no re-read.

## The kernels alone

`tools/measure_mixed_attention.py --backend flashinfer` makes vLLM's
calls, with its layout and plan arguments, over M69's 220-cell grid of
rows × context × chunk. The extra time, beyond the decode rows and the
chunk each run alone, is 0.7% at the median. FlashAttention-2's is 86%.
Per layer, at a 1,150-token context beside a 512-token chunk:

```
  decode rows   FlashInfer   FlashAttention-2
        8          121 µs         166 µs
       32          275            807
       63          482           1722
```

The decode kernel alone, timed inside CUDA graphs across batch 1–64 and
context 256–8192, is 0–16% faster than FlashAttention-2's. Two constants
fit it, as M68's did for FlashAttention-2: 6.0 µs per call, plus the KV
streamed at 1.02 of the calibrated DRAM rate. The mean error is 2.6%,
and 1.3% at batch 8 and above. The same session re-ran M68's
FlashAttention-2 sweep, and it reproduced within 1.5% in every cell.

FlashInfer's prefill kernel runs 0.85–1.10× FlashAttention-2's and is
faster from about 1k tokens. The model prices it like FlashAttention-2's;
at these chunk sizes attention is a few percent of the step.

## One parameter

`StepLatencyModel(attn_backend="flashinfer")` swaps the calibration's
attention fields for FlashInfer's. The decode kernel gets FlashInfer's
two constants, and the mixed-step re-read coefficients become None:
packed. Everything else is the same model. `"flash_attn"`, the default,
is M74 unchanged.

## In the engine, decode is a wash

Pure decode steps on the engine's clock, before anything was frozen,
read 0.985–0.994 under FlashInfer and 1.000–1.009 under
FlashAttention-2. In the engine, though, FlashInfer's decode steps are
not faster than FlashAttention-2's: +1.6% at batch 1 and −0.9% at 64.
Its kernel alone is faster, so about 0.4 ms a step at batch 1 goes
somewhere the kernel measurement does not see. The frozen file said so.

## Frozen, then measured

Predictions went out before any run, under both backends. Each test ran
under FlashInfer and then under FlashAttention-2 on the same GPU, so
FlashAttention-2 is the control.

**The mixed step on the engine's clock** uses M69's protocol. 8, 32 or 63
sequences decode at background contexts of 256, 1,024 or 2,048 tokens,
and prompts of 128–1,024 tokens are injected. At 63 decodes:

```
  context  chunk   FlashInfer   FlashAttention-2   ratio measured (predicted)
    256     128       45.3 ms        56.9 ms           0.80 (0.79)
   1024     128       55.0           91.7              0.60 (0.59)
   1024     512      107.4          146.6              0.73 (0.73)
   2048     128       69.1          146.3              0.47 (0.46)
   2048     512      120.5          199.9              0.60 (0.59)
```

All 36 FlashInfer cells are on the model: 0.961–1.024, mean error 1.5%.
The FlashAttention-2 control reads 0.968–1.059. The measured ratio
between the backends lands within 7% of the predicted one in every cell.
At a 2,048-token context with 63 decodes, a FlashInfer mixed step costs
about half of FlashAttention-2's beside a short chunk and 0.70 beside a
1,024-token one.

**Online**, both backends ran the same trace: prompts 512–1,536 tokens,
outputs 128–384, seed 3, 1.5–4 req/s. The criteria were M73's:

```
  rate   TTFT p50: FlashInfer (r)   FA2 (r)     TPOT: FlashInfer (r)   FA2 (r)    tok/s FI / FA2
   1.5         248 ms (0.99)       251 (1.03)         37.9 (0.97)      38.7 (0.98)       1.00
   2.5         301    (0.96)       348 (0.95)         56.1 (0.96)      64.6 (0.97)       1.00
   3           373    (0.93)      2579 (0.90)         70.5 (0.96)      82.0 (1.00)       1.05
   4          6312    (0.93)      8341 (0.98)         74.9 (0.98)      83.0 (1.00)       1.09
```

The knee moved as predicted. At 3 req/s FlashAttention-2 is past it and
FlashInfer is not: its median TTFT is 0.14 of FlashAttention-2's, where
0.15 was predicted. TPOT is 13% lower at 2.5 req/s (14% predicted), and
throughput 9% higher at 4 req/s (10% predicted).

The control meets every criterion. FlashInfer meets them at three of four
rates. The miss is its 3 req/s cell: TTFT p95 reads 0.72, with 1,186 ms
measured against 852 predicted, while p50 reads 0.93 and TPOT 0.96. The
engine's own clock explains it. The recorded steps price at 0.981–0.990
of measured, so the model's FlashInfer steps run about 1.5% fast, as they
did in-sample. At the knee's tail that much moves p95 a long way:
re-priced with every step 2% slower, the cell reads p95 1,218 ms and TPOT
70.6.

The GPU the runs used carried nothing else. During the mixed-step grids
the other GPU ran other jobs, and the pure decode steps stayed
on the model throughout.

## Exercises

1. FlashInfer's decode steps carry about 0.4 ms more in the engine at
   batch 1 than its kernel alone accounts for. Trace one step with Nsight
   Systems and find it. Is it a kernel inside the CUDA graph, or the
   graph's own replay?
2. FlashInfer's prefill kernel is 15% faster than FlashAttention-2's at
   1,920-token chunks. Give it its own efficiency. Does a long-prompt
   sweep notice?
3. A 2% step error moved this knee's p95 by 43%. From the queue, derive
   how large a step error a p95 prediction at a knee can tolerate, and
   where it stops mattering.
