---
layout: post
math: true
title: "Building tinyperf M80: A second MoE online: gpt-oss-20b on one GPU, and a pool nine times too small"
date: 2026-09-28 00:30:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "gpt-oss-20b served online for the first time, predictions frozen first: 9 of 18 values inside. Three things read off the run account for it: a KV pool nine times too small because capacity held MXFP4 experts at bf16, an online batch's wider routing, and an attention kernel that never paid the re-read the model charged. Frozen again on two new sweeps: 17 of 18 on long prompts; on short ones the residual stated before the run."
---

*Milestone 80 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `weight_only` in `capacity.py`, `gpt_oss_20b("online")` in `nets/transformer.py`, the `triton_attn` backend · Data: `data/validation/comparison_gpt_oss_20b_rtx_a6000_online.txt`, `engine_steps_m80_gpt_oss_20b_rtx_a6000.json`.*

Milestones 76–78 served an MoE online for the first time, Qwen3-30B-A3B
across two GPUs. They found things a fixed batch never shows: an online
batch's sequences share fewer experts, padded CUDA-graph rows route stale
tokens, and prefill steps touch fewer than all the experts. All of that
came from one model. gpt-oss-20b is a different MoE: 32 experts, top-4,
MXFP4 weights streamed by the Marlin kernel, on one GPU. Milestones 60–65
measured it on fixed batches only. This milestone served it online and
froze the model's intervals before the run.

## Frozen, then measured

The shape was M76's: prompts of 512–1,536 tokens, replies of 128–384, on
seed 12. Each measured value is shown with the frozen interval as ratios
to it; `*` marks a value outside:

```
  rate   TTFT p50 [low point high]    TTFT p95 [low point high]    TPOT [low point high]
   1      145 ms [0.97 1.05 1.11]      206 ms [0.97 1.05 1.19]     12.5 ms [0.97 1.01 1.05]
   2      159    [0.97 1.03 1.10]      257    [0.95 1.07 1.16]     19.4    [0.91 0.95 0.99]*
   3      179    [0.95 1.01 1.11]      325    [0.95 1.05 1.19]     27.0    [0.88 0.92 0.96]*
   3.5    185    [0.96 1.02 1.09]      354    [0.95 1.06 1.17]     31.0    [0.88 0.92 0.97]*
   4      200    [1.02 1.08 1.23]*     395    [2.55 3.15 3.80]*    35.6    [0.88 0.92 0.97]*
   5      328    [3.47 4.27 5.35]*    1886    [2.13 2.52 2.93]*    42.1    [0.87 0.90 0.93]*
```

Only 9 of the 18 values landed inside, against a criterion of 80%. The
errors pointed opposite ways. TPOT read low from 2 req/s, as the frozen
file had said it would. But the model put a queue at 4 req/s, and the
engine never formed one.

## The pool

The queue was not a step-time error. The simulator's KV pool held 67,312
tokens, and the engine logged 607,344. The model admits a request only
when its prompt and reply both fit. Sixty-seven thousand tokens held
about fifty of these requests, and 4 req/s needed more than that in flight.

The capacity model counted the weights at 38.9 GiB, where the checkpoint
loads 13.7. Milestone 60 taught the step pricing that gpt-oss's experts
are MXFP4, but the weight accounting never heard. It now takes the same
`weight_only` widths the latency model does. The pool comes out at 638,432
tokens, 5% above the engine's.

## The routing

M77's patch reads the router inside the CUDA graph. On gpt-oss it
crashed the engine at 2 req/s. vLLM 0.15.1 sizes the host copy of that
buffer for one KV-cache group, and gpt-oss has two, sliding-window and
full. The patch now skips that copy, which it never needed. Over 20,698
decode steps:

```
  decodes                   8      16      32      64
  online, the real rows    14.5    20.0    24.3    29.1
  a batch decoded in step  13.1    17.4    21.3    24.4
```

As with Qwen3, an online batch touches 11–19% more experts, and padded
steps touch more again. Seventeen decodes run as a 24-row graph touch 24.8
experts, where a full 24 touches 22.3. One difference showed up: up to 16
decodes, the engine ran this model's decode steps unpadded. Steps
carrying a prefill chunk touched 27–29 of the 32 experts, not all of
them. The tables are `gpt_oss_20b("online")`.

## The attention

The steps carrying a prefill chunk read high, by more the more decodes
they carried: 1.00 beside 3 decodes, 1.11 beside 48. That pattern belongs
to milestone 69's FlashAttention-2 re-read, where a mixed step's decode
rows each read their KV once per query head. But gpt-oss runs vLLM's
Triton attention, for its attention sinks. Its programs each take a block
of query tokens for all eight query heads of one KV head, in a mixed step
too.

Timed inside CUDA graphs on gpt-oss's attention, the mixed call cost less
than its decode rows and chunk run apart. That held in all 60 cells, full
and 128-token-window layers, by 0.01–1.0 decode KV reads. There is no
re-read. The calibration's `triton_attn` backend drops it, as FlashInfer's
did in M75.

## What was left

With all three changes, sweep A came in: 18 of 18 values inside,
re-predicted after the fact. The steps still showed one pattern. Steps
carrying a whole fresh prompt read low when small and high when large:

```
  prompt tokens in the step   513-768   769-1024   1025-1280   1281-1536   1537-2048
  model / engine               0.91      0.96       0.97        1.01        1.02-1.07
```

The re-read had hidden this. Marlin's expert math is one constant fitted
on prefills of thousands of tokens. A 600-token step gives each expert
about 75 rows, and the kernel runs slower there. This was stated in the
second frozen file, and not fixed.

> **Note (milestone 81).** The constant was the problem, but not only
> through Marlin. The model's tile steps, the router sending ~95% of a
> prompt's tokens to one expert, and Triton's prefill attention at a
> third of the price each carried part ([milestone 81]({% post_url 2026-09-28-building-tinyperf-m81 %})).

## Frozen again

Two new sweeps ran on seeds 13 and 14. B1 had short prompts (256–768
tokens), B2 long ones (1,024–3,072):

```
                                     B1, short prompts    B2, long prompts
  decode steps, 17-64 decodes           0.99-1.01            1.03-1.04
    the model as it stood               0.87-0.89            0.91-0.94
  prefill steps of 513-1024 tokens        0.91                 0.87
  values inside their intervals          10 of 18             17 of 18
  the cells' step bias                  0.965-0.983          1.000-1.020
```

B2 came in: 17 of 18 values inside and throughput 1.00 at every rate. The
model as it stood would have read its knee at 1.35–2.2× on TTFT. B1
missed at its knee. Its short prompts put a fifth of the engine's time in
the small steps the residual prices low. Its step bias fell outside
milestone 79's band, and at 5.5 req/s its median TTFT read 0.43. In all,
27 of 36 values landed inside. That is 75%, and the criterion was 80%.

Everything ran on GPU 1. At the first start, a job outside these runs
appeared on the same GPU. I stopped my run before it measured anything,
and restarted once the GPU had been free for a minute.

## Errata

- **M60, M74:** M60's `weight_only` priced the steps but never reached
  capacity. The pool held weight-only experts at bf16, nine times too
  small for gpt-oss-20b. No published figure used a weight-only model's
  pool.
- **M65:** its decode routing tables describe a batch decoding in step.
  Online, the real rows touch 11–19% more experts, and padded steps more.
  M65's fixed-batch validations stand.

## Exercises

1. Time vLLM's Marlin MoE on gpt-oss's shapes from 256 to 2,048 tokens,
   and fit its math rate by rows per expert. Does B1's step bias close?
2. Up to 16 decodes, vLLM ran gpt-oss's decode steps unpadded, where it
   padded Qwen3's at every size. Find out why, and what it saves.
3. The Triton mixed call costs less than its parts. Price the overlap:
   how much of a 48-decode mixed step does it save?
