---
layout: post
math: true
title: "Building tinyperf M81: gpt-oss's prefill, part by part: a hot expert, a slow attention kernel, and the constant that hid both"
date: 2026-09-28 17:45:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "A fitted constant priced gpt-oss's MoE prefill, and it was hiding three things: the model's own tile steps, a router that sends 95% of a prompt's tokens to one expert, and an attention kernel running at a third of the price the model charged. Measured alone, they bring every prefill step within 4% and retire the constant."
---

*Milestone 81 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `weight_only_expert_tflops` in `scheduler.py`, `tools/measure_marlin_moe.py` · Data: `data/validation/comparison_gpt_oss_20b_rtx_a6000_marlin_prefill.txt`, `data/calibration/marlin_moe_gpt_oss_20b_rtx_a6000.json`.*

Milestone 80 closed on a residual it had stated in advance. gpt-oss's
steps carrying a prompt priced low when small and high when large. M80
blamed the Marlin kernel's math: the model prices it with one constant,
0.66, fitted in milestone 60 on prefills of thousands of tokens. This
milestone took the step apart and measured each piece. The constant turned
out to be standing in for more than Marlin.

## Frozen, then measured

One fresh prompt at a time went to an idle server, and the engine's
clock timed the forward. The frozen model read:

```
  prompt tokens    128   256   384   512   640   768   1024  1280  1536  2048
  model / engine  1.01  0.95  0.91  0.93  0.90  0.96  1.05  0.96  1.05  1.02
```

It reads low below 768 tokens, as M80 said. Above that it zigzags.

## The kernel alone

`tools/measure_marlin_moe.py` loads the model and times one layer's own
Marlin call on the engine's repacked weights, inside a CUDA graph.
Against it, the model's per-layer price read 0.83–0.97 from 32 to 128 rows
per expert, and 1.03–1.15 from 192 up. The model prices an expert GEMM
with its tile-and-wave estimate times 0.66, and its tile choice jumps
between sizes; that was the zigzag. The kernel is smooth. Routed over 32,
16 or 8 experts, its rate follows rows per expert, within 8% of each other
from 64 rows up.

A curve by rows per expert, taken from the kernel, fixed the zigzag. It
still left three prompt lengths more than 5% low. The kernel had been
timed with random routing. The engine doesn't route randomly.

## A hot expert

So the tool took the routing from the engine instead. It prefilled a real
prompt, recorded every layer's router choices, and timed each layer's
experts on exactly those ids. The router sends about 95% of a prompt's
tokens to a single expert, in every layer:

```
  tokens                       128    256    384    512   1024   2048   8192
  the busiest expert's rows    120    240    364    483    974   1946   7801
  engine routing / uniform    0.90   1.16   1.21   1.05   1.06   1.04   1.06
```

vLLM chooses Marlin's row-block size from the mean rows over all 32
experts. That mean fits a spread of tokens no expert actually gets, and
the kernel runs up to 21% slower for it. vllm bench's prompts route within
2% of random token ids. The calibration's `weight_only_expert_tflops`
holds this rate by tokens per launch: 42 TFLOP/s at 128 tokens, rising to
90. One thing needed care. The kernel call also runs the activation and
the top-k sum, which the model prices as separate operations, so the
table is the kernel's time less those.

## The attention the constant hid

With the kernel measured, milestone 60's own validation cells fell. Its
8,192-token prompts read 0.83–0.85, where they had read 0.89–0.91. The
fitted 0.66 had been covering something else too.

gpt-oss on this GPU runs vLLM's Triton attention, because of its sinks.
The model had priced that attention's prefill like FlashAttention-2's.
Timed alone on one chunk, it runs three times slower:

```
  chunk tokens                     256   512   1024   2048   4096   8192
  FlashAttention-2's price / it   0.34  0.31  0.33   0.33   0.35   0.36
```

Its programs take two query tokens at a time. The calibration's
`triton_attn` backend now prices prefill at 0.215, a third of 0.65.
Decode attention keeps its own measured fields.

## Together

Neither change was fitted to a step. The lone-prompt steps:

```
  prompt tokens    128   256   384   512   640   768   1024  1280  1536  2048
  frozen          1.01  0.95  0.91  0.93  0.90  0.96  1.05  0.96  1.05  1.02
  now             1.02  0.98  1.00  0.98  0.96  1.00  1.02  0.97  0.99  1.00
```

Milestone 60's and 63's TTFT grids are now priced with the backend their
engine ran. They read 0.81–1.02, and the 8,192-token prompts 1.00–1.02.
The 0.66 constant no longer applies to any launch of 128 tokens or more.

## Held out

A new sweep of long prompts (1,536–4,608 tokens) ran online at 1–3.5
req/s on seed 15, frozen first. 17 of the 18 values landed inside their
intervals, and throughput read 0.99–1.00. Steps carrying a prompt chunk
read 0.97–1.00 by size, including second chunks attending to a
2,048-token prefix. The attention had only ever been timed on chunks
without one. The value outside was TPOT at 1 req/s, 10% high. That comes
from decode steps at contexts of 1.5k–4.6k tokens, which read 1.03–1.10;
Triton's decode kernel there is not measured.

That sweep does not tell the old model from the new one. On 2,048-token
chunks the faster Marlin and the slower attention cancel, and M80's model
reads it about as well. They differ on short prompt steps, on long
offline prompts, and on M80's own held-out sweeps. Re-priced with nothing
fitted to them, 45 of those 54 values now land inside, where the old
model had 36. Steps carrying a chunk beside many decodes still read 6–11%
low. The table came from a lone prompt's routing, and decodes spread
over more experts.

Everything ran on GPU 1. Once, during an exploratory replay that the
calibration does not use, a two-GPU job outside these runs shared it.

## Errata

- **M60:** its 0.66 constant, fitted on its TTFTs, stood in for the
  router's concentration and for Triton's prefill attention. Both are
  measured now. Its TTFT cells read 0.91–1.02, and its 512-token batch-1
  miss (0.80) is 0.91.
- **M80:** its residual was the constant, as it said, but not only
  Marlin. The tile model's steps, the hot expert and the attention each
  carried part.

## Exercises

1. The hot expert is a property of the workload, not the kernel. Replay
   real text through the router. Is the concentration there too, and how
   fast does Marlin run on it?
2. Steps carrying a chunk beside 30 decodes still read 6–11% low.
   Record a mixed step's routing and replay it. How much of the gap does
   it close?
3. Time Triton's decode kernel at contexts of 2k–5k tokens. Is it the
   decode residual on long prompts?
