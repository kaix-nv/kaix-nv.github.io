---
layout: post
math: true
title: "Building tinyperf M77: The padded rows, priced: the engine's own routing, read inside the CUDA graph"
date: 2026-09-26 20:22:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "A CUDA-graph decode step's padding rows are fed stale tokens, and an MoE router routes them. Read inside the graph by vLLM's own routed-experts capturer (whose hooks vLLM leaves unset), the engine's routing prices a new seed's padded decode steps at 0.991 where they had read 0.908, and all four frozen online cells land within the criteria."
---

*Milestone 77 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `routing_padded` in `nets/transformer.py`, `tools/vllm_patches/moe_padding_trace` · Data: `data/validation/comparison_qwen3_30b_a3b_rtx_a6000_tp2_moe_padding.txt`, `engine_steps_m77_qwen3_30b_a3b_rtx_a6000.json`.*

Milestone 76 ended on an unpriced mechanism. vLLM runs a decode step in
the CUDA graph for the next batch size up, and it fills the extra rows
with whatever tokens its input buffer still holds. An MoE router routes
those stale tokens to experts like any other. In M76's online sweep,
decode steps that filled a graph size priced at 1.004, and padded ones at
0.919. Pricing the padding needed the routing of steps nobody could see,
because Python hooks don't run inside a replayed CUDA graph.

## Reading the router inside the graph

vLLM already has the instrument. With `--enable-return-routed-experts`,
every fused-MoE layer writes each row's top-k experts into a device
buffer. The write is part of the layer, so it is captured into the CUDA
graph and runs on every replay, padded rows included. The engine clears
the buffer before each step and copies it out after.

In 0.15.1 it writes nothing, though. The layers look for the capturer
when they are built, and the capturer is created later, when the KV cache
is allocated. So every layer's hook is unset. `moe_padding_trace` binds
the hooks once the capturer exists, which is still before the graphs are
captured. After each step it reads the buffer on the first rank and
records the distinct experts per layer among the real rows, all rows,
and the padded rows alone.

It ran on M76's shape, from a new seed (7), at 1–3.5 req/s: 9,330 decode
steps covering every batch size from 1 to 64. It synchronizes every step,
so it measures routing, not time.

## What the padded rows do

With p padded rows beside the real ones:

```
  p padded rows               1     2     3     4     5     6     7
  experts they touch alone    8    14    19    23    27    29    33
```

That is roughly what p more decoding sequences would touch. What they add
to the real rows depends on how many real rows there are: at 9 decodes
padded to 16, 19 experts (37.6 to 56.4); at 15, 3.5. So a step of 9
decodes touches more experts than a full step of 16 (47.4), and the
engine's clock had shown exactly that.

A second finding came with it. The real rows of an online batch touch more
experts than a fixed batch's: 87 at 64 decodes, where the table M76 froze
(a batch started together) has 77. Online, sequences are at every point
of their replies and share less.

## One table, by real decodes

The measurement gives, for each number of real decodes, the experts the
step touches as it executes. At a graph size that is the real rows. In
between, it includes the padded rows' stale tokens. It is
`QWEN3_30B_A3B_PADDED_ROUTING`, used when a decode step is padded
(`qwen3_30b_a3b("online")`). Everything else in the step is unchanged:
the GEMMs still run every padded row, and attention reads only the real
ones.

## Frozen, then measured

M76's held-out run, re-priced with this table, has its decode steps at
1.010 unpadded and 1.009 padded. That run is in-sample for the mechanism,
so the predictions went out on a new seed (8) before it ran. The frozen
file also stated a residual in advance. Steps that carry a prefill chunk
had priced high in M76's run, up to 16% at small chunks, and would again.

```
  online decode steps, model / engine     unpadded   padded
  M77 (the engine's routing)                1.013     0.991
  M76's table                               1.007     0.908
```

```
  rate   TTFT p50 (r)    TTFT p95 (r)    TPOT (r)       tok/s   M76's table: TPOT, p95
   1       258 ms 1.07     437 ms 1.05    30.4 ms 1.01   1.00        0.89, 1.04
   2       372    1.04     755    1.05    64.5    1.01   1.00        0.89, 1.00
   2.5     538    1.04    4223    1.03    85.6    1.01   1.00        0.94, 0.72
   3.5    6920    1.09   19215    1.05    91.9    1.02   0.99        0.99, 0.97
```

All four cells meet M76's criteria, and the decode-step criterion holds.
With M76's table the same cells miss three times, twice on TPOT and once
on p95.

The residual landed as stated. By the step's size, prefill-carrying
steps read 1.14, 1.13, 1.07, 1.02 and 1.00 (stated: 1.16, 1.12, 1.06,
1.03, 1.01). The fused-MoE kernel's large-block regime was fitted on
gpt-oss-20b's 2880-wide experts, at 0.70 of the DRAM rate. It is too slow
for this model's 768-wide ones, which is why TTFT reads a few percent
high.

The GPU process log shows only the runs' own workers in every window.

## Exercises

1. Price the prefill-carrying steps. Time vLLM's fused-MoE kernel alone
   at this model's per-rank shapes from 129 to 2048 tokens, and compare
   it with gpt-oss-20b's. Is the 0.70 a property of the kernel's blocks,
   or of the expert size?
2. The padded rows route like extra decoding sequences. If vLLM zeroed
   them instead, they would all be one token and touch at most 8 experts.
   How much faster would a 9-decode step be?
3. The table is keyed by real decodes and was read off one workload.
   Measure it on real text. Does the padded rows' share change when the
   stale tokens are prose instead of random ids?
