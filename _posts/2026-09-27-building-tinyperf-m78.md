---
layout: post
math: true
title: "Building tinyperf M78: Prefill steps route to fewer experts, and two kernel stories that were true but not the answer"
date: 2026-09-27 00:01:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "A step carrying a prefill chunk touches about 100 of an MoE's 128 experts, not all of them. Read off the engine, that closes M77's residual on the workload it came from, with every frozen criterion met on a new seed. Long prompts miss: their prompt-tail steps route more concentrated still. On the way, the fused-MoE kernel and FlashAttention-2's re-read each followed a clean law that the engine does not."
---

*Milestone 78 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — the `online` routing table in `nets/transformer.py`, `tools/measure_fused_moe.py`, `tools/measure_mixed_attention.py --graph` · Data: `data/validation/comparison_qwen3_30b_a3b_rtx_a6000_tp2_moe_prefill_routing.txt`.*

Milestone 77 closed with a residual, stated before its run and landing as
stated. Steps that carry a prefill chunk priced high, by up to 14% at
small sizes, falling to nothing at 2,000 tokens. M77 blamed the fused-MoE
kernel's large-block constant, which gpt-oss-20b had set at 0.70 of the
DRAM rate. This milestone went looking for it in the kernels, and found
two true things that weren't the answer before finding the one that was.

## The kernel, alone

`tools/measure_fused_moe.py` times vLLM's fused-MoE kernel inside CUDA
graphs, with uniform random routing, at two very different shapes:
gpt-oss-20b's 32 experts of 2880, and this model's 128 experts of 768 per
rank. Once a launch carries more tokens than there are experts, the
weight-streaming rate falls the same way for both, by rows per expert:

```
  rows per expert      8      16      32      48      64
  gpt-oss-20b        0.72    0.70    0.66    0.63    0.58
  Qwen3-30B-A3B      0.73    0.71    0.67    0.63    0.58
```

That is a clean property of the kernel's 64-row blocks. As a model
constant, though, it priced this model's large steps 6–10% high and
gpt-oss-20b's already-validated prefill 22% high. The engine's routing
isn't uniform, so the kernel under uniform routing isn't the kernel the
engine runs.

## The re-read, in a graph

The model's mixed steps also pay FlashAttention-2's re-read (milestone
69). This model has 8 query heads per KV head, twice what M69 fitted on.
Timed inside CUDA graphs at four head layouts, the re-read's miss
fraction follows one variable: the re-read traffic against the 6 MB L2.
It is m = 0.13 × log2(re-read bytes / 18 MiB), fitted on 32/8 heads and
held out on 64/8, 16/4 and 16/2 within 11–15%. On this model's layout,
M69's fit is off by 38%.

That is also true, and also not what the engine pays. vLLM runs a mixed
step's attention eagerly, between the CUDA-graph pieces, so launch time
the graph removes is real there. The graph-timed fit priced Qwen3-8B's
long-validated mixed steps 1–5% low, and M69's eager fit stays.

## The routing

M77's routing trace had recorded every step, not only the decode steps it
was built for. For steps carrying a prefill chunk:

```
  step tokens          ~110   ~200   ~360   ~640   ~900  ~1260  ~1900
  experts touched        89     96    101    102    100    102    111
  the model assumed     128    128    128    128    128    128    128
```

A chunk of one `vllm bench` prompt, a run of consecutive token ids, routes
to a concentrated set. The model had streamed every expert's weights for
these steps, 25–45% too many at the sizes where streaming dominates. The
`online` table now carries these counts past 64 tokens. They are routing
measurements, like M77's; no kernel constant changed. Re-priced, M76's
and M77's prefill-carrying steps read 0.98–1.03 at every size.

## Frozen, then measured

Seed 9, two shapes. H1 was M76's shape; H2 was long prompts of 1,024–3,072
tokens.

```
  H1, prefill-carrying steps   <=512    <=1024   <=1536   more
  M78                          0.997    0.994    0.986    1.009
  M77's table                  1.099    1.063    1.026    1.000
```

H1 meets every criterion: all four cells, and both step criteria. At its
knee (2.5 req/s) the model predicted a median TTFT of 1,030 ms against
1,100 measured. M77's table would have said 1,941.

H2 misses. Its small prefill-carrying steps read 1.145 (M77's table:
1.317), and two of its four cells miss: TPOT 1.08–1.09, and at 2 req/s
TTFT outside its band. H2's small steps are
different animals: the tail of one long prompt beside 4 decodes, where
H1's are 240 prompt tokens beside 49 decodes. The trace says what that
does. At the same token count, a step with few decodes touches fewer
experts: 93 against 106 at 513–1,024 tokens. The decodes are many
sequences, and the chunk is one prompt. The table, keyed by tokens, was
read off a workload whose small steps always carried many decodes. H2's
fall outside it.

## Errata

- **M77:** its residual was routing, not the fused-MoE constant, which
  holds on the engine.

## Exercises

1. Route by composition: price a step's experts as the union of its
   decode rows' (the decode table) and each chunk's (measured by chunk
   size alone). Does H2's 1.145 close?
2. The fused-MoE kernel under uniform routing collapses on rows per
   expert. Replay the engine's recorded routing through it instead. Does
   the engine's rate follow?
3. Time a mixed step's attention eagerly and in a graph, both inside the
   engine's piecewise schedule. How much of the difference is launch time
   the GPU actually waits for?
