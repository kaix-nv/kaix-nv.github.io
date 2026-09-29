---
layout: post
math: true
title: "Building tinyperf M82: A chunk beside decodes: an expectation that failed, a cost that was hiding, and a correction that cost a sweep"
date: 2026-09-28 23:15:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "The frozen guess said Marlin would run 10-25% slower on a mixed step's routing. It didn't. Instead a lone chunk was priced high and every decode beside it too cheap, errors that half-cancelled. Priced by composition, the short-prompt sweeps come in; a derivation error found on the way, once corrected, costs the long-prompt sweep, and points at what to measure next."
---

*Milestone 82 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `weight_only_expert_mixed_us` in `scheduler.py`, `tools/measure_marlin_moe.py --routing mixed` · Data: `data/validation/comparison_gpt_oss_20b_rtx_a6000_mixed_routing.txt`.*

Milestone 81 left one online residual. Steps that carry a prompt chunk
beside many decodes read 6–11% low. M81's guess was routing. Its Marlin
table came from lone prompts, where one expert takes 95% of the tokens,
and a step's decodes spread over more experts. The frozen file put
it plainly: on a mixed step's routing, the kernel would run 10–25% above
the model.

## The kernel on a mixed step's routing

To get that routing, `tools/measure_marlin_moe.py --routing mixed` steps
the engine by hand. It admits d requests a few steps apart, so their
replies sit at different positions, and then a prompt of c tokens. It
records the routers on the forward that carries both, and times every
layer's experts on exactly those ids. The frozen price over the kernel:

```
  decodes    c=128   c=256   c=512   c=1024
     0        1.24    1.07    1.07    1.05
    16        1.11    1.03    1.01    1.01
    32        0.99    1.02    0.96    1.00
    64        0.98    0.99    0.96    0.98
```

The expectation failed. The kernel never ran more than 4% above the
model. What the grid showed instead was two errors pulling against each
other. A lone chunk was priced high: it touches about 22 experts, and the
model streamed the 27–29 in its online table. Each decode beside the
chunk was priced low. The kernel adds 4–6 µs per decode per layer, where
the model's table by tokens added about one.

The engine's own mixed steps agree. At 16–64 decodes beside a 512- or
1,024-token chunk, the decodes added 1.6–2.3 times what the model said.
No host time sat between the steps; the forward itself was longer.

## A table by composition

`weight_only_expert_mixed_us` is the kernel's time by decodes (0–64) and
chunk size (128–1,024 tokens). A longer chunk takes M81's rate, plus the
decodes' increment at the table's edge. Decode steps and lone chunks
keep their paths. M80's short-prompt sweep had read 0.89–0.95 on its
chunk-carrying steps. It now reads 0.96–0.99.

Building this turned up an error in M81. Its table took the kernel's time
and subtracted the model's own price for the activation and the top-k
sum, which the kernel call also runs. It priced those at the eager launch
cost, 25 µs each. Served steps pay the CUDA-graph cost, 3.5 µs. That made
Marlin 43 µs per layer too cheap. Both tables are now derived on the
graph stack.

## Frozen, then measured

Sweep D had B1's short prompts on a new seed. It landed 16 of 18 values
inside its intervals, with throughput at 1.01. Its chunk-carrying steps
read 0.95–0.98 by size, where M81's model reads 0.89–0.94. At the knee
M81's model would have read 0.48 on TTFT p95. The two values outside are
TPOT at 4.5 and 5 req/s, 4% low. Decode steps with only 4–8 decodes
read 0.91–0.94.

## What it cost

M81's own held-out sweep, long prompts on seed 15, now lands only 10 of
its 18 values. Mostly that's TPOT, 7–14% high. Its decode steps at
contexts of 1.5k–4.6k tokens read 1.03–1.10, and M81 had said so. Those
had been cancelled by the Marlin price that was 43 µs too cheap, and the
correction took the cancellation away. Across all five gpt-oss sweeps,
78 of 90 values now land inside, against 76 before. The short-prompt
sweeps gained and the long-prompt one lost. Both were measured, and the
loss points at the next thing to measure.

> **Note (milestone 83).** It was not the long contexts. On the online
> table those decode steps barely move with context. The sweep's replies
> were short, and a decode early in its reply touches fewer experts
> ([milestone 83]({% post_url 2026-09-29-building-tinyperf-m83 %})).

Everything ran on GPU 1. Only this milestone's own processes used it.

## Errata

- **M81:** its Marlin table was 43 µs per layer too cheap. The activation
  and sum were subtracted at the eager launch cost. Its lone-prompt steps
  read 0.978–1.034 re-derived. Its long-prompt sweep's 17 of 18 had
  leaned on the error.

## Exercises

1. Time Triton's decode kernel at contexts of 2k–5k tokens. Does it
   bring the long-prompt sweep back?
2. Decode steps of 4–8 sequences read 0.91–0.94. vLLM runs batches up to
   16 through the split-softmax (3D) kernel. Is that where the time goes?
3. The replay ran eagerly. In the engine, steps up to 1,024 tokens run in
   piecewise graphs whose padded rows hold stale tokens. Do those route
   differently?
