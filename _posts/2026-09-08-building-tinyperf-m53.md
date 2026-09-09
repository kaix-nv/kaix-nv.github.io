---
layout: post
math: true
title: "Building tinyperf M53: Prefix caching: the system prompt you only pay for once"
date: 2026-09-08 17:33:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "A cached prefix turns a prefill into a chunk step and stores the shared tokens once. Measured with vLLM's prefix cache switched back on: seven of eight TTFT cells within 10% of frozen predictions, one honest miss, and a decode surprise named cascade attention."
---

*Milestone 53 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `cached_tokens` / `shared_prefix` on the step model, the prefix pool in `simulate` · Example: `examples/36_prefix_caching.py`.*

Chat traffic repeats itself. A system prompt goes in front of every
request; a conversation's history goes in front of every turn; a batch of
few-shot examples goes in front of every query. A prefix cache keeps the
KV of those shared tokens resident and prefills only what is new — the
feature milestone 32 had to *switch off* to measure a prefill at all,
because identical prompts made TTFT vanish. This milestone models it,
then measures it with the switch on.

## Three places it shows up

**TTFT.** A prefill of `S` new tokens behind `P` cached ones is a step the
model already had: milestone 18's chunk step, `n` new tokens attending to
an existing cache. `prefill_us(total, n_seqs, cached_tokens=P)` prices a
hit as exactly that. Only the suffix runs through the projections and
FFNs; its queries attend to the whole prompt.

**Capacity.** A prefix shared by a batch is stored once. `llm_memory` and
`max_batch` take `shared_prefix_tokens` and charge one copy plus each
sequence's suffix — for a 3k system prompt in front of 512-token turns,
four times the concurrent sessions.

**The serving engine.** Requests carry a `prefix_id` and `prefix_tokens`.
The engine keeps a pool: the first request with a prefix computes it and
puts it in the pool; later ones hit and prefill their suffix; a prefix
stays resident after its last user leaves and is evicted LRU only when an
admission needs the room — RadixAttention's retention policy, first
order. `poisson_requests(shared_prefix=..., n_prefixes=...)` generates
the traffic, and the sweep pricer takes the same keys.

## LLaMA3-70B on eight H100s, a 3k system prompt and 512-token turns

```
                       TTFT b=1   TTFT b=8   max concurrent @4k
  no cache                 160       1213                  301
  prefix cached             31        198                 1197
```

Under Poisson load, 80 requests, median and p95 TTFT:

```
  rate req/s   no cache p50    p95   cached p50   p95   hits
         1.0            168    310           38    45     79
         2.0            171    511           40    67     79
         4.0            390    985           41    69     79
```

The first request pays for the system prompt; seventy-nine do not. At
four requests a second the uncached deployment's p95 is approaching a
second while the cached one is under seventy milliseconds, and the
difference is not a faster GPU — it is 3072 tokens per request that are
never computed twice.

## Silicon: the switch turned back on

Qwen3-8B on this box's A6000 under vLLM with prefix caching enabled:
4096-token prompts whose first `P` tokens are one shared system prompt,
distinct suffixes, at batch 1 and 8. Predictions frozen and committed
first.

```
   b  cached  suffix   TTFT meas   pred  ratio   vs P=0   TPOT meas
   1       0    4096       579.0  609.2   1.05     1.00       24.64
   1    2048    2048       319.6  345.3   1.08     0.55       24.71
   1    3584     512        97.2   94.0   0.97     0.17       24.70
   1    3968     128        47.2   31.8   0.67     0.08       25.14
   8       0    4096      4595.2 4839.2   1.05     1.00       32.06
   8    2048    2048      2473.9 2731.6   1.10     0.54       30.30
   8    3584     512       661.6  709.5   1.07     0.14       29.04
   8    3968     128       192.1  184.8   0.96     0.04       28.94
```

Seven of eight TTFT cells sit within 0.96–1.10 of the frozen
predictions, and the mechanism is confirmed: caching 87% of a prompt
takes 83% off its TTFT at batch 1 (the residual is the suffix's attention
over the cached prefix, which the chunk step prices). The eighth cell is
a real miss and stays pinned as one: a 128-token prefill at batch 1 takes
47 ms against a predicted 32. Fifteen milliseconds of engine cost that
the model has no constant for on this device — the milestone-39 prefill
overhead was measured on a B200 — and that a constant would misplace,
because adding 15 ms to the 512-token cell of milestone 32 would break
that envelope. Tiny prefills are engine-bound, and the model says so by
missing them.

## The decode surprise

The frozen predictions said TPOT would not move: a cached prefix is still
a prefix, and decode reads all of it. At batch 1 that held to within 2%.
At batch 8 decode got **faster** with the shared prefix — 32.1 ms down to
28.9 ms with 97% of the context shared.

That is cascade attention. When every sequence in the batch shares a
prefix, the engine reads the shared prefix's KV *once* for all their
queries and merges the result with each sequence's own suffix, instead of
reading the same blocks eight times. The graph builder now does the same:
`shared_prefix` splits attention into a batch-folded op over the shared
cache and a per-sequence op over the suffix, and the engine applies it
whenever its running batch shares one prefix.

Full cascade over-corrects. Measured decode at batch 8 sits between the
no-sharing prediction (4–9% high) and the full-cascade one (5–11% low) in
every cell: the engine gets part of the dedup, not all of it — merge
kernels, block granularity, whatever its heuristics leave on the table.
The comparison reports both bounds and the test pins the bracket.
Nothing was fitted to close it; when a cleaner measurement isolates the
cascade path, the constant goes there.

## Not modeled

Block-aligned matching (a hit is all-or-nothing on the declared prefix);
partial hits from diverging conversations; the hash and lookup cost;
composition with the paged allocator (refused, an exercise); the
engine-side fixed cost of very small prefills; and cascade attention's
exact accounting, bracketed above.

## Exercises

1. Compose the prefix pool with milestone 20's paged allocator: prefix
   pages are just pages nobody has freed. Where does eviction pressure
   bite first, the prefix or the running requests?
2. Measure the tiny-prefill cost on your engine (prompt 64, 128, 256
   tokens, batch 1) and decide whether it is per request or per step.
3. Cascade attention's benefit depends on how many sequences share and
   how much. Sweep both and find where the model's full-dedup bound and
   the no-share bound are farther apart than the model's error bar.
