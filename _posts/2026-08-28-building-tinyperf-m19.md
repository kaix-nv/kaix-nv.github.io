---
layout: post
math: true
title: "Building tinyperf M19: Chunked prefill — overlap, and the chunk-size knob"
date: 2026-08-28 16:35:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Riding prompt tokens under decode steps is a straight 2x throughput win at saturation, TPOT jitter turns out to be scheduling noise — and tiny chunks have their own failure mode."
---

*Milestone 19 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `chunk_tokens` in `serving.py` · Demo: `examples/17_chunked_prefill.py`.*

> **Erratum (milestone 32).** The throughput numbers in this post were
> measured against a prefill-first baseline that flattened each admitted
> batch into one concatenated sequence, making its attention quadratic in
> the combined length. Silicon validation caught it. Under the corrected
> batched-prefill baseline the headline win at this workload is **~1.3x**,
> not ~2x — chunking still wins throughput, TTFT p95 and TPOT
> simultaneously, so the qualitative conclusions stand at reduced
> magnitude.

Milestone 13's engine had one documented crudeness: prefill-prioritized
scheduling. When new requests arrive, the whole GPU turns into a prefill
machine and every running decode stalls. Milestone 18 built the price of a
mixed step; this milestone builds the scheduler that uses it.

## The engine change

With `chunk_tokens=N`, every step is a decode step for the running batch,
with up to N prompt tokens of one pending request riding along. The mixed
step is priced at the perfect-overlap bound — `max(decode_us, chunk_us)` —
which is honest by construction: weights are read once, and the chunk's
math pipelines under the decode's bandwidth. That overlap *is* the design
intent of chunked prefill; where both streams bind the same resource the
bound is optimistic, and we say so. (One request prefills at a time —
head-of-line — a documented simplification; real engines chunk several.)

## Three effects, one knob

7B on the calibrated A6000, prefill-heavy traffic (prompt 2048 / gen 128):

```
rate=4 req/s     TTFT p50   p95(ms)  TPOT p50  p95(ms)   tok/s
prefill-first      29769     64906      74.1    133.2      140
chunk=128          22926     40593      39.9     39.9      187
chunk=256          11742     20139      54.7     54.7      265
chunk=512           8401     14781      76.5     76.5      292
```

1. **The overlap win.** At prefill-heavy saturation, prefill-first spends
   whole steps on dedicated prefills while every decoder idles — and
   caps at 140 tok/s. Chunking rides the same prompt tokens under decode
   steps and reaches 292 tok/s. **Not a latency/throughput trade — a
   straight 2x win**, which is why every modern engine does this.
2. **The knob.** Chunk size trades admission speed against decode
   smoothness: 128 gives 39.9 ms decodes but 40 s admission tails; 512
   gives 76.5 ms decodes and 15 s tails. Pick the chunk from the TPOT
   SLO; take whatever TTFT it buys.
3. **Jitter vanishes.** Under chunking, TPOT p95 equals p50 to the first
   decimal — no step is ever a dedicated prefill, so decode pacing is
   uniform. The p95 >> p50 jitter of prefill-first was never physics;
   it was scheduling noise.

And one honest wrinkle the sweep surfaced: at *low* load, tiny chunks are
their own failure mode — chunk=128 at 1 req/s has **worse** TTFT than
prefill-first (3.8 s vs 0.3 s median), because a 2048-token prompt takes
16+ steps to trickle in when a single dedicated step would have finished
it. The knob has a floor: chunk too small starves admission even with an
empty queue. Real engines size chunks dynamically for exactly this reason.

## In production-scale models

Production schedulers run token-budget batching natively: each step packs
decode tokens plus as much prefill as the budget allows, across multiple
requests, with the budget tuned per SLO tier — plus preemption and
priority classes on top. The two-lever structure (budget for TPOT,
admission order for TTFT) is what this milestone's single knob distills.

## Exercises

1. Multi-request chunking: split the chunk budget across all waiting
   requests. Does head-of-line TTFT p95 improve without a TPOT cost?
2. Dynamic chunk sizing: shrink the chunk when the queue is empty (the
   low-load wrinkle above), grow it under backlog. How close to the
   per-rate-optimal static chunk does one rule get?
3. Re-run milestone 16's SLO price curve under chunked scheduling: how
   much cheaper does the TTFT-bound region get?
