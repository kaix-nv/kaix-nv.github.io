---
layout: post
math: true
title: "Building tinyperf M18: The chunk step — where verification stops being free"
date: 2026-08-28 15:20:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "One new step shape prices speculative verification honestly — and corrects the folklore: at long context, speculative decoding is a throughput tool."
---

*Milestone 18 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `chunk` phase in `nets/transformer.py`, `chunk_us` in `serving.py` · Demo: `examples/16_verify_regimes.py`.*

Milestone 17 shipped with a documented approximation — verification priced
as one decode step — and its blog repeated a piece of industry folklore:
that speculative decoding fades at throughput batch sizes. This milestone
replaces the approximation with a mechanism, and the mechanism corrects
the folklore. It's the M9 lesson again, one level up: honest machinery is
a debugger for your *beliefs*, not just your constants.

## One new step shape

The builder knew two step shapes: prefill (s tokens from scratch) and
decode (1 token against existing KV). Real engines have a third — **the
chunk**: m new tokens per sequence against existing KV. One `chunk=`
parameter on the decode phase prices it (queries m rows, KV = existing +
causal prefix of the chunk), and `StepLatencyModel.chunk_us` memoizes it.
This single shape is speculative verification (m = k+1), and it is also
chunked prefill — two serving techniques, one price.

`SpecStepLatencyModel` now verifies honestly: `chunk_us(batch, kv, k+1)`
instead of `decode_us(batch, kv)`. At b=1 nothing changes (all 20 prior
tests pass untouched) — which is itself the claim from M17, still holding.

## The regime map

Verify-cost ratio and spec-vs-plain effective TPOT, across batch and
context (7B + 1.1B draft, k=4, alpha=0.8, H100):

```
     kv           b=1          b=64         b=256         b=512
    256  1.00|0.62     1.38|0.65     2.00|0.76     2.10|0.77
   2048  1.00|0.60     1.12|0.44     1.19|0.43     1.20|0.42
  16384  1.00|0.53     1.02|0.36     1.03|0.36     1.03|0.36
```

(left of each cell: verify/decode cost; right: spec/plain TPOT — lower
means speculation wins.)

Two regimes fall out:

- **Short context, large batch** (top right): the target's GEMMs are
  math-bound, so k+1 verify rows cost real money — verification hits
  2.1x a decode step and the speculation win erodes toward parity.
  *Here* the folklore is true.
- **Long context** (bottom rows): KV traffic dominates every step, and a
  verify pass reads the same KV as a decode pass — verification stays
  free even at b=512, and speculation *amortizes the target's KV reads*
  over ~3.4 tokens. The win strengthens to 0.36x. At long context,
  speculative decoding is a **throughput** tool, not just an
  interactivity tool.

## Errata for milestone 17

M17's closing section claimed the speculation win "shrinks at large
batch" as a general rule. The honest model shows that is the
short-context special case; at the long contexts where modern serving
actually lives, the win grows with batch. The M17 post now carries a
pointer here. Two milestones in a row, the model has out-argued its
author — which is precisely what it is for.

## In production-scale models

Production engines schedule chunked steps natively (chunked prefill mixed
into decode batches under a token budget), and their speculation
planners choose k and draft size *per regime* — small k at short-context
high-batch, aggressive speculation for long-context traffic. The chunk
price is the primitive under all of it.

## Exercises

1. Chunked prefill in the M13 engine: replace prefill-prioritized steps
   with a per-step token budget mixing chunk + decode; measure TTFT p95
   vs TPOT p50 at high load.
2. Re-run the M16 SLO price curve at kv=16384: how much of the steep
   segment does speculation now erase?
3. Regime-aware k: sweep k per (batch, kv) cell and map the optimal
   speculation depth over the table.
