---
layout: post
math: true
title: "Building tinyperf M54: The engine under load: serving dynamics on silicon"
date: 2026-09-08 19:21:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Every earlier measurement timed a step. This one drives a live server with Poisson traffic and asks the continuous-batching engine where the knee is. Two corrections a step could never expose, one 25 ms online constant, and a knee within one sweep step."
---

*Milestone 54 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tools/measure_serving.py`, `mixed_step_us` in `serving.py`, `online_overhead_us` in `methodology.py`.*

Every silicon measurement in this series so far timed a *step*: one
prefill, one decode, a fixed batch, a known cache. A deployment is not a
step. It is a queue — requests arrive at random, wait for the running
batch, get scheduled with whatever else is there — and the numbers people
actually buy against, the TTFT tail and the goodput knee, come out of
that scheduling. Milestone 13 built the engine that models it and
milestones 15, 16 and 29 priced whole deployments on top of it, and none
of it had ever met a real server. This milestone drives one.

## The protocol

Qwen3-8B on this box's A6000 behind vLLM's OpenAI server (`max_num_seqs`
64, prefix caching off), loaded by vLLM's own generator: Poisson arrivals
at 1, 2, 3, 4, 5, 6 and 8 requests per second, 240 requests per rate,
1024-token prompts and 128-token outputs, `ignore_eos`. The generator
reports TTFT, TPOT and throughput percentiles per rate. tinyperf ran the
same traces through `simulate` under both of its schedulers — the
prefill-prioritized one of milestone 13 and the chunked one of milestone
19 — with predictions frozen and committed before the server started.

## What the frozen model got right, and wrong

Right: TPOT within 3% and throughput within 2% below the knee, and the
shape of the curve — a flat TTFT floor, a knee, a wall. Wrong, in three
ways worth separating.

**The knee came one step early.** Measured, TTFT p95 leaves its floor
between 3 and 4 requests per second; the frozen chunked model said 5, and
at saturation it had the server delivering 20% more tokens per second
than it did. The cause was in the mixed step. Milestone 19 priced a
chunked-prefill step as `max(decode, chunk)`, the perfect-overlap bound,
and said so. A real mixed step runs both row sets through the same GEMMs
— so the weights are streamed once, which is the real overlap — and then
runs both attentions and everything else. The honest price is the sum
less one weight pass, never less than the larger part. With that (and
letting several waiting prompts share the token budget in one step, as
the real scheduler does), saturation throughput lands 10% high instead of
20%, and the knee moves to within one sweep step of the measured one.
Milestone 19's throughput win for chunked prefill shrinks from 1.3× to
1.07× under this pricing; its tail-latency win stands; the post carries
an erratum.

**The server's budget was not what I assumed.** The frozen chunked
prediction used an 8192-token budget; the server log shows vLLM defaulted
to 2048 on this GPU. An input error, not a model error, disclosed as one
— the difference is small at this workload, and the comparison uses 2048.

**Every online request costs ~25 ms the offline steps never saw.** Below
the knee the measured TTFT medians ran 25% above the model even after the
fixes above. Two probes against an idle server (one request every five
seconds, so nothing to queue behind) measured it directly: a 1024-token
prompt takes 176 ms online against 146 ms for the offline step, a
128-token prompt 48 ms against 28. HTTP, tokenization, waiting for the
step boundary — an engine constant of the same kind as milestone 39's
prefill overhead and milestone 45's pipeline step cost. `online_overhead_us`
is 25 ms on this stack, measured apart from the sweep, and rides on every
request's TTFT in the engine and nowhere else: the offline envelopes are
untouched.

## The table

```
  rate   TTFT p50 meas  model   r     TTFT p95 meas  model   r     TPOT meas  model   r    tok/s meas  model   r
     1          221.7   198.0  0.89          366.6  337.9  0.92        30.56   29.56 0.97         126    128 1.01
     2          237.9   203.7  0.86          501.2  446.0  0.89        39.52   38.38 0.97         248    252 1.02
     3          328.2   228.8  0.70          769.1  588.5  0.77        59.45   53.18 0.89         364    372 1.02
     4         1063.3   371.3  0.35         2693.2  918.4  0.34       106.36   83.71 0.79         456    485 1.06
     5         5967.5  3198.3  0.54        12249.5 7886.6  0.64       112.86   97.51 0.86         468    517 1.11
     6         8341.7  5832.1  0.70        18671.3 14364.2 0.77       113.19  100.49 0.89         478    529 1.11
     8        11074.7  9262.0  0.84        26357.6 22351.1 0.85       111.42  101.31 0.91         493    541 1.10
```

Below the knee the engine is right to within 15% on the tail and 3% on
the steady state. At the knee it is optimistic: at 4 requests per second
the server has already tipped over and the model has not. Past it, the
server saturates with 10% less throughput and 10% longer steps than the
model, and the gap has a shape — 1 ms per step at four running sequences,
6 ms at ~twenty, 10 ms at sixty-four — that says *per-sequence per-step
CPU cost*: scheduling, sampling, detokenization, all of which scale with
the batch and none of which the graph prices. About 0.15 ms per sequence
per step on this stack. It is the next constant, and it is not fitted
here because the only data that would fit it is the data it would then
be checked against.

## What this validates

The engine's *mechanics*: continuous batching, admission against a KV
budget, a knee where prefill demand saturates, and the TTFT tail's shape
under Poisson load — on one workload shape, one GPU, one engine. And two
corrections to the model that a step measurement could never have
exposed, because they are about what happens *between* steps.

## Not validated

Other workload shapes (long prompts, short outputs, mixed lengths), the
paged allocator's preemption behaviour under memory pressure, the
disaggregated engine of milestone 37, multi-GPU serving dynamics, and any
engine but this one. The knee's position is within one sweep step; a
finer sweep between 3 and 5 would pin it better.

## Exercises

1. Sweep 3.0 to 4.5 requests per second in steps of 0.25 and locate the
   knee to a tenth. Does the model's knee move with `chunk_tokens`?
2. Fit the per-sequence per-step cost from a *different* workload (short
   prompts, long outputs) and check it against this sweep.
3. Run the same protocol with prefix caching on and a shared system
   prompt (milestone 53). Does the knee move as far as the step model
   says it should?
