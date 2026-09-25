---
layout: post
math: true
title: "Building tinyperf M72: The scheduler at KV-cache saturation: vLLM's preemption, priced, and a knee that was a slow GPU"
date: 2026-09-25 12:00:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "When the KV cache fills, vLLM allocates blocks as it goes, over-admits, and preempts its newest request to recompute it later, where tinyperf had reserved every request's full output up front. Priced as vLLM schedules, two held-out sweeps land within every frozen criterion, preemption counts within 9% and peak batch within 3%. And the knee that milestone 57 called bistable turns out, re-run with the step clock, to have been a slow GPU."
---

*Milestone 72 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `kv_paging="paged"` with chunked prefill and `kv_budget_tokens` in `serving.py`, `tools/vllm_patches/step_timing` · Data: `data/validation/comparison_qwen3_8b_rtx_a6000_kv_saturation.txt`, `engine_steps_m72_qwen3_8b_rtx_a6000.json`.*

Every serving sweep so far fit in the KV cache, about 190,000 tokens on
this GPU. When it doesn't fit, a scheduler has to decide who gets the
memory, and tinyperf's chunked engine decided differently from vLLM's.
It reserved each request's prompt plus its maximum output when it
admitted it, so it admitted cautiously and never had to take anything
back. vLLM does neither.

## What vLLM does when the cache fills

From vLLM 0.15.1's scheduler:

- **Running requests first.** Each allocates KV blocks (16 tokens each)
  for the tokens it schedules this step, with no reserve held back.
- **Out of blocks.** When a request cannot get its blocks, the scheduler
  preempts the newest running request. That request frees all its blocks
  and goes back to the front of the queue. When it resumes, it recomputes
  from scratch: its prompt, and every token it had generated.
- **No admissions.** A step that preempted anyone admits no one new.

So vLLM over-admits, and pays for it with the occasional recompute, where
tinyperf had admitted conservatively.

## Building it in

`simulate(kv_paging="paged")` had existed since milestone 20, but only
without chunked prefill ("an exercise", the code said). It now composes
with chunked prefill the way vLLM schedules. With an ample pool it
reproduces the reservation engine's schedule exactly, to the
microsecond. The step-timing patch now records the scheduler's
`preempted_req_ids` and resumed requests, so the engine's own log says
when it preempts.

The pool itself turned out to need care. vLLM prints the size it
allocates at start: 188,944 tokens for these runs at 90% memory
utilization. tinyperf's capacity model says 181,878, for two reasons:

- **Units.** It reads the A6000's "48 GB" as 48 × 10⁹ bytes, where CUDA
  sees 47.53 GiB.
- **Engine overheads.** It knows neither vLLM's profiled activation peak
  nor the sampler's rows.

So `kv_budget_tokens` takes the engine's figure. That figure is a property
of the configuration, readable from a server start before any run.

## In-sample: a small pool

The first sweep shrinks the pool to 50,896 tokens (50% memory
utilization), with 1024-token prompts, 512-token outputs and up to 256
sequences. Nothing was fitted. Measured, then priced by both schedulers:

```
  rate     TTFT p50    TTFT p50 ratio          preemptions        peak decodes
           measured    paged  reservation      engine / model     engine / model / reservation
  1.5/s     10.5 s     1.01      1.15            103 / 100          43 / 43 / 33
  2.5/s     34.0 s     0.99      1.12             95 / 97           45 / 45 / 33
  3.5/s     41.1 s     1.00      1.19            118 / 117          46 / 46 / 33
```

The paged scheduler's TPOT is 1.00–1.01. The reservation scheduler reads
TPOT 0.81–0.85: it keeps the batch at 33, so its steps are too short.

## Frozen, then measured

Two held-out sweeps ran on new seeds, with predictions pushed before
either ran:

- **H1:** the production pool (90%), prompts of 1038–3069 tokens and
  outputs of 517–1533, at 0.75, 1.25 and 2 req/s.
- **H2:** the small pool (50%), prompts of 54–973 tokens and outputs of
  79–1452, at 2, 3 and 5 req/s.

The criteria added the schedule itself: preemptions within 15%, peak
decodes within 5%, step counts within 3%.

```
  predicted / measured    TTFT p50   TTFT p95   TPOT    preemptions      peak decodes
                                                        engine / model   engine / model
  H1  0.75 req/s            1.00       1.00     1.01       0 / 0           68 / 70
  H1  1.25                  1.00       1.00     1.00     169 / 155         80 / 80
  H1  2                     1.01       1.02     1.02     168 / 172         86 / 87
  H2  2                     0.96       0.98     0.98     237 / 236         61 / 61
  H2  3                     0.98       0.98     0.99     235 / 238         63 / 64
  H2  5                     0.97       0.98     0.99     248 / 254         74 / 73
```

All six cells meet every criterion, and step counts agree within 0.3%.
At 1.25 req/s on the production pool the median request waits 2 seconds
for its first token, but the 95th percentile waits two minutes. The
model predicted 118.9 s and the engine measured 118.5.

Under the old reservation scheduler, the cells where the pool binds read
TTFT p50 1.4–16.5× and TPOT 0.68–0.73. It never over-admits, so its
batches peak at 37–63 where the engine's reach 61–86.

## The knee that was a slow GPU

Milestone 70 left one question open. On M57's knee sweep, the server had
queued worse at 3.5 and 3.75 req/s than at 4.0, on the same arrivals.
M57 had called it bistability, and nothing in the model could reproduce
it. I re-ran that sweep on the same trace, this time with the step clock
and GPU telemetry, on a GPU nothing else was using:

```
  rate    TTFT p50: re-run  (model)   M57's run     TPOT: re-run   M57's run
  3.25        360 ms  (1.09)            415 ms          64.7         84.6
  3.5         437 ms  (1.00)           4602 ms          80.0        133.4
  3.75        596 ms  (0.98)           2654 ms          92.9        121.1
  4.0        1146 ms  (1.20)           1144 ms         107.1        108.3
```

The queue grows with the rate, and the model is on it. In M57's run the
steps themselves were 30–67% slower at those rates. That was the GPU
running slow, as in M69's throttled run and M71's shared GPU. Neither
the scheduler nor the arrivals were the cause.

## Errata

- **M57:** its 3.5–3.75 req/s cells were a slowed GPU, not a bistable
  scheduler; its tipping fractions over random traces stand as a
  statement about the process.

## Exercises

1. Fix the capacity model: memory in GiB, the engine's activation
   profile, its memory-utilization setting. Does it reproduce 50,896 and
   188,944?
2. With prefix caching on, vLLM keeps a preempted request's full blocks
   cached, so part of its recompute becomes a cache hit. Compose paged
   KV with prefix caching.
3. At 1.25 req/s the production pool's TTFT has a two-minute tail
   behind a two-second median. Which requests are in it, and what does
   preemption have to do with it?
