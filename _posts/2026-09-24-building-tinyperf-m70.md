---
layout: post
math: true
title: "Building tinyperf M70: The scheduler at saturation: the miss was the trace, and three step mechanisms the engine's log exposed"
date: 2026-09-24 12:00:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Milestone 69 left the steps right and the queue at saturation wrong. Replayed on the requests the benchmark actually sent, the miss flipped sign: vllm bench serve's trace is deterministic in its seed, and near saturation one run's queue is set by its arrivals. The engine's step log then exposed three step mechanisms. Frozen on two new seeds, 13 of 14 cells land within the stated criteria, and at the knee two traces measured 1.54x apart are predicted 1.53x apart."
---

*Milestone 70 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `bench_requests`, the step `trace` of `simulate` and `decode_us(..., real=)` in `serving.py`, `tools/bench_trace.py` · Data: `data/validation/comparison_qwen3_8b_rtx_a6000_scheduler_saturation.txt`, `engine_steps_m70_qwen3_8b_rtx_a6000.json`, `bench_traces_m70.json`.*

Milestone 69 ended with the steps right and the queue wrong. On a
long-prompt sweep at 2–2.5 req/s, every step was priced within 1–2% at the
composition the engine recorded. TPOT, however, read 0.62–1.00 of
measured and TTFT 0.37–0.58. This milestone set out to compare the model's
schedule with the engine's, step by step.

## Same requests, opposite miss

A step-by-step comparison needs the same requests on both sides.
`vllm bench serve` records each request's send time and lengths, so the
run can be replayed through `simulate`. On the run's own requests, the
M69 model missed the other way: TTFT p50 1.48 at 2 req/s and 1.05 at 2.5,
TPOT 1.05–1.07. It was high, not low.

The difference was the requests themselves. M69's predictions, like every
online prediction since milestone 54, averaged eight Poisson traces drawn
by tinyperf. The benchmark sends one particular trace, and it is
deterministic in its seed:

- **Lengths.** Prompt and output lengths come from numpy's `default_rng(seed)`.
- **Arrivals.** Send times are gamma draws from numpy's global generator,
  rescaled so the last request goes out at exactly n/rate. So every rate
  sends the same draws, compressed.

`tools/bench_trace.py` rebuilds that trace from the run's arguments. On
M69's two sweeps it reproduces all 2,480 recorded requests: every prompt
length, every completed request's output length, and send times within
14 ms. `serving.bench_requests` prices it.

How much does the particular trace matter? At the M54 shape's knee
(4 req/s), other traces put the model's TTFT p50 anywhere from 0.39 to
2.26 of the measured run. The long sweep's seed-0 trace is a heavy one:
all 16 other traces predict 0.39–0.81 of its TTFT at 2 req/s. Near
saturation a run's queue is set by its arrivals, so a prediction of that
run has to be made on them.

## What the step log showed

With the trace fixed, the model was still 5–8% high on TPOT at 1–2 req/s.
The engine's step log (M67's `step_timing`, now read step by step against
a new `trace` from `simulate`) located three mechanisms.

**Attention in a padded graph.** vLLM pads a decode step to a CUDA-graph
size: 9 sequences run in the 16-row graph. The GEMMs run every padded
row, but a padded slot holds no KV, so attention reads only the real
sequences' caches. The model had priced attention for all 16. Over M69's
logs, the decode forward at batch 3–32 read 1.02–1.05 of measured; with
attention at the real batch it reads 0.99–1.02.

**A step of prefill chunks alone.** M69 priced a mixed step as one
forward, with the LM head only at the rows that sample. A step with no
decodes still went through the old chunk graph, which ran the LM head on
every chunk token, 22 ms too much at 2048 tokens. The first 2048-token
chunk of the replay was priced at 317 ms; the engine took 289; one
forward gives 292. Over 93 such steps the median moves from 1.08 to 1.00.

**The token budget.** vLLM schedules its running requests first, and a
decode spends a token of the step's 2048. A prompt arriving beside one
decoding request got a 2047-token chunk. Over 2,596 steps that carried
prefill, decodes plus prompt tokens never exceeded 2048. The model had
given prompts 2048 on top of the decodes.

A fourth effect needs no change. The engine sees a request 20–25 ms
after the client sends it, the frontend's intake. The model adds M54's
25 ms after scheduling, but a delay the same for every request shifts
every arrival alike, so where it is counted doesn't matter.

With the three mechanisms in place, the replayed schedule matches the
engine's. At 2 req/s the model takes 877 steps to the engine's 883, 203
of them carrying prefill (206), with 23.8 decodes per step (23.6). M69's
long sweep reads TTFT p50 0.93–1.05, p95 0.93–1.08 and TPOT 0.97–1.02 at
every rate.

## Every sweep, on its own trace

All online sweeps since milestone 54 ran with seed 0, and every one is now
priced on its own trace. On one GPU (M57's 3.25–3.75 req/s cells apart,
below), TPOT is 0.96–1.04 and throughput 0.99–1.01; TTFT p50 away from
the knee is 0.87–1.14. On two GPUs, the colocated pair reads TTFT p50
0.99–1.05 and TPOT 1.00–1.02 at every rate.
The pair's knee gap at 6–8 req/s, there since milestone 66, is gone.

The M54 knee itself reads 1.08–1.30 on TTFT p50 across its four runs at
4 req/s, and that is as close as a point prediction gets there. Against
M69's re-run (1.19), pricing every step 1% faster gives 0.91, 1% slower
1.35. At the knee, TTFT is a measurement of the step time to better than
1%, so the prediction is stated as a band.

Two sweeps do not come right:

- **M57's knee sweep at 3.25–3.75 req/s.** The real server queued
  worse there than at 4.0 on the same arrivals, rescaled: TTFT p50 4.6 s
  and 2.7 s at 3.5 and 3.75, against 1.1 s. No step price reproduces a
  queue that grows as the rate falls. Even with every step 6% slower the model stays on
  the quiet branch at 3.5. That run logged no step clock or GPU
  telemetry, so the cause is open.
- **1P1D above 3 req/s.** TTFT p50 reads 0.58–0.87. Its agreement had
  leaned on the chunk graph's overcharge, since every prefill-server step
  is a step of chunks alone. vLLM's own metrics give a lone 1024-token
  request 194 ms of prefill on that server, against ~150 ms for the
  plain step: the KV connector's per-layer work, which is not priced.

## Frozen, then measured

The predictions were committed and pushed before any run. They cover
three held-out sweeps on traces nobody had run, seeds 1 and 2, chosen
before any prediction was made:

- **Long prompts:** the long-prompt shape at 1–2.5 req/s.
- **M54 shape, twice:** 3–5 req/s on seed 1, then the same rates on seed 2.

The criteria were written into the file first:

- **Off the knee:** TTFT p50 and p95 within ±15% and TPOT within ±5%.
- **At the knee:** the measured TTFT p50 inside the ±1%-step band.
- **The schedule:** step counts and mean decodes within 3%.

Beside M70 the file carries the pre-M70 model on the same traces, and
M69's method (eight other traces).

```
  predicted / measured          TTFT p50     TTFT p95     TPOT
  long, seed 1, 1-2.5 req/s     0.97-1.09    0.98-1.17    0.97-1.03
  M54 shape, seed 1, 3-5        1.01-1.08    1.00-1.12    1.01-1.03
  M54 shape, seed 2, 3-5        1.01-1.09    0.99-1.07    1.01-1.02
  pre-M70, same traces          0.84-1.26                 1.01-1.06
  M69's method                  0.71-1.26                 0.86-1.03
```

Throughput is within 0.99–1.01 everywhere. The knee shows what the trace
does. At 4 req/s the two M54-shape runs measured TTFT p50 660 ms and
1015 ms, at the same rate and prompt length. The frozen predictions were
679 and 1042, and both measurements fall inside their bands.

Thirteen of fourteen cells meet the criteria. The miss is the long sweep
at 2.5 req/s: TTFT p95 1.17, with p50 1.09 and TPOT 1.03.

The engine's schedule matches the model's: step counts 0.98–1.04,
prefill-carrying steps 0.98–1.01, mean decodes 0.96–1.02. Thirteen of
fourteen cells are within 3%; the long sweep at 2 req/s is at 4%. My
first count read 0.81–0.92 on the M54 shape. That was my slip, not the
model's: each rate's window also holds vllm bench's warm-up request,
about 128 one-decode steps. Counted over the benchmark's own requests,
the two agree.

The GPU sat at its power cap throughout (293–298 W, SM clock
1500–1710 MHz). The neighbouring GPU carried another job during part of
the seed-2 sweep. Pure decode steps stayed within 0.98–1.04 of the model
in every 30-second window.

## Errata

- **M69:** the saturated rates' miss was the trace, not the queue.
- **M57:** its knee sweep sent the same arrivals at every rate; the 3.5
  and 3.75 req/s cells were not two draws from a distribution.
- **M66:** 1P1D's TTFT agreement at 4–8 req/s leaned on the chunk
  graph's LM-head overcharge.
- **M68:** a decode step between graph sizes was priced with attention
  for its padded slots.

## Exercises

1. Put the engine clock on the KV producer: run the 1P1D prefill server
   with `step_timing`. Is the connector's cost per step, per layer or per
   request, and does pricing it bring 1P1D back above 3 req/s?
2. Re-run M57's knee sweep with the step clock and telemetry. On the same
   trace, does 3.5 req/s tip again?
3. The frontend's intake ranges 15–32 ms on an idle server. Draw it per
   request: does the knee's band move?
