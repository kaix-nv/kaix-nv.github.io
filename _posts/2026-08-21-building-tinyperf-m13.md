---
layout: post
math: true
title: "Building tinyperf M13: A serving simulator — the analytical model becomes the delay model"
date: 2026-08-21 23:32:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Queues have no closed form: a discrete-event continuous-batching engine consumes tinyperf's step prices — and produces the serving hockey stick of a real, calibrated GPU in 0.35 seconds."
---

*Milestone 13 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tinyperf/serving.py`
· Demo: `examples/11_serving.py`.*

Twelve milestones priced one *step* in isolation. A serving system is what
happens between the steps: requests arrive on their own clock, batches form
and dissolve continuously, the KV cache fills, queues grow. TTFT under
load, the throughput/latency hockey stick, goodput against an SLO — these
are properties of the *schedule*, and no closed form captures a queue. So
this milestone changes paradigm: **discrete-event simulation, with the
analytical model as its delay model.**

## The bridge: a latency table filled by the model

The split is exactly how production tooling works: a steady-state
analytical model prices each step shape once; an event-driven system
simulator replays thousands of steps against those prices. Our bridge is
~30 lines:

```python
class StepLatencyModel:
    def decode_us(self, batch, kv_len):    # kv bucketed to 256
        key = ("decode", batch, bucket(kv_len))
        if key not in self._memo:
            g = build_llm_graph(self.p, "decode", batch, kv, tp, ep)
            fuse_attention(g)
            self._memo[key] = execute(g, device, methodology, calibration).total_us
        return self._memo[key]
```

Every earlier milestone rides through that `execute` call — fusion,
split-K, GQA, capacity, and crucially the **methodology axis**: run the
simulator with `CALIBRATED` and the serving curve carries real measured
constants. The demo below is LLaMA2-7B on the RTX A6000 with its
cuBLAS-fitted calibration — a *real GPU's* serving behavior, from
arithmetic.

## The engine

The loop models classic continuous batching, deliberately simple and
documented: prefill-prioritized scheduling (newly admitted requests
prefill as one batch; otherwise the whole running set takes one decode
step), and reservation-based KV admission (`prompt + gen` tokens must fit
the budget the capacity model derives — weights subtracted from headroom,
divided by KV bytes per token). Mixed chunked-prefill batches, paged KV,
and preemption are exercises, not accidents.

## The hockey stick, from arithmetic

Poisson arrivals, prompt 512 / gen 128, max batch 64, SLO: TPOT <= 60 ms
(`examples/11_serving.py`; anchors: prefill(512) = 74 ms, decode(1) =
25.7 ms, decode(64) = 66.1 ms):

```
 req/s  TTFT p50  p95 (ms)  TPOT p50  p95 (ms)   tok/s  SLO-good %
   0.5        84        99      27.9      31.2      70        100%
   1.0        88       110      31.1      36.3     137        100%
   2.0        92       140      38.6      44.9     263        100%
   3.0        96       176      49.8      57.3     377        100%
   4.0       107       205      66.8      75.3     471         25%
   5.0       127       760      83.0      95.9     537         16%
   6.0       175      3231      88.2      98.4     557         16%
```

Read it bottom-up like a serving operator:

- **Below saturation** every metric sits on its analytical anchor: TTFT
  p50 ~ one prefill, TPOT ~ decode at the ambient batch size. The
  event simulation adds nothing here — steady state is the analytical
  model's home turf.
- **Past ~4 req/s** the phase change: TTFT p95 explodes 205 -> 760 ->
  3231 ms (queueing, invisible to any per-step model), throughput
  plateaus at ~557 tok/s, and **goodput collapses to 16%** while raw
  throughput still climbs. Throughput and goodput part ways at exactly
  the load where an operator must stop trusting throughput.

The entire seven-point sweep — 700 requests, thousands of engine steps —
runs in 0.35 s, because only ~40 distinct step shapes ever get priced;
the memo replays them. That ratio (millions of events, dozens of prices)
is why the analytical/event split scales to datacenter questions.

## In production-scale models

Production serving simulators add what the exercises sketch: chunked
prefill mixed into decode batches, paged KV with fragmentation,
preemption and priority tiers, disaggregated prefill/decode with KV
transfer over the interconnect, multi-replica load balancing, and trace
replay from real traffic. All of it sits on the same two-layer design —
an event engine consuming a priced step table — which is why the
analytical layer's honesty (milestones 9–10) matters twice.

## Exercises

1. Chunked prefill: mix a bounded prefill chunk into each decode step and
   watch TTFT p95 at high load trade against TPOT p50.
2. Paged KV: replace reservation admission with page-granular allocation
   and an eviction/preemption policy; measure admission-rate gains.
3. Disaggregated serving: separate prefill and decode fleets with a KV
   transfer priced by the comm model. Where does the transfer pay for
   itself?
4. Replay a real trace (arrival times + lengths from any public serving
   log) instead of Poisson, and compare the p95s.

*Next — and last on the core roadmap: milestone 14, the sweep harness,
where thousands of these millisecond-priced configs become Pareto
frontiers.*
