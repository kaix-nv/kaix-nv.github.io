---
layout: post
math: true
title: "Building tinyperf M37: Disaggregated prefill/decode — the regime map"
date: 2026-08-31 15:37:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Same four GPUs, two designs: the TPOT-jitter collapse is structural, the saturation win is 4x on TTFT tails, and below saturation colocation wins — the regime map, priced."
---

*Milestone 37 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tinyperf/disagg.py` · Demo: `examples/26_disagg.py`.*

Milestone 13's engine colocates prefill and decode, and every milestone
since has managed the consequence: admitted prompts steal steps from
decoding requests. Chunked prefill (M19) softened the interference;
disaggregation removes it structurally. A prefill pool computes prompts,
ships each request's KV cache across the milestone-34 fabric, and a
decode pool runs continuous batching that never sees a prefill.

The mechanism is ~90 lines because everything it needs already existed:
the step prices come from `StepLatencyModel`, the KV sizes from the
capacity model, the transfer from the fabric fields, and recipes (M35/36)
flow through untouched. `kv_transfer_us` refuses on a device with no
fabric — disaggregation without an interconnect is not a deployment.

## Same four GPUs, two designs

Qwen3-8B on B200s, prompt 4096 / gen 256 (KV transfer: 6 ms/request):

```
  rate           config  TTFT p95  TPOT p50  TPOT p95  jitter   tok/s
    12   colocated tp=4        44      4.21      5.04    1.20    3042
    12     disagg 2P+2D        67      3.78      4.04    1.07    2987

    40   colocated tp=4      1115      7.24      8.67    1.20    5967
    40     disagg 3P+1D        98      9.56     10.21    1.07    4803
    40     disagg 2P+2D       269      5.41      5.76    1.06    6645
```

Three regimes, one table.

**Below saturation, disagg is not a free lunch.** Colocation wins the
TTFT tail at 12/s — no transfer hop, and all four GPUs can prefill when
a burst lands. Anyone selling disaggregation as unconditionally better
is reading only the bottom half of the table.

**The jitter collapse is structural.** Disagg TPOT p95/p50 sits at ~1.06
at every load; colocation runs 1.2-1.4. Decode cannot be interrupted by
a prefill because there is no prefill to interrupt it. If the SLO is on
the tail — and TPOT SLOs usually are — this column alone justifies the
architecture.

**At saturation the colocated design collapses and disagg does not.**
TTFT p95 over a second versus 269 ms, 25% faster p50 decode, 11% more
throughput — same four GPUs, different wiring.

**And the split is a real decision.** 3P+1D and 2P+2D differ by 40%
throughput at the same fleet size. The P:D ratio is now a knob the model
prices instead of folklore.

## Exercises

1. Sweep the P:D ratio against prompt/gen mix — derive the optimal split
   as a function of the prefill:decode FLOP ratio and check it against
   the simulation.
2. KV transfer is priced point-to-point; production systems layer it
   (NVLink inside the domain, IB across). Add domain-aware transfer.
3. Compose with M35 recipes: fp8 KV halves the transfer. When does that
   move the P:D optimum?
