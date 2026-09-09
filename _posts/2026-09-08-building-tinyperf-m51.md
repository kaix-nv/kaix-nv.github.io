---
layout: post
math: true
title: "Building tinyperf M51: Offloading: fitting is not the same as running"
date: 2026-09-08 16:50:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Half of a 70B model's weights on the host lets it fit one H100 and makes every token take a second. A quarter of an active 128k cache on the host multiplies the step by 5.7. The host link priced like any other resource."
---

*Milestone 51 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `offload_weights` / `offload_kv` on `llm_memory`, `StepLatencyModel`, `simulate` · Example: `examples/34_offloading.py`.*

Two things get pushed to host memory in practice. Weights, when a model
does not fit the GPUs you have; and the KV cache, when contexts are long
and the GPU is full of it. Both come back over the host link — PCIe, at a
public per-direction rate the device files now carry — and the model
prices that link like any other resource: streamed every step, hidden
under compute if the engine overlaps it, added if it does not.

## The mechanism

`offload_weights` and `offload_kv` are fractions. Capacity moves the
offloaded share to a `host_gb` column that does not count against the
GPU, so `fits` and `max_batch` see the room it frees. The step model
streams the offloaded weight share on every step — decode or prefill,
the weights are needed either way — and the offloaded share of whatever
cache the step touches: decode reads it, prefill writes it. A device with
no host path (`host_bw_gbps = 0`) refuses, as the fabric-less devices
refuse to span nodes.

## Weights: one H100, a 70B model

LLaMA3-70B in bf16 is 141 GB of weights and an H100 has 80 GB. Offload
half and it fits:

```
  weights on host   GPU GB   host GB   fits   TPOT b=1   TTFT 2k
              0.0      141         0  False          -         -
              0.5       71        71   True     1102ms    1102ms
              0.6       56        85   True     1323ms    1323ms
              0.8       28       113   True     1764ms    1764ms
```

And every decode step now moves 71 GB across a 64 GB/s link, so a token
takes 1.1 seconds. A second GPU (`tp=2`) makes the same model run at
31 ms per token. That is the whole story of weight offloading in one
ratio: it turns a model that cannot run into one that runs 35× slower
than it would on the hardware it needs. There are workloads for which
that is the right trade — offline batch generation with no latency
target, a laptop, a development box — and the number is now available
for making the trade knowingly. The TTFT column equals the TPOT column
for the same reason: the transfer dominates both, and 2048 tokens of
compute vanish under it.

## KV: the cache you offload comes back

The more tempting target is the cache. At 128k context and tp8 a
LLaMA3-70B batch of eight is 43 GB of KV per rank; offloading a quarter
of it would let twelve sequences fit instead of nine:

```
  KV on host   max batch   TPOT b=8   link ms   bound
        0.00           9       29.5       0.0   compute
        0.25          12      167.8     167.8   link
        0.50          18      335.5     335.5   link
        0.75          37      503.3     503.3   link
```

Offloading a quarter of the *active* cache multiplies the step by 5.7,
because every step reads all of it back: 10.7 GB per rank, 168 ms on the
link, against a 30 ms step. No fraction of an active cache is cheap on
PCIe. What engines actually offload is the cache of sequences that are
not currently decoding — paused sessions, preempted requests, a system
prompt that will be reused — and that costs nothing per step because the
step never touches it. That is a scheduler-level decision the serving
simulator of milestone 13 could make and the step model cannot see; the
fraction here is the honest price of the naive version, and the exercise
below is the useful one.

## What it composes with

The fractions thread through the precision and sparsity recipes (the
offloaded share is of the recipe's bytes), pipeline stages (a stage
streams its own share), attention DP and context parallelism (a rank
streams its own cache), the serving simulator's KV pool, and the sweep
pricer. `offload_overlap=False` gives the serial bound for an engine that
does not prefetch.

## Not modeled

DMA latency (only bandwidth and one launch); host memory capacity (the
host is assumed to have room); contention when several GPUs share a host
bridge — eight GPUs on one CPU do not each get 64 GB/s; and the smarter
policies above: layer-wise weight prefetch that hides under the previous
layer's compute (which is what makes weight streaming tolerable at large
batch), and idle-session KV eviction.

## Exercises

1. Give the serving simulator a host tier: evict the KV of requests that
   have waited more than N steps and charge their restore on re-admission.
   At what arrival rate does it start to pay?
2. Weight streaming at batch 256: the compute per step grows, the
   transfer does not. Find the batch at which the 0.5-offload TPOT is
   within 2x of tp2.
3. Add a `host_bw_shared_by` device field for GPUs behind one bridge and
   redo the KV table for eight GPUs on a two-socket host.
