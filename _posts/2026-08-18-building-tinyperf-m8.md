---
layout: post
math: true
title: "Building tinyperf M8: The memory capacity model — stop pricing impossible configs"
date: 2026-08-18 16:41:55 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Closed-form weights, KV-cache, and activation footprints add the fits? column — and reveal that 70B decode needs TP≥4 on 80 GB GPUs."
---

*Milestone 8 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tinyperf/capacity.py`
· Demo: `examples/06_fmha_capacity.py`.*

Milestone 7 fixed the first way the model lied to you; this one fixes the
second. Milestone 6 happily "ran" 141 GB of LLaMA3-70B weights on an 80 GB
H100. Capacity is often the *binding* constraint, and it's closed-form
(`capacity.py`):

```
weights/GPU = sharded params / tp + unsharded embed & LM head
KV/GPU      = 2 * layers * local_kv_heads * head_dim * bytes * batch * context
activations = 2 * tokens * widest intermediate    (small for decode)
fits        = total <= 0.90 * HBM                 (headroom stated, not hidden)
```

The TP table from milestone 6 now carries the truth column:

```
tp   ms/token   tok/s   mem/GPU GB   fits?
 1      67.78     472        184.1     NO
 2      36.33     881         94.1     NO    <- weights fit, KV doesn't!
 4      20.98    1525         49.2    yes
 8      14.35    2230         26.7    yes
```

The tp=2 row is the interesting one: 72.7 GB of weights looks like it fits
in 80 GB — but against the 90%-headroom budget of 72 GB, the weights
*alone* are already over, before a single KV byte. `max_batch` returns 0:
TP-4 is the floor for this model on 80 GB parts, and no batch tuning
rescues TP-2. And the derived table every serving engineer actually wants,
`max_batch(model, device, context)`:

```
context   7b@A100 tp1   70b@H100 tp4   70b@H100 tp8
   2048            54            199            600
   8192            13             49            150
  32768             3             12             37
```

Halve the context, double the batch — KV cache *is* the capacity story.
Cross-reference milestone 5's throughput curve (still rising at batch 128
for 2k context, which capacity confirms is reachable) and you have the
complete speed-vs-fit picture from two closed-form models.

> **Erratum (milestone 74).** The embedding and LM head were counted whole
> on every tensor-parallel rank. They are vocabulary-parallel, 1/tp per
> rank, as vLLM shards them. LLaMA3-70B's weights are 70.6 / 35.3 / 17.6 GB
> per GPU at tp=2/4/8 (the memory column: 92.0 / 46.0 / 23.0), so at tp=2
> the weights alone sit 1.4 GB under the 72 GB budget and `max_batch`
> returns 2 at 4k context, not 0. TP-4 is still the floor in practice. The
> 70B columns of the second table become 218 / 54 / 13 (tp4) and
> 645 / 161 / 40 (tp8) ([milestone 74]({% post_url 2026-09-25-building-tinyperf-m74 %})).

## In production-scale models

Production allocators don't estimate — they *walk the graph* and simulate
alloc/free against tensor lifetimes to find true peak memory, including
activation checkpointing and weight offloading, and emit per-component
memory-requirement reports that gate which batch-sweep configs even launch
(OOM filtering).

## Exercises

1. Add paged KV (block size 16/128) and re-derive max_batch with
   fragmentation.
2. Add weight quantization (INT8/FP4 storage, fp16 math) to both the speed
   and capacity models — watch max_batch and decode latency improve
   together.
3. Model prefill activation memory at long context — at what sequence
   length does it rival the KV cache?

*That closes the core series: ~700 lines of stdlib Python that price LLMs
on GPUs from first principles. The exercises across these posts (split-K,
MoE, calibration, an event-driven serving layer) are the map for what comes
next.*
