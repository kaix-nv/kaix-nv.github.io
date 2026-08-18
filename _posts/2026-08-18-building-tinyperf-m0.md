---
layout: post
math: true
title: "Building tinyperf M0: Why build an analytical GPU performance model?"
date: 2026-08-18 16:05:13 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Why replace execution with arithmetic: the questions analytical GPU performance models answer, an eight-milestone roadmap, and the validation anchors that keep the physics honest."
---

*Milestone 0 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf).*

## The question these models answer

Long before a GPU exists — and long after, when you want to sweep a million
configurations — you need to answer questions like:

- How fast will this LLM decode on next year's GPU?
- Is this workload math-bound or memory-bound? What if we double HBM bandwidth?
- What tensor-parallel degree minimizes latency for a 70B model at batch 32?

Cycle-accurate simulators answer these at ~kHz simulated speed; running a
real workload needs the silicon you don't have yet. An **analytical model**
answers in **milliseconds** by replacing execution with arithmetic: describe
the workload as math (FLOPs, bytes, shapes), describe the machine as a
handful of rates and capacities (TFLOPS, GB/s, cache sizes), and compute
their intersection carefully.

"Carefully" is the whole game. The naive roofline — `time = max(flops/peak,
bytes/bw)` — is off by 2–10x on real kernels because it ignores tiling,
cache reuse, wave quantization, launch overheads, and communication. The
craft of an analytical model is adding exactly the second-order effects that
matter, and no more.

## What we are building

Eight milestones, each with working code, a runnable example, a post, and
its own pull request in the repo:

| # | Milestone | Module |
|---|-----------|--------|
| 1 | Device model — a GPU as a bag of numbers | `tinyperf/device.py` |
| 2 | Analytical GEMM model — tile search, quantization, L2 reuse | `tinyperf/gemm_model.py` |
| 3 | Graph IR — describing workloads without executing them | `tinyperf/graph.py`, `operators.py` |
| 4 | Scheduler & reports — from graph to per-layer table | `tinyperf/scheduler.py` |
| 5 | Modeling an LLM — prefill, decode, KV cache, GQA | `tinyperf/nets/transformer.py` |
| 6 | Parallelism & communication — TP sharding, ring collectives | `tinyperf/comm_model.py` |
| 7 | FMHA fusion — price what the kernel library actually runs | `tinyperf/passes.py` |
| 8 | Memory capacity — stop pricing impossible configs | `tinyperf/capacity.py` |

Everything is stdlib-only Python (~700 lines total) using **public**
datasheet numbers for A100 and H100. The design deliberately mirrors how
production-scale analytical models are architected, at 1/1000th the size:

```
tinyperf                          production-scale analytical models
--------                          ----------------------------------
Graph/Operator/Tensor      <->    full graph IR with loaders (ONNX, ...)
nets/transformer.py        <->    declarative model zoo over a block builder
device.py + JSON           <->    layered product/architecture device specs
gemm_model.py              <->    curated per-arch kernel-config databases
scheduler.py dispatch      <->    layer-model factory per op type
comm_model.py              <->    topology-aware collective models
RunReport                  <->    report pipelines, databases, dashboards
```

## What we deliberately leave out

Power/energy modeling, training back-propagation passes, event-driven
kernel simulation, silicon calibration, pipeline-parallel schedules, and
batch-sweep infrastructure. Each is a natural extension once the skeleton
is understood — several appear as exercises at the end of the posts.

## A note on validation

An analytical model is only as good as its anchors. Throughout the series we
sanity-check against numbers you can verify independently: datasheet peak
TFLOPS, first-principles weight-streaming bounds for LLM decode
(`weights_bytes / HBM_bandwidth`), and the standard `2 * params`
FLOPs-per-token estimate. The test suite (`tests/test_core.py`) encodes
these anchors so refactors can't silently bend the physics — and a taste of
what they pin down: LLaMA2-7B decode on A100 lands at ~115 tok/s against a
6.6 ms/token weight-streaming bound, and LLaMA3-70B at batch 32 turns out
to need TP≥4 on 80 GB GPUs not for speed, but because the KV cache doesn't
fit at TP-2. Numbers like these — derived, not measured — are the entire
point of the exercise.
