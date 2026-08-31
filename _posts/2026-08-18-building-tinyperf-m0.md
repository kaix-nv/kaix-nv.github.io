---
layout: post
math: true
title: "Building tinyperf M0: Why build an analytical GPU performance model?"
date: 2026-08-18 16:05:13 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Why replace execution with arithmetic: the questions analytical GPU performance models answer, a 29-milestone roadmap, and the validation anchors that keep the physics honest."
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

The series so far — each milestone with working code, a runnable example,
and a post:

| # | Milestone | Module | Post |
|---|-----------|--------|------|
| 1 | Device model — a GPU as a bag of numbers | `tinyperf/device.py` | 01 |
| 2 | Analytical GEMM model — tile search, quantization, L2 reuse | `tinyperf/gemm_model.py` | 02 |
| 3 | Graph IR — describing workloads without executing them | `tinyperf/graph.py`, `operators.py` | 03 |
| 4 | Scheduler & reports — from graph to per-layer table | `tinyperf/scheduler.py` | 04 |
| 5 | Modeling an LLM — prefill, decode, KV cache, GQA | `tinyperf/nets/transformer.py` | 05 |
| 6 | Parallelism & communication — TP sharding, ring collectives | `tinyperf/comm_model.py` | 06 |
| 7 | FMHA fusion — price what the kernel library actually runs | `tinyperf/passes.py` | 07 |
| 8 | Memory capacity — stop pricing impossible configs | `tinyperf/capacity.py` | 08 |
| 9 | Methodologies & calibration — SOL/Proj/Calibrated, fitted to a real GPU | `tinyperf/methodology.py`, `tools/` | 09 |
| 10 | Split-K — rescuing launch-starved GEMMs | `tinyperf/gemm_model.py` | 10 |
| 11 | MoE & expert parallelism — routing, grouped GEMMs, all-to-all | `tinyperf/nets/transformer.py` | 11 |
| 12 | Training — backward as a graph pass, pipeline bubbles | `tinyperf/passes.py`, `training.py` | 12 |
| 13 | Serving simulator — event-driven engine on analytical delays | `tinyperf/serving.py` | 13 |
| 14 | Sweep harness — the answer space, Pareto frontiers | `tinyperf/sweep.py` | 14 |
| 15 | Capacity planning — sweeping the serving simulator | `tinyperf/sweep.py` | 15 |
| 16 | The SLO price curve — what interactivity costs | `tinyperf/sweep.py` | 16 |
| 17 | Speculative decoding — the thing that moves walls | `tinyperf/spec_decode.py` | 17 |
| 18 | The chunk step — where verification stops being free | `serving.py`, `nets/transformer.py` | 18 |
| 19 | Chunked prefill — overlap, and the chunk-size knob | `tinyperf/serving.py` | 19 |
| 20 | Paged KV — what optimism about output lengths buys | `tinyperf/serving.py` | 20 |
| 21 | DP & ZeRO — sixteen bytes, and where gradient sync binds | `tinyperf/training.py` | 21 |
| 22 | Datacenter calibration — H100 and B200 on real silicon | `tools/`, `data/calibration/` | 09 (update) |
| 23 | Attention model — accumulator precision, pipeline fill | `gemm_model.py`, `nets/transformer.py` | 23 |
| 24 | Hybrid linear attention — when the cache stops growing | `gemm_model.py`, `nets/transformer.py` | 24 |
| 25 | Linear-attention scaling measured on silicon | `tools/measure_linattn.py` | 24 (update) |
| 26 | The software stack is a calibration axis | `methodology.py`, `tools/measure_launch.py` | 26 |
| 27 | MTP — the draft that comes with the model | `tinyperf/mtp.py` | 27 |
| 28 | How far ahead should you speculate? | `tinyperf/spec_decode.py` | 28 |
| 29 | What a cheap draft does to the price of interactivity | `tinyperf/sweep.py` | 29 |

Everything is stdlib-only Python (~700 lines total) using **public**
datasheet numbers for A100 and H100. The design deliberately mirrors how production-scale analytical models
are architected, at 1/1000th the size:

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

Power/energy modeling and event-driven kernel simulation.
Each is a natural extension once the skeleton is understood — several appear
as exercises at the end of the posts.

## A note on validation

An analytical model is only as good as its anchors. Throughout the series we
sanity-check against numbers you can verify independently: datasheet peak
TFLOPS, first-principles weight-streaming bounds for LLM decode
(`weights_bytes / HBM_bandwidth`), and the standard `2 * params` FLOPs-per-token
estimate. The test suite (`tests/test_core.py`) encodes these anchors so
refactors can't silently bend the physics.
