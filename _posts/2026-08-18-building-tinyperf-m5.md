---
layout: post
math: true
title: "Building tinyperf M5: Modeling an LLM — prefill, decode, and the KV cache"
date: 2026-08-18 16:47:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Eight integers and a builder reproduce the economics of LLM serving: math-bound prefill, bandwidth-bound decode, and the KV-cache crossover."
---

*Milestone 5 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tinyperf/nets/transformer.py` · Demo: `examples/04_llm.py`.*

## A model is a parameter bag plus a builder

An LLM, to a performance model, is ~8 integers:

```python
@dataclass
class TransformerParams:
    hidden; n_layers; n_heads; n_kv_heads; ffn_hidden; vocab; dtype
```

`build_llm_graph(params, phase, batch, seq_len, tp)` emits the ops of one
decoder layer with per-GPU shapes, replicated `n_layers` times via the
`count` attribute, plus embedding and LM head. Presets (`llama2_7b()`,
`llama3_70b()`) are one-liners. Production LLM builders work the same way:
every model from GPT-3 to DeepSeek is a small declarative delta on a shared
block builder — new model support means writing a *config*, not a model.

Derived properties give free sanity checks before any simulation:

```
llama2-7b: 6.74B params, 13.5 GB fp16, KV cache 0.524 MB/token
```

and the test suite pins prefill FLOPs/token to the classic `2 x params`
estimate (we measure 13.8 GFLOP/token ≈ 2 x 6.74B ✓).

## Prefill and decode are different workloads

Same weights, same graph builder — but the shapes put them on opposite
sides of the ridge point (milestone 1):

**Prefill** (b=1, s=2048, A100): every GEMM has M = 2048, so arithmetic
intensity is high. Result: 108.7 ms, **90% of time in GEMMs, with the
projection GEMMs running at ~280–305 TFLOPS** — math-bound, ~83% MFU
(optimistic vs the ~50–65% real frameworks achieve; we model no framework
overhead — that's what a calibrated methodology corrects in production
tools).

**Decode** (b=1, kv=2048): every GEMM has M = 1. Each token streams all
13.5 GB of weights: the pure bandwidth bound is 6.6 ms/token, we predict
8.68 ms (launch overheads + KV reads) → **115 tok/s**, squarely in the
measured range for unoptimized fp16 7B serving on A100.

The batch sweep is the money table:

```
batch   ms/token   tok/s
    1       8.68     115
    8      12.45     642
   32      25.98    1232
  128      82.28    1556
```

Weights are streamed once per step *regardless of batch*, so throughput
scales steeply at first — then per-request KV-cache reads (0.524 MB/token x
kv_len x batch) take over as the dominant traffic. That crossover *is* the
economics of LLM serving, reproduced by a 700-line model.

## Two modeling tricks worth stealing

**GQA via batching.** Attention GEMMs are batched per *KV head*, with
M = (query heads sharing that KV head):

```python
g.BatchedMatMul("attn_qk", q, batch=b * n_kv_heads, m=group * s, n=kv, k=head_dim)
```

KV bytes are thus counted once per KV head, not once per query head — which
is the entire point of GQA, captured in shape arithmetic. LLaMA3-70B has
`group=8`, so its KV traffic is 8x smaller than an MHA equivalent; the model
gets that for free.

**Causal masking as shape arithmetic.** Prefill attention only computes the
lower triangle, so we set the effective KV length to `(s+1)/2` — half the
math and half the score traffic, no masks needed.

The KV cache itself never appears as a produced tensor: decode's QK^T
matmul takes explicit `n=kv_len`, and the KV read traffic rides in as the
"B matrix" bytes. State becomes shape.

## In production-scale models

Production LLM builders add: MLA/compressed attention, MoE with dispatcher
policies, speculative decoding (Medusa/EAGLE/MTP), in-flight batching
contexts with mixed prefill/decode chunks, mixed-precision recipes per op,
sliding windows, hybrid Mamba stacks... Every one is more shape arithmetic
on the same skeleton. And above the per-iteration model typically sits an
event-driven serving simulator for request-level dynamics that uses the
analytical model's steady-state numbers as its delay model.

## Exercises

1. Add a memory-capacity check: weights + KV + activations vs device HBM;
   report max batch at a given context length. (Implemented in milestone 8.)
2. Add MoE: `top_k` experts of an FFN each token, plus an AllToAll — find
   where DeepSeek-style MoE beats dense at equal params/token.
3. Model chunked prefill (mixed prefill+decode batches) and reproduce the
   latency/throughput tradeoff curve.

*Next: milestone 6 shards the graph across GPUs — tensor parallelism with explicit, priceable communication.*
