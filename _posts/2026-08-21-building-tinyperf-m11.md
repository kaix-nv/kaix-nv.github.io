---
layout: post
math: true
title: "Building tinyperf M11: MoE — active params set your FLOPs, total params set your bytes"
date: 2026-08-21 22:40:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Routing, grouped expert GEMMs, and all-to-all dispatch turn MoE into shape arithmetic — and the decode asymmetry that runs MoE serving economics falls out of the operand shapes."
---

*Milestone 11 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — MoE branch in
`nets/transformer.py`, `all_to_all_us` in `comm_model.py` · Demo:
`examples/09_moe.py`.*

## Sparsity as routing

A mixture-of-experts layer replaces the dense FFN with `n_experts`
independent FFNs and a tiny router; each token is sent to its `top_k`
best experts. Mixtral-8x7B (all numbers public): 8 experts, top-2,
**46.7B total parameters but only 12.9B active per token** — our
`TransformerParams` reproduces both to the decimal from the config.

To the performance model, MoE is three new pieces of shape arithmetic:

1. **Routing** — a `tokens x n_experts` GEMM plus a softmax. Noise.
2. **Dispatch/combine** — with expert parallelism, tokens travel to their
   experts' GPUs and back: two **all-to-all** collectives, each moving
   this rank's `(tokens x top_k / ep) x hidden` slice. A2A moves
   `(g-1)/g` of the buffer in `g-1` hops — half an all-reduce — which is
   why a2a dispatch beats replicate-and-allreduce once `ep > top_k`.
3. **Expert GEMMs** — a *grouped* GEMM: one batched matmul over the local
   active experts, each with `m = tokens x top_k / n_experts` rows
   (balanced routing assumed; imbalance is an exercise). At decode batch
   sizes this lands squarely in milestone 10's launch-starved regime —
   the milestones compose.

Two modeling assumptions worth stating: routing is balanced, and the EP
world equals the attention-TP world (attention-TP/FFN-EP hybrid). Both are
documented simplifications, not oversights.

## The asymmetry that runs MoE serving economics

Watch what the grouped GEMM's operands imply. Its FLOPs cover only the
routed tokens — proportional to **active** params. But its B-operand bytes
are *all the local experts' weights*: once `batch x top_k` covers the
experts, every expert is hit every step, so decode streams weights
proportional to **total** params. Equal compute, 3.6x the bytes. The model
captures this for free, because operators are shape-and-traffic contracts.

`examples/09_moe.py`, Mixtral vs LLaMA2-13B (equal active params, 2 GPUs
each, short context to isolate weights from KV):

```
model         phase     batch       ms  GFLOP/tok
mixtral-8x7b  prefill       1    38.51       13.1
llama2-13b    prefill       1    41.31       13.4   <- equal FLOPs, ~equal time
mixtral-8x7b  decode       32    16.18       13.0
llama2-13b    decode       32     7.78       13.2   <- MoE pays 2.1x in bytes
mixtral-8x7b  decode      256    19.74       13.0
llama2-13b    decode      256    23.49       13.2   <- crossover! (but read on)
```

Prefill: math-bound, active params rule, dead heat. Decode at batch 32:
the MoE tax in full — same FLOPs, 2.1x the time, all of it weight
streaming. Decode at batch 256: Mixtral *wins* — but honesty requires
reading the report, not the headline: the crossover is mostly **GQA vs
MHA**. LLaMA2-13B is an MHA-era model whose KV cache reads grow 5x faster
per token than Mixtral's 8-KV-head GQA. Model comparisons entangle design
axes; an analytical model lets you pull them apart — swap in a GQA dense
baseline (one preset away) and the MoE tax persists to much larger batch.

## Expert parallelism

EP shards experts, not tensors: each of `ep` ranks hosts `n_experts / ep`
whole experts, and the a2a pair prices the token travel
(`examples/09_moe.py`, Mixtral decode b=32, kv=4096, H100):

```
  ep  ms/token    tok/s  comm %  mem/GPU GB  fits?  max_batch
   1     34.46      929     0.0       110.6     NO          0
   2     18.42     1737     2.5        55.6    yes         93
   4     10.45     3061     6.8        28.0    yes        359
   8      6.90     4635    17.7        14.3    yes        890
```

The same three-way readout as milestone 6's TP table — speed, comm share,
and capacity — with the capacity column doing the gating: Mixtral fp16
simply does not exist on one 80 GB GPU. And the same Amdahl arc: by ep=8,
a sixth of every token is all-to-all.

## In production-scale models

Production LLM builders model MoE with pluggable dispatcher policies
(default, fine-grained, min-latency), expert-choice vs token-choice
routing, shared experts, imbalance factors fitted from real router
telemetry, and hybrid parallel schemes (attention-TP x expert-EP x DP)
whose a2a costs depend on topology. DeepSeek-class models add
fine-grained experts (hundreds, small) precisely to move the
active-vs-total tradeoff — all of it still shape arithmetic on this
skeleton.

## Exercises

1. Add a load-imbalance factor f >= 1 (hottest expert gets f x fair
   share): grouped-GEMM time follows the straggler. How fast does f
   erode the EP-8 speedup?
2. Add shared experts (DeepSeek-style): a small always-on dense FFN
   beside the routed ones. Where does it pay?
3. Model the replicate-and-allreduce EP variant and find the ep/top_k
   crossover where a2a wins (the comm model already prices both).
4. Build a GQA dense 13B baseline and re-derive the decode crossover
   batch without the MHA confound.

*Next: milestone 12 — training mode, where the backward pass arrives as a
graph transformation and pipeline parallelism as a bubble formula.*
