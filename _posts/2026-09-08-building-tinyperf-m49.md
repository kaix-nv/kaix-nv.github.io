---
layout: post
math: true
title: "Building tinyperf M49: 2:4 sparsity: a bytes story or a math story?"
date: 2026-09-08 16:45:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Sparse tensor cores promise 2x. The model says 21% on small-batch decode (bytes), 17% on prefill (the L2 feed binds once the rate doubles), 7% on large-batch decode (attention is untouched) — and 7.5 GB per GPU back."
---

*Milestone 49 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `apply_sparsity` in `passes.py`, `sparse=` in `gemm_model.py` · Example: `examples/32_sparsity.py`.*

Structured 2:4 sparsity is the tensor-core feature everyone quotes and
few deployments use: prune two of every four weights, keep a 2-bit index
per survivor, and the hardware multiplies at twice the dense rate on half
the bytes. The datasheet says 2×. What does the model say, and — the more
useful question — *where* does it say it?

## The mechanism, as a recipe

Sparsity is a property of specific weights, not of a run, so it takes the
same shape as the precision recipes of milestone 35: a pass over the
built graph that tags weight GEMMs by name prefix. Two recipes ship, the
FFN-only one that most pruning papers evaluate and the every-weight one.
A tagged GEMM runs at `sparse_math_multiplier` (2.0, the public figure;
set 1.0 for a part without sparse tensor cores) times the dense rate, and
its B operand costs `0.5 + 0.125 / bytes_per_element` of the dense bytes —
the nonzeros plus their indices, which is why fp4 sparse weights are 75%
of fp4 dense while fp16 sparse weights are 56%.

The pass refuses to tag an attention matmul. `QK^T` and `PV` multiply
activations by activations; there are no weights to prune, and a prefix
like `attn_` would otherwise claim a speedup that cannot exist. Precision
and sparsity are independent tags on the same op, so a sparse fp4 FFN
composes without new code, and capacity prices the smaller weights the
same way through `llm_memory`, `max_batch`, serving and sweeps.

## LLaMA3-70B, eight H100s

```
  recipe     TPOT b=8   TPOT b=256   TTFT 8x2k   GB/GPU   max b@8k
  dense         14.02        35.96         688     21.3        150
  2:4 FFN       11.72        34.14         592     15.1        169
  2:4 all       11.07        33.48         571     13.8        173
```

Three different answers to "how much does 2:4 buy":

- **Small-batch decode: 21%.** A batch-8 step streams weights; sparse
  weights are 56% of the bytes and the FFN time follows them (0.60×,
  launch and activation traffic making up the difference). This is the
  bytes story, and it is the one that matters for latency-bound serving.
- **Prefill: 17%.** The FFN GEMMs are math-bound, the rate doubles — and
  the FFN time falls to 0.71×, not 0.5×. With the multiply twice as fast,
  the L2 feed that stages tiles into shared memory becomes the binding
  term in the GEMM model. Sparse kernels in practice land at 1.3–1.6× over
  dense at large shapes, and the model reproduces that shape of answer
  without a fudge: the datasheet 2× is the rate of one unit, not of the
  kernel.
- **Large-batch decode: 7%.** At batch 256 with 4k contexts the step is
  attention, and attention is untouched.

And a fourth answer in the last two columns: 7.5 GB per GPU back, which
is 15% more concurrent 8k sequences. For a memory-bound deployment that
may be the number that pays.

## What is not modeled

The accuracy cost of pruning — a checkpoint's problem, not the runtime's,
but the reason most deployments do not do this. Sparse kernel efficiency
is assumed equal to dense (tile, pipeline and L2 terms unchanged); real
sparse kernels have their own tile shapes and metadata decode, and none
has been measured here. Activation sparsity, a different mechanism with
no tensor-core support, is not this.

## Exercises

1. The FFN prefill speedup is L2-limited. Raise `l2_bw_gbps` 2× on a
   what-if device and see how much of the 2× returns.
2. Pair `SPARSITY_2_4_ALL` with `RECIPE_FP4_SERVING` on a B200 and find
   the batch at which the fp4 sparse deployment stops being bytes-bound.
3. Add a per-op sparse efficiency (like `fmha_math_efficiency`) and fit
   it from a cuSPARSELt microbenchmark on a GPU you have.
