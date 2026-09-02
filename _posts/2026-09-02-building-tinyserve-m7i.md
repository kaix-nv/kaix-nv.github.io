---
layout: post
math: true
title: "Building tinyserve M7i: Pack Kimi experts, then batch the selected work"
date: 2026-09-02 13:58:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Profile Kimi serving, pack separately named experts before GPU transfer, batch selected expert projections, and retain a bounded-memory oracle for large prefills."
---

*Milestone 7i of [building an LLM inference engine from scratch](/series/tinyserve/).
Previous: [M7h — chunk and pack hybrid prefill]({% post_url 2026-09-02-building-tinyserve-m7h %}).*

Code: [`tinyserve` @ `973ce6b`](https://github.com/kaix-nv/tinyserve/tree/973ce6b0f80825c3e7c879006605d8a35ffbc7bf).

M7h made Kimi serving concurrent, but it did not make Kimi's model execution
fast. The first profile changed the next milestone. Copying request state into
a transient batch looked suspicious; it was not the dominant cost. On the
four-request tiny Kimi workload, cache gather and commit together consumed
about `0.39%` of wall time. The model consumed `99.2%`.

Nested CUDA events then pointed inside the model. Routed MoE accounted for
`106.9 / 142.2 ms` of prefill model time and `730.8 / 945.8 ms` of decode
model time—roughly three quarters of both phases. Optimizing persistent state
or KDA first would have attacked the wrong component.

The reason is not the MoE equation. It is how that equation reached the GPU.
For each active expert, M7h found its token assignments and invoked three tiny
linear projections. A decode batch has very few token rows but top-16 routing,
so Python dispatch and many undersized GPU launches dominate useful matrix
work.

![M7i replaces per-active-expert calls with bounded selected-weight batched
contractions](/assets/tinyserve/m7i-packed-kimi-experts.svg)

## Start with one concrete decode batch

The tiny fixture has `E = 64` routed experts. Each of `N = 4` decode tokens
selects `K = 16` experts in a `D = 256` latent space. The router returns:

```text
indices: [N, K] = [4, 16]       expert id for every assignment
weights: [N, K] = [4, 16]       normalized sigmoid route weight
```

There are 64 assignments, but usually fewer than 64 distinct active experts.
The grouped reference path processes them as follows:

```python
for expert_id in active_experts:
    token, slot = where(indices == expert_id)
    expert_input = routed_input[token]
    gate = linear(expert_input, w1[expert_id])
    up = linear(expert_input, w3[expert_id])
    selected[token, slot] = linear(situ(gate, up), w2[expert_id])
```

This is a good semantic oracle. All assignments for one expert share a call,
and it never copies weights for every route at once. It is a poor decode
kernel: one layer can execute close to three calls per active expert, and Kimi
has 16 routed-MoE layers.

## Pack storage once, without keeping a second GPU copy

The checkpoint names each expert separately:

```text
experts.0.w1.weight
experts.1.w1.weight
...
experts.63.w1.weight
```

Those names are useful for strict checkpoint loading, so the model initially
constructs the same `ModuleList`. Immediately after `load_state_dict`, while
the weights are still on CPU, `pack_experts()` stacks each projection along a
new expert dimension:

```text
w1: [E, I, D]
w3: [E, I, D]
w2: [E, D, I]
```

It then deletes the separate expert modules before `model.to(device)`. The
GPU receives one consolidated copy, not both the checkpoint-shaped modules
and packed tensors. For the 16 MoE layers in the tiny fixture, retaining both
would waste `384 MiB` of steady GPU memory.

This use of *packed* is distinct from the checkpoint's MXFP4 byte packing.
The loader first expands MXFP4 weights to BF16 as M7f defined; M7i then changes
only their execution layout.

## Turn assignment coordinates into batched contractions

Indexing an expert-major tensor with `[N,K]` router indices produces one
selected weight matrix per token-expert assignment. M7i computes:

```python
gate = einsum("nd,nkid->nki", routed_input, w1[indices])
up = einsum("nd,nkid->nki", routed_input, w3[indices])
activated = situ(gate, up)
selected = einsum("nki,nkdi->nkd", activated, w2[indices])
routed = (selected * weights[..., None]).sum(dim=1)
```

The first two contractions map `[N,D]` into `[N,K,I]`. The down contraction
returns `[N,K,D]`; the existing route weights reduce the `K` dimension. The
router, SiTU activation, shared experts, latent projections, and weighted
reduction are unchanged. Only the execution geometry changes.

For the concrete `N=4`, `K=16`, `I=D=256` batch, each projection replaces a
Python loop over active experts with one tensor contraction over all 64
assignments.

## Why the fast path needs a memory bound

Selected-weight execution trades launch count for temporary storage. One
advanced-index operation materializes:

$$
N \times K \times I \times D \times \text{element size}
$$

bytes. The three projection families are gathered sequentially, so one family
sets the selected-weight bound. M7i permits at most `64 MiB`. With the tiny
fixture's BF16 `K=16`, `I=D=256` geometry, one token costs `2 MiB` of selected
weights and at most 32 token rows take the fast path.

That covers decode and small packed prompts. A long or broadly padded prefill
can exceed the cap. It returns to the grouped oracle before gathering any
selected weights. This is intentionally a two-path teaching implementation:
the simple vectorized path demonstrates why launch geometry matters, while the
grouped path prevents chunk size from becoming an implicit memory hazard.

## Numerical boundary

The grouped and selected paths evaluate the same equation in a different
reduction geometry. The FP32 block test compares both paths within
`atol=rtol=2e-5`. The BF16 full-model test forces each path over the same four
prompts and requires maximum logit error below `0.004`, identical argmax, and
identical top-three candidate sets.

Generated strings are a weaker diagnostic for this random fixture. B1 and B2
remain token-identical in the retained benchmark. At B4, two rows eventually
choose different continuations after a near tie. As in M7h, that does not
indicate a routing or state-ownership error when the direct logit and leading-
candidate gates pass.

## Measured effect

The September 2, 2026 A/B run used the same loaded BF16 model for both paths,
one warmup, and four alternating measured repetitions. Prompts, token budget,
checkpoint, scheduler, backend, and GPU stayed fixed. The table reports median
generated-token throughput:

| batch | grouped oracle | selected path | speedup | selected peak-memory delta |
|---:|---:|---:|---:|---:|
| 1 | 9.13 tok/s | 24.28 tok/s | 2.66× | +1.5 MiB |
| 2 | 19.46 tok/s | 47.61 tok/s | 2.45× | +3.0 MiB |
| 4 | 33.99 tok/s | 93.73 tok/s | 2.76× | +13.9 MiB |

The gain follows the hypothesis: decode does the same expert arithmetic with
far fewer Python calls and small launches. The observed peak deltas stay well
below the 64 MiB selected-weight cap because tensors are materialized and
released by projection, and the cap is not the only live allocation.

## Same-checkpoint engine calibration

The fixed four-prompt calibration ran each adapter in both opposite orders,
with one warmup and three measured repetitions per order. Pooled medians are:

| execution path | batch 1 | batch 2 | batch 4 |
|---|---:|---:|---:|
| Tinyserve sequential `generate()` | 25.32 tok/s | 24.75 tok/s | 24.33 tok/s |
| Tinyserve M7i `serve()` | 24.42 tok/s | 47.44 tok/s | 92.35 tok/s |
| Transformers + FLA static batch | 9.60 tok/s | 17.99 tok/s | 35.93 tok/s |

This result is specific to the tiny random checkpoint and these adapters. The
reference uses a static cohort rather than Tinyserve's continuous scheduler,
and GPU telemetry was not clock-matched: reference samples reported 1800 MHz,
while some Tinyserve samples reported 210 MHz despite active P2 work. The
paired in-process grouped-versus-selected table is the causal M7i result. The
cross-engine table is a calibration showing no regression, not a general claim
that Tinyserve outruns an optimized production engine.

## Run and debug it

The **M7i: packed Kimi experts** launch configuration uses `generate.py` with
four requests. Set breakpoints in this order:

1. `KimiK3ForCausalLM.pack_experts()` — confirm packing occurs on CPU after
   strict checkpoint loading and before GPU transfer;
2. `KimiSparseMoeBlock._selected_weight_bytes()` — inspect `N*K` assignments
   and the pre-gather memory decision;
3. `_selected_experts()` — follow `[N,D]` and `[N,K,I,D]` through the two
   contraction shapes;
4. `_grouped_experts()` — lower `selected_weight_byte_limit` to `0` and watch
   the same assignments take the semantic oracle path;
5. `forward()` — verify route weighting and shared-expert addition are common
   to both paths.

```bash
.venv/bin/python examples/generate.py \
  --model /home/scratch.kaix_coreai/models/kimi-k3 \
  --serve --hybrid-slots 4 --chunk-size 512 \
  --prompts "2+2=" "Kimi" "A longer prompt" \
    "The capital of France is" \
  --no-chat --max-new-tokens 4 --backend gather --verbose
```

The paired mechanism benchmark is `examples/bench_kimi_experts.py`; it records
raw repetitions, outputs, token ids, peak allocations, checkpoint and source
hashes, and GPU telemetry in JSON.

## What M7i establishes

M7i demonstrates a recurring serving lesson: sparse arithmetic is not fast
merely because it computes fewer FLOPs. Route layout and launch geometry decide
whether those FLOPs reach useful GPU work. Expert-major storage lets a small
decode batch express all selected work as three batched contractions, while an
explicit byte cap preserves bounded prompt memory.

The contractions still materialize selected weights, and neither path is a
production grouped-GEMM kernel. KDA recurrence also remains a Python scan. The
next profile should determine which now dominates before choosing between a
Triton/grouped-GEMM expert kernel and a fused recurrent scan.
