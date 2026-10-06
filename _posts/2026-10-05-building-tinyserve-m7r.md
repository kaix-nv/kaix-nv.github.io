---
layout: post
math: true
title: "Building tinyserve M7r: Measure the MoE launch storm before writing a grouped kernel"
date: 2026-10-05 08:16:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Explain the MoE launch storm with a concrete route table before designing a grouped expert kernel."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m7r-kimi-moe-profile.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M7q — Submit KDA chunks once, carry state on the GPU]({% include tinyserve-post-url.html slug="building-tinyserve-m7q" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7q-multichunk-kda-prefill.md" %}) · Next: [M7s — Replace the Python expert loop with indexed grouped GEMMs]({% include tinyserve-post-url.html slug="building-tinyserve-m7s" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7s-native-indexed-moe.md" %})

M7q removed most of the host submission overhead around Kimi Delta Attention.
That optimization also changed the answer to “what is slow?” At 2,048 prompt
tokens, KDA fell to about one third of the model interval and routed MoE became
the largest measured module.

That is not enough evidence to start writing a generic MoE kernel. Kimi's
teaching implementation already has two expert paths with very different
costs. M7r changes no serving math. It identifies which path runs at each
prompt length, measures the stages around it, and counts the operators inside
one real-shape expert block.

[![M7r follows Kimi MoE through the 32-token selected-weight boundary, a
concrete grouped-expert trace, the measured launch storm, and the bounded M7s
kernel target.](/assets/tinyserve/m7r-kimi-moe-profile.svg)](/assets/tinyserve/m7r-kimi-moe-profile.svg)

## Start with the path boundary, not the MoE equation

The tiny checkpoint has:

```text
E = 64 experts
K = 16 selected experts per token
D = 256 routed hidden channels
I = 256 expert intermediate channels
weights = BF16
```

M7i packed the checkpoint weights into expert-major tensors:

```text
w1, w3: [E, I, D]
w2:     [E, D, I]
```

For a short input, advanced indexing creates one weight matrix for every
token-expert assignment. One projection family therefore materializes:

$$
N \times K \times I \times D \times 2\ \text{bytes}.
$$

With this checkpoint, that is exactly `2 MiB` per token row. The implementation
allows a `64 MiB` temporary:

| token rows | one selected family | path |
|---:|---:|---|
| 32 | 64 MiB | selected |
| 33 | 66 MiB | grouped |
| 128 | 256 MiB | grouped |
| 512 | 1 GiB | grouped |
| 2,048 | 4 GiB | grouped |

The three weight families are gathered sequentially, so the table is not
claiming that all three sizes are live simultaneously. It explains the guard:
raising the cap would turn a launch problem into an avoidable memory problem.
It also corrects an easy misconception about the M7q profile. The long-prompt
bottleneck is `_grouped_experts()`, not the selected-weight fast path.

## Follow four tokens through the current grouped path

Use a smaller `K=2` example. Suppose the router produces:

```text
token 0 → expert 2, expert 5
token 1 → expert 2, expert 7
token 2 → expert 5, expert 7
token 3 → expert 2, expert 5
```

The grouped implementation first counts assignments, moves the active expert
IDs to Python, and then repeats this work for expert 2, expert 5, and expert 7:

```python
for expert_id in active_experts:
    token, slot = torch.where(indices == expert_id)
    expert_input = routed_input[token]
    gate = F.linear(expert_input, w1[expert_id])
    up = F.linear(expert_input, w3[expert_id])
    activated = situ(gate, up)
    selected[token, slot] = F.linear(activated, w2[expert_id])
```

The algorithm is a useful memory-bounded oracle. It stores only the rows for
one active expert and invokes that expert once. Its GPU schedule is fragmented:
every active expert adds a route comparison, `torch.where`, token gather,
three linear projections, activation operations, and a scatter.

The retained prompts use all 16 routed-MoE layers. Each layer activates only
16–19 of the 64 experts for this random fixture, with a median of 17. Even that
concentrated routing means roughly 17 loop iterations and 51 expert projection
calls per layer. A trained model or a less repetitive prompt can activate more
experts; M7r does not generalize this fixture's route distribution.

## Attribute the real prefill before isolating an operator

Each condition contains one raw prompt that tokenizes to exactly 32, 128, 512,
or 2,048 tokens. The scheduler admits the whole prompt in one prefill action
and generates one token. Two warmups precede five retained repetitions.

CUDA events surround the whole model and these disjoint regions inside every
MoE block:

```text
router → routed down projection → selected/grouped experts
       → routed norm/up projection → route reduction and remaining work
       + shared expert
```

The table uses median CUDA-event intervals. “Expert share” is the selected or
grouped expert interval divided by its parent MoE interval:

| prompt | model | MoE | model share | path | expert interval | expert share of MoE |
|---:|---:|---:|---:|---|---:|---:|
| 32 | 69.8 ms | 24.3 ms | 34.8% | selected | 10.8 ms | 44.6% |
| 128 | 162.2 ms | 116.0 ms | 71.5% | grouped | 98.3 ms | 84.8% |
| 512 | 162.0 ms | 114.2 ms | 70.5% | grouped | 98.3 ms | 86.1% |
| 2,048 | 202.2 ms | 108.9 ms | 53.8% | grouped | 96.5 ms | 88.7% |

At 128 and 512 tokens, grouped expert execution alone consumes about 60% of
the complete model interval. At 2,048 tokens, KDA grows again, but grouped
experts remain the largest single measured region.

The other MoE stages are much smaller. At 128 tokens the router takes `3.9
ms`, the shared expert `5.1 ms`, and the grouped expert loop `98.3 ms`. Starting
with top-k, shared-expert fusion, or routed projection fusion cannot address
the dominant interval.

These are diagnostic intervals, not an M7r speed benchmark. Adding nested CUDA
events perturbs a launch-heavy program, and intervals can include idle stream
gaps while Python submits later work. M7q's uninstrumented paired numbers remain
the throughput result for the serving implementation.

## Count the launch geometry in one block

The second profile sends deterministic synthetic hidden states through one
checkpoint MoE block. The tensors use the real model dimensions and exercise
the same path guard, but their route distribution is deliberately separate
from the fixed full-model prompts.

| rows | path | active experts | CUDA kernel events | ATen self CPU | CUDA kernel time | temporary allocation |
|---:|---|---:|---:|---:|---:|---:|
| 32 | selected | 43 | 61 | 1.17 ms | 0.76 ms | 65.0 MiB |
| 128 | grouped | 50 | 1,215 | 14.80 ms | 2.65 ms | 2.6 MiB |
| 512 | grouped | 50 | 1,265 | 15.61 ms | 2.84 ms | 10.4 MiB |
| 2,048 | grouped | 54 | 1,352 | 16.63 ms | 3.82 ms | 41.4 MiB |

The selected path expresses the expert work with three indexed weight gathers
and three batched contractions. At 32 rows, `aten::index` takes `0.321 ms` and
the batched matrix multiplications take `0.310 ms` of self CUDA time. It has
good launch geometry but pays the 65 MiB temporary.

The grouped path preserves memory but submits more than 1,200 kernels in this
one-block trace. At 2,048 rows, the profiler records `16.6 ms` of ATen self CPU
time around only `3.8 ms` of CUDA kernel time. Those clocks are not additive
and the profiler adds overhead, but the combination of source structure,
operator counts, and the full-model expert interval identifies a launch-bound
dispatch problem rather than an expensive router.

`aten::nonzero` is the largest individual grouped CUDA operator in the trace,
but replacing only `torch.where` would leave three projections per active
expert. The optimization boundary needs to remove the expert loop and its
small GEMM launches together.

## M7s: the bounded experiment selected by M7r

The next milestone should build a checkpoint-specific native indexed expert
path with this contract:

```text
input:  routed rows [N,D], expert IDs [N,K], route weights [N,K]
weights: packed expert-major w1/w3 [E,I,D], w2 [E,D,I]
output: weighted routed rows [N,D]
```

The kernel path must read the packed expert weights by expert ID. It must not
materialize `[N,K,I,D]` selected weights, and it must not return to Python once
per active expert. The existing grouped implementation stays as the semantic
and memory-bounded oracle.

M7s is promoted only if it:

1. keeps full-model maximum logit difference at or below `0.004`, preserves
   ordered top-three candidates, argmax, and generated tokens;
2. reduces full-model latency by at least 10% at 128 and 512 tokens and 5% at
   2,048 tokens;
3. regresses the 32-token row by no more than 5%;
4. adds no more than 32 MiB of peak allocation;
5. removes the grouped launch storm in a post-change profile; and
6. passes the same-checkpoint cross-engine calibration after the feature is
   implemented.

M7r does not rerun external engines because it changes instrumentation only.
There is no new serving implementation to calibrate.

## Verify that the diagnostic stayed diagnostic

The Kimi implementation hash is unchanged from M7q. For the four exact
prompts, M7q and M7r choose the same first-token IDs:

```text
T=32                         → 76377
T=128, T=512, and T=2048    → 81998
```

The complete single-visible-GPU suite passes `125` tests with `9` expected
two-GPU TP, PP, EP, and CP skips. These checks do not prove a new model path—
M7r intentionally has none. They show that the profiler observes the existing
path without editing its implementation.

## Debug the measurement

The **M7r: profile Kimi MoE boundary** launch configuration uses 32 and 33
tokens so both paths can be stepped through quickly. Set breakpoints in this
order:

1. `profile_moe_operators()` — inspect the real `E=64`, `K=16`, `D=I=256`
   block and its synthetic input;
2. `KimiSparseMoeBlock.forward()` — calculate `last_selected_weight_bytes` and
   watch the 64 MiB decision;
3. `_selected_experts()` for 32 rows and `_grouped_experts()` for 33 rows; and
4. `KimiAttributor.summary()` — compare nested device intervals, host
   submission time, paths, and active-expert counts.

The retained evidence uses the same profiler with all four lengths. The
compact [M7r evidence manifest](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7r-kimi-moe-profile-a6000-2026-09-09.json)
records the protocol, medians, path boundary, operator counts, hashes,
limitations, and the M7s gate.

## Takeaway

Sparse arithmetic is not automatically an efficient GPU schedule. The
selected path batches work but scales temporary weight storage with every
token-expert assignment. The grouped path bounds memory but expands a small
set of active experts into hundreds of Python-driven operations across the
model.

M7r supplies the missing decision evidence: keep packed expert-major weights
and the grouped oracle, then test a native indexed path that eliminates both
selected-weight materialization and per-expert launch dispatch.
