---
layout: post
math: true
title: "Building tinyserve M6d: Context parallelism: split the KV, not the answer"
date: 2026-10-05 08:03:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Shard KV history and combine distributed softmax exactly, with a concrete two-GPU attention example."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m6d-context-parallelism.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M6c — Expert parallelism: route the token, not the whole model]({% include tinyserve-post-url.html slug="building-tinyserve-m6c" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m6c-expert-parallelism.md" %}) · Next: [M7a — Before optimizing, name the phase]({% include tinyserve-post-url.html slug="building-tinyserve-m7a" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7-phase-profiling.md" %})

Tensor parallelism splits heads and matrix channels. Pipeline parallelism
splits layers. Expert parallelism splits conditional MLP branches. None of
them directly answers this question:

> What if the model fits, but one request's KV cache does not?

Context parallelism partitions the sequence state. In M6d, model weights and
queries remain replicated, while persistent paged-KV blocks have exactly one
owner. Every rank scores its query against its local keys, then the ranks
reconstruct the same attention result that one GPU would have computed.

For Qwen3-0.6B BF16, one cached token occupies

$$
2 \times 28 \text{ layers} \times 8 \text{ KV heads}
\times 128 \text{ values/head} \times 2 \text{ bytes}
= 112 \text{ KiB}.
$$

An 8,192-token pool therefore consumes 0.875 GiB. At equal global capacity,
CP=2 stores 4,096 token slots, or 0.4375 GiB, on each rank. The weights remain
1.400 GiB per rank: this milestone adds KV capacity, not model-weight
capacity.

<a href="/assets/tinyserve/m6d-context-parallelism.svg"><img src="/assets/tinyserve/m6d-context-parallelism.svg"
     alt="A concrete CP=2 example maps three logical KV blocks to two rank-local pools, then reconstructs exact attention using MAX and SUM reductions"></a>

## One block table, two physical pools

Use the figure's block size of four. A 12-token sequence owns three global KV
blocks:

| logical positions | global block | owner | owner's local block |
|---|---:|---:|---:|
| 0–3 | 0 | rank 0 | 0 |
| 4–7 | 1 | rank 1 | 0 |
| 8–11 | 2 | rank 0 | 1 |

The mapping is intentionally mechanical:

```python
owner = global_block % cp_size
local_block = global_block // cp_size
```

The sequence and scheduler still carry global block IDs. Both ranks mirror
the same global allocator, refcounts, prefix hashes, and request state, so
admission and preemption make one decision. The physical pools differ: rank 0
stores only even global blocks and rank 1 only odd ones.

When attention writes a new K/V row, every rank computes the replicated QKV
projection, but only the owner commits K/V to persistent storage. Padding
writes still go to a rank-local scratch block. Before a forward, each rank
packs its owned block IDs once and retains their logical block positions.
Every layer reuses that metadata, including during:

- whole-prompt prefill;
- packed, left-padded prefill;
- a later prompt chunk attending over earlier chunks;
- batched decode over ragged sequence lengths.

Logical positions are essential. Compacting rank 0's blocks from global
positions `[0, 2]` into local rows `[0, 1]` must not make the second block look
adjacent to the first in the causal mask. Its keys still represent positions
8–11.

## Why two reductions are required

For one query, ordinary attention is

$$
o = \frac{\sum_j \exp(s_j) v_j}{\sum_j \exp(s_j)},
\qquad s_j = \frac{q k_j^\mathsf{T}}{\sqrt{d}}.
$$

Computing a softmax independently on each rank and adding the outputs is
wrong. If rank 0 has score `10` and rank 1 has score `0`, each one-element
local softmax assigns its own value weight 1. An average would treat them
equally, while the global softmax assigns approximately `0.99995` and
`0.00005`.

M6d reconstructs the global normalization in stable form. For rank `r`:

$$
m_r = \max_{j \in r} s_j,
\qquad m = \max_r m_r,
$$

then

$$
d_r = \sum_{j \in r} \exp(s_j - m),
\qquad
n_r = \sum_{j \in r} \exp(s_j - m) v_j.
$$

A MAX all-reduce produces `m`. A SUM all-reduce produces

$$
d = \sum_r d_r,
\qquad n = \sum_r n_r,
\qquad o = n / d.
$$

The reduced tensors follow the query shape, not the context length. K/V never
crosses ranks. A rank may own no visible block for a short sequence; its local
scores are masked, and the other rank still supplies a finite global maximum
and denominator.

In the code, `_context_attend()` performs the score math in FP32, concatenates
the numerator and denominator into one SUM payload, and casts the final output
back to the query dtype. That makes the communication visible: one MAX and one
SUM collective per attention layer per forward.

## The serving control plane does not split

Rank 0 remains the only owner of tokenization, arrivals, scheduling, sampling,
output, and metrics. It broadcasts the same M4/M5 step plan used by TP, PP,
and EP. Worker ranks mirror sequence mutations and global block operations,
then participate in attention collectives.

Prefix reuse remains valid because a full global block still denotes one
immutable token prefix. Adoption increments the same global refcount on both
ranks; only the physical owner has bytes to retain. Recompute preemption frees
the same global IDs everywhere and reconstructs them through ordinary chunked
prefill after readmission.

This first slice uses eager PyTorch attention. FlashInfer consumes a complete
local page table and cannot perform tinyserve's distributed normalization.
CUDA graphs are also disabled because the rank-local packed block width varies
with ownership. Those are later kernel/composition problems, not hidden behind
the M6d interface.

## Correctness evidence

The checks build from placement to the full serving lifecycle:

- CPU tests pin the even/odd owner mapping, global-to-local translation,
  owner-only writes, scratch writes, and packed local block metadata.
- A two-GPU FP32 gate compares CP last-token logits with the unpartitioned
  paged model within `3e-4`, including a prompt spanning both owners.
- Greedy generation and `serve()` preserve token IDs through whole and packed
  prefill, an awkward chunk budget, prefix adoption, and forced recompute
  preemption.
- A controlled BF16 chat workload preserves tokens and confirms that CP keeps
  FlashInfer and CUDA graphs disabled.
- Collective counters observe one MAX and one SUM per layer and forward; pool
  inspection proves disjoint physical block ownership.

As in M6c, raw text continuation is not used as semantic evidence. The BF16
gate uses the checkpoint's chat template so a close, underspecified first-token
distribution is not mistaken for a serving error.

## What the two A6000s measured

The retained run used Qwen3-0.6B, BF16, two RTX A6000 GPUs, PyTorch
2.13.0+cu130, an 8,192-token logical KV capacity, eager attention, no prefix
caching, four greedy output tokens, and three repeats. CP=1 used the existing
gather backend. CP=2 used the explicit distributed-softmax path. The synthetic
repeated-token prompt controls length; it does not measure answer quality.

At equal logical capacity:

| quantity | CP=1 | CP=2 |
|---|---:|---:|
| ranks | 1 | 2 |
| model weights per rank | 1.400 GiB | 1.400 GiB |
| KV token slots per rank | 8,192 | 4,096 |
| KV blocks per rank | 512 | 256 |
| global KV blocks | 512 | 512 |
| persistent KV pool per rank | 0.875 GiB | 0.4375 GiB |

The persistent KV result is exact: CP=2 halves the pool per rank. Giving each
CP rank the original 8,192 local slots would instead expose 16,384 global
slots, with the original 0.875 GiB pool on each GPU.

Median latency across three repeats was:

| prompt tokens | CP=1 TTFT | CP=2 TTFT | CP=1 ITL | CP=2 ITL |
|---:|---:|---:|---:|---:|
| 512 | 24.5 ms | 72.4 ms | 25.5 ms | 50.6 ms |
| 2,048 | 56.2 ms | 286.8 ms | 26.0 ms | 50.9 ms |
| 4,096 | 115.5 ms | 721.3 ms | 26.0 ms | 51.1 ms |

Four output tokens mean four forwards: one prefill and three decode calls.
With 28 layers, each CP rank therefore observed 112 MAX and 112 SUM
reductions. SUM payload per rank grew from 113.5 MiB at 512 prompt tokens to
452.2 MiB at 2,048 and 903.7 MiB at 4,096. The MAX payload stayed much smaller
because it carries one scalar per query head and position; the SUM also carries
the 128-value numerator.

A separately labelled 512-token, one-forward profile synchronized every
collective. It observed 28 MAX plus 28 SUM calls, 113.75 MiB of combined
payload per rank, and at most 34.4 ms of accumulated event time. This mode
distorts normal execution and does not isolate NCCL from synchronization
overhead, but it bounds how much of the measured path sits inside the
collectives.

Persistent pool memory and peak execution memory are different. At 512 tokens,
peak reserved memory fell from 2.492 GiB to 1.949 GiB per rank. At 4,096,
however, it rose from 2.648 GiB to 3.859 GiB: this readable implementation
materializes FP32 score, probability, and numerator tensors, while CP=1 uses a
fused SDPA kernel. M6d demonstrates correct ownership and normalization, not a
production long-context kernel.

The raw measurements are retained in
[`benchmarks/m6d-cp2-a6000-2026-08-29.json`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m6d-cp2-a6000-2026-08-29.json).

## Boundaries

- Weights, QKV projection work, and the residual stream are replicated. CP
  does not make a too-large model fit.
- Global block IDs are round-robin over the physical pool. A fresh long
  sequence is balanced, but reuse and allocator history can skew one
  sequence's owner count.
- The explicit attention path is quadratic in prefill length and allocates
  large FP32 temporaries. A fused ring/online-softmax kernel is outside M6d.
- FlashInfer, CUDA graphs, and combined TP×PP×EP×CP execution are not claimed.
  The two available GPUs establish CP=2 independently.

## Takeaway

Context parallelism is not “split the prompt and average attention.” The
placement invariant is that every persistent KV block has one physical owner.
The numerical invariant is that all owners contribute to one global softmax
normalization.

M6d completes tinyserve's four independent partition axes: TP splits a layer,
PP splits layers, EP splits experts, and CP splits sequence state. The next
milestone can now treat distributed attention and cache policy as explicit
research interfaces instead of implicit engine assumptions.
