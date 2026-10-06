---
layout: post
math: true
title: 'Tinyserve, Chapter 16: Tensor and pipeline parallelism'
date: 2026-10-06 00:00:00 -0700
categories:
- tinyserve
- llm-serving
excerpt: What work and communication follow from splitting tensors or layers?
book_chapter: 16
source_revision: 14b0c8dbb975350bbd730ac0c702eafce36eb95c
source_document: docs/book/16-tensor-and-pipeline-parallelism.md
code_revision: e20a34815498560f9226a9e057ade31b2b62743f
---

{% include tinyserve-article-style.html %}
{% include tinyserve-book-nav.html %}

*Chapter 16 · Distributed Serving*

One device may lack the capacity for a model, or a workload may benefit
from distributing its computation. Those are related but different goals.
Two complete replicas increase independent serving capacity, while each
request still needs a whole model on one device. Tensor parallelism and
pipeline parallelism instead make devices cooperate on one logical model.

Tensor parallelism divides operations inside a layer. Pipeline parallelism
divides the layer stack. We will place the same small decoder on two GPUs
in both ways and follow weights, residual activations, KV, and sampling.
The comparison makes communication and idle time as visible as the local
matrix work that distribution saves.

## Ranks and communication primitives

A rank is one participant in a distributed process group. In Tinyserve's
distributed paths, each process owns one GPU. A broadcast sends one rank's
value to its peers. An all-reduce combines equal-shaped tensors, for example
by summing their elements, and gives every rank the result. A point-to-point
send transfers a tensor from one specified rank to another.

These operations must be called in a compatible order. If one rank enters
an all-reduce while its peer follows a different request schedule, neither
the request IDs nor the tensor contents are sufficient to repair that
divergence. Tinyserve therefore gives rank 0 ownership of admission,
preemption, sampling, and returned results. Workers mirror explicit plans
and participate in the corresponding tensor operations.

The control plane is small compared with model tensors, but it is not
optional. A sampled token must reach every rank before the next forward,
and a page allocation must have the same logical meaning everywhere.
Distributed execution begins with an agreement about what work is running.

## Tensor parallelism through a numerical projection

Write a linear layer as `Y = X Wᵀ`, with W shaped `[output, input]`.
Splitting output features means splitting rows of the stored W. This is
conventionally called column parallelism because it creates columns of Y.
Each rank receives the same X and computes a different output feature slice.

For a tiny example, let `X = [1, 2]` and

```text
W = [[1, 0],     rank 0 owns the first two output features
     [0, 1],
     [1, 1],     rank 1 owns the last two output features
     [2, 0]]
```

Rank 0 produces `[1, 2]`; rank 1 produces `[3, 2]`. Together they represent
the four-feature result `[1, 2, 3, 2]`, but neither rank needs to gather
that entire vector if the next operation consumes its corresponding slice.

Suppose the next projection returns to width two with

```text
U = [[1, 0, 1, 0],
     [0, 1, 0, 1]]
```

This projection splits its input features, conventionally called row
parallelism. Rank 0 uses the first two columns of U and produces partial
output `[1, 2]`. Rank 1 uses the remaining columns and produces `[3, 2]`.
An all-reduce sum returns `[4, 4]` to both ranks, exactly the unsharded
`[1, 2, 3, 2] Uᵀ` in this small integer-valued example.

The general identity is

$$
Y = X_0 U_0^\mathsf{T} + X_1 U_1^\mathsf{T}.
$$

Each partial already has the full output width. Summation reconstructs
the output; concatenation would be wrong. Floating-point execution retains
this mathematical identity while potentially changing accumulation order.

[![Both TP ranks begin with the same residual. Local attention heads and MLP channels produce full-width partial outputs; one all-reduce after each output projection restores the replicated residual.](/assets/tinyserve/book16-tensor-shards.svg)](/assets/tinyserve/book16-tensor-shards.svg)

## Apply the pair twice in each decoder layer

Qwen3's Q, K, and V projections split by complete output heads. Attention
runs on local heads, including the local KV cache. The attention output
projection consumes those head slices and sums its residual-width partials
across ranks. The gated MLP repeats the pattern: gate and up projections
produce local intermediate channels, their nonlinear product remains local,
and the down projection ends with an all-reduce.

For Qwen3-0.6B, residual width is 1024, query heads are 16, KV heads are 8,
head width is 128, and MLP width is 3072. With TP=2, each rank sees:

| Tensor | Rank-local shape for one decode step |
|---|---|
| Residual input | `[B, 1, 1024]`, replicated |
| Q | `[B, 1, 8, 128]` |
| K and V | Each `[B, 1, 4, 128]` |
| Attention projection partial | `[B, 1, 1024]` |
| Gate and up outputs | Each `[B, 1, 1536]` |
| MLP down-projection partial | `[B, 1, 1024]` |

The residual is replicated at layer boundaries. Normalization and the next
layer can therefore begin on both ranks without a separate feature gather.
For this decomposition, query-head count, KV-head count, and MLP width must
divide by the TP degree; validation catches incompatible configurations
before useful distributed work starts.

Embeddings, normalization parameters, and endpoint weights are replicated.
The loader selects rank-specific Q/K/V, attention-output, gate/up, and down
weight shards. It can read the full CPU checkpoint in this educational
implementation, but only the selected shards move to the GPU. Host loading
capacity and GPU parameter capacity are consequently different questions.

A sequence's block table is mirrored, for example `[3, 9]`, while its
physical contents differ by rank. Rank 0's page 3 contains KV heads 0–3;
rank 1's page 3 contains heads 4–7. Local attention backends must plan with
four KV heads and eight query heads, not the global counts. No attention
collective is required here because complete head computations are local;
the combination occurs after the output projection.

## Tensor parallelism pays at every layer

There are two all-reduces per decoder layer. With 28 layers, one forward
contains 56 collective calls, even if the decode batch has only one token.
For N token rows and residual width H, each reduction passes an `[N, H]`
tensor. Its logical payload is `N × H × element_bytes` per rank; actual
link traffic depends on the collective algorithm and topology.

Small batches can make latency per collective more important than byte
volume. Large prefill batches increase both useful GEMM work and collective
payload. Halving sharded work is therefore only one term in the cost model.
Replicated operations and communication remain, and the slowest participant
can delay the next dependency on every layer.

Tinyserve supports eager gather attention, eligible local FlashInfer, and
TP decode graph replay. Captured collectives must execute in the same bucket
order on every rank. Graph resources are explicitly released before the
process group is destroyed; communication lifetime extends beyond a Python
function return when graph objects retain captured operations.

The retained Qwen3-0.6B BF16 experiment used two RTX A6000s, FlashInfer,
CUDA graphs, 16 simultaneous requests, 32 outputs each, no prefix reuse,
and three measured repetitions. Median useful throughput fell from
2487 tokens/s at TP1 to 1945 at TP2; median inter-token latency rose from
5.74 to 7.34 ms. The
[measurement record](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m6a-tp2-a6000-2026-08-28.json)
therefore shows a capacity mechanism with a latency cost on this small model.

Per-rank reserved memory fell from 3568 MiB to 2712/2672 MiB, while total
reserved memory rose to 5384 MiB. Replicated tensors, workspaces, and graph
allocations explain why a half-sized shard does not imply half the complete
process footprint. Twelve of sixteen broad BF16 benchmark rows matched TP1
tokens; controlled tests and FP32 comparisons establish narrower numerical
contracts. The measured workload kept fixed output lengths, but it is not
a universal bitwise-equivalence result.

## Pipeline parallelism changes the ownership boundary

Now keep each layer intact and split the same 28-layer model in half.
Rank 0 owns the embedding and layers 0–13. Rank 1 owns layers 14–27,
final normalization, and the vocabulary projection. Stage 0 sends hidden
activations `[B, T, 1024]`; stage 1 completes the remaining model.

KV follows layers rather than heads. Rank 0 stores all eight KV heads for
its fourteen layers; rank 1 stores all heads for the other fourteen. Global
layer names remain useful for checkpoint loading, while cache indices become
local: global layer 14 writes local cache slice zero on rank 1.

The embedding and LM head use tied learned values in this checkpoint. Since
their consumers sit on different GPUs, they cannot share one cross-device
allocation. The loader gives each endpoint a copy of the same checkpoint
tensor. Other stage-owned parameters occur on only their owning rank.

[![PP rank zero owns embedding and early layers; rank one owns late layers and logits. Four row microbatches overlap stage work after filling the pipeline, with visible initial and final bubbles.](/assets/tinyserve/book16-pipeline-timeline.svg)](/assets/tinyserve/book16-pipeline-timeline.svg)

One microbatch uses stage 0 while stage 1 waits, then stage 1 while stage 0
waits. These idle periods are pipeline bubbles. Splitting the scheduled
request rows into microbatches can overlap stage 0 for the next group with
stage 1 for the previous group.

Suppose decode selected request IDs `[7, 2, 9, 4]` and each microbatch has
one row. The pipeline processes MB0 for request 7, MB1 for request 2,
MB2 for request 9, and MB3 for request 4. Returned logits are retained in
that order and concatenated before the scheduler samples. Stages do not
independently reorder, admit, or retire requests.

This is row microbatching. Splitting one long prompt into chunks controls
how much of its token history is appended per iteration; it does not create
multiple independent request rows. A one-row prompt chunk still experiences
pipeline fill and drain in this implementation.

## Bubbles compete with matrix efficiency

For P balanced stages, M microbatches, and an ideal equal stage time t per
microbatch, the simple forward pipeline takes approximately

$$
\begin{aligned}
t_{\mathrm{pipeline}}&=(M+P-1)t,\\
\mathrm{utilization}&=\frac{M}{M+P-1}.
\end{aligned}
$$

For two stages and four microbatches, that ideal utilization is 4/5.
The formula assumes stage costs remain fixed and ignores communication.
When a fixed batch is split into smaller row groups, t changes: GEMMs have
less work per launch and may use the device less efficiently. More occupied
timeline slots need not mean more useful tokens per second.

Tinyserve sends each activation asynchronously and posts the corresponding
logit receive. The next stage receives, runs its layers, and sends its
logits back. Buffers remain alive until the communication handles complete.
Correct send/receive ordering prevents deadlock, while buffer lifetime
prevents a producer from overwriting a tensor before its peer consumes it.

Rank 0 remains the sampler. That choice requires returning full vocabulary
logits from rank 1. For one BF16 decode row, the stage activation is
`1024 × 2 = 2048` bytes, while the 151,936-entry logits vector is
303,872 bytes, or 296.75 KiB. The return transfer is about 148 times larger.
For prefill, activation bytes grow with T, while only the final logits row
returns. Sampling on the final stage could change this cost, but would
move the implemented policy boundary.

PP uses eager execution here, with local attention on each stage. The
end-to-end graph runner does not capture this variable microbatch and
point-to-point schedule. A comparison against a graph-enabled single-device
run would therefore combine distribution with a different execution mode.

## The measured microbatch trade

The retained PP comparison used the same small BF16 model on two A6000s,
eager FlashInfer for both the baseline and PP, eight requests, sixteen
outputs per request, no prefix reuse, and three repetitions:

| Execution | Rows per microbatch | Useful tokens/s | Median ITL |
|---|---:|---:|---:|
| One GPU | 8 | 320.4 | 24.8 ms |
| PP2 | 8 | 293.3 | 26.8 ms |
| PP2 | 4 | 200.8 | 40.0 ms |
| PP2 | 2 | 112.0 | 66.2 ms |
| PP2 | 1 | 64.2 | 114.1 ms |

Separately instrumented stage busy fractions rose from roughly 46/45
percent with one microbatch to 82/81 percent with eight, even as throughput
fell. The same logical payload became more, smaller transfers and smaller
matrix operations. Those busy fractions were measured with instrumentation
and are not components to subtract from headline wall time. The
[PP receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m6b-pp2-a6000-2026-08-28.json)
retains both measurement boundaries.

Per-rank reserved memory fell from 2098 to 1156 MiB in the one-microbatch
case. All eight rows in that case matched single-device greedy tokens;
the smaller-microbatch BF16 cases matched six of eight. FP32 stage and
lifecycle checks supply a stricter reference, while broad BF16 shape changes
retain their numerical limitation. Distribution did not win throughput in
this comparison, but it did reduce the maximum memory required on a device.

## Follow the implementation

| Source at runtime snapshot `e20a348` | Responsibility |
|---|---|
| [distributed.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/distributed.py) | Ranks, collectives, tensor shards, stage ranges, and communication accounting. |
| [loader.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/loader.py) | Select local weight shards and stage-owned checkpoint tensors. |
| [models/qwen3.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/models/qwen3.py) | Local head/channel computation and stage-local layer/cache mapping. |
| [pipeline.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/pipeline.py) | Row slicing, activation sends, ordered logits, and buffer lifetime. |
| [engine.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/engine.py) | Rank-0 scheduling plans, sampled-token broadcast, and mirrored request state. |

The constructor explicitly rejects combining TP, PP, EP, and CP degrees
greater than one. PP is the two-stage path described here; these independent
examples do not qualify a TP-by-PP topology. Combining axes would require
new process groups, placement, communication order, and validation.

TP keeps the residual replicated while splitting heads and channels inside
each layer. PP keeps layers whole while transferring residual activations
between owners. Both can solve a per-device capacity problem while making
the same small workload slower. Choosing a partition requires identifying
what cannot fit or what work dominates, then counting the communication
and synchronization introduced by that particular boundary.

{% include tinyserve-book-nav.html %}
