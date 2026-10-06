---
layout: post
math: true
title: 'Tinyserve, Chapter 17: Expert and context parallelism'
date: 2026-10-06 00:00:00 -0700
categories:
- tinyserve
- llm-serving
excerpt: How do distributed expert routes and KV shards preserve their local contracts?
book_chapter: 17
source_revision: 14b0c8dbb975350bbd730ac0c702eafce36eb95c
source_document: docs/book/17-expert-and-context-parallelism.md
code_revision: e20a34815498560f9226a9e057ade31b2b62743f
---

{% include tinyserve-article-style.html %}
{% include tinyserve-book-nav.html %}

*Chapter 17 · Distributed Serving*

Two models can exceed one GPU's memory for different reasons. A mixture of
experts may have far more expert weights than one device can hold, even
though each token activates only a few experts. A dense model may fit its
weights comfortably while a long attention history exhausts KV capacity.
Expert parallelism partitions the first object; context parallelism
partitions the second.

Both need communication, but their mathematical boundaries differ. Expert
parallelism sends token activations to selected weight owners and returns
weighted expert outputs. Context parallelism keeps queries local to every
rank, scores disjoint histories, and reconstructs one global attention
normalization. We will work through separate examples because confusing
these mechanisms hides both their correctness requirements and their costs.

## Expert owners and source-token owners

Chapter 7 introduced an MoE layer as a router followed by selected expert
MLPs. For token x, expert selection K(x), and selected routing weights w,

$$
y(x)=\sum_{e\in K(x)}w_e(x)\,\mathrm{MLP}_e(x).
$$

The router decides which functions contribute. Placing those functions on
different devices should not change the selection, weights, or identity
of the token receiving the result.

Use four experts and top-2 routing. Rank 0 owns E0 and E1; rank 1 owns E2
and E3. The surrounding dense attention and residual stream are replicated
in Tinyserve's EP path, so both ranks enter the MoE with the same four
hidden rows. To avoid dispatching every row twice, the implementation gives
rank 0 source ownership of t0 and t1, and rank 1 source ownership of t2
and t3.

An expert owner stores a set of weights. A source-token owner is responsible
for sending and reconstructing a subset of the current inputs. These are
different roles held by the same ranks. They need not imply that the token's
entire request or attention history belongs to that device.

| Source token | Source rank | Selected experts | Selected weights |
|---|---:|---|---|
| t0 | 0 | E0, E2 | 0.75, 0.25 |
| t1 | 0 | E1, E2 | 0.40, 0.60 |
| t2 | 1 | E0, E3 | 0.20, 0.80 |
| t3 | 1 | E2, E3 | 0.50, 0.50 |

There are eight routes from four tokens. Repeating t0 for E0 and E2 is
intentional: those copies will undergo different learned transformations.
It is not duplication of a finished expert result.

[![Four source tokens each select two experts. Routes cross to the owning rank, expert outputs return in route order, and each source applies routing weights before restoring token order.](/assets/tinyserve/book17-expert-routes.svg)](/assets/tinyserve/book17-expert-routes.svg)

## Dispatch and combine without losing identity

Rank 0 begins with route order `[t0/E0, t0/E2, t1/E1, t1/E2]`. Sorting
by global expert gives `[t0/E0, t1/E1, t0/E2, t1/E2]`. The first two
routes belong to rank 0 and the last two to rank 1. Rank 1 similarly sorts
its routes as `[t2/E0, t3/E2, t2/E3, t3/E3]`.

Each rank exchanges route counts before sending activations. The all-to-all
uses variable splits: every participant sends a different contiguous segment
to each destination and receives a segment from each source. Counts define
the buffer lengths even when a destination receives zero rows. The first
all-to-all delivers hidden rows to their expert owners.

At rank 0, E0 receives t0 and t2, while E1 receives t1. At rank 1, E2
receives t0, t1, and t3, while E3 receives t2 and t3. Received storage is
initially grouped by source rank, so the implementation collects segments
for the same local expert before calling its MLP. A zero-route expert
needs no MLP call.

The reverse all-to-all returns computed rows to their source-token owners.
Its send and receive split roles are swapped. The source inverts its
original sort, restoring the top-2 route slots for each token, then applies
the retained weights. If E0(t0) is `[2, 0]` and E2(t0) is `[0, 4]`,
t0's combined output is

$$
0.75[2,0]+0.25[0,4]=[1.5,1].
$$

Applying weights to returned rows before recovering their token/route
association could multiply a valid expert result by another route's weight.
The shapes would still look plausible. Correct dispatch needs a reversible
mapping among token index, top-k slot, packed position, and expert ID.

Finally, Tinyserve all-gathers combined source slices so both ranks regain
the original `[4, H]` residual order. This final gather is a consequence of
keeping subsequent dense layers replicated. It is not mandatory for every
possible EP design. For an odd token count, source slices differ in length;
the implementation pads them to an equal gather width, then removes those
extra rows using the known source ranges.

## What expert parallelism saves and costs

The Qwen3-30B-A3B path has 128 experts and selects eight per token. EP=2
assigns experts 0–63 to rank 0 and 64–127 to rank 1. Non-owned expert
modules have no parameters in that rank's state dictionary, and the loader
reads only its target checkpoint keys. Attention, routers, normalization,
embeddings, and endpoint weights remain replicated. KV also remains
replicated, so this placement expands expert-weight capacity rather than
long-context capacity.

With N tokens, K selected experts, hidden width H, and s bytes per element,
dispatch describes N K H s activation bytes before accounting for which
routes stay local and the communication algorithm. Returned expert outputs
have the same shape. Route counts and the final token gather add traffic.
As K increases, fewer active parameters than the full model can still mean
many activation routes per token.

Load balance matters because all ranks must finish their contributions.
Our example assigns three routes to rank 0 and five to rank 1. Even within
rank 1, E2 and E3 receive different row counts. Larger batches can improve
expert GEMM sizes, but the router follows the checkpoint; the implementation
does not drop tokens or change selections to equalize work.

The retained BF16 checkpoint occupied 56.871 GiB, including 54.000 GiB of
expert weights and 2.871 GiB of replicated weights. Each EP rank therefore
held 29.871 GiB of parameters, with 32.902 GiB peak reserved in the real-
model check. One 47.4-GiB device could not hold the complete BF16 artifact,
so there is no valid one-device latency row for that artifact.

On two RTX A6000s, eager paged gather attention, private KV, four greedy
outputs, and three repeats, batch one achieved 5.05 useful tokens/s with
168 ms median ITL. Batch eight reached 18.95 tokens/s with 362 ms median
ITL. More rows improved aggregate throughput while each request advanced
less frequently. At batch eight, the mean per-layer maximum-to-mean expert
load ratio remained 8.90. Activating more experts is not synonymous with
balanced work. The
[EP measurement record](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m6c-ep2-a6000-2026-08-28.json)
keeps the capacity and serving observations separate.

The distributed MoE executes ordinary expert MLP calls with variable route
splits. It does not automatically inherit every later single-device grouped
expert kernel, and CUDA graphs are disabled for this path. Local expert
optimization and cross-device route ownership are distinct integration tasks.

## Context parallelism partitions another tensor

Now return to a dense model. Each rank holds the same weights and computes
the same query, but persistent KV pages have one physical owner. With
CP degree R, Tinyserve maps a global physical page ID b as

$$
\begin{aligned}
\mathrm{owner}(b)&=b\bmod R,\\
\mathrm{localPage}(b)&=\lfloor b/R\rfloor.
\end{aligned}
$$

For a twelve-token history and four-token pages, a block table `[0, 1, 2]`
places positions 0–3 on rank 0 local page 0, positions 4–7 on rank 1
local page 0, and positions 8–11 on rank 0 local page 1. Both ranks keep
the global table and allocator metadata. Only an owning rank writes the
persistent bytes for a page; the other rank discards those writes.

Logical order must survive local compaction. Rank 0's second local page
still means logical positions 8–11, not 4–7. Its local attention metadata
therefore retains each page's logical position for validity and causality.
A query at position 9 may see rank 0's keys through 9 and rank 1's keys
through 7; it must not see positions 10 and 11 merely because their page
is physically present.

This arrangement replicates QKV projection work but shards storage and
history reads. It does not help a model whose weights already exceed one
device, and it is not expert dispatch: no routing network chooses which
KV owner contributes. Every owner with visible keys participates in one
attention result.

## One query and two numerical softmax summaries

Consider one query whose four already-scaled attention scores are
`[0, ln(2), ln(3), ln(4)]`, with scalar values `[10, 20, 30, 40]`.
Put the first two keys on rank 0 and the last two on rank 1. The exact
global exponential weights are proportional to `[1, 2, 3, 4]`, giving

$$
o=\frac{1(10)+2(20)+3(30)+4(40)}{1+2+3+4}=30.
$$

Independent local softmax outputs would be `50/3 ≈ 16.667` and
`250/7 ≈ 35.714`. Averaging them gives about 26.190, which is wrong.
They have different normalization masses and cannot be combined as equally
weighted answers.

[![Rank zero summarizes scores zero and log two, while rank one summarizes log three and log four. A global maximum and summed numerator/denominator reconstruct output thirty without transferring KV.](/assets/tinyserve/book17-context-softmax.svg)](/assets/tinyserve/book17-context-softmax.svg)

Online softmax supplies the correct summary. For each shard r, define a
maximum m_r, denominator d_r, and numerator n_r:

$$
\begin{aligned}
m_r&=\max_{j\in r}s_j,\\
d_r&=\sum_{j\in r}e^{s_j-m_r},\\
n_r&=\sum_{j\in r}e^{s_j-m_r}v_j.
\end{aligned}
$$

Rank 0 has `(ln(2), 1.5, 25)`; rank 1 has `(ln(4), 1.75, 62.5)`.
Choose global m = ln(4). Rescale rank 0's summary by `exp(ln(2)-ln(4))
= 0.5`; rank 1's factor is one. The merged denominator is
`0.5 × 1.5 + 1.75 = 2.5`, and numerator is
`0.5 × 25 + 62.5 = 75`. The result is `75 / 2.5 = 30`.

This is the same stable merge used across tiles in efficient attention,
now across devices. The communicated numerator becomes a vector of head
width D, while maxima and denominators remain scalars per query head.
Neither raw score arrays nor KV need cross the boundary to perform this
mathematical merge.

## How Tinyserve implements the reduction

The readable CP backend uses an equivalent ordering. It first all-reduces
local maxima with MAX, obtaining m on every rank. Then each rank directly
forms `exp(scores - m)`, its denominator, and its weighted value numerator.
It concatenates numerator and denominator into one SUM all-reduce and
divides the resulting global numerator by global denominator.

There is one MAX and one SUM per attention layer per forward. Statistics
follow query dimensions; their size does not grow with the number of cached
keys for a fixed query batch. Prefill has many query positions, however, so
the SUM payload can still be large. For Q query positions and Hq heads,
the FP32 payload sizes are approximately `4 Q Hq` bytes for MAX and
`4 Q Hq (D + 1)` for SUM, per rank before protocol overhead.

A rank can own no visible keys for a query. Its maximum is negative infinity
and its numerator and denominator contributions must be zero. Another rank
provides the finite global maximum for a real query with valid history.
Causal and valid-length masks prevent unused page tails and local packing
padding from becoming attention contributions.

The implementation performs score and reduction arithmetic in FP32, then
casts output back to query dtype. It materializes score and probability
tensors; it is not a fused distributed FlashAttention kernel. FlashInfer
and graph replay remain disabled for CP, whose variable local-page metadata
and global normalization differ from a complete local attention call.

## Context capacity is not peak execution memory

For the 28-layer, eight-KV-head, width-128 BF16 model, 8192 cached tokens
require 0.875 GiB of useful KV storage. At equal global capacity, CP=2
places 4096 token slots, or 0.4375 GiB, on each device, excluding scratch
overhead. Each rank still retains the model weights. Alternatively, keeping
8192 local slots per rank doubles global token capacity.

The historical two-A6000 CP comparison used Qwen3-0.6B BF16, eager execution,
private KV, four output tokens, and three repeats. At a 2048-token prompt,
median TTFT was 56.2 ms for CP1 and 286.8 ms for CP2; median ITL was
26.0 versus 50.9 ms. This explicit distributed backend was slower than
the single-device gather/SDPA path.

At 4096 prompt tokens, peak reserved memory rose from 2.648 GiB on CP1
to 3.859 GiB per CP2 rank despite the smaller persistent pool. FP32
attention intermediates and a different backend outweighed that storage
saving. The [CP receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m6d-cp2-a6000-2026-08-29.json)
therefore supports a precise placement claim, not a production long-context
memory or throughput claim. Three-repeat synthetic prompts and four outputs
also do not measure long-running service tails or answer quality.

## Follow the implementation

| Source at runtime snapshot `e20a348` | Responsibility |
|---|---|
| [models/qwen3_moe.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/models/qwen3_moe.py) | Source slices, route packing, expert execution, inversion, and weighted combine. |
| [distributed.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/distributed.py) | Expert/page owners, variable all-to-all, token gather, and CP reductions. |
| [paged.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/paged.py) | Owner-only KV writes and local tables retaining logical positions. |
| [attention.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/attention.py) | Global MAX/SUM normalization over local KV. |

The two-GPU paths are independent: EP partitions Qwen3 MoE experts, while
CP partitions ordinary dense attention history. They do not qualify combined
TP/PP/EP/CP topologies or the single-device Kimi expert backend. Controlled
FP32 and BF16 checks cover their stated placement and numerical contracts;
distributing arithmetic does not establish arbitrary BF16 token identity.

Expert parallelism must return every route to the right token and weight.
Context parallelism must normalize every key owner against the same global
softmax mass. Keeping those invariants distinct explains both why the
algorithms are correct and why their communication cannot be interchanged.

{% include tinyserve-book-nav.html %}
