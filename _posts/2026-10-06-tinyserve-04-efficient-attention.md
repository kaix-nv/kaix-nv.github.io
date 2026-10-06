---
layout: post
math: true
title: 'Tinyserve, Chapter 4: Efficient attention implementation'
date: 2026-10-06 00:00:00 -0700
categories:
- tinyserve
- llm-serving
excerpt: How can attention execute without unnecessary intermediate storage and traffic?
book_chapter: 4
source_revision: 14b0c8dbb975350bbd730ac0c702eafce36eb95c
source_document: docs/book/04-efficient-attention.md
code_revision: e20a34815498560f9226a9e057ade31b2b62743f
---

{% include tinyserve-article-style.html %}
{% include tinyserve-book-nav.html %}

*Chapter 4 · Model Architecture and Computation*

Dense attention specifies a weighted sum over visible values. It does not
require a program to write every score and probability to GPU memory. That
distinction matters most during prefill, where a prompt of length T has a
quadratic number of query/key pairs. A correct implementation can visit the
keys in tiles, maintain a small running summary, and produce the same
mathematical result.

We will derive that summary with four scores, then use it to understand
prefill tiling, decode parallelism, and direct reads from paged KV. Tinyserve
provides readable attention and the adapters to external kernels. Keeping
those responsibilities visible helps explain both the savings and the
numerical differences observed when switching execution paths.

## The intermediate matrix is optional

The reference operation for one head is

$$
\begin{aligned}
S&=QK^\mathsf T/\sqrt d+M,\\
P&=\operatorname{softmax}(S),\\
O&=PV.
\end{aligned}
$$

If Q and K each contain 4,096 positions, S has about 16.8 million entries.
For 16 heads, storing one FP32 score tensor would require 1 GiB. A separate
probability tensor could cost another 1 GiB. This is only intermediate
attention storage, excluding Q/K/V, output, model weights, and the cache.

A direct implementation writes scores, reads them for softmax, writes
probabilities, and reads probabilities for the value product. Most of those
intermediate values are used locally and then discarded. The original
[FlashAttention paper](https://arxiv.org/abs/2205.14135) reorganizes attention
around tiles and on-chip working storage to reduce that traffic. Its
operation remains exact dense attention in real arithmetic; floating-point
execution order can still change the returned values.

Tiling alone is not sufficient. A softmax denominator depends on every
visible key in the row. We need a way to combine partial normalizations
without retaining every score. The required state consists of a maximum,
a denominator, and an unnormalized weighted value sum.

## Accumulate one row across two tiles

Reuse the final head-A row from Chapter 3:

```text
scores:  [1, 0 | 1, -1]
values:  [1,0], [0,2] | [2,1], [-1,1]
```

Split after the first two keys. For any visited set of keys, maintain

$$
\begin{aligned}
m&=\max_j s_j,\\
\ell&=\sum_j e^{s_j-m},\\
z&=\sum_j e^{s_j-m}v_j.
\end{aligned}
$$

The normalized result is `z/ell`. Subtracting the maximum keeps every
exponential at most one, avoiding the overflow that large positive raw
scores could cause.

The first tile has `m=1`, denominator `1+exp(-1) = 1.367879`, and weighted
sum `[1, 2 exp(-1)] = [1, 0.735759]`. Its locally normalized output would
be `[0.731059, 0.537883]`, but we do not finalize that result yet.

The second tile also has maximum one. Its denominator is
`1+exp(-2) = 1.135335`, and its weighted sum is
`[2-exp(-2), 1+exp(-2)] = [1.864665, 1.135335]`. Since both maxima agree,
we can add the two denominators and weighted sums:

```text
m      = 1
ell    = 2.503215
z      = [2.864665, 1.871094]
z/ell  = [1.144394, 0.747476]
```

This is exactly the four-key answer, to the displayed rounding. Averaging
the two locally normalized outputs would be wrong because their
denominators differ. Each tile represents a different amount of softmax
mass.

[![A query accumulates two KV tiles using a maximum, denominator, and weighted value numerator. The merged result matches a single four-key softmax.](/assets/tinyserve/book04-online-softmax.svg)](/assets/tinyserve/book04-online-softmax.svg)

What if a later tile has a larger maximum? Let the accumulated summary be
`(m,ell,z)` and the new tile summary be `(mt,ellt,zt)`. Set

$$
m'=\max(m,m_t),\quad
\ell'=e^{m-m'}\ell+e^{m_t-m'}\ell_t,
$$

$$
z'=e^{m-m'}z+e^{m_t-m'}z_t.
$$

Both contributions now use the same exponential reference point. For
example, merging one old score zero with a new score two rescales the old
weight from one to `exp(-2)` before adding the new weight one. The result
is stable without revisiting the old score.

An empty accumulator has denominator and numerator zero. Production kernels
must also handle an entirely masked tile without evaluating an undefined
`-infinity - (-infinity)` expression. A query with no permitted keys needs
an explicit API contract; the causal model paths here always provide a
valid self or history key for real query rows.

## Turn the recurrence into tiled prefill

Now give a program Bq query rows and Bk key/value rows at a time. Its
score tile has shape `[Bq,Bk]`; its running numerator has shape
`[Bq,dv]`; and it keeps one maximum and denominator per query row.
For each KV tile it computes scores, applies visibility, updates the
summary, and discards the score tile. It writes `[Bq,dv]` output after
the final tile.

The large `[T,T]` intermediate disappears, but the dense comparisons remain.
Different query tiles may reread the same K/V tiles. Choosing Bq and Bk
balances reuse against the amount of fast storage each program needs and
the number of programs that can execute concurrently. Bigger tiles do not
automatically win: extra live values can reduce parallel occupancy or spill
into slower storage.

Causality also operates at two levels. A KV tile entirely in the future of
a query block can be skipped. A tile crossing the diagonal needs per-entry
masking. This preserves the same allowed pairs as the reference triangle;
it is not an approximation that drops inconvenient past tokens.

During long prefill there are many query blocks, so the GPU can distribute
them across its execution units. The score and value products have enough
rows to exploit matrix computation and K/V reuse. The reduction in temporary
traffic can be valuable even though arithmetic still scales quadratically
with prompt length. Fusing softmax with these products saves intermediate
reads and writes; it does not replace the learned attention equation.

Tinyserve's call to PyTorch scaled dot-product attention is an operation
request, not a guarantee that PyTorch materializes S and P. Its backend
may already use a fused kernel. A comparison must identify actual execution
paths before attributing a gain to “using FlashAttention.”

## Decode needs another source of parallel work

One decode request supplies one new query per head and potentially thousands
of keys. There are few query rows to distribute, and the whole history may
be read for little arithmetic per element. Head sharing can reuse K/V, but
a one-request workload may still provide too little parallel work.

Split-KV execution divides the history into several disjoint ranges. Each
program computes a partial summary for the same query against its assigned
range. A second operation merges those summaries with the maximum-rescaling
equations above. The final answer includes every visible key.

Suppose a history has 8,192 keys and four splits. Each program handles 2,048
keys, returning a maximum, denominator, and dv-coordinate numerator. A
reduction reads four summaries rather than the original scores. For one
query head and `dv=128`, an FP32 `(m,ell,z)` representation contains
`4 × (128+2) × 4 = 2,080` bytes across four splits. This is an illustrative
representation; kernels can instead store normalized outputs and
log-sum-exp statistics.

More splits add parallelism but also create intermediate writes, reduction
work, and launches. Short histories or already large batches may not need
them. Split count is consequently an implementation decision dependent on
shape and hardware. Tinyserve's integration delegates the external backend's
kernel planning; it does not contain a new handwritten split-KV attention
kernel in this chapter.

## Ragged queries and paged histories are different layouts

Ragged input packs unequal query sequences without padding. Two fresh
prompts of lengths two and three can share a flat query tensor
`[5,Hq,d]`, with boundaries `[0,2,5]`. Those boundaries keep attention
inside each request. Concatenating tokens does not create one five-token
conversation.

Paged history addresses persistent storage. Consider a request with five
cached tokens, appending four more. Its query positions are 5, 6, 7, and 8.
With page size four and page table `[5,2,4]`, logical positions 0–3 live
on physical page 5, positions 4–7 on page 2, and position 8 on page 4.
Physical IDs are not sorted because allocation order does not define token
order.

The direct paged-append plan uses:

```text
query indptr:           [0,4]      # query-token boundaries
paged-KV indptr:        [0,3]      # page-table boundaries
paged-KV indices:       [5,2,4]    # physical pages in logical order
last-page valid length: [1]        # only position 8 is valid there
```

The per-layer pool has shape `[num_pages+1,2,page_size,Hkv,d]` in
Tinyserve's layout. The extra page is part of the cache implementation;
valid request tables and lengths still determine which entries may be read.
The two middle payload choices are K and V. The query tensor has shape
`[4,Hq,d]` and the result the same shape.

[![A paged reader translates logical KV tiles through page IDs 5, 2, and 4. Query positions 5 through 8 use a shifted causal mask; the gather reader first copies the history.](/assets/tinyserve/book04-paged-reader.svg)](/assets/tinyserve/book04-paged-reader.svg)

A gather reader first copies those pages into a contiguous temporary,
then expands shared KV heads if its attention call requires that. A
direct reader resolves the page table while loading tiles into the
attention computation. Both must read the useful K/V data. The direct
reader avoids an extra full-history materialization and its associated
writes and rereads; it does not eliminate history traffic.

For query row i, the correct causal test is `j <= 5+i`. The first query
must not see positions 6–8, even though all four new K/V entries may already
have been written. The last page's other three slots must never count as
valid history. In the reference gather path, Tinyserve also makes unused
values finite: a zero probability multiplied by NaN is still NaN. Masking
scores alone cannot make poisoned value storage safe.

Allocation and reclamation are separate from this read algorithm. A backend
receives a current valid mapping; it does not decide which request gets a
page or whether a finished prefix should stay cached. Those lifecycle
decisions belong to Chapter 9.

## Follow the backend call

Tinyserve's `AttentionBackend` has two responsibilities: `prepare(ctx)`
prepares shape-dependent metadata before the model forward, and
`attend(...)` computes a layer's output. `Qwen3Attention._paged_attend()`
first writes the new K/V to `ctx.slots`, then calls
`ctx.attention.attend(...)`. It has already projected, normalized, and
rotated Q/K. The backend must not apply RoPE a second time.

`TorchAttentionBackend` uses in-flight K/V for fresh prefill, since those
tensors already cover the whole new history. For chunk append and decode,
it gathers the appropriate layer's cache, establishes valid positions,
expands KV heads, and invokes PyTorch attention with an explicit mask.
The chunk path slices to the actual append endpoint before the call.

`FlashInferAttentionBackend` owns wrappers and planning. Fresh packed
prefill uses the ragged wrapper. Single-token decode uses the paged decode
wrapper. The enabled multi-token append path uses a paged-prefill wrapper
with causal alignment at the end of the visible history. A “prefill” API
name does not mean that the old prefix is recomputed.

`plan_append()` checks `1 <= query_len <= kv_len` and that the page table
has the required number of pages. An exactly full final page has valid
length `page_size`, not zero. The plan is reused for all layers of that
forward, while their K/V payloads remain layer-specific. Length changes,
rollback, or page-table changes require a fresh plan for the next forward.
FlashInfer supplies its kernels; Tinyserve supplies model semantics,
addresses, lengths, and the call sequence.

## Faster reads do not establish full-model equivalence

Changing tile geometry, accumulation precision, or the order of sums can
perturb floating-point output. Comparing attention outputs on identical
rounded Q/K/V isolates a kernel's arithmetic. Comparing full-model logits
also includes projections, normalization, and errors propagated through
later layers. Comparing generated tokens adds a discontinuous argmax or
sampling decision. These are different validation boundaries.

The retained direct-paged experiment is a useful example. With BF16
Qwen3-4B as target and Qwen3-0.6B as draft on one RTX A6000, direct page
addressing and causal isolation passed their attention tests. The
predeclared full-model normalized logit RMS-error limit was 1%, however,
and all 18 teacher-forced comparisons exceeded it. Direct verification's
range was 1.25–2.07%; even the existing gather-verification control failed.
Matching early top-1 decisions and low KL divergence did not turn that
gate into a pass.

The same record retains diagnostic timing: draft-four speculative execution
with direct reads improved over its gather counterpart by paired median
ratios 1.153 for a 128-input/128-output workload and 1.116 for a
2,048-input/32-output workload. These are whole-request ratios from seven
rotating-order repetitions, including prefill, planning, and cleanup.
They are not attention-kernel speedups, and the failed numerical gate
keeps them unqualified for promotion. Optimized ordinary paged/graph
decoding remained faster than either speculative path. The
[committed receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m9c-direct-paged-attention-a6000-2026-09-20.json)
retains the conditions and failed comparisons.

The mechanism still teaches something precise: direct tile loads can avoid
a separate gather, and online summaries can avoid full score storage.
Whether those savings improve a serving path depends on the workload,
planning overhead, other kernels, and the numerical contract that path
must satisfy.

## Source map

| Source | Responsibility |
|---|---|
| [attention.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/attention.py) | Reference reads, wrapper planning, and paged append/decode dispatch. |
| [models/qwen3.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/models/qwen3.py) | Model-owned Q/K/V transformation and common cache writes. |
| [paged.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/paged.py) | Physical pool, page-table gathers, and forward metadata. |
| [backend separation](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/m7d-pluggable-attention-cache.md) | Attention mechanism versus cache lifecycle. |
| [direct paged attention](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/m9c-direct-paged-attention.md) | Addressing example and retained numerical/performance boundary. |

These source links describe runtime `e20a348` and require repository access.
The online-softmax arithmetic and address translation above are complete
without that access.

{% include tinyserve-book-nav.html %}
