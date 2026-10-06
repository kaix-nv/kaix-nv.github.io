---
layout: post
math: true
title: 'Tinyserve, Chapter 5: Sparse attention'
date: 2026-10-06 00:00:00 -0700
categories:
- tinyserve
- llm-serving
excerpt: Which history should a query read and what does selection cost?
book_chapter: 5
source_revision: 14b0c8dbb975350bbd730ac0c702eafce36eb95c
source_document: docs/book/05-sparse-attention.md
code_revision: e20a34815498560f9226a9e057ade31b2b62743f
---

{% include tinyserve-article-style.html %}
{% include tinyserve-book-nav.html %}

*Chapter 5 · Model Architecture and Computation*

Dense attention lets each query read every causally visible key. Efficient
kernels reduce the overhead of doing that work, but they still evaluate
the dense operation. Sparse attention changes the set of positions that a
query may read. A local window can avoid distant history; a selector can
retrieve a few older blocks; a model can combine local and global paths.

The difficult question is not how to draw fewer dots in a mask. It is how
to choose those dots, reach their physical KV storage, and preserve the
model behavior that matters while spending less time overall. We will
follow an eight-token example through that complete chain.

This chapter develops background and a possible design boundary. Runtime
`e20a348` has no qualified Tinyserve sparse-attention implementation. In
particular, `ModelConfig.is_sparse_layer()` selects mixture-of-experts
feed-forward layers, not sparse attention. The hypothetical reader below
is an explanatory design, not an available backend or a measured speedup.

## Restrict a single attention row

Let a query at logical position 7 have eight causally visible keys. Use
these synthetic scores and scalar values:

| Key position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scaled score | 0 | 2 | 0 | -1 | 1 | 0 | 2 | 1 |
| Value | 1 | 4 | 2 | -1 | 3 | 0 | 5 | 2 |

Dense attention exponentiates all eight scores, normalizes over all eight,
and returns approximately `3.507891`. A two-token causal window retains
only positions 6 and 7. Its output is

$$
o_{\mathrm{window}}=
\frac{e^2\cdot5+e^1\cdot2}{e^2+e^1}
\approx4.193176.
$$

Now divide the history into two-token selection blocks: block 0 holds
positions 0–1, block 1 holds 2–3, block 2 holds 4–5, and block 3 holds
6–7. Suppose a selector chooses old block 0 and always includes the
current local block 3. Restricted attention reads positions `[0,1,6,7]`
and returns approximately `3.943367`.

[![One eight-token history under dense attention, a two-token window, and selected old plus recent blocks. Each policy renormalizes over its retained positions and produces a different scalar output.](/assets/tinyserve/book05-eight-token-masks.svg)](/assets/tinyserve/book05-eight-token-masks.svg)

The selected blocks contain the two highest-scoring positions, but their
answer still differs from dense attention. The omitted positions have
nonzero probability and values. Even a selector that finds high scores
does not make dropping the remaining probability mass algebraically exact.

For retained set R, sparse attention computes

$$
o_R=\frac{\sum_{j\in R}e^{s_j}v_j}
{\sum_{j\in R}e^{s_j}}.
$$

This is a new normalization. Taking dense probabilities and simply zeroing
omitted terms would give a different result. Likewise, averaging separate
local and selected attention outputs is not equivalent to a single softmax
over their union. A model may deliberately learn separate branches and a
mixing gate, but that is another architectural equation.

## Static patterns choose positions without scoring the full history

A causal window of width w lets position t see
`max(0,t-w+1)` through t. The width includes the current token. With eight
tokens and `w=2`, full prefill evaluates 15 permitted pairs: one for the
first position, then two for each of seven remaining positions. Dense
causal prefill evaluates `8×9/2 = 36` pairs. For long sequences and fixed w,
the useful attention work grows as `T w` rather than `T^2`.

The window encodes a locality assumption. Information farther away can
still propagate through intermediate hidden states across layers, but a
single layer cannot directly retrieve an arbitrary distant value. For a
stack of L simple causal window layers of width w, a final representation's
maximum receptive distance grows to roughly `L(w-1)` positions. That
indirect route is not equivalent to giving the last query direct access to
every past token.

A local/global pattern preserves certain important positions for broader
communication. For a causal decoder, a query may read a designated global
position only if that position is not in its future. “Global” does not
override causality. An initial instruction token could remain visible
alongside the recent window, adding a small fixed number of reads per query.
Longformer provides a primary example of combining local windows and global
attention; its task-specific patterns must be distinguished from this
causal teaching example. [Longformer paper](https://arxiv.org/abs/2004.05150).

Other static patterns include strided positions and selected random
connections. BigBird combines local, random, and global connectivity.
Such patterns demonstrate that sparse attention is a family of communication
graphs, not just a shorter context window. Results for one trained pattern
do not establish quality for another pattern substituted at inference.
[BigBird paper](https://arxiv.org/abs/2007.14062).

Static patterns make selection cheap: positions follow a rule known before
examining the query's content. They can also make memory access regular.
Their limitation is that the rule may omit the token the current query
needs. That leads to content-dependent selection.

## Select blocks using the current query

A content selector might compare a reduced query with a representative of
each old block, choose a small number of blocks, and then run ordinary
attention over all fine-grained keys in those blocks. Compression is used
for selection; the final value retrieval can still use the original KV.

For our eight-token history, suppose each two-token block has a compact
representative. Four coarse comparisons produce selection scores, and the
selector chooses block 0 while retaining local block 3 unconditionally.
The four fine-grained scores in the selected set then determine the actual
weighted value sum. Coarse selection scores are not reused as if they were
the token probabilities.

An implementation must define what “choose blocks” means precisely. Is the
selection per query token, per query head, per group of heads, or per query
tile? Can different layers choose differently? How many old blocks are
retained in addition to the local region? Sharing selection across a query
tile reduces selector and metadata work, but all rows must then use an
appropriate causal mask within the selected blocks.

Block selection aligns the unit of useful work with a tiled attention
kernel. Selecting individual scattered tokens can minimize the mathematical
pair count while creating poorly coalesced loads and too little work per
program. Selecting an entire block reads some less useful tokens, but
gives the kernel regular tiles. The best block size balances retrieval
precision, selection cost, and execution efficiency.

Native Sparse Attention is a trained architecture that combines compressed,
selected, and sliding-window attention paths with hardware-aware blocking.
It illustrates why selector, attention equation, and training must be
considered together. Our four-position union is a smaller explanatory
construction, not an implementation of that architecture. [Native Sparse
Attention paper](https://arxiv.org/abs/2502.11089).

## Translate selected positions into physical storage

Selection blocks and cache pages need not have the same size. Keep
two-token selection blocks, but store KV in four-token cache pages. Let
the request's page table be `[9,2]`. Logical positions 0–3 occupy physical
page 9, and positions 4–7 occupy page 2.

The selector returns logical selection blocks `[0,3]`, which expand to
logical positions `[0,1,6,7]`. For page size four, the physical token slot
for position p is

$$
\operatorname{slot}(p)=4\,\operatorname{table}[\lfloor p/4\rfloor]+(p\bmod4).
$$

The corresponding physical slots are `[36,37,10,11]`. Sorting these slots
numerically would change their storage traversal order, not their logical
meaning. An attention implementation can choose a traversal order, but
must retain the correct position information wherever masking or position
bias depends on it. The selected keys already carry their correct RoPE
transform; fetching position 6 into compact row 2 must not rotate it as
though it were original position 2.

[![Selection blocks expand to logical positions before the request page table maps those positions to physical KV slots. A selector chooses reads; it does not own cache eviction.](/assets/tinyserve/book05-selection-addressing.svg)](/assets/tinyserve/book05-selection-addressing.svg)

A possible reader contract is therefore:

```text
query + selection metadata
    -> logical block IDs for this request and layer
    -> deduplicated valid logical token ranges
    -> page-table translation and intra-page offsets
    -> restricted causal attention with one normalization
```

Deduplication matters. If an old selected block overlaps the local window,
including its tokens twice changes the denominator and doubles their
contribution. A partial final block needs a valid-length bound. For a
prefill query tile, some selected blocks can contain future tokens for
earlier rows; selection does not remove the need for causal masking.
Every request also needs its own selection and address context. A physical
page ID alone is not a valid cross-request history reference.

## Skip reads without discarding history

The query at position 7 did not read blocks 1 or 2. A query at position 8
might select either one. Dynamic sparse reads therefore do not justify
freeing the unselected pages. The complete history may remain resident,
with only the per-query traffic reduced.

Eviction is a stronger policy. It says future queries will no longer be
able to retrieve the discarded keys and values, unless some other storage
tier or recomputation path restores them. Evicting based on this query's
importance scores can be wrong for the next query. Capacity savings and
read-traffic savings need separate measurements.

A model designed exclusively around bounded local windows can sometimes
reclaim history that no future layer operation will use. Even then, a
hybrid model with global or dense layers may still retain growing histories
for those layers. Memory accounting must sum the actual layer families.
It cannot multiply a local-window bound by every layer indiscriminately.

This is also why prefix reuse needs a clear state contract. Reusing dense
KV for an identical prefix may be straightforward, while learned selector
summaries or any irreversible evictions add state that must be restored
consistently. Sparse attention does not remove request ownership or cache
lifetime from the serving system.

## Count selection and irregular access in the cost

Let T be visible history length, b the selection-block width, k the number
of selected old blocks, w the number of local tokens, and ds the width of
coarse selection vectors. An illustrative selector scans about `T/b`
representatives, costing roughly `O((T/b) ds)` work per selection unit.
Fine attention then visits at most `kb+w` positions before deduplication,
with score/value work proportional to their count and the head widths.

The total includes more than those products:

$$
\begin{aligned}
t_{\mathrm{sparse}}\approx{}&t_{\mathrm{select}}+t_{\mathrm{metadata}}\\
&+t_{\mathrm{read}}+t_{\mathrm{restricted\ attention}}.
\end{aligned}
$$

This is a bookkeeping model, not an assertion that all GPU operations are
strictly serial. It identifies costs a pair-count comparison omits.
Representative construction may happen when KV is appended; top-k
selection needs work and temporary storage; irregular pages may reduce
memory efficiency; and a small selected set may underutilize the GPU.

At eight tokens, implementing our selector would almost certainly be an
exercise in overhead rather than a plausible speed experiment. The example
exists to make the semantics visible. At long contexts, skipping most
fine-grained reads can be attractive, but the selector's scan can itself
become significant. If it first computes every full attention score to
decide which scores to keep, much of the expensive work has already
happened.

Prefill and decode also favor different designs. Prefill has many queries
and can amortize shared selection and block metadata. Decode has few
queries and may be dominated by selecting and fetching scattered history.
The relevant result is latency or throughput for the intended phase and
workload, after numerical and quality checks, not the percentage of zeros
drawn in the mask.

## Decide what correctness means

For a model trained with a specific sparse pattern, reproducing that
pattern is model correctness. A dense implementation of the same mask can
serve as a slow numerical oracle. The sparse kernel should match it within
the chosen tolerance while reading only valid selected locations.

For a dense checkpoint modified at inference, the mask is an approximation.
No address-translation test can show that the missing information is
unimportant. Evaluation must cover model quality and the long-context
behaviors the change is intended to preserve, as well as kernel output
differences on fixed inputs.

One useful local diagnostic is retained dense probability mass. If R
retains mass alpha under the original dense distribution, then
`o_dense = alpha o_R + (1-alpha) o_omitted`. Consequently,

$$
\|o_R-o_{\mathrm{dense}}\|
=(1-\alpha)\|o_R-o_{\mathrm{omitted}}\|.
$$

Our selected blocks retain about 78.43% of the dense probability mass.
That explains why selecting both score-two positions is insufficient for
equality. If all value vectors have norm at most M, the error is bounded
by `2M(1-alpha)`. This is a one-layer bound under the stated norm
assumption, not a bound on final text quality or the effect of repeated
approximation through many layers.

Independent checks would isolate causal visibility, selection overlap,
page translation, poisoned unused storage, and output equivalence to a
dense masked oracle. Only after those pass does an experiment meaningfully
test whether selection overhead is repaid by skipped reads. Tinyserve
has not carried out that qualification for a sparse-attention backend at
the pinned runtime.

## Source map

| Source | What it establishes |
|---|---|
| [config.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/config.py) | `is_sparse_layer()` chooses MoE feed-forward layers. |
| [models/qwen3_moe.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/models/qwen3_moe.py) | Expert sparsity retains the ordinary attention implementation. |
| [attention.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/attention.py) | Existing dense attention reader boundary. |
| [paged.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/paged.py) | The page-table address translation a future reader would consume. |

The local source boundary is runtime `e20a348`; those links require
repository access. The hypothetical sparse reader and all numerical
examples are fully specified above and make no Tinyserve implementation
or performance claim.

{% include tinyserve-book-nav.html %}
