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
source_revision: 9de1765eb5ecc167dd5b46effba7792c5ada1a58
source_document: docs/book/05-sparse-attention.md
code_revision: 9de1765eb5ecc167dd5b46effba7792c5ada1a58
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

The chapter starts with the broader design space, then follows M13's bounded
implementation: a fixed causal window over paged KV. Dynamic block selection
remains background, not an implemented backend. M13 is qualified within the
bounded tests below, but stays opt-in: its measured speed benefits do not
preserve dense-model quality. `ModelConfig.is_sparse_layer()` selects mixture-of-experts
feed-forward layers, not sparse attention.

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

## Follow one windowed chunk through the implementation

M13 exposes `LLM(..., attention_window=W)`. `None` keeps full attention;
positive integer `W` includes the current token. The option deliberately
requires private KV, unquantized weights and KV, eager single-device dense
Qwen in FP32 or BF16, and greedy requests. Prefix sharing, graph replay,
mixed M12 execution, speculation,
disaggregation and distributed execution are not supported combinations.

Take a sequence whose first eight tokens are already cached. The scheduler
now selects the three-token chunk `[8,11)`, and `W=4`. The engine computes
only the new tokens at absolute positions 8, 9 and 10. Their windows differ:

| Query position | First allowed key | Last allowed key |
|---|---:|---:|
| 8 | 5 | 8 |
| 9 | 6 | 9 |
| 10 | 7 | 10 |

The Torch reader gathers their union, keys 5 through 10. It then masks each
row separately. Merely taking the final four keys, 7 through 10, would lose
valid keys for query 8 and would expose future keys unless also masked.

[![Three query windows share a gathered union of keys, then map through a noncontiguous block table. Old pages remain allocated and absolute positions do not change.](/assets/tinyserve/book05-windowed-chunk.svg)](/assets/tinyserve/book05-windowed-chunk.svg)

The code path makes responsibility explicit:

1. `LLM._paged_chunk_prefill()` supplies the original `start`, `end`, full
   block table and absolute RoPE positions. It does not truncate history.
2. `TorchAttentionBackend.prepare()` computes logical bounds, physical
   token slots and the row masks once for this forward, shared by all layers.
3. Each model layer writes its new K/V through the existing cache path.
   `PagedKVCache.gather_slots()` then reads only the prepared token addresses.
   It does not load every historical page and slice the resulting tensor.
4. Attention normalizes over each row's permitted keys. The engine advances
   `num_cached` from 8 to 11. All three history pages remain owned.

With page width four and block table `[9,2,7]`, keys `[5,6,7,8,9,10]` map
to slots `[9,10,11,28,29,30]`. Physical page 9 holds old keys 0 through 3;
this chunk does not read it, but the allocator still counts it as occupied.
The next decode query at position 11 needs only keys 8 through 11. Request
completion or cancellation—not the attention window—releases its pages.

Whole-prompt prefill uses in-flight K/V with a causal band mask. Left-padded
batches intersect that band with the existing padding mask. If `W` covers
the entire fresh row, the original causal kernel is already the correct
operation and is retained. Cached chunks
use the gathered union above; ragged decode gathers each request's last
`min(length,W)` tokens and fills temporary row padding with finite zeros.
No padded address points into unwritten tails or another request's history.

The FlashInfer adapter expresses the same policy as `window_left=W-1` for
native ragged prefill and windowed paged decode. Its block tables still
describe the complete cache. When every decode row fits inside `W`, it
retains the original full-context reader; once any row exceeds `W`, the
whole decode batch uses the tensor-core window reader. That boundary does
not change the attention equation: before it, there are no old keys to
exclude. Ordinary chunks retain the readable Torch path. This is an adapter
to existing native kernels, not a new CUDA implementation.

Why distinguish the native readers? In the installed FlashInfer 0.6.14,
the CUDA-core decode path can load excluded values before multiplying them
by zero attention weights. Poisoned old KV then exposes `0 * NaN = NaN`.
The tensor-core window path passes this poisoned-exclusion check; the test
alone does not measure physical memory traffic. Conversely, changing the
BF16 kernel when the window excludes
nothing can introduce rounding differences that accumulate across model
layers. Preserving the original full-context path addresses that separate
compatibility problem. The two failed qualification attempts are retained
in the [first](https://github.com/kaix-nv/tinyserve/blob/9de1765eb5ecc167dd5b46effba7792c5ada1a58/docs/benchmarks/m13-windowed-first-gate-a6000-2026-10-10.json)
and [second](https://github.com/kaix-nv/tinyserve/blob/9de1765eb5ecc167dd5b46effba7792c5ada1a58/docs/benchmarks/m13-windowed-second-gate-a6000-2026-10-10.json)
receipts; neither failure was repaired by widening a tolerance.

There are useful limits to this first implementation. When `W<T`, Torch fresh
prefill still constructs a quadratic mask, and chunk attention covers a union of
up to `W+Q-1` keys for `Q` queries. Neither should be advertised as a
specialized `Q×W` sparse prefill kernel. Decode cuts selected token reads,
but resident KV capacity and preemption pressure are unchanged.

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

Independent checks isolate causal visibility, page translation, poisoned
unused storage, and output equivalence to a dense implementation of the same
mask. Only after those pass does an experiment meaningfully test whether
selection overhead is repaid by skipped reads. M13's frozen protocol applies
these checks to its fixed window; it does not qualify the hypothetical
content-dependent selector or its selection-overlap handling.

## What the M13 measurements establish

These are local RTX A6000 measurements on Qwen3-0.6B and Qwen3-8B, not a
general recommendation to replace a dense model's attention. The window,
inputs, numerical budgets and timing protocol were frozen before candidate
execution. Runtime revision `c91f5b1` passed without changing those budgets.

### Correctness of the selected operation

The [numerical receipt](https://github.com/kaix-nv/tinyserve/blob/9de1765eb5ecc167dd5b46effba7792c5ada1a58/docs/benchmarks/m13-windowed-qualification-a6000-2026-10-10.json)
records 350 passing same-mask attention checks: FP32 and BF16 on the small
model's geometry, and BF16 on the 8B geometry. Windows 1, 15, 16, 17 and
512 cover page boundaries, noncontiguous page order, chunks, unequal rows,
and poisoned excluded storage. The logical-tensor oracle constructs each
query's permitted keys independently of the candidate's address code.

All 15 full-context model controls also pass, with bitwise-identical logits
and valid KV. Those controls verify compatibility when the window excludes
nothing. They do not establish that an active window preserves dense-model
outputs; that is a different experiment.

The broader [regression run](https://github.com/kaix-nv/tinyserve/blob/9de1765eb5ecc167dd5b46effba7792c5ada1a58/docs/benchmarks/m13-windowed-regression-a6000-2026-10-10.json)
has 1,023 passing tests, including all 167 M13 tests. Four existing M12 BF16
old-layout compatibility tests still fail; 15 tests are skipped and four
requiring an unavailable model are deselected. This is not an all-green
suite; the receipt preserves those boundaries and the earlier failed attempt.

### The quality cost of dropping old keys

For each model/backend pair, the quality diagnostic compares dense attention
with `W=512` on exactly 256 teacher-forced next-token predictions. Four
nonrepeated prose records are scored at context lengths 2,048 and 4,096,
with 32 target tokens each. Every distribution is finite and accounted for.
The backend arms are separate; their results are not averaged together.

| Model | Reader | Dense NLL | Window NLL | Mean KL, dense to window | Top-1 agreement |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | Torch gather | 3.718 | 7.008 | 3.378 | 32.4% |
| Qwen3-0.6B | FlashInfer | 3.726 | 7.011 | 3.366 | 32.4% |
| Qwen3-8B | Torch gather | 3.051 | 4.280 | 1.480 | 46.5% |
| Qwen3-8B | FlashInfer | 3.049 | 4.282 | 1.479 | 46.9% |

NLL is negative log probability of the actual next token, in nats/token;
larger is worse. KL measures the change in the full next-token distribution.
Top-1 agreement asks whether its highest-probability token stays the same,
not whether a generated answer is correct. The substantial NLL increases
show why a correctly implemented window is not a quality-preserving drop-in
optimization for these dense checkpoints.

The separate one-word password probe is inconclusive as a retrieval test.
Every arm scores `0/16` under its frozen whole-response matcher, including
the dense controls. Some dense answers begin with the correct password and
then continue with unrelated text; rejecting that response does not prove
the model could not retrieve the password. Each 8B windowed arm also has
one whitespace-only decoded response. All 128 responses contain 32 generated
token IDs. The harness did not retain finish reasons, so this is an
at-budget count, not a measured rate of length-stopped responses.

The [quality receipt](https://github.com/kaix-nv/tinyserve/blob/9de1765eb5ecc167dd5b46effba7792c5ada1a58/docs/benchmarks/m13-windowed-quality-a6000-2026-10-10.json)
retains every answer, original score, coverage check and logit-artifact hash.
There is no post-hoc prefix regrading, longer output budget, or tuned window.
These project-specific fixtures have no external benchmark corroboration.
The teacher-forced diagnostic establishes a quality cost on these inputs;
the answer probe does not establish retrieval preservation.

### Fewer reads help most when there is enough history and concurrency

The reader comparison holds the attention equation fixed. Its baseline
gathers the full history, zeros excluded copies and applies the same
`W=512` mask. The candidates use exact-slot Torch gathers or native
FlashInfer window reads. This is not a comparison against a dense
FlashInfer kernel, and it does not isolate memory traffic from the other
reader costs.

All reader arms first pass the independent same-mask numerical gate.
Timing then includes context construction, planning, address translation,
gathering and attention, but excludes projections, KV writes, and initial
workspace allocation. Each measurement repeats the same fixture 20 times;
two warmup pairs precede six measured pairs with alternating execution order.
The table shows the median **paired baseline/candidate time ratio**, with
the full measured minimum–maximum in parentheses. Above 1 means faster.

| Qwen3-8B head geometry | Batch | Exact-slot Torch | FlashInfer window reader |
|---|---:|---:|---:|
| 2,048 cached tokens | 1 | 0.75 (0.65–0.81) | 0.99 (0.85–1.62) |
| 2,048 cached tokens | 8 | 1.86 (1.72–2.02) | 1.48 (1.44–2.72) |
| 8,192 cached tokens | 1 | 1.15 (1.10–1.29) | 1.22 (1.11–1.68) |
| 8,192 cached tokens | 8 | 9.56 (6.65–9.99) | 5.23 (4.27–8.90) |

At 8,192 tokens and batch 8, the full-gather baseline takes about 7.43 ms
per reader call, versus 0.78 ms for exact-slot Torch and 1.42 ms for
FlashInfer. At 2,048 tokens and batch 1, selected-slot Torch is slower despite
reading fewer keys. Address preparation and small-kernel overhead are possible
costs, but this measurement does not attribute time among them.
The spread is part of the result, not a reason
to discard slower pairs. These are repeated-fixture reader measurements,
not a universal ranking of attention kernels.

### A reader improvement is not the whole serving improvement

Next, both serving arms use `W=512`, the same frozen 2,048-token prompts,
512-token prefill chunks, private pages and eager execution. This comparison
changes only the backend from the selected-slot Torch path to FlashInfer;
it does **not** use the full-gather baseline above. Timing includes the
request lifecycle and all model layers, but excludes model loading and
tokenization. Planning is shared across layers in a forward.

| Model | Batch | Torch median time | FlashInfer median time | Paired time ratio, median (range) |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 1 | 0.967 s | 0.896 s | 1.09 (1.02–1.12) |
| Qwen3-0.6B | 4 | 1.660 s | 1.542 s | 1.08 (1.07–1.12) |
| Qwen3-8B | 1 | 1.567 s | 1.433 s | 1.09 (1.08–1.11) |
| Qwen3-8B | 4 | 3.431 s | 3.152 s | 1.09 (1.08–1.10) |

Every request produces 32 tokens and ends at its length limit. The single-
request pairs produce identical token IDs; the four-request pairs do not.
Thus the work counts match, but this is not identical-output qualification.
Passing a BF16 numerical budget does not guarantee identical greedy choices
through an entire generated sequence. The median ratio is computed within
each pair, so it need not equal the ratio of the two marginal medians.

The [performance receipt](https://github.com/kaix-nv/tinyserve/blob/9de1765eb5ecc167dd5b46effba7792c5ada1a58/docs/benchmarks/m13-windowed-performance-a6000-2026-10-10.json)
retains every warmup and measured pair, actual token IDs, termination reasons
and telemetry provenance. These serving ratios measure a separate backend
substitution, not how the full-gather-to-window reader gain carries through
end to end. In general, optimizing attention cannot remove projection, MLP,
scheduler or other model costs; measuring that dilution requires matching
the baseline substitution at both levels. Resident KV is unchanged here.

### Check that the default path has not become a different experiment

The unchanged full-attention Qwen3-8B standard suite uses graph-enabled
decode, a prefill token budget of 8,192, no prefix sharing and five measured
repeats per condition. Its output throughput is 37.99, 283.57 and 907.93
tokens/s for 128-token prompts and 128-token outputs at batches 1, 8 and 32.
For 2,048-token prompts and 32-token outputs, the corresponding values are
26.70, 68.18 and 80.67 tokens/s. Every measured request reaches its specified
output count; the small arithmetic smoke check also passes.

Compared with the retained October 5 full-attention Tinyserve run, the six
throughput deltas range from -0.45% to +0.11%. These are separate runs under
dynamic clocks, not a paired demonstration of improvement or equivalence.
The October 10 results are also calibrated against the dated vLLM,
llama.cpp and Ollama runs after checking reference revisions, dependencies,
model artifacts and hardware conditions. Their raw comparisons and clock
ranges are in the same performance receipt. FreeToken's short-output
failure remains excluded; no new reference-engine run is claimed.

Tinyserve's engine-internal timing and the reference engines' streaming
HTTP timing have different boundaries, and their cache policies differ.
This calibration checks the surrounding serving baseline; it is not evidence
that a quality-changing window beats those engines at equivalent work.
M13 therefore ends as an explicit educational option, not a new default.

## Source map

| Source | What it establishes |
|---|---|
| [config.py](https://github.com/kaix-nv/tinyserve/blob/9de1765eb5ecc167dd5b46effba7792c5ada1a58/tinyserve/config.py) | `is_sparse_layer()` chooses MoE feed-forward layers. |
| [models/qwen3_moe.py](https://github.com/kaix-nv/tinyserve/blob/9de1765eb5ecc167dd5b46effba7792c5ada1a58/tinyserve/models/qwen3_moe.py) | Expert sparsity retains the ordinary attention implementation. |
| [attention.py](https://github.com/kaix-nv/tinyserve/blob/9de1765eb5ecc167dd5b46effba7792c5ada1a58/tinyserve/attention.py) | Window bounds, prepared token addresses, masks, and the FlashInfer adapter. |
| [paged.py](https://github.com/kaix-nv/tinyserve/blob/9de1765eb5ecc167dd5b46effba7792c5ada1a58/tinyserve/paged.py) | Exact-slot gathers; page ownership remains independent of selection. |
| [engine.py](https://github.com/kaix-nv/tinyserve/blob/9de1765eb5ecc167dd5b46effba7792c5ada1a58/tinyserve/engine.py) | Absolute chunk positions and configured backend propagation. |
| [M13 validation contract](https://github.com/kaix-nv/tinyserve/blob/9de1765eb5ecc167dd5b46effba7792c5ada1a58/docs/design/m13-windowed-attention.md) | Supported scope and frozen numerical/quality/timing boundaries. |

These source links require repository access. M13's local window is distinct
from the hypothetical content-dependent selector discussed earlier. The
worked examples and figures stand alone; the receipts preserve measured
benefits, quality costs, failed attempts and the limits of each comparison.

{% include tinyserve-book-nav.html %}
