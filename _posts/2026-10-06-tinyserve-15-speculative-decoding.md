---
layout: post
math: true
title: 'Tinyserve, Chapter 15: Speculative decoding'
date: 2026-10-06 00:00:00 -0700
categories:
- tinyserve
- llm-serving
excerpt: When can verified draft tokens save target-model work?
book_chapter: 15
source_revision: 14b0c8dbb975350bbd730ac0c702eafce36eb95c
source_document: docs/book/15-speculative-decoding.md
code_revision: e20a34815498560f9226a9e057ade31b2b62743f
---

{% include tinyserve-article-style.html %}
{% include tinyserve-book-nav.html %}

*Chapter 15 · Execution Optimization*

An ordinary decoder runs the target model once for each new token. The
next input is unknown until that forward finishes. A smaller draft model
can guess several inputs in advance, letting the target evaluate their
continuation together. If enough guesses agree with the target, one target
forward supplies several useful output tokens.

The important word is **useful**. Verification also computes states for
guesses that may be rejected. Those states must never become committed
history. We will follow a three-token proposal through verification, the
first mismatch, and rollback of both logical cursors and physical pages.

Tinyserve implements a single-request greedy path for unquantized dense
Qwen models. FP32 supplies its numerical reference. BF16 remains
experimental: the retained model-level logit gate fails, and the measured
speculative path does not beat optimized ordinary serving. Understanding
the mechanism includes understanding those boundaries.

## Known candidates let the target process several positions

The target cannot know its own future outputs without computing them.
It can, however, score a **supplied** continuation in parallel, just as it
processes a known prompt. Causal attention ensures that each position sees
only the prefix and earlier supplied candidates.

At a round boundary, let C tokens be cached and let `pending` be the last
token already emitted to the output history. It has not yet been processed
by either model. This is the same one-token gap that ordinary decode keeps
between sampling and cache insertion.

For three draft proposals, the target forward is

```text
input tokens:  [pending, d1,  d2,  d3]
positions:    [C,       C+1, C+2, C+3]
predictions:  [t1,      t2,  t3,  bonus]
```

The logits after an input predict the **following** token. Row zero checks
d1, not `pending`. Row one checks d2 using the history through d1. The
final row supplies a bonus only if all proposals survive. Verification
needs logits at every row, so it cannot use ordinary decode's final-row-only
vocabulary projection.

For target input shape `[1,k+1]`, logits have shape `[1,k+1,V]`, where V
is vocabulary width. Processing these rows together does not remove their
transformer computation; it can amortize weight loads and launch overhead.

## One mismatch determines the entire committed prefix

Suppose C=5 and token `4` is pending. These are illustrative integer IDs,
not claims about particular tokenizer words. The draft proposes `[7,9,2]`.
The target consumes `[4,7,9,2]` and predicts `[7,8,13,11]`.

| Target input | Position | Target predicts | Action |
|---|---:|---:|---|
| Pending 4 | 5 | 7 | Accept draft 7 |
| Draft 7 | 6 | 8 | Reject draft 9; emit correction 8 |
| Draft 9 | 7 | 13 | Discard prediction conditioned on rejected history |
| Draft 2 | 8 | 11 | Discard prediction conditioned on rejected history |

The round emits `[7,8]`. Even if a later proposal happened to equal its
target prediction, that later row would have used a history containing
rejected 9. Acceptance must be a prefix, stopping at the first mismatch.

[![A target verification call checks three proposals with four logit rows. Only draft token seven is accepted; target correction eight becomes the new pending token.](/assets/tinyserve/book15-verification.svg)](/assets/tinyserve/book15-verification.svg)

The essential acceptance logic in `_generate_tokens` is

```python
accepted = 0
for proposed, predicted in zip(proposals, predictions):
    if proposed != predicted:
        break
    accepted += 1
emitted = proposals[:accepted] + [predictions[accepted]]
```

If all three proposals match, `accepted=3` and `predictions[3]` supplies
the bonus. If the first proposal fails, `accepted=0` and only the target
correction is emitted. The draft never overrides the target's decision.

EOS and output limits refine this commit. An accepted EOS ends generation
without adding a bonus; a draft EOS can be rejected; a correction can itself
be EOS. The implementation reserves one output slot for a correction or
bonus when choosing the proposal count. With one slot remaining, it uses
zero proposals and performs ordinary target decode.

## Two caches have different temporary lengths

Both models begin the example with five cached positions. The target
processes all four supplied inputs and temporarily reaches length nine.
The draft processes `[4,7,9]` serially to **predict** `[7,9,2]`. It has not
processed its final proposal 2, so it reaches length eight.

After accepting 7 and emitting correction 8, the committed cache contains
the five old tokens, old pending 4, and accepted 7: seven positions.
Correction 8 becomes the new pending token. Thus both caches truncate to
seven, and the next forward writes 8 at position seven.

For a continuing request, the invariant is

$$
\texttt{committed}=|\texttt{prompt}|+|\texttt{outputs}|-1.
$$

The final output token is deliberately absent from KV. Confusing emitted
length with cached length would either cache an unprocessed correction or
skip processing it on the next round.

In an all-accepted three-proposal round, the target reaches C+4, while the
draft reaches C+3. The new bonus is pending; the draft is missing the KV
for its last accepted proposal. If generation continues, one extra draft
forward catches it up before the next round. That work belongs in draft
cost. A terminal request can release its caches without catching up state
that will never be read again.

Target prompt prefill produces the first output before draft prefill.
If that output is EOS or completes the requested budget, no draft cache is
needed. Otherwise the draft processes the same prompt and joins the shared
round-boundary invariant.

## Rollback changes visibility and page ownership

A contiguous `KVCache` has preallocated K/V buffers and a valid-prefix
cursor. `truncate(7)` shortens that cursor. It does not copy the retained
prefix or zero the suffix. Later attention sees only valid positions, and
the next append overwrites the rejected position before exposing it.

Paged storage adds physical reclamation. Use four-token pages and target
block table `[5,2]`. Logical positions zero through three live on physical
page 5; positions four onward begin on page 2. Verifying through position
eight allocates page 4, making the table `[5,2,4]`.

[![The target cache grows to nine positions over pages five, two, and four. Rollback retains seven positions, frees page four, masks page two's rejected tail, and overwrites that tail with correction eight on the next append.](/assets/tinyserve/book15-page-rollback.svg)](/assets/tinyserve/book15-page-rollback.svg)

The committed length seven needs $\lceil7/4\rceil=2$ pages. Page 4 returns
to the free list. Page 2 stays allocated, including a stale last slot
containing rejected 9's KV. That slot is outside the valid prefix. The next
pending token 8 maps to

$$
\begin{aligned}
\text{slot}(7)&=4\,\text{table}[\lfloor7/4\rfloor]\\
&\quad +(7\bmod4)\\
&=4\times2+3\\
&=11.
\end{aligned}
$$

Writing slot 11 replaces the stale KV. Freed page 4 may later be allocated
again, but new valid contents must be written before use. Clearing all
rejected memory is unnecessary when visibility, addresses, and ownership
are correct.

`PagedSpeculativeState` adapts append, truncate, and cleanup while reusing
the same acceptance loop. Target and draft have separate pools: matching
token IDs do not make their learned KV values or model dimensions equal.
The wrapper releases request pages in a `finally` block, including pages
allocated before an exception. Tests poison discarded tails and exercise
reuse, allocation failure, EOS, and repeated calls.

The rollback API rejects shared or published pages. This path has private
history, no speculative prefix publication, and no copy-on-write protocol.
GDN/KDA recurrent state is also outside scope: appending a token mutates a
summary state, so shortening a KV cursor cannot undo that recurrence.

## Verification attention begins after the cached prefix

For C cached tokens and T new queries, query row i occupies position C+i.
Its permitted keys satisfy

$$
j\le C+i.
$$

For C=5 and T=4, row zero can see keys zero through five, including its
own pending token but none of the proposals. A fresh-prompt triangle
starting at key zero would hide most of the old prefix. A noncausal mask
would expose future guesses. Both errors occur before acceptance compares
any IDs.

The Torch reference gathers paged history into contiguous tensors. The
optional FlashInfer adapter instead passes queries, physical pages, and
page-table metadata directly to attention. One-query draft/decode calls
use the paged-decode wrapper; multiquery verification uses the causal
paged-append wrapper. Fresh prefill retains the ordinary in-flight path.

For the example before rollback, append planning uses query boundaries
`[0,4]`, page boundaries `[0,3]`, page IDs `[5,2,4]`, and last-page valid
length 1. Query boundaries count tokens; page boundaries count page IDs.
After rollback, that plan is stale. The next forward must allocate, rebuild
the plan from its current table and length, and then run attention.

Direct reads remove an intermediate full-history gather, not attention's
own KV traffic. Tinyserve owns the metadata and lifecycle integration;
FlashInfer owns the selected GPU attention implementation. Neither the
reader change nor its unit tests resolve the full-model BF16 limitation.

## Greedy acceptance and stochastic correction are different algorithms

With stable target argmax decisions, greedy prefix acceptance gives the
same committed token history as ordinary target decoding. Each accepted
proposal equals the target choice for its preceding accepted history, and
the first correction comes directly from that target. Draft quality affects
how much work is accepted.

Sampling requires a different rule. If draft distribution q proposes token
x, an exact speculative-sampling construction accepts it with probability
$\min(1,p(x)/q(x))$, where p is the target distribution at the same history.
On rejection, it samples a correction from the normalized positive part
of $p-q$. If every proposal survives, it samples the bonus from the next
target distribution. The distributions include the chosen sampling
transformations. This preserves a distribution, not an identical random
sequence for an arbitrary shared seed. The construction is developed in
[Leviathan et al.](https://proceedings.mlr.press/v202/leviathan23a.html).

Tinyserve implements the greedy comparisons above. Independently sampling
from draft and target and accepting only matching IDs is not a substitute
for distribution correction, and request-local sampling from Chapter 11
does not automatically extend into this speculative path.

Compatibility also requires more than equal vocabulary sizes. The target
templates and tokenizes the prompt once. The facade compares complete fast
tokenizer definitions and special-token mappings before sharing those IDs
with the draft. Both models must fit the requested context, agree on EOS,
and satisfy the same-device, same-dtype, dense, unquantized restrictions.
Paged operation additionally needs idle private pools and disabled graphs.

## Floating point can change the target decision

The greedy argument assumes that one-token target decode and multi-token
verification compute the same argmax for the same history. Finite-precision
matrix operations can violate that assumption. Changing query count can
change reduction order or algorithm selection even before attention.

After two free-running paths choose different tokens, later differences
also include different text. The retained numerical closeout therefore
teacher-forces identical reference histories through query groups of one,
three, and five, and through contiguous, gather, and direct readers.
The first aligned output is the prompt's final prediction. Producing
32 output predictions needs only 31 continuation inputs.

The original gate requires finite logits, normalized RMS error at most
0.01, and mean KL at most 0.01. On Qwen3-4B/A6000, the
[matched-history record](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m9d-bf16-closeout-a6000-2026-09-20.json)
reports 42/42 passing FP32 comparisons and 0/90 passing BF16 comparisons.
BF16 failed the RMS condition, with maximum aggregate NRMSE 3.123%;
maximum mean KL was 0.001847. These overlapping comparisons cover six
prompts, not ninety independent prompts.

The first-layer normalized inputs were identical across query counts, but
Q/K/V projections already differed before the cached attention read.
Changing only the reader at fixed query count introduced additional
differences inside attention. The trace identifies both sources; it does
not identify a specific GEMM algorithm without profiling it.

A code-writing position makes the practical effect visible. Ordinary BF16
tied `def` and `Wait` at 22.375; argmax selected the lower token ID, `def`.
Five-query direct execution gave `def` 22.250 and `Wait` 22.375. At these
magnitudes one BF16 step is 0.125. A one-step change breaks the tie and
branches the subsequent text.

Low KL, matching early tokens, and small average errors are different
contracts from exact greedy agreement. Mean-centering logits is useful
diagnostically, but cannot replace the original raw-logit gate after it
fails. The closeout did not establish a cache-allocation or rollback bug
to repair; it retained the failed gate and experimental BF16 status.

## Accepted work must repay draft and verification cost

If a round accepts a proposals, it normally emits a+1 tokens. Its cost is

$$
\begin{aligned}
T_{round}&=T_{draft}+T_{verify}\\
&\quad +T_{bookkeeping}.
\end{aligned}
$$

A rough break-even condition compares that cost with a+1 ordinary steps:

$$
T_{round}<(a+1)T_{ordinary\ decode}.
$$

The draft term includes serial proposals and catch-up. Verification processes
k+1 inputs even if the first proposal fails, including all vocabulary rows.
Both models' weights and caches occupy memory. A longer proposal window
can increase wasted work faster than it increases the accepted prefix.

The [direct-reader diagnostic](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m9c-direct-paged-attention-a6000-2026-09-20.json)
kept Qwen3-4B target and Qwen3-0.6B draft resident on an A6000. It used
two warmups and seven rotating-order repetitions, one request, and four
proposals. Whole-request output rates include prompt prefill:

| Path | 128 input / 128 output tokens per request | 2,048 input / 32 output |
|---|---:|---:|
| Direct paged ordinary | 31.42 tok/s | 26.47 tok/s |
| Direct paged speculative | 33.08 tok/s | 26.18 tok/s |
| Ordinary paged with graphs | 62.45 tok/s | 44.02 tok/s |

All timed proposals were accepted on the repetitive timing fixture.
Despite that favorable case, speculation only modestly helped the short
workload against the matched direct reader and slightly lost on the long
prompt. It remained well below optimized ordinary serving. These timings
are diagnostic because numerical qualification failed; they do not justify
promotion. Timing-fixture token agreement does not erase the separate
real-prompt mismatches.

TTFT here is internal first-token availability, not streamed network
delivery. Target prefill supplies it before draft prefill, so draft setup
appears in the next-token gap and total request time. Pool construction
and compilation are outside warmed timing; per-forward plans, rollback,
draft work, and cleanup are inside. These are single-request results,
without speculative scheduling or concurrency qualification.

## Follow the acceptance and state boundaries

| File | Responsibility |
|---|---|
| [speculative.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/speculative.py) | Shared greedy draft, verify, acceptance, commit, and catch-up loop. |
| [kv_cache.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/kv_cache.py) | Contiguous valid-prefix cursor and truncation. |
| [speculative_paged.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/speculative_paged.py) | Request-owned paged state, append planning, and cleanup. |
| [paged.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/paged.py), [attention.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/attention.py) | Private-page rollback and gather/direct attention interfaces. |
| [diagnose_speculative.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/examples/diagnose_speculative.py) | Matched-history numerical comparisons and layer traces. |

This implementation is pinned to runtime snapshot `e20a348`. The worked
round exposes the governing invariant: only a verified prefix enters
committed history, and the last emitted token remains pending. Making that
invariant compatible with concurrent scheduling, quantized state, or
distributed models would require additional ownership and numerical
contracts beyond this single-request path.

{% include tinyserve-book-nav.html %}
