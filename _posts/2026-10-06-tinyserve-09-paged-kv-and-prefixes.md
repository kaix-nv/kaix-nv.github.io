---
layout: post
math: true
title: 'Tinyserve, Chapter 9: Paged KV memory and prefix reuse'
date: 2026-10-06 00:00:00 -0700
categories:
- tinyserve
- llm-serving
excerpt: How do logical token histories map to shared and private physical pages?
book_chapter: 9
source_revision: 14b0c8dbb975350bbd730ac0c702eafce36eb95c
source_document: docs/book/09-paged-kv-and-prefixes.md
code_revision: e20a34815498560f9226a9e057ade31b2b62743f
---

{% include tinyserve-article-style.html %}
{% include tinyserve-book-nav.html %}

*Chapter 9 · The Serving Runtime*

A contiguous cache gives each request a large rectangular reservation.
That is simple until prompts have different lengths, outputs stop early,
and new requests need the space held by old ones. Paged KV replaces that
reservation with a list of fixed-size storage blocks. Prefix reuse then
allows several requests to point to the same completed blocks.

Both mechanisms depend on a distinction between where a token belongs in
its history and where its keys and values happen to live. We will use
four-token pages, two requests with a common eight-token prefix, and a
small physical pool. Tinyserve ordinarily uses larger pages, but the
ownership rules are the same.

## A block table maps history to storage

Consider a ten-token request A:

```text
logical tokens: [a b c d] [e f g h] [i j]
logical page:       0         1       2
block table:       [7,        2,      9]
```

The physical IDs 7, 2, and 9 need not be consecutive or sorted. They say
where each logical page resides, not which token position it represents.
With page size S, token position p maps to

$$
\begin{aligned}
\mathrm{page}(p)&=T[\lfloor p/S\rfloor],\\
\mathrm{offset}(p)&=p\bmod S,
\end{aligned}
$$

and a convenient flattened slot identifier is

$$
\mathrm{slot}(p)=S\,T[\lfloor p/S\rfloor]+(p\bmod S).
$$

For A, logical position 5 is token f. Its logical page is 1, physical
page is 2, offset is 1, and slot is 9. Logical position 9 is token j,
stored in physical page 9 at offset 1, giving slot 37. The model still
uses logical positions 5 and 9 for RoPE; a physical slot ID never supplies
a token's position embedding.

[![Request A maps logical pages zero, one, and two to physical pages seven, two, and nine. Request B shares pages seven and two but has private page four.](/assets/tinyserve/book09-page-ownership.svg)](/assets/tinyserve/book09-page-ownership.svg)

The ordinary pool is laid out as

```text
[layers, physical pages, K-or-V, tokens per page, KV heads, head width]
```

Every layer stores its own K and V for the same page ownership. The slot
number identifies a page and offset within each layer; it does not flatten
the entire tensor. Because the K/V axis lies between page and offset,
blindly viewing storage as `[slots, 2, heads, width]` would rearrange the
meaning of its strides. Tinyserve decomposes slot IDs into page and offset
and indexes K and V explicitly.

That detail matters even if prefill text looks correct. A fresh prompt can
attend directly to the keys and values it just computed while writing them
into the pool. Only a later cache read exercises the physical write layout.
A meaningful cache check therefore writes a history and then reads it
through the block table during decode.

## Allocation and validity are separate

`ensure_blocks(seq, cache, upto)` extends a request's table until it covers
`ceil(upto / S)` tokens. It does not move earlier pages. A request growing
from four to five tokens acquires a second page; the first remains where
it was. Completing the request releases its references without copying
another request's data into the gap.

A's third page has capacity for four tokens but only two valid entries
after its ten-token prompt is computed. Its `num_cached` cursor is ten.
Attention must exclude the unused offsets even though the storage exists.
The block table says what storage the request owns; the cursor says which
part contains its computed history. A scheduler can reserve pages before
prefill reaches them, so allocation alone never establishes validity.

Padded batched prefill also needs a destination for unused input rows.
The ordinary paged pool includes one extra scratch page outside the
allocator. Padding writes can land there; useful attention never reads it.
Thus reported tensor allocation includes scratch storage even though the
request capacity advertised by the allocator does not.

Paging bounds tail waste to fewer than S token slots per independently
owned partial page. It removes the need to reserve every possible output
token at request creation, but it does not eliminate all waste or guarantee
admission. Active requests can still consume every available page, and
the scheduler needs a policy for the next decode boundary.

## Reuse requires the same preceding history

Now B arrives with

```text
A: [a b c d] [e f g h] [i j]
B: [a b c d] [e f g h] [x y]
```

A has already computed its prompt. The first eight tokens match, so B
can adopt physical pages 7 and 2, then allocate private page 4 for x and y.
Its table becomes `[7, 2, 4]`. A and B have separate histories and outputs,
but their first two pages contain the same mathematical keys and values.

Matching only the four tokens in the second page would be insufficient.
At deeper layers, their hidden states depend on the preceding page.
Tinyserve computes a chain of 128-bit BLAKE2b digests:

```text
h0 = hash(root, [a b c d])
h1 = hash(h0,   [e f g h])
```

`h1` identifies the whole eight-token prefix through its parent, not just
the most recent four IDs. Matching walks the chain from the beginning and
stops at the first miss. The cache belongs to one model instance and cache
configuration. Reusing entries across models, adapters, position policies,
or incompatible formats would require an explicit namespace beyond these
token hashes; this implementation does not provide that broader service.

The table stores only full, computed pages. Partial tails remain private.
If A and B shared a half-filled page and then wrote different continuations,
one could overwrite the other's history. Some designs solve that with
copy-on-write. Tinyserve's policy avoids the situation by sharing only
immutable full pages; it does not implement copy-on-write for these tails.

The token ledger is part of this contract. After sampling a generated token,
the engine appends its ID to `Sequence.tokens`. After a later forward
computes that token's KV, a newly completed page can be registered using
the corresponding token slice. Registering without the generated IDs would
hash the wrong history. The ordinary decode helper advances its cursor and
registers a just-filled page while preparing the forward, before launching
the actual write. The single owner does not admit another request during
that forward, so reuse becomes safe at the completed step boundary. On a
failed forward, the live session clears warm prefix entries after releasing
owners rather than trusting a potentially incomplete write. Registration
metadata alone is not evidence that device work has completed.

## A full KV hit still leaves prompt work

Suppose B's entire prompt is already represented in the cache. KV alone
cannot supply the logits at its final prompt position. Those logits require
the final hidden state and vocabulary projection, neither of which this
prefix cache stores.

The maximum adopted prefix therefore covers at most `len(prompt) - 1`
tokens, rounded down to full pages. With four-token pages:

| Prompt length | Maximum reused tokens | Tokens still computed |
|---:|---:|---:|
| 10 | 8 | 2 |
| 8 | 4 | 4 |
| 9 | 8 | 1 |

For B's ten-token example, adoption sets `num_cached = 8`. Prefill starts
at x, logical position 8, and finishes at y, position 9. x may attend to
the shared prefix and itself; y may also attend to x. The final logits
sample B1. As in Chapter 8, B1 is not cached until a subsequent decode
forward consumes it.

This is why a “100 percent prefix match” at the text level does not mean
zero model work. Caching final-position logits could change that contract,
but it would add another stored object and its own compatibility rules.
The minimal implementation keeps the KV-only boundary visible.

## Reference counts describe live ownership

After B adopts the prefix, pages 7 and 2 each have reference count two.
Private pages 9 and 4 each have count one. The reference count describes
live request owners, not how many token positions the page contains.

When A finishes, its references are released. Shared counts fall from
two to one, so B's prefix remains protected. A's unregistered private tail
returns to the free list. When B later finishes, counts for 7 and 2 reach
zero. Registered pages enter a warm least-recently-used tier instead of
becoming immediately available for arbitrary writes.

[![A registered page moves from two live owners to one, then to a warm zero-reference state. A match reactivates it, while allocation pressure removes its hash and reclaims its storage.](/assets/tinyserve/book09-page-lifetime.svg)](/assets/tinyserve/book09-page-lifetime.svg)

Warm means the data remains intact and matchable, but no live request pins
it. A later matching request can resurrect the page by removing it from
the warm tier and setting its live count to one. If allocation needs space,
the cache instead evicts the oldest warm entry, removes its hash mappings,
and returns its ID to the allocator. The next owner may overwrite it.

The free list and warm list thus mean different things. A free page has
no promised reusable identity. A warm page has an identity that the
allocator may revoke. The allocator uses free pages first and reclaims
warm ones when that list is empty. Free-list reuse is LIFO; warm eviction
is LRU. Neither ordering changes the logical-to-physical mapping rule.

Clearing the prefix cache for an independent benchmark condition is allowed
only when registered pages have no live owners. It drops warm identities
and returns their storage to the free list, while retaining the physical
pool allocation. This keeps tensor addresses stable for execution paths
that depend on them.

## Admission counts warm pages as capacity

The pool's immediately producible capacity is

$$
\mathrm{available}=\mathrm{free}+\mathrm{warm}.
$$

A live prefix match consumes no additional physical page: another request
already pins it. Adopting a warm match does consume available capacity,
because that page stops being reclaimable. A scheduler that subtracted
every hit equally would over-admit requests when hits came from warm state.

For example, let six pages be available and a ten-token prompt require
three pages. If two matched pages are live, only one new page must be
found. If both matches are warm, all three pages become newly unavailable
to other admissions: two through resurrection and one through allocation.
Tinyserve first looks up matches, counts live matches, checks capacity,
then commits adoption and reserves the rest. Its scheduler also retains
an admission watermark, discussed in Chapter 10.

Two identical prompts admitted together may still compute separate copies.
Neither has finished its KV when the other is considered. When both later
register the same hash, the first physical page becomes canonical and the
duplicate remains private. The implementation does not publish pending
computation or make one request wait on another producer. A miss causes
recomputation, preserving correctness while leaving a deduplication
opportunity unused.

## Capacity savings and execution costs

If N requests share F complete prefix pages and each needs one private tail,
their physical demand is F + N pages, compared with N(F + 1) without
sharing, assuming those are the only pages involved. Ten requests with
twenty shared pages and one tail each need thirty physical pages, while
their block tables still contain 210 references. Physical storage savings
do not shrink every metadata structure.

That distinction affects CUDA graph replay. A fixed metadata buffer must
fit the flattened table references, not merely the number of physical
pages in use. Tinyserve checks both batch size and reference capacity and
can use eager execution when the captured shape does not fit. A valid
fallback preserves the storage policy without making a replay promise.

Paging also changes attention reads. The reference backend gathers pages
into a convenient contiguous history; a paged backend can read through the
table directly. The latter can avoid copies, but page allocation by itself
does not supply such a kernel. Chapter 4 explains that execution boundary.

Historical measurements illustrate why the distinction matters. An older
RTX A6000 workload with 32 ragged prompts recorded 0.13 GB of peak paged KV
against 0.54 GB of contiguous reservation, yet useful throughput was
362 versus 406 tokens/s. Serialized prefill and host work accompanied the
memory change. The original full workload and traces were not retained,
so the difference cannot isolate the allocator's cost.

A separate prefix-reuse snapshot with 48 requests and a repeated roughly
1000-token policy prompt recorded 155 versus 372 useful tokens/s at
32 arrivals/s, and median TTFT of 1622 versus 90 ms. Its p99 inter-token
latency worsened from 67 to 152 ms. These Qwen3-0.6B BF16 FlashInfer and
CUDA-graph results are historical single-run observations, with incomplete
environment records. They demonstrate that saving prefill work can change
both capacity and interference; they do not establish an unconditional
latency improvement or a present-day speedup.

## Follow the implementation

| Source at runtime snapshot `e20a348` | Responsibility |
|---|---|
| [paged.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/paged.py) | Pool, slot mapping, refcounts, hash chains, warm eviction, and sequence release. |
| [scheduler.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/scheduler.py) | Match-aware admission and decode reservations. |
| [engine.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/engine.py) | Prefill/decode writes, cursor advancement, and full-page registration. |
| [test_prefix.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/test_prefix.py) | Sharing, warm resurrection, eviction, and cache-enabled output comparisons. |

The important lifetime is observable without a private code checkout:
computed full pages become shareable; live references pin them; the last
release makes registered pages warm; eviction removes their identity before
reuse. A page table lets those transitions occur without moving another
request's history. Scheduling decides when requests acquire and surrender
that ownership, which is the next chapter's subject.

{% include tinyserve-book-nav.html %}
