---
layout: post
math: true
title: 'Tinyserve, Chapter 8: KV caching and batched generation'
date: 2026-10-06 00:00:00 -0700
categories:
- tinyserve
- llm-serving
excerpt: How can several requests reuse their own histories in one batched loop?
book_chapter: 8
source_revision: 14b0c8dbb975350bbd730ac0c702eafce36eb95c
source_document: docs/book/08-kv-cache-and-batching.md
code_revision: e20a34815498560f9226a9e057ade31b2b62743f
---

{% include tinyserve-article-style.html %}
{% include tinyserve-book-nav.html %}

*Chapter 8 · The Serving Runtime*

A model has just predicted the next token. To predict another, it must
process that token with the preceding history. Chapter 1 did this by running
the entire growing sequence again. A KV cache lets the model keep the part
of that history that future attention needs. Batching then lets several
independent requests share a forward through the same weights.

These changes introduce a bookkeeping problem as important as the matrix
multiplications: the token just returned to the caller is usually not yet
in the cache. In a padded batch, its next physical storage location also
differs from its logical position in the request. We will follow two unequal
prompts through prefill and decode, keeping those facts visible throughout.

## What survives a forward

At each dense attention layer, the model projects hidden vectors into
queries, keys, and values. The query for a position is used to construct
that position's output. Its key and value can be used by every later
position, so those are the tensors worth retaining. Each layer needs its
own cache: a key from layer 3 cannot substitute for a key from layer 4.

For a causal model with fixed weights, positions, and evaluation behavior,
appending a future token does not change the allowed history of an earlier
token. Earlier hidden vectors, keys, and values therefore represent the
same mathematical quantities. Tinyserve caches keys after per-head
normalization and RoPE, and values after projection. Values do not receive
RoPE. The rotated key remains associated with its original logical position.

This equivalence is mathematical, not a promise that every execution shape
produces bit-identical floating-point results. A full prompt and a one-token
decode can select different kernels and reduction orders. Correctness
comparisons need fixed histories and a numerical contract, especially when
nearly tied logits can change a greedy choice.

The ordinary contiguous cache stores two tensors:

```text
K: [layers, batch, capacity, KV heads, head width]
V: [layers, batch, capacity, KV heads, head width]
```

`capacity` is reserved storage. `seq_len` is the physical prefix already
written. Allocating capacity does not make its unwritten contents valid.
Each layer writes at the same current offset; only after all layers finish
does the model advance the cursor. Advancing inside the layer loop would
make later layers write different token positions during the same forward.

The cache uses assignment into preallocated storage, rather than growing
K and V with a concatenation every step. Concatenation would repeatedly
allocate and copy the old history. A cache that removes repeated model work
should not replace it with repeated full-history copies.

## Two prompts in one rectangle

Let request A contain three tokens and request B contain five:

```text
A: [a, b, c]
B: [u, v, w, x, y]
```

Static batching uses one rectangular input of shape `[2, 5]`. Tinyserve
left-pads A, using the model's EOS token ID as the placeholder. The ID
alone does not distinguish padding; a separate validity tensor does.

| Row | Physical input columns 0 through 4 | Real-key flags | Logical positions |
|---|---|---|---|
| A | pad, pad, a, b, c | F, F, T, T, T | 0, 0, 0, 1, 2 |
| B | u, v, w, x, y | T, T, T, T, T | 0, 1, 2, 3, 4 |

Token a occupies physical column 2 but logical position 0. Token c occupies
physical column 4 but logical position 2. The pad positions receive zero
as an arbitrary position value; their keys are hidden from real queries.
Logical positions govern RoPE. Physical columns govern writes into this
particular cache layout.

[![Unequal prompts are left padded into five columns. Real tokens preserve their own logical positions, and decode appends both rows at physical column five with logical positions three and five.](/assets/tinyserve/book08-padded-cache.svg)](/assets/tinyserve/book08-padded-cache.svg)

Left padding places the final real prompt token in the same column in every
row. Consequently, selecting the last hidden column selects c for A and y
for B. Right padding could also be made correct, but it would require
selecting a different final column for each row.

Before attention, the hidden input has shape `[2, 5, H]`. Queries become
`[2, 5, Hq, D]`, while new keys and values become `[2, 5, Hkv, D]`.
Here H is residual width, Hq and Hkv are head counts, and D is head width.
The batch axis keeps the requests independent. No attention operation is
allowed to retrieve B's history for A.

## A mask answers which keys are visible

The prefill mask combines causal order and key validity. For a physical
query column i and key column j, Tinyserve permits attention when

$$
M_{b,i,j} = (j \leq i \;\land\; \mathrm{real}_{b,j})\;\lor\;(i=j).
$$

This is a Boolean mask: `True` means allowed, `False` means excluded.
It is passed with shape `[B, 1, P, P]`, broadcasting over query heads.
For A's real queries, a sees a, b sees a and b, and c sees a, b, and c.
Neither pad key is visible to a real query.

The diagonal term gives each pad query one permitted key: itself. Its
result is discarded, but it remains a well-defined row rather than relying
on a backend's handling of an entirely masked softmax. This does not expose
pad keys to real queries. Avoiding a NaN in a discarded intermediate can
still matter, because later arithmetic does not reliably neutralize NaN
merely by multiplying it by zero.

All ten physical input positions pass through the transformer. Padding is
hidden from useful attention, not removed from computation. The cache also
receives K and V for the two pad positions. Those physical entries remain
invalid for useful attention for the lifetime of the static batch.

At the end of prefill, the common physical cursor is five. The model uses
`last_token_only=True` to select the last hidden column before the vocabulary
projection. Instead of constructing `[2, 5, vocabulary]` logits, it produces
`[2, 1, vocabulary]`; the engine selects the singleton token axis and samples
one token per request. The optimization saves output-projection work, while
the preceding transformer still processes all padded prompt positions.

## Sampling is one step ahead of cached history

Suppose prefill samples A1 and B1. These tokens enter the requests' output
lists immediately. Their K and V do not exist yet: logits describe a choice
of next token, and choosing that token does not run it through the model.

To produce A2 and B2, the next forward consumes `[A1, B1]` as a `[2, 1]`
input. Both rows write physical cache column 5. Their logical positions
differ: A1 is position 3, while B1 is position 5. The engine passes
`positions = prompt_lengths + step`, with `step = 0` for this first decode.

The decode key mask is now

```text
A: [False, False, True, True, True, True]
B: [True,  True,  True, True, True, True]
```

Its shape is `[2, 1, 1, 6]`. There is only one query per row, and it may
attend to every real cached key, including its own new key. No triangle
starting at key zero is needed. An unqualified top-left causal mask for a
one-query, many-key attention call could incorrectly expose only the first
key; the explicit validity mask expresses the intended history directly.

After this forward, the physical cache cursor is six. Sampling A2 and B2
again leaves the newest output uncached. A third forward consumes those
tokens at physical column 6, using logical positions 4 and 6, and samples
A3 and B3.

| Model call | Input tokens | Cache after the call | Newly sampled tokens |
|---|---|---|---|
| Prefill | Padded prompts | Five physical columns | A1, B1 |
| Decode 1 | A1, B1 | Six physical columns | A2, B2 |
| Decode 2 | A2, B2 | Seven physical columns | A3, B3 |

If the output limit is three, generation ends here. A3 and B3 need never
enter the cache. Computing them merely to obtain unused next-token logits
would add a full forward without changing the response.

For one unpadded request with prompt length P and output count N, positive
N and no early completion, cached generation processes `P + N - 1` new
token positions. The uncached loop processes

$$
NP + \frac{N(N-1)}{2}
$$

positions. This is a count of transformer input positions, not a direct
latency formula. Cached attention still reads the growing history for each
new query; it has not become independent of context length.

## Finishing a row does not shrink this batch

Now suppose A2 is EOS while B requires another token. The engine records
A2, marks A done, and continues. A's row still participates in the next
forward, receiving whatever token the previous sampling call returned.
Its later logits and sampled tokens are ignored for A's returned output.
The final text omits the terminal EOS token.

This wasted row cannot modify B: their batch rows and histories are
separate. It does consume projection, attention, and cache work. The
`done` flag changes which outputs count, not the shape of the model call.
The loop stops when all rows are done or the common output limit is reached.

Request identity therefore must not be equated with a permanent tensor
row. In this static API, the initial list order happens to supply a stable
row-to-request mapping. Later serving paths remove completed requests and
rebuild a batch from the survivors. A row number then means “this request
in this forward,” while the request object owns history and completion.
That separation will make dynamic membership possible in Chapter 10.

This example uses a single-device dense model. At the pinned snapshot,
`generate_batch()` rejects distributed execution and hybrid models.
Recurrent-state ownership in a hybrid model needs more than
this rectangular ordinary KV cache; the scheduler-owned hybrid path is a
different implementation.

## Memory and useful throughput

If each cache element uses s bytes, a dense contiguous cache with L layers,
B rows, capacity C, Hkv KV heads, and head width D occupies

$$
\mathrm{bytes}_{KV}=2LBCH_{kv}Ds.
$$

The factor two counts keys and values. For the illustrative 28-layer,
8-KV-head, width-128 BF16 configuration, one position per request uses
114,688 bytes, or 112 KiB. Eight thousand valid positions use about
0.918 GB in decimal units. Reserved but unwritten positions consume the
same storage as valid positions.

Our two prompts reserve two rows of the longest prompt plus the output
allowance. That includes two pad slots and capacity for outputs that might
never be produced. Early EOS stops useful work but does not return a slice
of this tensor to another request. Paging, introduced next, changes that
allocation contract.

Batching can increase throughput because weight reads and operation launches
serve more token rows. It also increases per-request KV traffic. A historical
Qwen3-0.6B BF16 decode snapshot on one RTX A6000 illustrates both effects:

| Context length | Batch size | Step time | Raw decoded tokens per second |
|---:|---:|---:|---:|
| 512 | 1 | 22.1 ms | 45 |
| 512 | 32 | 24.1 ms | 1327 |
| 512 | 256 | 141.0 ms | 1815 |
| 2048 | 32 | 71.1 ms | 450 |

These are retained milestone measurements, not new measurements of the
current implementation. Raw throughput counts all B rows each step. A
separate ragged workload returned 406 useful tokens/s despite a roughly
1330-token/s decode microbenchmark: padded prefill, completed rows, and
other end-to-end work all contribute to that difference. The retained
record does not isolate their individual costs. The exact historical
environment and full workload were not retained, so these numbers explain
a scaling pattern rather than predict present performance.

Caching and batching solve complementary problems. Caching avoids repeated
old-token computation; batching gives each invocation more useful token
rows. Neither guarantees that cache reads, padding, or Python overhead
are cheap. A useful measurement counts returned tokens and states whether
it includes prefill, sampling, and early completion.

## Follow the implementation

| Source at runtime snapshot `e20a348` | Responsibility |
|---|---|
| [kv_cache.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/kv_cache.py) | Preallocation, layer writes, and one cursor advance per forward. |
| [engine.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/engine.py) | `generate()` and `generate_batch()`, padding, masks, sampling, and done rows. |
| [models/qwen3.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/models/qwen3.py) | Normalize and rotate keys, update caches, apply attention, and select final hidden rows. |
| [test_batching.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/test_batching.py) | Ragged batch versus individual generation and early completion checks. |

The tests compare fixed FP32 greedy outputs across cached, uncached, and
batched paths; they exercise the combined position and mask contract.
Those comparisons do not promise identical stochastic outcomes under
arbitrary regrouping or establish a GPU speedup. The concrete trace above
needs no repository access: after sampling, the newest token waits outside
KV; on its next forward, its logical position and physical destination
must both be correct.

{% include tinyserve-book-nav.html %}
