---
layout: post
math: true
title: "Building tinyserve M5b: Prefix caching"
date: 2026-08-23 18:14:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Identify complete token histories with chained hashes, share immutable KV blocks, and retain released blocks in a reclaimable warm LRU tier."
---

*Milestone 5, part 3, of [building an LLM inference engine from scratch](/series/tinyserve/).
Previous: [M5c — CUDA graph replay]({% post_url 2026-08-23-building-tinyserve-m5c %}).*

Code: [`tinyserve` @ `1967b96`](https://github.com/kaix-nv/tinyserve/tree/1967b9604bb9df53f0e03e35231f168f0cbf91d8).

## The problem: repeated prompts produce repeated KV

Many serving workloads repeat context. An assistant receives the same system
prompt on every request. A later chat turn sends the conversation history
again. A retry may differ only near the end. Without prefix caching,
tinyserve recomputes K/V for every repeated prompt token.

M3 already gave each sequence a block table, and M4 already let requests
arrive and leave independently. Those mechanisms solve *where* KV lives and
*who* may run, but neither recognizes that two requests can need the same KV
contents.

Within one fixed model and cache configuration, a token's K/V is determined
by its token history and position. If two requests have the same prefix, the
KV produced for that prefix is reusable. M5b turns that observation into a
cache with one strict invariant:

> Prefix caching may change how many prompt tokens are computed, but it must
> never change the generated result.

## Why hashing one block is not enough

For readability, imagine a block size of four tokens even though tinyserve
uses 16:

```text
request A: [a b c d] [e f g h] [i j]
request B: [a b c d] [e f g h] [x y]
```

The first two full blocks are reusable. The partial third blocks are not.

It is tempting to identify the second block by hashing only `[e f g h]`.
That is wrong. At deeper transformer layers, the hidden states—and therefore
K/V—for those tokens depend on everything they could attend to before them.
The same four token IDs after a different first block need different KV.

Tinyserve therefore builds a hash chain:

```text
h0 = H(root, [a b c d])
h1 = H(h0,   [e f g h])
```

The parent hash makes `h1` an identity for the entire eight-token prefix, not
just its last four tokens. The global map stores `h1 → physical block`.

[`hash_block`](https://github.com/kaix-nv/tinyserve/blob/1967b9604bb9df53f0e03e35231f168f0cbf91d8/tinyserve/paged.py)
uses a 128-bit BLAKE2b digest rather than Python's process-local `hash()`.
A collision would associate a request with unrelated KV, so this identity
needs a stable, collision-resistant digest. The map belongs to one `LLM`
instance, which implicitly fixes model weights, RoPE configuration, adapters,
dtype, and KV layout. A cache shared across models or persisted across runs
would need those values in an explicit namespace; tinyserve does not
implement that broader system.

## The three rules

### 1. Share only full blocks

A partial block can still receive more tokens. Sharing it would require
copy-on-write: when either sequence grows, it must first make a private copy.
Tinyserve avoids that machinery by sharing only full blocks.

Full blocks are immutable from the sequence's perspective. New K/V always
lands after `num_cached`, so M3's block refcounts are sufficient: adoption
increments a refcount, release decrements it, and no owner writes into a
shared block again.

### 2. Keep released blocks warm

When the last live reference to a registered block disappears, immediately
overwriting it would waste a potentially useful cache entry. Tinyserve moves
the zero-reference block into a warm LRU tier:

```text
active block, refcount 2
        │ one owner finishes
        ▼
active block, refcount 1
        │ last owner finishes
        ▼
warm block, refcount 0 ──match──▶ active again
        │ allocator needs space
        ▼
evicted: hash removed, storage returned to free list
```

A warm block is matchable and resurrectable, but it is also reclaimable.
That changes capacity accounting: admission and decode must reason about
`free blocks + warm blocks`, not only the free list. Warm storage is useful
state, not guaranteed residency.

### 3. Leave something to compute

The prefix cache stores K/V, not logits. If every prompt token were adopted,
there would be no forward pass from which to obtain the last prompt
position's logits and sample the first generated token.

Matching is therefore capped at `len(prompt) - 1` tokens. Because adoption
is block-granular, the exact remainder depends on alignment:

- a 10-token prompt with block size 4 may adopt 8 tokens and compute 2;
- an 8-token prompt may adopt only the first 4 and recompute the final
  4-token block.

Storing final-position logits alongside KV could remove this recomputation,
but that is a different cache contract. M5b keeps the minimal KV-only design.

![Hash-chained blocks move between shared ownership and a warm LRU tier](/assets/tinyserve/m5b-prefix-cache-lifecycle.svg)

*Figure 1. The chained hash identifies a complete token history. Request B
adopts A's two immutable full blocks, while each request keeps its partial
tail private. After the last owner releases a registered block, it remains
warm until a future match resurrects it or allocation pressure evicts it.*

## Admission: match, account, then adopt

When the FCFS scheduler considers a waiting request, it walks full prompt
blocks from left to right and stops at the first missing hash. It then
distinguishes two kinds of match:

- **Live match:** another request already pins the physical block. The new
  owner increments its refcount, but the pool consumes no additional block.
- **Warm match:** the block has refcount zero. Adoption resurrects and pins
  it, so the scheduler must count it as capacity that is no longer
  reclaimable.

Only after this accounting passes does the sequence adopt the matched blocks,
set `num_cached` to the matched token count, and reserve blocks for the
remaining prompt. M5a's chunked-prefill path already knows how to start from
a nonzero `num_cached`, so prefix hits need no separate model forward.

Blocks are registered as prompt or generated tokens fill them. Generated
tokens must therefore remain in `Sequence.tokens` until their K/V is written;
that token ledger is the source used to construct the next chained hash. A
later request can reuse generated text only when its serialized token prefix
is exactly the same—for example, a compatible multi-turn chat template.

## A deliberate limitation: same-step misses

Two identical requests admitted in the same scheduler call do not share with
each other. Their blocks do not contain computed KV yet, so neither request
can adopt the other's result. Both compute private copies; when registration
finds the same hash twice, the first physical block remains canonical and the
duplicate stays private.

A larger engine could register a block identity at allocation time and track
pending computation so later requests wait on the first producer. Tinyserve
does not add that state machine. The important correctness behavior is that a
miss causes recomputation, never reuse of unfinished KV.

## Interaction with CUDA graphs

Prefix sharing reduces physical KV usage but may increase the number of page
references: ten sequences can all point at the same twenty physical blocks,
creating two hundred entries in FlashInfer's flattened block-table metadata.

M5c captured fixed-size metadata buffers. M5b therefore checks both live
batch size and flattened page-reference count before replay. If the shared
table does not fit, decode uses eager FlashInfer for that step. This is a
useful systems lesson: composing two individually correct optimizations can
create a new boundary that neither feature had alone.

## The proof—and its boundary

[`tests/test_prefix.py`](https://github.com/kaix-nv/tinyserve/blob/1967b9604bb9df53f0e03e35231f168f0cbf91d8/tests/test_prefix.py)
and
[`tests/test_graphs.py`](https://github.com/kaix-nv/tinyserve/blob/1967b9604bb9df53f0e03e35231f168f0cbf91d8/tests/test_graphs.py)
check the following:

1. Hashes depend on their parent, and matching respects the
   `len(prompt)-1` cap.
2. A four-block CPU cache covers register → share → release → warm →
   resurrect → evict, with refcounts and hash entries checked at each state.
3. Clearing warm state for an independent benchmark condition returns every
   zero-reference block to the allocator.
4. `generate_paged()` keeps sampled tokens in the sequence ledger before a
   generated token fills and registers a block.
5. A second run reuses all but at most one block of each prompt and produces
   the same greedy fp32 gather text as the cold run.
6. Staggered requests adopt a previously computed shared system prefix and
   match a caching-disabled reference.
7. A warmed shared prefix remains correct while a small pool forces actual
   recomputation preemption.
8. Shared-prefix requests run through CUDA-graph replay and remain
   text-identical to a caching-disabled eager reference.

These are scoped claims. Greedy text parity does not establish equivalence
for every stochastic sampling regrouping, model, backend version, or hash
collision. The tests establish the invariants needed by this teaching
engine, not a production cache validation matrix.

## The payoff: retained milestone measurements

The original M5b experiment used 48 requests with an approximately
1000-token shared policy prompt, short per-request questions, Poisson
arrivals, Qwen3-0.6B bf16, FlashInfer, CUDA graphs, and a 512-token prefill
budget.

The raw logs, exact policy text, package versions, and repeated trials were
not retained. Treat the following numbers as a historical single-run
snapshot that motivates the mechanism, not as a result that can be replayed
bit for bit or generalized to other hardware.

At λ=8 requests/s, where the GPU had slack:

| configuration | goodput | TTFT p50 / p99 | prompt tokens cached |
|---|---:|---:|---:|
| prefix off | 112 tok/s | 99 / 329 ms | 0% |
| prefix on | 111 tok/s | **48 / 108 ms** | **98%** |

At λ=32 requests/s, where repeated prefill work saturated the snapshot:

| configuration | goodput | TTFT p50 / p99 | ITL p99 |
|---|---:|---:|---:|
| prefix off | 155 tok/s | 1622 / 3069 ms | 67 ms |
| prefix on | **372 tok/s** | **90 / 166 ms** | 152 ms |

The cached run computed far fewer prompt tokens and, in this snapshot,
delivered 2.4× goodput and an 18× smaller median TTFT at λ=32. ITL p99 became
worse. One plausible explanation is that faster admissions exposed decode to
more concurrent work and prefill interference, but the old run did not
retain enough traces to isolate that cause. The honest conclusion is simply
that removing one bottleneck changed the next visible tail.

A separate warm-repeat experiment retained this pair:

| run | TTFT | cached prompt tokens |
|---|---:|---:|
| cold | 170 ms | 0 |
| warm | **35 ms** | 2656 / 2657 |

The warm request computed one remaining prompt token and sampled its first
output from that forward. It did not run a 16-token chunk or a decode step
before TTFT. The result illustrates why KV-only prefix caching cannot quite
skip the entire prompt.

## Reproducing the current protocol

The checked-in serving benchmark now has an explicit shared-prefix workload,
feature switches, environment reporting, and a cold prefix-cache reset before
every reported condition. It prints the actual tokenized prefix and mean
prompt lengths rather than assuming the historical prompt length.

```bash
.venv/bin/pytest tests/test_prefix.py tests/test_graphs.py -q
.venv/bin/pytest tests/ -q

for prefix in off on; do
  git rev-parse HEAD | tee "m5b-prefix-${prefix}.txt"
  .venv/bin/python examples/bench_serve.py \
    --shared-prefix --rates 8 32 --n 48 --max-new 96 --long-frac 0 \
    --chunk-size 512 --seeds 0 1 2 \
    --cuda-graphs on --prefix-caching "${prefix}" \
    | tee -a "m5b-prefix-${prefix}.txt"
done
```

Run the `off` and `on` cases as separate processes, preserve both logs, and
compare the same seeds. These commands define a reproducible *current*
protocol; they do not recreate the historical table because its exact policy
text and environment were not retained.

For the cold/warm mechanism alone:

```bash
.venv/bin/python - <<'PY'
from tinyserve.engine import LLM
from tinyserve.scheduler import SamplingParams

llm = LLM("Qwen/Qwen3-0.6B")
prompt = "Summarize: " + "LLM serving is memory management. " * 300
for tag in ("cold", "warm"):
    out = llm.serve([prompt], params=SamplingParams(max_new_tokens=32))[0]
    print(
        tag,
        f"TTFT={out.ttft * 1e3:.0f} ms",
        f"cached={out.prompt_cached_tokens} tokens",
    )
PY
```

## What M5b teaches

Prefix caching is not merely a dictionary around prefill. It is a contract
between token identity, immutable paged storage, refcount ownership, warm
eviction, scheduler capacity, and the attention backend's metadata limits:

- hash the whole history, not an isolated block;
- share only immutable units unless copy-on-write is implemented;
- distinguish live sharing from warm pinning during admission;
- keep the token ledger aligned with KV registration;
- leave a prompt position to compute when the cache stores no logits;
- treat misses and eager fallback as correct behavior; and
- compare cache policies from the same cold starting state.

That is the minimal complete lesson. Cross-process cache coordination,
tenant isolation, persistent namespaces, pending-block deduplication, and
distributed invalidation belong to a different, production-oriented system.
