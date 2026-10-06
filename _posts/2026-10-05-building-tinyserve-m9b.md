---
layout: post
math: true
title: "Building tinyserve M9b: Speculative decoding over paged KV"
date: 2026-10-05 08:26:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Roll speculative KV back across physical page boundaries without exposing rejected tokens."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m9b-paged-speculation.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M9a — Speculative decoding: let the draft guess, let the target decide]({% include tinyserve-post-url.html slug="building-tinyserve-m9a" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m9-speculative-decoding.md" %}) · Next: [M9c — Verify draft tokens without gathering KV]({% include tinyserve-post-url.html slug="building-tinyserve-m9c" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m9c-direct-paged-attention.md" %})

Speculative decoding deliberately computes some KV that may never be used.
The draft proposes a continuation; the target verifies it; the first
disagreement invalidates the rest. With a contiguous cache, rollback only
shortens the visible prefix. With paged KV, we must also decide which physical
pages still belong to the request.

M9b connects the M9a acceptance loop to the page allocator from
[M3]({% include tinyserve-post-url.html slug="building-tinyserve-m3" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m3-paged-kv.md" %}). It changes storage, not who chooses the answer. Target
and draft have **separate private pools** because their layers, KV-head
counts, and KV values differ, even when their tokenizers agree.

This is a single-request, greedy reference path for unquantized dense Qwen
models. It uses PyTorch gather attention, with no scheduler, prefix sharing,
CUDA graphs, or speculative sampling. M9a remains the contiguous reference.
BF16 can still change near-tied greedy decisions; paging does not repair
that numerical limitation.

## The invariant: committed history ends before the pending token

At the beginning of a speculative round, both caches represent the same
committed token history. Its final emitted token, `pending`, has not yet
been processed. If `C` tokens are cached and the draft proposes `k` tokens,
the target verifies inputs `[pending, d1, ..., dk]` at positions
`C ... C+k`. The logits at each input predict the *following* token.

The target accepts only the matching prefix of proposals. It then emits
its correction, or a bonus token if every proposal matched. That final
emitted token becomes the next `pending` token. Therefore:

```python
committed = len(prompt_ids) + len(output_ids) - 1
```

The cache must stop at `committed`, not at the number of positions temporarily
written during verification. EOS and the output budget can stop the request
before another round; cleanup then releases all its pages.

## A concrete rejection crossing a page boundary

Use four-token pages to make the arithmetic visible. The implementation's
normal engine page size is 16; the rule is identical.

Start with `C=5` cached tokens and `pending=4`. The draft proposes `[7,9,2]`.
The target consumes `[4,7,9,2]` and predicts `[7,8,13,11]`.
It accepts `7`, rejects `9`, and emits correction `8`. The predictions
`13,11` used the rejected history and cannot be reused.

[![A target block table grows from two pages to three, rolls back to seven
cached positions, and overwrites the rejected suffix on the next append.](/assets/tinyserve/m9b-paged-rollback.svg)](/assets/tinyserve/m9b-paged-rollback.svg)

Suppose the target's physical block table starts as `[5,2]`. Logical page
zero lives in physical page 5; logical page one lives in physical page 2.
They need not be adjacent in memory. Verifying four inputs extends the
logical cache to nine positions and allocates physical page 4, giving
`[5,2,4]`.

The next committed length is seven: the five old tokens, `4`, and accepted
`7`. We keep `ceil(7/4)=2` pages, so the table becomes `[5,2]` and page 4
returns to the free list. Page 2 stays allocated even though its last slot
contains rejected token `9`'s KV. On the next append, pending correction `8`
at logical position 7 maps to:

```text
physical_slot = block_table[position // block_size] * block_size
                + position % block_size
              = block_table[7 // 4] * 4 + 7 % 4
              = 2 * 4 + 3 = 11
```

That write replaces rejected `9`'s KV. No prefix copy and no buffer clearing
are needed. At logical position 8, the allocator may hand out just-freed
page 4 again; the new value must be written before it is read.

The draft has its own table. It processed inputs `[4,7,9]` to *predict*
`[7,9,2]`, so its temporary length is eight, not nine. Both caches truncate
to seven in this example. In an all-accepted, nonterminal round, the draft
instead needs one catch-up forward on its last proposal before the next
round starts. These token-level rules are unchanged from M9a.

## Following the implementation

[`LLM.generate_speculative()`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/engine.py) selects `cache_mode`.
`"contiguous"` keeps M9a; `"paged"` calls
[`generate_paged_tokens()`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/speculative_paged.py). The paged API
requires idle local pools, no prefix caching, and separate target/draft
storage. The engine entry point also requires graphs disabled on both LLMs.

The wrapper gives the same
[`_generate_tokens()` acceptance loop](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/speculative.py) two
`PagedSpeculativeState` objects. Each state owns a `Sequence`: its token
history, valid cached length, and block table. The pool owns the physical
tensor and free list. That separation makes rollback small:

```python
keep = ceil(committed / block_size)
for block in seq.block_table[keep:]:
    pool.free_block(block)
del seq.block_table[keep:]
del seq.tokens[committed:]
seq.num_cached = committed
```

[`truncate_sequence()`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/paged.py) rejects shared or published
pages before mutating anything. M9b intentionally does not implement
copy-on-write or prefix-hash invalidation. After normal completion, EOS,
an exhausted pool, or a model exception, the wrapper's `finally` block calls
`free_sequence()` for each state it created. Partial allocation is covered:
every allocated page is immediately recorded in the request's table.

One append proceeds through these existing pieces:

| Step | Tensor or metadata | Responsibility |
| --- | --- | --- |
| Reserve | block table grows to `ceil((C+T)/B)` entries | `ensure_blocks()` |
| Map | `slots[T]` contains physical locations | `slot_ids()` |
| Forward | input IDs `[1,T]`, absolute positions `[T]` | existing dense model |
| Write | K and V each `[T,Hkv,D]` per layer | `PagedKVCache.write()` |
| Read | gathered valid prefix `[1,C+T,Hkv,D]` | Torch attention backend |
| Commit append | tokens extended, `num_cached = C+T` | state adapter, only after success |

Here `T` is the number of inputs in this forward, `B` is the page size,
`Hkv` is the model's KV-head count, and `D` is head width. For target
verification `T=k+1`; for a draft step `T=1`.

## Why a mask alone is not enough

The initial prefill attends directly to its in-flight K/V. Later appends
gather K/V through the block table and use the same offset causal mask as
chunked prefill: query `i` may read key `j` only when `j <= C+i`.
Verification candidates therefore cannot see future candidate tokens.

A gathered last page can also contain unwritten or rejected slots *past*
`C+T`. Those values are not legitimate context. The chunked Torch backend
now slices the gathered tensors to the actual append end **before** SDPA.
Masking unused values is insufficient if they contain NaNs: a zero attention
weight multiplied by NaN can still contaminate the reduction. The tests
poison unused and recycled slots explicitly, then compare replacement
appends with full recomputation.

This is a correctness backend, not a fused paged-attention kernel. It still
gathers referenced pages into temporary contiguous tensors, expands grouped
KV heads for the reference read, and incurs eager launch overhead.

## What paging saves—and what it does not

A request owns only the pages it currently needs. Rejected *whole* pages
become available for future allocation immediately; one retained partial
page can waste at most `B-1` slots. That is useful allocator behavior even
when single-request latency does not improve.

The physical pool itself remains allocated. Returning ten pages to its free
list does **not** release their GPU memory to PyTorch or the operating
system. Target and draft each reserve their own pool, and both model weights
remain resident. Pool capacity, peak request-owned pages, and wall-clock
speed are three different measurements.

M9b reports `peak_request_pages`, `free_pages_after`, `pool_pages`, and
`block_size` for each model. After an idle single request completes,
`free_pages_after == pool_pages` must hold. Repeated requests reuse the
pool's physical storage, not cached prefixes.

## Debugging the page lifecycle

The `M9b: paged speculative decoding and page rollback` entry in
[launch.json](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/.vscode/launch.json) runs `examples/generate.py` with the local
Qwen3-4B target, Qwen3-0.6B draft, FP32, and a small private pool.

Place breakpoints in `PagedSpeculativeState.forward()`, at `committed` in
the shared acceptance loop, and in `truncate_sequence()`. The verbose trace
prints target/draft tables after temporary writes and after rollback and
any required draft catch-up. These are physical page IDs, not token IDs.
The final statistics expose whether every page was released.

The [milestone example](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/examples/README.md#m9b-speculative-decoding-over-paged-kv)
uses `--speculative-cache paged`; omitting that option retains M9a. FP32 is
the debugging reference, not a universal proof of floating-point parity.

## Numerical qualification: paging is not a BF16 parity fix

On September 20, 2026, the local Qwen3-4B target / Qwen3-0.6B draft pair was
checked on six prompts, up to 32 output tokens, and draft windows 2 and 4
on RTX A6000 GPU 0. Every comparison uses emitted token IDs, not just text
that looks plausible.

| Comparison against contiguous ordinary greedy | FP32 matches | BF16 matches |
| --- | ---: | ---: |
| Paged ordinary, no draft | 6/6 | 5/6 |
| Paged draft/verify, both windows | 12/12 | 8/12 |
| Contiguous draft/verify control, both windows | 12/12 | 8/12 |

For both dtypes, paged speculation matched the corresponding contiguous
speculative mode on all twelve prompt/window pairs. That isolates the
storage change from the already observed BF16 verification limitation on
this fixture. It does not establish universal parity.

In BF16, the function-writing prompt differed even in paged ordinary decode.
Both speculative modes also differed on the KV-caching prompt. There is no
draft in the first comparison, so that discrepancy cannot be attributed to
accept/reject logic. The gather path and the contiguous path use different
attention layouts; the exact numerical source of this additional difference
has not been traced here. M9a's measured near-tie analysis remains useful
background, but it is not a diagnosis of every M9b mismatch.

The page-specific tests additionally force first/middle rejection, all
acceptance, EOS, zero/short budgets, exact and partial page boundaries,
noncontiguous physical tables, repeated requests, shared-page rejection,
partially successful allocation, and exceptions during either model's
forward. Poisoned unused slots must remain invisible; replacement appends
must match full recomputation within the FP32 test tolerance.

## Performance: a storage milestone, not a faster default

The BF16 timing sweep keeps both models resident, uses two warmups and seven
measured repetitions, and rotates the order of seven modes on the same
A6000. Batch size is one. Values below are median **end-to-end output
tokens/s**, including prefill, not decode-only rates.

| Mode | 128 prompt / 128 output | 2,048 prompt / 32 output |
| --- | ---: | ---: |
| Contiguous ordinary | 35.43 | 29.38 |
| Contiguous draft-2 | 32.25 | 26.01 |
| Contiguous draft-4 | 36.89 | 28.80 |
| Paged gather ordinary | 27.99 | 24.04 |
| Paged gather draft-2 | 25.96 | 21.59 |
| Paged gather draft-4 | 29.59 | 24.05 |
| Existing optimized paged/graph ordinary serving | 62.56 | 44.11 |

For paged versus contiguous draft-4, the paired median speed ratios are
**0.799×** (95% bootstrap interval 0.792–0.804) and **0.833×**
(0.831–0.842). Paging is slower in both measured cases. Draft-4 can recover
some of the short-prompt gather overhead relative to *paged gather ordinary*,
but it is still far below the optimized ordinary serving path.

Both synthetic timed prompts had 100% draft acceptance and matching
ordinary/speculative token IDs. That favorable case is not representative
of all real prompts; the six-prompt correctness fixture has rejections.
The optimized serving lane is a separate reference with FlashInfer and CUDA
graphs, not a matched-backend test of speculation alone.

M9b constructs its reusable pools before timing, then includes per-request
page allocation, draft prefill, rollback, and page release. Contiguous modes
include request-cache allocation. Token-level lanes exclude model loading
and tokenization; the serving lane includes tokenization. Speculative and
matched ordinary token-level lanes ignore EOS for fixed-length timing;
optimized serving honors EOS. All observed timed output counts equaled the
requested budgets. Neither pool creation nor first-use engine infrastructure
cost is a steady-state speed improvement.

Each pool has 512 allocatable 16-token pages plus a scratch page. Its physical
reservation is 1,210,318,848 bytes for the target and 941,359,104 for the
draft, unchanged when pages are freed. Peak request ownership was 16 pages
per model for the short workload and 130 for the long workload. Every
measured paged request returned to 512 free pages in each used pool.

The measured result fits the implementation: M9b adds page scatter/gather
and metadata work but has not introduced a fused verification kernel or
graph replay. The educational result is correct private-page ownership and
rollback. The default serving path remains unchanged.

### Fresh cross-engine calibration

The same day's separate ordinary-serving calibration uses Qwen3-4B BF16,
batch size one, two warmups, and five measured repetitions per workload.
These are fresh runs, not numbers copied from M9a.

| Engine / lane | 128 / 128 tokens/s | 2,048 / 32 tokens/s | Long-prompt TTFT (ms) |
| --- | ---: | ---: | ---: |
| Tinyserve optimized ordinary | 62.47 | 44.34 | 218.4 |
| llama.cpp | 74.29 | 37.75 | 409.0 |
| FreeToken | 73.15 | 51.15 | 173.5 |
| Ollama, native prompt-cache lane | 75.09 | 45.84 | 264.3 |

All arithmetic smoke checks passed. FreeToken reported 127/31 completion
tokens for the 128/32 budgets; the table uses those actual counts. Other
engines reported 128/32. External engines are timed through streaming HTTP,
whereas Tinyserve uses its engine-internal boundary. Prefix reuse is disabled
except for Ollama's separately labeled native-cache lane. These boundaries
make the table calibration context, not a perfectly controlled kernel race.

Tinyserve's retained M9a calibration was 62.46/44.36 tokens/s. The fresh
62.47/44.34 results show no meaningful observed ordinary-serving gain from
M9b; these are separate-run comparisons, not paired confidence bounds.
NInfer remains skipped: its checked-out build requires SM120a, while this
A6000 is SM86. No vLLM/SGLang result is implied.

The [measurement receipt](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m9b-paged-speculation-a6000-2026-09-20.json)
records source hashes, GPU identity, commands, per-repeat timings, output
counts, page statistics, numerical mismatches, external revisions, and raw
artifact hashes. The [M9a receipt](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m9a-speculative-a6000-2026-09-19.json)
retains the prior snapshot.

Final validation: **332 tests passed, 9 skipped**, including 48 M9b tests.
The FP32 `generate.py` debugger launch answered the arithmetic prompt with
`4`, exercised accepted and rejected proposals, and returned all eight
pages to each small debug pool. Ruff, diff checks, launch/receipt JSON,
local file links, and the rendered SVG were also checked. The skipped tests
are not claimed as validated configurations.
