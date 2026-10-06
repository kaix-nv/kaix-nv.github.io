---
layout: post
math: true
title: "Building tinyserve M9c: Verify draft tokens without gathering KV"
date: 2026-10-05 08:27:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Read paged KV directly during verification; retain the failed full-model BF16 gate despite passing kernel tests."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m9c-direct-paged-attention.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M9b — Speculative decoding over paged KV]({% include tinyserve-post-url.html slug="building-tinyserve-m9b" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m9b-paged-speculation.md" %}) · Next: [M9d — When the same history produces different logits]({% include tinyserve-post-url.html slug="building-tinyserve-m9d" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m9d-bf16-numerical-closeout.md" %})

M9b gave speculative decoding a page allocator, but its attention reader
still copied those pages into temporary contiguous tensors. The result was
easy to inspect, not fast: on the recorded A6000 short-prompt workload,
paged draft-4 reached 29.59 tokens/s versus contiguous draft-4's 36.89.
Those are **M9b measurements**, not results for the change in this chapter.

M9c changes how attention reads the cache. A FlashInfer kernel receives the
physical KV pool and its page table directly. The acceptance loop, physical
slot writes, private-page ownership, and rollback remain unchanged. This
is an opt-in, single-request, greedy path—not speculative continuous
batching, prefix sharing, or CUDA-graph replay.

**Status:** the direct-attention tests pass, but the preset full-model BF16
logit-drift gate fails. The implementation remains experimental; the
[numerical results](#numerical-results-kernel-tests-pass-the-model-level-gate-fails)
below must accompany any timing claim.

## Three forwards, two new readers

Let `C` tokens already be cached and `T` be the number of inputs in this
forward. Target verification supplies `[pending, d1, ..., dk]`, so `T=k+1`.
Each input produces logits for the following token, exactly as in M9a.

| Forward | Query count | M9c attention reader |
| --- | ---: | --- |
| Initial target or draft prefill | prompt length | Unchanged Torch attention over in-flight K/V |
| Draft step, catch-up, or ordinary target decode | 1 | Existing FlashInfer paged-decode wrapper |
| Target verification with proposals | greater than 1 | FlashInfer paged-prefill/append wrapper, causal |

The third API has “prefill” in its name, but it also handles appending
several queries to an existing paged cache. It does not recompute the old
prefix. M9c selects its FA2 backend for the current A6000 implementation.
FP32 remains on the Torch reference; explicitly requesting FlashInfer with
an unsupported dtype/device fails instead of silently running gather.

[![M9b gathers and expands the cached KV before attention; M9c sends queries,
the physical pool, and page-table metadata directly to a paged kernel.](/assets/tinyserve/m9c-direct-paged-attention.svg)](/assets/tinyserve/m9c-direct-paged-attention.svg)

## The metadata is the address translation

Use the same illustrative values as M9b: four-token pages, `C=5`, pending
token `4`, and proposals `[7,9,2]`. The target processes four inputs at
logical positions 5–8. After allocating enough space, its physical page
table is `[5,2,4]` and its visible KV length is nine.

`FlashInferAttentionBackend.plan_append()` builds:

```text
query indptr:           [0, 4]       # four query tokens
paged-KV indptr:        [0, 3]       # three physical page IDs
paged-KV indices:       [5, 2, 4]    # logical page order, not sorted physical order
last-page valid length: [1]          # only position 8 is valid on page 4
```

Both `indptr` arrays describe one request, but their units differ: **tokens**
for queries, **pages** for the KV table. The final-page length is
`(kv_len - 1) % page_size + 1`; an exactly full page has length `page_size`,
not zero. Page IDs and boundary arrays use INT32.

The model still writes each new K/V to
`block_table[position // page_size] * page_size + position % page_size`.
For each layer, the wrapper then receives:

- Queries: `[T, Hq, D]`.
- The layer's existing pool: `[num_pages + 1, 2, page_size, Hkv, D]`.
- The plan built from the request's page table.

There is no Python-side gather of the old prefix or repeated KV-head
tensor. The kernel loads the needed page tiles and handles grouped-query
attention directly. “Direct” does not mean zero data movement: attention
still reads K/V, computes scores and softmax, and writes an output
`[T,Hq,D]`. It avoids the separate full-context materialization before that
work.

## Causal alignment: the queries are at the end of the context

For verification, query row zero is not logical position zero. It is
position `C`. The permitted keys are:

```text
                     key positions
query position       0 1 2 3 4 5 6 7 8
5  (pending 4)       1 1 1 1 1 1 . . .
6  (proposal 7)      1 1 1 1 1 1 1 . .
7  (proposal 9)      1 1 1 1 1 1 1 1 .
8  (proposal 2)      1 1 1 1 1 1 1 1 1
```

This is a bottom-right-aligned causal mask:
`key_position <= kv_len - query_len + query_row`, or `j <= C+i`.
Using a fresh-prompt triangle starting at column zero would discard most
of the cached prefix. Making verification noncausal would let earlier
queries see later guesses. Neither error belongs in acceptance logic;
both are attention errors before token comparison even begins.

The plan uses `causal=True` and `pos_encoding_mode="NONE"`: the model has
already applied RoPE to Q and K. Applying RoPE again in the attention
wrapper would be another incorrect positional transformation.

## Rollback changes the next plan, not the kernel

Suppose the target predicts `[7,8,13,11]`. The loop accepts `7`, rejects `9`,
and emits correction `8`. Committed length becomes seven. The unchanged
M9b rollback returns page 4 and retains `[5,2]`, including the partially
filled final page. Correction `8` is pending and will overwrite logical
position 7 on the next forward.

An old attention plan still describes length nine and page table `[5,2,4]`.
It must not survive this ownership change. `PagedSpeculativeState.forward()`
therefore performs these steps in order:

1. Allocate any pages needed for the new append.
2. Plan from the **current** table and new visible length.
3. Run the model; each layer writes new K/V before attention reads it.
4. Advance valid cache metadata only after the model succeeds.

One plan is reused across all layers of that forward, not across subsequent
forwards with potentially different lengths or page ownership. Planning
uses the CPU-owned sequence metadata, so the adapter need not recover its
own page table from GPU tensors. FlashInfer's planning and metadata-transfer
costs are still included in the timed path.

The `finally` cleanup from M9b also covers planning exceptions. Target and
draft have separate wrappers and separate private pools. In the LLM facade,
each model's decode and append wrappers share its existing 128 MiB workspace;
their execution is sequential. There is no second 128 MiB float workspace,
but the append wrapper is not allocation-free: FlashInfer 0.6.14 also owns
an 8 MiB device integer workspace, an 8 MiB pinned host workspace, and a
128 KiB device length buffer, plus per-plan metadata. The standalone benchmark
factory creates the shared float workspace and both wrappers once per model
before timing.

## What should count as correctness?

The CPU tests check metadata units, exact-page-boundary lengths, direct pool
pointers, one plan per forward, re-planning after rollback, mode guards,
and cleanup when either model's plan raises. Wrapper spies exercise the
wiring; they do **not** establish that a CUDA kernel is correct.

GPU tests separately cover noncontiguous page tables, poisoned unused
slots, prefixes straddling page boundaries, and one-, two-, and five-query
forwards. Outputs are compared with FP32 attention on the same rounded
Q/K/V inputs. Changing the final candidate must not change earlier query
outputs. A small real model additionally tests rejection-tail replacement
and repeated requests with the same pool.

For the real Qwen3-4B / Qwen3-0.6B pair, the benchmark records exact token
agreement against contiguous ordinary, M9b gather, and direct-paged ordinary
references. It also feeds the **same continuation tokens** through gather
decode, gather verification, direct decode, and direct verification. That
teacher-forced comparison separates numerical drift from the cascading
effect of a different greedy decision.

The aggregate BF16 logit gates are fixed before measurement: finite outputs,
normalized RMS error at most 0.01, and mean KL divergence at most 0.01 against
gather decode. Token agreement and maximum absolute error are reported
separately. Passing these gates is not a claim of universal bitwise greedy
equivalence; M9a/M9b already demonstrated BF16 near-tie differences.

## Debugging and measuring the change

The `M9c: direct paged verification and draft decode` configuration in
[launch.json](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/.vscode/launch.json) uses `examples/generate.py` with:

```text
--speculative-cache paged --speculative-backend flashinfer --dtype bf16
```

Break in `PagedSpeculativeState.forward()` and
`FlashInferAttentionBackend.plan_append()`. Inspect `query_len`, `kv_len`,
the page table, and the last-page length before stepping into the model.
The final cache report counts `torch_prefill`, `paged_decode`,
`paged_append`, and `gather` forwards. In the direct path, cached forwards
must use decode/append, not gather; all request pages must be free at exit.

The checked debugger run used four target append forwards and fifteen draft
decode forwards, with zero cached gather forwards. It accepted 11 of 14
proposals, exercised rejection and page growth, and returned all eight pages
in each pool. Its answer began with `4`. This verifies that the public entry
point actually selects the new readers, not just that a standalone wrapper
can run.

`examples/bench_speculative.py --flashinfer-speculation` adds matched
direct-paged ordinary and speculative modes to the resident rotating-order
benchmark. Pool/wrapper construction and cold compilation are outside the
steady-state timing; per-forward planning, draft prefill, rollback, and
cleanup remain inside. The optimized ordinary paged/graph path is a separate
reference. The default backend remains gather until measurement justifies
a change; M9c does not assume that eliminating copies beats graph replay.

## Numerical results: kernel tests pass; the model-level gate fails

On September 20, 2026, all **29 M9c tests passed**: 16 CPU wiring/contract
cases and 13 CUDA cases on A6000 GPU 0. The full suite reported **361 passed,
9 skipped**; skips are not counted as verified behavior. GPU testing began
after the unrelated profiling jobs finished; no other process was stopped.
The FP32 regression fixture also retained all 30 ordinary/speculative
comparisons across six prompts and the contiguous/gather modes.

The BF16 real-model fixture, however, is **not numerically qualified by the
predeclared gate**. Against contiguous ordinary greedy, the up-to-32-token
continuations matched as follows:

| Mode | Matching prompts |
| --- | ---: |
| M9b gather ordinary | 5/6 |
| M9b gather draft-2 | 4/6 |
| M9b gather draft-4 | 4/6 |
| M9c direct ordinary | 6/6 |
| M9c direct draft-2 | 5/6 |
| M9c direct draft-4 | 4/6 |

These are agreement counts with one reference, not an accuracy benchmark.
Direct draft-2 still differs on the function-writing prompt; draft-4 also
differs on the KV-caching prompt. Passing six ordinary prompts does not
establish universal BF16 token equivalence.

The teacher-forced probe covers the first eight continuation inputs of each
prompt (48 rows). Every measured row had the same top-1 prediction, and KL
was below 0.01. The raw-logit RMS gate nevertheless failed:

| Mode versus gather decode | NRMSE range across prompts | Largest mean KL | Largest absolute logit difference |
| --- | ---: | ---: | ---: |
| Existing gather verification control | 1.21–1.83% | 0.002220 | 0.422 |
| Direct paged decode | 1.04–1.79% | 0.002294 | 0.438 |
| Direct paged verification | 1.25–2.07% | 0.001896 | 0.563 |

All 18 comparisons exceeded the fixed 1% NRMSE limit, including the six
existing gather-verification controls. The initial qualification run stopped
at this gate. The threshold was **not relaxed**. Subsequent timing is an
explicitly unqualified diagnostic experiment, not a successful numerical
promotion. Low KL and matching early top-1 choices do not erase a failed
logit gate or the later greedy divergences.

This bounds the conclusion: direct page addressing and causal isolation pass
their tests, but full-model BF16 execution does not meet the preset drift
contract. M9c stays opt-in and experimental. It changes neither the default
gather reference nor optimized ordinary serving.

## Diagnostic performance

The timing experiment asks two different questions: does removing the gather
improve M9b, and does speculation beat ordinary decoding with the same direct
reader? A faster cache reader alone does not establish the second claim.

The setup is one RTX A6000, unquantized BF16 Qwen3-4B target and Qwen3-0.6B
draft, PyTorch 2.13.0+cu130, and FlashInfer 0.6.14. These are single-request
results, not a concurrency or scheduler benchmark.

Both models remain resident for all seven modes. The draft window is four,
with two warmups and seven measured repetitions in rotating order. Reported
throughput is actual output tokens divided by whole-request elapsed time,
including initial prefill. Paired ratios compare modes within each repetition;
their 95% intervals bootstrap those seven ratios. They measure variability in
this run, not uncertainty over arbitrary prompts or machines.

| Mode | 128-in/128-out tokens/s | 2,048-in/32-out tokens/s |
| --- | ---: | ---: |
| Contiguous ordinary | 34.57 | 28.70 |
| Contiguous draft-4 | 35.83 | 28.16 |
| M9b gather ordinary | 27.42 | 23.35 |
| M9b gather draft-4 | 28.74 | 23.48 |
| M9c direct ordinary | 31.42 | 26.47 |
| M9c direct draft-4 | 33.08 | 26.18 |
| Ordinary paged/graph reference | 62.45 | 44.02 |

The paired comparisons answer the two questions:

| Draft-4 comparison | 128-in/128-out ratio [95% interval] | 2,048-in/32-out ratio [95% interval] |
| --- | ---: | ---: |
| Direct versus M9b gather | 1.153× [1.144–1.173] | 1.116× [1.111–1.121] |
| Direct versus direct ordinary | 1.051× [1.046–1.069] | 0.993× [0.992–0.999] |

Removing the gather improves the M9b speculative path on both workloads.
Speculation itself helps the short workload modestly against the matched
direct ordinary path, but loses slightly on the long-prompt/short-output
workload. Neither beats the optimized ordinary paged/graph reference.

Every timed proposal was accepted on this repetitive timing fixture, and
the timed speculative token IDs matched the contiguous ordinary reference.
That favorable acceptance rate is not representative of arbitrary prompts;
the separate six-prompt numerical fixture above still contains mismatches.
All modes generated the requested 128 or 32 tokens, and all private request
pages were released. Trace counters confirm the direct modes used paged
decode/append rather than the gather fallback.

The short-workload table uses a clean serial repeat. The first short sweep
is excluded because a CPU-only test overlapped it; its raw artifact is
retained. These timings remain diagnostic because the numerical gate above
failed. They justify keeping the reader as an educational opt-in, not
promoting speculative decoding to the default serving path.

## Fresh cross-engine calibration

The ordinary-serving calibration was also repeated on the same GPU:
Qwen3-4B, batch one, two warmups and five repetitions per
condition. This checks the surrounding serving baseline; these external
engines were **not** configured for speculative decoding.

| Engine / lane | 128-in/128-out tokens/s | TTFT (ms) | 2,048-in/32-out tokens/s | TTFT (ms) |
| --- | ---: | ---: | ---: | ---: |
| tinyserve ordinary, paged/graphs | 62.41 | 31.64 | 44.27 | 218.51 |
| llama.cpp, prompt cache disabled | 74.10 | 53.58 | 37.59 | 412.56 |
| FreeToken, naive cache | 73.13 | 29.48 | 51.16 | 173.46 |
| Ollama, native prompt-cache lane | 74.91 | 38.39 | 44.10 | 282.88 |

These are medians, and throughput includes prefill. FreeToken actually
reported 127 and 31 output tokens; its rate uses those counts, not the
requested 128 and 32. The other engines returned 128 and 32. Tinyserve's
boundary is engine-internal, while the external adapters measure streaming
HTTP. Ollama's native prompt caching is another distinct boundary. Do not
attribute differences in this table solely to an attention kernel. Ollama
also uses FP16 KV storage here, while tinyserve uses BF16 KV.

All eight arithmetic smoke checks passed; that is a launch sanity check,
not a task-accuracy evaluation. NInfer remains unmeasured because the local
checkout requires SM120a and this GPU is SM86. No vLLM or SGLang performance
result is claimed. M9c does not change the ordinary serving default measured
in this table.

## Evidence and boundary

The [measurement receipt](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m9c-direct-paged-attention-a6000-2026-09-20.json)
records source hashes, terminal commands, paired samples and intervals,
failed logit gates, token mismatch positions, pool-release counters, and
fresh cross-engine results. It retains the excluded run's artifact hash
rather than silently replacing it with the repeat.

The implementation reuses the existing attention backend and preserves
M9a's acceptance loop and M9b's rollback. The numerical follow-up is now
recorded in [M9d]({% include tinyserve-post-url.html slug="building-tinyserve-m9d" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m9d-bf16-numerical-closeout.md" %}), which extends the matched
history probe without relaxing this chapter's failed gate. Sampling,
concurrent requests, shared prefixes, and graphs remain outside M9c.
