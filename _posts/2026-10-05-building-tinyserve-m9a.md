---
layout: post
math: true
title: "Building tinyserve M9a: Speculative decoding: let the draft guess, let the target decide"
date: 2026-10-05 08:25:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Follow draft, verify, accept, and rollback through greedy speculation; distinguish FP32 correctness from BF16 drift."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m9-speculative-decoding.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M8g — Write compressed KV without the intermediate tensors]({% include tinyserve-post-url.html slug="building-tinyserve-m8g" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m8g-fused-kv-writes.md" %}) · Next: [M9b — Speculative decoding over paged KV]({% include tinyserve-post-url.html slug="building-tinyserve-m9b" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m9b-paged-speculation.md" %})

A decoder normally produces one token per target-model forward. Even when
the next few words are easy to predict, the large model still reads its
weights and walks through every layer once for each word.

Speculative decoding changes the amount of useful work in one target pass.
A smaller **draft** model proposes several tokens. The **target** evaluates
that proposed continuation in parallel, accepts its matching prefix, and
supplies the first correction. The draft is a guess, not a second authority
over the answer.

M9a implements the smallest complete version of that idea: one request,
greedy decoding, two dense Qwen models, and request-private contiguous KV
caches. It does not yet connect speculation to continuous batching, paged
KV, prefix sharing, CUDA graphs, sampling, or distributed execution.

The FP32 reference passes the measured token-parity checks. The BF16 path
is experimental: it has observed near-tie divergences and does not improve
on optimized ordinary serving. The [results below](#september-19-results-a-working-loop-not-a-serving-speed-promotion)
keep those boundaries explicit.

[![The request first prefills the target, then repeats draft, target
verification, prefix acceptance, and KV rollback.](/assets/tinyserve/m9a-speculative-lifecycle.svg)](/assets/tinyserve/m9a-speculative-lifecycle.svg)

## Why can the target verify several tokens at once?

Generation is sequential because the next input token is unknown. Once a
draft supplies candidate inputs, that particular dependency disappears:
the target can process them together using causal attention, just as it
processes a known prompt. It still computes every transformer layer for
every candidate. The opportunity is to reuse weight reads and amortize
launch overhead across several positions.

The logits **after** an input token predict the **following** token. Getting
this one-position shift wrong breaks verification even if every attention
and matrix multiplication is correct.

Let `C` tokens already be cached and `pending` be the final emitted token,
which has not been cached yet. For three proposals, the target receives:

```text
input:       [pending, d1, d2, d3]   shape [1, 4]
positions:   [C,       C+1,C+2,C+3]
predictions: [t1,      t2, t3, bonus]
logits:                            shape [1, 4, vocab_size]
```

Compare `d1` with `t1`, then `d2` with `t2`, and so on. Stop at the first
mismatch. If all three agree, the final row gives a fourth, bonus token.
Unlike ordinary prefill, verification needs **all** these vocabulary rows;
it cannot set `last_token_only=True`.

## One round, using actual implementation variables

Suppose five tokens are cached. Token `4` has already been emitted and is
the pending input. These are illustrative token IDs, not claims about which
English words Qwen assigns to them.

The draft proposes `[7, 9, 2]`. A single target call returns greedy
predictions `[7, 8, 13, 11]`:

| Target input | Logical position | Predicts | Decision |
|---|---:|---:|---|
| pending `4` | 5 | `7` | accept draft `7` |
| draft `7` | 6 | `8` | reject draft `9`; emit target `8` |
| draft `9` | 7 | `13` | discard; its history contains rejected `9` |
| draft `2` | 8 | `11` | discard for the same reason |

The round emits `[7, 8]`, not `[7, 8, 13, 11]`. In
[`generate_tokens()`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/speculative.py):

```python
base = 5
proposals = [7, 9, 2]
predictions = [7, 8, 13, 11]
accepted = 1
emitted = proposals[:accepted] + [predictions[accepted]]  # [7, 8]
committed = 7
pending = 8
```

The target temporarily wrote through position 8, so its cache length is 9.
The draft processed `[4, 7, 9]` to produce its three guesses; it did not
process its final guess `2`, so its cache length is 8. Both truncate to 7:
the five old tokens, pending `4`, and accepted `7`. Correction `8` is the
next pending input, not a cached token yet.

[![Concrete target and draft cache lengths during a partially rejected round,
including the prefix-offset attention mask.](/assets/tinyserve/m9a-kv-rollback.svg)](/assets/tinyserve/m9a-kv-rollback.svg)

## Rollback changes visibility, not the allocated buffer

[`KVCache`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/kv_cache.py) stores K and V separately, each shaped
`[layers, 1, capacity, kv_heads, head_dim]`. `seq_len` is the valid prefix
length shared by all layers. Each forward writes at that offset; the model
advances it once after the final layer.

`truncate(7)` sets `seq_len = 7`. It neither copies the accepted prefix nor
zeros the rejected suffix. The next forward writes new K/V starting at 7,
and `update()` exposes only the prefix through the new end. Bytes beyond
that view are invisible to attention. Tests deliberately fill the discarded
tail with NaNs, append a replacement, and compare with full recomputation.

This is possible because ordinary transformer KV is append-only. It is not
a rollback implementation for GDN/KDA recurrent state: those layers mutate
state that summarizes the entire history. They would need a different
checkpoint/replay design and are rejected by M9a.

When **all** proposals are accepted, there is no rejected history, but there
is still a bookkeeping detail: the draft has not processed its final guess.
If generation continues, one extra draft forward caches that guess. Both
caches then end immediately before the new target bonus token. This simple
catch-up step is included in the measured draft cost.

## The verification mask must include the old prefix

Previously the contiguous path supported an empty-cache prefill or one-token
decode. Verification needs a third case: `T > 1` with `C > 0` cached tokens.
Query row `i` must see keys `0` through `C+i`:

$$
\operatorname{visible}(i,j) = [j \le C+i].
$$

With `C=5, T=4`, the key axis has nine positions. Row zero sees six keys:
the five old ones and its own pending token. It cannot see any draft token.
The next row sees seven keys, and so on. A square, upper-left causal mask
would hide most of the prefix and give the wrong answer.

[`Qwen3Model.forward_layers()`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/models/qwen3.py) constructs the
offset mask once and shares it across layers. The existing explicit-mask
attention path expands KV heads for grouped-query attention and calls
PyTorch SDPA. This is a readable baseline, not a specialized verification
attention kernel. Paged attention takes its existing, separate path.

## EOS and output limits belong to the commit logic

- A target EOS from prompt prefill finishes without loading a draft cache.
- A draft EOS is only a proposal: the target may reject it.
- An accepted EOS ends the output immediately; do not append a bonus.
- A target correction can itself be EOS.
- The proposal count reserves one remaining output slot for the correction
  or bonus. With only one slot left, the round uses no proposals and becomes
  an ordinary target decode.
- A zero-token request returns immediately without a model forward.

Every request owns fresh caches, including on a repeated call. There is no
speculative suffix in a shared prefix cache and nothing to publish to another
request. Scheduler integration is a later milestone slice, not an implicit
property of this implementation.

## What does “same output” mean here?

For greedy decoding, accept a proposal only when it equals the target's
argmax. By accepting a prefix, every committed decision uses the same token
history as ordinary target decoding. A poor draft reduces acceptance and
speed; it does not get permission to select different tokens.

That algorithmic argument assumes the target computes the same decisions
in both execution shapes. Floating-point execution adds a separate caveat:
one-token and multi-token matmuls/attention can round differently. Nearly
tied BF16 logits can therefore flip an argmax. We test FP32 token parity,
test causal logits directly, and report real-model BF16 parity separately;
we do not claim universal bitwise equivalence from a small prompt fixture.

This is **not** the probabilistic speculative-sampling algorithm. At nonzero
temperature, matching draft and target sampled tokens is not sufficient to
preserve the target distribution; acceptance and correction distributions
must be handled explicitly. M9a rejects sampling options at the CLI.
The general approach and exact-distribution construction are described in
[Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*](https://proceedings.mlr.press/v202/leviathan23a.html).

## Compatibility is more than the model name

The local Qwen3-0.6B draft and Qwen3-4B target have identical `tokenizer.json`
files, including vocabulary and merge rules. Both model vocabularies have
151,936 output rows and use EOS ID 151645. The tokenizer itself exposes
151,669 entries; model padding rows are not additional text tokens.

`LLM.generate_speculative()` compares complete fast-tokenizer definitions
and special-token mappings before running. It templates and tokenizes the
prompt with the target **once**, then shares those IDs with the draft. It
does not separately apply the draft's chat template.

Both models must be unquantized dense Qwen, on the same device and dtype.
The first measured path is BF16; FP32 supplies the numerical checks. M8's
packed-weight paths remain available elsewhere but are not silently mixed
into this baseline.

## When can it be faster?

If a round accepts `a` draft tokens, it normally emits `a+1` tokens. A rough
break-even condition is:

$$
T_{\mathrm{draft}} + T_{\mathrm{verify}} + T_{\mathrm{bookkeeping}}
< (a+1) T_{\mathrm{ordinary\ decode}}.
$$

Draft time includes its serial proposals and any all-accepted catch-up.
Verification processes `k+1` inputs even if the first proposal is rejected.
It also projects all `k+1` vocabulary rows. Draft prefill and its additional
weight/KV memory are real costs. A longer proposal window is not inherently
better: it can add work without extending the accepted prefix.

The target emits the first token before draft prefill, so TTFT measures the
target's allocation and prompt forward. Draft prefill appears in the gap
before the next output and in end-to-end latency. Loading/tokenization are
outside the token-level timing boundary. M9a returns a completed result,
not a streaming API; TTFT is an internal first-token-available timestamp.

[`bench_speculative.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/examples/bench_speculative.py) keeps both models
resident, rotates ordinary/draft-window/optimized-serving order, warms each
path, and reports paired confidence intervals. Ordinary decoding shares the
same contiguous-cache implementation. A separate paged/graph lane prevents
an improvement over the educational eager baseline from being mistaken for
an improvement over Tinyserve's optimized serving path.

## September 19 results: a working loop, not a serving-speed promotion

The local RTX A6000 (SM86, GPU 0) ran Qwen3-4B as target and Qwen3-0.6B
as draft. No other compute process occupied that GPU during the runs.
Both models stayed resident in the paired comparison; two warmups preceded
seven measured, rotating-order repetitions for each workload.

| BF16 path | p128 / o128, B1 (tok/s) | p2048 / o32, B1 (tok/s) |
|---|---:|---:|
| Ordinary contiguous eager | 34.87 | 28.43 |
| Speculative, draft window 2 | 31.77 | 24.39 |
| Speculative, draft window 4 | 36.41 | 27.07 |
| Existing paged/graph serving | 62.35 | 44.20 |

These are median output tokens divided by whole-request duration, including
prefill—not steady-state decode-only rates. All paths produced the full
128/32-token budgets on these synthetic prompts. The speculative and
ordinary timed token IDs agreed, with **100% draft acceptance**: favorable
workloads, not evidence that arbitrary user prompts have that acceptance.

Window 4's median paired speed ratio against eager was **1.036×**
(95% bootstrap interval 1.029–1.048) on the short prompt and **0.984×**
(0.967–0.992) on the long prompt. Median paired ratios need not equal the
ratio of the two independently computed median throughputs. Neither case
beats the existing paged/graph path. M9a does not replace ordinary serving.

The real-model numerical fixture is a separate check: six ordinary prompts,
up to 32 output tokens each, with draft windows 2 and 4. **FP32 matched all
12 comparisons. BF16 matched 8 of 12**; the code-generation and KV-caching
prompts diverged under both draft windows. The BF16 API warns about this
boundary; the debugger example uses FP32. This is not a qualified lossless
BF16 deployment path.
Window-4 acceptance on those FP32 teaching prompts ranged from about 30%
to 100%, illustrating why the fully accepted timing fixtures are favorable.

A first-divergence diagnostic makes the numerical boundary concrete. On
the code prompt, ordinary BF16 tied `def` and `Wait` at 22.375; argmax chose
the lower token ID, `def`. Verification gave `Wait` 22.375 and `def` 22.25.
On the KV prompt, ordinary preferred ` Without`, while verification tied
it with the lower-ID ` For`. One-token replay from the **same speculative
prefix KV** retained the changed decisions in both cases. Thus differences
had already propagated into the prefix state; this is not just a claim
about rounding in the final vocabulary projection. The FP32 and mask tests
pass, but the observed BF16 token-parity failures remain explicitly open.

Target model-state tensors total 8,822,848,512 bytes; the draft adds
1,503,264,768 bytes before its KV allocation. Those are tensor-state bytes,
not a claim about peak allocator reservation. The current loader's tied
embedding/head allocation behavior is unchanged by M9a.

### Fresh cross-engine calibration

The same September 19 campaign ran ordinary BF16 Qwen3-4B on four engines,
with two warmups and five measured requests for each B1 shape. These are
fresh runs, not M8's 0.6B results. External engines did **not** use a draft.

| Engine / lane | p128/o128 throughput (tok/s) | p2048/o32 throughput (tok/s) | Long-prompt TTFT (ms) |
|---|---:|---:|---:|
| Tinyserve paged/graph | 62.46 | 44.36 | 216.6 |
| llama.cpp, BF16 GGUF | 74.32 | 38.34 | 401.7 |
| FreeToken, BF16 | 73.08 | 50.39 | 182.4 |
| Ollama, BF16 GGUF, native cache | 75.18 | 44.55 | 276.8 |

Tinyserve measures engine-internal time; external engines include streaming
HTTP. Prefix reuse is disabled for Tinyserve, llama.cpp, and FreeToken.
Ollama's native prompt-cache lane is labeled separately. All eight conditions
passed the arithmetic smoke check. FreeToken reports 127/31 completion
tokens for the 128/32 caps; its throughput uses those actual reported counts,
not an assumption of identical decoder-step counts across engines.

This comparison covers **B1 only**, matching M9a's supported scope. No
batched speculative performance is established. The local ninfer checkout
requires `sm_120a` and was not run on this SM86 GPU. No fresh vLLM/SGLang run
is claimed. Model precision, context/cache capacities, command
lines, and the paired measurements are retained in the
[M9a evidence receipt](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m9a-speculative-a6000-2026-09-19.json).

Full repository validation passed **284 tests, with 9 skipped** on the
single-visible-GPU test configuration. The 46 new focused cases cover
offset causality, poisoned-tail rollback, rejection/acceptance patterns,
EOS, token budgets, tokenizer/model compatibility, CLI guards, and recovery
after an injected forward failure. Ruff and local figure/link checks also
passed. The real-model FP32/BF16 results above are additional diagnostics,
not hidden inside that unit-test pass count.

## Code and debugger map

| Responsibility | Implementation |
|---|---|
| Prompt/tokenizer boundary | `LLM.generate_speculative()` |
| Draft, verify, acceptance, commit | `tinyserve/speculative.py:generate_tokens()` |
| Cache visibility rollback | `KVCache.truncate()` |
| Prefix-offset causal mask | `Qwen3Model.forward_layers()` |
| One-prompt CLI | `examples/generate.py --draft-model ...` |
| Paired timing and token checks | `examples/bench_speculative.py` |
| Acceptance tests | `tests/test_speculative.py` |

The **M9a: greedy draft, target verification, KV rollback** debugger entry
uses the local 4B/0.6B pair in FP32 and prints each round's proposals, target choices,
accepted count, temporary cache lengths, and committed length. The
[milestone example](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/examples/README.md#m9a-greedy-speculative-decoding)
uses the same CLI.

## Boundary for the next slice

M9a establishes a complete draft/verify/rollback loop. Batched speculation,
paged block reclamation, graph-friendly verification, quantized drafts,
probabilistic sampling, and recurrent-state rollback are not implemented
here. Each changes a different correctness or scheduling contract and
should be introduced separately.
