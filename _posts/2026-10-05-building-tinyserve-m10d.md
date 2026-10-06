---
layout: post
math: true
title: "Building tinyserve M10d: Batch the filtering, keep the randomness private"
date: 2026-10-05 08:30:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Batch filtering without mixing random streams; preserve the deterministic-scan fix and failed timing qualification."
source_revision: e20a34815498560f9226a9e057ade31b2b62743f
source_document: docs/m10d-batched-sampling.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `e20a348`](https://github.com/kaix-nv/tinyserve/tree/e20a34815498560f9226a9e057ade31b2b62743f). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M10c — Randomness belongs to the request]({% include tinyserve-post-url.html slug="building-tinyserve-m10c" fallback="https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/m10c-request-sampling.md" %}) · Next: [M11 — Move the KV, not the prompt work]({% include tinyserve-post-url.html slug="building-tinyserve-m11" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m11-prefill-decode.md" %})

Status: **deterministic scan passes correctness; candidate performance is still
48/54. The follow-up greedy reference trial passes only 42/54 stability blocks;
batched serving remains disabled**.
This chapter covers ordinary single-device dense-Qwen sessions. Filtering
remains PyTorch except for a small sampler-local CUDA prefix sum implemented
in Triton. Both the scalar control and batched candidate use that same scan.
Ordinary serving selects the scalar sampler with the corrected seeded scan.
The first candidate changed four CUDA top-p boundaries. V2 matched the
reference's row shape, but later repetition showed that even the old scalar
scan could change its answer between identical calls. The approved revision
therefore explicitly changes that numerical contract; it does not weaken the
support-mask, probability, token-ID, or RNG-state gates.

## The model is batched; the sampler is not

At the end of a decode forward, the model returns `logits[B, V]`: one row
per active request and one column per vocabulary token. M10c correctly
gives each stochastic request its own random generator. Its
`ServingSession._sample()` nevertheless calls the whole sampler once per
row: temperature, top-k, top-p, softmax, then a random draw. The next row
repeats the same chain of tensor operations.

The [M10c receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10c-request-sampling-a6000-2026-09-25.json)
measured the following median **sampler-only**, host-visible times on an
RTX A6000, using Qwen3-0.6B's vocabulary of 151,936 tokens:

| Batch | Batched greedy | Row-by-row stochastic |
|---|---:|---:|
| 1 | 0.028 ms | 0.538 ms |
| 8 | 0.038 ms | 3.927 ms |
| 32 | 0.068 ms | 16.469 ms |

These retained measurements used fixed FP32 logits, temperature 0.8,
top-k 20, top-p 0.9, two warmups, and five measured samples of 16 calls.
They include host dispatch and the final token-ID readback, but **no model
forward**. The BF16 model run hosting the measurement does not make these
sampler inputs BF16. This is motivation, not an M10d speedup result.

The implementation explains a plausible source of cost: separate dispatches
for every row, plus full-vocabulary filtering. In particular, top-k masks
unwanted logits but does not shorten the tensor; top-p still sorts `V`
entries. The timing alone does not tell us how much belongs to sorting,
softmax, random selection, memory allocation, or host dispatch. A diagnostic
operator trace must establish that breakdown before attributing a gain.

## Two kinds of work, two ownership rules

Filtering is deterministic arithmetic on a request's logits and parameters.
Rows can share an operator invocation without sharing probability mass:
reductions, sorting, and cumulative sums must stay on the vocabulary axis.
Random selection is different: it advances persistent state owned by the
request. Batch position is temporary and must not become RNG identity.

The proposed split is therefore **batch the deterministic filtering; retain
one draw per stochastic request**. This is still several PyTorch operations,
not one fused kernel.

[![Current row-by-row filtering versus proposed batched filtering, with greedy rows bypassing randomness and independent generators retained for each stochastic row](/assets/tinyserve/m10d-batched-sampling.svg)](/assets/tinyserve/m10d-batched-sampling.svg)

One `torch.multinomial()` call accepts one generator, not a list of
request-owned generators. Replacing the row loop with one matrix draw using
a shared generator would change the ownership contract. Keep the small draw
loop and first remove the repeated filtering work.
[PyTorch multinomial API](https://docs.pytorch.org/docs/2.14/generated/torch.multinomial.html)

## A three-request example

For a four-token vocabulary, let the input logits be the logarithms of
the probability vectors below. Token IDs are the column indices 0–3.

| Model row | Request | Input probabilities | Temperature / k / p | Generator |
|---|---|---|---|---|
| 0 | A | `[.55, .25, .15, .05]` | `1 / 3 / .8` | A, seed 17 |
| 1 | B | `[.10, .20, .60, .10]` | `0 / 0 / 1` | none: greedy |
| 2 | C | `[.10, .20, .30, .40]` | `1 / 0 / 1` | C, seed 29 |

1. Greedy row B returns token **2** without consuming randomness.
2. Gather stochastic rows `[0, 2]` into a `[2, 4]` tensor. Keep the matching
   request IDs `[A, C]`; their temperatures, k, and p are row parameters.
3. For A, top-k removes the last token. Renormalized probabilities become
   approximately `[.579, .263, .158, 0]`. Top-p keeps the first two, giving
   final probabilities `[.6875, .3125, 0, 0]`. C's filters are disabled,
   so its probabilities remain `[.10, .20, .30, .40]`.
4. Draw from A's full-vocabulary row with generator A, and from C's row
   with generator C. Call the resulting token IDs `a0` and `c0`; these are
   symbols, not claims about what those seeds produce on every device.
5. Scatter into the original row order: the result is `[a0, 2, c0]`.

If B finishes and the next model batch is `[C, A]`, the generator list is
`[generator_C, generator_A]`. Neither generator is reset or swapped with
the other request's state. Preemption keeps that state while recomputing
KV; partial prompt chunks still make no sampling call. M10c's cancellation,
EOS, budget, failure, and close cleanup remains responsible for retirement.

## Proposed tensor path

Use `B` for all model rows, `S` for stochastic rows, and `V` for vocabulary
size. Parameters arrive as validated Python values from the request objects;
the candidate builds row tensors on the logits device. Their construction
and row gathering count toward the measured cost.

1. **Preserve the all-greedy shortcut.** If every temperature is zero, return
   the existing batched argmax. With only one stochastic row, reuse the
   reference sampler initially; batching has no filtering cohort to share.
2. **Gather stochastic rows.** Convert to FP32 `[S, V]`. Subtract each row's
   maximum, then divide by its temperature `[S, 1]`, using M10c's lower
   clamp at the FP32 normal minimum. Never divide greedy rows by zero.
3. **Apply heterogeneous top-k.** Compute `K = max(k)` from the Python
   parameters. If `K > 0`, a batched `topk(K)` gives each enabled row its
   own kth threshold. Mask scores strictly below that threshold. A row
   with `k=0` bypasses the mask. Ties survive; this is not truncation to
   exactly k token IDs. A large k in one row can increase work for its peers.
4. **Apply heterogeneous top-p.** When any row has `p < 1`, sort the masked
   `[S, V]` logits along V and softmax along V. Apply the shared fixed-order
   prefix sum independently to each row, without a Python row loop. Then form
   the shifted `cumulative > p[:, None]` mask. Disable it for `p=1` rows.
   Scatter the mask back to vocabulary order and mask the logits there.
5. **Normalize and draw.** Softmax the filtered logits along V. Draw once
   per request using its own generator and a contiguous `[1, V]` row, as
   in M10c. Scatter the resulting IDs alongside greedy IDs, then perform
   one host readback for the final batch.

The algorithm sketch deliberately retains vocabulary order and width for
the random call. Drawing from a compact top-k vector or drawing in sorted
order can preserve the abstract distribution while changing how a seed maps
to token IDs and how generator state advances. Those are separate changes,
not shortcuts to include silently in this milestone.

Top-p's strict comparison is also intentional. If cumulative mass is
exactly p, the shifted `>` mask can retain one additional token. Changing it
to `>=` would change M10c's behavior. Equal logits are another boundary:
M10c does not request stable sorting, and stable tie ordering is an explicit
PyTorch option. Do not introduce a different tie policy just for the batch
path; require reference support-mask parity on ties and threshold cases.
[PyTorch sort API](https://docs.pytorch.org/docs/2.14/generated/torch.sort.html)

## A fixed addition order for top-p

In real arithmetic, `a + b + c` has one answer. FP32 rounds after each
addition: `(a + b) + c` need not equal `a + (b + c)`. A parallel prefix sum
chooses many such groupings. Near `p=0.9`, a one-ULP difference can change
which token crosses the threshold. Merely giving both callers a `[1,V]`
tensor does not fix an algorithm whose grouping depends on execution order.

[`sampling_scan.py`](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/sampling_scan.py) makes the grouping local
and fixed. Kernel 1 scans each 1024-token tile and writes its last value as
the tile total. Kernel 2 scans the tile totals in a fixed tree, selects the
sum of preceding tiles, and adds that offset to every local prefix. A short
last tile is padded with zero. One-tile rows skip kernel 2. Each output has
one writer: no atomics or inter-block look-back, and no global PyTorch flag.

[![Two scan passes: local prefixes within tiles, then fixed tile offsets; toy width four stands in for the implementation's 1024-token tile](/assets/tinyserve/m10d-fixed-scan.svg)](/assets/tinyserve/m10d-fixed-scan.svg)

The figure uses four-token tiles only to fit the page. At `p=0.9`, the
fifth token raises the cumulative mass to `0.91`; the shifted mask keeps
that token and drops tokens six through eight. Running this row alone,
alongside other requests, or after a row reorder uses the same tile shape
and addition tree. `num_warps=4` and the tile size are fixed, not autotuned.
The small scan of tile totals is repeated per tile to keep the teaching
implementation to two launches and avoid another global synchronization.

This is a **new numerical policy**, `fp32-fixed-1024-v1`, shared by the
scalar and batched private-generator paths. It can change rare seeded
outputs relative to M10c; matching a nondeterministic legacy boundary is
not a promise we can keep. The policy does not change strict `>`, shifted
masking, top-k ties, draw shape, token ordering, or request-owned RNGs.
CPU keeps its deterministic torch scan. Bitwise CPU/GPU equivalence, or
equivalence across compiler/device versions, is not promised. Those stacks
must qualify separately. The original failed receipts remain historical
evidence, not passing receipts for this new policy.

## What can improve—and what can get worse

The hypothesis is fewer repeated operator launches and larger independent
row workloads per invocation. It is **not** less model work or a reduction
in the vocabulary size. Full-vocabulary sorting and the per-request random
draws remain. If selection dominates, batching filters may have limited
payoff; operator traces will tell us where the time went.

Batching also increases simultaneous scratch storage. At `S=32` and
`V=151936`, one FP32 `[S,V]` tensor is **18.55 MiB** and one int64 index
tensor is **37.09 MiB**. Several such tensors, masks, and sort workspace can
coexist. These are size calculations, not measured peak memory. We must
measure peak allocated memory and report the latency/memory trade-off.

Mixed batches can waste work too: one top-p request may cause sorting for
otherwise unfiltered rows. Start with the simple path above and report
heterogeneous cases explicitly. Subgrouping, compact candidate sets, a new
counter-based RNG, fused filtering kernels, and CUDA-graph capture of the
sampler are outside this proposal. The deterministic prefix sum is the one
approved custom-kernel exception. A failed gate leaves scalar dispatch in place; it
does not automatically authorize another optimization series.

## Implementation boundary and defaults

The proposed helper belongs in `tinyserve/sampling.py`; the integration
point is `ServingSession._sample()` in `tinyserve/session.py`. Keep the
M10c `sample()` behavior as the reference except for the explicitly versioned
prefix-sum policy on the private-generator path. Legacy callers without a
generator keep their old arithmetic. The scheduler, cache ownership, model forward,
HTTP validation, and request-parameter copying do not need redesign.

Initially exercise the helper directly in tests and through a benchmark-only
sampler-selection hook. **No new public CLI flag, HTTP field, or sampler
backend option is proposed.** Existing temperature/k/p/seed defaults remain
unchanged. Only after the gates below pass would the ordinary session route
eligible stochastic batches through the new helper; all-greedy batches and
single-stochastic-row cases retain the reference shortcuts.

As in M10c, scope is ordinary single-device dense Qwen through live sessions,
offline `serve()`, and HTTP streams. Hybrid/distributed stochastic sessions,
quantized KV, speculative decoding, and M11 prefill/decode disaggregation
are not being enabled here. M8h remains separate. No model kernels change.

## Following the implementation

[`sample_batch()`](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/sampling.py) receives the model's `[B,V]`
logits plus parallel Python lists of request parameters and generators.
It first checks the row counts and rejects a stochastic row without its
private generator. All-greedy batches take one argmax; a single stochastic
row uses the scalar sampler with the shared scan. Neither shortcut enters the batched
filter helper.

For the three-request example, `stochastic == [0,2]`. `index_select()` gathers
A and C into `[2,4]`; `_batch_filtered_logits()` constructs their temperature,
k, and p columns, applies the batched filters, and returns vocabulary-order
scores. Inside top-p, `sampling_cumsum(sorted_probabilities)` uses fixed
1024-token tiles on CUDA; batch size does not change a row's addition tree.
The final softmax
gives two independent distributions. The short
draw loop slices `probabilities[j:j+1]`, preserving the reference's `[1,V]`
draw shape and request generator. `index_copy_()` restores model-row order.
There is no host token readback inside the helper; the test-only session
adapter performs one `.tolist()` after all rows have completed.

[`tests/test_batched_sampling.py`](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/test_batched_sampling.py) checks
the worked example and shortcuts. The existing request lifecycle and HTTP
tests run both the M10c path and a monkeypatched candidate path; this hook
is a test fixture, not a serving setting. The production
[`ServingSession._sample()`](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/session.py) remains unchanged.

[`qualify_batched_sampling.py`](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/examples/qualify_batched_sampling.py)
loads the sampler from the pinned Git revision, replaces exactly its seeded
prefix-sum operation, and checks the resulting function's syntax tree against
the current scalar implementation. It also checks that session dispatch has
not changed. The original source hash, adaptation name, and scan source hash
are retained separately. To compare support
masks and probabilities, it temporarily observes the reference's final
softmax input and stubs its draw. It restores both methods before real
seeded replay. The frozen [input fixture](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/data/batched_sampling.json)
and source hashes accompany each new, non-overwriting JSON receipt.

## Correctness gates before timing

Freeze and hash the test fixture before the first candidate GPU run. Pin
the M10c control to commit `31c240e`, with the explicit
`fp32-fixed-1024-v1` prefix-sum adaptation, not a moving branch or a copy of
the candidate filters. Test on CPU and the reserved A6000,
comparing each device to its own reference on the same runtime.

- Cover FP32 and BF16 input rows; vocabulary sizes 4, 64, and 151,936;
  batches 1, 2, 8, and 32; all-greedy, all-stochastic, and mixed parameters.
  Include disabled filters, `k=1`, `k=V`, ties at the kth score, equal-mass
  top-p boundaries, nearly tied scores, peaked/flat distributions, and
  M10c's finite-logit extreme-temperature cases (`1e-300`, `1e300`).
- Require identical retained-token masks. For final FP32 probabilities,
  require maximum absolute error at most `1e-7` and per-row L1 error at
  most `1e-6` against the reference, with finite normalized rows. Verify
  the four-token example independently of either implementation.
- Clone each request's generator state before reference/candidate calls.
  Require exact sampled token IDs and exact post-call generator states
  over 64 consecutive calls with seeds 0, 17, and 29. Small probability
  error alone cannot establish seeded compatibility. Check greedy and
  process-global RNG states are unchanged.
- Repeat fixed-logit streams with row reordering, unrelated random draws,
  peer admission/cancellation, preemption/recompute, zero output budgets,
  EOS, exceptions, and close. Retain the M10c lifecycle/HTTP tests and
  page/RNG cleanup checks. No silent fallback to the global generator.
- For Qwen3-0.6B, require FP32 and BF16 same-schedule reference/candidate
  token/event parity, plus repeated seeded replay under mixed live requests.
  Then run the full single-GPU suite with checkpoint-dependent KV tests.

A support-mask, sampled-ID, RNG-state, or lifecycle failure blocks promotion;
do not replace those gates with an aggregate frequency test. A tied-sort
discrepancy is evidence to stop and retain the reference, not permission to
quietly stabilize only the new path. Seeds do not promise matching text
across devices, runtime releases, or changed BF16 model schedules.
[PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

The linked API documentation is version 2.14; the retained M10c receipt used
`torch 2.13.0+cu130`. The installed signatures were checked while writing
this proposal. Qualification must record the runtime actually tested rather
than treating the documentation version as evidence of execution.

## Measurement and promotion gates

Use one reserved, idle GPU and stop only our run if a foreign job appears.
The following thresholds were frozen before measuring the candidate and
remain unchanged for the deterministic-scan revision.
Save source/artifact hashes, raw paired times, token counts, generator checks,
peak allocation, and GPU telemetry. Preserve failures and incomplete runs.

**Sampler-only pairs.** On the Qwen vocabulary, test batches 1/8/32 in FP32
and BF16, with all-greedy, homogeneous stochastic `(0.8,20,0.9)`, and mixed
rows cycling greedy, `(0.8,20,0.9)`, `(1,0,0.9)`, `(1,10,1)`. Use the M10c
monotonic logits plus frozen seeded-random and tied-logit panels. After two
warmup pairs, run ten alternating-order measured pairs, 16 calls per sample.
Reset matched private generator states outside each timed sample; include
row/parameter preparation and final host readback inside timing. Retain an
untimed operator trace separately; it must not contaminate paired timings.

Define the ratio as `reference time / candidate time`. To promote, require
both median and paired-bootstrap lower 95% bound to be at least **1.10**
for homogeneous stochastic batch 8 and 32 on every declared logit/dtype
panel. Require both to be at least **0.95** for all other declared conditions.
Use the existing bootstrap helper with a recorded seed. Require incremental
peak allocated scratch no more than **256 MiB above reference** at batch 32;
report full peaks too. These are acceptance budgets, not predicted gains.

**Whole-model pairs.** Reuse M10c's four Qwen3-0.6B BF16 conditions:
128/128 and 2,048/32 prompt/output budgets, batch 1/8, prefix caching off,
graphs on, prefill budget 8,192. Measure greedy and seeded stochastic modes
separately, using two warmup pairs and ten alternating measured pairs on
one loaded model. Require identical reference/candidate output token IDs
for each fixed schedule, and median plus lower-bound time ratios at least
**0.95** in every condition. A sampler win is not a serving win unless the
whole-model measurement supports that narrower claim. Different generated
lengths are not interchangeable work.

**Calibration.** Finally rerun the six Qwen3-8B standard cases against the
retained same-host baseline and the six Qwen3-4B HTTP conditions for each of
Tinyserve, llama.cpp, FreeToken, and Ollama. Keep these separate from paired
causal comparisons, including actual output counts, timing boundaries, and
native cache policies. M10c's FreeToken runs returned 127/31 tokens for the
128/32 budgets; a future run must report what it actually produced.

## First candidate: batching changed a top-p boundary

The [retained receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10d-batched-sampling-a6000-2026-09-26.json)
records the frozen fixture, source hashes, original failed attempts, and
GPU ownership checks. The runtime was `torch 2.13.0+cu130` on an RTX A6000.
These are correctness results, not timing measurements:

| Check | Result | What it establishes |
|---|---|---|
| CPU fixture | 24/24 cases pass | Reference masks/probabilities match exactly; token IDs and generator states match across 64 calls for each of three seeds |
| Focused helper, lifecycle, HTTP tests | 90 pass | The tested shortcuts and request ownership contracts survive the test-only binding |
| A6000 fixture | 22/24 cases pass | Both full-vocabulary mixed-batch cases fail the support-mask gate |
| Full single-GPU regression | 643 pass, 13 two-GPU tests skipped | Existing serving remains on M10c; checkpoint-dependent KV tests are enabled |
| Candidate model parity and performance | Not run | The sampler gate already blocks promotion |

The three seed values are base seeds 0, 17, and 29: original model row `i`
owns a generator seeded with `base_seed + i`. Each passing case therefore
checks 192 paired sampler calls, with identical row-to-generator mappings
in the reference and candidate.

The failing condition has `B=32`, `V=151936`, and 24 stochastic rows.
Its failing requests all use temperature 1, disabled top-k, and top-p 0.9.
FP32 input row 26 differs by one retained token. BF16 input rows 6, 14,
and 26 each differ by one. Filtering itself remains FP32 for both input
dtypes. Replay is explicitly skipped for these two failing cases, not
counted as a pass; the other 22 CUDA cases pass all three 64-call replays.

An untimed observation of the actual operators locates the first divergence:
sorted scores match exactly, sorted token IDs match exactly, and sorted
softmax probabilities match exactly. **The cumulative sums differ when
computed as one `[1,V]` row versus a `[24,V]` matrix.** The largest prefix-sum
error among these four rows is about `2.38e-6`. Floating-point addition is
not associative: independent mathematical rows do not guarantee identical
rounding when an operator's batch shape changes.

For FP32 row 26, the boundary looks like this (positions are zero-based
indices into the sorted row, not token IDs):

| Sorted position | M10c cumulative probability | V1 cumulative probability |
|---|---:|---:|
| 93028 | 0.899997592 | 0.899995863 |
| 93029 | **0.900000632** | 0.899998903 |
| 93030 | 0.900003433 | **0.900001884** |

M10c first crosses 0.9 at position 93029; the candidate crosses at 93030.
Because the shifted mask keeps the crossing token, the retained sets contain
93,030 versus 93,031 tokens. Vocabulary token **99474** is the extra survivor.
Its final probability is about `3.36e-6`, exceeding the `1e-7` probability
error budget; row L1 error is about `6.77e-6`, exceeding the `1e-6` budget.
This is a discrete filtering difference, not merely harmless rounding in an
otherwise identical distribution.

There was also a harness correction before CUDA qualification. The first
CPU attempt mistakenly applied the candidate-versus-reference L1 budget as
an absolute sum-to-one tolerance. The reference and candidate were identical,
but both sums missed one by up to `2.38e-6` on that CPU panel. The corrected
harness reports both normalization errors and checks candidate normalization
against the reference. It does not change the frozen input fixture or any
of the published support, probability-parity, token, or RNG gates. Both
attempts are retained. A separate full-suite invocation with CUDA hidden
was unsuitable for existing CUDA-only tests; it is not a passing regression.

The v1 outcome was therefore **keep M10c**. No serving selection, public option,
or default changes. No sampler timing, model timing, operator timing trace,
standard-suite performance run, or cross-engine calibration follows this
failed gate. Those remain necessary before claiming a completed optimization.
The subsequently approved v2 revision keeps the scan row-wise. It changes
neither tie policy nor tolerances and reuses the same frozen fixture. Fewer
filter launches remain a hypothesis: the extra scan loop and concatenation
must count toward the revised candidate's measured cost.

## V2: preserve the reference scan shape

The revision changes one operation, not the top-p definition:

```python
sorted_probabilities = sorted_scores.softmax(-1)  # still [S,V]
cumulative = torch.cat([
    row.cumsum(-1) for row in sorted_probabilities.split(1)
], dim=0)
```

Each scan sees `[1,V]`, just as in M10c. Sorting, softmax, mask construction,
and vocabulary-order normalization remain batched. The `[1,V]` private draw
loop is unchanged. The scan results are concatenated before applying the
same strict, shifted top-p mask; there is no new tie policy or tolerance.

The original frozen fixture passes **24/24 on CPU and 24/24 on the A6000**,
including all 192 paired calls per case. Qwen3-0.6B also passes reference/
candidate token and event parity in FP32 and BF16, with repeated seeded
offline/live replay, peer cancellation, and page/RNG cleanup. The full
single-GPU suite passes **648 tests**, with 13 two-GPU tests skipped and
checkpoint-dependent KV tests enabled.

[`bench_batched_sampling.py`](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/examples/bench_batched_sampling.py) keeps
its candidate session binding private to the harness. It preserves the
all-greedy session shortcut before preparing candidate metadata, and checks
the hashes of passing CPU/CUDA/model receipts before allowing timing. The
sampler matrix covers 54 conditions with the original two warmup pairs,
ten alternating measured pairs, 16 calls per sample, and bootstrap seed 0.
Parameter tensor creation, gathers, per-row scans/draws, concatenation,
scattering, and host token readback all count toward the candidate's time.
The operator trace is collected separately, after timed samples finish.

The [v2 receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10d-batched-sampling-v2-a6000-2026-09-27.json)
retains all 54 conditions and ten measured pairs per condition. Token IDs
and generator states match in every measured pair. **53/54 performance
conditions pass**, but promotion requires all of them:

| Sampler-only result | Measurement |
|---|---:|
| Homogeneous batch 8, across both dtypes and all three panels | 2.260–2.316× median speedup |
| Homogeneous batch 32, across both dtypes and all three panels | 2.987–3.401× median speedup |
| Largest extra peak allocated scratch above reference | 240.61 MiB, below the 256 MiB budget |
| Largest full peak allocated memory, reference / candidate | 24.13 / 264.74 MiB |
| Failing condition | FP32, tied logits, batch 1, homogeneous sampling |
| Failing condition's median reference/candidate ratio | 0.9883 |
| Failing condition's paired-bootstrap 95% interval | [0.5954, 1.3296], below the required 0.95 lower bound |

All twelve homogeneous batch-8/32 conditions pass both the median and
lower-bound 1.10 thresholds. The isolated failure is in a case with only
one stochastic row, which still uses the reference sampler through the
candidate wrapper. Its median is close to parity, but the measurements are
highly variable: mean per-call latency in the 16-call samples ranges from
543–1,590 microseconds for the reference, versus 556–1,561 for the candidate.
This interval **does not establish
a definite slowdown**, nor does it establish the required non-regression.
The source of that timing variability has not been diagnosed. No pairs are
discarded, no tolerance is widened, and no replacement run is substituted.

The separate operator trace uses FP32 random logits and homogeneous sampling
at batch 32. It helps explain the mechanism without being treated as another
timing sample:

| PyTorch operator invocations, one batch-32 call | M10c | V2 |
|---|---:|---:|
| Top-k | 32 | 1 |
| Sort | 32 | 1 |
| Softmax | 64 | 2 |
| Cumulative sum | 32 | 32 |
| Multinomial | 32 | 32 |

These are operator invocation counts, not GPU kernel-launch counts. The
trace supports the intended reduction in repeated filtering dispatch;
private random draws and reference-shaped scans remain. Larger concurrent
intermediates explain the need to report scratch memory alongside latency.

V2 is therefore a correctness-passing **experimental helper, not a promoted
serving optimization**. M10c remains selected. Whole-model paired timings,
the Qwen3-8B standard performance suite, and fresh cross-engine calibration
are not run after this sampler gate failure. The measured sampler gains do
not establish end-to-end serving speedup.

## A/A: can this timing window distinguish identical code?

Before changing the candidate, test the measurement itself. **A/A** means
both labels call the *same M10c function object*, `ServingSession._sample`.
There is no candidate wrapper. If their measured ratio varies widely,
the short-window procedure can report an apparent difference without any
implementation difference. That would not prove the cause of the earlier
candidate/reference failure: it is a new diagnostic on a new timing window.

The [diagnostic protocol](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/data/sampling_timing_aa.json) is frozen
before measurement. It reuses the failed condition's FP32 `[1,151936]` tied
logits, private input seed 101, temperature 0.8, top-k 20, and top-p 0.9.
Both labels reset the request's CUDA generator to seed 0 before each sample.
The [harness](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/examples/check_sampling_timing.py) runs three fixed blocks,
separated by ten-second gaps. Each block has two warmup pairs and ten measured
pairs, alternating A-first and B-first; each sample times 16 calls. Python
dispatch, filtering, random draws, and host token readback are all included,
exactly as in the v2 timed region. All tokens and post-call generator states
must agree; warmups are retained but excluded from timing statistics.

For each block, compute the median A/B latency ratio and its paired-bootstrap
95% interval (10,000 resamples, seed 0). This diagnostic asks whether both the
median and the entire interval fit inside `[0.95, 1/0.95]`, approximately
`[0.95, 1.0526]`. The reciprocal upper bound makes the check symmetric when
the two identical labels are swapped. All three blocks must satisfy it;
there is no selective retry. This is an operational stability check, not a
guarantee about every future measurement window or an independent-samples
claim about the calls within a sample.

This A/A criterion does not change M10d's promotion thresholds. Even a stable
A/A result cannot qualify a candidate that has not passed its own gates.

The [preflight receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10d-sampling-timing-aa-preflight-2026-09-27.json)
records the frozen hashes and CPU validation: 99 tests passed, with two
CUDA-only tests skipped. This includes a CPU-only orchestration test for
alternating order, generator resets, gaps, and retained warmup/measured pairs;
its timings are not performance evidence. At that checkpoint, GPU A/A
measurements were pending:
another workload claimed GPU 0 before launch, and the ownership check refused
both launch attempts on September 27. No active job was interrupted. That
receipt records preparation only; it is retained separately from the completed
GPU diagnostic below.

### The identical-reference check also exposes timing variability

After GPU 0 became available on September 28, the unchanged protocol completed
all three blocks. The [A/A receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10d-sampling-timing-aa-a6000-2026-09-28.json)
retains every warmup and measured pair, tokens, RNG-state hashes, telemetry,
and ownership samples. The monitor observed no foreign process on GPU 0.
All **36 paired token sequences and final generator states match exactly**;
six warmup pairs are excluded from the statistics, leaving ten measured pairs
per block. Each table entry is the median of paired A/B ratios, not the ratio
of two independently computed medians.

| Block | Median A/B | Paired-bootstrap 95% interval | Within `[0.95, 1.0526]`? |
|---|---:|---:|---|
| 1 | 0.9628 | [0.8906, 1.0296] | No |
| 2 | 1.0004 | [0.9901, 1.0075] | Yes |
| 3 | 1.0116 | [0.9951, 1.0363] | Yes |

The fixed three-block diagnostic therefore **fails its stability criterion**.
In block 1, mean per-call latency across the 16-call samples spans 496–1,307
microseconds for A and 518–1,445 for B. Because A and B are identical callbacks,
this apparent difference cannot be attributed to a candidate implementation.
The later two passing blocks do not erase the first block or justify a retry.

This result shows that the short timing procedure can fail its confidence-band
check on identical code. It does **not** isolate CPU scheduling, GPU clock
behavior, warmup, or any other particular source of variability. Nor does it
prove that the earlier candidate/reference failure had the same cause or
that v2 satisfies its non-regression requirement. M10c stays selected; the
original v2 result remains 53/54, and no candidate, model, standard-suite,
or cross-engine timing is rerun. The focused CPU regression still passes
99 tests, with two CUDA-only tests skipped; this is not a new full GPU suite.

The approved follow-up is one reference-only measurement change: increase calls
per sample from 16 to 256, keeping the inputs, warmup-pair count, measured-pair
count, gaps, bootstrap, and stability band fixed. Longer samples may reduce
the influence of brief timing disturbances, but that is a hypothesis to test,
not a correction to the retained results. The separate
[256-call protocol](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/data/sampling_timing_aa_256.json) differs from the
original JSON in exactly that field. `check_sampling_timing.py --long-samples`
selects it, rejects other protocol changes, and records the selected protocol's
path and hash. The default remains the original 16-call diagnostic. Neither
the timed loop nor the statistics function changes.

The warmup still uses two pairs per block, but those samples now contain 256
calls too: more calls per sample also means more warmup work. This experiment
does not separate that effect from averaging over longer measured windows.
The [256-call preflight receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10d-sampling-timing-aa-256-preflight-2026-09-28.json)
records the frozen hashes and 104 passing CPU tests, with two CUDA-only tests
skipped. It records preparation while GPU 0 was occupied, not a timing result.
Candidate requalification requires a separately agreed protocol and remains
downstream of a satisfactory measurement check; this reference-only trial
cannot enable M10d by itself.

### More calls did not establish a stable measurement window

The [256-call GPU receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10d-sampling-timing-aa-256-a6000-2026-09-28.json)
records the completed follow-up on September 28. All three fixed blocks ran
once, with no foreign process observed on GPU 0. Every one of the 36 paired
256-call token sequences and final RNG states matches exactly. The input hash
and initial RNG states also match the earlier trial, and each sequence's
first 16 tokens reproduce its earlier counterpart. Warmups remain excluded
from the statistics, and all measured pairs remain in the receipt.

| Block | Median A/B | Paired-bootstrap 95% interval | Within `[0.95, 1.0526]`? |
|---|---:|---:|---|
| 1 | 0.9904 | [0.9472, 1.0171] | No: lower bound |
| 2 | 1.0207 | [0.9992, 1.1034] | No: upper bound |
| 3 | 0.9958 | [0.9211, 1.0169] | No: lower bound |

All medians lie inside the band, but every confidence interval extends beyond
it. **The 256-call diagnostic therefore fails in all three blocks.** A close
median alone is insufficient to establish the required stability. The longer
sample size has not made this procedure reliable enough in this window; it
does not follow that 256 calls is inherently worse than 16. The two trials
occurred in different windows, and increasing call count also increased the
amount of warmup work. Neither trial isolates the source of the variability.

The intervals were independently recomputed from the retained pairs. The
post-run focused CPU regression passes 104 tests, with two CUDA-only tests
skipped; that is not another full GPU qualification. No threshold changed,
no block was dropped or retried, and no candidate, model, standard-suite,
or cross-engine timing followed. M10c remains selected and v2's original
53/54 result is unchanged.

The approved follow-up is a bounded, separately instrumented CPU/CUDA
trace of the reference path, to examine host-side gaps and device execution
before changing measurement settings again. Profiling overhead must not enter
promotion timings, and a trace alone cannot establish candidate non-regression.
The completed diagnostic follows; M10d remains an experimental helper.

### Reference trace: stable GPU work, variable host cost

The [trace protocol](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/data/sampling_trace.json) keeps the same FP32
`[1,151936]` tied logits and private seed. It is **not another A/A performance
gate**: two untraced warmup pairs precede three contiguous traced pairs, each
with 256 calls per label. Both labels still call `ServingSession._sample`.
The [capture harness](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/examples/profile_sampling_timing.py) adds CPU/CUDA
events and per-call annotations, main-thread CPU time, and context-switch
counts. Shapes, stacks, and allocation profiling are disabled, but the
remaining instrumentation still changes the workload.

The host does not permit CPU scheduler tracing (`perf_event_paranoid=4`), and
scheduler wait-time accounting is disabled. No permissions or host settings
were changed. The available counters distinguish executing CPU time from
elapsed time, but cannot identify which process delayed a thread. Time spent
spinning in a driver can also count as CPU time; it is not all Python work.

The [trace receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10d-sampler-reference-trace-a6000-2026-09-28.json)
records a clean GPU-0 ownership check and exact token/final-RNG agreement for
all ten samples, including warmups. All 2,560 tokens also reproduce the prior
256-call reference sequence. The trace contains 1,536 annotated calls across
six profiled samples, with **52 CUDA kernels and 113 CUDA API calls per call**.
These are observed counts for this one input and software stack, not universal
costs of sampling.

The [analysis](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/examples/analyze_sampling_trace.py) merges overlapping
kernel, copy, and memset intervals, then clips that union to each annotation.
This avoids counting simultaneous GPU activities twice:

| Traced sample, execution order | Host wall (ms) | Main-thread CPU (ms) | Traced GPU work (ms) |
|---|---:|---:|---:|
| Pair 1 / A | 321.269 | 313.914 | 80.554 |
| Pair 1 / B | 271.176 | 270.540 | 80.554 |
| Pair 2 / B | 275.643 | 274.825 | 80.549 |
| Pair 2 / A | 274.239 | 273.643 | 80.540 |
| Pair 3 / A | 274.189 | 273.456 | 80.542 |
| Pair 3 / B | 309.355 | 308.756 | 80.556 |

CPU and GPU work overlap; **do not add these columns**. The host counters
include the small annotation-entry/exit overhead, whereas GPU coverage is
clipped inside the annotation. Traced GPU work stays near 80.55 ms while host
wall time varies by about 50 ms. Main-thread CPU time covers 97.7–99.8% of
the host interval, so long periods when that thread is not executing do not
dominate this capture. The time inside the annotation without traced GPU
work ranges from 190.61 to 240.48 ms; this is not automatically an OS
scheduling delay or proof that the entire GPU was idle.

A concrete pair of **calls**, not implementations, illustrates the difference.
The median-ranked call (`sample/1/A/call/24`, zero-based trace indices) takes
1.024 ms, including 0.316 ms of traced GPU work. The slowest call
(`sample/2/B/call/186`) takes 4.012 ms, including 0.315 ms of GPU work. Both
launch 52 kernels. Their host time inside `cudaLaunchKernel` totals 0.202 ms
and 0.703 ms respectively; inclusive `aten::multinomial` regions take 0.236 ms
and 1.122 ms. These host intervals contain profiler/driver overhead, and the
operator regions are nested, so they are not independent costs to sum.

**Within this instrumented capture, the variation is mostly outside GPU
execution.** It does not point to kernels doing more work in the slower
samples. It does not isolate Python, PyTorch, driver, CPU-frequency, or
profiler costs, and it cannot establish the cause of the earlier unprofiled
A/A failures. The profiler changes the measured path; none of these times
can replace a promotion benchmark.

The interval unions were checked independently, and the focused CPU suite
passes 110 tests, with two CUDA-only tests skipped. M10c remains selected;
the original candidate gates and both failed A/A receipts remain unchanged.
The follow-up below checks wall/thread-CPU time without the full profiler.
Neither diagnostic changes the selected sampler.

### Unprofiled replay: variability persists while the CPU thread is executing

The [counter protocol](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/data/sampling_timing_cpu.json) pins the existing
256-call A/A protocol by hash. It keeps all three blocks, ten-second gaps,
two warmup pairs and ten measured pairs per block, alternating order, inputs,
seeds, and stability thresholds. `check_sampling_timing.py --thread-cpu`
adds just **two thread-CPU clock reads per sample**, outside the original
wall-clock interval. There is no CUPTI capture, profiler, or per-call marker.
The original 16-call and 256-call modes remain available without CPU counters.

What do the two clocks tell us? Wall time includes everything until the
sample completes, including waits. Thread CPU time counts time when the
calling thread is executing, including time spent spinning inside a driver;
it excludes time when that thread is not running. Their difference is **not
GPU execution time or an OS scheduling-delay measurement**. The CPU interval
also encloses the wall-clock reads, so its boundary is slightly wider. Small
negative wall-minus-CPU differences are possible and must not be clamped away.

The [September 28 replay receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10d-sampling-thread-cpu-a6000-2026-09-28.json)
retains every sample and the frozen preflight hashes. All 72 samples, including
warmups, reproduce the previous 256-token sequence and final RNG state exactly:
18,432 tokens in total. No foreign process was observed on GPU 0; GPU 1's
unrelated workload was left untouched. The host CPUs were still shared.

| Block | Median A/B | Paired-bootstrap 95% interval | Stability check |
|---|---:|---:|---|
| 1 | 1.0444 | [0.9396, 1.1261] | Fail: both bounds |
| 2 | 0.9387 | [0.8581, 0.9990] | Fail: median and lower bound |
| 3 | 0.9937 | [0.8588, 1.0958] | Fail: both bounds |

Across the 60 measured samples, wall time ranges from **135.609 to 334.562 ms**.
The main thread executes for 99.57–100.005% of each wall interval, with a
median of 99.84%. The fastest sample uses 135.282 ms of thread CPU time;
the slowest uses 333.883 ms. Wall-minus-CPU ranges from -6.25 to 686.44
microseconds; the tiny negative values reflect the different clock boundaries,
not negative waiting time.

**The variability survives removal of the full profiler, and long off-CPU
pauses do not dominate this replay.** This narrows the question, but does not
identify the cause. Driver spinning counts as CPU time, and without device
timestamps this replay cannot show whether GPU execution stayed constant.
It cannot isolate Python, PyTorch, driver, CPU-frequency, or GPU-wait costs,
nor explain a failure observed in an earlier measurement window.

All intervals were independently recomputed, and the focused CPU suite passes
117 tests with two CUDA-only tests skipped. All three blocks remain in the
receipt; no retry, threshold relaxation, candidate timing, model timing,
standard-suite run, or cross-engine calibration followed. GPU 0 was released.
M10c remains selected, and M10d v2's original 53/54 result is unchanged.
Further qualification needs a separately agreed, controlled experiment;
repeating the same trial until it passes would not establish a reliable win.

### A controlled CPU-affinity experiment

CPU affinity specifies **where a thread may execute**, not how much CPU time
it receives. An unpinned thread can move between eligible CPUs; pinning it
to one logical CPU removes that freedom, but does not reserve the core or
its SMT sibling. Pinning might change migration/cache effects, frequency,
or contention. It is therefore a useful intervention, not a direct count of
migrations or proof that migration caused earlier variability.

The [affinity protocol](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/data/sampling_timing_affinity.json) fixes
logical CPU 2 before measuring. Its SMT sibling, CPU 14, remains shared.
`check_sampling_timing.py --affinity` compares the original 24-CPU eligibility
mask with `{2}` on the **calling thread only**, in one process and GPU-0
window. It snapshots helper-thread masks outside timing and restores the
calling thread's original mask in `finally`, including after interruption.
No GPU clocks, thread counts, serving defaults, or other jobs are changed.

Four adjacent rounds use the fixed order `U/P, P/U, U/P, P/U`, where U is
unpinned and P is pinned. Each condition is first twice; both appear four
times. This extends the previous three-block diagnostic to eight blocks
for counterbalancing, rather than silently treating an old measurement
window as the control. Every block still uses two warmup pairs, ten measured
alternating A/A pairs, 256 calls per sample, the same inputs and private RNG,
and the same bootstrap and ratio band. Ten-second gaps separate all blocks.
Both labels still call the identical M10c function. Only sample-level
wall/thread-CPU counters are used; there is no profiler or per-call marker.

For each condition, every block's median and entire 95% interval must lie
inside `[0.95, 1.052631...]`. Keep all eight blocks and report both conditions;
do not substitute pooled results, select a different CPU afterward, or retry
until the criterion passes. Even a stable pinned condition cannot qualify
M10d or explain an older run. This is a reference-only diagnostic.

The [September 29 affinity receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10d-sampling-affinity-a6000-2026-09-29.json)
records the completed trial. Pinned passes **4/4 blocks**, while the
contemporaneous unpinned control passes **0/4**:

| Round | Condition, execution order | Median A/B | Paired-bootstrap 95% interval | Check |
|---|---|---:|---:|---|
| 1 | Unpinned, first | 0.9761 | [0.8390, 1.0256] | Fail |
| 1 | Pinned, second | 0.9979 | [0.9858, 1.0055] | Pass |
| 2 | Pinned, first | 0.9962 | [0.9891, 1.0040] | Pass |
| 2 | Unpinned, second | 0.9844 | [0.9268, 1.0834] | Fail |
| 3 | Unpinned, first | 1.0212 | [0.9095, 1.0764] | Fail |
| 3 | Pinned, second | 0.9884 | [0.9778, 1.0201] | Pass |
| 4 | Pinned, first | 1.0048 | [0.9942, 1.0189] | Pass |
| 4 | Unpinned, second | 1.0108 | [0.9747, 1.0787] | Fail |

All 192 samples, including 32 warmups, reproduce the prior 256-token sequence
and final RNG state exactly: **49,152 tokens**. The calling-thread mask is
verified at each switch and restored afterward; all ten observed helper
threads retain the original mask in the snapshots. No foreign process is
observed on GPU 0. These checks do not establish continuous CPU isolation:
the host cores and SMT siblings remain shared.

**Passing this check does not mean the latency tail is stable.** One measured
pinned pair takes 132.631 ms for A and 310.118 ms for B, a ratio of 0.4277.
That pair remains in the data and bootstrap; it was not discarded. The block
still passes because the declared statistic is the **median paired ratio**
and its interval, not the worst sample or a tail-latency bound. The measured
sample ranges are 127.794–194.133 ms unpinned and 129.830–310.118 ms pinned.
The isolated excursion remains unexplained; this is not a claim that every
pinned sample is faster or that pinning eliminates timing noise.

The intervention gives a usable result for this particular **256-call
reference procedure** in this window. It does not identify whether migration,
cache effects, frequency, shared-core contention, or driver behavior explains
the difference, and it does not retroactively qualify an earlier run. CPU
tests pass 127 cases, with two CUDA-only skips; the independent audit
recomputes all eight intervals and verifies the retained replay and masks.
M10c remains selected, the original 53/54 candidate result stands, and GPU 0
is released. No candidate or downstream performance phase was run.

At that point, the next proposed check was reference-only pinning at the **original 16-call
sample length**, frozen separately before measurement. Stability at 256 calls
does not establish it at 16. That check had not yet been run; neither candidate
qualification nor a serving change follows automatically from this result.

### Returning to the original sample length

The approved closeout starts with a separately frozen
[16-call affinity protocol](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/data/sampling_timing_affinity_16.json).
It differs from the 256-call affinity JSON only in its workload-protocol
hash: `check_sampling_timing.py --affinity-16` restores the original 16 calls
per sample while retaining CPU 2, both conditions, the eight-block order,
warmup-pair count, gaps, counters, and stability thresholds. Fewer calls also
mean less work inside each warmup; this is deliberately the original sample
length, not a claim that the earlier 256-call result automatically transfers.

All four pinned blocks must pass before a new candidate timing campaign.
That campaign will keep the candidate fixture, token/RNG requirements,
minimum speed ratios, and scratch-memory budget fixed. The remaining order
is sampler correctness, fixed-schedule model correctness, sampler timing,
whole-model timing, then the standard suite and HTTP cross-engine calibration.
Each phase retains its own evidence; a failed performance gate blocks the
downstream timing phases. Ordinary serving retains scalar dispatch until the full
qualification passes. No affinity change is proposed for production serving.

### Repetition exposed a separate correctness problem

Before the 16-call timing trial, a fresh full regression returned **685
passed, one failed, 13 skipped**. Its BF16 full-vocabulary mixed case changed
one support-mask bit, despite fresh standalone CPU and CUDA fixtures each
passing 24/24. The [repeatability receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10d-scan-repeatability-a6000-2026-09-29.json)
records why a one-shot success was insufficient:

| Same fixed BF16 fixture | Legacy CUDA scan | PyTorch deterministic scan, diagnostic only |
|---|---:|---:|
| Candidate/reference numerical passes | 17/32 | 32/32 |
| Reference row 10 support size across 64 calls | 92,823 or 92,824 | 92,823 |
| Repeated scan differences, each of eight inspected rows | 63/64 versus the first scan | 0/64 |

For row 10, token **150150** is the differing retained token. Identical sorted
probabilities are reused throughout; their hashes match between diagnostic
modes. Prefix-sum changes reach about `2.38e-7` across the inspected rows.
The installed PyTorch CUDA source routes a single floating row through CUB's
legacy scan unless deterministic algorithms are enabled. NVIDIA documents
that floating-point scan results need not be repeatable with that algorithm.
[CUB DeviceScan documentation](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceScan.html)

The diagnostic temporarily enabled PyTorch determinism in its own process
and restored the setting. It did **not** qualify that setting for serving,
measure performance, or authorize a global runtime change. It motivated the
explicitly approved sampler-local policy above. Correctness qualification
restarts against the adapted scalar reference, with all existing tolerances
and performance thresholds unchanged; downstream timing remains gated.

### Deterministic scan: correctness passes, promotion still blocked

The [new qualification receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10d-deterministic-sampling-a6000-2026-09-29.json)
records the shared `fp32-fixed-1024-v1` policy on the same A6000 and
`torch 2.13.0+cu130`. This is a new reference contract, not a retry of v2
until it happens to pass.

| Check | Result |
|---|---|
| CPU and CUDA sampler fixtures | 24/24 each; exact masks, tokens, and private RNG states |
| Sensitive full-vocabulary numerical check | 32/32 repetitions in each of FP32 and BF16 |
| Scan-specific checks | Repeated calls, row reordering, singleton/batch equality, tile boundaries, FP64 oracle, unchanged global settings |
| Qwen3-0.6B fixed-schedule parity | FP32 and BF16 pass, including live cancellation and seeded replay |
| Final single-GPU regression | 705 pass, 13 two-GPU/opt-in skips |
| Reference 16-call A/A | Pinned 4/4 blocks pass; unpinned 2/4 pass |
| Candidate sampler performance | 48/54 conditions pass; six greedy-only failures |
| Whole-model timing, standard suite, HTTP calibration | Not run: sampler performance gate failed |

The A/A trial retains all 192 samples and 3,072 token IDs, including
warmups. The candidate campaign then keeps CPU 2 fixed for the measuring
thread throughout each condition's ten measured pairs. Its two warmup
pairs run unpinned so lazy helpers initialize before pinning. Both methods
receive the same treatment; snapshots check helper masks, and `finally`
restores the caller's mask after each condition, including interruption.
Affinity setup is outside the unchanged timed call loop. This is benchmark
machinery, not a production affinity policy or a claim of CPU isolation.

All 540 measured candidate/reference pairs match token IDs and final RNG
states. The maximum incremental scratch allocation is **240.61 MiB**, below
the unchanged 256 MiB limit. All homogeneous stochastic and mixed-panel
conditions pass. Homogeneous batch-8 ratios span **2.72–2.79x**, and batch-32
ratios **3.80–3.86x**, across the six dtype/logit combinations. These are
**sampler-only** ratios against the scalar implementation with the same
new scan, not against the unmodified legacy M10c scan and not serving speedups.

These six greedy conditions fail the required median/lower-bound ratio of
at least 0.95:

| Input dtype | Logit panel | Batch | Median reference/candidate | Paired-bootstrap 95% interval |
|---|---|---:|---:|---:|
| FP32 | Random | 1 | 0.9931 | [0.9219, 1.0252] |
| FP32 | Tied | 1 | 0.9391 | [0.7888, 1.0500] |
| FP32 | Tied | 8 | 1.0075 | [0.9264, 1.0969] |
| BF16 | Monotonic | 8 | 0.9950 | [0.9414, 1.0020] |
| BF16 | Monotonic | 32 | 0.9842 | [0.9441, 1.0517] |
| BF16 | Tied | 8 | 1.0050 | [0.9486, 1.0220] |

Both greedy branches have the same syntax tree after renaming their loop
variable: parameter check, batched argmax, and host readback. Neither enters
the new scan. That observation does **not** override a failed performance
gate or prove the cause of the timing variation. The reference A/A workload
was one stochastic case; its success cannot certify the much shorter greedy
calls, whose control medians here range from roughly 37 to 77 microseconds.

An independent audit recomputes every A/A and candidate confidence interval,
checks saved token hashes, reported RNG matches, and affinity restorations, and reaches
the same result. No pair is discarded and no condition is selectively retried.
The numerical fix stays in the scalar sampler, while **batched dispatch is
not promoted**. These results motivated the separately approved greedy
measurement trial below; the full candidate and downstream gates still
need to pass. No additional sampling-kernel optimization is justified by
these greedy-only failures alone. GPU 0 was released after final regression;
GPU 1 and other users' jobs were left untouched.

### Qualifying the short greedy measurement separately

The approved next trial is **reference-only**, frozen in
[`sampling_timing_greedy_256.json`](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/data/sampling_timing_greedy_256.json).
It increases a greedy sample from 16 to **256 calls** to test whether a longer
measurement satisfies the existing stability criterion. It does not change
the sampler, assume the cause of the previous failures, or qualify a candidate.

| Part of the trial | Frozen choice |
|---|---|
| Distinct inputs | FP32/BF16 × monotonic/random/tied logits × batches 1/8/32; vocabulary 151,936 |
| Repetition | Three blocks per input; 54 blocks total, with ten-second gaps within an input |
| Each block | Two unpinned warmup pairs, then ten alternating A/B and B/A measured pairs |
| Each sample | 256 calls to the same scalar callback, including dispatch and token readback |
| Affinity | CPU 2 for only the calling thread during measured pairs; restore after each block |
| Pass condition | Every block's median and entire paired-bootstrap 95% interval must be inside `[0.95, 1/0.95]` |

The bootstrap still uses 10,000 resamples and seed 0. Both A and B name the
same function object. There are no profiler hooks or extra per-call markers.
All pairs, warmups, failures, and token/RNG checks remain in the receipt.
Greedy outputs are constant for these fixed logits, so the receipt stores
one token vector and a call count per sample, after verifying every original
vector. Expanding that encoding reproduces each original token-list hash.

The six batch-1 mixed cases in the candidate fixture have exactly the same
inputs and parameters as the corresponding greedy cases; CPU tests check
that equivalence. This reference trial tests the 18 distinct workloads once.
A later candidate campaign must still retain **all 54 original conditions**.

Only a complete 54/54-block reference pass authorizes 256-call greedy
samples in candidate timing. Stochastic samples would remain at 16 calls;
the original performance thresholds, memory limit, and correctness gates
remain unchanged. A failure blocks that transition: no pooled substitute,
selective retry, or silent sample-count increase. Whole-model timing and
cross-engine calibration remain downstream of the full candidate gate.

### The 256-call greedy reference trial also fails stability

The [complete receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10d-greedy-timing-256-a6000-2026-09-29.json)
retains all 54 blocks from the approved A6000 trial. **42 pass and 12 fail**;
only 10 of the 18 inputs pass all three blocks. Each table entry is passing
blocks out of three, not a candidate speedup or a correctness score.

| Input dtype | Logit panel | Batch 1 | Batch 8 | Batch 32 |
|---|---|---:|---:|---:|
| FP32 | Monotonic | 1/3 | 2/3 | 3/3 |
| FP32 | Random | 3/3 | 3/3 | 2/3 |
| FP32 | Tied | 3/3 | 2/3 | 1/3 |
| BF16 | Monotonic | 3/3 | 3/3 | 1/3 |
| BF16 | Random | 1/3 | 3/3 | 3/3 |
| BF16 | Tied | 3/3 | 3/3 | 2/3 |

Every block median lies inside the allowed band: they range from **0.9914
to 1.0188**. Twelve bootstrap intervals extend outside it. For example, the
FP32 monotonic batch-1 block 2 has median **1.0169** but interval
**[0.9618, 1.3422]**. A median near one cannot replace the predeclared interval
criterion, even when the two callbacks are identical.

All 648 pairs (including warmups) match tokens and RNG checks. The independent
audit reconstructs the hashes covering **331,776 calls and 4,534,272 token
IDs**, recomputes every interval, and verifies all 54 affinity restorations.
A separate CPU replay reconstructs all 18 input tensors and their argmax
outputs. Global CPU/CUDA RNG states are unchanged; the greedy path owns no
private generators. The 64 focused CPU harness tests pass. No foreign GPU 0
process was observed, and GPU 0 was released after the run.

Increasing the sample length alone is insufficient for this frozen test on
this host. This trial does **not** identify the source of variation, establish
a serving regression, or explain the historical candidate failures. CPU
pinning is still not CPU isolation. No sample is removed or selectively
retried, and no threshold is relaxed.

The original candidate timing stays at 16 calls and its retained result
stays **48/54**; it is not rerun after this failed prerequisite. Batched
dispatch remains disabled. Further qualification needs an explicitly
approved measurement/environment plan, not an automatic increase in sample
count. Whole-model, standard-suite, and cross-engine qualification remain
pending; M10d is **not complete**.

### Separating elapsed time from calling-thread CPU time

The approved diagnostic replays the same 18 greedy workloads and
54 blocks once, using `check_greedy_sampling_timing.py --thread-cpu` and the
separate [counter recipe](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/data/sampling_timing_greedy_cpu.json).
Two CPU-clock reads enclose each unchanged wall-clock sample. There is no
profiler, per-call marker, sample-count increase, CPU reassignment, or serving
change. Warmups remain unpinned; measured pairs keep the CPU-2 policy.

For each sample of 256 calls, retain elapsed time `W`, calling-thread CPU
time `C`, and the signed difference `W - C`. Do not clamp negative differences:
the CPU interval includes the wall-clock reads, so the timer boundaries are
slightly different. CPU time includes driver spinning; the difference is
neither pure scheduling delay nor GPU execution time.

Analysis keeps every raw sample and warmup. It reports measured samples per
workload, the slowest sample with its pair mate, and A-first/B-first strata
separately. A wall-time spike without a corresponding CPU-time increase
suggests waiting or descheduling. A spike in both can reflect CPU work or
driver spinning. Neither pattern identifies a specific cause; separate runs
cannot explain a historical failure or measure instrumentation overhead.

The existing wall-time intervals are still computed against the unchanged
band, but **this instrumented diagnostic cannot qualify the timing protocol
or promote M10d**, even if all intervals pass. An inconclusive result stays
inconclusive. There are no automatic retries or downstream calibration runs.

The [completed diagnostic receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10d-greedy-thread-cpu-a6000-2026-09-30.json)
records the single run on September 30 UTC. All 54 blocks completed;
**42 satisfy the wall-time band and 12 do not**. This happens to equal the
earlier uninstrumented count, but the failing blocks differ. It is not a
replacement qualification result or a controlled measurement of timer
overhead. The 83 focused CPU tests pass; token/RNG checks, all 54 affinity
restorations, source hashes, and a CPU reconstruction of all 18 inputs pass.
No foreign GPU process was observed, and GPU 0 was released.

The useful new observation is where the extra elapsed time goes. For each
workload, compare its slowest measured sample with the other sample in its
pair. The increase in calling-thread CPU time is **95.4–104.7%** of the
increase in elapsed time across these 18 comparisons. Values above 100%
are possible because the paired samples can have different signed gaps;
this is a comparison of differences, not a utilization percentage.

For example, the largest recorded sample is FP32, monotonic logits, batch 1,
block 1, pair 5, label B (zero-based indices). Both samples below contain
**256 calls**, not one unusually slow call:

| Sample in that pair | Elapsed time | Thread CPU time | Signed difference |
|---|---:|---:|---:|
| B, slowest recorded sample | 194.574 ms | 194.013 ms | 0.561 ms |
| A, its paired mate | 6.728 ms | 6.731 ms | -0.003 ms |

The large increase is mostly charged to the calling thread. Descheduling
alone does not explain this sample, but the counters cannot distinguish
Python work, CPU frequency effects, and CUDA-driver spinning. Nor does this
describe every delay: another sample has a **9.255 ms** elapsed-minus-CPU gap.
All signed gaps, including 906 small negative gaps among 1,080 measured
samples, are retained rather than clamped or filtered.

A second observation comes from the predefined slowest-sample analysis:
**16 of the 18 workload maxima occur in the first measured A sample**,
immediately after the calling thread is pinned to CPU 2. The warmups happen
before that affinity change. Exploratory inspection of all 54 blocks finds
the first measured A slower than the median of its later A samples in 53
blocks, with a median ratio of **1.81**. This position analysis is not a gate
and removes no sample; A-first/B-first summaries remain separate in the
receipt.

That pattern motivates a controlled study of warmup placement relative to
pinning; it does **not** prove that pinning caused the variation. The largest
spike above occurs later, so a transition effect cannot explain every slow
sample. No warmup policy, threshold, sampler, or serving path changed in
this diagnostic. Candidate qualification remains **48/54**, batched serving
stays disabled, and downstream calibration is still pending.

### Controlled comparison: warmups after pinning

The approved experiment changes one preparation step in a fresh,
reference-only comparison. Its [frozen recipe](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/data/sampling_timing_greedy_warmup.json)
keeps the preceding CPU counters and all 18 inputs, and compares:

| Arm | Unpinned preparation | After pinning the calling thread to CPU 2 | Measurement |
|---|---|---|---|
| Control | Two warmup pairs | No extra warmup | Ten A/B pairs |
| Post-pin | Two warmup pairs | Two extra warmup pairs | Ten A/B pairs |

Every sample still contains 256 calls. Initial unpinned warmups let runtime
helpers initialize without inheriting the caller's single-CPU mask. The
extra warmups are recorded, not silently discarded or relabeled measurements.
All historical samples and their failed gates remain unchanged.

For each workload, run three neighboring control/post-pin block pairs,
alternating order by round and by workload index. Across 18 workloads each
arm goes first in 27 of 54 neighboring pairs. This gives 108 blocks total,
with the original ten-second gaps between blocks within a workload, three
blocks per arm, and restoration after every block.

Recompute each block's original wall-time interval separately. Also compare
first measured A / median later A for wall time and CPU time within each
block, then compare those normalized ratios between neighboring arm blocks.
Keep every pair and report workload and arm-order summaries; a pooled result
cannot replace a failed block. This tests the complete extra-warmup procedure,
not a particular CPU-frequency or driver mechanism. It is diagnostic only:
no automatic protocol qualification, candidate rerun, or serving promotion.

The [completed comparison](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10d-greedy-warmup-a6000-2026-09-30.json)
supports adding warmups after pinning as a useful measurement change, but
**does not establish a fully stable protocol**:

| Diagnostic metric | Control | Two post-pin warmup pairs |
|---|---:|---:|
| Blocks satisfying the unchanged wall-time band | 38/54 | 53/54 |
| Median first measured A / median later A, wall time | 2.016 | 1.021 |
| Median first measured A / median later A, thread CPU time | 2.009 | 1.018 |

The last two rows summarize 54 per-block ratios in each arm. A value near
one means the first measured A sample resembles that block's later A samples;
it is **not a sampler or serving speedup**. All 54 neighboring arm comparisons
show a smaller normalized wall-time ratio with the extra warmups. The median
control/post-pin normalized ratio is 1.929 when control goes first and 1.972
when post-pin goes first. The effect is visible in both order strata, but
this tests the whole added-warmup procedure, not its CPU or driver mechanism.

One post-pin block still fails: **FP32, tied logits, batch 1, block 2**
(zero-based), with median A/B **0.9324** and paired-bootstrap interval
**[0.8768, 1.0368]**, outside the unchanged `[0.95, 1/0.95]` requirement.
Its first measured A takes about 30.33 microseconds per call; several later
pairs also differ, with B around 34–38 microseconds while A is around 30–33.
These are averages over 256 calls, and CPU time largely follows wall time.
The failure is not explained merely by removing a slow first measurement.
Large later spikes also remain elsewhere in the saved samples.

All **1,404 pairs**, including both kinds of warmup, are retained. The audit
recomputes every block interval, reconstructs token hashes covering
**718,848 calls and 9,824,256 token IDs**, verifies unchanged global RNG
states and all **108 affinity restorations**, and checks the frozen sources.
CPU reconstruction matches all 18 inputs and outputs; **100 focused CPU
tests pass**. No foreign GPU process was observed, and GPU 0 was released.

No historical sample was dropped or reclassified. The remaining failure
stays in the record; no extra warmup, retry, tolerance change, or serving
promotion follows automatically. M10d's candidate result remains **48/54**,
the uninstrumented reference result remains **42/54**, and whole-model and
cross-engine qualification are still pending. This experiment reduces a
measurement-boundary effect; it does not complete M10d.

### A bounded trace of the remaining greedy case

The remaining FP32/tied/batch-1 failure motivates one targeted diagnostic,
not another matrix or a larger warmup count. The frozen
[`sampling_trace_greedy.json`](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/data/sampling_trace_greedy.json)
recipe uses the same 151,936 logits and the identical scalar callback for
both labels. [`profile_greedy_sampling.py`](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/examples/profile_greedy_sampling.py)
runs one unprofiled control block with ten measured alternating pairs,
then, after a ten-second gap, one separately profiled block with three.
Each sample still contains 256 calls and host token-ID readback.

Both blocks retain two unpinned warmup pairs, pin only the calling thread
to CPU 2, and retain two additional pinned warmup pairs. CPU affinity is
restored even on interruption. The trace starts its profiler before the
unpinned warmups so new helper threads do not inherit the pin. This is CPU
eligibility, not exclusive ownership of that core or its SMT sibling.

Sample-level thread CPU time and voluntary/involuntary context-switch
counters sit outside the wall-clock region. The control has no call
markers or profiler; the trace records CPU/CUDA activities and sample/call
annotations, including its warmups. All outputs and global RNG states are
checked. Trace timings include instrumentation and must not be compared
with the control as a speedup. CUDA activity coverage is a clipped union,
not a sum of overlapping kernels or evidence that uncovered time is an OS
scheduling delay.

The control's original-band interval is descriptive for this one case,
not a replacement for the failed full-matrix prerequisite. A failure to
reproduce is inconclusive; even a reproduced slow sample cannot establish
the cause of the historical failure. There is no automatic retry, serving
change, candidate qualification, or downstream calibration.

The [completed capture](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10d-greedy-reference-trace-a6000-2026-09-30.json)
does **not reproduce the historical later-pair failure**. The unprofiled
control has median A/B **1.0010**, with a paired-bootstrap interval of
**[0.9946, 1.0161]**, inside the original band. Its 20 measured samples span
25.48–28.16 microseconds per call, with a median of 25.71. That is one case
in a new process, not evidence that all 54 reference blocks are now stable.

The six measured trace samples each contain 256 argmax kernels, 256 device
memsets, and 256 host readback copies. CPU operators include `aten::argmax`
and the nested CPU-transfer operations; their inclusive durations must not
be added as independent costs. The instrumented median is 76.48
microseconds per call, so it cannot substitute for unprofiled timing.
No voluntary or involuntary calling-thread context switches were counted
in those six measured trace samples. This describes the new trace, not
the historical failed block.

The first pinned trace warmup is still slow—841.84 microseconds per call
on average—and remains in the receipt as the warmup declared before the
run. Its largest gap without traced GPU work is about 176.55 milliseconds.
Neither that gap nor the absence of measured-sample context switches
identifies CPU frequency, Python work, driver spin, or profiler overhead
as the cause of the earlier failure.

All **42 samples / 10,752 calls** retain exact token outputs and hashes;
global RNG is unchanged, both affinity masks are restored, and **123
focused CPU tests pass**. The raw CPU/CUDA timeline, per-call analysis, and
ownership log are retained locally with hashes in the receipt. GPU 0 had
no foreign process and was released; GPU 1 was untouched.

This bounded investigation ends **inconclusive about the historical
instability**. There is no further retry or candidate run. The 48/54
candidate result and failed full-reference prerequisite remain unchanged;
the batched helper stays disabled. M10a–c's completed online-serving
capability is distinct from this unfinished M10d optimization.
