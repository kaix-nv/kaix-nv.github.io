---
layout: post
math: true
title: "Building tinyserve M12: Different requests, one model forward"
date: 2026-10-05 08:32:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Pack prompt chunks and decode tokens into one forward while keeping histories private and performance claims bounded."
source_revision: e20a34815498560f9226a9e057ade31b2b62743f
source_document: docs/m12-mixed-prefill-decode.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `e20a348`](https://github.com/kaix-nv/tinyserve/tree/e20a34815498560f9226a9e057ade31b2b62743f). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M11 — Move the KV, not the prompt work]({% include tinyserve-post-url.html slug="building-tinyserve-m11" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m11-prefill-decode.md" %})

**Status: complete for the approved opt-in slice. Qwen3-0.6B and Qwen3-8B pass the adopted
numerical and quality contract on RTX A6000 GPU 1 (2026-10-02). BF16 still
fails the original old-layout compatibility gate. Post-reset paired,
Qwen3-8B standard and reference timing runs are complete (2026-10-05).
FreeToken fails the matched-output-length check; its timings are excluded
from the accepted comparison. The user approved closing M12 with that
failed reference retained (2026-10-05); no failed gate is relabeled.** The
first slice is opt-in, eager, single-device, greedy, ordinary dense Qwen
with unquantized weights and KV. This is a new execution layout, not a
new scheduler policy or a claim of a serving speedup.

## 1. The gap after continuous batching and chunking

M4 decides which requests run in an iteration. M5a limits the prompt tokens
processed before the next decode opportunity. Those features already let
decoders progress while long prompts enter the cache incrementally. But
the ordinary session still executes the two kinds of work separately:

1. One model forward for the requests that are decoding.
2. One or more model forwards for whole prompts or prompt chunks.

Both forwards execute the same transformer weights: embeddings, Q/K/V
projections, attention, output projections, and MLPs. Their main difference
is how many new tokens each request contributes and what history those
tokens may attend to. M12 represents both as **append spans** and lets
the token-wise operations share one packed forward. Attention must still
keep the requests separate.

This is not M11 disaggregation. M11 uses two complete model replicas on
two GPUs; M12 uses one model on one device. It is also not simultaneous
prefill and decode kernels: one packed forward carries both workloads.

## 2. Ten query rows, three independent histories

Suppose A has five cached tokens and one uncached generated token. B has
three cached tokens and one uncached generated token. C is processing the
first eight tokens of a twelve-token prompt. With a prompt budget of eight:

| Request | New query tokens | Absolute positions | Visible KV after append | Sample now? |
|---|---:|---|---:|---|
| A, decoding | 1 | 5 | 6 | Yes |
| B, decoding | 1 | 3 | 4 | Yes |
| C, prefilling | 8 | 0 through 7 | 8 | No: four prompt tokens remain |

The packed token row has ten entries, not three padded rows of length eight.
Its query boundaries are `[0, 1, 2, 10]`. Positions are
`[5, 3, 0, 1, 2, 3, 4, 5, 6, 7]`, not a global `arange(10)`.
Only packed rows `[0, 1]` need the vocabulary projection in this iteration.
Rows for C compute hidden states and KV, but cannot produce an output token
until the last prompt chunk is complete.

[![Separate forwards versus one packed forward, with private attention histories and selective output rows](/assets/tinyserve/m12-mixed-prefill-decode.svg)](/assets/tinyserve/m12-mixed-prefill-decode.svg)

On the next iteration C contributes positions 8 through 11. Its final
query row now predicts its first output token. A and B each still contribute
only their newest, uncached token. As before M12, a sampled token enters
KV on the next iteration, not when the sampler emits it.

## 3. A common operation: append, then attend

An append span is `(sequence, start, end)`, where `start == num_cached` and
`[start, end)` contains tokens already present in that sequence's history.
Decode is simply a span of length one. The execution path builds:

- Token IDs and absolute positions in packed span order.
- Physical write slots from each sequence's own block table.
- Query boundaries, visible KV lengths, and visible physical page tables.
- Output indices for decoders and completed prompt chunks only.

With page size four, A's logical position five is page-table entry one,
offset one. If A's physical table is `[9, 2]`, that token writes slot
`2 * 4 + 1 = 9`. B's token can have a different logical position, table,
and physical slot; packing must not turn any of these into a shared history.
Reserved pages beyond the span's end are not visible attention history.

For request `r`, local query `i` may read key `j` exactly when:

$$
0 \le j \le \mathrm{start}_r + i.
$$
This is a prefix-offset causal mask. A decoder's one query sees its entire
committed prefix plus itself. A prompt query cannot see later prompt tokens
even though the layer writes every span's new KV before reading attention.
No query can read another request's private pages. Shared immutable prefix
pages remain valid and do not require copying.

The readable Torch backend gathers and slices each span's visible KV, then
runs its own shifted causal attention. It batches the projections and MLP,
but does **not** promise a single attention kernel. The FlashInfer backend
plans one paged append with ragged query lengths and bottom-right causal
masks, reusing the paged-append wrapper introduced in M9c. Neither path
changes model weights, RoPE, or the sampling algorithm.

## 4. Scheduling and lifetime stay explicit

The new session reuses M10's request registry, cancellation, token events,
and terminal cleanup. Its step performs:

1. Retire cancellations and zero-output requests.
2. Admit FCFS requests and reserve decoder growth, including existing LIFO
   preemption. Only then select spans from the remaining resident requests.
3. Include every surviving decoder once, then spend at most `chunk_size`
   prompt tokens across the FCFS prefill prefix. Never skip a large head
   to select a convenient shorter request.
4. Execute one packed forward when prompt work is present. Decode-only
   steps retain the existing eager decode helper.
5. Advance cache cursors and publish full blocks only after the packed
   forward succeeds. Sample only completed spans, with decode events first.
6. Synchronize before returning, even if every span was a partial prompt
   and no token-ID readback occurred.

Failed writes cannot become reusable warm prefixes. Cancellation and errors
must release every request-owned page exactly once. Already-computed shared
prefixes remain immutable. The ordinary M10 session stays available as the
reference; M10d's experimental batched sampler is not enabled.

Combining work also changes when decoder logits become available: a decode
token cannot be sampled until the mixed forward finishes. Fewer forwards
can improve throughput while first-text or inter-token latency worsens.
The prompt budget still bounds interference; it does not guarantee a latency
win. Event delivery remains at the end of a synchronous step.

## 5. First-slice limits and qualification

The option is `mixed_batching=True` on `LLM`, exposed as
`--mixed-batching` in the existing `generate.py` live/offline/HTTP examples.
CUDA graphs, quantized weights/KV, distributed and hybrid models, speculation,
and stochastic requests are outside this slice and must fail explicitly.
Prefix reuse remains supported. The option is disabled by default.

Validation proceeds from cheap invariants to measured serving:

- CPU: packed metadata, offset masks, poisoned page tails, mixed outputs
  against separate forwards, partial-chunk output selection, FCFS budget,
  preemption, prefix reuse, cancellation, EOS, zero/one-token limits, errors,
  and ownership after close. Verify one model forward actually covers a
  mixed iteration; token agreement alone does not prove packing happened.
- GPU: real Qwen3-0.6B, FP32 gather and BF16 gather/FlashInfer, fixed histories
  before generated continuations, exact page/position accounting, and live
  lifecycle checks. Freeze numerical tolerances before the first run.
  BF16 batch regrouping is not assumed to preserve every greedy near-tie.
- Performance: unprofiled separate/mixed whole-model pairs on fixed workloads,
  the same backend, eager execution, model, cache policy, token budgets and
  input histories. Retain all samples and actual output lengths. Report
  throughput, TTFT/ITL and forward counts separately; traces are diagnostic.
- Run the Qwen3-8B standard suite and compare the retained same-host
  cross-engine baseline. Refresh reference engines when revisions, runtime,
  model artifacts or hardware conditions differ. This is calibration, not
  proof that packing caused differences between engines.

No performance result is filled in before these checks run. A measured
regression stays in the report, and this eager teaching path does not
replace the graph-enabled default merely because it uses fewer forwards.

## 6. First implementation and the failed BF16 gate

[`mixed.py`](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/mixed.py) owns `AppendSpan`, `AppendAttention`,
`append_forward`, and `MixedServingSession`. The latter inherits terminal
ownership operations from M10 but implements its own packed step. The
ordinary session and sampler source files are unchanged. `LLM` selects the
new session only when the option is true. The existing HTTP worker and
offline driver use the same selection point.

The first dedicated GPU run uses Qwen3-0.6B on physical GPU 0, an RTX A6000.
The [prototype evidence receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m12-mixed-prefill-decode-prototype-a6000-2026-09-30.json)
retains the failed checks, diagnostic, source/model hashes, and GPU-ownership
logs separately from the existing-engine regression suite.
The tolerances were frozen before execution: FP32 `atol=rtol=1e-4`, BF16
`atol=rtol=3e-2`. The BF16 elementwise gate follows the existing small,
randomly initialized paged-append model test; it was not selected from
these results. That test does not establish how much rounding drift a
trained 28-layer model should accumulate when its GEMM shapes and attention
kernels change.

| Path | Two matched-history logit/KV checks | Live continuation and cancellation |
|---|---|---|
| FP32 / gather | Both pass | Pass; exact reference token IDs |
| BF16 / gather | Both fail at logits | Pass; token IDs match for this fixture |
| BF16 / FlashInfer append | Both fail at logits | Pass; token IDs match for this fixture |

The total is **5 passed, 4 failed**. The failed logit comparisons prevent
their later KV assertions from running: they are not BF16 KV-parity passes.
Elementwise logit failures cover 15.7–19.2% of the compared entries, with
maximum differences up to 0.5. Matching two short generated continuations
does not override those failures.

A separate, instrumented BF16/gather trace follows the first layout's A
query through the layers. Its first normalized projection input is exactly
equal between the single-row and ten-row forwards, but the first Q/K/V
projection outputs already differ, before attention reads the cache.
Repeating the first Q projection with identical input values at one versus
ten rows reproduces that projection's rounding difference. In the traced
final A logits the maximum difference is 0.21875 and relative RMS error
is about 1.645%. This identifies an early shape-dependent numerical
difference; it is not a proof that every downstream discrepancy has only
that cause, nor a quality assessment.

At that stage no tolerance, model computation, or default path was changed
to make the failed checks pass, and the unprofiled paired harness had not
run. The follow-up investigation below separates the numerical causes;
neither a performance claim nor milestone completion follows from it.

At the prototype revision, the full single-GPU regression passed **819 tests**, with **22 skipped**.
Those skips include the nine opt-in M12 checks whose separate 5-pass/4-fail
result is shown above. The focused CPU suite passed 193 tests with nine
CUDA-only skips; the later HTTP-focused suite passed 77 tests. An attempted
full CPU-only run retained 27 failures/errors from existing tests that
require CUDA; all of those passed in the subsequent single-GPU run. The
FP32 debugger configuration completed its three requests. No foreign GPU
process was interrupted, and no serving performance measurement followed
the failed numerical gate.

### Following the BF16 discrepancy

The [BF16 investigation receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m12-bf16-compatibility-investigation-a6000-2026-09-30.json)
records four isolated experiments on the same GPU and model. These are
diagnostics, not a replacement qualification run. Serving source and the
frozen `atol=rtol=3e-2` gate remain unchanged.

First, hold one layer's input values and weights fixed and repeat the input
row ten times. The first K projection changes from a one-row cuBLAS GEMV
to a tensor-core GEMM followed by a BF16 split-K reduction. Here *split-K*
means splitting a matrix dot product's reduction dimension into partial
sums, not splitting the attention key tensor or the KV cache. Rounding
those partial sums can change the result even though each row computes
the same mathematical dot products.

For this captured K projection, the single-row output exactly matches an
FP64 CPU dot product rounded once to BF16. The ten-row output differs in
406 of 1,024 entries, by at most 0.0078125. Temporarily disabling reduced-
precision reduction removes those differences. However, it does **not**
make the full model pass: other shape-dependent rounding differences
remain and propagate through later layers. The diagnostic restores the
backend setting; no process-wide arithmetic policy was added to serving.

Second, preserve each sequence's original linear row group *inside* the
packed forward: project A's one row, B's one row, and C's prompt rows
separately, then concatenate the results. Keep the packed positions,
page writes, normalization and attention unchanged. Also project the two
emitting output rows separately. This deliberately removes cross-sequence
GEMM packing, allowing us to isolate its numerical effect.

| Diagnostic | Gather, both layouts | FlashInfer, both layouts |
|---|---|---|
| Original packed execution | Logit/KV gate fails | Logit/KV gate fails |
| Disable reduced-precision reduction in both paths | Still fails | Still fails |
| Preserve reference linear row groups | Logits and all valid KV are bitwise identical | Still fails |

The last row isolates projection regrouping as the source of the gather
discrepancy **for these fixtures**, rather than incorrect page maps or
causal masks. It is not a performance fix: splitting the linears removes
an important part of the work-sharing that M12 is meant to teach.

With FlashInfer, preserving linear row groups still leaves a difference
at the first attention output projection, after matching Q/K/V projections
and Q/K normalization. The gather result cannot qualify FlashInfer.

A further attention-only diagnostic reconstructs each span's logical KV
directly from its physical pages and computes explicit FP32 CPU dot
products, causal softmax and weighted values. It uses the existing
attention-primitive tolerance, `atol=3e-3, rtol=3e-2`, declared before the
run. Across 168 layer/span checks per backend, gather passes 151 and
FlashInfer passes 157. There are respectively 42 and 19 failing entries
out of 1,777,664 compared elements per backend; all values are finite.
These runs use each backend's own evolving activations, so their failure
counts are not a head-to-head accuracy ranking. The small aggregate errors
do not override the failed elementwise checks or establish model quality.

On 2026-10-02 the [revised validation contract](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/design/m12-validation.md)
was explicitly approved: a shape-matched independent reference for packing,
direct attention checks against an FP32 oracle, and separately specified
whole-model BF16 quality gates. It distinguishes packing correctness from
compatibility with an older execution layout. The original gate remains
failed and reported; adopting new criteria does not retroactively pass it.

The contract fixes the references, fixture grid, numerical and quality
budgets, and the boundary before performance testing. Approval itself did
not qualify BF16; the subsequent execution is described below.

## 7. Qualifying the packed computation

There are three different questions here: did we pack the right rows and
histories, is the resulting BF16 computation sufficiently accurate, and
does it reproduce the old execution layout? One comparison cannot answer
all three. The [qualification receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m12-qualification-a6000-2026-10-02.json)
records them separately for both models and attention backends.

For the first question, the test runs each request independently while
matching the candidate's row geometry. For example, if A and B each supply
one decode row and C supplies seven prompt rows:

```text
Candidate:    [ A   B   C0  C1  C2  C3  C4  C5  C6 ]
Reference A:  [ A   0   0   0   0   0   0   0   0  ]
Reference C:  [ 0   0   C0  C1  C2  C3  C4  C5  C6 ]
              Same nine-row linear and RMSNorm geometry
              Each reference keeps only its own output rows
```

Those zero rows exist only in the test reference. Production execution
still packs real tokens. Each reference request constructs its own
positions, page addresses and causal history without the packing helpers;
it never copies hidden states or outputs from the candidate. The output
head is matched separately because incomplete prompts have no sampling row.

The first reference implementation matched linears but missed RMSNorm.
It passed only 10 of 16 BF16 layouts. A retained trace found identical inputs
at layer 12's input RMSNorm but one output element differing by
`0.000030517578125` at seven versus nine rows. Matching RMSNorm geometry
as well corrected the reference. Serving code, fixtures and acceptance
budgets were unchanged; the failed attempt remains in the receipt.

| Adopted check | Qwen3-0.6B | Qwen3-8B |
|---|---|---|
| FP32 packing, original tolerance | 16/16 layouts pass | 16/16 layouts pass |
| BF16/gather versus shape-matched reference | 16/16 pass; logits and valid KV bitwise equal | 16/16 pass; logits and valid KV bitwise equal |
| Gather attention versus explicit FP32 oracle | 1,232/1,232 layer/span checks pass | 1,584/1,584 pass |
| FlashInfer attention on the same operands | 1,232/1,232 pass | 1,584/1,584 pass |
| Isolation, page permutation and lifecycle | Both backends pass | Both backends pass |
| Original BF16 old-layout compatibility | Both backends fail | Both backends fail |

The attention oracle uses independent logical KV reconstruction and an
explicit causal softmax. Its adopted bounds check both vector-relative
error and error relative to the contributions before cancellation. The
old elementwise attention test remains reported separately and still has
failures. The new result does not retroactively pass that test.

Quality uses fixed next-token targets so early prediction differences
cannot change later inputs. Each model/backend scores all **768 predictions**
across 24 sequences, with contexts 128, 512 and 2,048. Each primary request
has at least eight scored decode steps inside a real mixed forward while
a 4,096-token background prompt is prefilling. The ordinary BF16 control
uses the same backend selection; ordinary FP32 supplies the KL reference.

| Model / mixed backend | Worst sequence NLL increase versus ordinary BF16 | Worst sequence mean KL from FP32 |
|---|---:|---:|
| 0.6B / gather | 0.000156 | 0.000111 |
| 0.6B / FlashInfer | 0.001461 | 0.0000684 |
| 8B / gather | 0.000329 | 0.00000589 |
| 8B / FlashInfer | 0.000133 | 0.00000611 |
| Frozen acceptance limit | 0.02 nats/token | 0.01 nats |

All sequence and overall NLL budgets pass. Each ordinary BF16 control and
mixed candidate also answers **12/12** known-answer cases correctly: neither
backend adds a failure. These are small synthetic regression fixtures, not
a broad accuracy certification. Both model sizes and backends qualify under
this adopted contract, not under the original old-layout tolerance.

The focused CPU suite passes **85 tests**. The subsequent single-GPU
regression passes **847 tests**, with **22 skipped**. Nine skips are the
unchanged opt-in historical M12 tests, not newly passed comparisons; the
dedicated runs above retain their old-layout failures alongside the adopted
gates. A regression attempt was stopped after discovering the system disk
was full; the complete rerun used scratch temporary storage. Both receipts
remain, and no foreign GPU process or files were removed.

Instrumented qualification does not establish a speedup or justify changing
defaults. The following measurements run separately, without the numerical
oracle or teacher-forcing instrumentation.

## 8. When does sharing a forward help?

The paired results below are the **October 5 post-reset run**, retained
with the [standard suites, reference results and artifact checks](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m12-post-reset-performance-a6000-2026-10-05.json). The user
confirmed GPU 1's graphics and memory clock locks were reset before this
fresh run. The [initial October 4 measurements](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m12-initial-performance-a6000-2026-10-04.json)
remain separate, including their raw-artifact hashes and incomplete
reference attempts. Resetting clock locks is not the same as benchmarking
at a controlled fixed frequency.

The 2026-10-05 paired run uses Qwen3-0.6B in BF16 on physical RTX A6000
GPU 1. Both arms are eager, use cold private KV, a 32,768-token pool and a
512-token prompt budget. Each workload has two warmup pairs followed by six
measured pairs, alternating separate-first and mixed-first. The ratio below
is the median of the paired **separate time / mixed time** ratios; greater
than one means mixed execution is faster. Brackets give the bootstrap 95%
interval, not a guarantee for other workloads or machines.

| Prompt / output / requests | Gather ratio [95% interval] | FlashInfer ratio [95% interval] | Separate → mixed forwards | Mixed forwards |
|---|---:|---:|---:|---:|
| 128 / 128 / 1 | 1.003 [0.971, 1.029] | 1.000 [0.964, 1.060] | 128 → 128 | 0 |
| 128 / 128 / 8 | 1.029 [0.985, 1.059] | 1.019 [1.000, 1.041] | 130 → 129 | 1 |
| 2,048 / 32 / 1 | 1.010 [0.995, 1.019] | 1.032 [1.000, 1.116] | 35 → 35 | 0 |
| 2,048 / 32 / 8 | **1.155 [1.121, 1.192]** | **1.537 [1.449, 1.583]** | **91 → 63** | **28** |

All measured pairs have identical token IDs and output counts. Both
single-request cases intentionally have no simultaneous prefill/decode
work. With eight short prompts, only one iteration can share a forward;
the remaining work is unchanged. These cases cluster much closer to parity.
The two FlashInfer intervals rounded to a lower bound of `1.000` are only
marginally above one before rounding (`1.000146` and `1.000053`). A
single-request case cannot demonstrate a gain from mixing different
requests; its layout and attention path can still differ.

The long-prompt, eight-request case exposes the mechanism. Earlier requests
are decoding while later prompts still need chunks. Packing removes 28
separate model calls, sharing the projections and MLPs across their rows.
On gather, median output throughput rises from **83.5 to 96.8 tokens/s**;
median request TTFT falls from **970 to 662 ms**, and median ITL from
**48.1 to 44.1 ms**. These latency summaries describe this fixed arrival
pattern, not an online tail-latency guarantee.

FlashInfer reaches **101.9 → 152.3 tokens/s**, with TTFT **919 → 465 ms**
and ITL **27.4 → 25.7 ms** for that case. Its 1.54× result is a **whole-path
comparison**, not a pure packing gain: ordinary execution still gathers KV
for partial prompt chunks, whereas mixed execution uses paged append
attention. Gather holds the attention algorithm fixed and more directly
isolates sharing the token-wise model work. Forward counts explain where
work was combined, but do not imply that every forward costs the same.

GPU ownership was monitored throughout; neither run encountered another
GPU 1 job. No fixed clock or power limit was imposed by the benchmark.
Telemetry still sometimes reported low clocks during active work after
the user-confirmed resets; the raw samples are retained. The counterbalanced
pairs reduce ordering effects, but do not turn this host into a
controlled-clock performance laboratory. We do not infer a causal benefit
from the clock resets themselves.

## 9. Qwen3-8B calibration is a different experiment

Both tables below use the October 5 post-reset calibration. All timing
runs finished, and their servers have stopped. FreeToken's suite ran but
failed the matched-output-length check; its measurements are retained as
diagnostics, not included as a valid comparison.

The standard suite uses prompts/outputs of 128/128 and 2,048/32 tokens,
each at one, eight and 32 simultaneous requests. Every condition has two
warmups and five measured repetitions. Tinyserve uses a 131,072-token KV
pool, cold prefixes and an **8,192-token prompt budget**, not the 512-token
budget in the paired test. We retain the graph-enabled default, both eager
ordinary controls, and both eager mixed candidates.

That larger budget changes how much work can overlap. All 32 short prompts
fit in the first prompt budget (`32 * 128 = 4,096`); there is no remaining
prefill to mix with their decode work. Eight long prompts take two budgets
instead of the 32 budgets needed by the small-model paired workload.
Packing therefore has fewer opportunities to remove a separate forward.
Model size, attention backend and prompt budget all matter; a result from
one row cannot be transplanted to another.

These suites run sequentially, not as randomized separate/mixed pairs.
Their medians describe this calibration window; small differences do not
establish a causal M12 speedup. The paired table above is the controlled
separate/mixed experiment. Neither table justifies enabling mixed execution
by default.

Median output tokens/s, including prompt processing time; each column is
**prompt / output / requests**:

| Tinyserve path | 128/128/1 | 128/128/8 | 128/128/32 | 2048/32/1 | 2048/32/8 | 2048/32/32 |
|---|---:|---:|---:|---:|---:|---:|
| Default, graphs + FlashInfer | 38.1 | 284.3 | 912.0 | 26.7 | 68.3 | 80.8 |
| Separate eager, FlashInfer | 28.9 | 221.3 | 780.2 | 22.3 | 65.3 | 79.2 |
| Mixed eager, FlashInfer | 26.9 | 213.0 | 752.3 | 21.3 | 64.8 | 80.0 |
| Separate eager, gather | 25.7 | 195.4 | 469.1 | 18.8 | 40.0 | 45.6 |
| Mixed eager, gather | 23.9 | 187.1 | 469.2 | 18.4 | 39.7 | 45.2 |

All 150 measured batches produce their full requested output counts. Mixed
execution is not consistently ahead of either eager control here, and the
graph-enabled default leads all six conditions. Latency is not uniformly
better either: for gather at 2,048/32/1, median TTFT rises from **365.1 to
420.0 ms**. This no-mixing control still takes the generic append path for
its prompt. The observation is retained; a causal explanation of that cost
would require a separate profile, not a guess from throughput alone.

Some eager samples also vary within a condition. For example, mixed gather
at 128/128/8 ranges from **169.9 to 193.4 tokens/s**, with a median of 187.1.
The receipt retains every repetition. We do not select the fastest sample
or interpret small differences between these sequential runs as a reliable
optimization gain.

External references use the same raw-prompt generator, greedy sampling,
request counts and output limits. Tinyserve times its engine call; external
engines include streaming HTTP client overhead. External TTFT is the first
nonempty text chunk, whereas tinyserve records its token event. HTTP chunk
gaps are not individual GPU decode-step timings. Native kernels, CUDA-graph
policies, batch/prefill limits, KV dtypes and runtime versions also differ.
An HTTP client's `chunk_size` argument does not configure the remote
scheduler; the retained server command defines that engine's limits.

Ollama belongs to the native-settings comparison: its retained revision
does not expose a true prompt-cache-off path. Its BF16-weight GGUF and FP16
KV are not an identical numerical execution to tinyserve's BF16 tensors.
Ollama's import rewrote the GGUF container, changing its file hash, but a
post-run audit confirms all **399 tensor types, shapes and payloads** are
identical to the input; the metadata also matches. Import and startup time
are excluded from the serving measurements. Input and served GGUF hashes,
external source revisions, executable hashes and launch settings are
retained with the measurements. NInfer is
not run: its current build requires `sm_120a`, while the A6000 is `sm_86`.
An unsupported engine is not a zero-throughput result.

vLLM, llama.cpp and Ollama passed the factual/arithmetic startup check and
the benchmark's `2 + 2` smoke check. All 90 measured batches across those
three engines reached the full requested output count for every request.
These **post-reset observations** retain the timing and native-settings
differences above; they are not a matched-kernel ranking. Values are median
output tokens/s, with the same prompt/output/request column notation:

| Reference | 128/128/1 | 128/128/8 | 128/128/32 | 2048/32/1 | 2048/32/8 | 2048/32/32 |
|---|---:|---:|---:|---:|---:|---:|
| vLLM | 41.7 | 311.2 | 1,015.9 | 29.5 | 80.4 | 97.8 |
| llama.cpp | 42.5 | 254.7 | 513.7 | 23.8 | 12.7* | 28.6 |
| Ollama, native cache / FP16 KV | 42.7 | 261.9 | 568.0 | 27.4 | 54.3 | 61.0 |

`*` The five llama.cpp long-prompt, eight-request runs measured **43.03,
12.13, 12.90, 12.65 and 12.65 tokens/s**. This repeats the initial run's
fast-first-sample pattern; resetting the clock locks did not eliminate it.
The cause remains unresolved, so the median must not be treated as a stable
baseline or used to claim a reliable engine ranking for that condition.
All samples are retained. No competing GPU 1 compute process was observed
during any of the four reference suites.

In the retained October 4 attempts, FreeToken failed during startup with an unresolved
`CuteDSLRT_TVMFFISetRaisedCudaError` runtime symbol. Ollama's own calibration
process was stopped during model import when the clock-reset rerun was
approved. Neither initial attempt produced a timing result; those failures
remain recorded separately from the completed October 5 suites.

An [October 5 startup diagnostic](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m12-startup-followup-a6000-2026-10-05.json)
subsequently recovered FreeToken by loading its existing CUDA 13 CuTe
runtime globally inside Python workers only. The server passed graph
capture, health/model checks and the Paris/391 generation check without
package or kernel changes. A broader `LD_PRELOAD` attempt had instead
crashed compiler subprocesses; that failed attempt is retained too. This
startup diagnostic supplies no throughput measurement. The subsequent
post-reset suite did run, but failed the matched-length check: FreeToken
emitted **127 instead of 128**, and **31 instead of 32**, tokens per
request. Its overlap scheduler checks the already-advanced device length
while draining the prior result, then discards the final in-flight result.
A CPU-only trace using the pinned request-state methods reproduces those
counts. These timings are diagnostic, not an accepted matched-workload
comparison. No request limit was increased to compensate, and no reference
source was patched. On October 5, the user explicitly approved closing M12
with FreeToken retained as a failed comparison. Repairing that reference
engine is separate work, not part of this milestone's accepted results.

M12 therefore closes as an educational, opt-in implementation of shared
prefill/decode forwards. The adopted correctness and quality gates pass,
and the timing evidence explains both a useful mixing case and its limits.
Completion does not mean original BF16 compatibility, passing results for
every reference engine, or a reason to replace the graph-enabled default.

## 10. Debugging the packing

The `M12: mixed prefill/decode, eager live requests` debugger entry uses
FP32/gather, the slice that passed the first dedicated correctness run.
In `MixedServingSession.step`, inspect the selected `requests`, `spans`,
and remaining `budget`. In `append_forward`, inspect `positions`,
`ctx.slots`, `attention.indptr`, and `output_rows`. Caching and sampling
should follow the ten-row example above even when physical pages are
non-contiguous. A debugger run demonstrates execution, not throughput.
