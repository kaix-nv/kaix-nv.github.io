---
layout: post
math: true
title: "Building tinyserve M10c: Randomness belongs to the request"
date: 2026-10-05 08:29:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Give each live request its own sampling parameters and randomness, with explicit seed and batching boundaries."
source_revision: e20a34815498560f9226a9e057ade31b2b62743f
source_document: docs/m10c-request-sampling.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `e20a348`](https://github.com/kaix-nv/tinyserve/tree/e20a34815498560f9226a9e057ade31b2b62743f). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M10b — From token events to an HTTP text stream]({% include tinyserve-post-url.html slug="building-tinyserve-m10b" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m10b-http-streaming.md" %}) · Next: [M10d — Batch the filtering, keep the randomness private]({% include tinyserve-post-url.html slug="building-tinyserve-m10d" fallback="https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/m10d-batched-sampling.md" %})

Status: **implemented and GPU-validated within the single-device scope below**.
This slice adds request-local sampling to ordinary, single-device dense-Qwen
live sessions and HTTP streams. It does not qualify the experimental M8h
reader or extend distributed, hybrid, speculative, or disaggregated sampling.

## Why a server needs more than a sampling function

M1 already implements temperature, top-k, and top-p. M10's live session uses
argmax, however, and its HTTP adapter only accepts temperature zero. The
missing work is not another sampling algorithm: it is deciding who owns the
parameters and random-number stream when requests arrive and disappear.

Suppose A and B share a global random generator. A samples, B samples, then
A samples again. Cancelling B removes its draw, so A receives a different
part of that stream. This can change A's output even when its logits are
identical. A process-wide seed reproduces an execution order, not an isolated
request. Resetting that global seed before each request is worse: it changes
the state used by requests already running.

Give each stochastic request its own `torch.Generator` on the model device.
The session owns it under the request ID, alongside the request's lifetime.
The HTTP/event-loop thread passes parameters, never a CUDA generator.

[![Each request owns its sampling parameters and RNG across admission, sampling, preemption, and retirement](/assets/tinyserve/m10c-request-sampling.svg)](/assets/tinyserve/m10c-request-sampling.svg)

## From logits to one token

For one request, the model returns a vector of vocabulary logits. The
existing sampler applies these operations in order:

1. At temperature zero, return argmax without a random draw.
2. Otherwise divide logits by the positive temperature. Lower temperature
   concentrates probability on the leading tokens; higher temperature
   flattens it.
3. With top-k enabled, mask logits below the kth-largest value. Ties at that
   boundary survive, so there can be more than k surviving tokens.
4. With top-p below one, sort by score and keep through the first token whose
   cumulative probability exceeds the threshold. The existing strict `>`
   comparison can keep one extra token at exact threshold equality. Apply
   this after top-k and its renormalization.
5. Softmax over the survivors and sample with this request's generator.

For probabilities `[0.55, 0.25, 0.15, 0.05]` at temperature one, top-k three
removes the last token. The remaining probabilities become approximately
`[0.579, 0.263, 0.158]`. Top-p `0.8` keeps the first two, whose cumulative
mass is about `0.842`; the final sampling probabilities are `[0.6875, 0.3125]`.
These controls change the distribution; they do not guarantee correctness,
creativity, or better model quality.

M10c reuses `sampling.sample()` with an optional explicit generator. Existing
callers without one keep their original behavior. The new path filters in
FP32 and subtracts the maximum logit before dividing by temperature. Extremely
small temperatures are clamped only below the FP32 normal range; tied maxima
retain mass without overflowing to NaN. Sampling stays outside
the captured model-forward CUDA graph. All-greedy batches retain one batched
argmax; mixed batches sample stochastic rows separately and preserve row IDs.
This is a readable implementation, not a fused high-throughput sampler.

## A concrete lifecycle

A arrives with seed 17 and B with seed 29. Each request copies its parameters
and initializes its own generator once. Submitting the same parameter object
twice creates two independent generators; it does not share mutable RNG state.

When A's final prefill chunk produces its first output, only A's generator
advances. An intermediate prompt chunk does not sample. A later decode step
may sample A and B in either row order: each call receives its owner's state.
Cancelling B removes its pages and generator without consuming randomness
from A. Greedy and zero-output requests need no generator.

Preemption discards A's KV, **not** its generated token history or generator.
Readmission recomputes that history without resampling emitted tokens. The
next emitted token consumes the next sampling call from A's existing stream.
EOS, budget exhaustion, cancellation, session close, and failed-forward
cleanup all retire that generator along with request ownership. Cached
prefix blocks contain model state, never sampling parameters or RNG state.

## What a seed promises—and what it cannot

With identical logits, parameters, device/runtime, and seed, a request gets
the same sampling sequence regardless of unrelated random draws, peer
cancellation, or row order. An omitted seed is chosen from operating-system
randomness; `torch.manual_seed()` is not the session's seed API.

This does **not** promise identical text across CPU/CUDA, library versions,
attention backends, hardware, or changed batching/chunking schedules. BF16
regrouping can change logits before sampling, as the M10/M9 investigations
already show. Request-local RNG removes random-stream interference, not
floating-point differences. CPU fixed-logit tests isolate the RNG contract;
real-model tests separately check repeated identical execution schedules.

## API and boundary

`SamplingParams` adds `seed=None`. Temperature must be finite and nonnegative;
top-k must be a nonnegative integer no greater than the model vocabulary;
top-p must be finite in `(0, 1]`; output budget must be a nonnegative integer.
Explicit seeds are integers in `[0, 2**63 - 1]`. Booleans are not numbers for
these API fields. Validate before registering a request. Greedy ignores the
sampling filters and seed after validating their values.

The HTTP completion subset accepts `temperature`, `top_k`, `top_p`, and `seed`
per request, with the same defaults as `SamplingParams`. `top_k` is a Tinyserve
extension; this remains a completion-stream subset, not a full compatible API.
Invalid requests return 400 without stopping other requests. Existing
disconnect cancellation, backpressure, and usage reporting are unchanged.

The live demo and ordinary `serve()` use the same session path. CLI seed
support is limited to those two entry points; HTTP parameters belong in the
request body. Quantized KV and M11 PD remain greedy-only, and unsupported
legacy serving paths must not silently ignore an explicit request seed.

## Acceptance checks

Qualification separates the following checks:

- Check filtering against independent probabilities, generator forwarding,
  fixed-logit distributions, and greedy's lack of RNG consumption.
- Exercise mixed parameters, copied parameters, row changes, live arrivals,
  preemption, warm/shared prefixes, cancellation at every state, zero budget,
  EOS, and failure cleanup. Check both request and page ownership.
- Check HTTP validation and parameter delivery, seeded replay, disconnects,
  backpressure, and healthy continuation after invalid requests.
- On the local Qwen3-0.6B checkpoint, check FP32 and BF16 seeded replay under
  a fixed execution schedule, plus mixed live lifecycle handling. BF16
  cross-schedule token identity is not an acceptance gate.
- Run the full regression suite and the fixed Qwen3-8B cross-engine
  calibration. Retain the same-host baseline provenance; do not silently
  treat historical reference rows as fresh runs. Re-run reference engines
  when their artifacts, executable revisions, runtime, or hardware changed.
- Measure greedy overhead against the pre-M10c path and report stochastic
  sampling cost separately. Different random outputs cannot establish a
  matched-work speedup. Preserve raw token counts, ordering, and receipts.

`examples/bench_request_sampling.py` pins the pre-M10c session from commit
`8e18498` and alternates it with the new session over one loaded model. The
shared model/runtime stays fixed. Dynamically loaded control events use the
current event classes after a schema check; otherwise the offline adapter's
`isinstance` check would reject an equivalent, separately defined class.
This preserves the control's serving logic. After two warmups, each of four conditions
(128/128 or 2,048/32 prompt/output budgets, batch 1 or 8) has five paired
repetitions. Require identical greedy token IDs and both median
control/candidate wall-time ratio and its paired-bootstrap lower 95% bound
to be at least `0.95`. This is a predeclared 5% overhead budget for a
functionality change, not a required speedup. Peak allocation is retained.
The fixed-logit sampler timing uses batch 1/8/32 and includes host dispatch
and token readback; it is not full-model throughput.

The results below retain failures and limits. A seeded sampler is a
functionality change; no serving speedup is promised.

## CPU validation, 2026-09-25

The focused suite passed **128 tests**, with **two CUDA tests skipped** under
`CUDA_VISIBLE_DEVICES=''`. It covers the new sampler and RNG lifecycle plus
the existing session, HTTP, overload, and CPU disaggregation tests. The
real-forward replay check uses a tiny random two-layer FP32 Qwen model on
CPU, not a pretrained-checkpoint quality evaluation. The
[CPU receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10c-request-sampling-cpu-2026-09-25.json) records
the tested source hashes and exact validation boundary.

Ruff, launch-config JSON validation, local document links, and SVG rendering
checks passed. The figure's labels fit and do not overlap. The benchmark's
correctness driver also ran against the CPU test model, and its pinned
pre-M10c control session loaded and executed successfully. These checks do
not substitute for GPU execution.

That receipt describes the initial CPU-only boundary, not the later GPU runs.
GPU validation exposed a benchmark-only event-class identity bug in the
historical control loader. A CPU regression reproduced it before the fix;
the focused suite then passed **129 tests, with two CUDA tests skipped**.
The initial failed benchmark attempt is retained, not counted as a timing result.

## GPU validation and sampling cost, 2026-09-25

On Qwen3-0.6B, FP32 and BF16 repeated fixed-schedule runs produced identical
seeded token/event sequences. Mixed greedy/stochastic arrivals and cancellation
drained correctly and released generators and pages. The seeded `generate.py`
debug example also ran successfully. Final checkpoint and timing runs use
physical GPU 0, an RTX A6000, after the initial GPU 1 reservation became
unavailable. No other jobs were interrupted.

All four paired greedy conditions passed the frozen overhead gate:

| Prompt/output budget | Batch | Median control/candidate | Lower 95% bound |
|---|---:|---:|---:|
| 128/128 | 1 | 0.996 | 0.958 |
| 128/128 | 8 | 1.002 | 0.997 |
| 2,048/32 | 1 | 1.005 | 0.995 |
| 2,048/32 | 8 | 0.998 | 0.963 |

Greedy token IDs matched the control in every pair. Ratios close to one
support keeping the feature; they do not establish a speedup.

The separate fixed-logit measurement exposes the cost of the readable
per-request sampler. For batch sizes **1 / 8 / 32**, median host-visible
sampling time was **0.028 / 0.038 / 0.068 ms** for greedy, versus
**0.538 / 3.927 / 16.469 ms** for stochastic sampling with temperature 0.8,
top-k 20, and top-p 0.9. Each stochastic row runs its own filtering and
sampling operations over the vocabulary; launches and full-vocabulary work
therefore accumulate with batch size. These are sampler-only measurements,
not whole-model throughput or a claim that stochastic generation is cheap.

## Standard-suite calibration

The standard Qwen3-8B BF16 suite keeps greedy sampling, disables prefix
caching, enables CUDA graphs, and uses an 8,192-token prefill budget.
Each condition has two warmups and five measured repetitions. Median output
throughput in tokens/second was:

| Prompt/output budget | Batch | Historical M8g | M10c |
|---|---:|---:|---:|
| 128/128 | 1 | 38.14 | 37.62 |
| 128/128 | 8 | 284.34 | 282.47 |
| 128/128 | 32 | 915.24 | 910.28 |
| 2,048/32 | 1 | 26.63 | 26.59 |
| 2,048/32 | 8 | 67.71 | 67.78 |
| 2,048/32 | 32 | 79.83 | 79.91 |

All cases completed their full output budgets. The `2 + 2` smoke check
returned 4. Both runs used physical GPU 0, but the baseline is historical,
not an alternating paired trial: ratios of 0.986–1.001 are calibration
evidence, not a claim that M10c caused a throughput change. The paired
session test above is the focused overhead gate.

## Fresh HTTP calibration

All four engines ran Qwen3-4B with unquantized BF16 weights on the same
physical A6000. The HTTP client measures whole-cohort output throughput,
including client/server overhead, after two warmups and over five measured
repetitions. These are fresh runs, not reused reference rows:

| Prompt/output budget | Clients | Tinyserve | llama.cpp | FreeToken | Ollama |
|---|---:|---:|---:|---:|---:|
| 128/128 | 1 | 59.40 | 73.68 | 72.80 | 74.20 |
| 128/128 | 4 | 223.73 | 242.28 | 284.59 | 245.67 |
| 128/128 | 8 | 410.64 | 401.93 | 543.40 | 406.62 |
| 2,048/32 | 1 | 40.81 | 37.04 | 49.64 | 44.33 |
| 2,048/32 | 4 | 85.77 | 48.06 | 106.82 | 61.10 |
| 2,048/32 | 8 | 103.48 | 62.57 | 132.21 | 86.18 |

Values are median output tokens/second. FreeToken reports 127/31 tokens
per request for the 128/32 budgets; the other engines report 128/32.
Tinyserve disables prefix caching, while external engines retain the cache
policies recorded in their launch commands, including Ollama's native
behavior. Their batching and termination behavior also differ. Keep these
rows as workload-specific calibration, not exact matched-work speedups or
a universal engine ranking. Executable hashes are recorded separately from
external source-checkout revisions; the retained binaries were not rebuilt.

Tinyserve's additional HTTP checks covered staggered arrivals, client
disconnect, overload rejection, and release of all request-owned KV
references. Together with the standard suite, the result is a new sampling
capability with greedy performance essentially unchanged—not a serving
speedup milestone.

## Completion boundary

The final single-GPU suite passed **593 tests**, with **13 two-GPU tests
skipped**, in 147.84 seconds. The checkpoint-dependent KV tests were enabled
with the local Qwen3-0.6B model. FP32/BF16 CUDA RNG-isolation tests, the new
control-loader regression, and the existing serving tests all passed.
Ruff, launch-config JSON, local links, SVG label/layout checks, and chapter
rendering also passed.

The [GPU receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10c-request-sampling-a6000-2026-09-25.json)
retains the trial/token counts, confidence bounds, source/model/binary hashes,
GPU summaries, process-ownership checks, and the rejected initial benchmark
attempt. No foreign GPU jobs were interrupted. The benchmark-loader repair
changed no serving code. M10c is complete for ordinary single-device dense
Qwen sessions and HTTP streams; stochastic distributed, hybrid, quantized-KV,
speculative, and M11 PD paths remain outside this milestone.
