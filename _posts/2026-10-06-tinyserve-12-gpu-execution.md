---
layout: post
math: true
title: 'Tinyserve, Chapter 12: Profiling GPU execution and reducing overhead'
date: 2026-10-06 00:00:00 -0700
categories:
- tinyserve
- llm-serving
excerpt: Which execution overhead should we change and does the gain survive measurement?
book_chapter: 12
source_revision: 14b0c8dbb975350bbd730ac0c702eafce36eb95c
source_document: docs/book/12-gpu-execution.md
code_revision: e20a34815498560f9226a9e057ade31b2b62743f
---

{% include tinyserve-article-style.html %}
{% include tinyserve-book-nav.html %}

*Chapter 12 · Execution Optimization*

A model can be mathematically efficient and still execute thousands of
small GPU operations per token. It can also spend most of its time in large
projections, making a conspicuous Python loop a poor optimization target.
The task is to connect a request-level delay to a concrete execution path,
change that path, and check whether the expected improvement survives an
unprofiled serving measurement.

We will follow one decode step through host submission, graph replay, and
kernel fusion. The distinction matters: a CUDA graph reduces repeated host
submission work, while fusion changes the GPU work being submitted. Neither
automatically changes the model's arithmetic, scheduler policy, or attention
algorithm.

## Read both the host and device timelines

Chapter 2 distinguished completed GPU work from asynchronous submission.
Tinyserve implements that distinction in `PhaseProfiler.region(name)`.
Entering a region starts a host timer and, on CUDA, records an event on the
current stream. Leaving records the ending event and the host duration.
`finish()` synchronizes once and resolves device elapsed times.

The serving loop labels scheduling, decode forwards, sampling, prefill
packing, prefill forwards, and paged KV writes. These labels identify
ownership boundaries, not disjoint counters that can all be summed.
KV writes are nested inside the forward; sampling's host interval can
include waiting for that forward to finish before a token ID is copied
to the CPU.

For a simple mental trace, imagine Python submits a forward, then asks for
the sampled integer. The forward region can finish on the host before its
GPU kernels finish. The integer conversion cannot complete until those
kernels and the sampling operation finish. Charging all of that wait to
sampling arithmetic would select the wrong optimization.

One retained Qwen3-0.6B smoke profile illustrates this distinction:

| Region | Host total | Device total |
|---|---:|---:|
| Prefill forward | 25.61 ms | 26.00 ms |
| Seven decode forwards | 2.27 ms | 36.83 ms |
| Eight sampling regions | 35.81 ms | 0.80 ms |

The complete call took 64.78 ms. These are historical single-run diagnostic
numbers from the [phase-profiling account](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/m7-phase-profiling.md), not a
new throughput comparison. They do not say that replacing 35.81 ms of
sampling arithmetic is possible: most of that host interval was waiting.

Graph replay introduces another boundary. Python does not reenter each
captured layer, so replay supplies one outer decode interval without eager
per-layer Python profiling regions. Use an eager trace when investigating
those regions and a kernel trace when opening the graph itself. Neither
should silently replace the ordinary unprofiled configuration used to report
serving latency.

## Separate capture from fusion

Suppose one operation executes three kernels, A, B, and C. Eager execution
submits them through the host repeatedly. A CUDA graph records their
dependencies and replays the recorded graph. The GPU still executes A, B,
and C. A fused kernel instead performs their combined work in one device
program.

[![Eager execution launches three kernels from the host, graph replay submits a recorded three-kernel dependency graph, and fusion changes those kernels into one device program.](/assets/tinyserve/book12-graphs-and-fusion.svg)](/assets/tinyserve/book12-graphs-and-fusion.svg)

Graphs can reduce framework and submission overhead. Fusion can additionally
remove kernel boundaries, temporary allocations, and intermediate memory
traffic. A fused implementation can also lose: larger live working sets
increase register pressure, spills can reintroduce traffic, and a generic
fusion may be worse than a well-tuned standalone matrix multiplication.
The intervention needs a predicted mechanism and a measurement, not just
a smaller source-code expression.

## Capture a decode shape and update its contents

Tinyserve's `GraphRunner` captures decode buckets of size 1, 2, 4, 8, 16,
32, and 64. With three live requests, the engine selects the batch-four
bucket. Each replay uses persistent storage for token IDs, logical positions,
KV write slots, attention metadata, and logits.

The important distinction is **contents versus addresses**. Request IDs,
positions, and page references change each iteration; the captured operations
must still refer to the same buffer addresses. `copy_` updates existing
storage. Replacing a buffer with a newly allocated tensor would not make
the captured graph follow the new Python variable.

For three live requests:

| Captured row | Token and position | KV destination and history | Output |
|---:|---|---|---|
| 0 | A's latest token and next logical position | A's real slot and page table | Used for A. |
| 1 | B's latest token and next logical position | B's real slot and page table | Used for B. |
| 2 | C's latest token and next logical position | C's real slot and page table | Used for C. |
| 3 | Dummy token 0, position 0 | Scratch slot and scratch page | Discarded. |

The dummy row must not overwrite a real request's KV or contribute a sampled
output. Its scratch storage is initialized to finite data; discarding its
logits does not justify feeding arbitrary NaNs through the recorded work.
The real rows keep their own positions and page tables. Padding a graph
bucket is an execution representation, not a change to request identity.

Before capture, the runner allocates its buffers, creates graph-compatible
attention wrappers, and warms each bucket. Warmup keeps one-time setup out
of the capture. Buckets are captured largest first and share a graph memory
pool under the implementation's nonconcurrent replay assumptions. The runner
copies new values, updates the wrapper's metadata, calls `replay()`, and
returns only the live rows.

This implementation graphs supported paged decode; prefill stays eager.
It also explicitly rejects quantized KV in the graph runner. The existence
of a graph optimization does not mean every model, cache representation,
or serving mode uses it.

## Metadata capacity can force eager fallback

Physical KV capacity is not the same as metadata capacity. Suppose the
pool has eight usable pages and three requests share four immutable prefix
pages, then each has one private tail page:

```text
A: [0, 1, 2, 3, 4]
B: [0, 1, 2, 3, 5]
C: [0, 1, 2, 3, 6]
```

Only seven physical pages are occupied, but the three page tables contain
15 references. The batch-four graph also needs one scratch-page reference
for its dummy row. Its captured index buffer has `num_blocks + bucket_size`
entries: `8 + 4 = 12`. Sixteen references do not fit in twelve slots.

[![Three requests use seven physical pages through fifteen references; adding the dummy row requires sixteen metadata entries, exceeding the batch-four buffer's twelve-entry capacity and triggering eager fallback.](/assets/tinyserve/book12-graph-metadata.svg)](/assets/tinyserve/book12-graph-metadata.svg)

`GraphRunner.fits()` checks both live batch size and flattened reference
count, including dummy rows. The engine must take the eager path if either
does not fit. Increasing sharing can reduce physical memory while increasing
the number of metadata references relative to that memory. Counting only
allocated pages would miss this overflow.

The test suite checks exact and padded buckets, changing page tables,
shrinking batches, prefix reuse, and preemption. Those tests support their
stated text and ownership contracts; they do not imply bitwise identity of
every intermediate tensor on every backend or device.

## Choose one operation inside the graph

After capture removes much repeated host dispatch, the next profile can
show a different bottleneck. In a retained Qwen3-8B BF16 graph trace on one
RTX A6000, batch 1 and context 128, projections dominated device time.
The graph also contained many short normalization operations.

There are four RMSNorm calls per layer—two block-level norms plus Q and K
norms—and one final norm. With 36 layers, that is `36 × 4 + 1 = 145` calls
per decode token. The readable reference formula reduced and scaled FP32
intermediates before returning to the input dtype. In that trace, each
normalization expanded to nine recorded operations.

Replacing those nine operations with one fused RMSNorm predicts a reduction
of `145 × (9 − 1) = 1,160` kernel events per token. This prediction is much
more useful than “fusion should be faster”: it gives the trace an observable
consequence to confirm or reject.

The [retained M7c receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m7c-fused-rmsnorm-a6000-2026-08-29.json)
records the result:

| Signal per token | Reference graph | Fused-norm graph |
|---|---:|---:|
| RMSNorm calls | 145 | 145 |
| Normalization kernel events | 1,305 | 145 |
| All kernel events | 2,353 | 1,193 |
| Summed profiled kernel duration | 27.38 ms | 25.52 ms |

The observed event reduction is exactly 1,160. Summed profiler durations are
still diagnostic: they are not substituted for an unprofiled serving timer.
A separate 40-step, five-warmup CUDA-event check measured 27.51 versus
25.81 ms/token, about a 6.2% reduction, with the same allocated memory.
Those measurements belong to that model, shape, software revision, and
hardware, not to normalization fusion in general.

The standard serving suite was checked separately. In its batch-one,
128-prompt/128-output-token condition, output throughput moved from 35.82
to 38.16 tokens/s with a 512-token prompt budget, two warmups, and five
measured repetitions. This serving comparison and the operator trace answer
different questions; neither number should be presented as the other.

## Keep a readable numerical reference

The first broad fused-norm dispatch was not accepted unchanged. A rare
rounding difference caused a ragged-versus-padded prefill comparison to fail
its existing logit tolerance. Instead of widening that tolerance, the
implementation restricted fused normalization to the operations recorded
during graph capture.

`_fused_rmsnorm_capture(model)` temporarily enables each relevant module's
fused flag, warms and captures the graphs, then restores the previous flags
even if capture fails. Replay retains the fused GPU operations already
recorded. Ordinary eager forwards and the FP32/CPU reference retain the
readable expression.

```python
previous = [norm.use_fused_kernel for norm in norms]
try:
    for norm in norms:
        norm.use_fused_kernel = True
    # warm and capture supported decode buckets
finally:
    for norm, enabled in zip(norms, previous):
        norm.use_fused_kernel = enabled
```

This is a bounded dispatch contract, not evidence that changing execution
order is always harmless. A reference path makes later layout changes easier
to diagnose: the optimized graph does not silently become the oracle for
every other feature.

Kimi's learned residual mixer supplies a second fusion pattern. It scores
a small set of saved residual vectors and the current prefix, then computes
a softmax-weighted sum of the original vectors. Its native Triton path keeps
an online maximum, normalization sum, and vector accumulator in registers
instead of materializing concatenated and normalized candidate tensors.
The native path applies only to its declared CUDA inference shapes and
dtypes; the PyTorch expression remains the fallback. The complete equations
and qualified boundaries are in the [residual-mixer account](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/m7j-fused-kimi-residual.md).
The general lesson is to fuse a specific producer/consumer chain, not to
replace a model's learned mixing rule with an easier approximation.

## A failed optimization can be the correct outcome

The sampling work illustrates why a good mechanism is not sufficient for
promotion. Batching deterministic filters can reduce repeated top-k, sort,
and softmax invocations while keeping each request's random generator.
However, floating-point cumulative sums changed a top-p cutoff, which
changed the set of retained tokens. A small arithmetic difference had a
discrete semantic effect.

The later approved revision used a shared, sampler-local deterministic
prefix-sum policy for reference and candidate, with the compatibility change
stated explicitly. Correctness passed that adopted contract. Performance
still passed only 48 of 54 conditions, and a subsequent 256-call greedy
reference qualification passed only 42 of 54 stability blocks. The candidate
remained disabled. The longer [sampling record](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/m10d-batched-sampling.md)
retains the failed prerequisites and diagnostics.

An unstable near-parity condition is neither a demonstrated slowdown nor
a demonstrated non-regression. A large speedup in other rows cannot erase
it when the declared gate requires every condition. Sampler-only gains also
cannot establish whole-model gains: the model, synchronization, and request
schedule may dominate the complete call.

## An implementation and evidence checklist

Before changing a hot path, write down the predicted observation. Examples
include fewer recorded kernels, fewer temporary bytes, unchanged page
ownership, or a shorter selected phase. Then check both the mechanism and
the boundary it is supposed to improve:

1. Establish the reference's numerical and ownership behavior.
2. Capture a diagnostic trace for the exact workload.
3. Change one scoped path and retain a usable reference dispatch.
4. Check intermediate correctness before headline timing.
5. Verify the predicted trace or memory change.
6. Measure unprofiled paired latency, then the relevant serving suite.
7. Preserve failed conditions and keep the old default when gates fail.

Read [profiling.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/profiling.py) for the two clocks,
[graph_runner.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/graph_runner.py) for capture buffers and
metadata fit, and [models/qwen3.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/models/qwen3.py) for
RMSNorm dispatch. [test_graphs.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/test_graphs.py) and
[test_rmsnorm.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/test_rmsnorm.py) name the tested boundaries.
The described runtime snapshot is `e20a348`; historical receipts identify
their own measured revisions. Source access is private, but no source access
is needed to follow the examples above.

Optimization is complete only when the changed mechanism explains the
observation and the accepted benefit reaches the intended scope. Graphs,
fusion, quantization, and speculation are different interventions; the next
chapters examine what each changes and what it must preserve.

{% include tinyserve-book-nav.html %}
