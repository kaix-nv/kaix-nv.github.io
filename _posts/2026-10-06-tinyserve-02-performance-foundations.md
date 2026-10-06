---
layout: post
math: true
title: 'Tinyserve, Chapter 2: How to reason about inference performance'
date: 2026-10-06 00:00:00 -0700
categories:
- tinyserve
- llm-serving
excerpt: What work are we measuring and can we trust the comparison?
book_chapter: 2
source_revision: 14b0c8dbb975350bbd730ac0c702eafce36eb95c
source_document: docs/book/02-performance-foundations.md
code_revision: e20a34815498560f9226a9e057ade31b2b62743f
---

{% include tinyserve-article-style.html %}
{% include tinyserve-book-nav.html %}

*Chapter 2 · Foundations and Measurement*

The uncached loop from Chapter 1 repeats work. That makes it an attractive
optimization target, but counting repeated operations does not tell us how
much faster a different implementation will run. An inference engine can
be limited by arithmetic, memory traffic, launch overhead, cache capacity,
or waiting behind other requests. A useful performance argument names the
resource and the measurement boundary before it names a speedup.

We will analyze one projection, then one request timeline. The numbers in
these worked examples are calculated or illustrative, not GPU measurements.
Afterward we will connect those models to Tinyserve's retained experiments
and the checks needed to trust a comparison.

## Separate capacity from traffic and compute

Memory capacity answers whether the model and its live state fit. Memory
traffic answers how many bytes must move while executing work. They are
related, but not interchangeable. A cache can occupy many gigabytes without
every operation reading all of it, and repeatedly reading a small allocation
can produce a large amount of traffic.

| Quantity | Typical unit | Question |
|---|---|---|
| Resident weights, KV, scratch | bytes or GiB | Will this workload fit? |
| Arithmetic work | floating-point operations, FLOPs | How much arithmetic does the operation require? |
| Data transferred | bytes | What must be fetched or written at the chosen memory boundary? |
| Compute rate | FLOP/s | How quickly can the relevant arithmetic execute? |
| Memory bandwidth | bytes/s | How quickly can that traffic move? |
| Elapsed time | milliseconds | What did the chosen observer wait for? |

Use GiB for powers of two and GB for powers of ten, or explicitly state a
different convention. Also specify what “memory” includes. GPU allocated
tensor memory, allocator-reserved memory, and device usage reported by a
driver are different measurements. Peak temporary workspace can be the
reason a workload fails even if the steady-state model and KV fit.

For a first approximation, separate model weights from state that scales
with requests. An ordinary KV cache scales with layers, KV heads, head
width, and cached tokens. Doubling concurrency can double private KV without
doubling a shared model's weights. Hybrid recurrent state has a different
shape, and prefix sharing changes how much KV is private. Later chapters
make those accounting rules explicit.

## Price one projection

Consider a linear projection with no bias:

$$
Y = XW^\mathsf{T}.
$$

Let X contain N input token rows of width 1024. W has 4096 output rows and
1024 columns, so Y has N rows of width 4096. Assume BF16 storage, two bytes
per value. Count a multiply-add as two FLOPs.

The weights occupy `4096 × 1024 × 2 = 8,388,608` bytes, or 8 MiB. The
arithmetic work is `2 × N × 1024 × 4096` FLOPs. A simplified traffic model
reads W and X once and writes Y once:

$$
F = 2N(1024)(4096),
$$

$$
B = 2(1024)(4096)+2N(1024+4096).
$$

F counts work and B counts bytes. This traffic model assumes useful reuse
inside the operation; it is not a measurement of DRAM transactions. Cache
residency, tiling, spills, and extra conversions can change actual traffic.

| Token rows N | Arithmetic | Modeled traffic | Arithmetic intensity F/B |
|---:|---:|---:|---:|
| 1 | 8.39 million FLOPs | 8.399 million bytes | 1.00 FLOP/byte |
| 128 | 1.074 billion FLOPs | 9.699 million bytes | 110.7 FLOP/byte |

Processing 128 rows does not require 128 independent reads of the weight
matrix in this idealized model. The additional rows reuse weights and give
the matrix multiplication more arithmetic per transferred byte. That is
one reason many-token prefill and small-batch decode behave differently.
It is also why batching can improve throughput without reducing the amount
of arithmetic needed for each token.

[![The same 8 MiB weight matrix serves one token row or 128 rows; work increases faster than modeled traffic, raising arithmetic intensity.](/assets/tinyserve/book02-projection-roofline.svg)](/assets/tinyserve/book02-projection-roofline.svg)

Now suppose a hypothetical device sustains 100 trillion FLOP/s for this
arithmetic and one trillion bytes/s at the memory boundary being modeled.
These are illustrative rates, not specifications or measurements for the
RTX A6000. Two resource lower bounds are

$$
t_{\text{compute}}=F/P,\qquad t_{\text{memory}}=B/W.
$$

Here P is compute rate and W is bandwidth, not the earlier weight tensor.
The simple roofline bound takes the larger time, rather than adding them:

$$
t \geq \max(t_{\text{compute}},t_{\text{memory}}).
$$

For one row, the bounds are about 0.084 microseconds of compute and 8.399
microseconds of traffic. For 128 rows, they are 10.737 and 9.699 microseconds.
The resource balance changes. The crossover is `P/W = 100 FLOP/byte`;
our 128-row example is just above it.

This is a lower bound under stated assumptions, not a latency prediction.
A tiny operation may attain neither rate. Launch overhead, insufficient
parallelism, shape inefficiency, dependencies, and synchronization can dominate.
Nor does a higher arithmetic intensity guarantee that an implementation is
faster: it may increase traffic elsewhere or perform unnecessary arithmetic.

## Prefill and decode have different work

Prefill processes the uncached prompt. There are many token rows, which often
lets projections reuse weights efficiently. Dense attention also has many
query/key pairs; its cost grows with prompt length. A long prompt can therefore
be expensive even when its matrix multiplications are efficient.

Cached decode adds one new token per active request. At low concurrency,
the projection matrices see only a few rows. Weight traffic and launch
overhead may matter more. Attention reads the existing KV history, so its
cost still grows with context length even though old projections are reused.
At larger batches or contexts the dominant resource can change again.

These are mechanisms to investigate, not universal labels. Saying “prefill
is compute-bound and decode is bandwidth-bound” is a starting hypothesis.
It leaves out batch size, context length, dtype, cache policy, model structure,
backend, and host overhead. A kernel trace and a matched timing experiment
must establish which resource matters for the actual condition.

Capacity also changes scheduling. A smaller KV representation may admit
more requests and increase throughput without speeding up one attention
call. Conversely, a faster kernel may not improve request latency if requests
spend most of their time waiting for admission. Always connect the local
operation to the serving workload it is supposed to improve.

## Name the request latency boundary

Imagine a request arrives at time 0 ms, begins model work at 30 ms, and has
its first token sampled internally at 82 ms. The client receives four
tokens at 90, 110, 137, and 160 ms. These are illustrative timestamps.

[![A request has distinct arrival, model-start, internal first-token, and client-receipt times; the four client tokens define TTFT and inter-token gaps.](/assets/tinyserve/book02-request-timing.svg)](/assets/tinyserve/book02-request-timing.svg)

Client-observed time to first token, or TTFT, is 90 ms. It includes waiting,
prompt processing, token selection, and delivery up to the first observable
token. It is not the 52 ms between model start and the first internal sample.
The inter-token latencies, or ITLs, are 20, 27, and 23 ms. Total request
latency through the last token is 160 ms.

The average time per output token after the first, often called TPOT, is

$$
\operatorname{TPOT}=\frac{160-90}{4-1}=23.33\text{ ms}.
$$

By contrast, dividing the four output tokens by total request time gives
25 tokens/s. Inverting TPOT gives about 42.86 tokens/s for the interval
after the first token. Both calculations are valid for their definitions;
they answer different questions. For a one-token response, TPOT based on
inter-token gaps is undefined, not zero.

At server level, aggregate throughput counts output tokens across all
requests over a defined observation window. Goodput additionally requires
an explicit criterion for useful or accepted work, such as completion within
latency targets. A median cannot reveal the worst tail behavior. Report
percentiles with their sample counts and avoid drawing tail conclusions
from a handful of requests.

Tinyserve's synchronous session records internal token events during an
iteration but returns the accumulated events only after the full step.
The online worker delivers them afterward. Internal event timestamps and
client HTTP timestamps therefore need distinct labels. Chapter 11 follows
that boundary through the streaming implementation.

## Observe asynchronous GPU work correctly

On CUDA, a model call generally enqueues work and returns before the GPU
finishes. A CPU timer around the Python call may measure submission rather
than completed execution. Two common measurements answer different questions:

- A synchronized host interval includes the host work and GPU completion
  inside the chosen boundary. Synchronize before starting it as well as
  after the measured work, so earlier queued work is not charged accidentally.
- CUDA events placed on the relevant stream measure the stream interval
  between those events after completion. Other streams require explicit
  dependency handling; one event pair is not automatically a whole-device
  or whole-request timer.

Tinyserve's `PhaseProfiler` keeps both host-region and device-event totals.
It resolves the events after the run rather than synchronizing at each small
region. A sampling region can have a long host time because reading a token
ID waits for the preceding forward, while its device sampling operation
is short. That does not mean sampling arithmetic consumed all the waiting
time.

Regions can also nest. KV writes inside a model forward are a diagnostic
subtotal; adding their time to the containing forward counts the same work
twice. Host time and device time should not be added either, because they
can overlap. Chapter 12 develops these attribution rules with the profiler.

## Make a comparison interpretable

A controlled comparison fixes the model artifact, input token IDs, requested
and actual output lengths, dtype, cache state, scheduling policy, hardware,
and measurement boundary. Then it changes the proposed intervention. If
several of those variables change, the result may still describe a useful
configuration comparison, but it cannot isolate one cause.

Warmup removes one-time setup from a steady-state question: loading, kernel
compilation, allocator growth, and graph capture can otherwise dominate a
short measurement. Cold-start behavior remains important, but it needs its
own experiment. State whether prefixes are already cached and whether graph
capture occurs inside or outside the timed interval.

Alternate reference/candidate order across paired trials to reduce monotonic
drift. Retain individual times, not just the fastest or the final average.
Define the ratio direction; in this book, a reference-time/candidate-time
ratio above one means the candidate is faster. Confidence intervals describe
uncertainty in the sampling protocol, not protection against a biased
protocol. Correlated trials on a drifting device do not become independent
just because a bootstrap was run.

Freeze correctness and non-regression gates before looking at performance.
Keep failed conditions. A change that reduces precision, returns fewer
tokens, or skips requests can be faster while failing the intended task.
If a numerical compatibility contract changes deliberately, name the new
contract and retain the old failed gate instead of relabeling it.

An A/A test runs the same implementation in both arms. If that experiment
cannot establish the intended near-one band, it warns that the timing
window is not stable enough for the proposed decision. It does not prove
that a candidate is slow, or give permission to discard inconvenient pairs.
Tinyserve's sampling work supplies a concrete example: the deterministic
sampler passed its adopted correctness checks, but candidate performance
passed only 48 of 54 conditions, and the follow-up uninstrumented greedy
reference trial passed 42 of 54 stability blocks. Batched serving remained
disabled. The [retained account](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/m10d-batched-sampling.md) explains the
separate numerical and measurement failures.

## Keep the evidence levels separate

| Evidence | What it can establish | What it does not establish alone |
|---|---|---|
| Analytical work/byte count | A resource hypothesis under stated assumptions | Actual latency or utilization. |
| Operator microbenchmark | Cost of an isolated operation and preparation included in its boundary | Whole-model or serving improvement. |
| Profiler trace | Where work occurs and which operations changed | Unperturbed request latency. |
| Paired whole-model run | Effect of an intervention on the fixed model workload | Generalization to other workloads or engines. |
| Standard serving suite | Behavior across the declared workload matrix | A causal decomposition of every difference. |
| Cross-engine calibration | Relative results for the captured engine/artifact/configurations | That a kernel alone explains the ranking. |

For example, a historical Qwen3-8B profile on an RTX A6000 attributed 98%
of the decode-heavy request's wall time to decode-forward device intervals.
The prefill-heavy workload instead spent 88.4% in prefill forwards. Those
profiles used different workload shapes and are evidence for choosing an
investigation, not an optimization gain. The subsequent unprofiled
instrumentation comparison was treated as unchanged, not credited with a
speedup for small movements. The [M7a record](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m7a-phase-profile-a6000-2026-08-29.json)
retains conditions and measurements.

Cross-engine comparison adds more variables: tokenizer/template differences,
weight formats, native prefix policy, graph defaults, stopping behavior,
and client interfaces. A run returning 31 tokens cannot silently stand in
for a 32-token run. Preserve failed or unmatched rows and state why they are
not included in a ratio. In particular, a quantized artifact and a BF16
artifact are not a pure engine comparison merely because their names refer
to the same model family.

## Follow the measurement implementation

The projection and timeline arithmetic above is checked in
[test_book_performance.py](https://github.com/kaix-nv/tinyserve/blob/14b0c8dbb975350bbd730ac0c702eafce36eb95c/tests/test_book_performance.py); it is not
a device benchmark. For runtime measurement, read
[profiling.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/profiling.py) for host/event ownership,
[profile_serve.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/examples/profile_serve.py) for phase attribution,
and [bench_cross_engine.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/examples/bench_cross_engine.py) for the
shared workload and adapter boundaries. These references describe snapshot
`e20a348`; source access is private, while the mechanisms above are complete
without it.

The useful optimization question is now precise: which resource limits this
workload, what intervention should change it, and what evidence would show
both correctness and improvement? The model-architecture chapters apply that
question first to attention and feed-forward computation.

{% include tinyserve-book-nav.html %}
