---
layout: post
math: true
title: "Building tinyserve M4: Continuous batching"
date: 2026-08-18 14:25:17 -0700
categories: [tinyserve, llm-serving]
excerpt: "Turn paged KV storage into a streaming scheduler with admission, prefill priority, and recomputation-based preemption."
---

*Milestone 4 of [building an LLM inference engine from scratch](/series/tinyserve/).
Previous: [M3 — paged KV]({% post_url 2026-08-17-building-tinyserve-m3 %}).*

Code: [`tinyserve` @ `3869d03`](https://github.com/kaix-nv/tinyserve/tree/3869d03bfe8e9ac8a011dffaab243db07b12b682).

## The problem

M3's `generate_paged()` improves a fixed list of prompts. Finished rows
leave the running batch and release their blocks, but the API cannot add
a new request to the same call. That freed capacity remains idle until
the rest of the list drains.

A server instead receives a stream. Requests arrive at different times,
have different prompt lengths, and finish after different numbers of
tokens. The engine now needs policy, not just storage:

- when an arrived request may enter the running set;
- whether the next GPU work is prefill or decode;
- which request gives up memory when the KV pool is full.

M3 built the allocator. M4 adds the scheduler that makes those choices.

## The request state machine

`serve()` keeps future arrivals outside the scheduler until their
arrival time is reached. From there, each request follows this loop:

```text
pending --arrival reached--> waiting --admit + prefill--> running
                               ^                            |
                               |       preempt + free KV    | EOS or token budget
                               +----------------------------+
                                                            |
                                                            v
                                                         finished
```

One engine-loop iteration makes one scheduling decision:

1. Move newly arrived requests from `pending` to the scheduler's
   first-come, first-served `waiting` queue.
2. Call `schedule()`. If one or more waiting prompts fit, admit them and
   prefill each request's current token history sequentially. For a new
   request that history is its prompt; for a preempted request it also
   includes generated tokens. Existing running requests do not decode
   during that iteration.
3. If nothing is admitted, call `reserve_for_decode()`, preempting under
   memory pressure if necessary, then advance every surviving running
   request by one token in a batched decode.
4. A request that reaches EOS or its token budget immediately releases
   its blocks. A later iteration can reuse them for new work.

![Continuous batching timeline showing a finished request replaced while
another request remains active](/assets/tinyserve/m4-continuous-batching-timeline.svg)

*Figure 1. M4 reconsiders batch membership at every engine-loop boundary.
A and B enter together and prefill sequentially. After A finishes and
returns its KV blocks, request C is admitted and prefills without
waiting for B to finish. Because M4 gives admission priority to prefill,
B pauses during that iteration. B and C then decode together in the next
iteration.*

The running batch can therefore change after every iteration. That is
the "continuous" in continuous batching. Notice that M4's *prefill
phase* may contain several model forwards—one per admitted prompt. It is
not a single packed prefill kernel.

## Admission and preemption

Admission preserves one watermark block as decode headroom. Within one
call to `schedule()`, a local `reserved` counter also tracks blocks
promised to earlier admissions that have not been allocated yet. A
request is admitted only when

```text
required history blocks + reserved in this call + watermark <= free blocks
```

M4 gives admission priority over decode. This reduces queueing for work
that fits, but a long prefill pauses token generation for every request
already running. M5 will split that work into chunks so decode can run
between them.

Decode does not allocate on every token. A sequence needs another block
only when `num_cached` crosses a block boundary. Before a decode step,
the scheduler counts how many running sequences are at such a boundary.
If the pool cannot cover them, it repeatedly removes the tail of the
running list—the most recently admitted request—until the survivors fit.

Preemption frees all of a victim's blocks and puts it at the front of the
waiting queue. When capacity later permits readmission, the normal
prefill path rebuilds K/V from its prompt plus generated tokens, then
generation continues. Front-of-queue placement preserves its waiting
turn, but it does **not** guarantee immediate readmission or starvation
freedom. LIFO is a small, deterministic teaching policy, not a latency
optimality claim.

![LIFO preemption freeing KV blocks before decode and later rebuilding
the victim's cache](/assets/tinyserve/m4-preemption-recompute.svg)

*Figure 2. Before decode, A and B each need one more block but only one is
free. `reserve_for_decode()` removes the tail of the running list, B,
and frees all of B's blocks. A can then grow and decode. B keeps its token
history and sampling/timing metadata, returns to the front of `waiting`,
and later rebuilds K/V through the normal prefill path when it fits
again.*

Recomputation is correct because, for this model, K/V is derived from the
token sequence and positions. The model-history state needed to rebuild
the cache is therefore the request's tokens; scheduler metadata such as
sampling parameters and timing remains on the `Request` object.

## The build

New: [`scheduler.py`](https://github.com/kaix-nv/tinyserve/blob/3869d03bfe8e9ac8a011dffaab243db07b12b682/tinyserve/scheduler.py) — `SamplingParams`
(per-request now, finally honoring a promise from M1's sampler),
`Request` (a `Sequence` plus params, arrival time, timing records),
`RequestOutput` (text + TTFT + per-token times), and `Scheduler`
(`schedule()` admits, `reserve_for_decode()` preempts, and `finish()`
releases). The engine gains `serve(prompts, params, arrivals)` and the
loop above.

The arrival schedule is synthetic, but the benchmark enforces it against
the wall clock: a future request is not made visible early, and reported
TTFT and inter-token latency surround real model execution. M3's
prefill/decode code was extracted into `_paged_prefill()` and
`_paged_decode()` helpers shared by `generate_paged()` and `serve()`.
Preemption recovery is the normal prefill helper applied to a longer
token sequence.

One real bug, caught by a unit test: admission originally checked each
waiting request against `num_free` — but admitted requests don't
allocate until their prefill runs, so admitting A (2 blocks) then
checking B against the *same* `num_free` over-admits the entire queue.
The fix is a `reserved` counter inside `schedule()`. This is the
classic TOCTOU gap between deciding and doing, and schedulers are made
of little else.

## The proof

For deterministic greedy decoding, scheduling should change *when* work
runs without changing the returned text
([`tests/test_scheduler.py`](https://github.com/kaix-nv/tinyserve/blob/3869d03bfe8e9ac8a011dffaab243db07b12b682/tests/test_scheduler.py)):

1. Unit tests on a microscopic pool (no GPU): admission respects the
   block budget including same-call reservations; preemption picks the
   LIFO victim, frees everything, and requeues at the *front*; an
   oversized request raises instead of deadlocking.
2. **Simultaneous-arrival parity:** in fp32 gather mode, `serve()` and
   `generate_paged()` return identical decoded strings for four prompts.
   This checks end-to-end output parity; it does not compare logits or
   token IDs directly.
3. **Preemption parity:** an eight-block pool forces at least one
   eviction on the same workload, and the returned strings must match a
   large-pool `serve()` run. Evict → recompute → continue is invisible in
   the returned text.
4. A staggered-arrival test checks that the second request cannot finish
   before its configured arrival time.

The complete M4 suite passes: 19 tests. The parity checks use greedy,
fp32 gather mode. They do not establish identical random samples when
stochastic requests are regrouped, or preemption parity for FlashInfer.

## A retained measurement snapshot

The original M4 run used 96 requests, `max_new=128`, bf16 + FlashInfer,
and synthetic Poisson arrivals at rate λ
([`examples/bench_serve.py`](https://github.com/kaix-nv/tinyserve/blob/3869d03bfe8e9ac8a011dffaab243db07b12b682/examples/bench_serve.py)). With seed 0,
10 of the 96 requests use one of the two long-prompt templates. The
baseline replays the same stream as three gated cohorts of 32; each
cohort starts only after its final member has arrived. The script calls
this path `static`, but it still uses M3's paged `generate_paged()` and
removes finished rows. The controlled difference is fixed cohort
membership—no mid-flight admission—not M2's padded cache layout.

TTFT is arrival-to-first-token latency. ITL contains gaps between tokens
after the first. Continuous goodput is returned, non-EOS token IDs
divided by wall-clock makespan. The static path reconstructs its token
count by re-tokenizing returned text.

| λ (req/s) | reported goodput (tok/s, cont / static) | continuous TTFT p50/p99 | static start delay p50/p99 |
|---:|---:|---:|---:|
| 2 | 87 / 81 | 38 / 79 ms | 7.3 / 15.0 **s** |
| 8 | 287 / 252 | 39 / 70 ms | 2.0 / 3.9 s |
| 16 | 453 / 287 | 43 / 113 ms | 3.1 / 5.9 s |
| 32 | 660 / 311 | 52 / 141 ms | 3.4 / 6.8 s |

| λ (req/s) | continuous ITL p50/p99 |
|---:|---:|
| 2 | 25 / 52 ms |
| 8 | 24 / 92 ms |
| 16 | 25 / 149 ms |
| 32 | 25 / 408 ms |

Treat this as a historical snapshot, not a statistically stable result:

- the raw command output, model revision, GPU, and complete software
  environment were not retained;
- each row is one seed and one run, with no error bars;
- for 96 request-level samples, the harness's p99 selector lands on the
  maximum TTFT or queue delay; ITL has many more samples;
- continuous goodput counts returned token IDs, whereas the static path
  re-tokenizes decoded strings. The two methods are not guaranteed to be
  identical for every tokenizer output.

Within those limits, three observations are useful:

- **Latency is the headline, not throughput.** At light load (λ=2) both
  paths report similar goodput, but the no-timeout static baseline makes
  a median request wait 7.3 seconds for its batch to start. Continuous
  batching reports a 38 ms median to the first generated token. Sparse
  arrivals are especially unfriendly to a fixed-size gate.
- **The reported throughput gap grows with load.** At λ=32 the snapshot
  reports 660 versus 311 tok/s. Continuous batching can refill capacity
  released by finished requests, while a fixed batch drains to
  stragglers. Because utilization was not profiled and token counting
  differs between paths, the table does not isolate that mechanism or
  establish a precise 2.1× speedup.
- **The ITL tail motivates M5.** Median ITL stays near 25 ms while the
  reported p99 grows from 52 to 408 ms. The engine structurally blocks
  decode while it serially prefills admitted prompts, so this pattern is
  consistent with prefill/decode interference. The retained data does
  not include a per-event trace proving how much of each spike came from
  a particular long prompt. M5's chunked prefill is designed to bound
  this blocking interval.

And preemption, isolated (λ=16 workload, shrinking pool):

| KV pool (token slots) | goodput (tok/s) | TTFT p99 | ITL p99 | preemptions |
|---:|---:|---:|---:|---:|
| 65,536 | 456 | 100 ms | 149 ms | 0 |
| 8,192 | 456 | 108 ms | 143 ms | 0 |
| 4,096 | 442 | 643 ms | 280 ms | 8 |

In this run, reducing the pool by 16× changed reported goodput from 456
to 442 tok/s and caused eight preemptions. TTFT p99 rose from 100 to
643 ms and ITL p99 from 149 to 280 ms. The correctness test proves that
recomputation preserves returned text; this single benchmark run shows
the latency and repeated-work cost, not a general SLO guarantee.

The useful mental model is: preemption turns insufficient resident KV
capacity into recomputation and waiting. That can let a workload finish
instead of failing allocation, but it is not free capacity.

## Reproducing

```bash
pytest tests/ -q                                    # 19 tests
python examples/bench_serve.py --rates 2 8 16 32 --n 96 --static
for pool in 65536 8192 4096; do
  python examples/bench_serve.py --rates 16 --pool-tokens "$pool"
done
```

The benchmark honors arrivals with real sleeps. With seed 0, the final
λ=2 arrival is about 44 seconds after the first, and the static
comparison replays that schedule, so the full sweep takes time.

These commands rerun the protocol; they do not guarantee the historical
numbers above. For a new result, retain the command output, prompt seed,
model revision, GPU, PyTorch/CUDA/FlashInfer versions, and repeat each
condition enough times to report variability.
