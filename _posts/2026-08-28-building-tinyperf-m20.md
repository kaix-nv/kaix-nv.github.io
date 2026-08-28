---
layout: post
math: true
title: "Building tinyperf M20: Paged KV — what optimism about output lengths buys"
date: 2026-08-28 16:45:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Schedulers know the API cap, not the output length. Reservation queues on phantom space; paged admission with recompute-preemption converts it into throughput."
---

*Milestone 20 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `kv_paging` in `serving.py` · Demo: `examples/18_paged_kv.py`.*

Every serving milestone so far shared a quiet assumption: the scheduler
knows each request's output length, so reserving `prompt + gen` KV tokens
at admission was nearly free. Real schedulers know no such thing — they
know the API cap (`max_tokens`), and honest reservation must plan for it.
That gap between the cap and reality is where paged KV lives, and it is
the technique's actual justification — not fragmentation trivia.

## Two allocators, one honest traffic model

The traffic model gains the uncertainty: actual `gen ~ U[gen/2, gen]`,
scheduler sees only `max_gen`. Then:

- **reservation** — admit only if `prompt + max_gen` fits; nothing is
  ever evicted. Safe, and honest about what a scheduler can know.
- **paged** — vLLM-style: requests hold `ceil(tokens/page)` pages of
  *current* KV and grow page by page. When the pool runs out, the newest
  running request is **preempted by recompute**: KV dropped, requeued at
  the front, `prompt + generated` re-prefilled on re-admission. Admission
  optimism, backed by an escape hatch.

(Single-request-at-a-time head-of-line chunking and paged×chunked
composition remain exercises; the pool guard rejects any request that
could never fit alone.)

## The table

7B on the calibrated A6000; prompt 512, actual gen ~U[128,256], cap 1024:

```
  rate    allocator  TTFT p50  p95 (ms)  TPOT p50   tok/s  preempt
   2.0  reservation        94       165      41.8     355        0
   2.0        paged        94       165      41.8     355        0
   4.0  reservation      4344      7055      60.6     518        0
   4.0        paged       121       209      79.7     592        0
   8.0  reservation      9061     20476      60.6     529        0
   8.0        paged       178     11151     111.7     631       22
```

Reservation holds 1536 tokens per request; the median request ever uses
~700. Three loads, three lessons:

- **2 req/s**: memory never binds — the rows are *identical*. Allocators
  only matter when the resource they allocate is scarce; below that,
  paged KV is complexity for nothing.
- **4 req/s**: the phantom-space regime. Reservation queues requests
  against memory that will never be used — TTFT p95 is **30x worse** —
  while paged admits them with *zero preemptions*: the optimism was
  simply correct. This is the regime the vLLM paper's headline numbers
  come from.
- **8 req/s**: the pool genuinely fills. Paged pays 22 recompute
  preemptions and still wins both tails and throughput. And the TPOT
  rise (60.6 → 111.7 ms) is the honest fine print: freed memory becomes
  bigger batches, and bigger batches decode slower. **Paging converts
  phantom space into throughput, and the exchange rate is set by your
  TPOT SLO.**

## In production-scale models

Production allocators add prefix sharing (paged KV's second superpower:
common prompt prefixes share pages), swap-to-host as an alternative to
recompute, watermark-based admission to damp preemption storms, and the
paged×chunked composition this milestone leaves as an exercise. The
two-line summary survives all of it: reserve for the cap and you queue on
phantoms; admit on actuals and you must be able to preempt.

## Exercises

1. Compose paged KV with chunked prefill (M19) — preemption interacts
   with the waiting queue; does the overlap win survive memory pressure?
2. Prefix caching: give requests a shared system prompt and share its
   pages. Re-derive max_batch and the 4 req/s row.
3. Swap vs recompute: price preemption as a host-transfer (PCIe
   bandwidth from the device spec) instead of a re-prefill. Which wins,
   where?
4. Preemption storms: push to 12 req/s and plot preemptions vs page
   size — small pages admit more but thrash more.
