---
layout: post
math: true
title: "Building tinyperf M69: The mixed step from its parts: two constants become two mechanisms, and the model checks its own measurement"
date: 2026-09-24 00:20:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "A step that carries a prefill chunk, priced as one forward from kernels measured alone: FlashAttention-2 re-reading each decode row's cache per query head, cuBLAS's row curve with its 128-row tile edges, and the LM head only where tokens are sampled. Frozen and timed on the engine's own clock, 36 mixed steps land within 1.9% (the previous model: 15.5%), and milestone 57's two fitted constants turn out to be those two mechanisms. The step model also caught a throttled GPU in its own measurement."
---

*Milestone 69 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `mixed_step_us` and `mixed_reread_us` in `serving.py`, `dense_gemm_row_factor` in `methodology.py`, `tools/measure_mixed_attention.py`, `tools/measure_prefill_step.py` · Data: `data/validation/comparison_qwen3_8b_rtx_a6000_mixed_step_engine.txt`, `engine_steps_m69_qwen3_8b_rtx_a6000.json`.*

Milestone 68 ended on a diagnosis. A step that carries a prefill chunk
beside decoding sequences was priced up to 40% low. The reason was on
the kernel: FlashAttention-2 packs a KV head's query heads into rows only
when every sequence has one query token, so beside a chunk each decode
row's cache is read once per query head. This milestone prices that, and
everything else a mixed step does, from kernels measured alone.

## The re-read, measured

`tools/measure_mixed_attention.py` times vLLM's FlashAttention-2 three
ways: the decode rows alone, the chunk alone, and both together. It
sweeps 220 cells of rows × context × chunk at Qwen3-8B's 32 query heads
per 8 KV heads, plus 24 cells at two other ratios. The extra time is
(query heads per KV head − 1) × m extra reads of the decode KV, where m
is the fraction that misses L2. It scales with the ratio: at 63 rows it reaches 0.9 of 1, 2.7 of 3, and 6.6
of 7. The fraction m rises with batch, from ~0.2 at 8 rows to ~0.9 at 63,
because fewer of a KV head's query heads are resident at once. It also
rises a little with context and chunk.

A linear form in those three terms, fitted on the 32/8 grid, prices the
mixed kernel within 11% per layer, against 52% with no term. Held out on
64/8 heads it lands within 12% (68% before).

## The step as one forward

M54 had priced a mixed step as decode plus chunk minus one weight pass.
That was a stand-in for what the engine does: run one forward over all
the step's rows. The step is now composed that way:

- **Token-proportional work** (GEMMs, norms, all-reduces) at the combined
  row count.
- **The LM head at the rows that sample.** The old chunk graph had priced
  it on every chunk token, about 10 ms at 1024 tokens.
- **Each side's attention,** plus the re-read.

Composing at the combined rows exposed the next thing: dense GEMMs are
not smooth in their row count. `tools/measure_decode_gemms.py` times
Qwen3-8B's layer GEMMs as vLLM calls them from 1 to 1280 rows:

- **Between the regimes they run slow.** Between weight-streaming and
  compute-bound, cuBLAS runs up to 1.4× the smooth model (at 129–320
  rows).
- **Its 128-row tiles leave edges.** Adding one row past 128, 256, 384,
  512 or 1024 costs 8–34% more.

The model now follows the measured curve: 33 row counts with both sides
of every edge, corrected times interpolated between them, and whole
128-row tiles beyond. Held out on tp=2's per-rank shapes, the curve is
within 3.7% on average.

The same kind of round-up that M67 found on the decode side turned up
in prefill: a prompt was priced at its length rounded up to 128 tokens.
With cuBLAS's edges, a 160-token prompt priced as 256 was 1.37× the
engine's own clock. At exact lengths, engine-clocked prefill steps from
32 to 1024 tokens land within 0.90–1.05.

## Where M57's constants went

M57 had fitted 3.3 ms + 213 µs per running sequence on every step that
carries prefill, as the cost of leaving the full CUDA graph. The engine's
clock says there is no host-side cost to find: a step's period is its GPU
forward plus the sampler, to 0.2 ms, in every sweep. The per-sequence
term was the re-read, measured at one short context. The fixed term was
mostly the 1024/1025-row tile edge: a 1024-token chunk plus a few
decodes crosses it, and the step's GEMMs grow 8% for a handful of extra
rows. Both are priced now, and the constants are zero.

## Frozen, then measured

Predictions were committed before the runs, with the pre-M69 model beside
them. The mixed step itself came first, on the engine's clock: 8, 32 or
63 decoding sequences at background contexts of ~450 to ~2150 tokens,
with 128- to 1024-token prompts injected. Over the 36 cells:

- **M69:** 1.9% mean error (0.97–1.07) at the context the engine recorded.
- **pre-M69:** 15.5% (0.56–1.03).

The M54 shape again: TPOT 0.93–1.04, throughput 0.98–1.01.

The long-prompt sweep (prompts 204–3891, outputs 12–243) needed two runs.
In the first, from 1.5 req/s the server's TPOT jumped to 257–284 ms.
The step model's own check caught the cause. Pure decode steps, which
the model prices within 2% everywhere else, ran 1.9–2.1× in alternating
30-second windows. That was a GPU running slow, not a pricing error, and
the neighbouring GPU was carrying another job at the time. The re-run
logged clocks and temperatures. The GPU sat at its power cap with
software thermal slowdown, 1470–1530 MHz at the high rates, and the pure
decode steps stayed on the model throughout.

In the re-run, every step that carries prefill, priced at the
composition the engine recorded, lands within 0.99–1.02 of measured at
the median at every rate. Up to 1.5 req/s the sweep is on the model
(TPOT 0.87–0.99, TTFT 0.88–0.95), and throughput is within 0.94–1.01 at
every rate. At 2–2.5 req/s, where the server only just keeps up, TPOT
reads 0.62–1.00 and TTFT 0.37–0.58. The steps are right; the queue near
saturation is not. That is a question about how a scheduler packs long
prompts when it is barely keeping up, and it is the next one.

## Errata

- **M56:** its mixed-step cells were the clients' largest inter-token
  gap. Under async scheduling that under-reads a mixed step, because the
  step's delay splits across two gaps. The engine clock reads 60.1 ms
  where M56's client read 50.0.
- **M57:** its constants were the re-read and the tile edge.
- **M68:** its 33–64-row GEMM constant was the first stretch of the row
  curve.

## Exercises

1. The long sweep's queue at saturation: compare the model's and vLLM's
   step compositions at 2 req/s (the engine clock records vLLM's). Which
   scheduling choice diverges first?
2. Fit the re-read's miss fraction from an occupancy model (resident
   query heads per KV head versus L2 capacity) instead of a linear form,
   and check it on the 16/8 grid.
3. Run `tools/measure_mixed_attention.py` with the FlashInfer backend.
   Does it pack GQA in mixed batches, and what does that do to M69's
   steps?
