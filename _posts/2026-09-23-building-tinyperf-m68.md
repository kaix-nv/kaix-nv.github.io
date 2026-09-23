---
layout: post
math: true
title: "Building tinyperf M68: The decode step from its kernels: the context term fixed, and the mixed step it uncovered"
date: 2026-09-23 00:10:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "The decode step, priced from vLLM's attention and GEMM kernels measured alone and the batch's mean context instead of its rounded-up maximum, lands within 1% on a frozen batch whose longest context is twice its mean, where the old pricing was 10-24% high. A frozen varied-length sweep then missed by 6-12% above 1 req/s, and the engine's clock found why: in a step that carries a prefill chunk, FlashAttention-2 stops packing GQA and re-reads every decode row's cache per query head."
---

*Milestone 68 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — mean-context pricing in `serving.py`, the decode-attention and GEMM constants in `methodology.py`, `tools/measure_decode_attention.py`, `tools/measure_mixed_attention.py` · Data: `data/validation/comparison_qwen3_8b_rtx_a6000_context.txt`, `engine_steps_varied_qwen3_8b_rtx_a6000.json`.*

Milestone 67 left the model with a known wrong term. `simulate` priced
a decode step at the running batch's *maximum* context, rounded *up* to a
multiple of 256 tokens. Attention reads each sequence's own cache, so the
true cost follows the mean. That overprice had been cancelling the
missing sampler.

The obvious fix, price the mean and interpolate between buckets, moved
seven envelopes at once. Milestone 45's tensor-parallel decode at batch
32 fell to 0.88. So this milestone took the step apart and priced each
piece from a kernel measured on its own. Only then did it freeze
predictions and run.

## What the round-up had been hiding

Three things, each measured alone.

**Decode attention.** `tools/measure_decode_attention.py` calls vLLM's own
FlashAttention-2 decode kernel the way its backend does: paged KV,
16-token blocks, one query token per sequence, inside a CUDA graph. It
sweeps batch 1–64 against context 256–8192. The model's generic price
(KV bytes at the calibrated DRAM rate) was 14% off on average and 57% at
worst. Short contexts miss a floor: one sequence at 256 tokens takes
11.7 µs a layer, not 5. Two constants fitted over the 36 cells bring it
to 5% mean and 3.8% at batch 8 and above:

- **7 µs per call,**
- **plus the KV at 0.96 of the calibrated rate.**

**The decode GEMMs.** `tools/measure_decode_gemms.py` times Qwen3-8B's
decode GEMMs as vLLM calls them, at every CUDA-graph capture size, with
shapes taken from the model's own graph. At 1–32 rows the 36 layers' sum
is on the model. At 40–64 rows it runs 5–12% slower: cuBLAS moves to
taller tiles and streams the weights more slowly. One constant, 0.92 of
the calibrated rate for 33–64 rows, fits tp=1 within 0.97–1.04. Held out
on tp=2's per-rank shapes, it lands within 0.94–1.03.

**The LM head under tensor parallelism.** The graph had it unsharded,
"for simplicity". vLLM shards the vocabulary across ranks and
all-gathers the logits. At tp=2 that overpriced batch 1 by 0.9 ms and
left out a gather that grows with the batch.

Two engine details came along:

- **The full graph:** a step whose requests all schedule one token stays
  in the full CUDA graph. That includes a request arriving on a
  disaggregated decode server with its KV already loaded. It therefore
  doesn't pay milestone 57's cost for leaving the graph.
- **The sampler:** it runs on every scheduled request, including a
  partial prefill whose token is discarded.

## Frozen, then measured

The predictions were committed before anything ran, with the pre-M68
model beside them for contrast. The sharpest test is a steady batch in
which half the sequences have 128-token prompts and half 1920. The
batch's maximum context is then about twice its mean, and the two
pricings disagree most:

```
  batch   engine step   M68     pre-M68
    16      28.36 ms    1.01    1.10
    32      32.90       1.01    1.19
    48      38.99       0.99    1.17
    64      43.38       1.01    1.24
```

The mean-context pricing is right to 1% at every batch size; the old
pricing was 10–24% high. Three new tensor-parallel cells (batch 16, 48,
64) came in at 1.01–1.02 on TTFT and 0.90–0.93 on TPOT. The remaining
tp=2 gap is NCCL, not the step. The all-reduce at 64–256 KB messages
runs ~13 µs a call slower than the ring model fitted on small messages,
and the logits all-gather runs at the all-gather's lower rate: 1–4 ms a
step, named in the frozen file and not priced.

The earlier sweeps, re-priced:

- **Decode-heavy TPOT:** 1.04–1.06 → **1.00–1.01**.
- **1P1D decode:** 1.02–1.06 → 1.00–1.03.
- **M54 saturation:** 0.95–0.99.

## The held-out sweep, and the step it found

The last frozen run was the one this milestone was designed around: an
online sweep with prompts drawn from 102–1945 tokens and outputs from 25
to 486, the server's default top-p sampling, and the engine's step clock
running.

At 1 req/s it landed: TTFT 0.99, TPOT 1.01. From 2 req/s it did not.
TPOT came in 6–12% low and saturated throughput 7–8% high, so the
model's queue grew too slowly. The pre-M68 model missed the other way,
7–10% high on TPOT and 4–5% low on throughput.

The engine's clock says where. Pure decode steps at 60 or more sequences
took 49.7 ms; the model says 48.8. The decode step is fine. The steps
that carry a prefill chunk beside 63 decodes are not:

```
  chunk tokens    measured    model    ratio
     185          122.3 ms    73.7     0.60
     523          160.9      128.8     0.80
    1165          239.1      225.8     0.94
    1844          345.0      328.8     0.95
```

FlashAttention-2 has a trick for decode. It packs each KV head's query
heads into rows, so each KV head's cache is read once. It applies the
trick only when every sequence in the call has one query token. In a
step that also carries a prefill chunk, the decode rows run unpacked,
and each of Qwen3-8B's 32 query heads re-reads its KV head's cache.
`tools/measure_mixed_attention.py` times this on the kernel. At 63 decode
rows and a 1150-token context, attention costs 1647 µs a layer mixed,
against 443 for the same rows alone and 18 for the chunk alone. That is
2.7 extra reads of the decode KV out of a possible 3; L2 catches the
rest. The effect falls to 0.6 extra reads at 8 rows.

That is also what milestone 57's mixed-step constants were. Its "213 µs
per running sequence" was fitted at a ~278-token context, where the
re-read measures about 0.16 ms per sequence. The re-read grows with context
and batch, which a per-sequence constant cannot, so a varied-length
workload at saturation found it.

## What changed, and what did not

- **Changed:**
  - the decode step is priced from three measured kernels and the mean
    context;
  - the LM head is sharded under tensor parallelism;
  - the one-token-step and sampler-row details above.
- **Unchanged:** prefill and mixed-step pricing; the mixed step is the
  next milestone.
- **Tests:** five envelopes pinned to the old pricing are re-pinned. The
  new test pins the kernels, the frozen results, and the miss.
- **Mean-context TTFT:** at saturation the varied-length sweep's TTFT
  runs 0.30–0.90, a consequence of the model's queue growing too slowly.
  It should close once the mixed step is priced.

## Exercises

1. Price the mixed step's decode attention as unpacked GQA with an L2
   filter, and freeze a varied-length sweep before measuring. Does M57's
   per-sequence constant go to zero?
2. Measure NCCL's all-reduce and all-gather at 16 KB – 16 MB on this pair
   and give the ring model two protocols. Does tp=2 decode close?
3. Re-run `tools/measure_decode_attention.py` with the FlashInfer backend.
   Does it pack GQA in mixed batches?
