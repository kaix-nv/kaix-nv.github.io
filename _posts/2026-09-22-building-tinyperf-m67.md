---
layout: post
math: true
title: "Building tinyperf M67: The step's last 4.7 milliseconds: the sampler, and the error it had been cancelling"
date: 2026-09-22 18:00:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Every serving sweep since milestone 54 had quietly run top-p sampling, because vllm bench serve no longer sends temperature 0. Top-p sorts each row's 152k-token vocabulary on every step: 4.7 ms of a 36 ms step at 64 sequences, the per-sequence cost milestone 55 fitted and milestone 57 removed. Measured on the engine's own GPU clock and priced from a kernel measurement, it lands within 0.3 ms at serving level, and exposes the decode price's context term that had been cancelling it."
---

*Milestone 67 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `SAMPLERS` and `StepLatencyModel(sampling=)` in `serving.py`, `tools/measure_sampler.py`, `tools/vllm_patches/step_timing` · Data: `data/validation/comparison_qwen3_8b_rtx_a6000_sampler.txt`, `decode_step_decomposition_qwen3_8b_rtx_a6000.json`.*

The oldest open miss in the envelope was the saturated decode step.
Three serving sweeps showed it:

- **Milestone 54, one GPU:** TPOT 0.86 at the knee and 0.91–0.95 at
  saturation.
- **Milestone 66, the two-GPU pair:** 0.81–0.92 at 6–8 req/s.
- **The decode-heavy sweep:** its median inter-token gap sat about 4.5 ms
  above the graph's decode step at 64 sequences.

Milestone 56 had measured that step directly at 64 sequences and found it
exact (32.1 ms against 32.6), so the missing cost appeared only under
load. This milestone went looking for it.

## The difference between the two measurements

Before any profiler, one question: what did the load sweeps do that
milestone 56's step measurement did not? The answer was in the request
bodies. Milestone 56's tool sent `temperature: 0`. `vllm bench serve`
sends no sampling parameters at all. Since 0.15 it no longer defaults to
greedy, and it prints a warning saying so. The server then applies the
model's own generation config, which for Qwen3 is temperature 0.6, top-k
20, top-p 0.95. The server log said it too: *"Default vLLM sampling
parameters have been overridden by the model's generation_config.json."*

Any top-p takes vLLM's native sampler down its sort path. Every row's
151,936 logits are sorted, softmaxed, cumulatively summed and scattered
back, every step. The FlashInfer sampler, which avoids the sort, is
opt-in.

## The sampler, on the kernel

`tools/measure_sampler.py` calls vLLM's own `Sampler` module with real
sampling metadata and bf16 logits, and times only its GPU work. A
sleeping kernel holds the GPU while the CPU enqueues the sampler, which
is exactly how a step meets it, queued behind the graph's forward:

```
  rows     greedy    temperature    top-k    top-p (Qwen3's config)
     1      0.015        0.069      0.370       0.262   ms
    16      0.048        0.207      0.474       1.505
    32      0.088        0.381      0.775       2.533
    64      0.164        0.741      1.460       4.681
```

Each mode is a fixed cost plus a per-row cost. Expressed as fp32 passes
over the vocabulary at the calibrated DRAM rate, the per-row cost is:

| mode | passes per row | fixed |
|---|---|---|
| greedy | 2.8 | 10 µs |
| temperature only | 12.7 | 25 µs |
| top-k | 23 | 140 µs |
| top-p | 76 | 410 µs |

That puts top-p at 66.6 µs per row. Those constants are
`serving.SAMPLERS`, fitted on this table and nothing else. The engine
pays the sampler on every step, for the rows that emit a token.

One line in milestone 55 now reads differently. It fitted "63 µs per
running sequence per step" on the decode-heavy sweep and called it
scheduling, sampling, detokenizing and streaming. Milestone 57 set that
constant to zero, because milestone 56's greedy steps did not show it.
The top-p sampler costs 66.6 µs per row.

## The engine's own clock

A client sees a step as the gap between two streamed tokens, blurred by
HTTP and its own event loop. `tools/vllm_patches/step_timing` reads the
step off the GPU instead. It records CUDA events on the model runner's
stream at each forward's start and around the sampler, and reads them
back later without a sync. It also records what each step held.

With predictions committed first, `tools/measure_decode_step.py` ran B
requests side by side under each sampling mode:

```
  top-p, B    step period   model    sampler in the engine   model
     1           24.29      24.32            0.29             0.48
    16           27.27      27.51            1.55             1.48
    32           30.09      30.99            2.58             2.54
    48           34.08      33.41            3.65             3.61
    64           36.23      37.28            4.71             4.68
```

- **The step:** priced within 0.95–1.04 across all four modes and five
  batch sizes.
- **The sampler inside the engine:** within 0.25 ms of a price fitted
  on an isolated kernel.
- **Top-p minus greedy at 64 rows:** 4.76 ms measured, 4.51 predicted.

That is the missing cost, at the size the sweeps needed.

## The same difference, at serving level

Two sweeps had been measured with top-p: milestone 54's (1024-token
prompts, 128 out) and milestone 55's decode-heavy one (128 in, 512 out).
Both were re-run with `--temperature 0`, predictions frozen first. The
measured difference in mean TPOT between the old top-p run and the new
greedy one is the sampler at serving level, and the model predicted it:

```
  decode-heavy, rate      1      2      4      8
  top-p minus greedy   1.45   3.23   3.94   4.05   ms measured
                       1.55   2.90   4.06   3.94   ms predicted
```

It is within 0.33 ms at every decode-heavy rate. On the M54 shape it is
within 0.9 ms away from the knee; at the knee, queueing amplifies any
step difference.

## What it had been cancelling

The frozen greedy sweeps did not land at 1.00. Decode-heavy TPOT came in
at 1.04–1.06, and M54 below the knee at 1.02–1.04. Re-pricing the older
top-p sweeps with the sampler shows the same split:

| sweep | before | after |
|---|---|---|
| M54 saturated TPOT | 0.92–0.95 | 0.95–0.99 |
| M54 saturated throughput | 1.06 high | 1.02 high |
| M57 saturated TPOT | 0.93 | 0.97–0.98 |
| M57 saturated throughput | 1.07–1.08 high | 1.04–1.05 high |
| M54 below the knee, TPOT | 0.98–1.04 | 1.02–1.08 |
| M55, TPOT | 0.93–1.00 | 1.04–1.06 |
| M66 1P1D decode, TPOT | 0.98–1.00 | 1.02–1.06 |

The engine's clock says why. `simulate` prices a decode step at the
running batch's *maximum* context, rounded *up* to a multiple of 256
tokens. Attention reads each sequence's own cache, so the true cost
follows the mean. On the decode-heavy shape at saturation, that is 768
tokens priced against a mean near 384. The control's cells, where every
sequence has the same context at each step, separate the rest:

- **At 1–32 rows,** the graph at the *exact* context matches the measured
  forward within 0.5 ms.
- **At 48–64 rows,** it runs 1.7–2.9 ms low. Most of that is in the
  GEMMs. `tools/measure_decode_gemms.py` times the 36 layers' decode
  GEMMs as vLLM calls them: 21.0–22.3 ms at 1–32 rows, on the model, but
  23.2–23.9 ms at 40–64 rows against 21.2–22.1 priced. cuBLAS picks
  different kernels above 32 rows (`gate_up` 353 µs against 313 at 40).
- **The round-up hides both.** At 32 and 64 rows it runs 1.1–1.3 ms
  high, which is why milestone 56's greedy step looked exact.

So three errors had been cancelling: a missing sampler, a context term
that runs high, and a GEMM regime that runs low. Pricing the context at
the mean, interpolated between buckets, was tried here. It moves six
envelopes, including milestone 45's tp=2 decode at batch 32 to 0.88, so
it needs its own frozen test on a workload with mixed context lengths.
It is recorded in `decode_step_decomposition_qwen3_8b_rtx_a6000.json`
and left for the next milestone. The envelopes are re-pinned to what the
model now does, with the reason written next to each bar.

## What changed, and what did not

- **Mechanism:** one new one, the sampler, as a per-step cost on
  emitting rows, with `sampling=` as a user input. The default is
  greedy, because a deployment's sampling config is a property of its
  requests. If you serve with a model's generation config, say so.
- **Instrument:** the engine's step clock, reusable for any vLLM 0.15.1
  run.
- **Offline prices:** unchanged. The offline tools were greedy and never
  carry the sampler.
- **Errata:** on milestones 55, 56 and 57.

## Exercises

1. Set `VLLM_USE_FLASHINFER_SAMPLER=1` and measure top-p with
   `tools/measure_sampler.py`. What is FlashInfer's rejection-sampling
   cost per row, and does it depend on the distribution's entropy?
2. Price the decode context at the batch mean, interpolated between
   buckets, then freeze a sweep whose prompts vary in length
   (`--random-range-ratio`) before measuring. Which of the six envelopes
   was right to move?
3. Add the capture sizes 40–64 to `tools/measure_gemm.py`'s grid and
   refit. Can one smooth calibration express a cuBLAS kernel switch, or
   does it need a regime constant like milestone 65's MoE one?
