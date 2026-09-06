---
layout: post
math: true
title: "Building tinyperf M45: Two GPUs on silicon: validating tensor and pipeline parallelism"
date: 2026-09-05 23:15:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Frozen predictions meet a real two-GPU vLLM run: tp=2 prefill is slower than one GPU on a PCIe link exactly as predicted, a host-loop NCCL benchmark turned out to be measuring Python, and the pipeline engine pipelines prompt chunks the model did not know about."
---

*Milestone 45 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tools/measure_vllm.py`, `tools/measure_nccl.py`, `tinyperf/serving.py` · Data: `data/validation/` · Tests: `test_tensor_parallel_silicon_envelope`, `test_pipeline_silicon_envelope`.*

Every multi-GPU number this series has produced so far — the TP cliff of
milestone 34, the pipeline layouts of milestone 44 — rested on a comm
model that had never met a real link. This milestone puts both schedules
on silicon: Qwen3-8B in bf16 under vLLM 0.15.1, on the two RTX A6000s in
the development box, once as `tp=2` and once as `pp=2`, on the same
eight-cell grid as the single-GPU validation of milestone 32. Predictions
were frozen and committed before each run was opened. The box is a hard
case on purpose: the two GPUs talk through the PCIe host bridge, no
NVLink bridge installed, so communication is expensive and every error in
the comm model is magnified.

## Measure the link first

The comm model needs a bandwidth and a per-hop latency. A 2-GPU NCCL
benchmark gave 4.0 GB/s for large all-reduces and an 87 µs floor for
small ones, flat from 8 KB to 256 KB. That flatness should have been a
warning. Fitted to the ring closed form it gave a 41 µs hop, and the
frozen tensor-parallel predictions carried it.

## Tensor parallelism: the prefill prediction that nobody wants

On a 4 GB/s link, a 2048-token prefill all-reduces 16 MB per layer, twice
per layer. The model said all-reduce would be two thirds of every
prefill step, and that `tp=2` prefill would therefore be *slower* than
one GPU by 1.4–1.6×, while decode — 8 KB messages — would be faster. That
is an uncomfortable prediction to freeze. Silicon agreed with it in
every cell:

```
                   TTFT ms               TPOT ms
  b  prompt   meas     pred  ratio  vs 1 GPU     meas   vs 1 GPU
  1    512    122.1   118.1   0.97    1.53       13.90    0.58
  1   2048    453.7   456.6   1.01    1.56       14.04    0.58
  1   8192   1813.5  1903.0   1.05    1.41       14.86    0.58
  8    512    880.8   892.8   1.01    1.56       17.50    0.70
  8   2048   3517.2  3603.8   1.02    1.54       18.75    0.67
  8   8192  14563.5 15178.2   1.04    1.40       24.42    0.62
 32    512   3470.4  3554.7   1.02    1.58       24.77    0.83
 32   2048  13952.3 14391.5   1.03    1.53       30.53    0.73
```

TTFT: geometric mean 1.02, every cell within 0.97–1.05, with the frozen
predictions. The bandwidth term is right.

Decode was another story. The frozen predictions came in 1.06–1.42×
*above* silicon, worst at batch 1 where a step is 72 tiny all-reduces
and per-message latency is everything. That 87 µs floor was the culprit:
a Python loop issuing `all_reduce` calls is host-bound, and what it
measured was the interpreter, not the wire. Re-measured with the
collectives captured in a CUDA graph — the way a serving engine actually
replays them — an 8 KB all-reduce takes 16.5 µs on this link, and every
message above 1 MB is unchanged. The hop latency refits to 8 µs from
that benchmark alone; nothing was fitted to the vLLM numbers. With it,
decode lands at 0.98 geometric mean, 0.91–1.08, and the direction of the
trade-off — prefill slower, decode 1.7× faster — is reproduced
cell by cell.

The lesson generalizes beyond this box: **a link latency measured from a
host loop is not a link latency.** A 5× error in one constant turned a
correct model into a 40% miss, and the tell was visible in the raw
benchmark before any silicon was touched.

## Pipeline parallelism: the engine pipelines more than you asked for

Milestone 44 priced a pipeline step as a steady state with one
micro-batch per stage, and noted that a single prefill cannot be split.
vLLM's scheduler forms micro-batches by rules the model did not know,
and the silicon exposed two of them.

**Chunks pipeline.** Prefill is chunked at the token budget, 8192 here,
and with pipeline parallelism the scheduler re-schedules a request's
remaining prompt chunks while earlier chunks are still in flight on the
other stage. So a 64k-token prefill wave — eight chunks — drains through
two stages in nine stage-times, not sixteen. The frozen prediction had
these cells 1.24–1.66× too slow; silicon ran them in 0.62–0.64 of the
single-GPU time. The wave formula is now

```
TTFT(wave) = (chunks + pp − 1) / pp × traverse(chunk)
```

with a chunk's work being its group's work over the chunk count. It is a
mechanism, not a fitted constant, and it moved TTFT from 1.22 to 0.99
geometric mean (0.91–1.05) across the eight cells.

**Groups persist into decode.** The chunks that were prefilled as
separate groups stay separate: the cells whose prefill spanned two token
budgets ran decode as two in-flight groups and beat the single GPU by up
to 16%, while single-group cells ran 8–11% slower than it. The step
model's default — `pp` micro-batches in flight — is right for a
continuously-fed server; for an offline burst the in-flight count is
`min(pp, ceil(tokens / budget))`, and the comparison uses that rule.

**One engine constant.** The single-group cells sat a flat 1.9–3.0 ms
per step above the model regardless of batch or context: the cost of
vLLM's two-process pipeline coordinating a step, the same kind of
software constant as the prefill overhead of milestone 39.
`pp_step_overhead_us = 2200` was fitted on those four cells; the four
two-group cells, held out, landed at 0.95–1.01.

```
  pp=2, eight cells        TTFT ratio      TPOT ratio
  frozen (v1)              1.22  (1.01–1.66)  0.99  (0.90–1.15)
  chunk wave + constant    0.99  (0.91–1.05)  0.99  (0.95–1.01)
```

## Two traps, recorded so nobody repeats them

**Capping admission is not micro-batching.** To force two micro-batches
I ran `pp=2` with the engine's batch cap set to half the burst. That does
not split the batch across stages; it admits half the requests and runs
the other half afterwards. TTFT matched one GPU, TPOT doubled (2.0–2.1×),
and the model reproduces it as two sequential waves (TTFT 1.02–1.08,
TPOT 0.85–1.02). The knob that creates in-flight groups is the token
budget. The run is kept in `data/validation/` as a protocol finding.

**The custom all-reduce hung.** vLLM's peer-to-peer custom all-reduce
deadlocked on the first step over this host bridge; the workers timed
out. `--disable-custom-all-reduce` forces NCCL and the run went through.
It also means the validated path is the NCCL one the link model
describes — an NVLink pair with the custom kernel enabled is a different
measurement, still to be made.

## Baseline control

Everything above compares against the single-GPU run of milestone 32,
taken a week earlier. Three of its cells were re-measured on the same day
as the multi-GPU runs: 1.02/1.00, 1.00/1.01, 1.00/1.00 for TTFT/TPOT.
The pipeline decode wins over one GPU are real, not drift.

## What this validates, and what it does not

Validated, and pinned in CI: tensor parallelism and pipeline parallelism
at width two, one model, one software stack, one PCIe link. The comm
model's bandwidth and latency terms are separately confirmed. The
pipeline model's wave formula, in-flight grouping and engine constant
are confirmed on held-out cells.

Not validated: an NVLink pair, widths beyond two, deeper pipelines, and
any of it on a datacenter GPU. The 2.2 ms engine constant is a property
of vLLM 0.15's multi-process executor and should be re-measured on any
other stack. The custom all-reduce path was not measured at all.

## Exercises

1. Run `tools/measure_nccl.py` in both modes on a machine you have and
   compare. How big is the gap on NVLink?
2. Set `--max-num-batched-tokens 2048` and re-run the `pp=2` burst: the
   single-group cells should split into groups. Does decode speed up as
   the model says, and does the 2.2 ms constant hold?
3. The pipeline hand-off is priced on the all-reduce link rate (4 GB/s)
   while send/recv measured 6.6 GB/s. Add a separate point-to-point rate
   and see whether any cell moves.
