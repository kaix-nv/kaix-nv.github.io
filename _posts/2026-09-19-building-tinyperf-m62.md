---
layout: post
math: true
title: "Building tinyperf M62: Expert parallelism on two GPUs: what the engine moves, the padded idle replica, and the router measured"
date: 2026-09-19 23:30:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Expert parallelism meets silicon: gpt-oss-20B as two attention replicas with sixteen experts each over a PCIe link. Predictions frozen from reading the engine got every direction right — including two GPUs being slower than one at batch 1, because the idle replica is padded and routed — and the misses led to three measurements: the link's per-collective rates, the router's real expert counts, and a decode kernel that streams weights at two thirds of the DRAM rate."
---

*Milestone 62 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `ep_dispatch`, lockstep padding and the routing rule in `nets/transformer.py`, per-collective link rates in `device.py` · Data: `data/validation/comparison_gpt_oss_20b_bf16_rtx_a6000_dp2ep2.txt`, `router_gpt_oss_20b_random_tokens.json`, `kernel_profile_gpt_oss_20b_rtx_a6000_decode.json`.*

Expert parallelism was the largest thing in this model that had never
met a GPU. The all-to-all pricing of milestone 11, the imbalance skew of
milestone 43 and the attention-replica semantics of milestone 46 all
carried error bars by construction. gpt-oss-20B in bf16 fits the two RTX
A6000s of the development box with room to spare, and vLLM runs it as
two attention replicas each holding sixteen of the thirty-two experts:
`tp=1, dp=2, ep=2`. Same eight cells as milestones 32 through 61, same
protocol — read the engine, price what it does, freeze, run.

## Two things read from the engine before the run

**The dispatch is not an all-to-all.** Without DeepEP-class kernels, the
engine's default expert-parallel path all-gathers every token's hidden
state *and* its router logits across the replicas, has each rank run its
local experts over the whole global batch, and reduce-scatters the
result. Each token crosses the link once each way regardless of how many
experts it was routed to. The milestone-11 pricing sends `top_k` copies
of a routed row and collects them back — right for a DeepEP kernel, and
for `top_k = 4` over two ranks twice the bytes this engine moves. The
graph builder gained `ep_dispatch="allgather"` and a `ReduceScatter`
op; the all-to-all stays the default and the other pricing.

**The idle replica is not idle.** The replicas step in lockstep because
every layer's collectives need both of them, and with CUDA graphs on the
engine pads every replica to the *largest* replica's token count. At
global batch 1 one replica has the request and the other runs a dummy
batch of the same size — 8192 dummy tokens for an 8192-token prompt —
and dummy tokens are routed to experts like real ones. So each rank does
the single GPU's expert work at batch 1, plus the collectives. The
expert layer sees `dp × ceil(batch/dp)` sequences.

The measurement tool learned to launch one engine process per GPU the
way `torchrun` would, split the global batch evenly, and have a replica
with no requests step dummy batches until the other is done. A cell's
time is the max over replicas; the two agreed within 1% in every cell.

## The frozen predictions, and what silicon said

The frozen file lists both dispatch pricings; the engine's was the
prediction. Every number below in milliseconds; `1 GPU` is the
single-GPU bf16 run of milestone 61.

```
                       TTFT                                TPOT
   b  prompt   meas  frozen    r   a2a    r  meas/1GPU    meas  frozen    r  meas/1GPU
   1     512   122    100   0.82  135  1.11    1.31       13.06  12.43  0.95   1.13
   1    2048   407    388   0.95  529  1.30    1.66       12.73  12.49  0.98   1.09
   1    8192  1871   1601   0.86 2164  1.16    1.67       13.21  12.71  0.96   1.09
   8     512   415    384   0.93  525  1.26    0.85       20.23  19.76  0.98   0.66
   8    2048  1795   1536   0.86 2099  1.17    0.94       19.95  19.98  1.00   0.66
   8    8192  7894   6391   0.81 8644  1.10    0.87       22.78  20.85  0.92   0.77
  32     512  1768   1519   0.86 2082  1.18    0.96       37.19  29.69  0.80   0.87
  32    2048  7201   6129   0.85 8381  1.16    0.95       39.05  30.56  0.78   0.85
```

Three things the frozen model got right, cell by cell. Two GPUs are
*slower* than one at global batch 1 — 1.3 to 1.7× on prefill, 1.1× on
decode — exactly the padded-replica prediction, and a result nobody
would guess from "twice the GPUs". At batch 8 and 32 they are faster in
both phases, by less than a factor of two, because each rank streams
about half the expert weights and pays two collectives per layer. And
the engine's dispatch pricing beat the all-to-all pricing at every
prefill cell: 0.81–0.95 against 1.10–1.30. On this link the choice of
collective pattern is worth 30%, and reading the engine picked the
right one.

Two things it got wrong. Prefill ran 13–19% under silicon at every cell
but one. Decode at batch 32 — sixteen sequences per rank — ran 20% low.
Neither had a fitted constant available to hide in, so both had to be
found.

## Reading the link again

Milestone 45 measured this PCIe pair's all-reduce (4.0 GB/s per
direction, 8 µs hop) and the model priced every ring collective at that
rate. NCCL's all-gather and reduce-scatter are different code paths, and
`tools/measure_nccl.py` now measures them the same CUDA-graph-captured
way:

```
  per direction, large messages     all_reduce  all_gather  reduce_scatter
  RTX A6000 pair, PCIe host bridge    4.0 GB/s    3.36 GB/s     3.13 GB/s
```

The device gained `all_gather_bw_gbps` and `reduce_scatter_bw_gbps`,
zero by default (equal to the link rate, which is what NVLink gives),
set from the benchmark here. That is half the prefill miss. The other
half is in the kernel: the fused-MoE kernel's activation and top-k sum
run over *all* gathered pairs, not the local rank's — the pairs owned by
the other rank are zero-filled, not skipped. Priced from the kernel
source, elementwise over `dp × tokens × top_k` rows. Prefill lands at
0.96 geometric (0.90–1.06), from 0.87.

## The router, measured

The decode miss led somewhere else. Milestone 61 had fitted a Zipf
exponent of 1.4 to step times and called it the router's skew. This time
the router was asked directly: forward hooks on the 24 routers of the
bf16 checkpoint, random-token prompts, the distinct experts selected at
the last position of every sequence in the batch.

```
  global batch          1     2     4     8    16    32
  router (measured)   4.0   5.5   8.4   9.7  12.5  15.9   of 32
  M61 fit (z = 1.4)   4.0   6.0   8.9  12.9  17.9  23.4
  uniform             4.0   7.5  13.2  21.0  28.2  31.6
  Zipf z = 2.15       4.0   5.3   7.2   9.6  12.8  16.7
```

The router is more concentrated than the timing fit said — a batch-8
step touches 10 experts, not 13, and a batch-32 step 16, not 23. Per
layer, two or three experts are picked by essentially every random
token. The two halves of a contiguous 2-way expert split saw equal
counts at every batch (4.7 / 4.9 at batch 8; 7.8 / 8.1 at 32), so the
per-rank count is the global count over `ep`, not a popularity law
re-drawn on the local pool — the builder's earlier local-pool rule
over-counted the skewed case by 40%. The preset carries `routing_skew =
2.15`, and the number now means what it says.

That leaves a question. If the router touches fewer experts than
milestone 61 assumed, why did milestone 61's step times fit? Because
something else was hiding in the count.

## What the kernel does with its time

vLLM's torch profiler, around 64 decode steps of a 64-token prompt, on
one GPU, for both checkpoints; kernel durations summed per step. The
decode step is GPU-bound to within a millisecond at every batch, and
the expert kernel is where the time goes:

```
                              expert kernel   router     touched-expert    effective  vs calibrated
                              ms per step     experts    weights per step  GB/s       DRAM rate
  bf16 (Triton)   batch 1        7.6           4.0        4.8 GB            631        0.91
                  batch 8       25.4           9.7       11.6 GB            457        0.66
                  batch 32      41.6          15.9       19.0 GB            457        0.66
  MXFP4 (Marlin)  batch 1        2.5           4.0        1.3 GB            511        0.74
                  batch 8        8.1           9.7        3.1 GB            381        0.55
                  batch 32      14.3          15.9        5.0 GB            353        0.51
```

With one row per expert the kernels stream the touched weights at about
the calibrated rate. With more than one — batch 8 and up — the bf16
kernel drops to two thirds of it and the Marlin kernel to half, and
stays there. That is what the 1.4 skew had been paying for: a larger
expert count standing in for a slower kernel. The model now carries the
two as separate, separately measured constants: `moe_dram_efficiency =
0.70` and `weight_only_dram_efficiency = 0.53`, applied to expert GEMMs
with more than one row per expert, alongside milestone 61's math-side
constants for prefill. Neither was fitted to a step time. The
single-GPU grids re-land at 0.98–1.13 (bf16) and 0.91–1.18 (MXFP4) on
decode, with batch 1 unchanged at 0.99–1.02 — the case that never had a
constant.

The same profile on rank 0 of the two-GPU run shows the last residual:
with sixteen local sequences (thirty-two gathered) the kernel streams
its eight touched experts at 0.50 of the calibrated rate, below the 0.70
the constant says. The decode step at batch 32 stays 27% low, and the
envelope says so. Rows per touched expert seem to be the variable — 1
row 0.9, 3–8 rows 0.65, 16 rows 0.5 — but three points are not a
mechanism, and the kernel-profile file carries them for whoever fits
the fourth.

## After

```
                     TTFT                  TPOT
   b  prompt   meas    live     r    meas    live     r
   1     512    122     111   0.91   13.06   12.90  0.99
   1    2048    407     432   1.06   12.73   12.96  1.02
   1    8192   1871    1775   0.95   13.21   13.17  1.00
   8     512    415     427   1.03   20.23   18.26  0.90
   8    2048   1795    1709   0.95   19.95   18.48  0.93
   8    8192   7894    7086   0.90   22.78   19.36  0.85
  32     512   1768    1693   0.96   37.19   27.32  0.73
  32    2048   7201    6823   0.95   39.05   28.19  0.72
```

TTFT 0.96 geometric (0.90–1.06); TPOT 0.99–1.02 at batch 1, 0.85–0.93
at batch 8, 0.72–0.73 at batch 32. Everything between the frozen and
the live columns is a measurement — a link benchmark, a kernel read, a
router histogram, a kernel profile — and none of it is a fit to these
cells.

> **Note (milestone 63).** The two streaming constants became curves in
> rows per touched expert, derived from unprofiled step times at batch 1
> to 64 rather than from profiler traces — which, milestone 63 found, read
> these kernels 7–15% long. The router's counts moved onto the preset as
> a measured table. With both, the single-GPU decode grids land at
> 0.93–1.11 (bf16) and 0.95–1.09 (MXFP4) over batch 1–64, and this
> two-GPU grid at 0.93–0.96 / 0.84–0.92 / 0.71–0.73 for batch 1 / 8 / 32
> — the batch-32 rank is still the open item.
>
> **Erratum (milestone 65).** The router histogram and the kernel profile
> in this post were both right about what they measured and wrong as
> inputs. The histogram counted experts at the prompt's last position;
> decode routing spreads wider over the steps a TPOT covers (13.1 at batch
> 8, 21.3 at batch 32 on the random-token protocol). Divided by the true
> counts, the fused-MoE kernel streams at the calibrated DRAM rate — not
> 0.66–0.70 — whenever a launch's tokens do not exceed the rank's
> experts. The batch-32 rank's "0.50" is vLLM's default Triton config
> switching to 64-row blocks because the rank gathers 32 tokens but holds
> only 16 experts (0.70 of the rate, measured on one GPU between batch 32
> and 33). With both, this grid lands at 0.84–0.98 on decode, batch 32 at
> 0.91–0.93; the idle replica's dummy tokens route to their own four
> experts, which closes batch 1 (0.95–0.98).

## Two notes for whoever runs this next

**A GPU that says "busy or unavailable" with nothing on it is hung.**
Between the frozen commit and the run, GPU 0 stopped accepting CUDA
contexts while GPU 1 worked; `nvidia-smi -q` showed 100% utilization,
120 W, no process, and `GPU Recovery Action: Reset`. Only a reset
clears it, and a reset needs the display session off the device and
root — and, on this box, a kernel module reload, because an unattended
upgrade had replaced the userland and the GSP firmware under the loaded
module two days earlier. Everything two-GPU in this post ran after that
reload; the driver version is in the measurement file.

**The profiler's wall time is not a wall time.** Under CUPTI a
430-kernel decode step reads 50% longer than it runs; the kernel
durations are fine. Kernel sums go in the tables above; step times come
from the unprofiled runs.

## What this validates, and what it does not

Validated and pinned: expert parallelism at width two on a PCIe pair,
bf16, one model, one engine — the dispatch pattern, the lockstep
padding, the link's per-collective rates, the direction of the
two-GPU trade-off in both phases, and the decode decomposition into a
measured routing count and a measured kernel efficiency.

Not validated: DeepEP-class all-to-all kernels (the other `ep_dispatch`
is still the milestone-11 pricing, unmeasured), expert parallelism over
NVLink or beyond two ranks, the fused-MoE kernel's efficiency beyond
eight rows per expert, and the router on real text — 2.15 is random
tokens, and a chat workload with a hot domain will be more concentrated
still.

## Errata

Milestone 61's post carries a note: its routing skew was an effective
constant that absorbed the decode kernel's inefficiency; the router's
own number is 2.15, and the kernel constants are now separate. Milestone
60's post carries a second note: the Marlin kernel's streaming
efficiency, not only the routing, was behind its batch-8/32 overshoot.

## Exercises

1. The profile gives the fused-MoE kernel's streaming efficiency at 1,
   3, 8 and 16 rows per expert. Measure 2, 4, 32 and 64 with
   `fused_experts` directly, plot against rows per expert, and decide
   whether it is a curve or two regimes.
2. Set `VLLM_ALL2ALL_BACKEND=naive` and re-run one prefill cell. The
   naive path broadcasts per rank and combines with an all-reduce; does
   the `all_to_all` pricing or the `allgather` pricing come closer?
3. Run the router histogram on a real corpus. If the distinct-expert
   curve moves, so does every decode number in this post — by how much?
