---
layout: post
math: true
title: "Building tinyperf M46: Attention data parallelism: what batch means on eight GPUs"
date: 2026-09-08 12:30:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "When experts span more GPUs than a tensor-parallel group, the extra GPUs are data-parallel attention replicas serving their own share of the batch. Making that the model's semantics fixed a quiet inconsistency, a MoE-under-TP bug, and two columns of the milestone-44 table."
---

*Milestone 46 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `dp` in `nets/transformer.py`, `capacity.py`, `serving.py`, `sweep.py` · Example: `examples/29_dp_attention.py`.*

Milestone 11 gave the model expert parallelism, and from then on a layout
like `tp=4, ep=8` was legal. Eight GPUs hold the experts; four form a
tensor-parallel group. Nobody asked what the other four were doing during
attention. The graph priced them as four more copies of the same TP group
computing attention over the same batch — which is not a deployment
anyone runs, and the accounting was quietly inconsistent: the expert side
already spread the global batch's token-expert assignments over all eight
ranks, while the attention side charged every rank for all of it.

## The layout engines actually run

When experts span more GPUs than a TP group, the extra GPUs are
**data-parallel replicas of the attention**. Each replica serves its own
requests — `batch/dp` of them — and the replicas meet only inside the
expert all-to-all. vLLM spells it `--tensor-parallel-size 4
--data-parallel-size 2 --enable-expert-parallel`; the DeepSeek serving
papers call it DP attention. This milestone makes it the model's
semantics:

- `batch` is the **global** batch across the deployment.
- Attention, norms, the dense first layers and the head run on
  `ceil(batch/dp)` sequences per rank, with weights sliced `1/tp`.
- Experts either **span all attention ranks** (`ep = tp × dp`, whole
  experts on ranks, the global batch dispatched over `ep`) or are
  **TP-sharded like a dense FFN** (`ep = 1`, every rank holds a `1/tp`
  slice of every expert, runs its own rank's tokens, and all-reduces).
- Any other `ep` is refused with a message, and `ep > tp` with no `dp`
  given is read as `dp = ep / tp`, the only physical meaning it has.

The second bullet on the expert side is a fix as much as a feature. Under
pure tensor parallelism the MoE block used to ignore `tp` entirely — full
experts, all rows, no reduction — so a Mixtral on `tp=2` paid the same
expert time as on one GPU. It now slices, runs half, and all-reduces, and
its expert weights per GPU halve as they should.

Capacity follows the same rule: a rank holds KV and state for its own
`batch/dp` sequences, and `max_batch` returns the global batch the
deployment holds, `dp` times what one replica fits.

## Eight B200s, one MoE, four layouts

gpt-oss-120B, global decode batch 64 at 4k context, prefill of eight
2048-token prompts:

```
  layout          attn b/rank  TPOT ms  comm %  TTFT ms  GB/GPU  max b
  tp8 (MoE-TP)             64     8.41   18.1%       44      32   6578
  tp4 dp2 ep8              32     7.86   16.3%       32      33   6634
  tp2 dp4 ep8              16     7.74   14.3%       29      33   6644
  tp1 dp8 ep8               8     7.69   11.5%       27      34   6608
```

Every row is the same eight GPUs holding the same model, and the memory
column says so. What moves is where the collectives are. Pure TP pays
two all-reduces per layer over eight ranks and slices attention thin;
`dp8` pays one all-to-all pair per layer and runs each attention rank on
eight sequences with unsliced heads. Decode improves 9%, prefill 38%, and
the comm share falls from 18% to 12%. On an NVLink node the layout choice
is worth less than the model's own ~10% error bar for decode — and quite
a lot for TTFT.

## Erratum for milestone 44

The Kimi-K3 layout table in milestone 44 used the old accounting, and two
of its columns were artifacts. Under the corrected semantics the three
64-GPU layouts (`tp8 ep64 pp1`, `tp8 ep32 pp2`, `tp8 ep16 pp4`) hold
**the same per-GPU footprint** — 50, 47 and 47 GB — and the same global
batch, about 3300 at 16k context. Deeper pipelines do not buy memory at a
fixed GPU count; they cannot, it is the same model on the same GPUs. The
decode column also changes (21.3, 17.5, 15.6 ms rather than 39.1, 24.8,
17.9): the old numbers had every rank computing attention for the whole
batch. The gain that remains with depth is real and has two sources,
micro-batching across stages and the narrower expert all-to-all (fewer
inter-node hops as `ep` shrinks from 64 to 16). The TTFT column was
unaffected — a single prompt is one sequence on one replica either way —
and the conclusion that narrow EP hurts single-request TTFT stands. The
milestone 44 post carries a note pointing here.

## What this does not model

Load imbalance across replicas: `batch/dp` assumes the router or the
serving front-end spreads requests evenly, and a hot replica gates the
step the same way a hot expert rank does in milestone 43. Attention DP
with `ep` between `1` and `tp × dp` (experts both sharded and
distributed, the "tensor-expert" layouts) is refused rather than
approximated. And nothing here has met silicon: the M45 two-GPU runs
were dense, so the DP-attention and MoE-TP paths inherit the single-GPU
and TP error bars by construction, not by measurement.

## Exercises

1. Add a `dp_imbalance` knob mirroring `moe_imbalance` and find the skew
   at which `tp1 dp8` loses to `tp2 dp4`.
2. Price the tensor-expert middle ground: `tp=4, ep=2` with each expert
   sliced across two ranks and dispatched across two groups. Where does
   it land in the table above?
3. Re-run `examples/28` under the new semantics and check the M44
   erratum's numbers.
