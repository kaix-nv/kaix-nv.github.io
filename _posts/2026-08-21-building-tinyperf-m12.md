---
layout: post
math: true
title: "Building tinyperf M12: Training — backward as a pass, pipelines as algebra"
date: 2026-08-21 23:15:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "A 40-line graph pass appends the backward pass (the 6N rule emerges from shape arithmetic), a two-line formula prices pipeline bubbles — and GPT-3 on 64 H100s lands at 33-60% MFU."
---

*Milestone 12 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `add_backward` in
`tinyperf/passes.py`, `tinyperf/training.py` · Demo:
`examples/10_training.py`.*

Eleven milestones modeled inference. Training needs two additions — and
neither one touches the model definitions.

## The backward pass is a graph transformation

Model code describes the forward pass; a ~40-line pass appends the
backward, exactly the way production graph IRs implement training support.
Walking forward ops in reverse:

- **gemm** → two more GEMMs of equal FLOPs: dgrad (`dA = dC @ B^T`) and
  wgrad (`dB = A^T @ dC`). This *is* the classic 1:2 fwd:bwd ratio, and
  with it the famous **6N FLOPs/token** estimate (2N forward + 4N
  backward) emerges from shape arithmetic — the test suite pins the ratio
  at exactly 3.00 and measures 42.9 GF/token for LLaMA2-7B against
  6N = 40.4 (attention accounts for the excess).
- **fmha** → one backward fused-attention op with doubled query rows:
  ~2x forward cost (dQ/dK/dV plus recompute), matching published
  FlashAttention backward ratios.
- **comm** → mirrored collectives (Megatron TP: forward's identity is
  backward's all-reduce and vice versa — net, a 1:1 mirror).
- **rw** → mirrored bandwidth ops at 1.5x bytes (backward reads the saved
  activation *and* the incoming gradient).
- **optimizer** → one final bandwidth-bound op: mixed-precision Adam
  touches ~26 bytes per local parameter (fp32 master + two moments read
  and written, fp16 grad read, fp16 weight written).

Because the pass operates on op *types*, everything earlier composes for
free: MoE training gets mirrored all-to-alls, fused attention gets its
fused backward, TP training gets its collective pairs — no new model code.

## Pipeline parallelism is a schedule, not a graph

PP shards *layers* across `pp` stages; a step pushes `mb` microbatches
through. The pipeline is full for `mb` slots and filling/draining for
`pp - 1`, so to first order (GPipe and 1F1B alike):

```
step_time  = (mb + pp - 1) x stage_time_per_microbatch
efficiency = mb / (mb + pp - 1)
```

```
  pp     mb=1     mb=4    mb=16    mb=64   mb=256
   2      50%      80%      94%      98%     100%
   8      12%      36%      70%      90%      97%
  16       6%      21%      52%      81%      94%
```

All of PP economics is in that ratio — and in its tension with the rest of
the model: more microbatches shrink the bubble but shrink the GEMMs too,
sliding them toward the launch-overhead floor (and, at scale, toward
milestone 10's starvation regime). The formula alone can't see that
tradeoff; the formula *composed with the graph model* can.

## GPT-3 175B on 64 H100s, priced in milliseconds

Composing everything — the public GPT-3 config (175.2B params from our
preset), tp=8 within nodes, pp=8 across them, microbatch b=1 s=2048:

```
 microbatches   step ms  tokens/step    tok/s  pipeline eff   MFU
            8     847.7       16,384   19,328           53%   33%
           16    1288.4       32,768   25,433           70%   44%
           32    2170.0       65,536   30,202           82%   52%
           64    3933.0      131,072   33,326           90%   57%
          128    7459.1      262,144   35,144           95%   60%
```

MFU in the 30–60% band, rising with microbatch count, saturating as the
bubble closes — the shape (and magnitude) of every published large-model
training study, from a model that runs in milliseconds. As always, PROJ
is the optimistic tier: real runs pay for data loading, checkpointing,
stragglers, and imperfect overlap; a training-measurement calibration
(milestone 9's pipeline, pointed at step times instead of GEMMs) is the
natural next turn of the crank.

## What we still don't model

Data parallelism and its gradient all-reduce (ZeRO sharding included),
activation memory and recompute (the capacity model only knows inference),
and interleaved-1F1B's smaller bubbles. Each is an exercise below — none
requires new machinery, only new composition.

## Exercises

1. Add DP: gradient all-reduce of local params per step, overlappable
   with backward. At what DP degree does a 400 GB/s inter-node link
   become the binder?
2. Extend `capacity.py` for training: weights + grads + optimizer state +
   activations(mb, recompute). Re-derive the famous "175B needs hundreds
   of GPUs just to *fit*" from first principles.
3. Model activation recompute: forward runs twice, activation memory
   drops ~sqrt(L). Find where recompute beats a bigger pp.
4. Interleaved 1F1B: v virtual stages per GPU shrink the bubble to
   (pp-1)/v; charge the extra p2p sends and find v's sweet spot.

*Next: milestone 13 — the event-driven serving simulator, where requests
arrive, batches form continuously, and tinyperf's per-step latencies
become the delay model of a system.*
