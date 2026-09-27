---
layout: post
math: true
title: "Building tinyperf M76: An MoE online: the router touches far fewer experts, and a padded row is not idle"
date: 2026-09-26 17:48:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "The first MoE served under load in the series: Qwen3-30B-A3B across two GPUs. Its router touches half the experts uniform routing assumes, and with that measured the engine-clock decode steps land within 7.5% in all six frozen cells. Online, two of four cells missed, and the engine's step clock showed why: a CUDA-graph padding row is fed a stale token, and the router routes it."
---

*Milestone 76 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `QWEN3_30B_A3B_DECODE_ROUTING` in `nets/transformer.py`, `tools/measure_moe_routing.py` · Data: `data/validation/comparison_qwen3_30b_a3b_rtx_a6000_tp2_moe_serving.txt`, `engine_steps_m76_qwen3_30b_a3b_rtx_a6000.json`.*

The series has checked an MoE on silicon before: gpt-oss-20b, on one GPU,
at fixed batches. It has never served one under load. Qwen3-30B-A3B has
128 experts, of which each token uses 8. Its weights are 61 GB in bf16,
so it needs both GPUs: tensor parallel 2, with the M73 and M74 work under
it.

## What a decode step streams

At small batches an MoE decode step is mostly expert weights, and it
streams only the experts its batch touches. The model had assumed uniform
routing: every expert equally likely. Milestone 65 showed that has to be
measured, per model and per workload.

`tools/measure_moe_routing.py` hooks each layer's router. It had run in
the engine's own process, which doesn't work for a model that needs two
GPUs. It now installs the hooks inside vLLM's tensor-parallel workers, so
it measures the model exactly as it is served. It also samples with the
model's generation config, as a server would (temperature 0.6, top-p
0.95, top-k 20), not greedily.

It measured 256 decode steps of 1024-token random-token prompts, with six
prompt sets per batch. Distinct experts per layer:

```
  batch         2     4     8    16    32    64   128
  measured    12.3  22.7  33.0  43.0  58.5  77.0  93.6
  uniform     15.5  29.1  51.6  82.3 112.7 126.0 128
```

At batch 32 the router touches half the experts uniform routing says.
The six prompt sets also differ a lot from each other: at batch 8, from
26 to 42. Which experts a step touches depends on what its sequences
happen to be saying.

Everything else is held out on this model. The fused-MoE kernel constants
were fitted on gpt-oss-20b, whose experts are 2880 wide and 32 in number.
vLLM has no tuned kernel configuration for either model on this GPU, so
both run its default blocks. The two-GPU link is M73's curve.

## Frozen, then measured

Predictions were pushed before any run, with uniform routing beside them.

**Decode steps on the engine's clock**, B requests decoding side by side:

```
  batch   measured    model (r)    uniform routing (r)
     1     9.27 ms      1.026           1.026
     8    19.04         1.073           1.407
    16    28.27         0.971           1.431
    32    38.86         0.972           1.435
    48    50.05         0.935           1.288
    64    55.20         0.978           1.275
```

All six cells meet the criteria. The fused-MoE constants carry over to
experts a quarter the width, and the router's concentration is the
difference between within 7.5% and 28–44% high.

**Online**, 200 requests of 512–1536 prompt tokens and 128–384 output
tokens, seed 4:

```
  rate   TTFT p50: measured (model / uniform)   TPOT: measured (model / uniform)   verdict
   1           268 ms (1.05 / 1.16)                  30.7 ms (0.89 / 1.54)         miss (TPOT)
   2           370    (1.02 / 1.36)                  65.7    (0.96 / 1.47)         within
   2.5        3159    (0.69 / 2.34)                  90.0    (0.98 / 1.15)         miss (the knee)
   3.5       11380    (1.02 / 1.32)                  91.9    (0.99 / 1.14)         within
```

Two of four cells meet the criteria, and throughput is within 1% at every
rate. Both misses are decode steps priced about 5% cheap online, where
the fixed-batch steps had read 0.97–1.07.

## A padded row is not idle

The engine's step clock shows where those 5% are. vLLM runs decode steps
in CUDA graphs captured at fixed batch sizes (1, 2, 4, 8, 16, 24, 32…). A
step with 9 decodes runs the graph for 16 and pads the other 7 rows.
Split by that:

```
  online decode steps                     count    model / engine
  decodes fill a graph size                5701         1.004
  decodes padded up to one                 7339         0.919
```

On unpadded steps the model is right under load. Padded ones cost about
8% more than it says. On the engine's clock, a step with 9–15 decodes
takes 30.4–31.5 ms, while a full graph of 16 takes 29.5.

A padded row isn't empty. vLLM feeds it whatever token is left in its
input buffer, usually the tail of the last prefill, and the router routes
that token like any other, to 8 experts. The likely reason padded steps
cost more: stale prompt tokens have little in common with what the batch
is decoding, so they bring experts of their own. Neither simple rule
prices them:

- **Identical dummy tokens** (milestone 65's rule for an idle replica)
  would add 8 experts, so a padded step would never cost more than a full
  one.
- **Independent tokens** would add 33 at 9 decodes, several times what the
  clock shows.

The stale rows are consecutive prompt tokens, correlated somewhere in
between. Pricing them needs their routing measured, and they stay
unpriced here.

## Out of step

A second, smaller effect. The routing tables were read off batches whose
sequences all start together, so at any step they are at the same point
in their replies and say similar things. In a server they aren't.
Re-counting the same measurement with each sequence at its own random
step adds 2–5% experts.

Measured on prompts built exactly as `vllm bench` builds them (runs of
consecutive token ids across the whole vocabulary, from another seed),
that table becomes `routing="serving"`. It puts the knee inside its band:
2.5 req/s reads a median TTFT of 0.84 of measured, and its band,
1880–3173 ms, contains the measured 3159. At 1 req/s TPOT reads 0.90; that
cell is padding. `vllm bench`'s prompts route like independent random
ids, within 6% at batch 8 and above, so the prompt style was not the
cause.

The GPU process log shows only the runs' own workers in every window.

## Exercises

1. Price the padded rows. Record the router's choices for a padded step
   (an eager step with the same stale buffer), and fit nothing: how many
   experts do 1–7 stale prompt tokens add beside a real batch?
2. vLLM could zero the padded rows' tokens instead. Then every padded row
   is one identical token, adding 8 experts at most. How much of a
   padded step's cost would that save at batch 9?
3. The six prompt sets at batch 8 range from 26 to 42 experts. How many
   prompt sets does a routing table need before its mean is good to 2%,
   and does that change with real text?
