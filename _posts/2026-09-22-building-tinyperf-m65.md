---
layout: post
math: true
title: "Building tinyperf M65: Real text, and the routing nobody had measured: a decode step touches the experts its tokens pick, all 128 steps of them"
date: 2026-09-22 13:00:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Reading the router inside the serving engine showed that routing spreads as a decode proceeds, so the table the model had used undercounted decode by 25-40% — and three milestones had built on it. With decode routing measured per workload and kernel rates paired launch by launch, predictions frozen before three real-text runs landed within 7%, and a mixed batch decodes 31% slower than chat replies on the same GPU."
---

*Milestone 65 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `GPT_OSS_20B_DECODE_ROUTING` in `nets/transformer.py`, `tools/measure_moe_routing.py` · Data: `data/validation/comparison_gpt_oss_20b_rtx_a6000_realtext.txt`, `routing_decode_gpt_oss_20b.json`, `kernel_pairing_gpt_oss_20b_rtx_a6000.json`.*

Every gpt-oss number since milestone 60 was measured on random-token
prompts, and the router table on the preset came from random tokens too.
A real server decodes prose, code and chat, and a router is a function of
content. This milestone set out to measure the router on real text and
check the decode predictions on real prompts. It found that the table was
wrong for random tokens as well, and that three milestones had built on
that error.

## Reading the router inside the engine

The table on the preset came from milestone 62: forward hooks on a
separate copy of the model, counting the experts chosen at the *last
position of the prompt*. This time the hooks went on the 24 router
modules inside vLLM itself, in eager mode, recording the top four experts
of every token of every forward, and only pure decode steps were kept.

The first number back was wrong by a margin that could not be noise.
Random-token prompts at batch 8, decoded for 32 steps, touched 12.2
experts per step and layer. The table said 9.7. The reason shows in the
per-step record: routing *spreads as the decode proceeds*.

```
  random tokens (64-token prompts), batch 8, decode step    1    2    4    8   16   24   32
  distinct experts (of 32)                                9.0  9.9 11.1 12.5 12.1 13.1 13.8
```

A count at the prompt's last position describes the first decode step at
best. A TPOT measurement averages 128 steps. For chat replies the drift is
extreme: at batch 16 a reply touches 5 experts over its first 8 steps and
18 over steps 33–128, because every reply opens with the same few tokens
and then diverges. So the tables are now measured the way a TPOT is: 128
decode steps, averaged.

## Undoing three milestones

If decode touches 30% more experts than the table said, every constant
fitted against the table absorbed the difference. Milestone 62 read the
fused-MoE kernel's streaming rate as 0.70 of the calibrated DRAM rate.
Milestone 63 turned that into a curve falling to 0.53 at fourteen rows per
expert. Milestone 64 found the kernel 1.4× faster replayed alone than in
the step, and concluded that the step reads bytes its kernels don't need.

Pairing settles it. Under nsys, with the router hooks on, every fused-MoE
launch was matched to its layer's routing in the same step:

```
  bf16 Triton fused MoE          batch 1   8     24    32    33    48    64
  gate-up streaming, GB/s          692    702   675   662   492   492   489
  down streaming, GB/s             681    695   686   680   473   474   473
  Marlin MXFP4 (gate-up / down)    561/486 603/564       544/516
```

The calibrated DRAM rate is 691 GB/s. Inside the engine's step, the bf16
kernel streams the touched experts' weights at exactly that rate, and its
duration is linear in the experts the layer touched: 47 µs per expert,
from 6 experts to 21. There is no pipeline effect. Milestone 64's replay
had captured the second decode step, about 10 experts, while the in-step
averages covered all 32 steps, about 13. Every "extra" byte was an expert
the table had not counted.

## One real effect: the kernel's config switch

One thing in the table above is not routing. Between batch 32 and 33 the
rate drops 30% and stays there. vLLM has no tuned fused-MoE config for
this GPU, so it uses its default: 16-row blocks while a launch's tokens do
not exceed the experts, and 64-row blocks above. Each touched expert's
rows are then padded to 64, and the math stops hiding under the weight
stream. On one GPU the switch sits between batch 32 and 33, against 32
experts. On the two-GPU expert-parallel rank of milestone 62 it sits much
lower: the rank gathers 32 tokens at global batch 32 but holds only 16
experts. That was milestone 62's open item, a rank "streaming at 0.50".

The model now tags every expert GEMM with its launch's token count and the
rank's expert count. It prices bf16 decode at the calibrated rate in the
small-block regime and at 0.70 of it in the big-block one. Marlin gets a
single 0.80.

## The dummy token

The two-GPU runs had one more miss, at global batch 1, where one replica
idles and runs a dummy batch in lockstep. Hooking both replicas' routers
showed the real token and the dummy touching 7.25 distinct experts, where
two random tokens would touch 6.1. A dummy batch is identical tokens that
all pick the same four experts, disjoint from the real token's. So
padding now adds `top_k` experts, not a random token's share. The batch-1
cells move from 0.82–0.85 to 0.95–0.98.

## Checking the existing grids exactly

The earlier grids used seeded random prompts, and greedy decoding is
deterministic, so each timing run's prompts could be regenerated and
their routing measured. With each cell's exact routing, the two paired
kernel rates and nothing fitted, every cell measured since milestone 60
lands here:

```
                                       decode, model / silicon
  bf16, one GPU, batch 1-64 (16 cells)      0.90 – 1.02
  MXFP4, one GPU, batch 1-64 (16 cells)     0.93 – 1.03
  bf16, two GPUs dp2 ep2 (8 cells)          0.84 – 0.98
```

With the random-token table in place of each cell's exact routing, which
is what the envelope tests pin, the ranges are 0.90–1.04, 0.93–1.03 and
0.84–0.98. Batch 1, where routing is exactly four experts, sits at
0.97–1.03 with no MoE constant at all.

## Real text

The workloads come from `tools/build_prompt_pools.py`: WikiText-103
prose, the vLLM Python sources as code, CNN/DailyMail articles wrapped in
the gpt-oss chat template with a summarize request, and a mixed pool that
interleaves the three. Routing, averaged over the 128-step decode:

```
  distinct experts (of 32)   batch 2    8     32    64   128
  chat replies                  6.4  13.2  19.3  21.6  23.8
  random tokens                 6.1  13.1  21.3  24.4  26.5
  prose                         7.1  14.6  22.6  26.5  28.6
  code                          7.1  16.7  25.9  29.0  30.3
  mixed batch                   7.2  18.3  28.0  30.2  31.1
```

Content matters, and mixing matters most: a batch drawn from three
domains touches the union of their experts.

The tables were measured on one set of pool prompts. Predictions for
three timing runs, on different prompts from the same pools, were then
committed before the engine was opened. They list the model with the
workload's table, the same model with the random-token table, and the
pre-M65 model from master.

```
                     decode, model / silicon (batch 1-64 x prompt 512/2048)
                     workload table   random table   pre-M65 model
  bf16  mixed          1.00 – 1.04     0.76 – 1.03     0.81 – 1.03
  bf16  chat           0.96 – 1.02     0.96 – 1.09     1.02 – 1.13
  MXFP4 mixed          1.02 – 1.07     0.84 – 1.03     0.87 – 1.05
```

Prefill landed at 0.93–1.08 except the MXFP4 batch-1 × 512 cell (0.79,
the tiny-prefill miss). The workload effect is large and the model
carries it: at batch 32 × 512 a mixed batch decodes in 53.5 ms and chat
replies in 40.8 ms on the same GPU. The model said 54.8 and 39.1. The
pre-M65 model said 45.3 for both.

A post-hoc replay of the exact prompts each run drew touched almost
exactly the table's counts (18.4 against 18.3 at batch 8 mixed, 19.35
against 19.33 at batch 32 chat). So the remaining error belongs to the
model, not to routing noise. The MXFP4 overshoot of 3–7% is the single
Marlin factor: pairing shows 0.77 to 0.85 depending on batch, and the
model uses 0.80.

## What changes in the model

- `gpt_oss_20b(routing="mixed")` picks a workload's measured decode table
  from `GPT_OSS_20B_DECODE_ROUTING`: random, prose, code, chat or mixed.
  The default is the most spread realistic workload.
- Calibration: `moe_dram_efficiency` 1.0,
  `moe_dram_efficiency_big_block` 0.70, `weight_only_dram_efficiency`
  0.80. The milestone-63 curves are removed.
- The prefill constants of milestones 60 and 61 (0.44 and 0.66 of dense
  math) are unchanged. Prefill touches every expert, and they were fitted
  on prefill.
- `tools/measure_moe_routing.py` measures a routing table for any model
  and workload. `tools/build_prompt_pools.py` builds the pools, and
  `tools/measure_vllm.py --prompt-pool` times them. The builder reproduces
  the pools used here byte for byte, and the routing tool reproduces the
  table cells.

## What this does not model

Workloads outside the table: a real serving mix probably falls between
"chat" and "mixed", but it has to be measured. Decodes much longer than
128 tokens, which may spread further. Routing that differs by layer; the
table averages 24 layers. A tuned fused-MoE config, which moves or
removes the 64-row switch. And, still, gpt-oss-120B, datacenter GPUs,
NVLink, and expert parallelism beyond two ranks.

## Errata

Milestone 64's conclusion is reversed, and its post now opens with the
correction. Milestone 63's curve is withdrawn. Milestone 62's streaming
constants and its batch-32 open item are explained by routing and by the
config switch. Milestones 60 and 61 carry notes: milestone 61's fitted
skew of 1.4 was closer to the truth than milestone 62's correction of it.
The lesson is recorded in the envelope: a routing count must come from the
steps being timed.

## Exercises

1. Run `tools/measure_moe_routing.py` on a chat workload at 512 and 1024
   generated tokens. Does the drift saturate, and where?
2. Tune the fused-MoE Triton config for `E=32, N=2880` on this GPU with
   vLLM's benchmark script and re-pair batch 33. Does the 64-row switch
   move, or disappear?
3. Measure routing per layer rather than pooled. If a few layers carry
   most of the spread, a per-layer table would let the model price
   layer-sharded pipelines correctly.
