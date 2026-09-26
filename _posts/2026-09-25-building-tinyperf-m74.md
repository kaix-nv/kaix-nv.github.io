---
layout: post
math: true
title: "Building tinyperf M74: The KV pool from first principles: vLLM's start-up accounting, priced, and a MoE workspace that is never freed"
date: 2026-09-25 19:00:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "vLLM sizes its KV cache at start from a handful of terms: CUDA's total, the weights, a profiling pass's peak, the memory outside torch. Priced term by term, eight held-out dense configurations on four models land within 1%, where the old estimate read 0.83-1.76. The MoE missed twice, and the reason was a scratch buffer vLLM never frees."
---

*Milestone 74 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `vllm_kv_pool` in `capacity.py`, `simulate(gpu_memory_utilization=)` in `serving.py`, `tools/measure_kv_pool.py` · Data: `data/validation/comparison_rtx_a6000_kv_pool.txt`, `vllm_kv_pool_rtx_a6000_m74.json`.*

Milestone 72 needed the size of vLLM's KV cache and read it off the
server's log. tinyperf's capacity model said 181,878 tokens where the
engine allocated 188,944. It also ignored the memory-utilization
setting, so at 0.5 it gave the same answer. This milestone derives the
pool.

## How vLLM sizes the pool

At start, vLLM 0.15.1 does five things:

1. It asks CUDA for the device's total memory and takes
   `gpu_memory_utilization` of it.
2. It loads the weights.
3. It runs a profiling pass and records torch's peak. The pass is one
   forward over `max_num_batched_tokens` tokens, then a sampler over
   `max_num_seqs` rows.
4. It measures the memory held outside torch: the CUDA libraries'
   workspaces.
5. It gives the rest to the KV cache, in 16-token blocks.

With `VLLM_LOGGING_LEVEL=DEBUG` it logs every term.
`tools/measure_kv_pool.py` starts a server, reads the terms, and stops
it. For Qwen3-8B at 0.9, 2,048 tokens and 256 sequences:

```
  term                         vLLM    tinyperf (GiB)
  requested, 0.9 x 47.40       42.66     42.66
  weights                      15.27     15.26
  profiling peak                1.40      1.40
  non-torch                     0.04      0.04
  KV cache                     25.95     25.97    188,944 vs 189,072 tokens
```

`capacity.vllm_kv_pool` prices each line.

## What the capacity model had wrong

- **Units.** It read the A6000's "48 GB" as 48 × 10⁹ bytes, or 44.7
  GiB. CUDA reports 47.40 GiB, so the 48 is roughly GiB less the
  driver's share.
- **The engine's own terms.** It knew neither the profiling peak nor the
  utilization setting.
- **The embedding and LM head at tensor parallel 2.** It counted both
  whole on every rank. vLLM shards them by vocabulary, so each rank
  holds half. For Qwen3-8B at tp=2 the model said 8.79 GiB of weights
  per rank, where vLLM loads 7.64. This one reaches back: nine earlier
  posts quote capacity figures it inflated (errata below).
- **Tied embeddings.** Qwen3-0.6B and 4B use one matrix as both
  embedding and head, and the model had counted it twice.

## The profiling peak

The peak is the larger of the two passes.

- **The sampler.** It computes fp32 logits for every row over the whole
  vocabulary, then does a full sort for top-p, with its index buffers.
  That costs 38.6 bytes per logit, fitted on Qwen3-8B at 128–512 rows.
  256 rows of Qwen3's 151,936-token vocabulary take 1.40 GiB, the room
  for about 10,000 tokens of Qwen3-8B's KV.
- **The forward.** It costs 3 × (FFN width + hidden) values per token.
  For Qwen3-8B it sets the peak above about 3,800 tokens at 64
  sequences. At 8,192, 10,240, 12,288 and 16,384 tokens the model says
  0.75, 0.94, 1.12 and 1.50 GiB, and vLLM logged 0.76, 0.95, 1.13 and
  1.51.

One thing the model does not price. At 64 sequences the peak reads
0.47–0.50 GiB on every model, where the sampler term is 0.35. At 32
sequences the floor is absent. It costs 0.5–1.1% of the pool.

## The first start is different

The first time vLLM starts a configuration, `torch.compile` runs inside
the profiling pass, and the peak rises by 0.4–0.7 GiB. For Qwen3-8B at
4,096 tokens and 64 sequences, the first start read 1.22 GiB and the
second 0.50. The pool you get therefore depends on whether the compile
cache is warm. Qwen3-14B's first start at 0.9 allocated 89,232 tokens
and its second 95,808.

So every configuration was started twice, and the criterion applies to
the second, compile-cached start.

## Frozen, then measured

The model has five inputs: the model, the utilization, the token budget,
the sequence count and tp. None of the held-out models had been started
on this machine. The predictions were pushed before any start, with the
old estimate beside them. The criterion was the pool within 2%:

```
  model           util  tokens  seqs  tp    measured       M74     r      old     r
  Qwen3-0.6B       0.9    2048   256   1     375,552   375,520  1.000   363,567  0.97
  Qwen3-0.6B       0.5    8192    64   1     206,576   207,840  1.006   363,567  1.76
  Qwen3-4B         0.9    2048   128   1     250,192   250,688  1.002   233,137  0.93
  Qwen3-4B         0.8   16384   256   1     210,160   211,088  1.004   233,137  1.11
  Qwen3-14B        0.9    2048    64   1      95,808    96,736  1.010    83,399  0.87
  Qwen3-14B       0.95    4096   512   1      96,096    96,240  1.002    83,399  0.87
  Qwen3-14B        0.9    2048   256   2     358,272   360,144  1.005   328,079  0.92
  Qwen3-30B-A3B    0.9    2048    64   2     296,432   302,448  1.020   245,071  0.83
  Qwen3-30B-A3B   0.85    8192   256   2     215,840   227,760  1.055   245,071  1.14
  Qwen3-8B         0.8    8192   128   2     429,424   430,528  1.003   457,965  1.07
```

All eight dense configurations are within 1%, across a 23× range of
model size and both tensor-parallel widths. The weights agree within 1%.
The old estimate reads 0.83–1.76.

Both misses are the MoE. Its profiling peak read 0.60 and 1.92 GiB where
the model had 0.35 and 1.40.

One process note: the MoE's first start at 0.9 timed out loading 61 GB
from a cold disk, so the next start was the one that compiled. Two
further starts, both compile-cached, gave 296,432 tokens, and the row
uses those.

## A workspace that is never freed

vLLM's fused-MoE layer takes two scratch buffers from a workspace
manager: tokens × top_k × max(n/2, hidden) and tokens × top_k × max(n,
hidden), where n is the rank's gate-and-up width. The manager keeps one
buffer, grows it to the largest request and never frees it. So the
workspace is still held when the sampler peaks, and it adds to the peak
rather than competing with it. That is 0.125 GiB at 2,048 tokens and
0.5 at 8,192. With this term the two cells read 1.011 and 1.005.

The term was added after seeing the misses. So it was frozen in turn,
before three new configurations were started:

```
  Qwen3-30B-A3B, tp=2   measured   frozen      r     peak measured / model
  0.9, 16384, 128        271,072   272,960  1.007        1.76 / 1.70
  0.9,  4096, 512        243,312   243,536  1.001        3.03 / 3.05
  0.8, 32768, 256        148,368   151,408  1.0205       2.64 / 2.52
```

Two are within 2%. The 32k token budget misses by 0.05%. Its peak is
0.12 GiB over the model, which is the size of the forward's hidden
states at that budget (32,768 × 2,048 × 2 bytes), perhaps still held
when the sampler peaks. That term is not priced; exercise 2 asks
whether it should be. The model as first frozen reads 1.02–1.19 on these
three.

The GPU process log, sampled every 2 seconds through every start, shows
only the probes' own servers.

## What it changes

`simulate` now derives its pool as vLLM does whenever none is given.
The derivation covers one pipeline stage, with no attention data
parallelism, context parallelism, offload or sparsity. Milestone 72's
held-out sweeps, re-priced on the derived pools of 189,072 and 51,008
tokens, read TTFT p50 0.97–1.01 and TPOT 0.99–1.02.

`llm_memory` and `max_batch` keep their 90%-of-HBM budget, but the
vocabulary-parallel embedding and head apply to them too.

## Errata

- **M8, M11, M14, M44, M46, M49, M51, M53, M58:** the embedding and head
  were counted whole on every tensor-parallel rank. LLaMA3-70B at tp=8
  holds 3.7 GB less per GPU than they said. Three findings shift:
  - M8's tp=2 row holds 2 sequences at 4k, not 0.
  - M14 finds 35 feasible configurations, not 31.
  - M49's 2:4 gain is 7.4 GB and 14% more 8k sequences.

  Each post except M11 and M58, which move by 1–4%, carries a note.
- **M72:** CUDA reports 47.40 GiB on this GPU, not 47.53.
- **M19, M20:** their demos now run on the derived pool. Where the pool
  binds, the tables move by up to 28%, and no claim changes.

## Exercises

1. At 64 sequences the profiling peak reads 0.47–0.50 GiB on every
   model, and at 32 it follows the forward. Find the allocation with
   `torch.cuda.memory._record_memory_history` around vLLM's profiling
   pass.
2. Price the forward's hidden states as held through the sampler pass
   (tokens × hidden × 2 bytes, added to the peak). It closes the 32k
   miss. What does it do to the 512-row cells, which it would put about
   0.03 GiB high, and to the 38.6 bytes per logit fitted without it?
3. The fraction of the nameplate memory that CUDA reports was measured
   on one A6000. Read it on an H100 or a B200 and check whether it is a
   constant.
