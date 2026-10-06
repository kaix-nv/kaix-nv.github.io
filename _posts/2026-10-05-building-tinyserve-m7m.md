---
layout: post
math: true
title: "Building tinyserve M7m: Count the launches before writing the next KDA kernel"
date: 2026-10-05 08:11:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Use attribution and launch counts to decide which KDA boundary is worth changing next."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m7m-kimi-prefill-profile.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M7l — Make KDA prefill chunkwise]({% include tinyserve-post-url.html slug="building-tinyserve-m7l" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7l-chunkwise-kda.md" %}) · Next: [M7n — Tune the KDA chunk before fusing the kernel]({% include tinyserve-post-url.html slug="building-tinyserve-m7n" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7n-kda-chunk64.md" %})

M7l removed Kimi Delta Attention's token-by-token prefill recurrence. That was
a large improvement, but “KDA is faster” does not tell us what to optimize
next. The remaining time could sit in expert routing, projections, causal
convolution, the triangular solve, pair construction, or Python dispatch
between all those operations.

M7m makes no serving-path change. It fixes one exact prompt per condition,
runs the prompt in one prefill forward, attributes the model by module, drills
into KDA, and finally tests the cheapest intervention suggested by the trace.

[![The M7m profile narrows 2,048-token Kimi prefill from 87 percent KDA, to 976
milliseconds in the chunkwise rule, to many small dispatched operators rather
than the triangular solve. An unprofiled control selects chunk size 64 over 32
and 128 for the next milestone.](/assets/tinyserve/m7m-kimi-prefill-profile.svg)](/assets/tinyserve/m7m-kimi-prefill-profile.svg)

## Keep the measurement shape boring

Each condition uses one raw prompt that tokenizes to exactly 32, 128, 512, or
2,048 tokens. The scheduler budget equals the prompt length, so the trace must
contain exactly one action:

```text
[(request 0, prompt token 0, prompt token T)]
```

There is one generated token and therefore no decode forward. This isolates
prefill without changing the model's real cache allocation, scheduler,
sampling, or output path. `KDA chunk size = 32` is held at the M7l default.

The profiler records CUDA events around the whole model, MoE, KDA, MLA,
AttnRes, and the vocabulary head. Inside KDA it separately wraps projections,
causal convolution, Q/K normalization, and the recurrent or chunkwise rule.
Nested times are diagnostic subtotals; they must not be added to their parent.

Two warmups precede five retained measurements. The fixed prompt within each
condition also keeps expert routing constant, rather than confusing input
variation with runtime variation.

## First cut: which module grows?

The table reports median CUDA-event intervals and each module's share of the
model interval:

| prompt | model | KDA | MoE | MLA | AttnRes | other model |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 74.5 ms | 37.4 ms (50.1%) | 19.9 ms (26.7%) | 6.1 ms (8.2%) | 3.1 ms (4.1%) | 7.8 ms (10.5%) |
| 128 | 210.3 ms | 84.6 ms (40.2%) | 107.3 ms (51.0%) | 6.3 ms (3.0%) | 3.3 ms (1.6%) | 8.1 ms (3.9%) |
| 512 | 393.6 ms | 267.2 ms (67.9%) | 109.5 ms (27.8%) | 6.6 ms (1.7%) | 3.4 ms (0.9%) | 8.3 ms (2.1%) |
| 2,048 | 1,148.3 ms | 999.2 ms (87.0%) | 109.5 ms (9.5%) | 22.4 ms (2.0%) | 3.2 ms (0.3%) | 7.4 ms (0.6%) |

MoE is the largest module at 128 tokens, but its absolute time is nearly flat
from 128 through 2,048 tokens for this tiny checkpoint. KDA grows with prompt
length and owns 87% of the long-prompt model interval. A kernel aimed at MLA
or AttnRes cannot materially change that row.

## Second cut: what inside KDA grows?

At 2,048 tokens, the twelve KDA layers contribute these median nested
intervals:

| KDA region | total | share of KDA |
|---|---:|---:|
| chunkwise rule | 976.0 ms | 97.7% |
| input/output projections | 5.0 ms | 0.5% |
| causal convolutions | 4.6 ms | 0.5% |
| Q/K normalization | 3.5 ms | 0.3% |
| unclassified KDA work | 10.2 ms | 1.0% |

The projection and convolution costs that matter in other models are not the
long-prompt problem here. The target is now one function:
`chunkwise_kda()`.

## A CUDA-event interval is not all arithmetic

Wrapping `chunkwise_kda()` with two CUDA events measures the stream interval
from its first submitted work to its final submitted work. The GPU can be idle
inside that interval while Python constructs tensors and submits the next
operator. We therefore run one isolated 2,048-token rule through the PyTorch
operator profiler at the checkpoint's real shape:

```text
B=1, T=2048, H=8, K=128, V=128, C=32
```

The call records `70.7 ms` of self CPU operator time but only `16.8 ms` of
self CUDA operator time. These clocks are not additive, but their gap plus the
call counts reveal a launch-bound implementation:

| operator | calls | self CUDA time |
|---|---:|---:|
| multiply | 1,153 | 3.401 ms |
| reduction | 512 | 3.056 ms |
| batched matrix multiply | 256 | 3.463 ms |
| subtraction | 384 | 1.107 ms |
| in-place addition | 640 | 0.926 ms |
| exponential | 448 | 0.855 ms |
| masked fill | 256 | 0.738 ms |
| triangular solve | 64 | 0.516 ms |

The triangular solve is only 3.1% of self CUDA time. Replacing that operation
alone would optimize the most visually exotic line rather than the measured
bottleneck. Pair construction and state/output updates are fragmented into
hundreds of small elementwise, reduction, and matrix launches.

## Test the cheapest explanation first

If dispatch per chunk is expensive, a larger token chunk should help by
reducing the number of Python iterations. M7m therefore runs an unprofiled,
paired geometry sweep on one loaded model. Each prompt has two warmups and
five repetitions; path order is forward on even repetitions and reversed on
odd repetitions. Every path generates the same first token.

| prompt | recurrent | chunk 32 | chunk 64 | chunk 128 | best |
|---:|---:|---:|---:|---:|---|
| 128 | 261.8 ms | 150.9 ms | 131.4 ms | 141.6 ms | chunk 64 |
| 512 | 724.1 ms | 289.3 ms | 219.0 ms | 224.4 ms | chunk 64 |
| 2,048 | 2,144.5 ms | 780.0 ms | 455.8 ms | 485.1 ms | chunk 64 |

Relative to M7l's 32-token chunk, 64 tokens improves whole-model latency by
12.9%, 24.3%, and 41.6% as the prompt grows. Doubling again to 128 loses at
all three lengths because quadratic within-chunk work starts to outweigh the
saved dispatch.

The memory result has the same shape. At 128 tokens, peak allocation grows
from `475.2 MB` for chunk 32 to `491.9 MB` for chunk 64 and `554.2 MB` for
chunk 128. At 512 tokens, chunk 64 costs `9.8 MB` over chunk 32 while chunk
128 costs `75.3 MB`. At 2,048 tokens all paths share the same `843.9 MB` peak
because other prompt activations dominate.

This control does not prove a fused kernel is unnecessary. It proves that a
one-line geometry change should be exhausted and verified before committing
to a much larger implementation.

## The M7n decision and gate

The next milestone should promote KDA's default kernel chunk from 32 to 64,
not rewrite the triangular solve. The change remains deliberately bounded:

1. update the KDA model, engine, and CLI defaults together;
2. extend equation and poisoned-padding parity across 64-token and partial
   final chunks;
3. compare full-model logits, ordered top candidates, state, and generated
   token decisions against the 32-token oracle boundary;
4. repeat the paired 32/64 control at 32, 128, 512, and 2,048 tokens; and
5. accept only if 512- and 2,048-token latency improve by at least 10%, no
   retained short row regresses by more than 5%, and the memory increase stays
   below 32 MB at 512 tokens.

After promotion, run this profile again. If `chunkwise_kda()` still owns more
than half of 2,048-token model time, the next experiment should fuse the
repeated pair-construction elementwise/reduction chain and state/output
updates. The trace specifically argues against starting with a custom
triangular solver.

## Run and inspect the profile

The **M7m: profile Kimi prefill** launch configuration uses shorter retained
lengths for interactive debugging. Set breakpoints in this order:

1. `profile_kda_operators()` — inspect the real KDA tensor shape;
2. `KimiAttributor.install()` — see which module/function boundaries are
   wrapped without changing the model equation;
3. `run()` — verify the one-action whole-prompt scheduler trace; and
4. `summarize()` — distinguish top-level shares from nested KDA subtotals.

The complete retained command is:

```bash
.venv/bin/python examples/profile_kimi_prefill.py \
  --device cuda:0 --prompt-tokens 32 128 512 2048 \
  --kda-chunk-size 32 --kda-kernel-backend torch \
  --operator-tokens 2048 \
  --warmups 2 --repeats 5 \
  --output .tinyserve-bench/m7m/kimi-prefill-profile.json
```

The geometry control reuses the paired M7l benchmark:

```bash
.venv/bin/python examples/bench_kda.py \
  --device cuda:0 --prompt-tokens 128 512 2048 \
  --chunk-sizes 32 64 128 --micro-lengths 128 512 2048 \
  --kda-kernel-backend torch \
  --micro-iterations 5 --warmups 2 --repeats 5 \
  --output .tinyserve-bench/m7m/kda-geometry.json
```

The compact [M7m evidence](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7m-kimi-prefill-profile-a6000-2026-09-04.json)
retains the medians, operator counts, geometry control, hashes, and
measurement limitations.

## Takeaway

Chunkwise algebra removed serial token dependence, but a Python loop still
submitted dozens of operations per chunk. At 2,048 tokens, KDA owns 87% of
prefill model time even though the useful CUDA operators in one rule call take
only 16.8 ms. The next speedup does not require guessing: halve the number of
chunks with the measured 64-token geometry, verify the numerical and memory
boundary, and profile the result before deciding how much fusion to build.
