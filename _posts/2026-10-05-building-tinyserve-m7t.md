---
layout: post
math: true
title: "Building tinyserve M7t: Profile the state-carrying KDA tail"
date: 2026-10-05 08:18:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Trace occupancy, state tiles, and register spills in the state-carrying KDA tail."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m7t-kda-tail-profile.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M7s — Replace the Python expert loop with indexed grouped GEMMs]({% include tinyserve-post-url.html slug="building-tinyserve-m7s" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7s-native-indexed-moe.md" %}) · Next: [M7u — When a faster kernel is still the wrong default]({% include tinyserve-post-url.html slug="building-tinyserve-m7u" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7u-kda-state-tile.md" %})

M7s removed Kimi's long-prompt expert loop. At 2,048 prompt tokens, the model
interval fell to `112.37 ms`, routed MoE fell to `21.03 ms`, and KDA became the
largest measured module at `64.73 ms`. The chunkwise KDA solve alone took
`46.27 ms` across twelve KDA layers.

That result identifies a module, not an optimization. M7t changes no serving
math. It profiles the two native kernels inside one real-shape KDA solve and
asks a narrower question: is the remaining time caused by pair construction,
the ordered state carry, arithmetic throughput, or memory traffic?

[![A 2,048-token prompt creates 32 KDA chunks. Pair construction has tens of
thousands of independent blocks, while the ordered tail has only 64 large
state-tile blocks. The profile selects a smaller value tile as the next
experiment.](/assets/tinyserve/m7t-kda-tail-profile.svg)](/assets/tinyserve/m7t-kda-tail-profile.svg)

## Follow one 2,048-token prompt

The tiny Kimi-K3 checkpoint has twelve KDA layers. In one layer, the promoted
M7q path sees:

```text
B = 1 request                  H = 8 heads
T = 2,048 prompt tokens        K = 128 key channels
C = 64 tokens per chunk        V = 128 value channels
N = T / C = 32 chunks
```

Pair construction and state carry have different dependency graphs.

`_kda_pair_kernel` builds a lower-triangular correction matrix `L[n]` and a
query-pair matrix `P[n]` for each chunk. Chunk 17 does not need chunk 16's
matrix, so every target row and 16-source tile can be launched independently.
At `T=2048`, that is:

$$
B \times H \times N \times C \times (C / 16)
= 1 \times 8 \times 32 \times 64 \times 4
= 65{,}536\ \text{programs}.
$$

`_kda_multichunk_tail_kernel` consumes those matrices, solves correction rows,
emits outputs, and updates recurrent state. Value columns are independent, so
one program owns 16 of them. Chunks are not independent: chunk `n+1` must start
from the state produced by chunk `n`. The current grid is therefore:

$$
B \times H \times (V / 16)
= 1 \times 8 \times 8
= 64\ \text{programs}.
$$

Each of those 64 programs owns a logical `[K,Vtile] = [128,16]` FP32 state
tile and carries it through all 32 chunks. This avoids 32 host submissions, but
it also creates one large, long-lived working set per program.

The word “tail” is easy to misread here. The kernel is named *tail* because it
does the work after pair construction. A GPU *tail effect* means a few
straggling blocks run after others finish. M7t finds static underfill instead:
all 64 blocks do equal work, but an A6000 has 84 SMs, so 20 SMs receive no
block at all.

## Capture the exact dispatch without loading the model

The NCU harness imports the unchanged M7s kernel and submits one
`fused_kda_multichunk()` call at 512 and 2,048 tokens. It uses deterministic
FP32 tensors with the real shapes and the same channel-strided query/key layout
produced by causal convolution. No checkpoint weights are needed: this kernel's
dispatch and loop bounds depend on shape, strides, dtype, and device rather
than the tensor values.

Each Triton specialization was warmed once before collection. Nsight Compute
then selected one kernel name and one launch. Full and source-counter reports
were collected separately, so pair and tail metrics are never averaged.

| kernel | tokens | NCU duration | grid | waves/SM | registers/thread | achieved occupancy | local loads / stores |
|---|---:|---:|---:|---:|---:|---:|---:|
| tail | 512 | 1.414 ms | 64 | 0.76 | 255 | 16.65% | 1.75M / 0.50M |
| tail | 2,048 | 5.647 ms | 64 | 0.76 | 255 | 16.43% | 6.98M / 1.97M |
| pair | 512 | 0.215 ms | 16,384 | 21.67 | 56 | 72.18% | 0 / 0 |
| pair | 2,048 | 0.872 ms | 65,536 | 86.69 | 56 | 72.75% | 0 / 0 |

NCU replays kernels and controls clocks, so these durations are diagnostic, not
serving latency. Their scaling is still informative. Four times as many chunks
produce almost exactly four times the instructions, bytes, and duration for
both kernels. The same tail pathology repeats for every chunk; there is no
one-time setup cost hiding the result.

The earlier lightweight 2,048-token operator trace provides a separate timing
boundary: pair construction took `0.310 ms`, while the tail took `4.383 ms`.
The exact durations differ under NCU, but both measurements agree about which
kernel dominates.

## The tail is resource-bound and spilling

The current tail launches one 256-thread block per state value tile. Nsight
Compute reports:

```text
grid                           64 blocks on 84 SMs
registers                     255 per thread
dynamic shared memory         36.86 KiB per block
theoretical occupancy         16.67%
achieved occupancy            16.43%
eligible warps                0.19 per scheduler per cycle
issue slots busy              16.77%
```

Registers and shared memory each limit the launch to one eight-warp block per
SM. Even on an occupied SM, only about one fifth of one warp is ready to issue
per scheduler per cycle. NCU estimates a `23.81%` opportunity from filling the
currently empty SMs and a larger `62.99%` local opportunity from occupancy.
Those estimates overlap; adding them would be meaningless.

The spill evidence is more direct:

```text
T=2048 local-load instructions       6,980,096
T=2048 local-store instructions      1,974,272
local sector utilization             1 byte out of 32
DRAM throughput                      12.92% of peak
```

“Local” memory is not an on-chip scratchpad. It is compiler-managed per-thread
storage backed by the memory hierarchy. A kernel at the 255-register ceiling
with millions of local instructions has more live values than registers can
hold. Low DRAM utilization rules out a saturated-memory explanation; the
kernel lacks enough resident and ready warps to hide its spills and reductions.

The source counters point to the causal solve. At 2,048 tokens, the largest
kernel-source long-scoreboard locations are:

- `tinyserve/kimi_kernels.py:814`, the `lower_row × solved_rows` reduction:
  15,384 samples;
- line 831, the `pair_row × solved_rows` output correction: 10,668 samples;
- lines 773–774, the state projection used for base output; and
- line 877, the outgoing-state update.

The complete sampled-state mix is `28.40%` short scoreboard, `20.85%` barrier,
`18.08%` long scoreboard, and `13.45%` wait. Shared-memory accesses also show
3.2-way load and 3.5-way store bank conflicts. The profile does not isolate one
magic instruction; it shows a large live state/solve tile stressing registers,
shared memory, and synchronization together.

## Pair construction has a real but secondary problem

The pair kernel looks almost opposite:

```text
grid                           65,536 blocks
registers                     56 per thread
achieved occupancy            72.75%
local loads / stores          0 / 0
SM throughput                 59.15% of peak
```

Its main problem is access layout. `44.93%` of sampled warp states are MIO
throttle. The source-key load at `tinyserve/kimi_kernels.py:287` accounts for
13,419 MIO-throttle samples, and NCU labels 67% of global sectors excessive.
Each sector carries only about `10.3` useful bytes out of 32.

That could justify a later coalescing experiment. It is not the first move:
pair construction is only `0.872 ms` beside a `5.647 ms` tail in NCU, and
`0.310 ms` beside `4.383 ms` in the lightweight trace. Optimizing the smaller
stage first would ignore the moved bottleneck.

## M7u: split value columns more finely

The profile selects one bounded experiment: reduce the tail's value tile from
16 columns to 8 and retune its warp count.

```text
current                         candidate to measure
state tile [128,16]             state tile [128,8]
64 programs                     128 programs
32 chunks in causal order       same 32 chunks in causal order
FP32 equations                  same FP32 equations
```

This choice attacks both strongest facts at once. It halves the width of the
long-lived state, correction, and output rows, while doubling the grid enough
to cover all 84 SMs. It does **not** parallelize causally dependent chunks and
does not change accumulation precision.

The tradeoff is falsifiable. Smaller value tiles reread Q, K, decay, `L`, and
`P` more often. Reduced spilling and higher occupancy must outweigh that extra
traffic. M7t therefore does not edit `BLOCK_V` yet.

M7u is promoted only if it:

1. keeps the PyTorch oracle bounds and full-model token decisions;
2. removes or materially reduces local-memory instructions;
3. improves the isolated 2,048-token tail by at least 10%;
4. lowers full-model 2,048-token latency by at least 5%;
5. regresses 32- and 128-token full-model latency by no more than 5%;
6. adds no meaningful peak-memory growth; and
7. reruns the same-checkpoint external calibration after the serving change.

If smaller tiles do not pass, keep M7s and record the candidate as rejected.
The next investigation would then separate the forward substitution or change
its shared-memory layout—not silently weaken FP32 math to activate tensor
cores.

## Inspect the profile locally

The **M7t: KDA NCU harness** launch configuration runs the exact 512-token
harness without NCU so the Python dispatch is easy to step through. Break at:

1. `fused_kda_multichunk()` — inspect `chunks=8`, `pair_grid`, and `tail_grid`;
2. `_kda_pair_kernel` launch — map rows and 16-source tiles to `L/P`;
3. `_kda_multichunk_tail_kernel` launch — map eight value tiles per head; and
4. the harness's final shapes and finite check.

For device metrics, open the ignored local reports with `ncu-ui` or regenerate
the compact analysis:

```bash
PYTHONPATH=/usr/local/cuda-12.8/nsight-compute-2025.1.0/extras/python \
  .venv/bin/python \
  profile/kda-multichunk-tail-m7t-a6000/analysis/analyze_reports.py
```

The full technical report at
`profile/kda-multichunk-tail-m7t-a6000/REPORT.md` records the collection
boundary, six-dimension diagnosis, commands, and caveats. The compact [M7t evidence
manifest](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7t-kda-tail-profile-a6000-2026-09-12.json) retains the
decision metrics and report hashes.

M7t changes instrumentation and documentation only. There is no new serving
implementation to compare with external engines; that calibration belongs to
M7u if the smaller tile passes its local gates.

## Takeaway

Moving a loop from Python into one GPU launch is not the end of scheduling.
M7q removed 32 host submissions by carrying recurrent state through every
chunk inside one kernel. M7t shows the price: only 64 large programs, 20 idle
SMs, the register ceiling, low occupancy, and heavy spills.

The next step is now narrow and testable. Split independent value columns more
finely while preserving causal chunk order and FP32 equations, then keep the
change only if the complete model—not just the launch geometry—gets faster.
