---
layout: post
math: true
title: "Building tinyserve M7u: When a faster kernel is still the wrong default"
date: 2026-10-05 08:19:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "A smaller value tile improves an isolated kernel but fails the whole-model promotion gate."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m7u-kda-state-tile.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M7t — Profile the state-carrying KDA tail]({% include tinyserve-post-url.html slug="building-tinyserve-m7t" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7t-kda-tail-profile.md" %}) · Next: [M7v — When panelization increases register spills]({% include tinyserve-post-url.html slug="building-tinyserve-m7v" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7v-kda-panel-solve.md" %})

M7t found an attractive target: Kimi's state-carrying KDA tail launches only
64 blocks on an 84-SM A6000, uses 255 registers per thread, and spills millions
of values to local memory. The proposed fix sounded almost mechanical: halve
the value tile from 16 columns to 8, halve the block from eight warps to four,
and launch twice as many programs.

The experiment makes the isolated 2,048-token tail faster, but it does not pass
the milestone gate. Across three paired runs, full-model latency falls `4.62%`
at the pooled median, below the required
`5%`, and total local-memory instructions rise `1.82%` instead of falling. M7u
therefore keeps the M7q `BLOCK_V=16, num_warps=8` launch as the serving default.

That rejected result is useful. It shows why “more blocks” is not the same as
“more parallel work,” and why a kernel-local speedup cannot promote itself.

[![The original KDA tail owns sixteen value columns in each eight-warp program.
The candidate owns eight columns in each four-warp program. It doubles blocks
but keeps the total warp count fixed, speeds up the isolated tail, and still
fails the spill and full-model gates.](/assets/tinyserve/m7u-kda-state-tile.svg)](/assets/tinyserve/m7u-kda-state-tile.svg)

## Start with one concrete prompt

Use the same Kimi-K3 layer shape as M7t:

```text
B = 1 request                  H = 8 KDA heads
T = 2,048 prompt tokens        C = 64 tokens per algebra chunk
K = 128 key channels           V = 128 value channels
N = T / C = 32 ordered chunks
```

For one head, the recurrent state is a `[K,V] = [128,128]` FP32 matrix. Value
columns are independent, so the kernel can partition the state along `V`.
Chunks are not independent: chunk 1 consumes the state emitted by chunk 0,
chunk 2 consumes chunk 1's state, and so on.

The original M7q ownership is:

```text
one program owns state[:, 0:16]     → [128,16]
one program owns state[:, 16:32]    → [128,16]
...
one program owns state[:, 112:128]  → [128,16]

8 value tiles/head × 8 heads = 64 programs
8 warps/program × 64 programs = 512 launched warps
```

Each program visits all 32 chunks in order. Keeping that state tile live avoids
32 host submissions, but the solve rows, output rows, and state projection must
coexist for a long time.

M7u changes ownership, not the equation:

```text
one program owns state[:, 0:8]      → [128,8]
one program owns state[:, 8:16]     → [128,8]
...
one program owns state[:, 120:128]  → [128,8]

16 value tiles/head × 8 heads = 128 programs
4 warps/program × 128 programs = 512 launched warps
```

The block count doubles, but the total number of warps does not. That detail is
easy to miss and predicts much of the result.

## Make the hypothesis selectable

The serving interface does not need a tile-size option. Kernel geometry is an
implementation detail, so `fused_kda_multichunk()` gets two keyword-only
experiment knobs while its defaults stay on the M7q launch:

```python
def fused_kda_multichunk(
    ...,
    *,
    tail_value_block: int = 16,
    tail_num_warps: int = 8,
):
    tail_grid = (
        batch * heads,
        triton.cdiv(value_dim, tail_value_block),
    )
    _kda_multichunk_tail_kernel[tail_grid](
        ...,
        BLOCK_V=tail_value_block,
        num_warps=tail_num_warps,
    )
```

The benchmark temporarily selects a geometry at this internal boundary. Both
paths run in one loaded model, in alternating order, with the same prompts.
The model constructor, scheduler, KDA equation, chunk order, and public backend
switches remain unchanged.

A small warp-count screen prevents us from assuming that fewer threads are
always better. At `BLOCK_V=8`, eight warps were about `0.63×` baseline tail
speed and two warps about `0.95×`; four warps was the only faster candidate.
Those screening runs choose the candidate. The retained paired and NCU runs
make the promotion decision.

## Numerical behavior stays fixed

The candidate still performs the same FP32 operations for each owned value
column. Changing the program boundary can change reduction scheduling, so M7u
checks both the isolated recurrence and the complete checkpoint.

Across 32, 128, 512, and 2,048 tokens, the retained full-model comparison has:

```text
maximum logit difference       0
maximum recurrent-state diff   0
top-5 token candidates         equal
argmax token                   equal
generated token IDs            equal
```

Separate kernel tests cover the 2-, 4-, and 8-warp narrow variants against the
original 16-wide kernel. The 4-warp candidate also stays within the existing
PyTorch-oracle tolerances. Padding and partial-chunk behavior still use the
unchanged M7q/M7p fallback paths.

## The isolated tail passes

The final paired benchmark uses one loaded model, two warmups, nine measured
repeats, and alternating path order. Its isolated operator timing includes five
KDA calls per repeat and extracts the tail kernel's GPU time.

| tokens | original tail | 8-wide / 4-warp tail | latency reduction | complete KDA speedup |
|---:|---:|---:|---:|---:|
| 128 | 0.280 ms | 0.263 ms | 6.23% | 1.038× |
| 512 | 1.100 ms | 1.021 ms | 7.17% | 1.060× |
| 2,048 | 4.436 ms | 3.902 ms | **12.03%** | **1.110×** |

The predeclared isolated gate asks for at least a 10% latency reduction at
2,048 tokens. The candidate passes.

## NCU explains what improved—and what did not

The post-change NCU run uses the exact M7t shape, strides, dtype, input seed,
and kernel filter. Only `BLOCK_V` and `num_warps` differ.

| T=2,048 tail metric | M7t: 16 × 8 warps | M7u: 8 × 4 warps | change |
|---|---:|---:|---:|
| NCU duration | 5.647 ms | 4.871 ms | **-13.74%** |
| grid / block | 64 / 256 | 128 / 128 | 2× blocks, half threads |
| waves per SM | 0.762 | 0.762 | unchanged |
| registers per thread | 255 | 255 | unchanged |
| shared memory per block | 36.86 KiB | 35.00 KiB | -5.41% |
| theoretical occupancy | 16.67% | 16.67% | unchanged |
| achieved occupancy | 16.43% | 12.21% | -25.73% |
| local-load instructions | 6.980M | 6.586M | -5.64% |
| local-store instructions | 1.974M | 2.531M | +28.19% |
| total local instructions | 8.954M | 9.117M | **+1.82%** |
| SM throughput | 37.01% | 46.84% | +26.57% |

Why does doubling the grid leave `waves/SM` unchanged? NCU normalizes a wave
by the number of blocks that can reside concurrently. The original block has
eight warps and fits once per SM. The candidate block has four warps and fits
twice per SM:

$$
\text{original waves/SM} = \frac{64}{84 \times 1} = 0.762,
\qquad
\text{candidate waves/SM} = \frac{128}{84 \times 2} = 0.762.
$$

The candidate covers every SM with at least one smaller block, which improves
SM throughput. It does not create another wave of warps or lower the
255-register ceiling. It also makes two programs reread the Q/K/decay and pair
data previously shared by one wider program.

The source profile confirms that the same causal reductions remain hot. At
2,048 tokens, `tinyserve/kimi_kernels.py:814` (forward substitution) carries
17,383 long-scoreboard samples, and line 831 (query-pair correction) carries
16,403. Sampled states remain dominated by short scoreboard (`24.72%`), long
scoreboard (`20.78%`), barriers (`18.14%`), and fixed-latency wait (`16.04%`).
The tile got faster, but it did not remove the live-range and synchronization
problem selected by M7t.

## The complete model does not pass

The same nine-repeat run times one-token generation through the complete
Kimi-K3 checkpoint. The paths use separate medians; paired per-repeat
reductions are also retained so one noisy run cannot hide direction.

| prompt tokens | original median | candidate median | median reduction |
|---:|---:|---:|---:|
| 32 | 50.123 ms | 50.352 ms | -0.46% |
| 128 | 49.876 ms | 49.823 ms | 0.11% |
| 512 | 53.491 ms | 52.687 ms | 1.50% |
| 2,048 | 108.762 ms | 102.177 ms | **6.05%** |

That last point estimate crosses 5%, but it does not replicate. Two independent
five-repeat runs measured `4.56%` and `4.18%` paired medians; the nine-repeat
run measured `5.86%`. Pooling all 19 paired reductions gives a `4.62%` median
and a deterministic 10,000-resample bootstrap interval of `[4.39%, 5.41%]`.
The result is positive, but it does not establish the required 5% improvement.
Peak allocated memory is identical at `843,904,512` bytes in every run.

Two promotion conditions therefore fail:

1. total local-memory instructions must materially decrease, but rise 1.82%;
2. full-model T=2,048 latency must decrease at least 5%, but the pooled paired
   median is 4.62% and its confidence interval crosses the threshold.

Parity, short-prompt, isolated-tail, and memory gates pass. A majority is not
enough: each gate protects a different failure mode.

## Decision: keep the original launch

M7u retains the internal benchmark geometry and parity tests so the result is
inspectable, but does not switch the serving default. Consequently there is no
new serving path to calibrate against Transformers + FLA: an external run would
measure the unchanged M7q default. Ollama, llama.cpp, FreeToken, and ninfer
still do not share a supported Kimi-K3 checkpoint boundary.

The result also rejects “make the tile even smaller” as the next blind move.
The kernel needs a mechanism that shortens live ranges or reduces shared/local
traffic, not another repartition that preserves 512 total warps. M7v should
isolate the 64-row forward substitution and test a separate solve/stitch or
shared-memory layout while preserving the readable FP32 oracle.

## Inspect it locally

Use **M7u: compare KDA state tiles** in `.vscode/launch.json`. Break in:

1. `configure_path()` to see the current `16×8` or candidate `8×4` geometry;
2. `fused_kda_multichunk()` to inspect `tail_grid`;
3. `_kda_multichunk_tail_kernel` to map one program to `[128,BLOCK_V]`; and
4. the promotion-gate assembly to see why a local speedup is rejected.

The retained paired command is:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \
  examples/bench_kda_kernel.py --comparison state_tile \
  --candidate-tail-warps 4 --device cuda:0 \
  --prompt-tokens 32 128 512 2048 \
  --micro-lengths 32 128 512 2048 \
  --chunk-size 64 --micro-iterations 5 --warmups 2 --repeats 9 \
  --output .tinyserve-bench/m7u/kda-state-tile-w4-final9.json
```

The candidate NCU harness and parsed metrics live under
`profile/kda-state-tile-m7u-a6000/`; the direct M7t comparison lives under
`profile/kda-state-tile-m7u-vs-m7t-a6000/`. Raw `.ncu-rep` files stay local and
ignored, while the compact [M7u evidence
manifest](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7u-kda-state-tile-a6000-2026-09-12.json) records the
measurement boundary and decision.

## Takeaway

Parallelism is counted in schedulable work, not grid dimensions. M7u doubles
programs by halving both value columns and warps per program, so it still
launches 512 warps. That repacking helps the isolated kernel, but it leaves the
register ceiling and total spill traffic intact.

The serving rule is stricter than “the kernel got faster.” Preserve the
equation, measure the exact target, check the complete model, and keep the old
default when the mechanism does not satisfy the gates that justified it.
