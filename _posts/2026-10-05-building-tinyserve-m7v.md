---
layout: post
math: true
title: "Building tinyserve M7v: When panelization increases register spills"
date: 2026-10-05 08:20:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "A panelized solve increases register spills and loses; retain the measured failure and the row-solve default."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m7v-kda-panel-solve.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M7u — When a faster kernel is still the wrong default]({% include tinyserve-post-url.html slug="building-tinyserve-m7u" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7u-kda-state-tile.md" %}) · Next: [M7w — Pay once, reuse eight times]({% include tinyserve-post-url.html slug="building-tinyserve-m7w" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7w-kda-solve-factors.md" %})

M7u taught us that splitting a KDA value tile creates more blocks but does not
remove the state-carrying kernel's real constraint. The retained M7q tail still
uses 255 registers per thread and executes millions of local-memory loads and
stores.

M7v attacks a more causal target: the 64-row forward substitution inside each
chunk. The source-level idea is clean. Divide the triangular system into four
16-row panels, consume each solved panel immediately, and stop keeping all 64
solved rows alive.

The equations remain exact, but the optimization fails. Triton still compiles
the candidate at 255 registers per thread, local-memory instructions rise
`58.84%`, the ordinary 2,048-token tail becomes `4.78%` slower, and paired
full-model latency becomes `15.65%` slower at the median. M7v therefore keeps
the original row solve as the serving default.

This is a useful failure because it distinguishes a source-code lifetime from
a compiled GPU lifetime.

## The problem inside one concrete chunk

Use the Kimi-K3 shape from the benchmark:

```text
chunk width C = 64
key dimension K = 128
value tile Vt = 16
incoming state = [128,16]
lower matrix L = [64,64]
right-hand side B = [64,16]
solution X = [64,16]
```

For one chunk, the tail must solve the unit-lower system

$$
L X = B.
$$

Row `r` depends on every earlier solved row:

$$
X_r = B_r - \sum_{j < r} L_{r,j} X_j.
$$

The M7q implementation follows that equation directly. It creates
`solved_rows [64,16]`, fills one row at a time, and keeps the complete solution
for two later consumers:

$$
O = O_{\text{base}} + P X
$$

and

$$
S_{n+1} = D_n S_n + K_n^\mathsf{T} X.
$$

That is easy to read and easy to compare with the PyTorch oracle. It also means
the solution, output accumulator, state tile, RHS, and several matrix operands
overlap inside one large Triton program.

## The proposed four-panel solve

Partition the rows into four groups:

```text
panel 0: rows  0..15
panel 1: rows 16..31
panel 2: rows 32..47
panel 3: rows 48..63
```

Then view the lower matrix as a block-triangular system:

$$
\begin{aligned}
X_0 &= \operatorname{solve}(L_{00}, B_0), \\
X_1 &= \operatorname{solve}(L_{11}, B_1 - L_{10}X_0), \\
X_2 &= \operatorname{solve}(L_{22}, B_2 - L_{20}X_0 - L_{21}X_1),
\end{aligned}
$$

with the same pattern for $X_3$.

The important implementation trick is not merely “use blocks.” After solving
$X_0$, the kernel immediately:

1. subtracts $L_{j0}X_0$ from every future RHS panel;
2. adds $P_{:,0}X_0$ to the chunk output;
3. adds $K_0^\mathsf{T}X_0$ to the outgoing state;
4. discards $X_0$ before solving $X_1$.

The same sequence repeats for all four panels. There is no complete inverse,
no approximation, no changed causal order, and no extra serving launch.

[![The original row solve and the intended four-panel dataflow, followed by the compiled spill result.](/assets/tinyserve/m7v-kda-panel-solve.svg)](/assets/tinyserve/m7v-kda-panel-solve.svg)

## How the implementation maps the equations

The candidate is selected only by the internal
`tail_solve="panel16"` argument of `fused_kda_multichunk()`. The normal serving
call does not pass that argument, so it continues to compile the M7q row path.

Inside `_kda_multichunk_tail_kernel`, one candidate iteration looks like this:

| source object | shape | role |
|---|---:|---|
| `rhs` | `[64,16]` | original RHS plus corrections from finished panels |
| `solved_panel` | `[16,16]` | current diagonal-panel solution |
| `query_panel` | `[64,16]` | maps the current solution into all output rows |
| `lower_panel` | `[64,16]` | maps the current solution into future RHS rows |
| `state_block` | `[128,16]` | incoming state reused as the outgoing accumulator |
| `output_rows` | `[64,16]` | base output reused as the correction accumulator |

For panel 0, `solved_panel` owns rows 0–15. `query_panel @ solved_panel`
updates all 64 output rows, and a `[128,16] @ [16,16]` product updates the
state. `lower_panel @ solved_panel` corrects rows 16–63 of `rhs`. The next
iteration repeats this for rows 16–31.

This is the exact source-level lifetime we wanted to test: only one
`[16,16]` solution panel is named at a time.

## Correctness comes before performance

Changing the grouping changes the FP32 operation order, so bitwise state
identity is not required. The retained tests compare the complete output and
outgoing state against the row implementation. The checkpoint comparison also
checks logits, top-five IDs, argmax, generated token IDs, and the recurrent
cache.

Across 32, 128, 512, and 2,048 prompt tokens:

- maximum absolute logit difference is `0`;
- maximum absolute recurrent-state difference is `2.33e-10`;
- top-five IDs, argmax, and generated token IDs are equal;
- measured peak allocation is identical for both paths.

The candidate is numerically acceptable. Its performance is not.

## Ordinary timing says the panel path is slower

The retained run loads Kimi-K3 once, warms both paths, alternates their order,
and records five paired repetitions. CUDA-event timing isolates the KDA path
and PyTorch's CUDA profiler extracts the state-carrying tail.

| tokens | row KDA | panel KDA | row tail | panel tail | tail change |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.476 ms | 0.484 ms | inactive | inactive | — |
| 128 | 0.470 ms | 0.478 ms | 0.297 ms | 0.303 ms | +1.83% |
| 512 | 1.481 ms | 1.525 ms | 1.169 ms | 1.209 ms | +3.50% |
| 2,048 | 5.409 ms | 5.625 ms | 4.499 ms | 4.714 ms | **+4.78%** |

Positive change means slower. The regression grows with the number of chunks,
which is consistent with extra work inside every panel of every chunk.

The complete model shows the same direction where the KDA work is large:

| prompt tokens | row median | panel median | paired median change |
|---:|---:|---:|---:|
| 32 | 49.40 ms | 49.11 ms | -0.62% |
| 128 | 49.59 ms | 49.05 ms | -0.74% |
| 512 | 55.00 ms | 58.68 ms | +6.60% |
| 2,048 | 106.28 ms | 123.07 ms | **+15.65%** |

The five-repeat T2,048 bootstrap interval is wide because one baseline sample
is slow, but that uncertainty cannot rescue the candidate: both the isolated
tail gate and the independent hardware-mechanism gate fail.

## NCU explains why the source intuition failed

The fresh NCU comparison uses identical shapes, seeds, strides, launch
geometry, and FP32 precision. It repeats at T512 and T2,048.

| T2,048 metric | row solve | panel solve | change |
|---|---:|---:|---:|
| NCU duration | 5.648 ms | 5.336 ms | -5.53% |
| grid / block | 64 / 256 | 64 / 256 | unchanged |
| waves per SM | 0.762 | 0.762 | unchanged |
| registers/thread | 255 | 255 | unchanged |
| shared memory/block | 37 KiB | 33 KiB | -10.81% |
| local loads | 6.980M | 9.159M | +31.21% |
| local stores | 1.974M | 5.065M | +156.54% |
| total local instructions | 8.954M | 14.223M | **+58.84%** |
| long-scoreboard samples | 18.07% | 38.01% | +19.94 points |
| SM throughput | 36.50% | 20.02% | -16.48 points |

T512 repeats the same result: local instructions increase `59.00%` and
long-scoreboard samples rise from `18.18%` to `37.91%`.

Why can NCU duration improve while ordinary latency regresses? NCU replays a
kernel many times to collect incompatible counter sets. Its duration is useful
inside that controlled diagnostic run, but it is not the serving-latency
arbiter. The candidate also misses the NCU-local 10% threshold and, more
decisively, moves spill traffic in the wrong direction.

The memory hierarchy confirms the diagnosis. At T2,048, DRAM-read throughput
rises from `4.17%` to `23.48%`, L1 hit rate falls from `37.80%` to `13.16%`,
and L2 hit rate falls from `83.74%` to `61.71%`. The candidate is not saturating
DRAM; it is waiting more often on additional spill traffic.

The panel source removed one complete `solved_rows` name, but the compiler
statically expands four panel stages while the full RHS, output accumulator,
state accumulator, and dot operands remain live. A smaller source array did not
produce a smaller compiled live set.

## Decision: retain the row solve

M7v leaves `tail_solve="row"` as the default. The panel path remains available
to tests, the benchmark, and the debugger because a rejected implementation is
valuable when its boundary and evidence are explicit.

No external Transformers, Ollama, llama.cpp, FreeToken, or ninfer calibration
is run. The serving path is unchanged, so such a run would only remeasure M7q.

The next question is different: can the solve preparation be computed once per
`[batch, head, chunk]` and reused by all eight value tiles? M7w should measure
that factor boundary in isolation before adding storage or launches to the
serving path.

## Inspect it locally

Use **M7v: compare KDA causal solves** in `.vscode/launch.json`. Break in:

1. `configure_path()` to see `row` versus `panel16` selection;
2. `fused_kda_multichunk()` to see the unchanged `16 × 8-warps` geometry;
3. `_kda_multichunk_tail_kernel()` at the panel loop;
4. the three immediate consumers of `solved_panel`.

The retained benchmark output is intentionally outside Git under
`.tinyserve-bench/m7v/`. Its SHA-256 and measurements are recorded in the
[M7v evidence manifest](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7v-kda-panel-solve-a6000-2026-09-13.json).
The compact NCU report is under `profile/kda-panel-solve-m7v-a6000/`; raw
`.ncu-rep` files stay local and ignored.

## Takeaway

Source-level scope is a hypothesis about register lifetime, not proof. A GPU
compiler can keep other values live across statically expanded panel stages or
spill them around new matrix products. The only reliable loop is the one used
here: preserve the equations, isolate one mechanism, inspect compiled counters,
measure the complete model, and reject the candidate when those layers disagree
with the source intuition.
