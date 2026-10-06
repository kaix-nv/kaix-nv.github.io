---
layout: post
math: true
title: "Building tinyserve M7w: Pay once, reuse eight times"
date: 2026-10-05 08:21:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Prepare state-independent KDA factors once per chunk and reuse them across value tiles."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m7w-kda-solve-factors.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M7v — When panelization increases register spills]({% include tinyserve-post-url.html slug="building-tinyserve-m7v" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7v-kda-panel-solve.md" %}) · Next: [M8 — Quantization: fewer bits are a systems contract]({% include tinyserve-post-url.html slug="building-tinyserve-m8" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m8-quantization.md" %})

M7v tried to make one KDA value-tile program smaller. It did not work because
the compiler still kept a large live set and spilled more data. M7w changes a
more important boundary: instead of solving the same 64-row triangular system
inside every value tile, it prepares two state-independent transforms once per
chunk and shares them across all eight tiles.

That change is exact FP32 algebra, not an approximation. On the retained RTX
A6000 run, the complete T2,048 KDA path becomes `2.15x` faster and prompt plus
one-token model latency falls from `105.81 ms` to `92.55 ms`. The paired median
reduction is `12.76%`, with a bootstrap interval of `[12.04%, 13.48%]`.
Logits are bit-identical and the largest recurrent-state difference is
`4.66e-10`.

The lesson is broader than KDA: when many GPU programs repeat the same causal
work, first ask whether the dependency belongs inside that ownership boundary.

## The repeated work

For one Kimi-K3 chunk:

```text
chunk width C = 64
key dimension K = 128
value dimension V = 128
value tile Vt = 16
value-tile programs = V / Vt = 8
```

After pair construction, KDA has a unit-lower matrix $L$ and a right-hand side
$R$. The current row path solves

$$
L X = R
$$

and then consumes $X$ twice:

$$
O = O_{\text{base}} + P X,
$$

$$
S_{n+1} = D_n S_n + K_{d,n}^{\mathsf T} X.
$$

$P$ is the causal query-pair matrix, $K_d$ is the key matrix scaled for the
chunk's final decay, and $D_n$ decays the incoming state. The RHS also depends
on that incoming state:

$$
R = \beta \odot \left(V - K_c S_n\right).
$$

The state dependency means chunks must still be applied in order. But $L$,
$P$, and $K_d$ do **not** depend on $S_n$. That separation is the opening M7w
uses.

The old kernel owns only 16 value channels, so its RHS is `[64,16]`. It walks
the 64 dependent solve rows once for channels 0–15, again for channels 16–31,
and so on. All eight programs read the same $L$, but each repeats the forward
substitution.

## A four-row example

Shrink the chunk to four tokens and the value dimension to eight channels,
split into two four-channel tiles:

```text
L  = [4,4]
R0 = [4,4]  # value channels 0..3
R1 = [4,4]  # value channels 4..7
```

The old ownership solves two systems:

$$
L X_0 = R_0, \qquad L X_1 = R_1.
$$

Both solves walk the same dependency graph: row 0, then row 1 using row 0,
then row 2 using rows 0–1, then row 3 using rows 0–2.

M7w instead solves four identity columns once:

$$
L Z = I.
$$

This gives the solve transform $Z=L^{-1}$. The implementation does not call a
generic inverse routine. Four Triton programs each forward-substitute 16
identity columns for the real C64 shape. Before discarding its local tile of
$Z$, each program forms two transforms:

$$
A = P Z \in \mathbb{R}^{64 \times 64},
$$

$$
U = K_d^{\mathsf T} Z \in \mathbb{R}^{128 \times 64}.
$$

Every value tile can now consume its state-dependent RHS with matrix products:

$$
O = O_{\text{base}} + A R,
$$

$$
S_{n+1} = D_n S_n + U R.
$$

For the four-row example, both $R_0$ and $R_1$ reuse the same $A$ and $U$.
For Kimi-K3, eight `[64,16]` RHS tiles reuse them.

[![M7w moves the triangular solve before the value-tile fan-out while preserving ordered state carry between chunks.](/assets/tinyserve/m7w-kda-solve-factors.svg)](/assets/tinyserve/m7w-kda-solve-factors.svg)

## The implementation has two phases

`fused_kda_multichunk()` still starts with `_kda_pair_kernel`, which constructs
$L$ and $P$ for every `[batch, head, chunk]`. M7w changes what follows.

### Phase 1: prepare factors in parallel

`_kda_solve_factor_kernel` launches four programs per head and chunk. Program
`j` owns identity columns `16j..16j+15`:

```text
inverse_rows [64,16] = solve(L, identity columns)
output tile  [64,16] = P @ inverse_rows
state tile  [128,16] = Kd.T @ inverse_rows
```

It stores the complete transforms as:

| object | shape per head and chunk | role |
|---|---:|---|
| `output_factor` | `[64,64]` | maps an RHS tile directly into output correction |
| `state_factor` | `[128,64]` | maps an RHS tile directly into state correction |

The temporary inverse tile never becomes a global `[64,64]` allocation. Only
the two useful transforms are stored. Preparation is independent across all
heads and chunks, so T2,048 launches `8 heads × 32 chunks × 4 column tiles =
1,024` blocks.

### Phase 2: apply factors while carrying state

`_kda_multichunk_factor_tail_kernel` retains M7q's causal ownership: one
program owns a `[128,16]` state tile and walks chunks in order. Inside each
chunk it:

1. computes `remembered = (K * exp(G)) @ state_block`;
2. builds the state-dependent `rhs [64,16]`;
3. computes the output with `output_factor @ rhs`;
4. advances state with `state_factor @ rhs`;
5. continues to the next chunk using that new state.

Only factor preparation crosses the chunk loop. The recurrent-state hand-off
does not. M7w therefore preserves the semantic order `S0 → S1 → ... → SN`.

## The memory tradeoff is explicit

For one FP32 head and chunk, the factors cost:

$$
(64 \times 64 + 128 \times 64) \times 4\ \text{bytes} = 48\ \text{KiB}.
$$

At T2,048 with eight heads and 32 chunks, that is 12 MiB. The standalone
benchmark measures exactly `12,582,912` extra peak bytes. The full checkpoint's
peak allocation remains `843,904,512` bytes for both paths because other model
buffers set a higher peak.

This is a deliberate compute-for-storage exchange. CPU, autograd, partial
chunks, different K/V dimensions, and non-C64 paths keep the readable PyTorch
or existing kernel fallbacks.

## Three gates before promotion

The experiment is staged so a launch-count artifact cannot promote it.

First, `bench_kda_factors.py` flattens independent chunks into one batch. Both
paths receive the same prebuilt $L$ and $P$ and use two launches. Pair
construction is excluded, but candidate factor preparation is included.

| total tokens | row tail | factor tail | speedup |
|---:|---:|---:|---:|
| 128 | 0.272 ms | 0.121 ms | 2.24x |
| 512 | 0.953 ms | 0.320 ms | 2.98x |
| 2,048 | 3.335 ms | 1.223 ms | 2.73x |

Second, the same benchmark restores one real sequence and ordered state carry:

| prompt tokens | row multichunk | factor multichunk | speedup |
|---:|---:|---:|---:|
| 128 | 0.356 ms | 0.201 ms | 1.77x |
| 512 | 1.309 ms | 0.613 ms | 2.14x |
| 2,048 | 5.203 ms | 2.314 ms | 2.25x |

Third, `bench_kda_kernel.py --comparison solve_factors` loads the checkpoint
once, alternates path order, and runs five paired repetitions. Its profiler
extracts factor preparation plus factor application as one tail boundary:

| tokens | row KDA | factor KDA | row tail | factor tail | tail speedup |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.461 ms | 0.461 ms | inactive | inactive | — |
| 128 | 0.446 ms | 0.413 ms | 0.280 ms | 0.113 ms | 2.49x |
| 512 | 1.402 ms | 0.714 ms | 1.102 ms | 0.409 ms | 2.69x |
| 2,048 | 5.340 ms | 2.481 ms | 4.430 ms | 1.568 ms | 2.82x |

T32 does not enter the multichunk kernel, so its small difference is ordinary
measurement noise. At T128, complete-model paired latency regresses `1.31%`,
inside the declared 5% guard.

## The complete model keeps the long-prompt win

The end-to-end measurement includes tokenization, cache allocation, all 17
layers, logits, and one generated token:

| prompt tokens | row median | factors median | paired median reduction | 95% bootstrap interval |
|---:|---:|---:|---:|---:|
| 32 | 50.74 ms | 50.78 ms | 0.01% | [-6.32%, 2.76%] |
| 128 | 49.99 ms | 50.62 ms | -1.31% | [-1.94%, 0.02%] |
| 512 | 53.70 ms | 52.72 ms | 0.51% | [-0.16%, 2.33%] |
| 2,048 | 105.81 ms | 92.55 ms | **12.76%** | **[12.04%, 13.48%]** |

Every generated token sequence, top-five set, and argmax matches. Logit
difference is zero across all four lengths. The maximum state difference is
`4.66e-10`, far below the `5e-5` gate.

## NCU shows why this boundary works

Fresh profiles retain T512 and T2,048 for the old row tail, factor preparation,
and factor application. Candidate time below is the sum of its two kernels.

| metric | T512 row | T512 factors | T2,048 row | T2,048 factors |
|---|---:|---:|---:|---:|
| NCU duration | 1.416 ms | 0.467 ms | 5.647 ms | 1.757 ms |
| speedup | — | 3.03x | — | 3.21x |
| preparation blocks | — | 256 | — | 1,024 |
| application blocks | 64 | 64 | 64 | 64 |
| registers/thread | 255 | 255 / 255 | 255 | 255 / 255 |
| total local instructions | 2.245M | 1.814M | 8.954M | 7.212M |

M7w does not magically eliminate register pressure. Both candidate kernels
still compile at 255 registers per thread. The application kernel still owns
the large state tile, and source counters now point to state projection and
factor loads. Its long-scoreboard fraction rises because factor traffic is
more visible in the much shorter kernel.

The causal improvement is elsewhere. The old row solve's reduction and
pair-correction lines account for the largest implementation-owned
long-scoreboard hotspots. Factor preparation pays its 64-row identity solve in
a 1,024-block grid at T2,048, then the 64 application blocks consume matrix
products instead of repeating forward substitution eight times. Combined
local loads fall enough that total local instructions drop `19.47%`, even
though factor storage increases local stores and global traffic.

NCU replay time is diagnostic, not the serving-latency headline. Here it agrees
with ordinary CUDA-event and full-model timing, so all three evidence layers
support the same mechanism.

## Decision: promote factors for the measured path

The default multichunk Kimi-K3 geometry now selects `tail_solve="factors"`.
The row and rejected panel paths remain available for tests, benchmarks, and
debugging. Alternate value-tile geometries still default to the row solve so
their experiment boundary does not silently change.

There is no Ollama, llama.cpp, FreeToken, or ninfer number for this milestone:
the tiny structural Kimi-K3 fixture is not a shared supported checkpoint for
those engines. M7w also does not introduce FLA as an implementation or
calibration dependency. The causal cross-engine boundary is therefore the
same loaded checkpoint with row and factor kernels selected explicitly.

## Inspect it locally

Use **M7w: measure reusable KDA solve factors** in `.vscode/launch.json`. Useful
breakpoints are:

1. `configure_path()` in `bench_kda_kernel.py`, where row and factor paths are
   selected without reloading the model;
2. `fused_kda_multichunk()`, where $L/P$ storage and factor storage are
   allocated;
3. `_kda_solve_factor_kernel()`, especially the identity-column forward solve;
4. `_kda_multichunk_factor_tail_kernel()`, where state crosses chunk
   boundaries but factors do not.

The [standalone factor evidence](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7w-kda-factor-boundary-a6000-2026-09-13.json)
and [checkpoint promotion evidence](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7w-kda-solve-factors-a6000-2026-09-13.json)
retain raw paired samples, gates, source hashes, and memory. The compact NCU
analysis is under `profile/kda-solve-factors-m7w-a6000/`; raw `.ncu-rep` files
remain local and ignored.

## Takeaway

M7v shortened a source-level lifetime but left the same work inside every
value-tile owner. M7w moves the dependency itself: prepare a state-independent
solve transform once, share it across all RHS tiles, and keep only true state
causality in the ordered chunk loop. A good kernel boundary is not merely one
that makes a program smaller. It is one that stops independent programs from
repeating the same serial work.
