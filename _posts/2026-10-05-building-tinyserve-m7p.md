---
layout: post
math: true
title: "Building tinyserve M7p: Fuse KDA's correction and state tail"
date: 2026-10-05 08:14:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Split KDA correction and state updates into kernels with distinct dependency and ownership patterns."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m7p-fused-kda-tail.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M7o — Fuse KDA's pair construction]({% include tinyserve-post-url.html slug="building-tinyserve-m7o" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7o-fused-kda-pairs.md" %}) · Next: [M7q — Submit KDA chunks once, carry state on the GPU]({% include tinyserve-post-url.html slug="building-tinyserve-m7q" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7q-multichunk-kda-prefill.md" %})

M7o made KDA's two causal pair matrices inexpensive. Its post-change profile
then moved 68.7% of the isolated PyTorch CUDA time into four small batched
matrix multiplications and one triangular solve. A 2,048-token prompt repeats
that tail 32 times in every KDA layer.

M7p keeps the pair kernel and the mathematical boundary unchanged. Two small
Triton kernels replace the fragmented tail: one builds the right-hand side,
performs forward substitution, and emits token outputs; the other updates the
recurrent state passed to the next chunk. The readable PyTorch equations remain
selectable as the oracle.

[![M7o launches four small matrix multiplications and a library triangular
solve after every KDA pair kernel. M7p assigns one batch-head and a 16-channel
value tile to its correction/output program, solves rows in causal order, and
then updates independent 16-by-16 recurrent-state tiles in a second kernel.
The pair matrices are built once and reused by all value tiles.](/assets/tinyserve/m7p-fused-kda-tail.svg)](/assets/tinyserve/m7p-fused-kda-tail.svg)

## Start from the tail that M7o left behind

For one batch row and one head, use these dimensions:

- `C`: tokens in this inner chunk, at most 64;
- `K`: key channels, 128 for the tiny Kimi-K3 checkpoint;
- `V`: value channels, also 128;
- `S_in`: incoming recurrent state with shape `[K,V]`;
- `L` and `P`: M7o's causal pair matrices with shape `[C,C]`.

Let `G[t,k]` be cumulative log decay inside the chunk and
`D[t,k] = exp(G[t,k])`. The PyTorch oracle evaluates:

$$
R = b \odot \left(V - (K \odot D)S_{in}\right),
\qquad
W = L^{-1}R,
$$

$$
Y = (Q \odot D)S_{in} + PW,
$$

$$
S_{out} = e^{G_{last}} \odot S_{in}
          + \left(K \odot e^{G_{last}-G}\right)^T W.
$$

Here `R`, `W`, and `Y` are `[C,V]`. `W` is called `correction` in the
implementation. It tells KDA how much each token writes after removing what
the decayed incoming state and earlier writes already explain.

Written as code, the boundary is deliberately direct:

```python
rhs = b * (v - (k * exp(G)) @ S_in)
correction = torch.linalg.solve_triangular(L, rhs, upper=False)
output = (q * exp(G)) @ S_in + P @ correction
S_out = exp(G_last) * S_in + (k * final_decay).T @ correction
```

The four `@` operations and the solve are individually reasonable. Their
shape is the problem. At `C=64`, each operation is small, and Python submits
the whole chain once per chunk. One isolated 2,048-token rule therefore calls
128 BMMs and 32 triangular solves. The complete 12-layer KDA model repeats
that pattern 12 times.

## Forward substitution with three concrete tokens

The triangular solve is easier to understand row by row. Suppose one head has
three tokens, two value channels, and:

```text
L = [[ 1.0, 0.0, 0.0],       R0 = [1.0, 2.0]
     [ 0.2, 1.0, 0.0],       R1 = [3.0, 1.0]
     [-0.1, 0.3, 1.0]]       R2 = [2.0, 4.0]
```

Because `L` has a unit diagonal, `L @ W = R` can be solved without dividing:

```text
W0 = R0
   = [1.00, 2.00]

W1 = R1 - 0.2 W0
   = [2.80, 0.60]

W2 = R2 - (-0.1 W0 + 0.3 W1)
   = [1.26, 4.02]
```

Token 2 cannot be solved before tokens 0 and 1. Value channels do not depend
on one another, however. The same row coefficients in `L` independently solve
all 128 columns of `W`. That separation determines the kernel grid.

If the last row of `P` is `[0.2, -0.1, 0.3]`, its correction contribution is:

```text
(P @ W)2 = 0.2 W0 - 0.1 W1 + 0.3 W2
           = [0.298, 1.546]
```

The final token output adds this vector to its incoming-state read
`((q * D) @ S_in)2`. Nothing about fusion changes those numbers or the causal
order.

## Kernel one owns causal order

`_kda_correction_output_kernel` uses a grid over `(batch × head, value tile)`.
One program owns all `C` token rows for 16 value channels:

```text
program (batch=0, head=3, values=32..47)
    project all C keys through S_in        -> remembered[C,16]
    project all C queries through S_in     -> base_output[C,16]
    row 0: solve W0, then form Y0
    row 1: read W0, solve W1, then form Y1
    ...
    row C-1: read W0..W(C-2), solve final row, form final output
```

The two state projections reuse the same incoming `[K,16]` state tile. The
program keeps solved correction rows locally, so the causal dependency does
not cross programs and does not need a global synchronization. At the end it
writes `W[:, values]` and `Y[:, values]` once.

The value tile is intentionally 16 rather than 32. The first 32-channel
prototype kept too many `[C,V_tile]` intermediates live and improved the
2,048-token isolated rule by only about 4%. Reducing the tile to 16 gives the
compiler a smaller live working set; the retained kernel is 1.53–1.98× faster
than the M7o tail path across the measured lengths. This is a measured launch
shape, not a new model constant.

## Kernel two owns independent state tiles

The correction matrix is still needed to construct `S_out`, so M7p keeps it
as the explicit bridge between the two kernels. The state-update kernel uses
a three-dimensional grid over `(batch × head, key tile, value tile)`. Each
program owns one `[16,16]` output tile and computes:

```text
decayed incoming tile
    exp(G_last[keys]) * S_in[keys, values]

plus all token writes
    (key * exp(G_last - G)).T @ correction
```

Different state tiles have no dependency and run in parallel. The completed
`[K,V]` state is the only object carried into the next 64-token chunk. Decode
still uses the one-token recurrent equation; these kernels are a prefill-only
implementation choice.

## Why the pair kernel remains separate

`L` and `P` depend on query, key, decay, and beta, but not on a value-channel
tile. M7o builds each matrix once per chunk. Kernel one then reuses them for
eight 16-channel value tiles.

A monolithic kernel organized by value tiles would either reload or recompute
the same pair coefficients eight times. A monolithic kernel organized around
all 128 values would create the register-pressure problem the smaller tile
avoids. Two kernel boundaries therefore express the real dependency cleanly:

```text
q, k, G, beta ──> pair kernel ──> L, P
                                      │ reused
v, S_in ─────────> correction/output ─┴─> W, Y
q, k, G, S_in, W ─> state update ───────> S_out
```

## Fallback is the executable specification

The public `--kda-kernel-backend` option retains the same three values:

- `auto` uses native pair and tail kernels when their contracts fit;
- `torch` runs the complete readable PyTorch equation; and
- `triton` requires the native path and raises on an unsupported input.

The tail contract is deliberately narrow: CUDA inference, FP32 working
tensors, `1 ≤ C ≤ 64`, and `K=V=128`. Partial chunks use masks rather than
semantic padding. CPU, autograd, and future Kimi shapes fall back under
`auto`.

Padding remains an identity transition. `chunkwise_kda()` zeros padded query,
key, value, decay, and beta rows before cumulative decay or either native
stage. The poisoned-padding test writes `NaN` into those inputs and requires
finite, oracle-matching output and state.

For performance experiments, the model setter accepts a separate internal
tail selector. That is how the benchmark holds M7o's Triton pair kernel fixed
while alternating only the PyTorch and Triton tail. Normal serving exposes one
kernel selector and enables both native stages together.

## Numerical boundary

Focused CUDA tests compare the complete PyTorch equation with both native
stages at lengths 17, 64, and 65. These cover a partial chunk, a complete
chunk, and a complete-plus-one boundary using Kimi's channel-strided query and
key layout. A two-row 67-token test gives the second row only 33 real tokens,
poisons the rest, and verifies that padding cannot leak.

The retained full-model check holds M7o pair construction fixed and compares
the two tails on fresh BF16 caches:

| prompt | max logit difference | max state difference | state P99 | ordered top 5 | argmax |
|---:|---:|---:|---:|:---:|:---:|
| 32 | 0 | 6.98e-10 | 2.91e-11 | equal | equal |
| 128 | 0 | 2.33e-10 | 7.28e-12 | equal | equal |
| 512 | 0 | 2.33e-10 | 7.28e-12 | equal | equal |
| 2,048 | 4.88e-4 | 3.34e-5 | 1.61e-6 | equal | equal |

The 2,048-token logit difference is one BF16 step. Its mean absolute state
difference is `8.80e-8`; the larger maximum is localized after 32
floating-point chunk hand-offs. All states pass `atol=5e-5, rtol=5e-5`, and
all paired generated first-token decisions are equal. The normal suite passes
116 tests, with nine expected two-GPU tests skipped under one visible GPU.

## Paired promotion gate

The causal performance comparison uses one loaded BF16 model and alternates
M7o and M7p order on every retained repetition. Both paths use the same Triton
pair kernel and `C=64`; only the tail changes. Each exact-length prompt has two
warmups and five retained runs, including the complete Tinyserve generation
call and one output token.

| prompt | M7o tail | M7p tail | lower latency | peak-memory change | first token |
|---:|---:|---:|---:|---:|:---:|
| 32 | 44.46 ms | 41.98 ms | 5.57% | 0 MiB | equal |
| 128 | 126.86 ms | 114.13 ms | 10.04% | −0.75 MiB | equal |
| 512 | 150.78 ms | 129.00 ms | 14.45% | 0 MiB | equal |
| 2,048 | 292.76 ms | 211.38 ms | 27.80% | 0 MiB | equal |

The result passes the declared gate: at least 10% lower latency at 512 and
2,048 tokens, no short-prompt regression above 5%, less than 32 MiB additional
peak allocation at 512, bounded state and logit differences, and equal token
decisions.

The isolated rule shows that the intended local boundary improved:

| tokens | M7o tail | M7p tail | speedup |
|---:|---:|---:|---:|
| 32 | 0.874 ms | 0.572 ms | 1.53× |
| 128 | 0.977 ms | 0.567 ms | 1.72× |
| 512 | 3.430 ms | 1.830 ms | 1.87× |
| 2,048 | 13.257 ms | 6.691 ms | 1.98× |

## Profile again: this is mainly a launch win

The separate post-change diagnostic explains an important result. At 2,048
tokens, M7p removes the 128 `aten::bmm` calls and 32
`aten::linalg_solve_triangular` calls from the isolated rule. The retained
PyTorch self-CPU operator time falls from 14.83 ms to 6.95 ms.

| diagnostic | M7o | M7p | reduction |
|---|---:|---:|---:|
| model CUDA-event interval | 406.28 ms | 316.82 ms | 22.02% |
| KDA interval | 265.80 ms | 174.61 ms | 34.31% |
| chunkwise-rule interval | 244.44 ms | 153.14 ms | 37.35% |
| isolated PyTorch self-CPU time | 14.83 ms | 6.95 ms | 53.15% |

The raw GPU work does **not** shrink by the same amount. M7p's 32 pair,
correction/output, and state-update kernels consume 0.386, 3.440, and 0.325 ms
respectively. Adding the remaining 0.624 ms of PyTorch CUDA operators gives
about 4.78 ms, close to M7o's approximately 4.88 ms including its pair kernel.

The end-to-end gain comes mainly from replacing fragmented Python/library
submission with a smaller launch graph and fewer stream gaps. Counting only
new kernel arithmetic would miss the bottleneck that the profile exposed.

KDA still owns 55.1% of the profiled 2,048-token model interval, and the
chunkwise rule owns 87.7% of KDA. The next experiment should therefore measure
the remaining per-chunk cumulative-decay setup and Python loop before deciding
whether multi-chunk submission or another fusion is justified.

## Same-checkpoint calibration

The paired M7o/M7p run above is the causal result. A separate-process
calibration measures distance to the checkpoint's optimized Transformers +
FLA reference:

| prompt | Tinyserve M7p | reference | Tinyserve / reference |
|---:|---:|---:|---:|
| 32 | 39.28 ms | 125.49 ms | 0.31× |
| 128 | 108.30 ms | 122.73 ms | 0.88× |
| 512 | 126.19 ms | 125.77 ms | 1.00× |
| 2,048 | 202.66 ms | 136.38 ms | 1.49× |

At 2,048 tokens the gap narrows from M7o's 1.99× to 1.49×. Tinyserve's
implementation is written from scratch and contains no FLA dependency; FLA is
used only as an external calibration boundary. Ollama, llama.cpp, FreeToken,
and ninfer remain excluded because the tiny structural Kimi-K3 checkpoint is
not a supported common model boundary for those engines.

The compact [M7p evidence
manifest](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7p-fused-kda-tail-a6000-2026-09-08.json) retains the
promotion gate, full-model numerical boundary, post-change profile, external
calibration, source hashes, and limitations.

## Follow one chunk in the debugger

The VS Code configuration `M7p: fused KDA correction and state tail` runs
`examples/generate.py` with a prompt that crosses the 64-token inner boundary.
Break in `chunkwise_kda()` after `L` and `P` are built, then step into
`fused_kda_tail()`.

For the first chunk, inspect these shapes:

```text
q, k, cumulative_g: [1, 8, 64, 128]
L, P:                [1, 8, 64, 64]
S_in:                [1, 8, 128, 128]
W, Y:                [1, 8, 64, 128]
S_out:               [1, 8, 128, 128]
```

The next partial chunk receives only `S_out`; it does not receive the previous
chunk's `L`, `P`, or `W`. Force `--kda-kernel-backend torch` to walk the same
calculation through named PyTorch tensors.

M7p is successful because it changes execution granularity without hiding the
equation: causal rows remain sequential, value and state tiles remain
parallel, recurrent state still crosses chunk boundaries, and the old path
continues to define correctness.
