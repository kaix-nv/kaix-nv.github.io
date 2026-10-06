---
layout: post
math: true
title: "Building tinyserve M7l: Make KDA prefill chunkwise"
date: 2026-10-05 08:10:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Turn channel-wise Kimi Delta Attention into bounded chunk computations with explicit state carry."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m7l-chunkwise-kda.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M7k — Make GDN prefill chunkwise]({% include tinyserve-post-url.html slug="building-tinyserve-m7k" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7k-chunkwise-gdn.md" %}) · Next: [M7m — Count the launches before writing the next KDA kernel]({% include tinyserve-post-url.html slug="building-tinyserve-m7m" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7m-kimi-prefill-profile.md" %})

M7f began with Kimi Delta Attention (KDA) written as a token recurrence. That
implementation made the unusual state transition visible and gave us a
trustworthy oracle, but it also launched a long chain of small GPU operations
during prefill. M7l keeps that recurrence for decode and short inputs, then
solves longer prefills in 32-token chunks.

This is not the GDN function renamed. GDN applies one scalar decay to a whole
head; KDA applies a different decay to every key channel. That extra dimension
changes both the algebra and the memory problem.

[![KDA uses the recurrent equation for decode and short inputs. Longer prefills
compute per-channel cumulative decay, reduce bounded channel tiles into causal
token-pair matrices, solve one lower-triangular system, and carry only the
final state to the next chunk.](/assets/tinyserve/m7l-chunkwise-kda.svg)](/assets/tinyserve/m7l-chunkwise-kda.svg)

## The serial equation

For one head, KDA keeps a `[K,V]` matrix $S$. Token $t$ supplies key $k_t$,
value $v_t$, scaled query $q_t$, an effective update rate
$b_t=\sigma(\beta_t)$, and a vector of log-decays $g_t\in\mathbb{R}^K$:

$$
D_t=\operatorname{diag}(e^{g_t})S_{t-1},
\qquad
u_t=b_t\left(v_t-k_t^\mathsf{T}D_t\right),
$$

$$
S_t=D_t+k_tu_t^\mathsf{T},
\qquad
o_t=q_t^\mathsf{T}S_t.
$$

Read the four operations in order:

1. decay each row of the old state by its own channel gate;
2. ask what value the decayed state recalls for the new key;
3. write the gated prediction error back as a rank-one update; and
4. read the updated state with the query.

That direct loop is ideal at decode, where `T=1`. During a known prompt, token
$t+1$ waits for token $t$ even though the GPU could perform much larger matrix
operations.

## A concrete two-token trace

Use two key channels, one value channel, and `state_in = [0, 0]`. The numbers
below use the effective $b$ after the sigmoid:

| token | channel decay $a=e^g$ | key $k$ | value $v$ | $b$ | recalled value | correction $u$ | new state |
|---:|---|---|---:|---:|---:|---:|---|
| 0 | `[0.5, 0.25]` | `[1, 2]` | 3 | 0.5 | 0 | 1.5 | `[1.5, 3.0]` |
| 1 | `[0.5, 1.0]` | `[2, 1]` | 5 | 1.0 | 4.5 | 0.5 | `[1.75, 3.5]` |

Token 1 recalls `2×0.75 + 1×3.0 = 4.5`. Its correction therefore depends on
token 0. Computing both corrections independently would return the wrong
state.

Chunkwise KDA preserves the dependency by moving it into a triangular system.
Let the cumulative channel decay inside a chunk be

$$
G_t=\sum_{j=0}^{t}g_j,
\qquad
A_t=e^{G_t}.
$$

Expanding the recurrence gives

$$
u_t+
\sum_{i<t} b_t
\left[\sum_{d=1}^{K}
k_t[d]k_i[d]e^{G_t[d]-G_i[d]}\right]u_i
=
b_t\left(v_t-k_t^\mathsf{T}\operatorname{diag}(A_t)S_\text{in}\right).
$$

For the example, $A_0=[0.5,0.25]$ and $A_1=[0.25,0.25]$. The dependency from
token 0 to token 1 is

$$
L_{1,0}
=2\cdot1\cdot\frac{0.25}{0.5}
+1\cdot2\cdot\frac{0.25}{0.25}
=3.
$$

The two serial corrections are recovered together:

```text
u0          = 1.5
u1 + 3 * u0 = 5.0
u1          = 0.5
```

Stacking a whole chunk produces $(I+L)U=R$, where $L$ is strictly lower
triangular. The solve packages the same causal dependencies into matrix-shaped
work; it does not make KDA non-causal.

## Why KDA needs channel tiling

For scalar-decay GDN, the pairwise decay ratio has shape `[B,H,C,C]`. KDA's
ratio is different for every key channel, so the direct intermediate would be
`[B,H,C,C,K]`. If `C` grew with the prompt, this tensor would grow
quadratically in tokens and linearly in key width.

`chunkwise_kda()` bounds both dimensions:

- it limits the token chunk to `C=32` by default;
- it visits the key dimension in 32-channel tiles; and
- it immediately reduces each tile into `[B,H,C,C]` key-pair and query-pair
  matrices.

The implementation never materializes a prompt-sized `[T,T,K]` tensor. Only
the fixed `[K,V]` recurrent state crosses a chunk boundary.

There is one numerical detail worth seeing in the code. For a causal pair we
need $G_t-G_i$ only when $i\le t$. Above the diagonal that difference can be
positive, so exponentiating first can overflow even though the value will be
masked away later. M7l writes `-inf` above the diagonal *before* `exp()`.

## Follow the implementation

The core path is deliberately a short sequence of recognizable operations:

1. `cumulative_g = g_chunk.cumsum(...)` builds $G_t$ per channel;
2. the `channel_start` loop accumulates decayed key and query pair scores;
3. `lower` and `rhs` form the causal system $(I+L)U=R$;
4. `torch.linalg.solve_triangular()` obtains every correction in the chunk;
5. `query_pairs @ correction` produces every token output; and
6. the final cumulative decay and corrections reduce the chunk to one state.

This is a from-scratch algorithm built from PyTorch/CUDA primitives. Tinyserve
does not call FLA for this path, and it retains `kda_recurrence()` as a
selectable semantic oracle. A future fused kernel can replace these primitives
only after profiling identifies the next dominant cost.

## Decode and the crossover

Chunk setup has a cost. On the tiny Kimi checkpoint, the isolated 8-token rule
was slower chunkwise, while 16 tokens was already faster. `kda_rule()` therefore
dispatches by the work presented to one model forward:

| input length | `auto` path | reason |
|---:|---|---|
| `T = 1` | recurrent | decode needs exactly one state update |
| `2 ≤ T < 16` | recurrent | measured setup cost exceeds saved loop overhead |
| `T ≥ 16` | chunkwise, `C = 32` | matrix work wins at the measured crossover |

`--kda-prefill-backend recurrent` forces the oracle.
`--kda-prefill-backend chunkwise` forces the new prefill implementation, but
the explicit `T=1` rule still keeps decode recurrent. `--kda-chunk-size`
changes only the inner KDA computation.

This is separate from M5/M7h scheduler chunking. The scheduler decides how
many prompt tokens may enter a model step and which requests are packed into
that step. KDA kernel chunking decides how one KDA layer evaluates the tokens
it has already received.

## Packed padding is an identity transition

M7h may pack unequal request chunks into one right-padded batch. Padding must
not age or update KDA state. Before any exponential or matrix product, M7l
maps padded positions to

```text
g = 0       -> channel decay is 1
b = 0       -> correction is zero
k = v = 0  -> padded data cannot enter a pair reduction
```

The focused test replaces padded `Q`, `K`, `V`, `g`, and `beta` with NaNs. Real
outputs and final state still match the recurrence. Zero-only padding would
not catch a masked value consumed before the mask was applied.

## Verification boundary

The equation tests compare chunk sizes 1, 8, 16, and 32 across a 17-token
input, including partial final chunks and a key dimension that is not divisible
by the 3-channel test tile. Separate dispatch tests prove that `T=1` and `T=8`
remain recurrent and that `T=16` switches to chunkwise execution.

At the full-model boundary, fixed 24-, 32-, and 128-token BF16 inputs produced
zero representable logit difference, identical ordered top-five candidates,
and the same argmax. The paired benchmark also required every recurrent and
chunk-size path to generate the same first token for the same prompt.

## Measured effect

The September 4, 2026 paired run used one loaded tiny Kimi-K3 checkpoint in
BF16 on one RTX A6000. It used two warmups and five measured repetitions, with
forward path order on even repetitions and reverse order on odd repetitions.
Whole-model timing includes tokenization, Kimi cache allocation, prompt
forward, sampling, and one generated token.

First isolate the KDA rule with chunk size 32:

| tokens | recurrent | chunkwise | speedup |
|---:|---:|---:|---:|
| 8 | 0.927 ms | 1.113 ms | 0.83× |
| 16 | 1.782 ms | 1.135 ms | 1.57× |
| 32 | 3.405 ms | 1.147 ms | 2.97× |
| 128 | 14.019 ms | 4.369 ms | 3.21× |

The full prompt-plus-one-token measurement shows the effect after the other Kimi
layers are included:

| prompt | recurrent | chunk 8 | chunk 16 | chunk 32 | chunk-32 speedup |
|---:|---:|---:|---:|---:|---:|
| 24 | 73.3 ms | 84.2 ms | 68.4 ms | 53.2 ms | 1.38× |
| 32 | 78.9 ms | 88.3 ms | 65.5 ms | 52.5 ms | 1.50× |
| 128 | 300.4 ms | 323.6 ms | 241.8 ms | 185.5 ms | 1.62× |

Chunk 32 wins all three retained full-model cases. At 128 tokens it increases
peak allocated memory from `451.1 MB` to `455.5 MB`; the bounded pair matrices
cost about `4.4 MB` while removing the serial prompt loop.

## Same-checkpoint calibration

The external reference is the checkpoint's Transformers implementation, which
currently imports FLA. Tinyserve's implementation does not. This separate
process is a calibration point, not the causal M7l comparison:

| prompt | Tinyserve recurrent | Tinyserve chunk 32 | Transformers + FLA | chunk 32 vs reference |
|---:|---:|---:|---:|---:|
| 24 | 73.3 ms | 53.2 ms | 151.4 ms | 2.84× faster |
| 32 | 78.9 ms | 52.5 ms | 180.3 ms | 3.44× faster |
| 128 | 300.4 ms | 185.5 ms | 153.5 ms | 1.21× slower |

Only recurrent versus chunkwise alternates inside one loaded process, so that
is the feature result. The reference timing is noisy at these tiny prompt
lengths and includes a different framework path. It shows that the new
implementation is competitive for short prefill but still leaves a long-prompt
gap worth profiling; it does not establish general engine superiority.

## Run and debug it

The **M7l: chunkwise KDA prefill** launch configuration uses `generate.py`, a
prompt longer than the 16-token crossover, and an 8-token KDA chunk so the
debugger crosses several boundaries. Set breakpoints in this order:

1. `kda_rule()` — inspect `T` and the phase dispatch;
2. `chunkwise_kda()` at `start`/`end` — identify one token chunk;
3. `cumulative_g` — see `[B,H,C,K]` channel-wise prefix decay;
4. the `channel_start` loop — watch channel tiles reduce into `[B,H,C,C]`;
5. `solve_triangular()` — inspect the causal corrections; and
6. the final `state += ...` — see the only state carried to the next chunk.

```bash
.venv/bin/python examples/generate.py \
  --model /home/scratch.kaix_coreai/models/kimi-k3 \
  --prompt "Explain why Kimi Delta Attention uses a recurrent update for one-token decode but a channel-wise chunked solve for a known prompt. Include a two-token example." \
  --no-chat --max-new-tokens 1 \
  --kda-prefill-backend chunkwise --kda-chunk-size 8 \
  --kda-kernel-backend torch
```

Force `recurrent` to single-step the oracle. The paired benchmark records
individual runs, output tokens, peak allocation, source/checkpoint hashes, and
GPU telemetry:

```bash
.venv/bin/python examples/bench_kda.py \
  --device cuda:0 --prompt-tokens 24 32 128 \
  --chunk-sizes 8 16 32 --micro-lengths 8 16 32 128 \
  --kda-kernel-backend torch \
  --warmups 2 --repeats 5 \
  --output .tinyserve-bench/m7l/kda-ab.json
```

The compact [M7l evidence](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7l-chunkwise-kda-a6000-2026-09-04.json)
retains the protocol, numerical checks, medians, hashes, and limitations.
