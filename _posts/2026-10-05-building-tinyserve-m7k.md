---
layout: post
math: true
title: "Building tinyserve M7k: Make GDN prefill chunkwise"
date: 2026-10-05 08:09:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Derive chunkwise Gated DeltaNet prefill from its token recurrence; keep single-token decode recurrent."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m7k-chunkwise-gdn.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M7j — Fuse Kimi's learned residual choice]({% include tinyserve-post-url.html slug="building-tinyserve-m7j" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7j-fused-kimi-residual.md" %}) · Next: [M7l — Make KDA prefill chunkwise]({% include tinyserve-post-url.html slug="building-tinyserve-m7l" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7l-chunkwise-kda.md" %})

M7e implemented Qwen3.5's Gated DeltaNet as a readable token recurrence. That
was the right correctness baseline and the wrong execution geometry for a long
prompt. Every one of the 18 GDN layers ran a Python loop whose next iteration
waited for the previous recurrent state. On one RTX A6000, a 2,048-token BF16
prompt plus one generated token took a median `4.401 s`.

Decode should remain recurrent because it receives one new token. Prefill has
many known tokens and can reorganize the same causal equations into bounded
matrix operations. M7k derives that chunkwise form directly in Tinyserve,
keeps the M7e loop as an executable oracle, and adds no FLA dependency.

[![A single decode token uses one recurrent state update, while prefill solves
each 64-token GDN chunk with cumulative decay, a lower-triangular system, and
matrix multiplications before carrying one final state to the next
chunk.](/assets/tinyserve/m7k-chunkwise-gdn.svg)](/assets/tinyserve/m7k-chunkwise-gdn.svg)

## Two meanings of chunking

M5 and M7h already split prompt work at the scheduler. That mechanism answers:
**how many prompt tokens may run between decoder iterations?** A scheduler
budget of 512 can admit 512 real tokens, possibly packed from several requests.

M7k adds chunking *inside the GDN computation*. It answers: **how should one
model forward evaluate the admitted tokens?** The default kernel chunk is 64
tokens. For a 512-token scheduler action, one GDN layer therefore executes
eight 64-token chunks and carries its fixed recurrent state between them.

Changing the GDN chunk size does not change FCFS order, request admission,
decode interleaving, or how many prompt tokens the scheduler spends.

## Start from the token recurrence

For one head, let $S$ be the `[K,V]` recurrent matrix. Token $t$ has scalar
decay $a_t=e^{g_t}$, update rate $\beta_t$, key $k_t$, value $v_t$, and query
$q_t$. The readable implementation performs:

$$
D_t = a_t S_{t-1},
\qquad
u_t = \beta_t\left(v_t-k_t^\mathsf{T}D_t\right),
$$

$$
S_t = D_t+k_tu_t^\mathsf{T},
\qquad
o_t=q_t^\mathsf{T}S_t.
$$

The key first reads what the decayed state remembers. The error between that
memory and the new value becomes a rank-one correction. The recurrence is
causal because $u_t$ depends on every earlier correction through $S_{t-1}$.

The M7e implementation expresses those four lines almost literally. For
`T=1`, that is ideal teaching code and a reasonable decode path. For
`T=2048`, it creates 2,048 host-driven iterations in every GDN layer.

## A two-token example

Use scalar keys, values, and state so every matrix becomes one number. Start
with `state_in = 2`:

| token | decay $a$ | key $k$ | value $v$ | $\beta$ | decayed state | correction $u$ | new state |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5 | 1.0 | 3.0 | 0.5 | 1.0 | $0.5(3-1)=1$ | $1+1(1)=2$ |
| 1 | 0.5 | 0.5 | 2.0 | 1.0 | 1.0 | $1(2-0.5)=1.5$ | $1+0.5(1.5)=1.75$ |

The second correction depends on the first, so computing the two rows as if
they were independent would be wrong. Chunkwise execution must preserve that
dependency rather than discard it.

## Turn the dependencies into a triangular solve

Within one chunk, define cumulative decay

$$
A_t=\prod_{j=0}^{t} a_j=e^{\sum_{j=0}^{t}g_j}.
$$

Expanding the state recurrence lets every correction be written in terms of
the incoming chunk state and earlier corrections:

$$
u_t+
\sum_{i<t}
\beta_t\frac{A_t}{A_i}(k_t^\mathsf{T}k_i)u_i
=
\beta_t\left(v_t-A_tk_t^\mathsf{T}S_{\mathrm{in}}\right).
$$

All coefficients with $i>t$ are zero. Stacking the token rows therefore gives
a lower-triangular system:

$$
(I+L)U=R.
$$

For the scalar example, $A_0=0.5$ and $A_1=0.25$. The same two corrections
come from:

```text
u0                         = 0.5 * (3 - 0.5 * 1.0 * 2) = 1.00
u1 + 1*(0.25/0.5)*(0.5*1)*u0 = 1.0 * (2 - 0.25 * 0.5 * 2)
u1 + 0.25*u0              = 1.75
u1                         = 1.50
```

Once `U` is known, masked matrix multiplications produce every query output
and the final state:

$$
S_t=A_tS_{\mathrm{in}}+
\sum_{i\le t}\frac{A_t}{A_i}k_iu_i^\mathsf{T}.
$$

The triangular solve does not make causality disappear. It packages the same
causal dependencies into GPU-friendly matrix work. Only the final `[K,V]`
state crosses from one chunk to the next.

## How the implementation maps to the math

`chunkwise_gated_delta()` keeps activations and recurrent state in FP32 for
the GDN calculation, matching the oracle's numerical boundary. One loop now
runs over chunks rather than tokens:

1. slice `Q`, `K`, `V`, `g`, and `beta` as `[B,H,C,D]`;
2. compute `cumulative_g` and pairwise relative decay;
3. build the strict lower triangle from `K @ K.T`;
4. solve for all correction rows with `torch.linalg.solve_triangular`;
5. compute all chunk outputs with causal matrix multiplication; and
6. reduce the chunk into one final state for the next iteration.

This is a from-scratch chunkwise algorithm composed from PyTorch/CUDA matrix
primitives. It is not yet one monolithic Triton kernel. That boundary keeps
the derivation inspectable; a future fusion should be justified by a new
profile rather than assumed necessary.

`gated_delta_rule()` owns phase dispatch:

| input | `auto` behavior |
|---|---|
| `T = 1` | readable recurrent update |
| `T > 1` | chunkwise prefill, default `C = 64` |

`--gdn-prefill-backend recurrent` forces the old prefill oracle.
`--gdn-prefill-backend chunkwise` forces the new prefill policy, while decode
still stays recurrent. `--gdn-chunk-size` selects the inner kernel chunk and
is deliberately separate from the scheduler's `--chunk-size`.

## Padding must be an identity transition

M7h packs unequal prompt chunks into a right-padded batch. A padded GDN row
must not decay or update request state. Before the triangular system is built,
M7k maps every padded position to:

```text
g = 0       -> exp(g) = 1, so state does not decay
beta = 0    -> correction is zero
k = v = 0  -> poisoned padding cannot enter a matrix product
```

The focused test fills padded `K`, `V`, `g`, and `beta` with NaNs. Real outputs
and final states still match the recurrent oracle. This test is stronger than
using zero padding because it catches a masked value that was read too early.

## Numerical boundary

The direct equation tests compare chunk sizes 1, 16, 32, and 64, including a
partial final chunk, against the token recurrence at `atol=rtol=1e-5`. A
separate dispatch test replaces the chunkwise function with a failure and
proves that single-token decode never calls it.

BF16 matrix reductions do not execute in token-loop order. On fixed 128- and
2,048-token prompts, the full model produced these differences relative to
recurrent prefill:

| prompt | maximum logit difference | maximum recurrent-state difference | top five | argmax |
|---:|---:|---:|---|---|
| 128 | 0.125 | 0.0347 | identical and ordered | identical |
| 2,048 | 0.141 | 0.0373 | identical and ordered | identical |

The paired performance run also required the first generated token to match
for every repetition and chunk size. Exact long continuations are not a sound
BF16 acceptance rule: on the short `2+2=` smoke, both paths first choose `4`,
but a later near-boundary decision can choose a newline or comma after the
state reductions have been reordered. The FP32 equation, leading-candidate,
padding, and cache tests define the correctness boundary instead.

## Measured effect

The September 4, 2026 paired run used one loaded Qwen3.5-0.8B model in BF16 on
one RTX A6000. Each path had two warmups and five measured repetitions. Path
order was forward on even repetitions and reversed on odd repetitions. Timing
includes tokenization, cache allocation, one prompt forward, sampling, and one
generated token.

| prompt | recurrent | chunk 16 | chunk 32 | chunk 64 | chunk-64 speedup |
|---:|---:|---:|---:|---:|---:|
| 128 | 314.8 ms | 106.0 ms | 68.3 ms | 48.6 ms | 6.48× |
| 2,048 | 4,401.1 ms | 1,322.8 ms | 678.2 ms | 360.1 ms | 12.22× |

Chunk 64 wins both retained lengths. Its peak-allocation increase over the
recurrent path is about `0.8 MiB` at 128 tokens and `1.3 MiB` at 2,048 tokens,
consistent with bounded per-chunk matrices rather than prompt-length state.

## Same-checkpoint calibration

The external calibration ran the local Transformers PyTorch implementation
with the same checkpoint, dtype, prompt lengths, warmups, repetitions, and
one-token output boundary:

| prompt | Tinyserve recurrent | Tinyserve chunk 64 | Transformers | Tinyserve chunk 64 vs Transformers |
|---:|---:|---:|---:|---:|
| 128 | 314.8 ms | 48.6 ms | 151.5 ms | 3.12× faster |
| 2,048 | 4,401.1 ms | 360.1 ms | 326.3 ms | 1.10× slower |

Only the Tinyserve recurrent/chunkwise comparison alternates paths inside one
loaded process, so it is the causal M7k result. The Transformers process is a
calibration, not a claim of general engine superiority. It shows that M7k
removes the old 2,048-token gap and reaches the same latency class without an
FLA backend; the remaining `33.7 ms` gap is the next profiling question.

## Run and debug it

The **M7k: chunkwise GDN prefill** launch configuration uses `generate.py` and
a 16-token inner chunk so a short teaching prompt crosses several boundaries.
Set breakpoints in this order:

1. `gated_delta_rule()` — inspect `T` and the phase dispatch;
2. `chunkwise_gated_delta()` at `start` and `end` — identify one kernel chunk;
3. `cumulative_g` and `relative_decay` — follow scalar decay through the
   `[B,H,C,C]` causal matrix;
4. `lower` and `correction` — inspect `(I+L)U=R`; and
5. the final `state += ...` — see the only value carried to the next chunk.

```bash
.venv/bin/python examples/generate.py \
  --model /home/scratch.kaix_coreai/models/Qwen3.5-0.8B \
  --prompt "Explain why a long language-model prompt should use matrix-shaped chunkwise Gated DeltaNet prefill while one-token decode should keep a recurrent state update. Use a short concrete example and distinguish kernel chunks from the serving scheduler token budget." \
  --no-chat --max-new-tokens 1 \
  --gdn-prefill-backend chunkwise --gdn-chunk-size 16
```

Force `recurrent` to single-step the original token equation. The paired
benchmark records every order, latency, first token, peak allocation,
checkpoint/source hashes, and GPU telemetry:

```bash
.venv/bin/python examples/bench_gdn.py \
  --device cuda:0 --prompt-tokens 128 2048 \
  --chunk-sizes 16 32 64 --warmups 2 --repeats 5 \
  --output .tinyserve-bench/m7k/gdn-ab.json
```

The structured [M7k evidence](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7k-chunkwise-gdn-a6000-2026-09-04.json)
retains the medians, parity limits, implementation fingerprint, and raw
artifact hashes.

## Takeaway

Linear recurrence describes model semantics, not the only valid execution
order. Decode has one new token and should update state once. Prefill has many
known causal transitions and should expose them as chunk-local matrix work.

GDN is the right first lesson because its decay is one scalar per token and
head. KDA's decay is a vector over key channels, so it needs a different
chunkwise derivation. M7l can reuse the phase dispatch, padding contract,
benchmark protocol, and state-carry idea without pretending the two kernels
have identical algebra.
