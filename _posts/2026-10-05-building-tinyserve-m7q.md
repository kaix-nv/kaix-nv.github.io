---
layout: post
math: true
title: "Building tinyserve M7q: Submit KDA chunks once, carry state on the GPU"
date: 2026-10-05 08:15:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Replace repeated Python chunk submissions with a chunk grid and GPU-resident recurrent state carry."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m7q-multichunk-kda-prefill.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M7p — Fuse KDA's correction and state tail]({% include tinyserve-post-url.html slug="building-tinyserve-m7p" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7p-fused-kda-tail.md" %}) · Next: [M7r — Measure the MoE launch storm before writing a grouped kernel]({% include tinyserve-post-url.html slug="building-tinyserve-m7r" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7r-kimi-moe-profile.md" %})

M7p made the arithmetic inside one 64-token KDA chunk inexpensive, but a
2,048-token prompt still returned to Python 32 times in every KDA layer. The
isolated rule launched 32 cumulative sums, 32 exponentials, 32 pair kernels,
32 correction/output kernels, and 32 state-update kernels. Only 4.78 ms was
useful GPU work; the complete chunkwise interval was much larger because the
GPU repeatedly waited for the next submission.

M7q changes the submission granularity without changing the KDA recurrence.
One cumulative-sum launch covers every chunk, one pair-kernel grid builds all
independent pair matrices, and one ordered tail launch carries recurrent state
from chunk 0 through the final chunk. The M7p loop and the readable PyTorch
equation remain selectable controls.

[![M7p returns to Python for cumulative decay, pair construction, correction,
and state update for every 64-token chunk. M7q computes cumulative decay and
pair matrices for the entire chunk grid, then one tail program visits chunks
in order while keeping its recurrent-state value tile local.](/assets/tinyserve/m7q-multichunk-kda-prefill.svg)](/assets/tinyserve/m7q-multichunk-kda-prefill.svg)

## The equation was already fast; its launch schedule was not

For one batch row and one head, M7p processes a prompt of length `T` as
`N = T / C` inner chunks. The measured Kimi shape uses:

- `C=64` tokens per complete inner chunk;
- `K=128` key channels;
- `V=128` value channels; and
- recurrent state `S` with shape `[K,V]`.

Inside chunk `n`, cumulative log decay is `G_n`, and M7o's pair kernel builds
two `[C,C]` matrices: the unit lower-triangular system `L_n` and the query-key
pair matrix `P_n`. M7p then evaluates:

$$
R_n = b_n \odot \left(V_n - (K_n \odot e^{G_n})S_n\right),
\qquad
W_n = L_n^{-1}R_n,
$$

$$
Y_n = (Q_n \odot e^{G_n})S_n + P_nW_n,
\qquad
S_{n+1} = \operatorname{update}(S_n, K_n, G_n, W_n).
$$

The last expression creates a real dependency:

```text
S0 --chunk 0--> S1 --chunk 1--> S2 -- ... --> SN
```

That does **not** mean every operation depends on the prior state. `G_n`,
`L_n`, and `P_n` depend only on tokens inside chunk `n`, so all chunks can
build them together. Only the correction/output/state tail must visit chunks
in order.

## A concrete 128-token trace

Take a 128-token prompt. It has two 64-token chunks. M7p asks Python to submit
this sequence:

```text
chunk 0:
  cumsum(g0) -> exp(G0) -> build L0/P0
  solve W0 -> emit Y0 -> write S1

chunk 1:
  cumsum(g1) -> exp(G1) -> build L1/P1
  solve W1 using S1 -> emit Y1 -> write S2
```

The second solve really needs `S1`, but the second cumulative sum and pair
matrices do not. M7q submits the same work as:

```text
one cumsum:     [g0, g1] -> [G0, G1]
one pair grid:  chunk 0 -> L0/P0       chunk 1 -> L1/P1

one tail program for values 0..15:
  load S0[:, 0:16]
  chunk 0: solve W0[:, 0:16], emit Y0[:, 0:16], form S1[:, 0:16]
  chunk 1: solve W1[:, 0:16], emit Y1[:, 0:16], form S2[:, 0:16]
  store S2[:, 0:16]
```

Seven other tail programs do the same for values `16..31`, ..., `112..127`.
Value tiles are independent, so they run in parallel. Within one tile, chunks
and the 64 forward-substitution rows inside each chunk remain causal.

## Stage one: expose a chunk grid

`fused_kda_multichunk()` views token-major inputs as:

```text
Q, K, G: [B,H,N,C,128]
V:       [B,H,N,C,128]
beta:    [B,H,N,C]
```

One `cumsum(dim=-2)` computes the 64 prefix values independently for every
`[B,H,N,K]` row. The pair kernel adds `N` to its launch grid, so programs for
different chunks build `L_n` and `P_n` concurrently. No prompt-sized
`[T,T,K]` tensor is created: pair storage remains block diagonal, with shape
`[B,H,N,C,C]`.

M7q also stops materializing `exp(G)` as a prompt-sized tensor. The ordered
tail loads `G` and computes the exponential where the two incoming-state
projections consume it. This removes the separate exponential launch and
temporary without changing its FP32 value.

## Stage two: carry one state tile through all chunks

`_kda_multichunk_tail_kernel` launches a grid over `(batch × head, value
tile)`. One program owns a `[128,16]` slice of recurrent state:

```text
state_block = S_in[:, values]
for chunk in prompt order:
    remembered = (K_chunk * exp(G_chunk)) @ state_block
    base_output = (Q_chunk * exp(G_chunk)) @ state_block
    solve correction rows 0, 1, ..., 63
    emit this chunk's 16 output channels
    state_block = decay(state_block) + token_writes
S_out[:, values] = state_block
```

The state no longer makes 32 round trips through separate state-update and
correction launches. It remains live across the chunk loop and is written once
at the end. This is a persistent *program lifetime* within one kernel launch;
it is not a background service kernel.

The pair stage deliberately remains separate. `L_n` and `P_n` are shared by
all eight value tiles. Computing them inside the tail would repeat the same
pair reductions eight times, trading launch overhead for redundant math.

## Padding is still an identity transition

The scheduler represents a ragged batch with a Boolean token mask. Before the
multi-chunk view is formed, M7q zeros masked query, key, value, decay, and beta
rows in five vectorized operations. A padded row therefore contributes:

```text
g = 0       -> no additional decay
k = 0       -> no state read or write
beta = 0    -> zero correction
q = v = 0   -> zero output
```

The test uses a two-row 128-token batch, gives the second row only 65 real
tokens, fills its padding with `NaN`, and obtains finite output and state that
match M7p. The mask work is submitted once for the whole prompt, rather than
once per chunk.

## The narrow native contract and fallback

The multi-chunk path is intentionally limited to CUDA inference with FP32
working tensors, `K=V=128`, `C=64`, at least two chunks, and a prompt width
that is exactly divisible by 64. It supports an optional right-padding mask.

`--kda-submission-backend` makes the boundary visible:

- `auto` selects M7q when the contract fits and otherwise keeps the M7p loop;
- `chunk_loop` forces M7p's per-chunk submission; and
- `multichunk` requires M7q and raises for an unsupported input.

Lengths 63, 64, 65, 127, and 129 exercise the `auto` fallback. Short and
partial prompts retain M7p because padding every request up to another full
chunk was not justified by the retained measurements. Decode remains the
one-token recurrent equation.

## Storage tradeoff

M7p keeps `L` and `P` for one chunk at a time. M7q must make every chunk's pair
matrices available before the ordered tail begins. For the measured
2,048-token shape:

```text
2 matrices × 1 batch × 8 heads × 32 chunks × 64 × 64 × 4 bytes
    = 8 MiB
```

This is bounded block-diagonal storage, not quadratic storage in the whole
prompt. The retained peak-allocation measurement does not increase at 512 or
2,048 tokens because M7q removes the per-chunk correction list and final
concatenation while adding the all-chunk pair matrices. The exact balance is
shape-dependent and must be remeasured for a different Kimi configuration.

## Correctness boundary

The focused CUDA tests compare M7q with the native M7p loop for one and two
batch rows, nonzero incoming state, two and three chunks, partial-boundary
fallbacks, and poisoned right padding. The complete model comparison uses
fresh BF16 caches and exact prompt lengths:

| prompt | max logit difference | max state difference | ordered top 5 | argmax |
|---:|---:|---:|:---:|:---:|
| 32 | 0 | 0 | equal | equal |
| 128 | 0 | 0 | equal | equal |
| 512 | 0 | 0 | equal | equal |
| 2,048 | 0 | 0 | equal | equal |

The complete suite passes 125 tests; nine two-GPU TP/PP/EP/CP tests are
expected to skip when only one GPU is visible.

## Paired promotion gate

The causal comparison keeps one BF16 model loaded and alternates M7p and M7q
on every retained repetition. Both use `C=64`, the same pair arithmetic, and
the same fused correction/state equations; only submission granularity and
state lifetime change. Each prompt has two warmups and five retained runs and
generates one token.

| prompt | M7p loop | M7q multi-chunk | lower latency | peak-memory change | first token |
|---:|---:|---:|---:|---:|:---:|
| 32 | 40.56 ms | 39.35 ms | 2.98% | 0 MiB | equal |
| 128 | 108.51 ms | 104.36 ms | 3.83% | −0.50 MiB | equal |
| 512 | 127.34 ms | 113.15 ms | 11.14% | 0 MiB | equal |
| 2,048 | 207.74 ms | 168.01 ms | 19.13% | 0 MiB | equal |

M7q passes the declared gate: at least 10% lower latency at 512 and 2,048
tokens, no short-prompt regression above 5%, less than 32 MiB extra peak
allocation at 512, and equal state, logit, top-5, argmax, and sampled-token
decisions.

The isolated rule shows where the improvement begins:

| tokens | M7p loop | M7q multi-chunk | speedup |
|---:|---:|---:|---:|
| 32 | 0.304 ms | 0.310 ms | 0.98× |
| 128 | 0.524 ms | 0.382 ms | 1.37× |
| 512 | 1.739 ms | 1.347 ms | 1.29× |
| 2,048 | 6.739 ms | 5.249 ms | 1.28× |

## Profile again: the host loop is gone

The separate post-change diagnostic makes the launch reduction explicit for
one 2,048-token KDA rule:

| operation | M7p calls | M7q calls |
|---|---:|---:|
| `aten::cumsum` | 32 | 1 |
| `aten::exp` for cumulative decay | 32 | 0 |
| pair kernel | 32 | 1 |
| correction/output kernel | 32 | 0 |
| state-update kernel | 32 | 0 |
| ordered multi-chunk tail | 0 | 1 |

The pair and ordered-tail kernels consume 0.310 ms and 4.382 ms of self CUDA
time. That 4.69 ms total is close to M7p's 4.78 ms of listed GPU work: M7q
does not win by deleting the recurrence. It wins by submitting the same work
at a useful granularity. Retained PyTorch self-CPU operator time falls from
6.95 ms to 0.35 ms.

| diagnostic interval | M7p | M7q | reduction |
|---|---:|---:|---:|
| model | 316.82 ms | 199.75 ms | 36.95% |
| KDA | 174.61 ms | 65.29 ms | 62.61% |
| chunkwise KDA rule | 153.14 ms | 46.19 ms | 69.84% |
| isolated PyTorch self-CPU operators | 6.95 ms | 0.35 ms | 94.90% |

CUDA-event module intervals include stream gaps and profiler overhead; they
are diagnostic rather than additive useful-kernel time. The interleaved table
above is the causal end-to-end latency result.

The bottleneck has now moved. At 2,048 tokens, KDA is 32.7% of the profiled
model interval while MoE is 53.2%. Another speculative KDA fusion is therefore
less justified than profiling the selected-expert path.

## Same-checkpoint calibration

A separate process compares the current Tinyserve implementation with the
checkpoint's optimized Transformers + FLA reference. FLA is a calibration
dependency only; M7q's kernel is written from scratch and imports no FLA code.

| prompt | Tinyserve M7q | reference | Tinyserve / reference |
|---:|---:|---:|---:|
| 32 | 38.97 ms | 127.26 ms | 0.31× |
| 128 | 104.29 ms | 128.34 ms | 0.81× |
| 512 | 109.53 ms | 127.50 ms | 0.86× |
| 2,048 | 167.52 ms | 142.68 ms | 1.17× |

The long-prompt calibration gap is now 1.17×. This is not a paired causal
comparison—the candidate and reference load in separate processes—so it does
not replace the M7p/M7q promotion experiment. Ollama, llama.cpp, FreeToken,
and ninfer remain outside this calibration because the tiny structural
Kimi-K3 checkpoint is not a supported common model boundary for them.

The compact [M7q evidence
manifest](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7q-multichunk-kda-prefill-a6000-2026-09-08.json) retains
the gate, numerical boundary, profile, calibration, source hashes, and
limitations.

## Follow two chunks in the debugger

The VS Code configuration `M7q: multi-chunk KDA prefill` uses
`examples/generate.py` with exactly 128 raw prompt tokens and forces the native
path. Break in `chunkwise_kda()` at `native_multichunk`, then step into
`fused_kda_multichunk()`.

Inspect these shapes:

```text
query:          [1,128,8,128]
query_chunks:   [1,8,2,64,128]
cumulative_g:   [1,8,2,64,128]
lower, pairs:   [1,8,2,64,64]
output:         [1,8,2,64,128]
state:          [1,8,128,128]
```

The Python debugger cannot step through individual Triton program instances.
Use the figure and kernel comments to follow the device loop: pair programs
may process chunk 0 and chunk 1 in any order, while each tail program must
update `state_block` in chunk order. Change the launch option to
`--kda-submission-backend chunk_loop` to walk M7p's visible Python loop with
the same model and prompt.
