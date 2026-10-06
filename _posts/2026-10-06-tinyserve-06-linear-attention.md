---
layout: post
math: true
title: 'Tinyserve, Chapter 6: Linear attention and hybrid models'
date: 2026-10-06 00:00:00 -0700
categories:
- tinyserve
- llm-serving
excerpt: How do recurrent memory and chunkwise prefill work together?
book_chapter: 6
source_revision: 14b0c8dbb975350bbd730ac0c702eafce36eb95c
source_document: docs/book/06-linear-attention.md
code_revision: e20a34815498560f9226a9e057ade31b2b62743f
---

{% include tinyserve-article-style.html %}
{% include tinyserve-book-nav.html %}

*Chapter 6 · Model Architecture and Computation*

An ordinary attention layer remembers one key and value per token. A
recurrent linear-attention layer instead updates a fixed matrix that
summarizes earlier tokens. Decode reads and modifies that matrix once;
it does not scan a growing list of keys. The change is architectural:
the model must be trained for this memory and its update rule.

Tinyserve implements two related recurrences: Gated DeltaNet, or GDN, in
its Qwen hybrid model, and Kimi Delta Attention, or KDA, in Kimi. Their
state updates look similar, but scalar versus channel-wise decay changes
the prefill algorithm. We will derive both, connect serial updates to
chunkwise matrix work, and account for the ordinary attention history
that remains in a hybrid model.

## Read and write a matrix memory

For one head, let the state S have shape `[K,V]`. Here K and V are
**dimension sizes**, not the key and value tensors. A key vector k has
K coordinates, a value v has V coordinates, and a query q has K
coordinates. Reading `k^T S` returns a V-coordinate value prediction;
reading `q^T S` returns the head's output.

A simple additive memory would write the outer product `k v^T`:

$$
S_t=S_{t-1}+k_tv_t^\mathsf T,\qquad o_t=q_t^\mathsf TS_t.
$$

Expanding S shows how earlier keys and values contribute to the read:
`o_t = sum_{i<=t}(q_t^T k_i) v_i`, plus any incoming-state contribution.
This identity explains the recurrent representation. It is not softmax
attention: there is no exponential row normalization, and dot products
can be negative. Practical linear-attention architectures define their
own normalization, gating, and update equations.

An additive write also ignores what the state already knows about the
key. A delta rule first recalls that association and writes a correction.
GDN adds a learned decay before that correction:

$$
D_t=a_tS_{t-1},\qquad
u_t=\beta_t(v_t-k_t^\mathsf TD_t),
$$

$$
S_t=D_t+k_tu_t^\mathsf T,\qquad
o_t=q_t^\mathsf TS_t.
$$

The decay `a_t = exp(g_t)` is one scalar per token and head. The write
strength beta is also scalar per token and head. The vector u has V
coordinates and contains the scaled prediction error.

Use a two-by-two incoming state equal to the identity, key `[1,0]`,
value `[3,4]`, decay `0.5`, and beta `0.25`. Decay gives
`D=[[0.5,0],[0,0.5]]`. The key recalls `[0.5,0]`; the error is `[2.5,4]`;
and the scaled correction is `[0.625,1]`. The new state is

```text
S = [[1.125, 1.0],
     [0.0,   0.5]]
```

A query `[0.5,1]` reads `[0.5625,1]`. The example treats q as the query
already supplied to the recurrence, including any intended scale. In
the model, Q/K normalization and query scaling happen at defined boundaries
before this read. These synthetic vectors explain the update rather than
claiming to be a complete model-layer fixture.

## Follow two tokens without hiding the dependency

Shrink to scalar state, key, and value so every operation is visible.
Start from `S_in=2`, and let both queries equal one:

| Token | Decay a | Key k | Value v | Beta | Decayed state D | Correction u | New state and output |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5 | 1 | 3 | 0.5 | 1 | 1 | 2 |
| 1 | 0.5 | 0.5 | 2 | 1 | 1 | 1.5 | 1.75 |

Token 1's recalled value is `0.5 × 1 = 0.5`, so its correction is
`2 − 0.5 = 1.5`. That decayed state contains token 0's write. Computing
both corrections independently from the original state would be wrong.

[![Two sequential delta updates decay memory, recall through the key, write the error, and read with the query. A triangular system recovers the same two corrections within one chunk.](/assets/tinyserve/book06-recurrence-chunk.svg)](/assets/tinyserve/book06-recurrence-chunk.svg)

For decode, this dependency is natural: one new input token means one
state transition. A long prompt has many known input tokens, however,
and a Python loop issues a succession of tiny dependent operations.
Chunkwise prefill reorganizes the dependency into a small triangular
system without changing the intended recurrence.

## Derive the chunkwise GDN equation

Within a chunk, define cumulative decay

$$
A_t=\prod_{j=0}^{t}a_j=e^{\sum_{j=0}^{t}g_j}.
$$

Expanding the state through all earlier writes yields

$$
S_t=A_tS_{\mathrm{in}}+
\sum_{i\le t}\frac{A_t}{A_i}k_i u_i^\mathsf T.
$$

Substitute the corresponding expansion of the decayed previous state
into the correction equation:

$$
u_t+\sum_{i<t}\beta_t\frac{A_t}{A_i}
(k_t^\mathsf Tk_i)u_i
=\beta_t(v_t-A_t k_t^\mathsf TS_{\mathrm{in}}).
$$

The coefficient of u_i is zero for every `i>=t` on the left-hand sum.
Stacking C correction vectors gives a unit-lower-triangular system
`(I+L)U=R`, with `L [C,C]`, `U [C,V]`, and `R [C,V]`. The incoming
state affects R; the token-pair coefficients in L depend only on the
chunk's keys, gates, and beta values.

For the scalar example, cumulative decays are `A0=0.5` and `A1=0.25`:

```text
u0             = 0.5 × (3 − 0.5 × 1 × 2) = 1
u1 + 0.25 u0   = 1   × (2 − 0.25 × 0.5 × 2) = 1.75
u1             = 1.5
```

The final state is `0.25×2 + (0.25/0.5)×1×1 + 0.5×1.5 = 1.75`.
The solve recovers the same corrections and final state as the token
loop. Causality remains in the triangular system; it has not disappeared.

`chunkwise_gated_delta()` constructs cumulative log decay, a key Gram
matrix, and the strict lower triangle, then calls
`torch.linalg.solve_triangular()`. Matrix products produce the output for
every query and the final state for the next chunk. This is Tinyserve's
own algorithm built from PyTorch operations, with the token recurrence
retained as its semantic oracle.

For fixed C, the pair matrices are bounded by C rather than total prompt
length T. Across the prompt there are roughly `T/C` chunks and `TC`
token-pair entries per head. The dependence on T is linear at fixed
chunk size, although its constant includes K, V, pair construction,
and triangular work. Larger C reduces chunk boundaries while increasing
within-chunk work and storage.

## KDA decays each key channel separately

GDN multiplies every row of one head's state by the same decay. KDA uses
a vector `a_t [K]`:

$$
D_t=\operatorname{diag}(a_t)S_{t-1},\qquad
u_t=b_t(v_t-k_t^\mathsf TD_t),
$$

$$
S_t=D_t+k_tu_t^\mathsf T,\qquad o_t=q_t^\mathsf TS_t.
$$

Each row corresponds to a key channel and has its own forgetting rate.
This is not a scalar broadcast over the entire head. In Tinyserve's
KDA function, the supplied beta is a logit and `b=sigmoid(beta)` is formed
inside the rule. GDN's function receives beta after its sigmoid. Confusing
those interfaces would apply the gate twice or omit it.

For a two-key-channel, one-value-channel algebra example, start at
`S_in=[0,0]`. Token 0 has decay `[0.5,0.25]`, key `[1,2]`, value 3,
and effective write rate 0.5. Its correction is 1.5 and new state
`[1.5,3]`. Token 1 uses decay `[0.5,1]`, key `[2,1]`, value 5, and
effective rate one. Decay gives `[0.75,3]`, recall gives 4.5, correction
is 0.5, and the new state is `[1.75,3.5]`. These illustrative keys omit
the surrounding model's L2 normalization to keep the arithmetic small.

Let `G_t[d]` be cumulative log decay through token t in channel d. The
strict lower-triangular coefficient becomes

$$
L_{t,i}=b_t\sum_{d=1}^{K}
k_t[d]k_i[d]e^{G_t[d]-G_i[d]},\qquad i<t.
$$

The scalar GDN decay can factor outside the key dot product; the KDA
channel decay cannot. In the example, the coefficient linking token 0
to token 1 is `2×1×0.5 + 1×2×1 = 3`. The chunk system is
`u0=1.5`, `u1+3u0=5`, giving the same `u1=0.5`.

A naive implementation would build `[B,H,C,C,K]` relative decays.
Tinyserve's readable `chunkwise_kda()` reduces channel tiles into
`[B,H,C,C]` key-pair and query-pair matrices instead. It masks the
noncausal region before exponentiation. Above the diagonal, reversed
log-decay differences can be positive and overflow even if a later mask
would discard them. Valid causal differences use nonpositive accumulated
log decays and do not require that invalid intermediate.

## The surrounding layer owns more than a matrix

The rule receives `Q,K [B,T,H,K]`, `V [B,T,H,V]`, and state
`[B,H,K,V]`. GDN log decay is `[B,T,H]`; KDA log decay is
`[B,T,H,K]`. Output is `[B,T,H,V]`. Both readable rules convert the
calculation to FP32, scale queries by `1/sqrt(K)`, and cast output back
to its input dtype. The carried state has its own storage contract.

`QwenHybridGatedDeltaNet.forward()` first projects Q/K/V together, applies
a causal depthwise convolution and SiLU, splits heads, and L2-normalizes
Q/K. If there are more value heads than key heads, it repeats Q/K over
the corresponding groups. Separate projections produce decay, beta,
and an output gate. The recurrence's result passes through gated RMSNorm
and `out_proj`.

`KimiDeltaAttention.forward()` uses separate Q/K/V projections and
convolutions, then L2 normalization for Q/K. A two-stage gate projection
creates one decay per channel. Its output normalization uses a sigmoid
output gate, whereas the Qwen hybrid uses a SiLU output gate. These
surrounding equations are part of each checkpoint's architecture.

A width-four causal convolution requires the previous three **raw
projected inputs**, before convolution and SiLU. State ownership therefore
includes convolution tails as well as S. A request resumed with the correct
matrix but the wrong tail will compute the next Q/K/V incorrectly.
Neither memory can be substituted for ordinary KV pages.

## Hybrid memory still grows with context

The Qwen3.5-0.8B fixture has 18 GDN layers and six full-attention layers.
Each request's recurrent tensors total 18 MiB in FP32; its BF16
convolution tails add about 0.63 MiB. The six full-attention layers add
12 KiB of valid BF16 KV per token.

| Valid cached tokens | Fixed recurrent and convolution state | Full-attention KV payload | Combined valid payload |
|---:|---:|---:|---:|
| 128 | 18.63 MiB | 1.50 MiB | 20.13 MiB |
| 2,048 | 18.63 MiB | 24 MiB | 42.63 MiB |
| 32,768 | 18.63 MiB | 384 MiB | 402.63 MiB |

The cache implementation can allocate history buffers up to a configured
maximum ahead of use. This table counts valid payload, not necessarily
the current allocator's reserved bytes. Concurrency multiplies the fixed
state too: constant per request does not mean constant for the server.

[![A hybrid request owns recurrent matrices, convolution tails, and a growing ordinary or MLA history. Temporary batch rows gather and return that request-owned state; only the final state crosses a kernel chunk boundary.](/assets/tinyserve/book06-hybrid-memory.svg)](/assets/tinyserve/book06-hybrid-memory.svg)

Kimi similarly combines 12 KDA layers with five MLA layers. Its matrix
state is `[12,B,8,128,128]` in FP32, alongside three convolution-tail
families. The MLA history stores the latent and separate component
described in Chapter 3 and grows with sequence length. It is therefore
incorrect to call the entire model constant-memory.

`HybridCache` and `KimiCache` keep these state families together for a
request. A shared sequence cursor advances after every layer has consumed
the same token span. Their transient batch wrappers gather request-owned
states into current batch rows and scatter updated states back afterward.
Batch row identity is temporary; request identity determines ownership.

For unequal padded prompt chunks, padding must be an identity transition:
log decay zero means decay one, update strength zero suppresses writes,
and padded Q/K/V are sanitized before matrix products. The convolution
tail ends at each row's last real token. Only real tokens append KV/MLA
and advance the cursor. Reusing a prefix would require a consistent
snapshot of all state families, not just attention history.
[Chapter 10](/tinyserve/llm-serving/2026/10/06/tinyserve-10-scheduling.html) explains admission and scheduling around
those ownership requirements.

## From chunkwise algebra to native kernels

At the pinned runtime, GDN uses recurrence for one-token decode and
chunkwise prefill otherwise under `auto`, with inner chunk size 64.
KDA keeps inputs shorter than 16 tokens recurrent under `auto`, and
uses chunkwise computation for longer inputs; one-token decode always
remains recurrent. An inner chunk of 64 is unrelated to the scheduler's
prompt-token budget. A 512-token scheduler action can contain eight
inner chunks for one layer.

The first KDA chunk path uses ordinary tensor operations. Native Triton
kernels then fuse pair construction and the correction/output/state work.
The current multi-chunk path can prepare state-independent pairs for all
chunks in parallel, while each state-owning program visits chunks in
order. That distinction exposes parallel work without breaking causal
state carry. Tinyserve implements these kernels itself; it does not call
FLA for the serving path.

The later factor path removes repeated triangular solves across value
tiles. Let `A=I+L`, let P map correction rows into outputs, and let
`Kd^T` map them into the outgoing state. With `U=A^-1 R`, ordinary
execution computes `P U` and `Kd^T U` separately for each value tile.
The factor path first prepares

$$
F=P A^{-1},\qquad J=K_d^\mathsf T A^{-1}.
$$

It then computes `F R` and `J R` for every value tile. A triangular
solve against tiled identity columns constructs the factors; there is
no generic matrix-inverse call. F and J depend on the chunk's projected
inputs, not its incoming recurrent state. R still depends on that state,
so applying successive chunks remains ordered.

At `C=64`, `K=V=128`, and a 16-value tile, eight programs share those
factors. Their FP32 storage is `(64×64 + 128×64)×4 = 48 KiB` per head
per chunk. Eight heads over 32 chunks require 12 MiB. This spends
temporary storage to avoid repeating the same causal solve eight times.

Dispatch is deliberately narrow: native multi-chunk execution requires
CUDA inference with FP32 working tensors, K and V width 128, chunk width
64, at least two complete chunks, and compatible optional token masks.
The default factor tail additionally uses the retained 16-value/eight-warp
geometry. Partial chunks, other dimensions, CPU execution, and autograd
use the applicable readable or existing kernel fallbacks. BF16 model
activations are converted before this FP32 working boundary.

## Measure at the boundary being improved

The initial GDN chunkwise comparison used BF16 Qwen3.5-0.8B on one RTX
A6000, two warmups, and five alternating-order repetitions. Whole
single-request prompt-plus-one-output-token latency fell from 4,401.1 ms
with recurrence to 360.1 ms with C64 at 2,048 prompt tokens. At 128
tokens it fell from 314.8 to 48.6 ms. These figures include tokenization,
cache allocation, the model, and sampling; they are not isolated kernel
times. The [GDN receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m7k-chunkwise-gdn-a6000-2026-09-04.json)
also retains numerical differences and the first-token checks.

KDA's earliest chunk experiment demonstrated the short-input crossover:
at eight tokens the isolated chunk rule took 1.113 ms versus recurrence's
0.927 ms; at 16 it took 1.135 versus 1.782 ms. That is why avoiding the
token loop is not sufficient reason to use chunkwise setup everywhere.
The retained [KDA chunk receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m7l-chunkwise-kda-a6000-2026-09-04.json)
keeps the separate full-model measurements.

Later experiments tested the native implementation on the tiny structural
Kimi fixture. Two plausible changes failed. Halving a state value tile
improved the isolated tail, but the pooled paired full-model reduction
was 4.62%, below the required 5%, and total local-memory instructions
increased. A panelized solve preserved logits but made the 2,048-token
tail 4.78% slower; its full-model paired change was a 15.65% slowdown.
Neither became the default. Their [state-tile](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m7u-kda-state-tile-a6000-2026-09-12.json)
and [panel-solve](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m7v-kda-panel-solve-a6000-2026-09-13.json)
receipts retain those failures.

Reusable factors did pass their declared comparison. On one RTX A6000,
five alternating paired repetitions reduced the 2,048-token Kimi
prompt-plus-one-token median from 105.81 to 92.55 ms, with a paired
median reduction of 12.76% and bootstrap interval `[12.04%,13.48%]`.
Logits matched exactly in the fixture and maximum recurrent-state
difference was `4.66e-10`. At 128 tokens, however, paired model latency
regressed 1.31%, within the accepted 5% guard. The
[factor receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m7w-kda-solve-factors-a6000-2026-09-13.json)
supports this bounded result, not an architecture-wide speedup or a
production-language-quality claim for the small structural checkpoint.

Changing chunk boundaries and reductions changes floating-point association.
Equation tests, poisoned-padding tests, state comparisons, model logits,
and generated decisions therefore answer different questions. A native
kernel can preserve the recurrence while still perturbing a near-tied
BF16 greedy decision. Conversely, a matching first token cannot establish
that every state entry or every future continuation is correct.

## Source map

| Source | Responsibility |
|---|---|
| [models/qwen_hybrid.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/models/qwen_hybrid.py) | GDN layer, recurrence, and chunkwise equations. |
| [models/kimi.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/models/kimi.py) | KDA gates, recurrence, chunkwise fallback, and dispatch. |
| [kimi_kernels.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/kimi_kernels.py) | Native pair, tail, multi-chunk, and reusable-factor kernels. |
| [hybrid_cache.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/hybrid_cache.py) | Qwen request-owned state and transient batches. |
| [kimi_cache.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/kimi_cache.py) | KDA state, convolution tails, and growing MLA history. |

These implementation references describe runtime `e20a348` and require
repository access. The recurrence examples and chunk derivations above
are self-contained.

{% include tinyserve-book-nav.html %}
