---
layout: post
math: true
title: "Linear attention from first principles: chunked prefill, ReplaySSM, and DASC"
date: 2026-09-06 10:00:00 -0700
categories: [linear-attention, llm-serving]
excerpt: "One recurrent state, three serving questions: how to compute it in chunks, how to avoid rewriting it at every decode step, and which parts to keep in a prefix cache."
---

*A standalone note from the hybrid-model work in
[building tinyserve](/series/tinyserve/).*

Linear attention replaces a growing history of keys and values with a
fixed-size recurrent state. That changes the serving problem: we need to
build the state efficiently during prefill, update it during decode, and
sometimes save it for prefix reuse.

This post follows one state matrix through those three jobs:

| Job | Question | Method |
|---|---|---|
| [Prefill](#prefill-runs-the-same-recurrence-in-chunks) | How can many token updates become matrix multiplications? | Chunked evaluation of the recurrence |
| [Decode](#what-one-decode-token-actually-costs) | Can we defer writing the full state back to memory? | ReplaySSM |
| [Prefix reuse](#dasc-do-not-store-what-the-model-has-already-forgotten) | Which parts of a saved state can we omit? | DASC |

The distinction matters. Chunked prefill and ReplaySSM reorganize the same
computation in exact arithmetic. DASC deliberately approximates a saved state.

## Start with an associative memory
{: #from-softmax-to-a-matrix }

For one attention head, use this notation throughout:

| Symbol | Shape | Meaning |
|---|---|---|
| $q_t, k_t$ | $d_k$ | Query and key, written as column vectors |
| $v_t$ | $d_v$ | Value |
| $S_t$ | $d_v \times d_k$ | State after processing token $t$ |
| $r_t$ | $d_v$ | Correction written by token $t$ |
| $\alpha_t$ | scalar | Decay of the previous state |
| $\beta_t$ | scalar | Strength of the correction |
| $y_t$ | $d_v$ | Output, $S_t q_t$ |

Batch, layer, and head indices are omitted. Some kernels store the transpose
of $S$; that changes the written matrix order, not the recurrence.

Ordinary causal attention reads the entire key/value history:

$$
y_t =
\frac{\sum_{i=1}^{t} \exp(q_t^\top k_i)\,v_i}
     {\sum_{i=1}^{t} \exp(q_t^\top k_i)}.
$$

To see how a recurrent state becomes possible, replace the exponential
similarity by a feature inner product, $\phi(q)^\top\phi(k)$. This is a
different attention rule unless the feature map exactly represents the
original kernel. Its numerator and denominator can both be accumulated:

$$
S_t = \sum_{i=1}^{t} v_i\phi(k_i)^\top,
\qquad
z_t = \sum_{i=1}^{t}\phi(k_i),
\qquad
y_t = \frac{S_t\phi(q_t)}{z_t^\top\phi(q_t)}.
$$

The key identity is $(v k^\top)q = v(k^\top q)$: an outer product stores a
value along a key direction, and a query reads it back according to its
overlap with that key. This recurrent formulation is the starting point of
[linear attention](https://arxiv.org/abs/2006.16236).

For the rest of this post, we study an **unnormalized recurrent layer**.
Here $q$ and $k$ denote the vectors actually passed to the recurrence, after
any feature transformation or normalization. The simplest update is

$$
S_t = S_{t-1} + v_t k_t^\top,
\qquad
y_t = S_t q_t.
$$

This is an associative memory. It has a fixed $d_v \times d_k$ shape,
regardless of sequence length. It is also imperfect: repeated writes can
interfere, and old information never fades.

## From additive writes to Gated DeltaNet
{: #three-refinements-that-make-it-work }

### Correct what the state already predicts

Before storing $v_t$ at key $k_t$, read the current prediction $S_{t-1}k_t$.
The delta rule writes the prediction error:

$$
r_t = \beta_t(v_t-S_{t-1}k_t),
\qquad
S_t = S_{t-1}+r_t k_t^\top.
$$

Why is this useful? If $\|k_t\|_2=1$, reading the same key after the update
gives

$$
S_t k_t
= (1-\beta_t)S_{t-1}k_t+\beta_t v_t.
$$

At $\beta_t=1$, the state now returns $v_t$ for that key. At smaller
$\beta_t$, it moves partway toward the new value. Other key directions can
still be affected when they overlap with $k_t$.

### Decay first, then correct

Gated DeltaNet also forgets some of the old state. The prediction must use
that **decayed** state:

$$
\boxed{
\begin{aligned}
r_t &= \beta_t\bigl(v_t-\alpha_t S_{t-1}k_t\bigr),\\
S_t &= \alpha_t S_{t-1}+r_t k_t^\top,\\
y_t &= S_t q_t.
\end{aligned}
}
\tag{1}
$$

Equation (1) is the recurrence used in every derivation below. In particular,
the $\alpha_t$ inside the correction is essential. Predicting from
$S_{t-1}$ instead would define a different update.

Equivalently,

$$
S_t
= \alpha_t S_{t-1}(I-\beta_t k_t k_t^\top)
  + \beta_t v_t k_t^\top.
\tag{2}
$$

Equation (1) exposes the rank-one write; equation (2) exposes how old state
is transformed. We will need both views.

This is the scalar-decay setting of
[Gated Delta Networks](https://arxiv.org/abs/2412.06464).
Kimi Delta Attention uses a decay vector over key channels: with our
$d_v \times d_k$ convention, decay becomes right multiplication by a
diagonal matrix. The scalar formulas below should not be reused unchanged
for that case.

## Prefill: solve the corrections, then carry the state
{: #prefill-runs-the-same-recurrence-in-chunks }

During decode, only one new token is available. During prefill, all the
tokens in the prompt are available, so we would like to process many at once.

The obstacle is in equation (1): $r_t$ depends on $S_{t-1}$, which depends
on earlier corrections. Chunked prefill separates those dependencies into
two parts:

1. Interactions among tokens inside a chunk.
2. Dependence on the state entering that chunk.

Take a chunk of $C$ tokens. Reset its local token indices to $1,\ldots,C$,
and call its incoming state $S_0$. Define cumulative decay

$$
g_0=1,
\qquad
g_i=\prod_{j=1}^{i}\alpha_j.
$$

Thus $g_i$ is how much of the incoming state remains at token $i$, while
$g_i/g_j$ is the decay from just after token $j$ through token $i$.
Assume $\alpha_j>0$ for this notation.

### Step 1: expand two tokens to find the coefficients
{: #move-1-the-state-inside-a-chunk-in-terms-of-the-boundary-state }

The first correction is already an affine function of $S_0$:

$$
r_1
= \beta_1 v_1-S_0(\beta_1 g_1 k_1)
= \widetilde u_1-S_0 w_1,
$$

where

$$
\widetilde u_1=\beta_1 v_1,
\qquad
w_1=\beta_1 g_1 k_1.
$$

For token 2, substitute $S_1=\alpha_1S_0+r_1k_1^\top$ into equation (1):

$$
\begin{aligned}
r_2
&= \beta_2\bigl(v_2-\alpha_2 S_1 k_2\bigr)\\
&= \beta_2 v_2-\beta_2 g_2 S_0 k_2
   -\beta_2\alpha_2(k_1^\top k_2)r_1.
\end{aligned}
$$

Name the coefficient of $r_1$:

$$
L_{21}
= \beta_2\frac{g_2}{g_1}(k_1^\top k_2).
$$

Now substitute $r_1=\widetilde u_1-S_0w_1$ and collect terms:

$$
\begin{aligned}
r_2
&= \beta_2v_2-\beta_2g_2S_0k_2
   -L_{21}(\widetilde u_1-S_0w_1)\\
&= \underbrace{\bigl(\beta_2v_2-L_{21}\widetilde u_1\bigr)}
      _{\widetilde u_2}
   -S_0\underbrace{\bigl(\beta_2g_2k_2-L_{21}w_1\bigr)}
      _{w_2}.
\end{aligned}
$$

This is where $W$ comes from. Each $w_i$ collects the coefficients multiplying
the incoming state, including the effects of earlier corrections.
$\widetilde u_i$ collects everything independent of that state.

Neither is a learned weight matrix. Both are computed from the current
chunk's inputs.

### Step 2: turn the substitution into a triangular solve
{: #move-2-derive-the-corrections-for-two-tokens }

Unrolling the state through token $i-1$ gives

$$
S_{i-1}
= g_{i-1}S_0
  +\sum_{j<i}\frac{g_{i-1}}{g_j}r_jk_j^\top.
$$

Substituting this into the correction equation yields

$$
r_i
= \beta_i v_i-\beta_i g_i S_0k_i
  -\sum_{j<i}
    \beta_i\frac{g_i}{g_j}(k_j^\top k_i)r_j.
\tag{3}
$$

Stack token vectors as rows:

$$
K\in\mathbb R^{C\times d_k},
\qquad
V,R\in\mathbb R^{C\times d_v}.
$$

Row $i$ of $R$ is $r_i^\top$. Define diagonal matrices
$D_\beta=\operatorname{diag}(\beta_1,\ldots,\beta_C)$ and
$D_g=\operatorname{diag}(g_1,\ldots,g_C)$, and a strictly lower-triangular
matrix

$$
L_{ij}=
\begin{cases}
\beta_i(g_i/g_j)(k_i^\top k_j), & j<i,\\
0, & j\ge i.
\end{cases}
$$

Equation (3) is then

$$
(I+L)R=D_\beta V-D_\beta D_g K S_0^\top.
$$

Solve the two state-independent systems

$$
\boxed{
(I+L)U=D_\beta V,
\qquad
(I+L)W=D_\beta D_g K.
}
\tag{4}
$$

The diagonal of $I+L$ is all ones, so these are triangular solves.
$U$ has shape $C\times d_v$ and $W$ has shape $C\times d_k$.
Their rows are exactly the $\widetilde u_i^\top$ and $w_i^\top$ from
the two-token derivation.

By linearity of the solve,

$$
\boxed{R=U-WS_0^\top.}
\tag{5}
$$

That is the full derivation of the kernel expression `R = U - W @ S`.
A kernel that stores the incoming state as $d_k\times d_v$ calls
$S_0^\top$ its `S`, so the transpose disappears in code.

The placement of $D_g$ matters: it belongs **inside the right-hand side
of the solve for $W$**. Scaling an ungated solution afterward is generally
different because $D_g$ and the triangular solve do not commute.

### Step 3: use the corrections for outputs and the next state
{: #move-3-compute-those-coefficients-with-a-triangular-solve }

Once the actual corrections are known,

$$
S_i=g_iS_0+\sum_{j\le i}\frac{g_i}{g_j}r_jk_j^\top.
\tag{6}
$$

Apply this state to query $q_i$:

$$
y_i=g_iS_0q_i+
    \sum_{j\le i}\frac{g_i}{g_j}r_j(k_j^\top q_i).
$$

Let $Q\in\mathbb R^{C\times d_k}$ and $O\in\mathbb R^{C\times d_v}$
stack queries and outputs as rows. Define the causal decay matrix

$$
\Gamma_{ij}=
\begin{cases}
g_i/g_j, & j\le i,\\
0, & j>i.
\end{cases}
$$

Then the whole chunk's outputs are

$$
\boxed{
O=D_g Q S_0^\top+
  \bigl(\Gamma\odot QK^\top\bigr)R.
}
\tag{7}
$$

The state passed to the next chunk is

$$
\boxed{
S_C=g_CS_0+
R^\top\operatorname{diag}(g_C/g_1,\ldots,g_C/g_C)K.
}
\tag{8}
$$

These are matrix multiplications. The sequential dependency has been
concentrated into the state passed between chunks; the local preparation
and output work can run across chunks in parallel.

### Where `fla_chunk_delta_h` fits

In the FLA-style pipeline, the stages have different responsibilities:

| Stage | Inputs it needs | What it computes |
|---|---|---|
| Prepare each chunk | Keys, values, gates, correction strengths | $U,W$ from equation (4) |
| Carry state between chunks | $U,W$, keys, gates, incoming state | $R$ and $S_C$ from equations (5) and (8) |
| Produce outputs | Queries, keys, gates, $R$, saved incoming states | $O$ from equation (7) |

`fla_chunk_delta_h` is the middle stage. In the implementation discussed
here, `h` saves the state **entering** each chunk, and `v_new` stores the
corrected values $R$. The kernel also advances the carried state for the
next chunk. The decay factors needed to reach the chunk boundary are
applied during the state update; they are not already folded into `v_new`.

This explains why the kernel can look simple once $U$ and $W$ exist:
the triangular solve has already accounted for the interactions inside
the chunk. What remains is to insert the actual incoming state.

Implementations usually represent cumulative decay in log space rather
than forming long products directly. A code path using `exp2` expresses
those logs in base 2. Floating-point precision and tiling still affect
numerical agreement; the identities above describe exact arithmetic.

The chunkwise delta-rule construction is developed in
[Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)
and extended to gated updates in
[Gated Delta Networks](https://arxiv.org/abs/2412.06464).
The FLA [forward caller](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py),
[state kernel](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_delta_h.py),
and [output kernel](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_o.py)
show the corresponding pipeline.

## Decode: the cost of materializing state
{: #what-one-decode-token-actually-costs }

During ordinary decode, a fused recurrent kernel reads $S_{t-1}$, computes
the correction and output, and writes $S_t$ back to memory.

For batch size $B$, $N_h$ heads per layer, and $b$ bytes per state element,
the dense state occupies

$$
M=B N_h d_v d_k b
$$

bytes per layer. Reading and writing it moves roughly $2M$ bytes per token,
before counting projections, gates, convolution state, or other model work.

For example, an FP32 state with $B=256$, $N_h=4$, and
$d_k=d_v=256$ occupies 256 MiB per layer. A dense update moves about
512 MiB per layer. Across 24 layers, that is 12 GiB of state traffic per
decode step.

This can make large-batch recurrent decoding bandwidth-bound. It is not a
universal statement about the whole model: at smaller batches, launch
overhead and other operations can dominate, and the importance of state
traffic depends on the architecture and hardware.

The state is needed to compute future predictions. But must its newest
version be written to global memory after every token?

## ReplaySSM: keep an anchor and recent corrections
{: #replayssm-why-write-the-state-at-all }

Equation (6) already provides another representation of the state.
Take $S_0$ to be the last materialized checkpoint, or **anchor**.
After $t$ additional tokens,

$$
S_t=g_tS_0+\sum_{i=1}^{t}\frac{g_t}{g_i}r_i k_i^\top.
\tag{9}
$$

Instead of immediately writing the dense $S_t$, keep $S_0$ plus a short
buffer of corrections, keys, and decay information. The logical state
changes every token; its dense representation need not.

For a scalar illustration, let $S_0=8$, every decay be $1/2$, and the
successive additive writes $r_i k_i$ be $4,2,6$:

| Token | Recurrent update | State |
|---|---|---|
| 1 | $\frac12\cdot8+4$ | $8$ |
| 2 | $\frac12\cdot8+2$ | $6$ |
| 3 | $\frac12\cdot6+6$ | $9$ |

The anchor representation gives the same answer:

$$
S_3=
\underbrace{\tfrac18\cdot8}_{\text{anchor}}
+\underbrace{\tfrac14\cdot4+\tfrac12\cdot2+6}_{\text{recent writes}}
=9.
$$

No information was discarded. This equality does not require the anchor
to have decayed to a small value.

### What a GDN replay step has to compute

There are two state reads in the mathematics of equation (1):

$$
p_t=\alpha_t S_{t-1}k_t,
\qquad
r_t=\beta_t(v_t-p_t),
$$

followed by

$$
y_t=\alpha_t S_{t-1}q_t+r_t(k_t^\top q_t).
$$

Both $S_{t-1}k_t$ and $S_{t-1}q_t$ can be obtained from the anchor plus
buffer using equation (9). A fused kernel can load an anchor tile,
reconstruct the required current-state tile, use it for these products,
and avoid storing that reconstructed tile.

For GDN, the [ReplaySSM construction](https://dao-lab.ai/blog/2026/replayssm/)
caches **correction vectors** together with keys and log decays.
In our notation, a record is $(r_i,k_i,\log\alpha_i)$; the source calls
the correction $u_i$. These are not raw values $v_i$, because each
correction has already incorporated the prediction from its preceding
state.

After a window of $P$ tokens, materialize equation (9) once, make it the
new anchor, and clear the buffer. A window bounds both storage and replay
work. Longer windows save writes but increase reconstruction cost.

### What the traffic saving does and does not mean

If the implementation reads one full anchor per token and fuses the flush
write into the last step, a $P$-token window moves:

- Dense recurrence: $2P$ full matrices.
- Replay: $P+1$ full matrices, plus buffer traffic.

The ratio for dense-state traffic alone is

$$
\frac{2P}{P+1}.
$$

At $P=8$, this is $16/9\approx1.78$. It approaches 2 as the window grows.
It is not an end-to-end speedup prediction: buffer accesses, extra
arithmetic, occupancy, and the rest of the model still matter.

![Dense recurrence moves sixteen full state matrices over eight tokens; replay moves nine, plus vector traffic.](/assets/linear-attention/replayssm-write-traffic.svg)

Replay is exact as an algebraic reorganization, subject to floating-point
rounding and the chosen buffer precision. Omitting or quantizing the
anchor would introduce an additional approximation.

Speculative decoding adds another benefit: accepted history can be tracked
through buffer records rather than a full matrix snapshot for every draft
token. For GDN, computing the draft corrections still requires accounting
for their causal interactions, using a triangular solve closely related
to prefill. Rollback must also restore decay bookkeeping; simply truncating
a buffer is insufficient if its older entries were rescaled in place.

### Published results and a local prototype

The [ReplaySSM report](https://dao-lab.ai/blog/2026/replayssm/) gives
end-to-end standard-decode speedups up to $1.48\times$. Its speculative
results reach $1.87$–$1.96\times$ relative to vLLM **standard** decoding.
These are different comparisons. Its fixed-memory speculative tests also
report roughly $3$–$3.3\times$ batch capacity.

My early A6000 prototype used a 340M Gated DeltaNet with 24 layers,
four heads per layer, and $d_k=d_v=256$. With an eight-token window:

| Batch | Attention-kernel speedup | Full 24-layer decode speedup |
|---|---:|---:|
| 256 | $1.742\times$ | $1.126\times$ |
| 512 | $1.746\times$ | $1.282\times$ |

These are historical measurements from that prototype, not measurements
of the published implementation or new results from this write-up.
The gap between kernel and full-decode speedup is the practical reason
to keep those measurements separate.

## DASC: decide what a prefix checkpoint must retain
{: #dasc-do-not-store-what-the-model-has-already-forgotten }

ReplaySSM keeps enough information to reproduce the state. DASC asks a
different question: for **prefix reuse**, can we save a smaller checkpoint
and tolerate a small reconstruction error?

A hybrid model may cache recurrent states at prefix boundaries alongside
its full-attention KV cache and convolution states. With many cached
prefixes, these recurrent checkpoints can consume substantial memory.
DASC targets that persisted state.

The intuition is that some parts of the state forget old history quickly.
A recent suffix may therefore be enough to reconstruct them approximately,
while slower-forgetting parts still need their saved checkpoint.

### Why decay suggests a retention horizon

Consider two copies of the same layer with different initial states, but
the **same subsequent keys, values, and gates**. From equation (2), their
state difference satisfies

$$
\Delta_t
=\alpha_t\Delta_{t-1}(I-\beta_t k_tk_t^\top).
\tag{10}
$$

If $\|k_t\|_2=1$ and $0\le\beta_t\le2$, the matrix
$I-\beta_tk_tk_t^\top$ has spectral norm at most one. Therefore

$$
\|\Delta_t\|_F
\le
\left(\prod_{i=1}^{t}\alpha_i\right)\|\Delta_0\|_F.
\tag{11}
$$

This gives a precise, conditional version of “the layer forgets its
starting state.” For a constant decay $\alpha$, the number of steps
needed to attenuate that starting-state difference by a factor
$\epsilon$ is

$$
H_{\mathrm{ret}}=\frac{\log\epsilon}{\log\alpha}.
\tag{12}
$$

For $\epsilon=10^{-3}$:

| Decay $\alpha$ | Approximate horizon |
|---|---:|
| $0.95$ | 135 tokens |
| $0.99$ | 687 tokens |
| $0.9999$ | 69,074 tokens |

The horizons can differ by orders of magnitude. Treating every head as
equally dependent on distant history wastes that distinction.

But equation (11) is not a full-model quality guarantee. Real gates are
input-dependent, the initial error has its own magnitude, and errors in
earlier layers can change later layers' inputs. Attenuating a state
difference by $10^{-3}$ does not by itself bound output error by
$10^{-3}$.

### Store the long-memory units; approximate the others

[DASC](https://arxiv.org/abs/2608.30386) estimates static horizons from
learned decay parameters evaluated at a nominal gate input. For a chosen
suffix budget $W_{\max}$, it retains units whose estimated horizons exceed
that budget.

The unit is a whole head for scalar-decay GDN. For KDA, it is a key channel:
a **column** of our $d_v\times d_k$ state, or a row in the transposed layout.

The paper describes two restoration modes:

- **No recovery (NR):** restore retained units and fill omitted units with zeros.
- **Window recovery (WR):** replay a bounded prefix suffix from zero in scratch
  state, then use the reconstructed omitted units alongside the retained ones.

WR remains approximate: information from before the suffix is absent.
Its corrections must be recomputed during replay; corrections from a
different incoming state cannot simply be reused. The method preserves
the other hybrid-model cache components, including convolution state.

The [paper's serving experiments](https://arxiv.org/abs/2608.30386) report
up to $2.63\times$ recurrent-checkpoint capacity, 42.6% lower time to first
token, and 68.4% higher input throughput under matched memory budgets.
The strongest reported speed point uses NR; those latency numbers should
not be presented as a measurement of suffix replay.

### What the local checkpoint experiment showed

A separate A6000 experiment used
[`linear-moe-hub/Gated-Deltanet-340M`](https://huggingface.co/linear-moe-hub/Gated-Deltanet-340M),
with 96 heads across 24 layers. An offline calibration selected a
256-token recovery window.

| Per-prefix checkpoint | Dense | Compressed |
|---|---:|---:|
| Persisted recurrent heads | 96 | 40 |
| Heads reconstructed from the suffix | 0 | 56 |
| Bytes including unchanged convolution state | 24.56 MiB | 10.56 MiB |

That saved about 57% of checkpoint bytes, or allowed about
$2.33\times$ as many such checkpoints in the same space. It is not a
$2.33\times$ reduction in total serving memory.

Across 4,096 evaluated continuation tokens from WikiText-2 and
CNN/DailyMail, the worst-slice top-1 agreement with dense restoration was
98.828%. This is encouraging evidence for the selected model and test
slices, not general qualification for every prompt or model.

Recovery added roughly 60–65 ms for a batch of four in the Python
prototype. That was measured recovery work, not end-to-end time to first
token. Whether it pays off depends on how much extra prefix-cache capacity
improves reuse, and how often recovery is needed.

## Three methods, three different decisions

The same recurrence connects the whole story, but each method changes
a different part of serving:

| Method | What changes | What is preserved in exact arithmetic | Main tradeoff |
|---|---|---|---|
| Chunked prefill | Evaluation order | The recurrent outputs and final state | Local preparation and intermediate storage in exchange for matrix operations |
| ReplaySSM | Frequency of dense-state writes | The state represented by the anchor and buffered corrections | Fewer writes in exchange for replay work and buffer storage |
| DASC | Contents of persisted prefix checkpoints | Only the explicitly retained state units | More cache capacity in exchange for approximation and possibly recovery work |

ReplaySSM does not rely on fast forgetting: equation (9) is exact even
when every $\alpha_i=1$. DASC uses forgetting to justify dropping
information, so its quality must be evaluated.

They can be combined conceptually: a DASC-restored state can initialize
a ReplaySSM anchor. That preserves the approximation already introduced
by DASC; it does not remove it. A serving implementation must also handle
sequence admission, checkpoint restore, buffer reset, and rollback
consistently.

Likewise, a small anchor coefficient is a reason to investigate skipping
an anchor read, not proof that the read is unnecessary. Quantizing or
omitting the anchor requires measuring the resulting recurrent error
and model quality as well as speed.

The useful question is always concrete: are we changing **when the same
state is computed**, **when it is written**, or **which information is
kept**? Keeping those decisions separate makes both the algebra and the
performance claims easier to follow.

## References

- Katharopoulos et al., [*Transformers are RNNs: Fast Autoregressive
  Transformers with Linear Attention*](https://arxiv.org/abs/2006.16236).
- Yang et al., [*Parallelizing Linear Transformers with the Delta Rule over
  Sequence Length*](https://arxiv.org/abs/2406.06484).
- Yang et al., [*Gated Delta Networks: Improving Mamba2 with Delta Rule*](https://arxiv.org/abs/2412.06464).
- Kimi Team, [*Kimi Linear*](https://arxiv.org/abs/2510.26692).
- Dao Lab, [*ReplaySSM*](https://dao-lab.ai/blog/2026/replayssm/).
- Yu et al., [*DASC: Decay-Aware State Compression for Hybrid
  Linear-Attention Serving*](https://arxiv.org/abs/2608.30386).
- [*Flash Linear Attention*](https://github.com/fla-org/flash-linear-attention).
