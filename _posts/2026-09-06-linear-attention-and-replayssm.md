---
layout: post
math: true
title: "Linear attention from first principles: ReplaySSM and DASC"
date: 2026-09-06 10:00:00 -0700
categories: [linear-attention, llm-serving]
excerpt: "How softmax attention becomes a fixed-size matrix, why decoding that matrix is a memory-bandwidth problem, and how two methods, ReplaySSM at decode time and DASC in the prefix cache, both stop storing state the model can rebuild."
---

*A standalone note. It grew out of the hybrid-model milestones in
[building tinyserve](/series/tinyserve/), where Qwen3.5 and Kimi-K3 spend
most of their decode time in linear-attention layers.*

Transformers pay for long context with a KV cache that grows every token.
Linear attention pays instead with a fixed-size matrix per head that must be
read and rewritten every token. That trade looks like a clear win on paper, and
in practice it moves the bottleneck rather than removing it: a decode step in a
Gated DeltaNet layer is a pure memory-bandwidth problem, and the matrix traffic
is the whole problem.

This post derives the linear-attention family from softmax attention in five
short steps, works out exactly what a decode token costs, and then explains two
methods that attack the state from opposite sides. ReplaySSM, from the Dao Lab,
removes half of the decode traffic by asking why the state should be written at
all. DASC, from Yu et al., shrinks prefix-cache checkpoints by asking which
heads have already forgotten the prefix. Both turn out to be the same idea.

## From softmax to a matrix

Start with ordinary causal attention for token $t$:

$$
y_t = \frac{\sum_{i \le t} \exp(q_t^\top k_i)\, v_i}{\sum_{j \le t} \exp(q_t^\top k_j)}.
$$

Every previous key and value participates. The cache holds all of them, so
memory is $O(t \cdot d)$ and so is the work per token.

Replace $\exp(q^\top k)$ by an inner product of feature maps,
$\phi(q)^\top \phi(k)$. The query is the same in every term, so it factors out
of the sum:

$$
y_t = \sum_{i \le t} \phi(q_t)^\top \phi(k_i)\, v_i
    = \Big( \sum_{i \le t} v_i\, \phi(k_i)^\top \Big) \phi(q_t)
    = S_t\, \phi(q_t).
$$

Two facts make the middle step legal. First, $\phi(q_t)^\top \phi(k_i)$ is a
scalar, and a scalar times a vector can be written in either order, so
$\big(\phi(q_t)^\top \phi(k_i)\big) v_i = v_i \big(\phi(k_i)^\top \phi(q_t)\big)$.
Second, that product is $v_i \phi(k_i)^\top$ applied to $\phi(q_t)$, because for
any vectors $a, b, c$ the outer product obeys $(a b^\top) c = a (b^\top c)$: the
matrix $a b^\top$ times $c$ is $a$ scaled by the dot product $b^\top c$. Since
$\phi(q_t)$ does not depend on the summation index $i$, it can be pulled outside
the sum, leaving $\sum_i v_i \phi(k_i)^\top$ as a matrix that no longer knows
anything about the query.

The entire history has collapsed into one matrix $S_t \in \mathbb{R}^{V \times K}$
with a one-step recurrence:

$$
S_t = S_{t-1} + v_t k_t^\top, \qquad y_t = S_t q_t.
$$

The recurrence is just the sum written incrementally: $S_t = \sum_{i \le t}
v_i k_i^\top$ and $S_{t-1} = \sum_{i \le t-1} v_i k_i^\top$ differ by exactly
the $i = t$ term.

Modern variants drop $\phi$ and the normalizer, L2-normalize $q$ and $k$, and
put an RMSNorm on the output instead. Memory is $O(VK)$ regardless of context.
Each decode token costs one read and one write of $S$.

## Three refinements that make it work

**Forgetting.** The plain recurrence never forgets. A scalar gate
$\lambda_t \in (0,1)$ per head fixes that:

$$
S_t = \lambda_t S_{t-1} + v_t k_t^\top.
$$

Unrolling means substituting the recurrence into itself until only inputs
remain. Start from $S_0 = 0$ and apply the rule three times:

$$
\begin{aligned}
S_1 &= v_1 k_1^\top, \\
S_2 &= \lambda_2 S_1 + v_2 k_2^\top
     = \lambda_2\, v_1 k_1^\top + v_2 k_2^\top, \\
S_3 &= \lambda_3 S_2 + v_3 k_3^\top
     = \lambda_3 \lambda_2\, v_1 k_1^\top + \lambda_3\, v_2 k_2^\top + v_3 k_3^\top.
\end{aligned}
$$

Each substitution multiplies everything already present by one more $\lambda$
and adds one new undecayed term. After $t$ steps the term from token $i$ has
been multiplied by $\lambda_{i+1}, \lambda_{i+2}, \dots, \lambda_t$, one factor
for every token that arrived after it, and the newest term by nothing:

$$
S_t = \sum_{i \le t} \Big( \prod_{j=i+1}^{t} \lambda_j \Big) v_i k_i^\top.
$$

The empty product for $i = t$ is 1. If every $\lambda$ equals a constant
$\lambda$, the weight on token $i$ is $\lambda^{\,t-i}$, which is exponential
forgetting with a horizon set by how close $\lambda$ is to 1.

This is the Mamba2 and gated-linear-attention family.

**The delta rule.** Treat $S$ as an associative memory. Reading key $k_t$
returns $S_{t-1} k_t$. To *store* $v_t$ under $k_t$, take one gradient step of
size $\beta_t$ on the retrieval error $\tfrac{1}{2}\lVert S k_t - v_t \rVert^2$:

$$
S_t = S_{t-1} - \beta_t (S_{t-1} k_t - v_t) k_t^\top
    = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top.
$$

The gradient comes from the chain rule. Write the residual $r = S k_t - v_t$,
so the loss is $\tfrac{1}{2} r^\top r$. Its derivative with respect to $r$ is
$r$, and $r$ depends on $S$ through $S k_t$, whose derivative with respect to
the matrix $S$ is "multiply on the right by $k_t^\top$". So
$\nabla_S \tfrac{1}{2}\lVert S k_t - v_t \rVert^2 = (S k_t - v_t)\, k_t^\top$,
a $V \times K$ matrix of rank one. One gradient-descent step with learning rate
$\beta_t$ gives the first form.

The second form is the first with the bracket expanded and regrouped:
$S_{t-1} - \beta_t S_{t-1} k_t k_t^\top + \beta_t v_t k_t^\top$, then factor
$S_{t-1}$ out of the first two terms. It says the update is a rank-one
projection applied to the old state, followed by a rank-one write.

The exact-overwrite claim is a two-line check. Apply $S_t$ to $k_t$ using the
first form:

$$
S_t k_t = S_{t-1} k_t - \beta_t (S_{t-1} k_t - v_t)\,(k_t^\top k_t).
$$

With $\lVert k_t \rVert = 1$ the last factor is 1, and with $\beta_t = 1$ the
right side collapses to $S_{t-1} k_t - S_{t-1} k_t + v_t = v_t$. So reading
$k_t$ back immediately after the write returns exactly $v_t$, whatever was
stored there before.

**A worked example.** Take $K = V = 2$, start from $S_0 = 0$, and use the unit
key $k = (1, 0)$ throughout. First store $v_1 = (2, 3)$ under $k$. Both rules
agree when the memory is empty:

$$
S_1 = v_1 k^\top = \begin{pmatrix} 2 & 0 \\ 3 & 0 \end{pmatrix},
\qquad S_1 k = (2, 3) = v_1 .
$$

Now store a different value $v_2 = (5, 1)$ under the *same* key.

Plain linear attention adds another outer product:

$$
S_2 = S_1 + v_2 k^\top = \begin{pmatrix} 7 & 0 \\ 4 & 0 \end{pmatrix},
\qquad S_2 k = (7, 4) = v_1 + v_2 .
$$

Reading $k$ back returns the sum of both values, which is neither of them. The
memory has been corrupted by the repeated key.

The delta rule with $\beta = 1$ first reads what is already there, $S_1 k =
(2, 3)$, computes the error against the new value, $(2, 3) - (5, 1) =
(-3, 2)$, and subtracts that error along $k$:

$$
S_2 = S_1 - \begin{pmatrix} -3 \\ 2 \end{pmatrix} k^\top
    = \begin{pmatrix} 2 & 0 \\ 3 & 0 \end{pmatrix}
    - \begin{pmatrix} -3 & 0 \\ 2 & 0 \end{pmatrix}
    = \begin{pmatrix} 5 & 0 \\ 1 & 0 \end{pmatrix},
\qquad S_2 k = (5, 1) = v_2 .
$$

Reading $k$ back now returns exactly the new value. The old one is gone. A
different key $k' = (0, 1)$ reads $(0, 0)$ from both $S_1$ and $S_2$, so the
correction touched only the direction of $k$ and left the rest of the memory
alone. That is what the projection $(I - \beta k k^\top)$ does: with unit $k$
and $\beta = 1$ it zeroes the component of every row along $k$ and leaves the
orthogonal components untouched.

With $\beta = 0.5$ the same step gives $S_2 k = (3.5, 2)$, halfway between the
old and new values. The learned $\beta_t$ therefore controls how much the
model trusts the new observation over the existing memory, per token and per
head.

Plain linear attention would have *added* $v_t$ on top of
whatever was already stored under $k_t$. That is the difference between an
accumulator and an error-correcting memory, and it is why DeltaNet models
handle repeated keys so much better than their predecessors.

**Both together: Gated DeltaNet.** Decay first, then the delta step on the
decayed state:

$$
S_t = \lambda_t S_{t-1} + \beta_t\big(v_t - \lambda_t S_{t-1} k_t\big) k_t^\top.
$$

To see where this comes from, take the delta rule's second form and replace
$S_{t-1}$ by the decayed state $\lambda_t S_{t-1}$ everywhere:

$$
S_t = \lambda_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top
    = \lambda_t S_{t-1} - \beta_t \lambda_t S_{t-1} k_t k_t^\top + \beta_t v_t k_t^\top .
$$

The last two terms share the factor $\beta_t(\cdot) k_t^\top$; pulling it out
gives $\beta_t (v_t - \lambda_t S_{t-1} k_t) k_t^\top$. The quantity in the
bracket is the retrieval error measured against the *decayed* memory, which is
the right thing to correct since the decayed memory is what the next reader
will see.

Name the rank-one innovation $u_t := \beta_t(v_t - \lambda_t S_{t-1} k_t)$.
Then the update and readout are

$$
S_t = \lambda_t S_{t-1} + u_t k_t^\top, \qquad
y_t = S_t q_t = \lambda_t S_{t-1} q_t + u_t\,(k_t^\top q_t).
$$

The readout on the right is the update rule multiplied through by $q_t$:
$(\lambda_t S_{t-1} + u_t k_t^\top)\, q_t = \lambda_t S_{t-1} q_t + u_t k_t^\top q_t$,
and $k_t^\top q_t$ is a scalar, so $u_t k_t^\top q_t = u_t\,(k_t^\top q_t)$ by
the same outer-product rule used in the first section. That is the whole
derivation, but it changes what a kernel has to do.

The second form matters for what follows: the output needs two matrix-vector
products against the *old* state, one with $k_t$ to form the innovation and one
with $q_t$, and never needs $S_t$ to exist as a matrix.

Where the scalars come from, in the FLA implementation: $q$, $k$, $v$ are
projections followed by a width-4 depthwise causal convolution and SiLU; $q$
and $k$ are L2-normalized; $\lambda_t = \exp(-\exp(A_{\log}) \cdot
\mathrm{softplus}(x_t W_a + \mathrm{dt\_bias}))$ with learned per-head
$A_{\log}$ and $\mathrm{dt\_bias}$, which is Mamba2's discretized decay
reused verbatim; and $\beta_t = \sigma(x_t W_\beta)$. Kimi's KDA differs in
one place: $\lambda_t$ is a vector over key channels instead of a scalar per
head.

## What one decode token actually costs

Take a small concrete model: 24 layers, 4 heads, $K = V = 256$, state in fp32.
One head's state is $256 \times 256 \times 4 = 256$ KiB. One sequence carries
1 MiB per layer and 24 MiB across the model.

The dense kernel handles a batch of $B$ sequences in one launch. Per layer per
token it must load every state, apply $\lambda_t S + u_t k_t^\top$, and store
every state back. Every entry changes when $\lambda_t \ne 1$, so nothing can
be skipped:

$$
\text{bytes} = \underbrace{2}_{\text{read+write}} \cdot
\underbrace{4}_{\text{fp32}} \cdot B \cdot H \cdot V \cdot K .
$$

| batch | state moved per layer per token | 24 layers |
|---:|---:|---:|
| 64 | 134 MB | 3.2 GB |
| 256 | 537 MB | 12.9 GB |
| 512 | 1,074 MB | 25.8 GB |

An RTX A6000 sustains about 768 GB/s. At batch 256 the floor is 0.70 ms per
layer; the measured kernel takes 0.80 ms. The arithmetic is 65,536 multiply-adds
per head and takes microseconds. Every tiling and warp-count sweep I ran
landed within noise of the same number, because none of them changes the bytes.

Two consequences. First, this is bandwidth-bound in the strictest sense, and
the only lever is to move fewer bytes. Second, the cost is linear in batch while
the weight traffic for the projections is not, so at small batch the state is a
rounding error and at batch 256 and above it is most of the layer.

## ReplaySSM: why write the state at all?

The Dao Lab's
[ReplaySSM](https://dao-lab.ai/blog/2026/replayssm/) starts from the
observation that half of the traffic above is the write, and the write is not
needed to produce the output. The output needs the *old* state. The new state
is needed only so that the *next* token can read it. So instead of eagerly
summarizing history into $S$ every step, cache the recent inputs and
reconstruct $S$ only when the cache fills.

![Dense decoding moves the state matrix twice per token; ReplaySSM reads it
every token and writes it once per eight-token window](/assets/linear-attention/replayssm-write-traffic.svg)

### The identity, with one number first

Forget matrices and let the state be a single number updated by
"decay, then add": $S_{\text{new}} = \lambda S_{\text{old}} + u$. Take
$\lambda = 0.5$, start at $S = 8$, and feed in $u = 4, 2, 6$. The dense
way rewrites $S$ each step: $8 \to 8 \to 6 \to 9$.

Now keep the starting value $A = 8$ untouched, a scale $\alpha$ that starts at
1, and a list $R$. On each token, multiply $\alpha$ by $\lambda$, multiply every
entry already in $R$ by $\lambda$, and append the new input unchanged:

| after | $\alpha$ | $A$ | $R$ | $\alpha A + \sum R$ |
|---|---:|---:|---|---:|
| start | 1 | 8 | [ ] | 8 |
| token 1 | 0.5 | 8 | [4] | 8 |
| token 2 | 0.25 | 8 | [2, 2] | 6 |
| token 3 | 0.125 | 8 | [1, 1, 6] | 9 |

Same answers, and the slot holding 8 was never written. The list holds each
past input at its *current* decayed value.

### The same thing with matrices

Let $A$ be the state matrix stored at the start of a window, never written
during the window. Let $\alpha_t$ be the product of decays since then. Store
each innovation as a pair $(R_i, K_i)$ with $K_i = k_i$ and $R_i$ the
value-side vector $u_i$ at its current decayed scale. Then for every $t$ in
the window,

$$
S_t = \alpha_t A + \sum_{i} R_i^{(t)} K_i^\top,
\qquad R_i^{(t)} = u_i \prod_{j=i+1}^{t} \lambda_j ,
\qquad K_i = k_i .
$$

The proof is one substitution. Assume it holds at $t-1$, then

$$
\begin{aligned}
S_t &= \lambda_t S_{t-1} + u_t k_t^\top \\
    &= \lambda_t\Big(\alpha_{t-1} A + \sum_{i \le t-1} R_i^{(t-1)} K_i^\top\Big) + u_t k_t^\top \\
    &= (\lambda_t \alpha_{t-1})\, A + \sum_{i \le t-1} \big(\lambda_t R_i^{(t-1)}\big) K_i^\top + u_t k_t^\top \\
    &= \alpha_t A + \sum_{i \le t-1} R_i^{(t)} K_i^\top + R_t^{(t)} K_t^\top
        \qquad\text{with } R_t^{(t)} := u_t,\; K_t := k_t \\
    &= \alpha_t A + \sum_{i \le t} R_i^{(t)} K_i^\top .
\end{aligned}
$$

The last line renames three things without changing any value: $\lambda_t\alpha_{t-1}$
is $\alpha_t$ by the definition of $\alpha$ as a running product; each
$\lambda_t R_i^{(t-1)}$ is $R_i^{(t)}$ by the definition of $R$ as the input
times every decay since it arrived; and the fresh $u_t k_t^\top$ is the
$i = t$ term with $R_t^{(t)} = u_t$, because the product of decays *after*
token $t$ up to token $t$ is empty and equals 1. Since the form holds at the
start of the window, where the list is empty and $\alpha = 1$, it holds at every
token in the window. Nothing is approximated. It is the same matrix written a
different way, exactly as the unrolled sum in the gating section was the same
matrix as the recurrence.

**A worked example.** Take $K = V = 2$, $\beta = 1$, and start a window with the
anchor $A = I$ in memory, $\alpha = 1$, and an empty list. Run two tokens
through both the dense rule and the factored rule.

*Token 1:* $\lambda = 0.5$, $k = (1, 0)$, $q = (0, 1)$, $v = (2, 0)$.

Dense: the decayed state is $0.5\,I$, its prediction for $k$ is
$0.5\,I\,k = (0.5, 0)$, so $u_1 = v - (0.5, 0) = (1.5, 0)$ and

$$
S_1 = 0.5\,I + u_1 k^\top
    = \begin{pmatrix} 2 & 0 \\ 0 & 0.5 \end{pmatrix},
\qquad y_1 = S_1 q = (0, 0.5).
$$

Factored: $\alpha \leftarrow 0.5$. The list is empty, so the prediction is
$\alpha A k = 0.5\,(1, 0) = (0.5, 0)$, the same $u_1 = (1.5, 0)$, and the
output is $\alpha A q + u_1 (k^\top q) = (0, 0.5) + 0 = (0, 0.5)$. Append
$R_1 = (1.5, 0)$, $K_1 = (1, 0)$. The memory holding $A$ still says $I$.
Check the implied state:

$$
\alpha A + R_1 K_1^\top
= \begin{pmatrix} 0.5 & 0 \\ 0 & 0.5 \end{pmatrix}
+ \begin{pmatrix} 1.5 & 0 \\ 0 & 0 \end{pmatrix}
= \begin{pmatrix} 2 & 0 \\ 0 & 0.5 \end{pmatrix} = S_1 .
$$

*Token 2:* $\lambda = 0.8$, $k = (0, 1)$, $q = (1, 1)$, $v = (0, 1)$.

Dense: decayed state $0.8\,S_1$, prediction $0.8\,S_1 k = (0, 0.4)$,
innovation $u_2 = (0, 0.6)$, and

$$
S_2 = 0.8\,S_1 + u_2 k^\top
    = \begin{pmatrix} 1.6 & 0 \\ 0 & 0.4 \end{pmatrix}
    + \begin{pmatrix} 0 & 0 \\ 0 & 0.6 \end{pmatrix}
    = \begin{pmatrix} 1.6 & 0 \\ 0 & 1.0 \end{pmatrix},
\qquad y_2 = S_2 q = (1.6, 1.0).
$$

Factored: $\alpha \leftarrow 0.5 \cdot 0.8 = 0.4$, and the existing residual is
decayed in place, $R_1 \leftarrow 0.8\,(1.5, 0) = (1.2, 0)$. The prediction is

$$
\alpha A k + R_1 (K_1^\top k) = 0.4\,(0, 1) + (1.2, 0)\cdot 0 = (0, 0.4),
$$

so $u_2 = (0, 0.6)$ as before. The output is

$$
\alpha A q + R_1 (K_1^\top q) + u_2 (k^\top q)
= 0.4\,(1, 1) + (1.2, 0)\cdot 1 + (0, 0.6)\cdot 1 = (1.6, 1.0).
$$

Append $R_2 = (0, 0.6)$, $K_2 = (0, 1)$. The memory holding $A$ still says
$I$. Check the implied state:

$$
\alpha A + R_1 K_1^\top + R_2 K_2^\top
= \begin{pmatrix} 0.4 & 0 \\ 0 & 0.4 \end{pmatrix}
+ \begin{pmatrix} 1.2 & 0 \\ 0 & 0 \end{pmatrix}
+ \begin{pmatrix} 0 & 0 \\ 0 & 0.6 \end{pmatrix}
= \begin{pmatrix} 1.6 & 0 \\ 0 & 1.0 \end{pmatrix} = S_2 .
$$

Both outputs and both implied states match the dense computation exactly,
while the dense path wrote the $2 \times 2$ matrix twice and the factored path
wrote it zero times. At the flush the kernel would evaluate that last sum once
and store $S_2$ into $A$'s memory, then reset $\alpha$ to 1 and empty the list.
Notice also that $R_1$ changed from $(1.5, 0)$ to $(1.2, 0)$ between tokens:
that is the superscript in $R_i^{(t)}$ made concrete.

### Why the rewritten form is cheap

Multiply the identity by a vector and every rank-one term collapses to a dot
product:

$$
\lambda_t S_{t-1} k_t = \alpha_t A k_t + \sum_{i \le t-1} R_i^{(t)}\,(K_i^\top k_t),
\qquad
\lambda_t S_{t-1} q_t = \alpha_t A q_t + \sum_{i \le t-1} R_i^{(t)}\,(K_i^\top q_t).
$$

This is the identity with both sides multiplied by $k_t$ or $q_t$ on the
right, then $(R_i K_i^\top)\, k_t = R_i\,(K_i^\top k_t)$ applied to each term
of the sum, the third time the same outer-product rule has done the work. The
upper limit is $t-1$ because $u_t$ has not been computed yet; it depends on
the first of these two products.

$A k_t$ is a matrix-vector product that *reads* $A$ and never writes it. Each
$K_i^\top k_t$ is a single number, and $R_i$ times that number is a scaled
256-vector. For a window of eight, the extra reads are a few thousand floats
against 65,536 in $A$. From these two products the kernel forms
$u_t = \beta_t(v_t - \lambda_t S_{t-1} k_t)$, emits
$y_t = \lambda_t S_{t-1} q_t + u_t (k_t^\top q_t)$, and appends $(u_t, k_t)$
to the list.

Per token this writes one scalar, rescales a handful of short vectors, and
appends two more. It never writes the $V \times K$ matrix. When the list
reaches the window length, one flush computes $\alpha_t A + \sum R_i K_i^\top +
u_t k_t^\top$ as a full matrix and stores it back into $A$'s memory. That is
the one dense write per window.

ReplaySSM's own framing is slightly different in what it caches: it keeps the
raw $(v_i, k_i)$ inputs and the decay factors and replays the recurrence at
flush time. For Mamba2, where the update does not depend on the state, that
gives an "output-only" kernel with no reconstruction at all until the flush.
For delta-rule models the innovation $u_t$ depends on what the state predicts
for $k_t$, so the state must be read every token anyway. The blog calls this
"state-and-output reconstruction". Storing the innovation rather than the raw
input, as above, is a small variation that makes the flush a plain sum.

### What it saves and what it costs

Over an eight-token window the dense kernel moves 16 matrix-equivalents and
the replay kernel about 9, so the kernel-level ceiling is roughly $1.78\times$.
The price is a sidecar buffer of eight $(R, K)$ pairs per head, $8 \times 512$
floats against the $65{,}536$ in the matrix, or 6.25% extra state, plus one
scalar. Longer windows save a little more traffic and cost proportionally more
sidecar; eight is where the curve flattens for this head size.

ReplaySSM reports 1.43 to 1.48$\times$ end-to-end decode speedup on models from
4B to 550B parameters. My own implementation of the same idea, inside FLA's
Gated DeltaNet layer on a 340M model and a single A6000, lands at about
$1.5\times$ at batch 256 and 512 under CUDA-graph replay. The agreement is not
a coincidence. Both are bounded by the same ratio of matrix reads to matrix
writes.

### The second gift: rollback for free

Speculative decoding drafts several tokens and verifies them in one pass. With
a transformer that is natural: rejected drafts are dropped from the KV cache.
With a recurrent state it is painful, because the state after draft $k$ has
irreversibly summarized drafts $1$ through $k$, so an engine must snapshot the
state per draft position and serially rebuild on rejection.

In the replay representation, drafts are just list entries. Rejecting the last
three drafts means deleting three pairs. Verifying $k$ drafts becomes one
matrix multiply of the cached keys against the draft queries instead of $k$
sequential state updates. ReplaySSM reports 1.87 to 1.96$\times$ throughput
over its own standard decoding from this, and 3.0 to 3.3$\times$ more
concurrent requests at a fixed memory budget once per-draft snapshots are gone.

## Two lessons from building it independently

I built this before connecting it to ReplaySSM, and two things went wrong in
ways the equations above hide.

**Do not divide by $\alpha$.** The obvious way to keep everything under one
common scale is to store $U_i = u_i / \alpha_i$ so that a single $\alpha_t$
multiplies the anchor and every pair together. It saves the per-token rescale
of the list. It also divides by a product of up to eight decays, and for a
strongly forgetting head that product reaches fp32 zero under ordinary gate
values. The dense kernel is perfectly happy with $\lambda = 0$; the
divide-by-$\alpha$ kernel produces `inf` then `NaN`. Decaying the stored
residuals in place, as in the tables above, costs a few short-vector writes and
is finite for every input, because each $R_i$ is only ever multiplied by
numbers in $(0, 1)$.

**Stagger the flush across layers.** If all 24 layers flush on the same token,
that token carries 24 dense writes and every eighth step is a latency spike.
Giving each layer a different-length first window spreads the flushes so about
three layers flush on every token. Mean cost is identical. P99 latency at batch
512 fell by about 23% in my runs. Nobody who only looks at mean throughput will
see this problem, and nobody who runs a latency SLO can ignore it.

## DASC: do not store what the model has already forgotten

ReplaySSM applies "reconstruct instead of store" to the state *within* a decode
window. DASC, from Yu et al., applies the same instinct to the state *across
requests*, in the prefix cache.

### The problem

Many requests share a long identical beginning: a system prompt, tool
definitions, a document. Servers compute that prefix once and save whatever
the model needs to continue from it. For a transformer that is the KV cache of
the prefix. For a linear-attention model it is the recurrent state after the
prefix: the matrix $S$ for every head in every layer, plus the short
convolution tails. It does not grow with prefix length, but it is not small.
For the 340M GDN model above it is 24.5 MiB per prefix; for the 48B Kimi Linear
model it is 40 MiB.

A prefix cache lives in a pool of fixed size. At 24.5 MiB per entry a 2 GB pool
holds about 80 prefixes, and every eviction costs a full recompute on the next
hit. Whoever shrinks the entry holds more prefixes in the same pool.

### The observation

Go back to the unrolled gated recurrence:

$$
S_T = \sum_{i \le T} \Big( \prod_{j=i+1}^{T} \lambda_j \Big) v_i k_i^\top .
$$

The weight on a token that arrived $n$ positions ago is roughly $\lambda^n$,
and $\lambda$ is a learned property of the head. Define a head's *retention
horizon* as the number of tokens after which that weight has fallen below
$10^{-3}$:

$$
H = \frac{\log 10^{-3}}{\log \lambda} .
$$

Two heads on the same 8,000-token prefix:

| $n$ tokens ago | weight, $\lambda = 0.95$ | weight, $\lambda = 0.9999$ |
|---:|---:|---:|
| 10 | 0.60 | 0.999 |
| 135 | 0.001 | 0.987 |
| 1,000 | $6 \times 10^{-23}$ | 0.905 |
| 8,000 | $10^{-179}$ | 0.449 |

The first head has $H \approx 135$. Every term older than 135 tokens weighs
less than one part in a thousand, so to that tolerance its final state is a
function of the last 135 tokens only. The second head has $H \approx 69{,}000$.
A token from the start of the prefix still carries 45% of its weight, so every
term matters.

That asymmetry is the whole method. A short-horizon head's state can be
*rebuilt* by feeding its last $H$ tokens through the model from a zero state,
because the terms the zero start omits are the ones that had already decayed
away. A long-horizon head's state cannot be rebuilt without re-running the whole
prefix, which is the cost the cache exists to avoid.

![Ninety-six heads sorted by retention horizon, with a cutoff at 256 tokens
separating 56 heads that are rebuilt from 40 that are stored](/assets/linear-attention/dasc-horizon-split.svg)

### The method

1. **Score each head once.** In GDN, $\lambda = \exp(-\exp(A_{\log}) \cdot
   \mathrm{softplus}(g + \mathrm{dt\_bias}))$. Evaluate it at a nominal gate
   input, take the log, and compute $H$ per head. This depends on weights, not
   on any prompt, so it is done offline.
2. **Save only long-horizon heads.** Choose a cutoff $W_{\max}$. At checkpoint
   time, write to storage only heads with $H > W_{\max}$.
3. **Rebuild the rest on load.** Run the model over the last $W_{\max}$ tokens
   of the prefix from a zero state. Every skipped head has $H < W_{\max}$, so
   this replay reproduces its state to within $10^{-3}$.
4. **Merge and continue.** Saved heads are exact, rebuilt heads are
   approximate, and ordinary decode proceeds on the assembled state.

The paper calls the variant with replay DASC-WR, for *with recovery*. Dropping
short heads without any replay also exists and saves more, but at a quality
cost that did not pass my gates.

### What it gives

The paper reports $2.63\times$ compression of KDA recurrent-state checkpoints
on Kimi Linear with a 42.6% reduction in time to first token on prefix hits,
and results on Qwen-family GDN models. On the 340M GDN checkpoint I have been
using throughout, with $W_{\max} = 256$:

| quantity | value |
|---|---:|
| heads with $H > 256$, stored | 40 of 96 |
| heads rebuilt by replay | 56 of 96 |
| checkpoint size | 24.56 → 10.56 MiB |
| bytes saved | 57.0% |
| prefixes per fixed budget | $2.33\times$ |
| worst-slice perplexity retention after load | 99.81% |
| worst-slice top-1 agreement | 98.8% |

The control that matters: keep the *same* 40-of-96 budget but pick the 40 heads
at random instead of by horizon. Five random draws average 96.7% perplexity
retention and 92.5% top-1 agreement, a broken model. The saving comes from
choosing the heads whose decay has already done the forgetting, not from
dropping state in general.

### What it costs

Loading is no longer a memory copy. It is a forward pass over $W_{\max}$
tokens, which on the A6000 took 60 to 65 ms for a batch of four at 256 tokens.
Against recomputing an 8,000-token prefix that is nothing. Against a plain load
of a full checkpoint it is added latency, and the right $W_{\max}$ depends on
where the pool is actually bottlenecked: capacity or load time. The paper's
42.6% TTFT win says the trade favors compression at their scale; I have not
measured it end to end on mine.

Two things to be honest about. The horizon formula evaluates $\lambda$ at a
nominal gate input, but in GDN the gate is data-dependent, so $H$ is a static
estimate of a quantity that varies per token. The quality gates, not the
formula, are what qualify a cutoff. And for KDA the decay is per key channel
rather than per head, so the unit of selection becomes a (head, channel) row;
the paper's $2.63\times$ is on those finer units.

### The two methods are one idea

ReplaySSM keeps a stale anchor plus a short list of recent inputs and rebuilds
the state when it must. DASC keeps the heads that remember and rebuilds the
heads that do not from the recent inputs. Both replace a stored matrix with
"the recent past plus something cheap", and both work because the recurrence
is a decayed sum in which recent terms dominate exactly when $\lambda$ is
small.

They also compose without effort. A restored DASC checkpoint is a dense state,
which is a ReplaySSM anchor with an empty list. And the horizon idea points at
a decode-time optimization ReplaySSM does not have: when a head's $\alpha_t$
inside a window has fallen below $10^{-3}$, its anchor contributes nothing to
the output, and the kernel can skip loading it.

## What ReplaySSM does not touch

The write is gone. The read is still there, and it is now the entire cost of
the kernel. That points at the next set of ideas.

- **Quantize the anchor.** It is read every token and written once per
  window, so an int8 anchor with fp32 residuals cuts the dominant remaining
  term by 4$\times$ and injects quantization error once per flush rather than
  per token.
- **Skip the anchor when $\alpha$ is tiny.** For a fast-forgetting head the
  anchor's contribution is numerically zero after a few tokens. The kernel
  already knows $\alpha$; it can skip the load.
- **Admission without a flush.** The list rank is shared across a batch. A
  sequence joining at rank 0 can be padded with zero pairs, which is exact.
  Without this, continuous batching forces a flush on every admission.
- **Low batch.** None of this helps at batch 2, where the state is small and
  launch overhead dominates. That is a different problem with a different fix.

The DASC section above is the prefix-cache half of the same story, and its
horizon test is the natural trigger for the anchor-skipping item in this list.

## References

- Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive
  Transformers with Linear Attention*, 2020. The factorization in the first
  section.
- Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*,
  ICLR 2025, [arXiv:2412.06464](https://arxiv.org/abs/2412.06464).
- Dao et al., *Mamba2 / State Space Duality*, 2024. The decay
  parameterization GDN reuses.
- Kimi Team, *Kimi Linear*,
  [arXiv:2510.26692](https://arxiv.org/abs/2510.26692). Channel-wise decay.
- Dao Lab, *ReplaySSM*, 2026,
  [dao-lab.ai/blog/2026/replayssm](https://dao-lab.ai/blog/2026/replayssm/).
- Yu et al., *DASC: Decay-Aware State Compression for Hybrid Linear-Attention
  Serving*, 2026, [arXiv:2608.30386](https://arxiv.org/abs/2608.30386).
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
  the kernels referred to throughout.
