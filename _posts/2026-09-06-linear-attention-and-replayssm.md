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
y_t = \frac{\sum_{i=1}^{t} \exp(q_t^\top k_i)\, v_i}{\sum_{j=1}^{t} \exp(q_t^\top k_j)}. \tag{1}
$$

Every previous key and value participates. The cache holds all of them, so
memory is $O(t \cdot d)$ and so is the work per token.

Replace $\exp(q^\top k)$ by an inner product of feature maps,
$\phi(q)^\top \phi(k)$. The query is the same in every term, so it factors out
of the sum:

$$
y_t = \sum_{i=1}^{t} \phi(q_t)^\top \phi(k_i)\, v_i
    = \Big( \sum_{i=1}^{t} v_i\, \phi(k_i)^\top \Big) \phi(q_t)
    = S_t\, \phi(q_t).
$$

Two facts make the middle step legal. First, $\phi(q_t)^\top \phi(k_i)$ is a
scalar, and a scalar times a vector can be written in either order, so
$\big(\phi(q_t)^\top \phi(k_i)\big) v_i = v_i \big(\phi(k_i)^\top \phi(q_t)\big)$.
Second, that product is $v_i \phi(k_i)^\top$ applied to $\phi(q_t)$, because for
any vectors $a, b, c$ the outer product obeys $(a b^\top) c = a (b^\top c)$: the
matrix $a b^\top$ times $c$ is $a$ scaled by the dot product $b^\top c$. Since
$\phi(q_t)$ does not depend on the summation index $i$, it can be pulled outside
the sum, leaving $\sum_{i=1}^{t} v_i \phi(k_i)^\top$ as a matrix that no longer knows
anything about the query.

The entire history has collapsed into one matrix $S_t \in \mathbb{R}^{V \times K}$
with a one-step recurrence:

$$
S_t = S_{t-1} + v_t k_t^\top, \qquad y_t = S_t q_t. \tag{2}
$$

The recurrence is just the sum written incrementally: $S_t = \sum_{i=1}^{t}
v_i k_i^\top$ and $S_{t-1} = \sum_{i=1}^{t-1} v_i k_i^\top$ differ by exactly
the $i = t$ term.

Modern variants drop $\phi$ and the normalizer, L2-normalize $q$ and $k$, and
put an RMSNorm on the output instead. Memory is $O(VK)$ regardless of context.
Each decode token costs one read and one write of $S$.

## Three refinements that make it work

### Forgetting

The plain recurrence never forgets. A scalar gate
$\alpha_t \in (0,1)$ per head fixes that:

$$
S_t = \alpha_t S_{t-1} + v_t k_t^\top. \tag{3}
$$

Unrolling means substituting the recurrence into itself until only inputs
remain. Start from $S_0 = 0$ and apply the rule three times:

$$
\begin{aligned}
S_1 &= v_1 k_1^\top, \\
S_2 &= \alpha_2 S_1 + v_2 k_2^\top
     = \alpha_2\, v_1 k_1^\top + v_2 k_2^\top, \\
S_3 &= \alpha_3 S_2 + v_3 k_3^\top
     = \alpha_3 \alpha_2\, v_1 k_1^\top + \alpha_3\, v_2 k_2^\top + v_3 k_3^\top.
\end{aligned}
$$

Each substitution multiplies everything already present by one more $\alpha$
and adds one new undecayed term. After $t$ steps the term from token $i$ has
been multiplied by $\alpha_{i+1}, \alpha_{i+2}, \dots, \alpha_t$, one factor
for every token that arrived after it, and the newest term by nothing:

$$
S_t = \sum_{i=1}^{t} \Big( \prod_{j=i+1}^{t} \alpha_j \Big) v_i k_i^\top. \tag{4}
$$

The empty product for $i = t$ is 1. If every $\alpha$ equals a constant
$\alpha$, the weight on token $i$ is $\alpha^{\,t-i}$, which is exponential
forgetting with a horizon set by how close $\alpha$ is to 1.

This is the Mamba2 and gated-linear-attention family.

### The delta rule

Treat $S$ as an associative memory that maps keys to
values: reading key $k$ returns $S k$. After storing $v_1$ under $k_1$ we want
$S k_1 \approx v_1$. Plain linear attention stores by adding an outer product,
$S \leftarrow S + v k^\top$, so reading $k$ back returns $v\,(k^\top k)$ plus
interference from every other stored key. If the same key is written twice,
the two values add and a read returns their sum. This memory can append a fact
but never update one.

The delta rule fixes that by treating a write as an *error correction*. It is
built in three steps.

*Step 1: measure how wrong the memory currently is.* Read $k_t$ from the old
state. $S_{t-1} k_t$ is what the memory would answer right now. Compare it to
what we want it to answer:

$$
e_t = S_{t-1} k_t - v_t .
$$

This is a vector of length $V$. If the memory already held $v_t$ under $k_t$,
the error is zero and there is nothing to do.

*Step 2: turn the error into a matrix that lives only along $k_t$.* We want
to change the memory's answer for $k_t$ without disturbing its answer for
other keys. The outer product $e_t k_t^\top$ is a $V \times K$ matrix with
exactly that property: multiplying it by $k_t$ gives $e_t\,(k_t^\top k_t) =
e_t$ when $k_t$ has unit length, and multiplying it by any vector orthogonal
to $k_t$ gives zero.

*Step 3: subtract a fraction of it.* Subtracting all of $e_t k_t^\top$ would
remove the whole error at once. A factor $\beta_t \in (0, 1)$ removes part of
it:

$$
S_t = S_{t-1} - \beta_t (S_{t-1} k_t - v_t) k_t^\top
    = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top. \tag{5}
$$

The first form is Step 3 with $e_t$ written out. It is still a rank-one
update, the same shape as plain linear attention, but the vector being written
is the error rather than the raw value.

*Check that it does what we wanted.* Read $k_t$ from the new state:

$$
S_t k_t = S_{t-1} k_t - \beta_t\, e_t\,(k_t^\top k_t)
        = S_{t-1} k_t - \beta_t (S_{t-1} k_t - v_t)
        \qquad\text{using } k_t^\top k_t = 1 .
$$

With $\beta_t = 1$ the right side is $S_{t-1} k_t - S_{t-1} k_t + v_t = v_t$.
The read returns exactly the new value, whatever was stored before. With
$\beta_t = 0.5$ it returns the midpoint of old and new.

*Where "gradient step" comes in.* Steps 1 to 3 are one step of gradient
descent on the loss $\tfrac{1}{2}\lVert S k_t - v_t \rVert^2$ with learning
rate $\beta_t$. By the chain rule the derivative of $\tfrac{1}{2}\lVert r
\rVert^2$ with respect to $r$ is $r$, and $r = S k_t - v_t$ depends on $S$
through right-multiplication by $k_t$, which contributes $k_t^\top$. So the
gradient is $(S k_t - v_t)\, k_t^\top$, the matrix from Step 2. The gradient
view is not needed to understand the rule; it explains why this particular
update is the natural one.

*The second form is the first form rearranged.* Expand the bracket:
$S_{t-1} - \beta_t S_{t-1} k_t k_t^\top + \beta_t v_t k_t^\top$. The first two
terms both have $S_{t-1}$ on the left; factor it out to get
$S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$. This form says the
same thing in two stages. First, multiply the old state by $(I - \beta_t k_t
k_t^\top)$; with unit $k_t$ and $\beta_t = 1$ that matrix zeroes the $k_t$
component of every row of $S$ and leaves the other components alone. Second,
add $\beta_t v_t k_t^\top$, which writes the new value into the now-empty slot.
Erase, then write.

Plain linear attention would have *added* $v_t$ on top of
whatever was already stored under $k_t$. That is the difference between an
accumulator and an error-correcting memory, and it is why DeltaNet models
handle repeated keys so much better than their predecessors.

Where the two ingredients come from in Gated DeltaNet: the layer L2-normalizes
$q$ and $k$ inside the kernel, which is what makes $k_t^\top k_t = 1$ and the
overwrite exact; and $\beta_t = \sigma(x_t W_\beta)$ is one scalar per head in
$(0, 1)$, computed from the current token. An optional flag doubles it to
$(0, 2)$, so the eigenvalue $1 - \beta_t$ of the projection can go negative and
the model can flip the sign of a stored association rather than only shrink it.

What the rule buys is retrieval. In-context recall, copying, and key-value
lookup all need a memory that can update a slot, and DeltaNet-family models
beat gated linear attention on exactly those tasks. What it costs shows up
later in this post: because the innovation depends on the current state through
$S_{t-1} k_t$, the state must be *read* on every token even when it is not
written, which is why ReplaySSM cannot use its cheapest output-only path for
delta-rule models.

Two things the rule glosses over. It is one gradient step, not a solve, so
with $\beta_t < 1$ or a non-unit key the write is partial and the old value
lingers. And each write only corrects the $k_t$ direction, so a key that is not
exactly orthogonal to earlier keys still picks up interference from them; the
state dimension and the number of stored facts both matter.

### Both together: Gated DeltaNet

Decay first, then the delta step on the
decayed state:

$$
S_t = \alpha_t S_{t-1} + \beta_t\big(v_t - \alpha_t S_{t-1} k_t\big) k_t^\top. \tag{6}
$$

To see where this comes from, take the delta rule's second form and replace
$S_{t-1}$ by the decayed state $\alpha_t S_{t-1}$ everywhere:

$$
S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top
    = \alpha_t S_{t-1} - \beta_t \alpha_t S_{t-1} k_t k_t^\top + \beta_t v_t k_t^\top .
$$

The last two terms share the factor $\beta_t(\cdot) k_t^\top$; pulling it out
gives $\beta_t (v_t - \alpha_t S_{t-1} k_t) k_t^\top$. The quantity in the
bracket is the retrieval error measured against the *decayed* memory, which is
the right thing to correct since the decayed memory is what the next reader
will see.

Name the rank-one innovation $u_t := \beta_t(v_t - \alpha_t S_{t-1} k_t)$.
Then the update and readout are

$$
S_t = \alpha_t S_{t-1} + u_t k_t^\top, \qquad
y_t = S_t q_t = \alpha_t S_{t-1} q_t + u_t\,(k_t^\top q_t). \tag{7}
$$

The readout on the right is the update rule multiplied through by $q_t$:
$(\alpha_t S_{t-1} + u_t k_t^\top)\, q_t = \alpha_t S_{t-1} q_t + u_t k_t^\top q_t$,
and $k_t^\top q_t$ is a scalar, so $u_t k_t^\top q_t = u_t\,(k_t^\top q_t)$ by
the same outer-product rule used in the first section. That is the whole
derivation, but it changes what a kernel has to do.

The second form matters for what follows: the output needs two matrix-vector
products against the *old* state, one with $k_t$ to form the innovation and one
with $q_t$, and never needs $S_t$ to exist as a matrix.

### Where the scalars come from

In the FLA implementation: $q$, $k$, $v$ are
projections followed by a width-4 depthwise causal convolution and SiLU; $q$
and $k$ are L2-normalized; $\alpha_t = \exp(-\exp(A_{\log}) \cdot
\mathrm{softplus}(x_t W_a + \mathrm{dt\_bias}))$ with learned per-head
$A_{\log}$ and $\mathrm{dt\_bias}$, which is Mamba2's discretized decay
reused verbatim; and $\beta_t = \sigma(x_t W_\beta)$. Kimi's KDA differs in
one place: $\alpha_t$ is a vector over key channels instead of a scalar per
head.

## Prefill runs the same recurrence in chunks

Everything above is written one token at a time. That is the right form for
decode, where there is exactly one new token per step. It is the wrong form for
prefill, where thousands of prompt tokens arrive at once: applying equation 6
sequentially means thousands of dependent $V \times K$ updates, none of which
can use tensor cores, and the GPU idles. The other extreme, materializing the
full $T \times T$ causal attention matrix that equation 4 implies, is quadratic
in sequence length and defeats the point of a fixed-size state.

The chunkwise form sits between the two. Cut the sequence into chunks of $C$
tokens, typically 64. Carry the state across chunk boundaries as in the
recurrent form. Inside a chunk, compute all $C$ outputs with a few dense matrix
multiplies. Sequential work drops from $T$ steps to $T / C$ steps, and the work
inside each step is what tensor cores are built for.

Getting there takes three moves, and it helps to know the plan before the
details. First, write the state at any position inside a chunk as the boundary
state plus a sum over the chunk's own innovations. Second, notice that for the
delta rule the innovations themselves are unknowns, and that they satisfy a
small triangular linear system that can be solved in one shot. Third, arrange
that solve so it does not depend on the boundary state, so every chunk's solve
can run in parallel before the sequential pass over states. Once the
innovations are in hand, the outputs and the next boundary state are two
matmuls each.

### Move 1: the state inside a chunk, in terms of the boundary state

Take one chunk. Let $S_0$ be the state at its start, and index positions inside
the chunk by $j = 1, \dots, C$. Write $g_j = \alpha_1 \alpha_2 \cdots \alpha_j$
for the cumulative decay from the chunk start to position $j$, with $g_0 = 1$.
Unrolling the innovation form of the recurrence, equation 7, from $S_0$ instead
of from zero gives the state at any position in the chunk as

$$
S_j = g_j\, S_0 + \sum_{m=1}^{j} \frac{g_j}{g_m}\, u_m k_m^\top . \tag{8}
$$

This is equation 4 with two changes: the sum starts at the chunk boundary
rather than at token 1, and everything before the boundary is compressed into
the single term $g_j S_0$. The ratio $g_j / g_m$ is the product of the decays
strictly after position $m$ up to $j$: dividing $g_j = \alpha_1 \cdots \alpha_j$
by $g_m = \alpha_1 \cdots \alpha_m$ cancels the shared prefix and leaves
$\alpha_{m+1} \cdots \alpha_j$. It is the same weight that appeared in equation
4, written as a ratio of two per-position numbers so that it can later sit in a
matrix.

Equation 8 needs two ingredients: the cumulative decays $g_j$, which are a
running product of known gates, and the innovations $u_1, \dots, u_j$. For
plain gated linear attention the innovation is just the value, $u_m = v_m$,
and equation 8 is already computable. For the delta rule it is not, because
each $u_m$ is defined through the state it is written into. That is the
problem the next move solves.

### Move 2: the delta rule's innovations satisfy a triangular system

For Gated DeltaNet the innovation at position $m$ is the write strength times
the error against the decayed state just before it, $u_m = \beta_m (v_m -
\alpha_m S_{m-1} k_m)$. That looks like it forces a sequential loop: $u_m$
needs $S_{m-1}$, which needs $u_{m-1}$, and so on. But $S_{m-1}$ is given by
equation 8 at $j = m-1$. Substitute it in, and use $\alpha_m g_{m-1} = g_m$:

$$
u_m = \beta_m v_m - \beta_m g_m\, S_0 k_m
      - \beta_m \sum_{n=1}^{m-1} \frac{g_m}{g_n}\, (k_n^\top k_m)\, u_n .
$$

Read the three terms. The first is the raw value. The second is what the
boundary state predicts for $k_m$, after decay; it is known before the chunk
starts. The third is what the earlier writes *in this chunk* predict for
$k_m$: each $u_n$ was written along $k_n$, reading $k_m$ picks it up scaled by
the key overlap $k_n^\top k_m$, and the decay between $n$ and $m$ scales it
again. The unknown $u_m$ appears on the left, and the unknowns $u_1, \dots,
u_{m-1}$ appear on the right, each multiplied by a known number. That is a
linear equation, and the dependence only runs backwards, so it is triangular.

Stack the $C$ equations. Let $U, V \in \mathbb{R}^{C \times V}$ and $K \in
\mathbb{R}^{C \times K}$ hold the chunk's innovations, values, and keys as
rows. Moving the $u_n$ terms to the left gives

$$
(I + L)\, U = \operatorname{diag}(\beta)\big(V - \operatorname{diag}(g)\, K S_0^\top\big),
\qquad
L_{mn} = \beta_m \frac{g_m}{g_n}\, k_n^\top k_m \;\; (n < m). \tag{9}
$$

$L$ is strictly lower triangular, so $I + L$ has ones on the diagonal, is
always invertible, and is inverted by forward substitution in $O(C^2)$
operations per chunk, negligible next to the matmuls. Solving equation 9 gives
all $C$ innovations at once, with no loop over positions.

### Move 3: take the boundary state out of the solve

As written, the right-hand side of equation 9 contains $S_0$. Chunk $t$'s
solve cannot start until chunk $t-1$'s state is known, so the solves would be
as sequential as the states, and the parallelism would be lost again. The
Gated DeltaNet paper avoids this by splitting the right-hand side. The system
is linear, so solve it once against $V$ and once against $K$, and combine
afterwards:

$$
T = (I + L)^{-1} \operatorname{diag}(\beta), \qquad
\tilde U = T\, V, \qquad
W = T\, K, \qquad
U = \tilde U - \operatorname{diag}(g)\, W S_0^\top . \tag{10}
$$

$T$, $\tilde U$, and $W$ depend only on the chunk's own keys, values, gates,
and write strengths. They are computed for every chunk in parallel, in one
batched pass, before any state exists. The boundary state then enters through
one matmul per chunk, $W S_0^\top$, inside the sequential pass that carries $S$
from chunk to chunk. Substituting the last line of equation 10 into equation 9
and using $(I + L)\,T = \operatorname{diag}(\beta)$ confirms it solves the
system.

### Putting it together: outputs and the next boundary state

With $U$ known, apply equation 8 to each query and stack the $C$ results as
rows of an output matrix $O \in \mathbb{R}^{C \times V}$. Let $Q \in
\mathbb{R}^{C \times K}$ hold the queries as rows, and let $\Gamma$ be the
$C \times C$ matrix with entries $\Gamma_{jm} = g_j / g_m$ for $m \le j$ and
zero above the diagonal. Then

$$
O = \underbrace{\operatorname{diag}(g)\, Q\, S_0^\top}_{\text{inter-chunk}}
  + \underbrace{\big(\Gamma \odot Q K^\top\big)\, U}_{\text{intra-chunk}} . \tag{11}
$$

The first term is what the carried state contributes to every position: one
$C \times K$ by $K \times V$ matmul. The second is a masked $C \times C$
attention matrix applied to the chunk's own innovations: entry $(j, m)$ of
$Q K^\top$ is $q_j^\top k_m$, the mask $\Gamma$ supplies both causality and the
decay weight, and the product with $U$ sums $\frac{g_j}{g_m} u_m (k_m^\top q_j)$
over $m \le j$, which is row $j$ of equation 8 applied to $q_j$.

The state handed to the next chunk is equation 8 at $j = C$, also as matmuls:

$$
S_C = g_C\, S_0 + U^\top \operatorname{diag}\!\Big(\frac{g_C}{g_m}\Big) K . \tag{12}
$$

The whole algorithm is therefore:

1. For every chunk in parallel: cumulative decays $g$, the decay-weighted key
   overlaps that fill $L$, the triangular solve for $T$, and $\tilde U = T V$
   and $W = T K$.
2. Sequentially over chunks: $U = \tilde U - \operatorname{diag}(g) W
   S_0^\top$, the outputs by equation 11, and the next boundary state by
   equation 12.

Step 1 is where all the parallel matmul work lives. Step 2 touches the
$V \times K$ state once per chunk, and each of its operations is a matmul with
$C$ as one dimension. Every operation in both steps is a dense matmul, an
elementwise product on a $C \times C$ tile, or the small triangular solve.

### Where this comes from, and where it lives in the code

This is the Gated DeltaNet paper's algorithm. Its equation 10 is this post's
equation 6 with the factors written as $S_{t-1}\big(\alpha_t (I - \beta_t k_t
k_t^\top)\big) + \beta_t v_t k_t^\top$, the same recurrence. Its equations 6
and 7 are this post's equation 10 for the ungated case, and its $\tilde U$
formula in section 3.3 is the gated one, where the matrix being inverted is
written $I + \mathrm{strictLower}\big(\operatorname{diag}(\beta)\,(\Gamma
\odot K K^\top)\big)$; $\Gamma \odot K K^\top$ is exactly the decay-weighted
key overlap $\frac{g_m}{g_n} k_n^\top k_m$ that fills $L$ here. The paper calls
the split in Move 3 the UT transform, and the pair $(W, \tilde U)$ the WY
representation, after the classical result that a product of Householder
reflectors $\prod (I - \beta_m k_m k_m^\top)$ equals $I$ minus a rank-$C$
matrix $K^\top W$. Notation differs in one place: the paper writes the
within-chunk cumulative decay as $\gamma^{\,r}_{[t]}$, which is $g_r$ here;
$\gamma$ is reserved in this post for ReplaySSM's running product.

FLA follows the paper's split exactly. In
[`chunk.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py)
the step labelled "fused kkt + solve_tril + recompute_w_u" is Step 1 above,
run over all chunks at once: form the decay-weighted $K K^\top$, invert the
unit lower-triangular matrix, and produce $W$ and $\tilde U$. Only afterwards
does the state kernel run Step 2, sweeping the chunks in order. The triangular
solve lives in `fla/ops/utils/solve_tril.py`, which inverts $16 \times 16$
blocks and assembles the $64 \times 64$ result. The decode kernel never touches
any of this: with one token per step the system in equation 9 is a single
scalar.

### What this buys and where it stops

| form | sequential steps | work | tensor cores |
|---|---:|---:|---|
| recurrent, equation 6 | $T$ | $O(T \cdot VK)$ | no |
| fully parallel, equation 4 | 1 | $O(T^2 (K + V))$ | yes, but quadratic |
| chunkwise, equations 8 to 12 | $T / C$ | $O(T C (K + V) + (T/C)\, VK \cdot C)$ | yes |

With $C = 64$ the chunkwise form is linear in $T$, uses matmuls for everything
but a small triangular solve, and this is what FLA's chunk kernel implements
for Gated DeltaNet. It is compute-bound in the way prefill should be.

It does nothing for decode. With one new token per step the chunk has size 1,
equation 11 collapses back to equation 7, and the cost is once again the read
and write of the $V \times K$ state. That is why the two halves of this post
look so different: prefill is a matmul problem solved by chunking, while decode
is a memory-traffic problem that the next section quantifies. It also shows
what a prefix checkpoint is: the carried state $S_C$ at a chunk boundary,
which is exactly what DASC compresses later in the post, and what the DASC
replay recomputes over its last 256 tokens using this chunk kernel. And it
previews ReplaySSM: that method touches the state once per window of decode
tokens, the way this form touches it once per chunk of prefill tokens. Its
end-of-window flush is a chunk update of the form of equation 12 with $U$
already known, since each innovation was computed at its own token, so the
flush needs no triangular solve at all.

## What one decode token actually costs

Take a small concrete model: 24 layers, 4 heads, $K = V = 256$, state in fp32.
One head's state is $256 \times 256 \times 4 = 256$ KiB. One sequence carries
1 MiB per layer and 24 MiB across the model.

The dense kernel handles a batch of $B$ sequences in one launch. Per layer per
token it must load every state, apply $\alpha_t S + u_t k_t^\top$, and store
every state back. Every entry changes when $\alpha_t \ne 1$, so nothing can
be skipped:

$$
\text{bytes} = \underbrace{2}_{\text{read+write}} \cdot
\underbrace{4}_{\text{fp32}} \cdot B \cdot H \cdot V \cdot K . \tag{13}
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
"decay, then add": $S_{\text{new}} = \alpha S_{\text{old}} + u$. Take
$\alpha = 0.5$, start at $S = 8$, and feed in $u = 4, 2, 6$. The dense
way rewrites $S$ each step: $8 \to 8 \to 6 \to 9$.

Now keep the starting value $A = 8$ untouched, a scale $\gamma$ that starts at
1, and a list $R$. On each token, multiply $\gamma$ by $\alpha$, multiply every
entry already in $R$ by $\alpha$, and append the new input unchanged:

| after | $\gamma$ | $A$ | $R$ | $\gamma A + \sum R$ |
|---|---:|---:|---|---:|
| start | 1 | 8 | [ ] | 8 |
| token 1 | 0.5 | 8 | [4] | 8 |
| token 2 | 0.25 | 8 | [2, 2] | 6 |
| token 3 | 0.125 | 8 | [1, 1, 6] | 9 |

Same answers, and the slot holding 8 was never written. The list holds each
past input at its *current* decayed value.

### The same thing with matrices

Let $A$ be the state matrix stored at the start of a window, at step $s$, and
never written during the window. Let $\gamma_t$ be the product of decays since
then. Store each innovation as a pair $(R_i, K_i)$ with $K_i = k_i$ and $R_i$
the value-side vector $u_i$ at its current decayed scale. Then for every $t$
in the window, summing over the tokens $s+1, \dots, t$ seen since the anchor,

$$
S_t = \gamma_t A + \sum_{i=s+1}^{t} R_i^{(t)} K_i^\top,
\qquad R_i^{(t)} = u_i \prod_{j=i+1}^{t} \alpha_j ,
\qquad K_i = k_i . \tag{14}
$$

The proof is one substitution. Assume it holds at $t-1$, then

$$
\begin{aligned}
S_t &= \alpha_t S_{t-1} + u_t k_t^\top \\
    &= \alpha_t\Big(\gamma_{t-1} A + \sum_{i=s+1}^{t-1} R_i^{(t-1)} K_i^\top\Big) + u_t k_t^\top \\
    &= (\alpha_t \gamma_{t-1})\, A + \sum_{i=s+1}^{t-1} \big(\alpha_t R_i^{(t-1)}\big) K_i^\top + u_t k_t^\top \\
    &= \gamma_t A + \sum_{i=s+1}^{t-1} R_i^{(t)} K_i^\top + R_t^{(t)} K_t^\top
        \qquad\text{with } R_t^{(t)} := u_t,\; K_t := k_t \\
    &= \gamma_t A + \sum_{i=s+1}^{t} R_i^{(t)} K_i^\top .
\end{aligned} \tag{15}
$$

The last line renames three things without changing any value: $\alpha_t\gamma_{t-1}$
is $\gamma_t$ by the definition of $\gamma$ as a running product; each
$\alpha_t R_i^{(t-1)}$ is $R_i^{(t)}$ by the definition of $R$ as the input
times every decay since it arrived; and the fresh $u_t k_t^\top$ is the
$i = t$ term with $R_t^{(t)} = u_t$, because the product of decays *after*
token $t$ up to token $t$ is empty and equals 1. Since the form holds at the
start of the window, where the list is empty and $\gamma = 1$, it holds at every
token in the window. Nothing is approximated. It is the same matrix written a
different way, exactly as the unrolled sum in the gating section was the same
matrix as the recurrence.

**A worked example.** Take $K = V = 2$, $\beta = 1$, and start a window with the
anchor $A = I$ in memory, $\gamma = 1$, and an empty list. Run two tokens
through both the dense rule and the factored rule.

*Token 1:* $\alpha = 0.5$, $k = (1, 0)$, $q = (0, 1)$, $v = (2, 0)$.

Dense: the decayed state is $0.5\,I$, its prediction for $k$ is
$0.5\,I\,k = (0.5, 0)$, so $u_1 = v - (0.5, 0) = (1.5, 0)$ and

$$
S_1 = 0.5\,I + u_1 k^\top
    = \begin{pmatrix} 2 & 0 \\ 0 & 0.5 \end{pmatrix},
\qquad y_1 = S_1 q = (0, 0.5).
$$

Factored: $\gamma \leftarrow 0.5$. The list is empty, so the prediction is
$\gamma A k = 0.5\,(1, 0) = (0.5, 0)$, the same $u_1 = (1.5, 0)$, and the
output is $\gamma A q + u_1 (k^\top q) = (0, 0.5) + 0 = (0, 0.5)$. Append
$R_1 = (1.5, 0)$, $K_1 = (1, 0)$. The memory holding $A$ still says $I$.
Check the implied state:

$$
\gamma A + R_1 K_1^\top
= \begin{pmatrix} 0.5 & 0 \\ 0 & 0.5 \end{pmatrix}
+ \begin{pmatrix} 1.5 & 0 \\ 0 & 0 \end{pmatrix}
= \begin{pmatrix} 2 & 0 \\ 0 & 0.5 \end{pmatrix} = S_1 .
$$

*Token 2:* $\alpha = 0.8$, $k = (0, 1)$, $q = (1, 1)$, $v = (0, 1)$.

Dense: decayed state $0.8\,S_1$, prediction $0.8\,S_1 k = (0, 0.4)$,
innovation $u_2 = (0, 0.6)$, and

$$
S_2 = 0.8\,S_1 + u_2 k^\top
    = \begin{pmatrix} 1.6 & 0 \\ 0 & 0.4 \end{pmatrix}
    + \begin{pmatrix} 0 & 0 \\ 0 & 0.6 \end{pmatrix}
    = \begin{pmatrix} 1.6 & 0 \\ 0 & 1.0 \end{pmatrix},
\qquad y_2 = S_2 q = (1.6, 1.0).
$$

Factored: $\gamma \leftarrow 0.5 \cdot 0.8 = 0.4$, and the existing residual is
decayed in place, $R_1 \leftarrow 0.8\,(1.5, 0) = (1.2, 0)$. The prediction is

$$
\gamma A k + R_1 (K_1^\top k) = 0.4\,(0, 1) + (1.2, 0)\cdot 0 = (0, 0.4),
$$

so $u_2 = (0, 0.6)$ as before. The output is

$$
\gamma A q + R_1 (K_1^\top q) + u_2 (k^\top q)
= 0.4\,(1, 1) + (1.2, 0)\cdot 1 + (0, 0.6)\cdot 1 = (1.6, 1.0).
$$

Append $R_2 = (0, 0.6)$, $K_2 = (0, 1)$. The memory holding $A$ still says
$I$. Check the implied state:

$$
\gamma A + R_1 K_1^\top + R_2 K_2^\top
= \begin{pmatrix} 0.4 & 0 \\ 0 & 0.4 \end{pmatrix}
+ \begin{pmatrix} 1.2 & 0 \\ 0 & 0 \end{pmatrix}
+ \begin{pmatrix} 0 & 0 \\ 0 & 0.6 \end{pmatrix}
= \begin{pmatrix} 1.6 & 0 \\ 0 & 1.0 \end{pmatrix} = S_2 .
$$

Both outputs and both implied states match the dense computation exactly,
while the dense path wrote the $2 \times 2$ matrix twice and the factored path
wrote it zero times. At the flush the kernel would evaluate that last sum once
and store $S_2$ into $A$'s memory, then reset $\gamma$ to 1 and empty the list.
Notice also that $R_1$ changed from $(1.5, 0)$ to $(1.2, 0)$ between tokens:
that is the superscript in $R_i^{(t)}$ made concrete.

### Why the rewritten form is cheap

Multiply the identity by a vector and every rank-one term collapses to a dot
product:

$$
\alpha_t S_{t-1} k_t = \gamma_t A k_t + \sum_{i=s+1}^{t-1} R_i^{(t)}\,(K_i^\top k_t),
\qquad
\alpha_t S_{t-1} q_t = \gamma_t A q_t + \sum_{i=s+1}^{t-1} R_i^{(t)}\,(K_i^\top q_t). \tag{16}
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
$u_t = \beta_t(v_t - \alpha_t S_{t-1} k_t)$, emits
$y_t = \alpha_t S_{t-1} q_t + u_t (k_t^\top q_t)$, and appends $(u_t, k_t)$
to the list.

Per token this writes one scalar, rescales a handful of short vectors, and
appends two more. It never writes the $V \times K$ matrix. When the list
reaches the window length, one flush computes $\gamma_t A + \sum R_i K_i^\top +
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

**Do not divide by $\gamma$.** The obvious way to keep everything under one
common scale is to store $U_i = u_i / \gamma_i$ so that a single $\gamma_t$
multiplies the anchor and every pair together. It saves the per-token rescale
of the list. It also divides by a product of up to eight decays, and for a
strongly forgetting head that product reaches fp32 zero under ordinary gate
values. The dense kernel is perfectly happy with $\alpha = 0$; the
divide-by-$\gamma$ kernel produces `inf` then `NaN`. Decaying the stored
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

Go back to the unrolled gated recurrence (4), written for a $T$-token prefix:

$$
S_T = \sum_{i=1}^{T} \Big( \prod_{j=i+1}^{T} \alpha_j \Big) v_i k_i^\top . \tag{17}
$$

The weight on a token that arrived $n$ positions ago is roughly $\alpha^n$,
and $\alpha$ is a learned property of the head. Define a head's *retention
horizon* as the number of tokens after which that weight has fallen below
$10^{-3}$:

$$
H = \frac{\log 10^{-3}}{\log \alpha} . \tag{18}
$$

Two heads on the same 8,000-token prefix:

| $n$ tokens ago | weight, $\alpha = 0.95$ | weight, $\alpha = 0.9999$ |
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

1. **Score each head once.** In GDN, $\alpha = \exp(-\exp(A_{\log}) \cdot
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

Two things to be honest about. The horizon formula evaluates $\alpha$ at a
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
is a decayed sum in which recent terms dominate exactly when $\alpha$ is
small.

They also compose without effort. A restored DASC checkpoint is a dense state,
which is a ReplaySSM anchor with an empty list. And the horizon idea points at
a decode-time optimization ReplaySSM does not have: when a head's $\gamma_t$
inside a window has fallen below $10^{-3}$, its anchor contributes nothing to
the output, and the kernel can skip loading it.

## What ReplaySSM does not touch

The write is gone. The read is still there, and it is now the entire cost of
the kernel. That points at the next set of ideas.

- **Quantize the anchor.** It is read every token and written once per
  window, so an int8 anchor with fp32 residuals cuts the dominant remaining
  term by 4$\times$ and injects quantization error once per flush rather than
  per token.
- **Skip the anchor when $\gamma$ is tiny.** For a fast-forgetting head the
  anchor's contribution is numerically zero after a few tokens. The kernel
  already knows $\gamma$; it can skip the load.
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
- Yang et al., *Gated Linear Attention Transformers with Hardware-Efficient
  Training*, ICML 2024, [arXiv:2312.06635](https://arxiv.org/abs/2312.06635).
  The chunkwise form for gated linear attention.
- Yang et al., *Parallelizing Linear Transformers with the Delta Rule over
  Sequence Length*, NeurIPS 2024,
  [arXiv:2406.06484](https://arxiv.org/abs/2406.06484). The WY-based chunkwise
  algorithm for the delta rule.
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
