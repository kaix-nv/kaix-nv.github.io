---
layout: post
math: true
title: 'Tinyserve, Chapter 7: Feed-forward networks and mixture of experts'
date: 2026-10-06 00:00:00 -0700
categories:
- tinyserve
- llm-serving
excerpt: How do tokens choose expert computations and recover their combined outputs?
book_chapter: 7
source_revision: 14b0c8dbb975350bbd730ac0c702eafce36eb95c
source_document: docs/book/07-feed-forward-and-moe.md
code_revision: e20a34815498560f9226a9e057ade31b2b62743f
---

{% include tinyserve-article-style.html %}
{% include tinyserve-book-nav.html %}

*Chapter 7 · Model Architecture and Computation*

Attention mixes information across token positions. The feed-forward
sublayer transforms the features at each position independently. In a
dense model, every token uses the same learned matrices. A mixture of
experts, or MoE, gives the model many alternative feed-forward networks
and routes each token to a small subset.

Choosing fewer experts controls active arithmetic, but it does not by
itself organize efficient GPU work. A token may produce several expert
assignments, many tokens may choose the same expert, and their results
must return to the original token order. We will follow four tokens
through that routing and then connect the example to Tinyserve's packed
weights and native indexed expert kernels.

## Start with a dense feed-forward layer

A conventional two-projection feed-forward network expands a hidden
vector x from width H to intermediate width I, applies a nonlinearity,
then projects back:

$$
f(x)=W_{\mathrm{down}}\,\phi(W_{\mathrm{up}}x).
$$

The output has width H so it can be added to the residual stream.
For a batch of N token rows, the same function acts on every row; there
is no mixing between token positions inside these matrix products.

Qwen3 uses a gated version, SwiGLU:

$$
f(x)=W_{\mathrm{down}}
\left[\operatorname{SiLU}(W_{\mathrm{gate}}x)
\odot(W_{\mathrm{up}}x)\right].
$$

The activation is

$$
\operatorname{SiLU}(z)=z\,\sigma(z).
$$

Both branches produce I coordinates, their elementwise product produces
I coordinates, and the down projection returns H. “Gate” here describes
continuous feature modulation; it is not the expert router introduced
below. A feature with a small gate contribution still participates in
the dense matrix products.

For a numerical example, let `x=[1,2]`, gate and up projections both be
the identity, and the down projection be `[1,-1]`. The intermediate is
`[SiLU(1)×1, SiLU(2)×2] ≈ [0.731059,3.523188]`, and the output is
`−2.792130`. This deliberately small one-output example shows the
branching arithmetic; an actual residual block's down projection returns
the full input width.

With no biases, a dense SwiGLU layer has `3HI` weights and approximately
`6NHI` projection FLOPs, counting each multiply-add as two. The nonlinear
activation and elementwise multiplication add work. During prefill, many
token rows can reuse each weight tile; during small-batch decode, loading
weights can dominate the short matrix multiplication.

## Let the router choose functions

An MoE layer replaces one such function with a collection
`f_0,...,f_(E-1)` and a router. Given hidden row x, the router produces
E scores, chooses K experts, and supplies a weight for each selected
route. A simple routed result is

$$
y(x)=\sum_{r=0}^{K-1}w_r(x)f_{e_r(x)}(x).
$$

The expert ID e determines which learned matrices run. Route slot r
identifies one of this token's chosen experts. The route weight w
controls its contribution after evaluation. These three concepts must
remain separate in the implementation.

Routers differ across checkpoints. Tinyserve's Qwen3-MoE applies softmax
over expert logits, chooses top-k probabilities, and optionally
renormalizes the selected weights. Its Kimi router applies sigmoid to
FP32 expert scores, adds a learned correction bias **for selection**,
and gathers combination weights from the original sigmoid scores. It
can renormalize those selected weights and multiplies them by the
configured routed scaling factor. Adding the correction bias to the
final combination weights would change Kimi's equation.

Sparse expert selection is independent of sparse attention. Every token
can still attend to its entire causal history while evaluating only
K of E feed-forward experts. Tinyserve's `is_sparse_layer()` refers
to this choice of feed-forward architecture.

## Route four tokens and retain every identity

Use four token rows and select two experts per token from a toy expert
pool. Suppose the retained, normalized weights are:

| Token row | Slot 0 expert and weight | Slot 1 expert and weight |
|---:|---|---|
| 0 | Expert 2, 0.6 | Expert 5, 0.4 |
| 1 | Expert 2, 0.8 | Expert 7, 0.2 |
| 2 | Expert 5, 0.3 | Expert 7, 0.7 |
| 3 | Expert 2, 0.5 | Expert 5, 0.5 |

For example, token 0 could have sigmoid scores 0.6, 0.4, and 0.1 for
experts 2, 5, and 7, zero correction bias, and scaling factor one.
Top-two selection retains experts 2 and 5; their sum is one, so
renormalization leaves `[0.6,0.4]`. The particular slot ordering is an
illustration; `topk(sorted=False)` does not promise ascending expert IDs.

There are eight assignments, not four. Flatten the `[N,K]` route table
in token order and define `a = token*K + slot`:

```text
assignment a     0  1  2  3  4  5  6  7
token            0  0  1  1  2  2  3  3
slot             0  1  0  1  0  1  0  1
expert           2  5  2  7  5  7  2  5
```

Group the assignments by expert:

```text
expert 2: assignments [0,2,6] -> input tokens [0,1,3]
expert 5: assignments [1,4,7] -> input tokens [0,2,3]
expert 7: assignments [3,5]   -> input tokens [1,2]
```

The same token appears in multiple groups because it deliberately uses
multiple experts. Within one expert, the grouped input rows form a
matrix. That matrix can reuse a single expert's weight tiles.

[![Four tokens create eight weighted expert assignments. Grouping by expert keeps original assignment IDs, and scattering those IDs restores token and route-slot order before weighted combination.](/assets/tinyserve/book07-routing-example.svg)](/assets/tinyserve/book07-routing-example.svg)

Suppose expert outputs for token 0 are `[2,0]` and `[0,4]`. After returning
them to slots 0 and 1, the routed sum is
`0.6×[2,0]+0.4×[0,4]=[1.2,1.6]`. For all four tokens, illustrative
returned values could give:

| Token | Slot 0 output | Slot 1 output | Weighted routed result |
|---:|---|---|---|
| 0 | `[2,0]` | `[0,4]` | `[1.2,1.6]` |
| 1 | `[1,1]` | `[3,-1]` | `[1.4,0.6]` |
| 2 | `[-1,2]` | `[2,0]` | `[1.1,0.6]` |
| 3 | `[4,1]` | `[0,3]` | `[2,2]` |

Those are routed-branch outputs before any architecture-specific output
normalization, projection, shared expert, or residual addition. The
key invariant is that assignment 6 always returns to token 3, slot 0,
even if it ran beside token 0 on expert 2.

## Kimi uses latent experts and a shared branch

The tiny Kimi fixture has 64 routed experts and selects 16 per token.
Its router reads the original H-wide hidden state. A separate down
projection maps that state into a 256-coordinate routed latent space,
and each selected expert transforms this latent with intermediate width
256. After weighted combination, routed RMSNorm and an up projection
return it to H. Shared experts process the original hidden state and
are added to the routed result.

Thus Kimi's full block is more than the simple sum above. Its two shared
experts are represented as one wider shared MLP with the corresponding
combined intermediate width. They run for every token and are outside
top-k selection. Layer 0 uses a dense MLP; the fixture's remaining
16 layers use the routed structure.

The experts use Kimi's SiTU activation rather than Qwen's SwiGLU. In
Tinyserve's notation, with a gate branch g and up branch u,

$$
\operatorname{SiTU}(g,u)=
\left[b\tanh(g/b)\sigma(g)\right]\odot\tilde u,
$$

where `u_tilde = c tanh(u/c)` if a linear-branch bound c is configured,
otherwise `u_tilde=u`. The activation is evaluated in FP32 before
returning to the input dtype. The native expert path preserves the
BF16 projection-rounding boundary before that activation. A mathematically
similar fusion that moves rounding can change final logits.

## Pack weights once and choose the execution path

Checkpoint names identify individual experts, such as
`experts.5.w1.weight`. Those names are useful for strict loading, but
execution benefits from an explicit expert dimension:

```text
w1, w3: [E,I,D]    gate and up projections
w2:     [E,D,I]    down projection
```

`pack_experts()` stacks the separately loaded expert weights on CPU,
then deletes the original expert modules before transferring the model
to the GPU. Here “packed” means expert-major tensor layout. The
checkpoint's MXFP4 bit packing is a separate representation: the teaching
loader expands those weights to BF16 before this execution packing.
This path does not imply native FP4 expert arithmetic.

For a small number of token rows, `w1[indices]` materializes
`[N,K,I,D]` selected weights. Three batched contractions evaluate every
assignment. The readable implementation is:

```python
gate = einsum("nd,nkid->nki", x, w1[indices])
up = einsum("nd,nkid->nki", x, w3[indices])
act = situ(gate, up)
selected = einsum("nki,nkdi->nkd", act, w2[indices])
routed = (selected * weights[..., None]).sum(dim=1)
```

This removes many small expert calls, but repeated routes also duplicate
selected weight matrices in temporary storage. One BF16 projection
family consumes `N K I D × 2` bytes. For `K=16`, `D=I=256`, that is
2 MiB per token row: 32 rows reach the 64 MiB guard, while 128 rows
would require 256 MiB. The three families are gathered sequentially;
the guard estimates one family rather than summing three simultaneous
copies or measuring total peak allocation.

For larger inputs, the memory-bounded PyTorch oracle groups assignments
by active expert and invokes each expert once. It stores activations
and results rather than one weight copy per assignment. Its disadvantage
is the Python loop and many small operations around each expert.

## Build an indexed expert schedule on the GPU

The native long-input path preserves grouped reuse while removing the
per-expert Python loop. `fused_indexed_moe()` flattens expert IDs, sorts
their assignment IDs by expert, counts assignments per expert, and
constructs cumulative offsets. Each sorted assignment still carries
its original `a`; `a//K` recovers the source token and `a%K` its route
slot. Equal-expert ordering need not be stable because those identities
travel with every row.

With row tile size 16, an expert receiving n assignments needs
`ceil(n/16)` tiles. Another prefix sum of these tile counts maps a
kernel task to its expert and row tile. The actual tile total remains
on the GPU. The host launches a safe upper bound,

$$
\sum_e\lceil n_e/16\rceil\le\lceil NK/16\rceil+E,
$$

and surplus programs mask themselves out. This avoids copying the
task count to Python merely to construct the launch grid.

The first expert kernel gathers each group's input rows, computes gate
and up projections using the expert's weights, applies SiTU, and writes
sorted activations `[NK,I]`. The second computes the down projection
and scatters directly to original assignment positions, yielding
`[N,K,D]`. Existing model code applies route weights, sums K, normalizes,
projects up, and adds the shared expert.

[![Expert-major storage supports either selected-weight temporary gathers for short inputs or a GPU-sorted indexed grouped path for long inputs. Both return the same token-by-route result shape before common weighted reduction.](/assets/tinyserve/book07-execution-paths.svg)](/assets/tinyserve/book07-execution-paths.svg)

“Two kernels” describes these two expert-computation launches, not the
entire MoE block. Sorting, counting, offsets, routing, and the common
combination still execute operations. The improvement comes from
removing the multiplication of small projection calls by active expert
count while preserving enough rows per tile to reuse weights.

The dispatch contract is narrow: more than 32 token rows, CUDA inference,
contiguous BF16 inputs and packed weights, INT64 contiguous expert IDs,
`E=64`, `K=16`, `D=I=256`, and supported positive SiTU bounds.
`auto` uses the native path only when the long-input guard and this
contract agree; unsupported long shapes use the grouped oracle.
Even explicitly requesting `triton` retains the selected short path,
and requires native support only after the 64 MiB guard selects the
long path.

## Active parameters are not total storage

For three projections of shape D by I, all E experts contain `3EDI`
parameters while one token activates `3KDI`. The corresponding expert
projection work is approximately `6NKDI` FLOPs. Router, latent
projections, shared experts, activation, and combination add costs.

With the fixture's dimensions, one expert contains 196,608 parameters,
or 384 KiB in BF16. Sixty-four experts occupy 24 MiB per routed layer;
the 16 selected experts for one token contain 6 MiB of weight payload.
The entire expert pool still needs storage somewhere. Sparsity reduces
per-token activation of experts, not the number of trained expert
weights that exist.

Across a batch, the union of chosen experts can approach all E. That
can increase total weight traffic while also increasing reuse within
each expert. A balanced route distribution may supply many useful
matrix rows. A highly skewed distribution leaves a few large groups
and many tiny or empty ones. On one GPU, uneven groups affect tile
utilization and scheduling; across devices, they additionally create
owner imbalance and communication, the subject of Chapter 17.

This makes “active parameters” an incomplete latency predictor. Two
models with equal active expert arithmetic can differ in router cost,
expert width, route concentration, temporary materialization, launch
count, and weight reuse. Count the work, identify the actual selected
path, and measure the whole boundary that matters.

## The measured result includes a numerical exception

The retained one-block profile found 1,215 CUDA kernel events for the
128-row grouped reference, including routing support work and three
projections per active expert. Native indexed execution reduced that
profile to 66 events, with one up/activation and one down expert launch.
The sorting work did not vanish; the per-expert launch chain did.

In the paired full-forward comparison, one loaded BF16 tiny Kimi model
on one RTX A6000 used two warmups and five alternating repetitions:

| Prompt rows | Grouped or selected reference | Auto execution | Full-forward reduction |
|---:|---:|---:|---:|
| 32 | 38.23 ms | 38.11 ms | 0.3%; both use selected weights |
| 128 | 102.09 ms | 38.87 ms | 61.9% |
| 512 | 107.54 ms | 41.89 ms | 61.0% |
| 2,048 | 158.62 ms | 95.68 ms | 39.7% |

The long rows switch from grouped Python execution to native indexed
experts. Measured peak allocation was unchanged within each pair;
temporary activation/result storage scaled with `[NK,I]` and `[NK,D]`,
not `[N,K,I,D]` selected weights. These are model-forward timings,
not request throughput or a cross-engine ranking.

Maximum model logit difference was `0.001953125`, below the retained
`0.004` bound, and ordered top-three lists matched. The 128-token
fixture nevertheless had a greedy-token exception: the grouped result
gave two candidates exactly the same BF16 maximum, and a one-ULP-scale
perturbation changed which candidate won. The record explicitly retains
that exception rather than claiming exact token parity. A structural
random checkpoint also does not establish production language quality.
The [committed indexed-MoE receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m7s-native-indexed-moe-a6000-2026-09-10.json)
keeps both the timings and this numerical boundary.

## Source map

| Source | Responsibility |
|---|---|
| [models/qwen3.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/models/qwen3.py) | Dense SwiGLU feed-forward network. |
| [models/qwen3_moe.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/models/qwen3_moe.py) | Softmax routing and single-device expert oracle. |
| [models/kimi.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/models/kimi.py) | Sigmoid routing, correction bias, latent/shared branches, packing, and dispatch. |
| [kimi_kernels.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/kimi_kernels.py) | GPU route schedule and indexed expert kernels. |
| [loader.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/loader.py) | Checkpoint expansion and CPU packing before GPU transfer. |

The implementation references describe runtime `e20a348` and require
repository access. The four-token routing and restoration example above
contains the complete index contract without those links.

{% include tinyserve-book-nav.html %}
