---
layout: post
math: true
title: 'Tinyserve, Chapter 3: Dense attention and its memory'
date: 2026-10-06 00:00:00 -0700
categories:
- tinyserve
- llm-serving
excerpt: How does a token retrieve information from its history?
book_chapter: 3
source_revision: 14b0c8dbb975350bbd730ac0c702eafce36eb95c
source_document: docs/book/03-dense-attention.md
code_revision: e20a34815498560f9226a9e057ade31b2b62743f
---

{% include tinyserve-article-style.html %}
{% include tinyserve-book-nav.html %}

*Chapter 3 · Model Architecture and Computation*

A token has a vector of features, but those features alone cannot tell it
what an earlier token said. Attention supplies that communication. The
current token creates a query, compares it with keys associated with visible
tokens, and combines their values. During generation, the old keys and values
become persistent memory; the query is usually needed only for the current
forward.

We will calculate one attention row, place it inside a four-token causal
matrix, and follow the tensors through Tinyserve. Then we will change the
number and representation of stored heads. This separates three questions
that are easy to confuse: which tokens are visible, how many learned heads
exist, and how the implementation stores and reads their memory.

## Calculate one query by hand

Consider four token positions, numbered 0 through 3. There are two query
heads, one shared key/value head, and two coordinates per head. Treat the
following vectors as the inputs to the attention dot products: any learned
projection, Q/K normalization, and positional transformation has already
happened. They are a synthetic arithmetic example, not checkpoint values.

| Key position | Key vector k | Value vector v |
|---:|---|---|
| 0 | `[1, 0]` | `[1, 0]` |
| 1 | `[0, 1]` | `[0, 2]` |
| 2 | `[1, 1]` | `[2, 1]` |
| 3 | `[-1, 0]` | `[-1, 1]` |

At position 3, head A's query is `[sqrt(2), 0]`. Dividing its dot products
by the square root of head width, `sqrt(2)`, produces scores
`[1, 0, 1, -1]`. Softmax converts them to weights:

$$
p_j=\frac{\exp(s_j)}{\sum_i\exp(s_i)}.
$$

The four weights are approximately
`[0.399486, 0.146963, 0.399486, 0.054065]`.

The output is the weighted sum of **values**, not keys:

$$
o_A=\sum_jp_jv_j\approx[1.144394,\ 0.747476].
$$

For example, its first coordinate is
`0.399486 × 1 + 0.146963 × 0 + 0.399486 × 2 − 0.054065`.
The scores are similarities in a learned coordinate system. The output can
contain information that does not appear in the query, because the value
vectors carry the information being retrieved.

Head B can ask a different question of the same memory. With query
`[0, sqrt(2)]`, its scores are `[0, 1, 1, 0]`. Its probability row is
approximately `[0.134471, 0.365529, 0.365529, 0.134471]`, producing
`[0.731059, 1.231059]`. Sharing a KV head therefore does not force query
heads to have the same weights or outputs.

The two outputs concatenate into four coordinates. A learned output
projection maps those coordinates back to the residual stream width.
Attention's internal width and the model's residual width need not match.

## Put the row inside a causal matrix

For one head, arrange queries and keys as matrices with one token per row:
`Q [Tq,d]`, `K [Tk,d]`, and `V [Tk,dv]`. Ordinary scaled dot-product
attention is

$$
O=\operatorname{softmax}\left(QK^\mathsf T/\sqrt d+M\right)V.
$$

Softmax operates independently across each query row. An allowed entry of
M is zero; a forbidden entry is negative infinity. Exponentiating negative
infinity gives zero, so forbidden keys receive no probability.

For a fresh four-token prompt, `Tq = Tk = 4`. Query 0 sees key 0; query 1
sees keys 0 and 1; and query 3 sees all four. The ten allowed query/key pairs
form a lower triangle. Both heads follow the same visibility rule even
though their scores differ.

[![Four causal query rows share a key and value history. The final row is calculated numerically, and its two head outputs are joined before projection.](/assets/tinyserve/book03-attention-example.svg)](/assets/tinyserve/book03-attention-example.svg)

If the same head-A query appeared at position 1, masking would leave only
scores `[1, 0]`. Their probabilities would be `[0.731059, 0.268941]` and
the output `[0.731059, 0.537883]`. We must normalize over the permitted
keys. Computing the four-key softmax and zeroing the last two probabilities
afterward would leave a row that sums to less than one and would produce a
different operation.

Causality permits parallel prompt processing. All four query rows can run
together because their masks prevent future information from entering earlier
positions. It also explains caching: adding position 4 cannot change what
positions 0 through 3 were allowed to see. Their previously computed K/V
remain valid for this dense causal model in evaluation mode.

The square-root scale controls the growth of dot-product magnitude as head
width grows. For independent zero-mean, unit-variance coordinates, a sum of
d products has variance d; dividing by `sqrt(d)` gives unit variance under
those assumptions. Actual learned Q/K distributions need not satisfy them.
The scale is nevertheless part of the trained attention equation, not an
optional numerical tuning parameter for the serving implementation.

## Position changes the comparison

The causal mask says which positions may communicate. It does not by itself
give the dot product a numerical representation of their distance. Qwen3
uses rotary position embeddings, or RoPE, to rotate pairs of query and key
coordinates according to position. Values are not rotated.

For one coordinate pair and angle theta, the operation is

$$
R(\theta)\begin{bmatrix}x_0\\x_1\end{bmatrix}
=\begin{bmatrix}
x_0\cos\theta-x_1\sin\theta\\
x_0\sin\theta+x_1\cos\theta
\end{bmatrix}.
$$

Take unrotated query `[1,0]` at position 2 and key `[1,0]` at position 1.
For an illustrative frequency of `pi/2` radians per position, the rotated
query is `[-1,0]` and key `[0,1]`; their dot product is zero. If both were
at position 1, both would rotate to `[0,1]` and their dot product would be
one. More generally,

$$
(R(p\omega)q)^\mathsf T(R(s\omega)k)
=q^\mathsf TR((s-p)\omega)k.
$$

The relative displacement appears because the two absolute rotations
combine. Each pair uses its own frequency. This positional construction is
described in the [RoFormer paper](https://arxiv.org/abs/2104.09864).

The coordinate pairing must agree with the checkpoint. Tinyserve's Qwen3
implementation uses `rotate_half`: at head width eight, the pairs are
`(0,4)`, `(1,5)`, `(2,6)`, and `(3,7)`. They are not adjacent pairs.
`apply_rope()` computes `x*cos + rotate_half(x)*sin` using the token's
logical position, with head and batch dimensions broadcast as necessary.

Qwen3 first applies learned per-head RMSNorm to Q and K, then applies RoPE.
Although a rotation preserves Euclidean norm, learned coordinate gains do
not generally commute with it. Reversing these operations can change the
model. The rotated key can be cached permanently at its own position; a
later query uses a new rotation without rotating old keys again. A physical
cache relocation does not change the key's logical position.

## Follow the shapes through Tinyserve

Let B be request batch size, T the number of newly processed tokens, H the
residual width, Hq the number of query heads, Hkv the KV-head count, and d
the head width. For ordinary Qwen3, key and value widths are both d.

| Operation | Result shape |
|---|---|
| Receive normalized hidden states | `[B,T,H]` |
| `q_proj`, then split heads | `[B,T,Hq,d]` |
| `k_proj` and `v_proj`, each | `[B,T,Hkv,d]` |
| Normalize Q/K and rotate Q/K | Shapes unchanged |
| Append K/V to an existing prefix of P tokens | Each `[B,P+T,Hkv,d]` |
| Attend with the new queries | `[B,T,Hq,d]` |
| Join query-head outputs | `[B,T,Hq*d]` |
| `o_proj` | `[B,T,H]` |

In `Qwen3Attention.forward()`, the projection results are reshaped, Q/K
are normalized, and RoPE is applied. A contiguous cache's `update()` returns
the visible K/V history. The tensors transpose to `[B,heads,tokens,d]` for
PyTorch scaled dot-product attention. The result transposes back, joins
heads, and passes through `o_proj`.

During fresh prefill, every prompt token is a query. During single-token
decode, `T = 1`, but keys cover `P+1` tokens, including the new token's key.
That query may attend to the entire valid history. A top-left causal
triangle over a `1 × (P+1)` matrix would incorrectly allow only key 0.
Tinyserve uses no causal triangle for this maskless single-token case.

For a multi-token append, query row i has logical position `P+i` and may
see keys `j <= P+i`. The caller supplies this shifted causal mask. With
left padding or unequal valid lengths, explicit masks also identify real
tokens. These are position and visibility contracts, independent of whether
memory is contiguous or paged. [Chapter 4](/tinyserve/llm-serving/2026/10/06/tinyserve-04-efficient-attention.html)
develops the readers; [Chapter 8](/tinyserve/llm-serving/2026/10/06/tinyserve-08-kv-cache-and-batching.html) develops the
generation loop that owns the history.

## Share heads to reduce persistent memory

Multi-head attention, multi-query attention, and grouped-query attention
change how query heads share keys and values:

| Architecture | KV heads for Hq query heads | Memory relationship |
|---|---:|---|
| MHA | Hq | Each query head has its own K/V projections. |
| MQA | 1 | All query heads share one K/V head. |
| GQA | An intermediate divisor of Hq | Each group of query heads shares one K/V head. |

Our two-query-head, one-KV-head example is the MQA endpoint of grouped-query
attention. An eight-query-head GQA model with two KV heads might map query
heads 0–3 to KV head 0 and heads 4–7 to KV head 1. The mapping is architectural,
not chosen separately for each token. The [GQA paper](https://arxiv.org/abs/2305.13245)
studies this intermediate representation and training conversions.

For L layers, B requests of equal cached length T, and s bytes per stored
element, ordinary KV storage is

$$
M_{KV}=2LBTH_{kv}d\,s.
$$

The factor two is K plus V. This counts valid tensor payload, excluding
allocator slack, page metadata, alignment, scales, and temporary expanded
heads. For the Qwen3-0.6B dimensions used earlier, `L=28`, `Hkv=8`, `d=128`,
and BF16 storage gives `114,688` bytes per cached token across the model,
or 112 KiB. A single 4,096-token history occupies 448 MiB of KV payload.
An otherwise equal hypothetical MHA layout with 16 KV heads would require
896 MiB; this arithmetic does not imply that the trained GQA checkpoint can
be converted to MHA or back without changing its weights and behavior.

[![Eight query heads use eight MHA KV heads, two GQA heads, or one MQA head; MLA instead stores a shared latent and a separate positional component.](/assets/tinyserve/book03-memory-representations.svg)](/assets/tinyserve/book03-memory-representations.svg)

Sharing reduces stored K/V and can reduce decode traffic if the kernel reuses
a KV load across its query group. A reference implementation may explicitly
repeat KV heads to match Hq, spending temporary memory and traffic. Thus
the architecture creates an opportunity; the execution path determines how
much of it is realized.

## Store a latent instead of expanded keys and values

Multi-head latent attention, MLA, uses a learned compressed representation
for the ordinary content part of keys and values. Let the cached latent for
token j be `c_j [r]`, with head-specific matrices `W_K [d_k,r]` and
`W_V [d_v,r]`. Conceptually,

$$
k_j=W_Kc_j,\qquad v_j=W_Vc_j.
$$

The content score can be rearranged as
`q^T W_K c_j = (W_K^T q)^T c_j`. After softmax, the value sum can be
rearranged as `W_V sum_j(p_j c_j)`. These identities move expansion outside
the history dimension. The cache stores r latent values per token instead
of expanded content K/V for every head. MLA also separates a positional
component whose treatment must match the model. The [DeepSeek-V2
report](https://arxiv.org/abs/2405.04434) introduces this architecture.

Tinyserve's `KimiMLAAttention.forward()` makes the rearrangement concrete.
`kv_a_proj_with_mqa` creates a latent and a separate component;
`kv_a_layernorm` normalizes the latent before `KimiCache.update_mla()` stores
it. `kv_b_proj.weight` is viewed as head-specific key and value weights.
`absorbed_query` multiplies the content query by the key weight, scores
compare it with cached latents, and a second score term uses the separate
component. Attention combines latents before applying the value weight.

The tiny Kimi fixture stores 512 latent coordinates and 64 separate
coordinates per token per MLA layer. In BF16 that is `576 × 2 = 1,152`
bytes; five such layers require 5,760 bytes per token. The checkpoint's
reference calls the separate component rotary-related but does not actually
rotate it. Tinyserve preserves that executable equation. Applying Qwen3's
RoPE here because of a variable name would break checkpoint parity.

MLA's compressed history still grows with context. It also changes the
dimensions and projection work of attention; its cache compression ratio
alone is not a latency prediction. The surrounding Kimi model contains
recurrent layers too, which [Chapter 6](/tinyserve/llm-serving/2026/10/06/tinyserve-06-linear-attention.html) treats
separately.

## Count comparisons as well as bytes

Ignoring projection costs and counting one multiply-add as two FLOPs, dense
attention over N permitted query/key pairs costs approximately
`2 Hq N (d + dv)` FLOPs for scores and value accumulation. When `dv=d`,
that becomes `4 Hq N d`. Fresh causal prefill has
`N = T(T+1)/2`; one decode query over T keys has `N=T`.
Softmax reductions and exponentials add work beyond that count.

MHA, GQA, and MQA retain Hq query outputs, so KV sharing does not eliminate
the comparisons for those query heads. It primarily changes projections,
storage, and potential data reuse. A materialized prefill score matrix would
have `B Hq T^2` entries, even though roughly half are masked. That quadratic
temporary is avoidable without changing dense attention's mathematical
answer. The next chapter shows how online softmax makes that possible.

## Source map

| Source | Responsibility |
|---|---|
| [models/qwen3.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/models/qwen3.py) | Q/K/V projections, head normalization, RoPE, and attention shapes. |
| [kv_cache.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/kv_cache.py) | Contiguous KV append and visible history. |
| [models/kimi.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/models/kimi.py) | Absorbed MLA content projections and latent attention. |
| [kimi_cache.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/kimi_cache.py) | Per-request latent and separate-component histories. |
| [model reference](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/design/model.md) | Architecture dimensions and supported model contracts. |

The implementation references describe runtime `e20a348`. Repository access
is required for those source links; all arithmetic and tensor definitions
needed for this chapter appear above.

{% include tinyserve-book-nav.html %}
