---
layout: post
math: true
title: 'Tinyserve, Chapter 1: From weights to the first token'
date: 2026-10-06 00:00:00 -0700
categories:
- tinyserve
- llm-serving
excerpt: How does a checkpoint become a generated token?
book_chapter: 1
source_revision: 14b0c8dbb975350bbd730ac0c702eafce36eb95c
source_document: docs/book/01-first-token.md
code_revision: e20a34815498560f9226a9e057ade31b2b62743f
---

{% include tinyserve-article-style.html %}
{% include tinyserve-book-nav.html %}

*Chapter 1 · Foundations and Measurement*

A language model does not return a finished answer in one operation. It
turns a token history into scores for the next token. A generation loop
chooses a token, appends it to that history, and calls the model again.
Our first serving system needs only a correctly loaded model and this loop.
It will recompute more than necessary, but every later optimization needs
an understandable baseline to compare against.

We will follow a three-token prompt through a small dense decoder, produce
three output tokens, and count the work. The implementation is Tinyserve's
single-device Qwen3 path, using ordinary floating-point weights and greedy
selection. The goal is to understand the complete path before introducing
KV caches, request scheduling, specialized kernels, or distributed execution.

## What comes with a checkpoint

Three different kinds of information are needed to turn text into a model
response. None can substitute for the others.

| Component | What it supplies | Example |
|---|---|---|
| Model configuration | The architecture and tensor dimensions | Layer count, hidden width, attention heads, position settings, and whether input/output weights are tied. |
| Tokenizer and chat template | The agreement between text and token IDs | Vocabulary, tokenization rules, special tokens, and conversation formatting. |
| Weight tensors | The values learned during training | Embedding vectors, projection matrices, and normalization gains. |

`resolve_model_path` accepts a local directory or resolves a model repository
into a local snapshot. `ModelConfig` selects the supported architecture;
`AutoTokenizer` handles its text representation; `load_model` constructs the
model and loads its tensors. Tinyserve implements the model forward itself.
Using a tokenizer library does not mean delegating that forward to the
library's model implementation.

[![Configuration builds the architecture, weights populate it, and the tokenizer connects text to token IDs. A forward produces scores; the generation loop chooses and feeds back a token.](/assets/tinyserve/book01-checkpoint-to-token.svg)](/assets/tinyserve/book01-checkpoint-to-token.svg)

Before tokenization, a chat request may be wrapped in a template containing
role markers and an assistant-generation prefix. Tinyserve's `generate`
method does this when `chat=True`; `chat=False` encodes the supplied text
directly. Two engines given the same visible sentence can therefore receive
different token histories if their templates or special-token settings differ.

The model sees integer IDs, not strings and not necessarily whole words.
For example, let our already-tokenized prompt be `[7, 12, 5]`. These are
illustrative IDs in a synthetic 16-token vocabulary; they are not a claim
about how a real Qwen tokenizer encodes a particular sentence. We will call
the generated tokens `y1`, `y2`, and `y3`, without assigning them English
meanings.

## A small example with the real model structure

The worked example uses the actual `Qwen3ForCausalLM` class with small,
untrained weights. Its dimensions are deliberately small enough to follow:

| Dimension | Worked example | Qwen3-0.6B configuration |
|---|---:|---:|
| Decoder layers | 2 | 28 |
| Residual hidden width, H | 8 | 1024 |
| Query heads / KV heads | 2 / 1 | 16 / 8 |
| Width of each attention head, d | 8 | 128 |
| Feed-forward intermediate width, I | 12 | 3072 |
| Vocabulary size, V | 16 | 151,936 |

The small model demonstrates tensor shapes and control flow, not language
quality. The larger configuration tells us which dimensions come from the
checkpoint rather than from assumptions in the implementation.

Our input IDs have shape `[1, 3]`: one request and three token positions.
The position IDs are `[0, 1, 2]`. Looking up the token IDs `[7, 12, 5]` in an embedding matrix
of shape `[V, H]` produces a hidden tensor `[1, 3, 8]`. Each token now has
eight features. Those features are learned coordinates, not eight
human-assigned properties such as noun, color, or sentiment.

The hidden tensor then passes through two decoder blocks. Each block returns
the same shape, `[1, 3, 8]`, but with updated values. A final normalization
and a vocabulary projection produce logits of shape `[1, 3, 16]`.

The three logit rows have different meanings:

| Logit row | History visible at that position | What its scores predict |
|---|---|---|
| 0 | `[7]` | The token after ID 7. |
| 1 | `[7, 12]` | The token after that two-token history. |
| 2 | `[7, 12, 5]` | The first new token after the complete prompt. |

To extend this prompt, we select **row 2**, not row 0 and not all three rows.
The prompt already supplies its own three tokens. We do not replace them
with predictions from the earlier rows.

## Inside one decoder block

A decoder block first exchanges information between token positions through
attention, then transforms each position's features through a feed-forward
network. Both operations add their result back to a residual stream:

```python
x = x + self.self_attn(self.input_layernorm(x), cos, sin)
x = x + self.mlp(self.post_attention_layernorm(x))
```

This is the uncached form of `Qwen3DecoderLayer.forward`. The additions are
why both sublayers must return the residual width H, even when their
internal tensors are wider.

[![One decoder block keeps a residual stream while applying normalized causal attention and a normalized gated feed-forward network. Attention mixes positions; the feed-forward network transforms each position independently.](/assets/tinyserve/book01-decoder-block.svg)](/assets/tinyserve/book01-decoder-block.svg)

### Normalize the features

RMSNorm rescales a token's feature vector using its root-mean-square
magnitude, then multiplies by a learned gain for each feature. With input
coordinates $x_i$ and gains $g_i$,

$$
z_i = g_i\,\frac{x_i}{\sqrt{\frac{1}{H}\sum_{j=1}^{H}x_j^2+\epsilon}}.
$$

The small positive epsilon prevents division by zero. Unlike mean-centered
LayerNorm, this operation does not subtract the feature mean. It does not
mix different token positions. Tinyserve's reference implementation computes
the normalization in FP32 and casts the result back to the input dtype.
This matters when the surrounding model uses BF16: the reduction is not
simply performed in the lower-precision storage format.

The two block-level normalizations operate across H features. Qwen3 also
has a separate normalization inside attention, across each Q or K head's
d features. They act on different tensors and are not interchangeable.

### Retrieve information with causal attention

Three learned linear projections create queries, keys, and values:

- A query describes what a token is looking for.
- A key describes how a token can be matched.
- A value supplies the information to combine when that token is attended to.

These descriptions are intuition for learned vectors, not explicit semantic
labels. For each query head, the attention output O is

$$
O = \operatorname{softmax}\!\left(\frac{QK^\mathsf{T}}{\sqrt d}+M\right)V.
$$

Here V denotes the **value tensor**, not the vocabulary-size symbol used in
the shape tables. The softmax converts the permitted scores in each query
row into weights that sum to one. M is zero for allowed keys and negative
infinity for forbidden keys, excluding their scores before softmax.
For our three-token prompt, the allowed positions are

```text
              key position
                 0  1  2
query 0          Y  -  -
query 1          Y  Y  -
query 2          Y  Y  Y

Y = may attend; - = future position, excluded
```

All three query rows can be computed in the same forward. “Causal” does
not mean computing prompt tokens one at a time; it means that row 0 cannot
use row 1 or 2, and row 1 cannot use row 2. This independence from future
tokens will later make KV caching possible.

The tiny configuration has two query heads but one key/value head. This is
grouped-query attention: both query heads use that shared KV head. The
relevant tensors before the attention calculation are

| Tensor | Shape in the worked example |
|---|---|
| Residual input | `[1, 3, 8]` |
| Projected Q, split into heads | `[1, 3, 2, 8]` |
| Projected K and V, each | `[1, 3, 1, 8]` |
| Attention output, heads joined | `[1, 3, 16]` |
| Output projection back to H | `[1, 3, 8]` |

Do not derive the head width by dividing H by the number of query heads.
In this example, two heads of width eight require a Q projection from 8 to
16 features. In Qwen3-0.6B, the corresponding projection is 1024 to 2048:
16 heads times 128 features. The output projection returns that wider
attention result to the residual width. The configuration specifies both
widths explicitly.

For the uncached path without an explicit mask tensor, Tinyserve calls
PyTorch scaled dot-product attention with `is_causal=True` for multi-token
inputs and `enable_gqa=True`. The model need not materialize the score
matrix shown in the equation; that equation specifies the operation, not
the backend's intermediate storage.

### Attach positions to queries and keys

The content vectors alone do not encode a token's numerical position. The
causal mask controls visibility; it does not directly supply the distance
between a query and an allowed key.

Qwen3 uses rotary position embeddings, or RoPE. After projecting Q and K,
the implementation normalizes them per head, then rotates pairs of their
coordinates by angles determined by position. It does not rotate values.
The same content vector at position 0 and position 2 therefore reaches the
attention dot product in different orientations. Comparing rotated Q and K
introduces a dependence on their relative positions.

The exact coordinate pairing is part of checkpoint compatibility.
Tinyserve follows the half-split convention: for a head of width eight,
the pairs are `(0,4)`, `(1,5)`, `(2,6)`, and `(3,7)`. An adjacent-pair
implementation is not a drop-in replacement for these trained weights.
Likewise, preserve the normalize-then-rotate order. A rotation preserves
vector magnitude, but the learned per-coordinate normalization gains do
not generally commute with rotation. Dense attention and position encoding
receive a fuller treatment in Chapter 3; here the essential contract is
to use the checkpoint's dimensions, positions, and operation order.

### Transform each token with a gated feed-forward network

The feed-forward sublayer does not attend to other positions. It applies
the same learned transformation to each position independently:

```python
return self.down_proj(
    F.silu(self.gate_proj(x)) * self.up_proj(x)
)
```

In the tiny example, `gate_proj` and `up_proj` each map 8 features to 12.
SiLU supplies a nonlinear gate; elementwise multiplication combines the
two branches; `down_proj` maps 12 features back to 8. This gated structure
is commonly called SwiGLU. The result is added to the residual stream, and
the next decoder block receives the updated `[1, 3, 8]` tensor.

Attention mixes information across permitted positions. The feed-forward
network transforms the resulting features within each position. The model
needs both; attention is not the whole transformer.

## Load learned values into the structure

Creating the module tree establishes shapes and operations. Loading assigns
the checkpoint's values to that structure. For this ordinary dense path,
Tinyserve's module names match names such as
`model.layers.0.self_attn.q_proj.weight` in the checkpoint.

The loader builds the model on PyTorch's `meta` device: parameters have
shape metadata but no backing weight storage yet. It reads the required
safetensors entries, converts floating weights to the requested dtype,
and calls `load_state_dict(..., strict=True, assign=True)`. Assigning avoids
first allocating ordinary initialized parameters just to overwrite them.
Strict loading checks that the required model keys and shapes are supplied;
the loader itself filters checkpoint tensors to those the selected model
needs.

Some checkpoints tie the input embedding and output projection weights.
With an embedding matrix E of shape `[V, H]`, the output logits can be
computed by multiplying the final hidden vectors by its transpose. A tied
checkpoint may omit a duplicate `lm_head.weight` entry. Tinyserve supplies
the embedding tensor under that key before strict loading.

There are two distinct meanings of “shared” here: both operations must use
the same learned values, while avoiding duplicate storage is an allocation
property. Aliasing the loader's input tensors is not by itself a guarantee
that a later device or dtype conversion keeps one physical allocation.
Do not infer a memory saving just from the checkpoint's tying flag.

RoPE tables are derived buffers rather than learned checkpoint weights.
Because the model was constructed on `meta`, the loader recreates these
tables with real storage before transferring the model to the chosen device
and putting it in evaluation mode. The architecture, learned tensors, and
derived buffers must all be ready before the first forward.

## Choose one token and feed it back

A logit is a score, not a probability and not a token ID. For greedy
generation, choosing the largest score is sufficient; applying softmax
first would not change its argmax. Stochastic sampling is a different
selection policy, introduced later with request-local randomness.

Here is a simplified version of the single-request, uncached greedy path
for `max_new_tokens > 0`. `ids` starts as a `[1, P]` tensor, where P is the
prompt length, and `model` is already loaded. Run it under
`torch.inference_mode()` so inference does
not record a gradient graph:

```python
max_new_tokens = 3
eos_id = model.config.eos_token_id
positions = torch.arange(ids.shape[1], device=ids.device)
logits = model(ids, positions)[0, -1]

for step in range(max_new_tokens):
    next_id = logits.argmax()
    ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
    if next_id.item() == eos_id or step == max_new_tokens - 1:
        break
    positions = torch.arange(ids.shape[1], device=ids.device)
    logits = model(ids, positions)[0, -1]
```

The first forward processes the prompt and supplies the scores for `y1`.
That prompt processing is **prefill**. Choosing `y1` does not itself run
the model on `y1`. To obtain `y2`, the next forward must include `y1`.
In this uncached baseline, it also recomputes every earlier position.
The repeated generation phase is **decode**, even though this deliberately
simple implementation reprocesses the whole history on each decode step.

[![Three no-cache forwards process histories of lengths three, four, and five. They sample y1, y2, and y3 respectively; the last sampled token is not fed back after reaching the output limit.](/assets/tinyserve/book01-naive-generation.svg)](/assets/tinyserve/book01-naive-generation.svg)

The prompt IDs keep positions 0, 1, and 2. On the second forward, `y1`
occupies position 3; on the third, `y2` occupies position 4. Rebuilding
`arange(length)` for a whole-history forward preserves those logical
positions. It does not assign position 0 to every newly generated token.

We stop after sampling `y3`, so there is no fourth forward and no need to
process `y3`. An end-of-sequence token can stop the loop earlier. Finally,
`generate` decodes only the newly appended IDs, with special tokens skipped,
into the returned text. This basic API returns a completed string; emitting
network events as tokens arrive is a separate serving concern.

One small optimization already exists in the model interface:
`last_token_only=True` selects the final hidden row **before** the vocabulary
projection. It produces `[1, 1, V]` logits instead of `[1, T, V]`. It does
not avoid computing the earlier hidden states or create a KV cache. Here T
is the current input-history length, including any generated tokens fed back.
Tinyserve's current `generate(use_cache=False)` keeps the full-logit baseline;
the cached prefill path uses the last-row projection optimization.

## Count the recomputation

Our three calls process 3, then 4, then 5 token positions: 12 positions in
total to produce three new tokens. More generally, with prompt length P
and N output tokens, assuming no early termination and positive N,

$$
\sum_{s=0}^{N-1}(P+s)=NP+\frac{N(N-1)}{2}.
$$

This counts token positions passed through the transformer, **not total
FLOPs or elapsed time**. Token-wise projections and feed-forward layers
repeat work for every old position. Attention also compares each query
with its permitted history: a full causal pass at length T has
`T × (T + 1) / 2` permitted query/key pairs per head. Its work is not
captured by the token-position count alone. Backend implementation,
matrix sizes, memory traffic, and launch overhead determine how those
operations turn into time.

The reason old states can be reused is causality. At fixed positions, adding
a future token does not change an earlier token's allowed history. For this
dense causal model in evaluation mode, earlier keys and values therefore
do not need to be recomputed just because the sequence grew.

A cached execution of the same three-token example would process the prompt
once and then only `y1` and `y2`: five newly processed positions instead of
twelve. It still performs attention against the growing history. Storing KV
removes repeated work at the cost of memory; it does not make all decode
cost independent of context length. Chapter 8 develops that trade.

Before comparing speeds, we need to name what is being timed. Model-forward
time, time to the first generated token, time between tokens, and total
request time are different quantities. The next chapter develops that
measurement vocabulary and the compute-versus-memory cost model. A slower
reference remains valuable because it gives faster paths a correctness
target.

## Check the boundaries of the example

The executable companion
[test_book_first_token.py](https://github.com/kaix-nv/tinyserve/blob/14b0c8dbb975350bbd730ac0c702eafce36eb95c/tests/test_book_first_token.py) checks the
small model's shapes, output-weight tying at construction, final-row
projection, greedy feedback, and causal independence of earlier logits.
Its synthetic CPU model makes the example inspectable; passing these checks
is not evidence that random weights can produce useful language.

Checkpoint correctness needs a separate comparison. Tinyserve's
[parity tests](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/test_parity.py) compare the trained model against
the Transformers implementation using fixed inputs, FP32 logit tolerances,
and a bounded BF16 greedy-prefix criterion. Fluent-looking output alone
cannot reveal a wrong RoPE convention or missing normalization. Conversely,
floating-point changes near a tied top score can alter a greedy token, so
the numerical contract must specify more than “the text looks similar.”

## Follow the implementation

| File | Responsibility in this chapter |
|---|---|
| [config.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/config.py) | Select the architecture and read explicit dimensions. |
| [loader.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/loader.py) | Resolve the artifact, construct on meta, load weights, and rebuild derived buffers. |
| [models/qwen3.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/models/qwen3.py) | Embeddings, decoder blocks, position handling, and vocabulary logits. |
| [engine.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/engine.py) | Tokenization and `generate(..., use_cache=False)`. |
| [sampling.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/sampling.py) | Turn logits into the next token ID; temperature zero selects argmax. |

This chapter describes the ordinary dense, uncached path at code snapshot
`e20a348`. The linked source repository requires access; the worked example
and code excerpts above are self-contained. The
[book outline](/series/tinyserve/) shows where attention, caching, sampling,
and scheduling are developed further.

We now have the smallest complete generation path: configuration gives the
structure, weights supply its learned values, tokenization gives the input
IDs, a forward predicts the next token, and a loop turns those predictions
into a response. The next question is how to measure the cost of doing so.

{% include tinyserve-book-nav.html %}
