---
layout: post
math: true
title: "Building tinyserve M0: A correct model, the slowest possible way"
date: 2026-07-14 15:41:31 -0700
categories: [tinyserve, llm-serving]
excerpt: "Implement Qwen3 from scratch, prove parity with Transformers, and measure the cost of naive autoregressive decoding."
---

*Milestone 0 of [building an LLM inference engine from scratch](/series/tinyserve/).
Code: [`tinyserve` @ `f5c3627`](https://github.com/kaix-nv/tinyserve/tree/f5c3627).*

Every inference engine is, at its core, three things: a model that turns
token IDs into next-token logits, a loop that feeds tokens back into it,
and machinery to make that loop fast. M0 builds the first two and
deliberately skips the third. No KV cache, no batching, no kernels beyond
what stock PyTorch gives us. The result is an engine that produces
*exactly* the right tokens at an embarrassing 3.5 tok/s once the context
gets long — and that number is the entire reason M0 exists. You can't
appreciate a KV cache until you've paid for not having one.

The other rule of the game: **no peeking**. vLLM, SGLang, and nano-vllm
sit one directory up from this repo, and the rule is to write each
milestone before reading how they did it.

## The target: Qwen3-0.6B

Small enough to iterate on in seconds, modern enough that nothing about it
is a toy — it has every architectural feature its 235B sibling has:

| | Qwen3-0.6B |
|---|---|
| layers | 28 |
| hidden size | 1024 |
| attention heads (Q / KV) | 16 / 8 → GQA, group size 2 |
| head dim | **128** (note: *not* 1024/16 = 64) |
| MLP intermediate | 3072 (SwiGLU) |
| vocab | 151,936 |
| position encoding | RoPE, θ = 1M |
| normalization | RMSNorm, pre-norm + **QK-norm** |
| lm_head | tied to embeddings |

Three of those rows are traps for anyone who has only implemented
Llama-style models before. We'll hit each one.

## The forward pass, from scratch

The whole model is [`tinyserve/models/qwen3.py`](https://github.com/kaix-nv/tinyserve/blob/f5c3627/tinyserve/models/qwen3.py)
— about 150 lines. The skeleton is the standard pre-norm decoder:

```python
def forward(self, x, cos, sin):
    x = x + self.self_attn(self.input_layernorm(x), cos, sin)
    x = x + self.mlp(self.post_attention_layernorm(x))
    return x
```

Embeddings in, 28 of these blocks, a final RMSNorm, then a projection to
vocab-size logits. Everything interesting hides inside `self_attn`.

### RMSNorm: the easy one (with one footgun)

```python
x = x.float()
x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
return (x * self.weight.float()).to(dtype)
```

LayerNorm minus the mean-centering: rescale each vector to unit RMS,
then multiply by a learned per-channel gain. The footgun is the
`.float()`: the model runs in bf16, which has ~3 decimal digits of
precision, and squaring-then-averaging 1024 values in bf16 loses enough
of them that logits drift. Norms are cheap; compute them in fp32 and
cast back. Every serious implementation does this, and it's invisible
until you diff logits against a reference and wonder where your 1e-2
error comes from.

### Position first: attention cannot see order by itself

The dot product inside self-attention compares token content; it has no
term saying where either token came from. If a query sees the same keys
and values in a different order, their weighted sum is unchanged. A
causal mask says which earlier tokens are visible, but not how far apart
two visible tokens are. The model therefore needs an additional position
signal: this token is at position 0, the next at position 1, and so on.

Some models add a position vector to each token embedding. Qwen3 instead
uses **rotary position embeddings (RoPE)**: after producing a query `q`
and key `k`, it changes their coordinates according to the token's
position. Values are not rotated.

### RoPE: represent position as rotation

Take one pair of coordinates $(a,b)$ from an attention head. Rotating it
by an angle $\phi$ produces:

$$
a' = a\cos\phi - b\sin\phi, \qquad
b' = a\sin\phi + b\cos\phi.
$$

You can picture $(a,b)$ as an arrow on a sheet of paper. RoPE does not
append the position number to that arrow; it turns the arrow by an amount
determined by the position. The same content at two positions therefore
has the same length but a different orientation.

Qwen3's head dimension is 128, so RoPE treats it as 64 such pairs. Pair
$i$ at token position $p$ uses the angle
$\phi_{p,i} = p\,\theta^{-2i/128}$. At position 0 there is no rotation.
Moving forward one token advances each pair by a fixed angle. Different
pairs rotate at different speeds, like 64 clock hands, giving attention
several scales on which to measure distance.

The useful part appears when attention takes a dot product. Rotating two
arrows by the same amount does not change the angle between them, so that
shared rotation cancels. Let $R_p$ mean "apply all the RoPE rotations for
position $p$." For a query at position $p$ and a key at position $s$:

$$
(R_p q)^\mathsf{T}(R_s k) = q^\mathsf{T}R_{s-p}k.
$$

The token content still comes from $q$ and $k$, but the positional part of
their attention score depends on the relative offset $s-p$, not their
absolute positions separately. Each token can therefore apply RoPE on
its own; no pairwise position matrix is needed.

This also fits KV caching cleanly. A key at position $s$ is rotated once
with $R_s$ and stored. Every later query can attend to that stored key
without changing it. Strictly speaking, causal attention is what makes
old K/V states reusable; RoPE makes the position-adjusted key itself
self-contained.

### The channel-pairing convention

"Pair the coordinates" still leaves an implementation choice. For an
8-dimensional vector, an interleaved implementation pairs
$(0,1), (2,3), (4,5), (6,7)$. Hugging Face's Llama implementation instead
pairs the two halves: $(0,4), (1,5), (2,6), (3,7)$. For Qwen3's
128-dimensional heads, those pairs are $(0,64), (1,65), \dots, (63,127)$.

That second convention explains `rotate_half`:

```python
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

q = q * cos + rotate_half(q) * sin
```

For an 8-dimensional input, `rotate_half` changes
`[x0, x1, x2, x3, x4, x5, x6, x7]` into
`[-x4, -x5, -x6, -x7, x0, x1, x2, x3]`. Combined with `cos` and `sin`,
this rotates each pair from opposite halves using the equations above.

The two pairing conventions describe the same rotation mathematics after
a fixed permutation of channels. They are not interchangeable inside an
already-trained checkpoint, however: the learned query and key weights,
and the arrangement of `cos` and `sin`, expect the convention used during
training. Choose the other one and the model still runs, but its attention
scores and logits are wrong. The parity test catches exactly this mistake.

### GQA: the KV cache diet, implemented before the cache exists

Qwen3-0.6B has 16 query heads but only 8 KV heads; each KV head serves a
group of 2 query heads. The motivation is entirely about M1 and beyond:
KV cache size scales with the number of KV heads, and at scale the cache
— not the weights — is what limits how many sequences fit on a GPU.
Halving KV heads halves the cache for a small quality cost. (Qwen3-8B is
more aggressive: 32 Q heads, 8 KV heads.)

In M0, with no cache, GQA is just a shape detail:

```python
k = k.transpose(1, 2).repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
```

materializing each KV head twice so `scaled_dot_product_attention` sees
16 of each. Copying the thing you invented GQA to avoid copying is
comically wasteful — a real kernel indexes KV heads in place (and newer
PyTorch has `enable_gqa=True`). Noted for M3, kept dumb for M0.

### QK-norm: the Qwen3-specific bit

Qwen2 put biases on the QKV projections; Qwen3 removed them and instead
RMS-normalizes q and k **per head, before RoPE**:

```python
q, k = self.q_norm(q), self.k_norm(k)   # RMSNorm over head_dim=128
q, k = apply_rope(q, k, cos, sin)
```

The training-time motivation is attention-logit stability: q·k can blow
up as scale grows, and normalizing both vectors bounds it. For us it's
two extra 128-dim RMSNorms — but note both order dependencies: per-head
(not over the full hidden dim), and norm-then-rotate (RoPE is a rotation,
so it preserves the norm; rotating first then normalizing gives the same
k... but normalizing *before* projecting or after merging heads does
not). Get either wrong and you fail parity by a lot.

### The rectangular projection trap

For most models `head_dim = hidden_size / num_heads`. Qwen3 breaks this:
hidden is 1024, but the config says `head_dim: 128`, so 16 heads × 128 =
2048 — attention runs in a space *twice as wide* as the residual stream.
`q_proj` is 1024→2048 and `o_proj` is 2048→1024, both rectangular.
If you compute head_dim the traditional way you get 64, and every
attention weight fails to load with a shape error. The config field
exists; read it, don't derive it.

### SwiGLU and tied embeddings

The MLP is three matmuls: `down(silu(gate(x)) * up(x))` — a SiLU-gated
linear unit, standard since Llama. And because a 151,936 × 1024 embedding
matrix is 156M parameters — a quarter of this "0.6B" model — Qwen3-0.6B
uses the same matrix for input embeddings and the output lm_head.
That tie caused the milestone's one real bug. Which brings us to loading.

## Loading the weights (and the one bug that bit)

Trick #1: name your modules exactly like HF names theirs
(`model.layers.3.self_attn.q_proj`, ...) and checkpoint loading is a
no-op — read the safetensors shards into a dict, `load_state_dict`,
done. No remapping table to maintain. This holds until tensor
parallelism (M5), where weights get sharded and a real mapping layer
becomes unavoidable.

Trick #2: build the model under `torch.device("meta")` — shape metadata
only, no allocation — then `load_state_dict(..., assign=True)` to adopt
the checkpoint tensors directly. Skips allocating 1.2GB of random-init
weights just to overwrite them. For 0.6B this is cosmetic; for 8B+ it's
the difference between loading in seconds and paging.

The bug: the checkpoint contains no `lm_head.weight` (it's tied, so
storing it would be redundant), and my first loader *popped* the key and
relied on the tie created in `__init__`. But `load_state_dict` with
`strict=True` walks the module tree, finds an `lm_head.weight` parameter,
and demands a value for it:

```
RuntimeError: Missing key(s) in state_dict: "lm_head.weight".
```

Worse, the "obvious" fix — load non-strict, re-tie afterwards — hides a
subtler trap: `assign=True` *replaces* parameter objects, so any tie
created in `__init__` is silently severed by loading. The right fix is
to alias the tensor in the state dict itself:

```python
if config.tie_word_embeddings:
    state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
```

Same tensor object under both keys; `assign=True` installs it in both
places; the tie survives because it *is* the same storage. Lesson filed
away: weight tying and `assign=True` interact, and every engine has a
loader full of exactly this kind of edge case.

## Proving it's right

"It generates fluent text" is not a correctness test — a model with
mismatched RoPE layout still writes plausible English while being
measurably worse. The test is `transformers` itself
([`tests/test_parity.py`](https://github.com/kaix-nv/tinyserve/blob/f5c3627/tests/test_parity.py)):

1. **Logits parity, fp32**: one forward pass, ours vs
   `AutoModelForCausalLM`, max absolute logit difference. Measured:
   **1.6e-4** (threshold 1e-3). fp32 because bf16 noise would mask real
   convention bugs at this tolerance.
2. **Greedy decode parity, bf16**: generate 32 tokens with both, compare
   token IDs — this exercises the actual dtype and the actual loop. One
   wrinkle: in bf16, when the top-2 logits are within noise of each
   other, ours and HF can legitimately pick different tokens and the
   sequences diverge from there. So the test requires a long exact
   prefix (≥30/32) rather than full equality. In practice: 32/32.

Both pass. The model is *right*. Now let's see how it's *slow*.

## The cost of no KV cache

The M0 generate loop re-runs the full forward pass over the *entire*
sequence for every new token, and throws away everything except the last
position's logits:

```python
for step in range(max_new_tokens):
    logits = self.model(ids, positions)[0, -1]   # recompute EVERYTHING
    ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
```

Per-step cost is a full forward over $T$ tokens, so a generation of $N$
tokens does $O(N^2)$ token-forwards. Measured (bf16, one RTX A6000,
10-run average per point):

| seq len | ms/step | tok/s |
|---:|---:|---:|
| 128 | 21.4 | 46.7 |
| 256 | 20.4 | 49.0 |
| 512 | 21.7 | 46.1 |
| 1024 | 29.6 | 33.8 |
| 2048 | 61.5 | 16.3 |
| 4096 | 127.3 | 7.9 |
| 8192 | 284.5 | 3.5 |

Two regimes, both instructive:

- **Flat until ~512.** Recomputing 512 tokens costs the same as
  recomputing 128?! Yes: at tiny sizes the GPU is bound by kernel-launch
  and Python overhead, not math. 28 layers × ~10 kernels each, ~2300
  launches per step at ~µs-scale each, while the actual matmuls on a
  0.6B model are trivially small. The GPU is idling between launches.
  This is the regime CUDA graphs (M5) exist for.
- **Linear growth after.** Once there's enough work to saturate the GPU,
  per-step cost grows linearly with $T$ (doubling 4096→8192 doubles
  ms/step almost exactly: 127→285), i.e. total generation cost grows
  quadratically. At 8k context we're 13× slower per token than at 512.

Worth internalizing what the linear term *is*: it's mostly the
re-projection of K and V for thousands of tokens whose K and V **have
not changed since the last step**. RoPE guaranteed they never will.
Every byte of that recomputation is a pure waste, and storing it instead
costs `2 × layers × kv_heads × head_dim × 2 bytes` = **112 KB per token**
(28 × 8 × 128 × 2 × 2) — about 0.9 GB for an 8k context. Trading 0.9 GB
of a 48 GB card to delete the quadratic term is the single best deal in
inference.

That trade is M1: split the loop into *prefill* (one pass over the
prompt, cache K/V) and *decode* (one token in, attend against the
cache). Same tokens out — that's the acceptance test — with per-step
cost that no longer grows with... well, mostly. The attention itself
still reads a linearly-growing cache; where compute goes after that is
the story of the entire rest of this series.

## Reproducing

```bash
uv venv && uv pip install -e '.[dev]'
pytest tests/ -x                                   # parity vs transformers
python examples/generate.py --verbose              # watch it generate
```

The benchmark table: time `model(ids, positions)` for random sequences
of each length, 3 warmup + 10 timed iterations, `torch.cuda.synchronize()`
around the timer (CUDA is async — time without syncing and you measure
kernel *launches*, not kernels).
