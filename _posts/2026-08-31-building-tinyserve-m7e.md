---
layout: post
math: true
title: "Building tinyserve M7e: Linear attention is a different kind of memory"
date: 2026-08-31 12:34:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Follow one Gated DeltaNet token update, account for its fixed recurrent state and growing hybrid KV cache, and measure the gap between a readable recurrence and fused implementations."
---

*Milestone 7e of [building an LLM inference engine from scratch](/series/tinyserve/).
Previous published article: [M5b — prefix caching]({% post_url 2026-08-23-building-tinyserve-m5b %}).*

Code: [`tinyserve` @ `8b1a41b`](https://github.com/kaix-nv/tinyserve/tree/8b1a41bf753a379809114aaedd68abc0f3420c3a).

> M6 and M7a–M7d are already implemented and documented in the repository.
> This article focuses on the first milestone that changes the model's state
> family rather than another way to execute or store ordinary attention.

The earlier tinyserve milestones all served one family of decoder: every
attention layer remembered one key and one value per token. Paging changed
where those tensors lived. Prefix caching changed how long they lived.
FlashInfer changed how they were read. None changed the fact that the state
grew with the token history.

M7e adds a model whose memory has a different shape. Qwen3.5-0.8B repeats
three **Gated DeltaNet** linear-attention layers followed by one gated
full-attention layer, six times. Eighteen layers compress their history into
fixed recurrent matrices. Six layers still keep ordinary growing KV.

![Qwen3.5 combines fixed recurrent state with growing full-attention
KV](/assets/tinyserve/m7e-hybrid-linear-state.svg)

This is not a drop-in replacement for Qwen3's softmax. Qwen3.5 was trained
with this recurrence. Replacing a Qwen3 attention backend with a convenient
linear formula would run different mathematics under incompatible weights.

## Start from one token, not the name

“Linear attention” is often introduced as an asymptotic claim. The serving
mechanism is easier to see by following one head through one token.

Let `S` be a matrix with key dimension along its rows and value dimension
along its columns. At token `t`, Qwen3.5 produces a normalized key `k`, a
normalized and scaled query `q`, a value `v`, a negative decay log `g`, and a
write strength `beta`:

$$
\begin{aligned}
\bar S_t &= \exp(g_t) S_{t-1}, \\
e_t &= v_t - k_t^\top \bar S_t, \\
S_t &= \bar S_t + k_t (\beta_t e_t)^\top, \\
o_t &= q_t^\top S_t.
\end{aligned}
$$

The first line forgets some old memory. The second asks what value the old
state already associates with this key. The third writes only the error. The
last reads the updated association with the current query.

Consider a two-dimensional toy head:

```text
S       = [[1, 0],       decay = 0.5
           [0, 1]]       beta  = 0.25
k       = [1, 0]         v     = [3, 4]
q       = [0.5, 1]
```

After decay, `Sbar = [[0.5, 0], [0, 0.5]]`. Reading with `k` returns
`[0.5, 0]`, so the error is `[2.5, 4]` and the beta-scaled write is
`[0.625, 1]`. The new matrix is:

```text
S = [[1.125, 1.0],
     [0.0,   0.5]]
```

The query reads `[0.5625, 1.0]` before the model's `1/sqrt(key_dim)` scale.
Token 1,000 updates the same matrix; it does not append a 1,000th matrix.

## What Qwen3.5 puts around that recurrence

The recurrence is the center, not the whole layer. In
[`QwenHybridGatedDeltaNet.forward()`](https://github.com/kaix-nv/tinyserve/blob/8b1a41bf753a379809114aaedd68abc0f3420c3a/tinyserve/models/qwen_hybrid.py#L185-L218)
one normalized residual row takes this path:

```text
h
├─ in_proj_qkv ─> depthwise causal conv ─> SiLU ─> q, k, v
├─ in_proj_a   ─> g = -exp(A_log) * softplus(a + dt_bias)
├─ in_proj_b   ─> beta = sigmoid(b)
└─ in_proj_z   ─> output gate z

q, k ─> L2 norm ─> recurrent delta rule ─> RMSNorm * SiLU(z) ─> out_proj
```

The width-4 convolution means a layer needs the previous three raw projected
QKV columns in addition to `S`. During decode, the new projected column joins
that three-column tail, produces one convolved column, and becomes part of the
next tail.

For the local 0.8B checkpoint, one live request therefore owns across its 18
linear layers:

| state | per-layer shape | dtype | all 18 layers |
|---|---|---|---:|
| convolution tail | `[6144, 3]` | BF16 | 0.63 MiB |
| recurrent matrices | `[16, 128, 128]` | FP32 | 18.00 MiB |
| **fixed linear state** | | | **18.63 MiB** |

“Constant in sequence length” does not mean “free.” The recurrent allocation
is fixed for a request but grows with concurrency.

## Why the cache is still hybrid

Layers 3, 7, 11, 15, 19, and 23 are gated full attention. They use two KV
heads of width 256 and append BF16 K and V for every token:

$$
6\ \text{layers} \times 2\ \text{heads} \times 256 \times
2\ (K,V) \times 2\ \text{bytes} = 12\ \text{KiB/token}.
$$

The combined state for one request is consequently:

| cached tokens | fixed linear state | full-layer KV | total state |
|---:|---:|---:|---:|
| 128 | 18.63 MiB | 1.50 MiB | 20.13 MiB |
| 2,048 | 18.63 MiB | 24.00 MiB | 42.63 MiB |
| 32,768 | 18.63 MiB | 384.00 MiB | 402.63 MiB |

Linear layers flatten most of the context-dependent growth. They do not make
Qwen3.5 a cache-free model.

## The smallest correct M7e cache

[`HybridCache`](https://github.com/kaix-nv/tinyserve/blob/8b1a41bf753a379809114aaedd68abc0f3420c3a/tinyserve/hybrid_cache.py#L16-L81)
deliberately supports one fixed batch. It contains:

```python
conv       # [18, B, 6144, 3]       model dtype
recurrent  # [18, B, 16, 128, 128]  FP32
k, v       # [6, B, max_length, 2, 256] model dtype
seq_len    # shared append cursor for the six full-attention layers
```

One full prompt starts all recurrent state at zero, runs the same token update
from left to right, stores the final matrices and convolution tails, and fills
the six KV layers. Decode then passes one token at a time and mutates those
same tensors.

That is enough for `generate()`. It is not enough for `serve()`:

- a batch needs a stable state slot for each live request;
- finishing or preempting must release or reset that slot;
- packing prompts needs sequence boundaries for the recurrence;
- prefix reuse must restore recurrent state and the convolution tail at the
  same boundary as full-attention KV.

Reusing only the six KV layers would silently combine the correct visible KV
with a recurrent summary of some other history. M7e therefore rejects paged
and batched Qwen3.5 entry points instead of pretending the old cache lifecycle
is sufficient.

## Checkpoint details that are easy to miss

The module tree mirrors the checkpoint's `model.language_model.*` names, so
the loader can select all 320 text tensors while ignoring 168 vision/MTP
tensors. Three architectural details differ from Qwen3:

1. Ordinary RMSNorm stores a zero-centred scale and applies `1 + weight`.
2. A full-attention query projection interleaves each head's query and output
   gate; splitting the projection into two large halves is wrong.
3. Only the first quarter of each 256-wide Q/K head receives RoPE. For
   text-only input the three multimodal position axes coincide, reducing
   interleaved MRoPE to ordinary one-dimensional partial RoPE.

These are checkpoint contracts, not optional optimizations.

## Numerical validation

The
[`test ladder`](https://github.com/kaix-nv/tinyserve/blob/8b1a41bf753a379809114aaedd68abc0f3420c3a/tests/test_linear_attention.py)
separates the recurrence from the model:

1. A random FP32 recurrent trace matches the four token equations above.
2. Qwen3.5-0.8B cached generation matches full recomputation in FP32.
3. The raw completion `2+2=` begins with `4`.
4. Against Hugging Face BF16 eager inference, the final logit vector has a
   maximum absolute difference of 0.1875 and mean difference of 0.0253. The
   top-five token order and argmax are identical.

Hugging Face uses a parallel chunk algorithm for a multi-token prompt;
tinyserve M7e intentionally executes the readable recurrent equation. BF16
reassociation can move later near-tied logits enough for this small model to
diverge after the shared first token. FP32 cached versus recomputed generation
is the semantic oracle; the BF16 difference is retained as execution evidence.

## Calibration snapshot: semantics before speed

The first performance question is not “did linear attention beat Qwen3?” That
would compare different checkpoints and different mathematics. The useful
question is whether this implementation leaves an obvious execution gap on
the **same** Qwen3.5 weights.

The August 31, 2026 snapshot used one RTX A6000, BF16 weights, greedy raw-text
generation, exact 128- and 2,048-token prompts, 32 requested output tokens,
two warmups, and five measured repeats. Tinyserve and Hugging Face ran twice
in opposite order; the table pools the ten samples. llama.cpp ran five times
through its local streaming HTTP endpoint, so its timing boundary is broader,
not narrower.

| engine | execution path | 128 prompt + 32 output | 2,048 prompt + 32 output |
|---|---|---:|---:|
| tinyserve M7e | readable token recurrence | 0.964 s | 3.976 s |
| Transformers 5.15.1 | PyTorch chunk/recurrent fallback | 0.979 s | 1.137 s |
| llama.cpp `d7bd3bfcad3e` | CUDA graph + native delta-net operators | 0.180 s | 0.336 s |

The 128-token Tinyserve and Transformers rows are effectively tied. At 2,048
tokens, Tinyserve is 3.5x slower than Transformers. The prompt loop makes the
cause visible: `gated_delta_recurrence()` launches one small chain per token,
whereas the other engines evaluate prompt chunks with a parallel scan or
fused operator. llama.cpp is 5.3x faster at 128 tokens and 11.8x faster at
2,048 despite including HTTP streaming in its measurement.

This is not an optimization win, nor was one expected in M7e. It is the first
native baseline and a falsifiable target: M7g must close the long-prompt gap
without changing the recurrent-state result established here. M7f comes first
because kernel speed does not solve request ownership, preemption, or prefix
state restoration.

The original Hugging Face index SHA-256 was
`d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e`.
The llama.cpp row used a lossless 320-tensor BF16 text conversion with GGUF
SHA-256 `981854ca1ec27c079bc91c80d026284b75229df3c2382b684d8a09d9b958e1bd`.
All three passed the `2+2=` -> `4` smoke check. GPU clocks were stable within
each run, but differed across engines, so these numbers locate the engineering
gap rather than certify a production ranking.

Two requested local engines do not produce honest rows on this host.
FreeToken `58f4b9ec0e16` contains a scheduler-owned fused GDN path, but its
pinned Torch 2.11/Triton 3.6 runtime is not installed in Tinyserve's Torch
2.13/Triton 3.7 environment. NInfer `6b94b8c5721f` rejects GPUs other than
`sm_120a`, while the A6000 is `sm_86`. Ollama `f96e7aa0513b` recognizes
Qwen3.5, but its CUDA path uses the same llama.cpp mechanism already measured,
so listing it as an independent engine would double-count one implementation.

## Run and debug it

```bash
.venv/bin/python examples/generate.py \
  --model /home/scratch.kaix_coreai/models/Qwen3.5-0.8B \
  --prompt "2+2=" --no-chat --max-new-tokens 4 --verbose
```

The **M7e: native linear attention** launch configuration runs this command.
Set breakpoints in this order:

1. `QwenHybridGatedDeltaNet._causal_conv()` at `previous` and `joined`;
2. `gated_delta_recurrence()` at `remembered`, `error`, and the new `state`;
3. `QwenHybridAttention.forward()` at `cache.update_kv()`;
4. `HybridCache.advance()` at the shared token cursor.

At the second forward, the input length is one. The recurrent matrix shape is
unchanged, the convolution tail shifts by one column, and each full-attention
KV tensor grows by one token.

## What M7e establishes

M7d separated attention execution from KV lifecycle. M7e shows why a modern
engine also needs to model **state families**. A recurrent matrix is neither a
KV page nor a warm prefix block. It has its own shape, precision, update rule,
and request lifecycle.

M7f can now add scheduler-owned hybrid slots without guessing the algebra. It
will keep one readable state transition as the oracle, then decide how prompt
packing, preemption, and continuous batching move those states through the
server.
