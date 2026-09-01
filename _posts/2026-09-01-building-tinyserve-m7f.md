---
layout: post
math: true
title: "Building tinyserve M7f: Kimi is KDA + MLA + MoE"
date: 2026-09-01 12:40:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Use a tiny Kimi-K3 checkpoint to follow channel-wise KDA state, compressed MLA history, latent MoE routing, MXFP4 loading, and the serving state each request must own."
---

*Milestone 7f of [building an LLM inference engine from scratch](/series/tinyserve/).
Previous: [M7e — linear attention is a different kind of memory]({% post_url 2026-08-31-building-tinyserve-m7e %}).*

Code: [`tinyserve` @ `10f7648`](https://github.com/kaix-nv/tinyserve/tree/10f7648e7104c6ce09e80f77612d249e2d4400a2).

M7e added Qwen3.5's Gated DeltaNet. It would be tempting to call that
“linear attention support” and map every later checkpoint onto the same
module. Kimi-K3 shows why that shortcut is wrong. Its text backbone combines
three contracts:

1. **Kimi Delta Attention (KDA)** with a different decay value for every key
   channel;
2. **Multi-head Latent Attention (MLA)** whose useful cache is a compressed
   latent, not expanded per-head K/V;
3. a **latent mixture of experts** with sigmoid top-16 routing, shared experts,
   SiTU activation, and block-level attention residuals.

All three affect checkpoint compatibility. Loading only the KDA tensors would
still execute the wrong model.

![The tiny Kimi-K3 text stack combines KDA, MLA, and latent MoE](/assets/tinyserve/m7f-kimi-k3-stack.svg)

## A real structural fixture, not a useful language model

The local `/home/scratch.kaix_coreai/models/kimi-k3` checkpoint is a random
debug model derived from `moonshotai/Kimi-K3`. It is intentionally small enough
to inspect and run on one A6000:

| property | tiny fixture |
|---|---:|
| checkpoint size | 133.5 MB |
| residual width | 8 |
| decoder layers | 17 |
| attention schedule | 12 KDA + 5 MLA |
| KDA geometry | 8 heads × 128 dimensions |
| MLA latent | 512 + 64 designated rotary dimensions |
| routed experts | top-16 of 64 |
| shared experts | 2 |

The unusual width is deliberate: the surrounding projections, per-head
dimensions, latent cache, routing pattern, and MXFP4 checkpoint format remain
real. The model weights are random, so generated prose is not an accuracy
test. M7f instead checks tensor parity, cached-state parity, and deterministic
greedy decisions against the checkpoint's Transformers + FLA implementation.

Like M7e's Qwen3.5 support, M7f serves only the text backbone. The vision tower
and multimodal projector are present in the checkpoint but deliberately not
loaded.

## KDA: every channel decides how much to forget

Qwen3.5 Gated DeltaNet uses one decay value per head at a token. KDA uses a
vector: every row of the recurrent matrix can decay by a different amount.
For one head, with key dimension `d`, the state is
`S[d, value_dimension]`:

$$
\begin{aligned}
\bar S_t[d,:] &= \exp(g_t[d]) S_{t-1}[d,:], \\
e_t &= v_t - k_t^\top \bar S_t, \\
S_t &= \bar S_t + k_t\left(\sigma(\beta_t)e_t\right)^\top, \\
o_t &= \frac{q_t^\top S_t}{\sqrt{d}}.
\end{aligned}
$$

The delta-rule part is familiar from M7e: read what the state already predicts,
then write only the error. The new detail is the first line. `g_t` has 128
entries for each K3 head, so one channel can retain old state while another
forgets it quickly.

The released K3 checkpoint bounds each log-decay between `-5` and `0`:

$$
g_t = -5\,\sigma\!\left(\exp(A_{\log})
\left(f(h_t) + \mathrm{dt\_bias}\right)\right).
$$

The implementation keeps this equation in `kda_recurrence()` rather than
calling an optimized FLA kernel. Queries and keys are L2-normalized; `beta` is
sigmoided inside the update; the recurrent matrix remains FP32.

KDA also uses three independent width-4 depthwise convolutions before the
recurrence. A live request therefore owns, per KDA layer:

| state | shape | dtype | purpose |
|---|---|---|---|
| Q convolution tail | `[8×128, 3]` | BF16 | previous three raw Q columns |
| K convolution tail | `[8×128, 3]` | BF16 | previous three raw K columns |
| V convolution tail | `[8×128, 3]` | BF16 | previous three raw V columns |
| recurrent matrix | `[8, 128, 128]` | FP32 | complete summarized history |

That allocation is fixed in sequence length, but it is not free: every live
request needs its own copy.

## MLA: keep the token history without expanded K/V

The five global layers must still remember every token. A direct
implementation first expands a 512-value latent into a separate key and value
for all eight heads, then caches the expansion. That is mathematically simple
but misses the reason MLA exists.

For token `t`, let `c_t` be its normalized 512-value latent and `r_t` its
64-value rotary-designated key component. Split the MLA up-projection into
per-head key and value matrices `W_K` and `W_V`:

$$
k_t^{\text{nope}} = W_K c_t, \qquad v_t = W_V c_t.
$$

There is no need to materialize either tensor in the cache. Move `W_K` to the
query side when computing scores:

$$
\mathrm{score}(q,t)
= (q^{\text{nope}} W_K)c_t
+ q^{\text{rope}}r_t.
$$

After softmax, aggregate latents first and apply `W_V` once:

$$
o = \left(\sum_t p_t c_t\right) W_V^\top.
$$

M7f therefore stores only `(c_t, r_t)` and compares this absorbed computation
against an expanded-K/V oracle. For the tiny fixture:

| cache layout | bytes per MLA layer per token | all 5 MLA layers |
|---|---:|---:|
| expanded `8 × (192 K + 128 V) × BF16` | 5,120 | 25 KiB/token |
| compressed `(512 + 64) × BF16` | 1,152 | 5.625 KiB/token |

The compressed cache is 4.44× smaller on this eight-head fixture. The ratio is
larger on a production-width model because the 576-value latent is shared by
all heads.

One checkpoint-specific detail is worth stating explicitly: the local K3
eager reference names the 64-value component `qk_rope_head_dim`, but does not
apply a rotation in `modeling_kimi_linear.py`. M7f follows that executable
checkpoint contract for parity; it does not invent a position transform that
the reference path does not run.

![Kimi-K3 uses fixed KDA state and a compressed MLA cache](/assets/tinyserve/m7f-kimi-cache.svg)

## The feed-forward path is part of Kimi support

Layer 0 has a dense SiTU MLP. SiTU transforms both halves before multiplying:

$$
\mathrm{SiTU}(a,u) =
4\tanh(a/4)\sigma(a)\;
25\tanh(u/25).
$$

Layers 1–16 use a latent MoE:

1. project the 8-value residual row into a 256-value routed-expert space;
2. compute 64 sigmoid router scores in FP32;
3. add the learned correction bias only for expert selection;
4. select top-16 experts and renormalize their original sigmoid scores;
5. combine routed outputs, normalize, project back to width 8;
6. add two shared experts evaluated from the original residual row.

The checkpoint stores the 3,072 routed expert matrices in MXFP4. Each byte
contains two E2M1 values and every group of 32 columns has an E8M0 power-of-two
scale. `dequantize_mxfp4()` expands them once during loading. Its output is
bit-identical to `compressed-tensors 0.18.0` for the local checkpoint.

This is intentionally a readable baseline, not an optimized MXFP4 execution
path. Dequantization makes the 133.5 MB file roughly 408 MiB of BF16 model
parameters in memory. A later kernel milestone can keep routed experts packed.

K3 also replaces the ordinary `x + sublayer(x)` chain with **AttnRes**. Every
eight layers, a new block residual stream begins. Learned normalized scores
softmax over the saved block streams plus the current prefix sum. Omitting this
mixing produces plausible tensor shapes and wrong logits, which is why it lives
in the first compatibility milestone.

## A concrete cached trace

The tokenizer maps the raw prompt `2+2=` to `[17, 10, 17, 28]`.

During prefill:

1. KDA layer 0 begins with zero convolution tails and zero `S`, processes four
   token updates, then retains only the final tails and matrix.
2. Layers 1 and 2 do the same independently.
3. MLA layer 3 appends four normalized latents and four 64-value key components
   to its compressed cache.
4. The pattern repeats through layer 16.
5. `KimiCache.seq_len` advances from 0 to 4 only after every layer finishes.

Suppose greedy sampling chooses token `42`. Decode sends only `[42]` through
the model. Each KDA layer shifts one column into each convolution tail and
updates the same recurrent matrix. Each MLA layer appends one `(512 + 64)`
entry and attends over five cached positions. The shared cursor advances to 5.

Recomputing all five tokens in FP32 and performing this cached transition choose
the same argmax; the largest logit difference is `3.73e-8`.

## Validation

The acceptance ladder separates storage, layer algebra, cache lifecycle, and
the full checkpoint:

| gate | retained result |
|---|---|
| K3 config and schedule | 12 KDA, 5 MLA, top-16 of 64 experts |
| MXFP4 expansion | bit-identical to `compressed-tensors` |
| KDA token equation | exact FP32 unit parity |
| compressed MLA vs expanded K/V | within `2e-6` FP32 |
| cached decode vs recomputation | max logit error `3.73e-8`, same argmax |
| Tinyserve vs Transformers + FLA BF16 | max `0.00159`, mean `0.000268` |
| decision parity | identical argmax and top-five token order |
| focused tests | `5 passed` |

The small BF16 difference comes from two intentional associations: FLA runs a
parallel chunk KDA kernel, while Tinyserve executes token order; the reference
expands MLA K/V before attention, while Tinyserve absorbs the expansion around
the latent cache.

## Same-checkpoint calibration

The August 31, 2026 snapshot used one RTX A6000, BF16 weights, greedy raw-text
generation, one warmup, and three measured repeats. Each request generated
eight tokens. Model loading, MXFP4 decompression, and the reference's initial
Triton compilation were outside measured runs.

| engine | 64 prompt + 8 output | 128 prompt + 8 output |
|---|---:|---:|
| tinyserve M7f readable KDA + latent MLA | 1.148 s / 6.97 tok/s | 1.134 s / 7.06 tok/s |
| Transformers 5.15.1 + FLA 0.5.2 | 0.875 s / 9.15 tok/s | 0.845 s / 9.46 tok/s |

The reference is 1.31–1.34× faster. That is the expected baseline: its KDA
prompt scan and one-token transition use Triton kernels, while M7f launches a
Python-visible recurrence and loops over selected experts. The tiny random
model and eight-token output make this useful for locating mechanism overhead,
not predicting production Kimi throughput.

The checkpoint SHA-256 is
`8ec5cd600c8c3c487654d1d59bebb57e3bbce3ce28cb415b564cb154f62424db`.

## Run and debug it

Generation uses the ordinary M1 entry point:

```bash
.venv/bin/python examples/generate.py \
  --model /home/scratch.kaix_coreai/models/kimi-k3 \
  --prompt "2+2=" --no-chat --max-new-tokens 4 --verbose
```

The output is random because the checkpoint is random. Inspect the state, not
the prose. The **M7f: Kimi-K3 text backbone** launch configuration runs this
command. Set breakpoints in this order:

1. `KimiDeltaAttention._causal_conv()` — watch three independent tails shift;
2. `kda_recurrence()` — compare channel-wise `decay_log`, `remembered`, and
   the new recurrent matrix;
3. `KimiMLAAttention.forward()` — inspect `latent`, `absorbed_query`, and the
   576-value cache append;
4. `KimiMoEGate.forward()` — separate correction-biased selection from the
   original combining weights;
5. `KimiLinearModel.forward()` — see AttnRes accumulate block streams and the
   cache cursor advance once.

The stronger cross-engine check needs the local FLA checkout and the
`compressed-tensors` reference loader:

```bash
PYTHONPATH="$PWD" .venv/bin/python examples/validate_kimi.py \
  --model /home/scratch.kaix_coreai/models/kimi-k3 \
  --device cuda:1
```

## What M7f establishes

M7f supports the Kimi-K3 **text execution contract** on one request: exact
checkpoint loading, readable KDA, compressed MLA state, latent MoE, AttnRes,
tokenization, prefill, and cached decode.

It does not yet provide vision input, scheduled batches, prefix reuse,
preemption, packed MXFP4 kernels, or production Kimi-K3 validation. M7g can now
make GDN matrices, KDA matrices, convolution tails, MLA latents, and ordinary
KV explicit scheduler-owned resources without guessing what one “hybrid
cache” means.
