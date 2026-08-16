---
layout: post
title: "Building tinyserve M1: The KV cache"
date: 2026-07-14 15:41:40 -0700
categories: [tinyserve, llm-serving]
excerpt: "Split inference into prefill and decode, cache every layer's keys and values, and make decode cost flat with context length."
---

*Milestone 1 of [building an LLM inference engine from scratch](/).
Previous: [M0 — a correct model, the slowest possible way]({% post_url 2026-07-14-building-tinyserve-m0 %}).*

## The problem

M0 ended with this table — per-token decode cost of an engine that
recomputes the entire sequence for every new token:

| seq len | M0 ms/step | M0 tok/s |
|---:|---:|---:|
| 512 | 21.7 | 46.1 |
| 2048 | 61.5 | 16.3 |
| 8192 | 284.5 | 3.5 |

Per-step cost grows linearly with context, total generation cost grows
quadratically, and at 8k tokens the engine produces 3.5 tokens per
second on a 48GB GPU running a 0.6B model. That's the problem.

## The idea

Look at what step $T$ actually computes: q, k, v for every token, then
attention of each query against each key. But only the *last* row of that
attention matters — we keep `logits[0, -1]` and throw the rest away. The
queries for old tokens are pure waste. What about their keys and values?

Here's the load-bearing fact from M0: **RoPE rotates k by the token's own
absolute position**. Token 17's k is the same tensor whether we're
generating token 18 or token 8000. Same for v (no position at all). So
every k and v we computed last step is *bit-identical* this step — and
recomputing them is the entire linear term in the table above.

So: store them. After processing each token, keep its per-layer k and v
in a preallocated buffer. Generation splits into two phases:

- **Prefill** — one forward over the whole prompt. Computes and caches
  K/V for every prompt token, yields logits for the first new token.
- **Decode** — one token per step. Project q/k/v for *one* token, append
  k,v to the cache, attend the single query against the whole cache.

Decode step work drops from $O(T)$ token-forwards to $O(1)$, plus an
attention read over the cache that grows with $T$ — we'll see below what
that read actually costs.

The price is memory. Per token: 2 (K and V) × 28 layers × 8 KV heads ×
128 dims × 2 bytes = **112 KB**. GQA already halved this (16 query heads,
8 KV heads). An 8k context costs 0.9 GB — noticeable, and this arithmetic
is why KV-cache size dominates serving-engine design from M3 onward.

## The build

Three new pieces, one modified one. ([Diff](https://github.com/kaix-nv/tinyserve/compare/f5c3627...87ab05b), ~150 lines.)

### The cache ([`kv_cache.py`](https://github.com/kaix-nv/tinyserve/blob/87ab05b/tinyserve/kv_cache.py))

One tensor pair, preallocated at `prompt_len + max_new_tokens`:

```
k, v: [num_layers, batch, max_seq_len, num_kv_heads, head_dim]
```

Preallocated because appending via `torch.cat` every step would copy the
whole cache — quadratic again, just in memcpy. Contiguous because it's
the simplest correct thing for one sequence. Both simplifications have
expiration dates: guessing `max_seq_len` up front over-reserves memory
(a request that *might* generate 32k tokens reserves 3.6 GB even if it
stops after 10), and that waste is exactly the motivation for paged KV
in M3.

One protocol subtlety: all 28 layers write their K/V *at the same token
offset* during a single forward, so the cache's write pointer must not
advance until the whole forward is done:

```python
def update(self, layer, k_new, v_new):       # called by each layer
    self.k[layer, :, self.seq_len:end] = k_new   # seq_len is read-only here
    ...
def advance(self, num_tokens):               # called once, by the model
    self.seq_len += num_tokens
```

My first sketch advanced `seq_len` inside `update()` — layer 0 advances
the pointer, layer 1 writes its K/V one slot too far, everything is
garbage from layer 1 down. The split into a read-only `update` and an
explicit end-of-forward `advance` makes that mistake structurally
impossible.

### Attention learns two shapes ([`models/qwen3.py`](https://github.com/kaix-nv/tinyserve/blob/87ab05b/tinyserve/models/qwen3.py))

The only change inside attention: after RoPE, push the new k,v through
the cache and attend against everything cached:

```python
q, k = apply_rope(q, k, cos, sin)
if kv_cache is not None:
    k, v = kv_cache.update(self.layer_idx, k, v)   # [B, past+T, H_kv, D]
...
o = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
```

That `is_causal = T > 1` line encodes the two M1 shapes:

- **Prefill** (T > 1, cache empty): q and k cover the same tokens →
  ordinary causal mask.
- **Decode** (T = 1): the single query is the newest token, so it's
  allowed to see *every* cached key — no mask at all. `is_causal=True`
  here would be wrong in a fun way: SDPA aligns the causal mask to the
  *top-left*, so a 1×N mask would let the query see only key 0.

The shape this can't handle — T > 1 into a non-empty cache — is chunked
prefill, which is M5's problem. An assert in `Qwen3Model.forward` makes
the limitation loud instead of silently wrong.

Positions matter too: decode must pass the token's *absolute* position
(`seq_len`, not 0) so RoPE rotates its q/k correctly. Get this wrong and
generation degrades subtly rather than crashing — the kind of bug the
parity test exists for.

### Sampling ([`sampling.py`](https://github.com/kaix-nv/tinyserve/blob/87ab05b/tinyserve/sampling.py))

Greedy was fine for correctness testing; a real engine samples.
Standard filter chain, in the standard order: divide logits by
temperature → **top-k** (keep the k best) → **top-p** (sort, softmax,
cumsum, keep the smallest prefix whose mass exceeds p) → sample from the
renormalized survivors. The classic top-p off-by-one: the boundary token
that *crosses* p must be kept (shift the drop mask by one), otherwise
the kept set can cover less than p — or with a peaked distribution and
small p, be empty.

### The engine loop ([`engine.py`](https://github.com/kaix-nv/tinyserve/blob/87ab05b/tinyserve/engine.py))

Prefill once, then a decode loop feeding one token back per step.
The M0 path survives behind `use_cache=False` — it's the correctness
oracle and the baseline. One small fix found while writing the loop:
sample first, *then* decide whether to run another forward — otherwise
the last iteration computes logits nobody ever samples from.

## The proof

The KV cache is a pure optimization: prefill+decode does *the same math*
as full recomputation minus the redundancy. So the bar is exact
reproduction, not "close" ([`tests/test_kv_cache.py`](https://github.com/kaix-nv/tinyserve/blob/87ab05b/tests/test_kv_cache.py)):

1. **Token-exact match, fp32**: 32 greedy tokens from a random 64-token
   prompt, cached vs naive — `torch.equal`, no tolerance. (fp32 because
   in bf16 the two paths hit different kernel shapes, and ~tied top-2
   logits can legitimately flip; that's kernel noise, not a cache bug.)
2. **Decode-logit match, fp32**: after prefill + several decodes, the
   next step's logits vs a full recompute of the same sequence —
   max diff **< 1e-3** observed.
3. Sampling unit tests: greedy = argmax; top-k=2 only ever emits the two
   best tokens; top-p tight enough to isolate the argmax is greedy.

All pass, plus M0's HF parity tests, unchanged.

## The payoff

Same sweep as M0's table (bf16, one RTX A6000, 20-run average;
`ctx` = tokens in cache when the decode step runs):

| ctx | M0 ms/step | M1 ms/step | speedup |
|---:|---:|---:|---:|
| 128 | 21.4 | 20.7 | 1.0× |
| 512 | 21.7 | 22.5 | 1.0× |
| 1024 | 29.6 | 20.7 | 1.4× |
| 2048 | 61.5 | 20.8 | 3.0× |
| 4096 | 127.3 | 21.2 | 6.0× |
| 8192 | 284.5 | 21.2 | **13.4×** |

Decode is now **dead flat**: ~21 ms/step whether the cache holds 128
tokens or 8192. The quadratic term is gone. Prefill, measured separately,
runs at ~30k tok/s (8192 tokens in 287 ms) — three orders of magnitude
faster per token than decode, which is the prefill/decode asymmetry
everything from M2 onward is built around: prefill is a few enormous
compute-bound matmuls, decode is thousands of tiny bandwidth-and-overhead-
bound ones.

But look closer at that flat 21 ms, because it's hiding the next two
milestones:

- **It should be ~2 ms, not 21.** A decode step must at minimum read
  every weight (1.5 GB in bf16) plus the KV cache (0.9 GB at 8k) from
  HBM: ~2.4 GB at the A6000's 768 GB/s ≈ 3 ms. We're 7× off that
  roofline, and the flatness across context lengths is the tell — even
  the 0.9 GB cache read at 8k doesn't move the needle, so the time isn't
  going to memory *or* math. It's going to ~2,300 kernel launches and
  the Python between them. Deleting that overhead is CUDA graphs, M5.
- **The GPU is almost idle either way.** One sequence decoding at
  47 tok/s uses a sliver of the machine — the same hardware just did
  prefill math at 30k tok/s. The way to spend the idle capacity is to
  decode many sequences at once: reading the weights once per step is
  ~free for 1 sequence or 64, so throughput scales almost linearly with
  batch size until bandwidth saturates. That's M2, static batching.

## Reproducing

```bash
pytest tests/ -q                     # 5 tests: parity, cache, sampling
python examples/generate.py --verbose                # flat tok/s now
python examples/generate.py --no-cache --verbose     # the M0 baseline, for contrast
python examples/generate.py --temperature 0.7 --top-p 0.95   # sampling works
```

Benchmark: prefill a random context of length L into a cache, then time
20 single-token decode forwards (3 warmup), `torch.cuda.synchronize()`
around the timer.
