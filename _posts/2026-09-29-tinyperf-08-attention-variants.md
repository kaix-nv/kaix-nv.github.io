---
layout: post
title: "Building tinyperf, chapter 8: Attention variants"
date: 2026-09-29 12:08:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/08-attention-variants/
excerpt: "Chapter 6 found two costs that grow with every token a sequence holds: the KV cache, which takes memory, and its read in every decode step, which takes time. At long contexts they decide the step and how many sequences fit, so most recent models bound attention somehow. What does each design change in the price, and where does the cost go?"
redirect_from:
  - /tinyperf/perf-modeling/2026/08/29/building-tinyperf-m24.html
  - /tinyperf/perf-modeling/2026/08/31/building-tinyperf-m38.html
  - /tinyperf/perf-modeling/2026/08/31/building-tinyperf-m40.html
  - /tinyperf/perf-modeling/2026/08/31/building-tinyperf-m41.html
  - /tinyperf/perf-modeling/2026/09/09/building-tinyperf-m58.html
  - /tinyperf/perf-modeling/2026/09/10/building-tinyperf-m59.html
---

*[Building tinyperf](/series/tinyperf/) · Part II: One model on one GPU · Code: [`tinyperf/nets/transformer.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/nets/transformer.py), `TransformerParams.kv_local_bytes` and the attention branches of `build_llm_graph`, and [`tinyperf/gemm_model.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/gemm_model.py), `estimate_linear_attention` · Every table and the plots in this chapter come from `python3 book/scripts/ch08_variants.py`.*

Chapter 6 found two costs that grow with every token a sequence holds:
the KV cache, which takes memory, and its read in every decode step,
which takes time. At long contexts they decide the step and how many
sequences fit, so most recent models bound attention somehow. What
does each design change in the price, and where does the cost go?

The short answer: each design bounds one term and pays somewhere else.

- A **sliding window** stops a layer looking back more than W tokens;
  the full layers carry the growth.
- **Linear attention** replaces the cache with a fixed-size state, but
  the deployed kernel that builds it runs, by the constants fitted to
  it, at about a hundredth of the tensor cores' rate.
- A **latent cache** (from multi-head latent attention, MLA) stores one
  small vector per token, 53 times smaller than per-head keys and values
  here, but it cannot be split across GPUs by heads.
- **Sparse attention** reads only the top k tokens, and an indexer that
  scores every token takes over the growth.
- **Compressed attention** pools every r tokens into one entry.

No released design stops the growth outright; each changes the slope.

By the end of this chapter you will know:

- what one layer of each design stores, and the formula for it;
- how the builder prices each design, and how capacity (the most
  sequences that fit in memory; chapter 6) counts it;
- where each design moves the cost;
- how close the model gets where there are measurements, and what can
  only be checked on paper.

## Six ways to keep the past

![Six rows of token cells, one per design, showing what each layer
stores and what a query reads.](/assets/tinyperf-book/ch08-designs.svg)

*Figure 8.1. What one layer stores for one sequence under each design,
and what a query reads. Only the window and linear attention store an
amount that stops growing with the context.*

For one sequence of c tokens, one layer stores:

```
full attention     2 · kv_heads · head_dim · bytes · c
sliding window     2 · kv_heads · head_dim · bytes · min(c, W)
linear attention   v_heads · k_dim · v_dim · 4                    fp32 state, whatever c is
                   + (2·k_heads + v_heads) · k_dim · (conv − 1) · bytes   a short convolution's history
latent (MLA)       (latent + rope) · bytes · c
compressed         (latent + rope) · bytes · (min(c, W) + c / r),  plus an index key per entry if sparse
```

DeepSeek-V4's compressed layers use r = 4 (compressed sparse, CSA) or
128 (heavily compressed, HCA). Sparse attention changes what is read,
not what is stored. Table 8.1 puts four released models' configs into
these formulas.

```
Table 8.1  What one layer keeps for one sequence, by design: 16-bit cache, fp32 state
  design            model               bytes/token  MB at 8k     128k        1M
  full attention    gpt-oss-20b               2,048     16.78   268.44  2,147.48
  window of 128     gpt-oss-20b                   0      0.26     0.26      0.26
  full attention    qwen3.8-27b               4,096     33.55   536.87  4,294.97
  linear attention  qwen3.8-27b                   0      3.21     3.21      3.21
  per-head K and V  kimi-k3                  61,440    503.32 8,053.06 64,424.51
  latent            kimi-k3                   1,152      9.44   150.99  1,207.96
  4:1 sparse (CSA)  deepseek-v4-flash           352      3.03    46.28    369.25
  128:1 dense (HCA) deepseek-v4-flash             9      0.22     1.33      9.58
```

![Megabytes per layer per sequence against context, log-log: five
rising lines, two flat.](/assets/tinyperf-book/ch08-layer-cache.svg)

*Figure 8.2. The same designs from 1k to 1M tokens. The rising lines
are parallel wherever c/r outweighs the window: a latent is the same
fraction of per-head K and V at every context. The flat lines cross
them: the 3.2 MB linear-attention state outweighs a latent layer's
cache up to about 2,800 tokens.*

One function answers "how many bytes does one sequence's cache hold?"
for every design, and chapter 6's `max_batch` calls it (docstring and
comments trimmed):

```python
    def kv_local_bytes(self, context_len: int, tp: int = 1,
                       kv_dtype_bytes: float | None = None) -> float:
        nkv_local = max(1, self.n_kv_heads // tp)
        b = kv_dtype_bytes if kv_dtype_bytes is not None else self.dtype.nbytes
        if self.is_compressed:
            lat_entries, idx_entries = self._compressed_kv_tokens(context_len)
            return (lat_entries * (self.kv_lora_rank + self.qk_rope_head_dim)
                    + idx_entries * self.dsa_index_dim) * b
        if self.is_mla:
            per_layer_tok = (self.kv_lora_rank + self.qk_rope_head_dim) * b
        else:
            per_layer_tok = 2 * nkv_local * self.head_dim * b
        full = (self.n_fullctx_layers + self.mtp_layers) * context_len
        swa = self.n_swa_layers * min(context_len, self.sliding_window or context_len)
        return per_layer_tok * (full + swa)
```

`kv_lora_rank` is the latent's width and `qk_rope_head_dim` the
position part every head shares (both explained under latent caches);
`dsa_index_dim` is the width of a sparse layer's index key.
`_compressed_kv_tokens` counts entries by the compressed formula;
`mtp_layers` is a token-prediction head with its own cache (chapter
19). The linear state is a separate property, `lin_state_bytes`, and
`max_batch` charges every request both. The script checks that Table
8.1's formulas add up to `kv_local_bytes` for each model.

## Sliding windows

A windowed layer's query attends only to the last W tokens, so an
engine can drop older keys. gpt-oss alternates full layers and windowed
ones with W = 128. The builder gives the windowed layers (`L_swa`) their
own attention with the length capped:

```python
    L_ctx, L_swa = p.n_fullctx_layers, p.n_swa_layers
    kv_swa = min(kv, p.sliding_window) if p.sliding_window else kv
    ...
    if L_swa and not skip_gqa:
        sw_scores = g.BatchedMatMul(
            "attn_qk_swa", q, batch=b_local * nkv_l, m=group * s, n=kv_swa,
            k=hd, count=L_swa, out_dtype=DType.FP32, q_len=s,
        )
```

(The softmax and second matmul follow.) In prefill `kv` is the causal
mean, `(s + 1) / 2`, so a long prompt's windowed queries read W keys
each. Table 8.2 prices gpt-oss-20b, 12 full and 12 windowed layers,
against a twin whose layers are all full, on a B200 at the rates fitted
to it (chapter 4), kernels issued from a CUDA graph (launches recorded
once and replayed together).

```
Table 8.2  gpt-oss-20b against its full-attention twin, one B200 at its fitted rates
             KV MB/sequence    decode step ms       prefill ms        max batch   
   context   window     twin   window     twin    window      twin  window    twin
      2048     53.5    100.7     7.09     7.33      12.2      12.5    2229    1188
      8192    204.5    402.7     7.85     8.84      45.3      51.1     586     298
     32768    808.5   1610.6    10.87    14.88     250.1     346.7     148      74
    131072   3224.4   6442.5    22.95    39.04    2165.9    3723.5      37      18
  decode steps of 32 sequences; prefill of one prompt of that length
  at 131072: one full layer's attention 1346 us, one windowed layer's 5 us (the same at every context)
```

The window is first a capacity feature: from 8k tokens the cache is
half the twin's and twice as many sequences fit. In time it halves the
slope and no more. A windowed layer's attention costs 5 µs at any
context, while the 12 full layers' attention is 16.2 ms of the 23 ms
step at 131,072 tokens; the step still triples from 2k to 128k.
Prefill saves 2% at 2,048 tokens, where GEMMs dominate, and 42% at
131,072.

## Linear attention: a state instead of a cache

Linear attention keeps no per-token keys and values. The variant priced
here, the *gated delta rule*, keeps for each value head a state matrix
S of `k_dim × v_dim`. Each token decays S by a learned gate, writes a
rank-one correction that moves S toward mapping its key to its value,
and reads S with its query. Qwen3.8-27B, a hybrid, has 16 full layers
(24 query heads, 4 KV heads of 256) and 48 gated-delta layers with 48
value heads of 128 × 128: 3.15 MB of state plus 0.06 MB of convolution
history, 3.21 MB per layer per sequence.

The two phases run different kernels:

- **Decode** is a recurrent step: read the state, update it, write it
  back, `4 · k_dim · v_dim` FLOPs per head. Its cost is `2 × state` of
  traffic, whatever the context.
- **Prefill** runs a chunked form: inside a chunk of C = 64 tokens the
  work is quadratic and suits tensor cores; the state carries the rest
  across chunks. Per head that is `2·n·C·(k_dim + v_dim) +
  4·n·k_dim·v_dim` FLOPs for n tokens: linear in n.

`estimate_linear_attention` prices both. Trimmed: the signature (the
device, the shapes, `math_efficiency`, `fresh_state`, `floor_us`), the
docstring, some comments, the q/k/v bytes (`stream`) and the result
bookkeeping:

```python
LIN_CHUNK = 64
LIN_MATH_EFFICIENCY = 0.55   # more elementwise/gating than a dense GEMM
    ...
    eff = math_efficiency if math_efficiency is not None else LIN_MATH_EFFICIENCY
    state_bytes = batch * v_heads * k_dim * v_dim * state_dtype.nbytes
    n = tokens_per_seq
    if n <= 1:
        # recurrent step: state in, state out, tiny projections
        flops = batch * v_heads * 4.0 * k_dim * v_dim
        dram_bytes = 2 * state_bytes + ...
    else:
        c = min(LIN_CHUNK, n)
        per_head = 2.0 * n * c * (k_dim + v_dim) + 4.0 * n * k_dim * v_dim
        flops = batch * v_heads * per_head
        dram_bytes = stream + (1 if fresh_state else 2) * state_bytes
    math_s = flops / (device.peak_tflops(dtype) * 1e12 * eff)
    dram_s = device.dram_time_s(dram_bytes)
    floor_s = (floor_us * 1e-6) if n >= LIN_CHUNK else 0.0
    ...  max(math_s, dram_s, floor_s) * 1e6 + device.kernel_launch_us
```

The builder emits one such op per linear layer (`tokens_per_seq=s`,
`fresh_state=(phase == "prefill")`), between a fused QKV projection with
a short convolution and a gated output projection. A calibration (a
GPU's file of fitted constants; chapter 4) can supply the efficiency and
a per-layer floor, which applies only when the chunked kernel runs.
Table 8.3 prices the hybrid against a twin in which each linear layer is
replaced by the model's own full layer.

```
Table 8.3  Qwen3.8-27B (48 linear + 16 full layers) against an all-full-attention twin, one B200
            KV + state MB/seq    decode step ms    max batch  
   context    hybrid      twin   hybrid     twin  hybrid  twin
      2048       297       545    13.17    13.17     364   200
      8192       724      2181    15.19    21.22     149    50
     32768      2436      8724    23.24    53.43      44    12
    131072      9281     34897    55.45   182.28      11     3
  one linear layer: 3.15 MB of state + 0.06 MB of convolution history = 3.21 MB, a full layer's cache at 783 tokens
  decode, 32 x 2048: the 48 linear cores 1.69 ms against 2.19 ms of attention in the layers they replace;
    their projections and gating cost 0.51 ms more than those layers'
```

The architecture is a trade. At 2,048 tokens the two decode steps cost
the same, 13.17 ms: the 48 linear cores cost 1.69 ms against the 2.19
ms of attention they replace, and the linear layers' wider projections
and gating give the 0.51 ms back. The gain comes with length: at
131,072 tokens the step is 3.3 times cheaper and 11 sequences fit
where 3 do. The full layers keep the growth, 9.1 GB of cache per
sequence at that length.

Prefill is where the constants matter. Table 8.4 prices one layer's
attention in a prefill with the linear core two ways: at the asserted
efficiency, the 0.55 in the code, set by hand and never measured; and at
the constants fitted to vLLM's TTFT (time to first token) on a B200
(below).

```
Table 8.4  One layer's attention in a one-prompt prefill: Qwen3.8-27B on a B200, us per layer
   tokens  linear, asserted    linear, fitted  full attention
      512               6.6             553.5             6.5
     2048              14.5             553.5            51.2
     8192              45.9            2154.7           766.2
    32768             172.5            8608.3         12204.6
   131072             679.4           34422.8        195211.7
  asserted: 0.55 of the fitted tensor rate, no floor; fitted: 0.0108 and a 550 us floor (0.0081 of the datasheet peak)
  the fitted linear core costs more than a full layer's attention below 23,093 tokens
  whole prefill, hybrid / all-attention twin: 2.21 at 512, 1.35 at 2048, 1.24 at 8192, 0.92 at 32768, 0.54 at 131072
```

![Per-layer prefill time against prompt length for full attention and
the linear core at two sets of constants.](/assets/tinyperf-book/ch08-linear-prefill.svg)

*Figure 8.3. The scaling is as designed: the linear core grows with
the prompt, full attention with its square. The level is not: with the
fitted constants the linear core costs more than full attention below
about 23,000 tokens, and the whole hybrid prefill costs more than its
twin's below some length between 8k and 32k tokens.*

At the asserted efficiency the linear core is almost free. At the
fitted one it runs at 0.8% of the datasheet peak, and the floor
dominates short prompts: the hybrid's 512-token prefill costs 2.2
times its twin's. Above the floor both columns grow in proportion to
the prompt, so no check of scaling alone could tell them apart.

> **Field note: a scaling check that could not fail.** The first
> silicon check timed a plain PyTorch version, a loop over chunks, on
> a B200. Doubling the prompt from 2,048 to 8,192 tokens multiplied its
> time by 1.98, then 1.94: linear, as claimed. But at 0.1% of the
> datasheet peak its time was the loop, linear by construction. It said
> nothing about the level, where the asserted 0.55 was 51 times too
> high.

## Latent caches

Multi-head latent attention (MLA) projects each token's hidden state
down to a latent vector c of 512 values, plus a 64-value key part that
carries the position (rope, for rotary position embedding), shared by
every head. The cache holds only these 576 values per token per layer.
Per-head keys and values, `K_h = W_uk,h · c` and `V_h = W_uv,h · c`, are
rebuilt in one of two ways:

- **Materialize** (prefill): up-project every latent to per-head K and
  V and run ordinary attention. With thousands of queries per sequence,
  the up-projections are cheap next to the attention.
- **Absorb** (decode): since `q_h · (W_uk,h c) = (W_uk,hᵀ q_h) · c`,
  fold `W_uk` into the query, and `W_uv` into the output. Each head's
  query becomes a 512-value vector that scores the latents directly, so
  attention runs on the shared latent itself: multi-query attention
  (every query head sharing one key and value head) with a head dim of
  576 for the scores and 512 for the values.

The saving, derived from the config:

```
Worked example  Kimi-K3's latent against per-head K and V, values per token per layer
  per-head K and V: 96 heads x (128 + 64 + 128) = 30,720
  latent: 512 + 64 = 576: 53.3x smaller
  per-head at head dim 128 for both K and V (24,576): 42.7x
  DeepSeek-V3's published config: 40,960 / 576 = 71.1x; 61 layers x 576 x 2 bytes = 70,272 bytes per token
```

The absorbed decode path in the builder (comments trimmed):

```python
        else:
            qn = Tensor("mla_qn", (tokens, nh_l * nope), dt)
            g.BatchedMatMul("mla_q_absorb", qn, batch=b_local * nh_l, m=s,
                            n=lat, k=nope, count=L_full)
            qa = Tensor("mla_qa", (tokens, nh_l * (lat + rope_d)), dt)
            scores = g.BatchedMatMul(
                "attn_qk", qa, batch=b_local, m=nh_l * s, n=kv_eff,
                k=lat + rope_d, count=L_full, out_dtype=DType.FP32)
            probs = g.Softmax("attn_softmax", scores, count=L_full, out_dtype=dt)
            g.BatchedMatMul("attn_pv", probs, batch=b_local, m=nh_l * s,
                            n=lat, k=kv_eff, count=L_full)
            av = Tensor("mla_av", (tokens, nh_l * lat), dt)
            g.BatchedMatMul("mla_v_absorb", av, batch=b_local * nh_l, m=s,
                            n=vd, k=lat, count=L_full)
```

Attention batches over sequences with every head stacked as rows,
chapter 6's grouped-query attention (GQA) trick with one group, so the
shared latent is read once per sequence. The prefill branch emits
`mla_k_up` and `mla_v_up` GEMMs and per-head attention instead.
Table 8.5's twin has 96 KV heads at a head dim of 160 for both K and V,
the mean of K's 192 and V's 128, so its bytes and FLOPs match per-head
K and V. The last columns split the layer over eight GPUs by tensor
parallelism (tp=8: each GPU holds an eighth of every weight matrix and
of the heads; chapter 11).

```
Table 8.5  One Kimi-K3 latent layer against per-head K and V: decode, 32 sequences, B200
             cache MB/seq    attention us, one GPU   absorb   tp=8, per GPU  
   context  latent per-head     latent     per-head      us   latent per-head
      8192     9.4      503         94         2520     143       93      318
     32768    37.7     2013        361        10070     143      360     1262
    131072   151.0     8053       1431        40269     143     1430     5037
  absorption at batch 1: 11 us; the model charges its weights once per sequence
  prefill of 8192 tokens, one layer: latent path 2.06 ms (K and V up-projections 0.15), per-head attention 1.91 ms
```

- **The cache** shrinks 53 times.
- **Decode attention** is 28 times cheaper, not 53: the fused-attention
  price reads the latent twice, as keys (576 values) and as values
  (512).
- **Absorption** adds two small GEMMs per layer. Their weights are the
  same at any batch, so at 32 sequences they should cost close to the
  11 µs of batch 1 (not measured), not the table's 143.
- **Tensor parallelism** splits per-head K and V eight ways but not the
  latent, which every head needs. Each GPU keeps and reads all of it, so
  per GPU the attention gain falls from 28 to 3.5 times. Chapter 12's
  data-parallel attention, which gives each GPU whole sequences
  instead, avoids that.

## Sparse attention: an indexer picks the tokens

DeepSeek sparse attention (DSA) caps what attention reads. For each
query, a small *indexer* scores every cached token and keeps the top k
for the main attention. GLM-5.3-Flash uses it on its latent layers,
with 32 index heads of 128 and k = 2,048; each token also caches one
128-value index key. Per query per layer, attention costs as if the
context were `min(c, k)`, while the indexer costs about `2 ·
index_heads · index_dim · c` FLOPs. In the builder (comments trimmed):

```python
        kv_eff = min(kv, p.dsa_topk) if p.dsa_topk else kv
        ...
        if p.dsa_index_heads and (not p.dsa_topk or kv > p.dsa_topk):
            g.Linear("dsa_index_proj", y,
                     out_features=p.dsa_index_heads * p.dsa_index_dim,
                     count=L_full)
            iq = Tensor("dsa_iq", (tokens, p.dsa_index_heads * p.dsa_index_dim), dt)
            g.FusedAttention("dsa_index_scores", iq, batch=b_local,
                             m=s * p.dsa_index_heads, kv=kv, k=p.dsa_index_dim, out_dim=1,
                             count=L_full)
            g.Elementwise("dsa_index_topk", Tensor("dsa_logits", (tokens, kv), DType.FP32),
                          count=L_full)
```

Three decisions are in them.

- **Skip when there is nothing to select.** If the context is no
  longer than k, the top k is everything, and an engine that knows it
  runs no indexer.
- **One fused streaming op.** The index heads stack as rows against
  one shared key per token, so the keys are read once, and the per-head
  scores never reach memory. What lands is one fp32 score per (query,
  key), summed over the heads, which the top-k pass reads back.
- **Not divided by tp.** `m` uses all of `p.dsa_index_heads`. A GPU
  could score with a share of the index heads, but the top k needs each
  key's score summed over all of them: a tensor the size of the scores
  would have to cross the GPUs. So every GPU runs the whole indexer.

The twin in Table 8.6 is the same layer without its indexer, reading
every token.

```
Table 8.6  One GLM-5.3-Flash sparse-attention layer in decode: 8 sequences, one B200, us per layer
   context  full-context twin  top-2048 attention   indexer    total  indexer share
      2048                8.9                 8.9       0.0      8.9            0%
      4096               14.1                 8.9      17.1     26.1           66%
      8192               24.6                 8.9      18.5     27.4           68%
     16384               45.6                 8.9      21.2     30.1           70%
     65536              171.4                 8.9      37.6     46.5           81%
    131072              339.2                 8.9      59.4     68.3           87%
   1048576             2688.0                 8.9     364.4    373.3           98%
  the sparse layer costs less than full-context attention above 9,443 tokens
  each token also caches a 256-byte index key beside its 1,024-byte latent
  at tp=8, per GPU at 131072: attention 8.8 us, indexer 59.4 us (at tp=1: 8.9 and 59.4)
```

![Per-layer decode time against context for full-context attention, a
sparse layer, and its attention and indexer apart.](/assets/tinyperf-book/ch08-sparse-decode.svg)

*Figure 8.4. The cap and what pays for it. Attention stops at 2,048
keys, and the indexer grows with the context in its place, more slowly
than full attention did.*

The cap is not free. Just past k, the indexer's projection and two
extra launches make the layer dearer than full attention (26.1 against
14.1 µs at 4,096 tokens). It wins above 9,443 tokens and at a million
costs a seventh, nearly all of it indexer, which tp does not shrink.

## Compressed attention

DeepSeek-V4-class models compress the cache along the sequence, with a
ratio per layer. A CSA layer pools every 4 tokens' latents into one
entry (a learned, softmax-weighted pool), indexes the pooled entries as
above and attends to the top 512. An HCA layer pools 128 tokens into one
and attends to all the pooled entries. Every layer also keeps its last
128 tokens raw, as a window. V4-Flash has 20 CSA, 20 HCA and 3
window-only layers, with multi-query attention on a 512-value latent
plus 64 of rope. The builder gives each layer class one attention op
over the window plus its pooled entries, `min(k, c/r)` of them for CSA
and `c/r` for HCA, after a projection and a pooling pass for the
compressor. Table 8.7's twin attends to its whole context through the
latent in every layer: no pooling, no indexer. The model needs eight
GPUs (tp=8, and ep=8: its mixture-of-experts layers spread their experts
over the eight GPUs; chapter 12); the tp=1 column prices its prefill as
if on one.

```
Table 8.7  DeepSeek-V4-Flash against its uncompressed twin, 8 B200s (tp=8, ep=8)
             cache GB/sequence     decode ms      prefill s    max batch   indexer share
   context     V4   twin  ratio     V4    twin     V4    twin    V4  twin    tp=1   tp=8
     32768   0.25   1.66  0.152   8.83   11.75   0.30    0.59   360    54      9%    24%
    131072   0.99   6.64  0.149   9.03   23.25   1.74    6.81    91    13     21%    47%
   1048576   7.88  53.15  0.148  10.95  130.56  54.84  387.37    11     1     58%    84%
  decode steps of 8 sequences; prefill of one prompt; indexer share of that prefill at tp=1 and tp=8
```

Compression buys a cache 0.15 of the twin's, 11 million-token sequences
where one fits, and a decode step that grows 24% from 32k to 1M tokens
where the twin's grows elevenfold. The prefill shows where the cost
went. The indexer scores every pooled key for every query, so its work
grows with the square of the prompt, and it is not split: at a million
tokens it is 58% of the prefill priced as if on one GPU and 84% on
eight, where the rest is split eight ways.

## Sidebar: an image encoder

A vision-language model first runs each image through a vision
transformer. Qwen3.8's tower cuts an image into 16-pixel patches, runs
27 layers over them and merges each 2 × 2 group into one token for the
language model. Its attention is *bidirectional*: every patch attends
to every patch of its image, so `build_vit_graph`'s score GEMM is
square (`m = n = patches`), with no causal halving.

```
Table 8.8  One image through the Qwen3.8 vision tower, B200 at its fitted rates
  pixels  patches  tokens for the LLM  encode ms  attention share  LLM prefill of those tokens ms
     512     1024                 256       1.80              12%                             8.0
    1024     4096                1024       5.49              37%                            34.2
    2048    16384                4096      42.32              73%                           167.8
  the tower: 27 layers, 16 heads of 72, 0.458B parameters; LLM prefill after a 512-token text prompt
```

Doubling the resolution quadruples the patches and multiplies the
attention's FLOPs by 16: from 12% of the encode to 73%. The image's
larger cost is still the language model's prefill of its tokens. None
of this has been measured.

## How close is it?

**Windows, timed.** Table 8.9 times gpt-oss-20b's two kinds of attention
layer alone in vLLM's Triton kernel on an RTX A6000, inside CUDA graphs.
The decode prices use chapter 7's decode kernel price (a fixed cost per
call plus the cache at a fitted rate), fitted on another kernel and
another model's heads, so the decode cells are held out; chapter 7's
Table 7.6 prices the full layers (0.90–1.16). Ratios are predicted ÷
measured: above 1 the model reads high. A set of ratios is summarized by
its range or its typical error, the geometric mean distance from 1
(chapter 3).

```
Table 8.9  gpt-oss-20b's two kinds of attention layer, timed alone in vLLM's Triton kernel, RTX A6000
  decode, one layer, us      context     512    2048    6144
    full layer, 8 seqs    measured      22.1    60.6   163.0
    full layer, 64 seqs   measured     102.1   387.6  1150.2
    window layer, 8 seqs  measured      14.0    19.7    19.7
                          model/meas    0.99    0.70    0.71
    window layer, 64 seqs measured      31.1    30.8    32.3
                          model/meas    1.20    1.21    1.16
    window layers, all 24 cells (8-64 seqs, context 512-6144): model/measured 0.56-1.41
  prefill, one prompt, us     tokens    2048    4096    8192   growth per doubling
    full layer            measured      1323    5094   20048   x3.85, x3.94  (model x3.99, x4.00)
    window layer          measured       235     502    1059   x2.14, x2.11  (model x1.98, x1.99)
```

The mechanism holds. A windowed layer's decode time is flat from 2,048
tokens (a little lower below, not profiled) while a full layer's grows.
In prefill, full layers grow 3.9 times per doubling and windowed ones
2.1, against the model's 4.0 and 2.0 (the levels are in-sample: this
kernel's prefill rate was fitted to these timings, chapter 7). The
model misses the windowed layer's level, 0.56–1.41: low at 8
sequences, high at 64, consistent with the kernel's switch between
splitting each context and not (chapter 7; not profiled). It is a small
term: at 6,144 tokens a windowed layer costs 12% of a full one at 8
sequences and 3% at 64.

**The hybrid, served.** Qwen3.8-27B in bf16 on one B200 under vLLM, five
cells, each timed in forward and reverse cell order (the table uses the
midpoint). TTFT and TPOT (time per output token) are as in chapter 6.

```
Table 8.10  Qwen3.8-27B served by vLLM on one B200: model/measured, calibrated model
  batch prompt             TTFT ms model/meas  without the three constants   TPOT ms model/meas
      1    512  fitted        56.7      0.988                        0.419     11.41      0.954
      1   2048  fitted       106.0      1.012                        0.711     11.38      0.958
      1   8192  held out     417.7      0.946                        0.689     11.46      0.957
      8   2048  fitted       764.3      0.992                        0.719     12.09      0.944
     32   2048  held out    2995.0      1.000                        0.728     15.03      0.878
  the three: the gated-delta prefill kernel at 0.0108 of the fitted tensor rate, a 550 us floor per layer, 6 ms per prefill pass
  TPOT, nothing fitted to this grid: 0.878-0.958, typical error 6.6%
```

```
Recorded  Predictions committed before the B200 measurement, model/measured
  TTFT: the column without the three constants, cell for cell; TPOT: 0.882-0.958
```

Decode passed, 0.88–0.96 on every cell with nothing fitted to it: a
decode step here is bytes (the weights, the states, the caches) at a
memory rate fitted on GEMMs, and nothing specific to linear attention
enters it. Prefill failed, 27–58% low on every cell: the linear kernel
was priced at the asserted 0.55, with no floor. Three constants, now in
the B200's calibration file, fix it: the kernel's efficiency, a floor
per layer, and a fixed engine cost per prefill pass. They were fitted on
the three cells marked; the other two, held out, read 0.946 and 1.000
(the file recorded 5.3% and 0.0% at fitting). The floor's cause has not
been profiled; the model carries the fitted value.

That is the chapter's lesson in miniature: the bytes-over-bandwidth
part of a price carried over to a new architecture unchanged, and the
kernel's efficiency had to be fitted to the engine that runs it.

**The rest, on paper.** No latent, sparse or compressed model has run
here; their prices rest on the mechanisms above, untested. What can be
checked is that the presets are read correctly:

```
Check  Presets against published figures, parameters in billions
  gpt-oss-120b         116.79 against 116.80
  qwen3.8-27b           26.87 against 27.25 with its token-prediction layer (0.37), which the count leaves out
  deepseek-v4-flash    283.66 against 284.00; active 14.2 against 13, 13.2 without the two embedding tables
  qwen3.8 vision        0.458 against 0.461
  glm-5.3-flash         313.4 from config arithmetic, no published total checked
  kimi-k3              5483.3 from config arithmetic, no published total checked
```

and that the cache formula is the published one: the DeepSeek-V2 paper
gives MLA's cache as `(d_c + d_R) · l` elements per token (latent
width, rope width, layers): 576 per layer at its dims and DeepSeek-V3's,
as `kv_local_bytes` charges.

## Where it breaks

- **The latent is read twice.** Fused attention charges keys and values
  as separate reads, but in a latent layer the values are the first 512
  of the 576 key values. A kernel that loads each latent once reads
  about half what Table 8.5 charges (not measured here).
- **Absorption per sequence.** `mla_q_absorb` and `mla_v_absorb` batch
  over sequence × head, so each head's matrix is read once per
  sequence: 143 µs a layer at 32 sequences, where the weights alone
  take about 11.
- **Index keys are not stored.** The latent path of `kv_local_bytes`
  counts no index keys, so GLM-5.3's sparse layers hold 25% more cache
  than capacity charges. The compressed path counts them.
- **The skip rule in prefill** compares the causal mean, `(s + 1) / 2`,
  with k. A prompt between k and 2k tokens skips the indexer, though
  its later queries need one.
- **Linear attention** is measured for one kernel version in one engine
  on one GPU. Other gated linear layers (GLM-5.3's, Kimi-K3's) inherit
  Qwen's constants on the B200 and 0.55 elsewhere. TPOT reads 0.878 at
  32 sequences, a gap not yet explained.
- **Windows** that also keep the first few tokens (sinks) pay for them
  unpriced; gpt-oss's sinks are a logit (chapter 7) and cost nothing.
- **Compressed attention:** the pooling and V4's residual stream, four
  copies wide (hyper-connections), are priced as passes over memory,
  and context parallelism (splitting a sequence's tokens over GPUs;
  chapter 12) is refused rather than approximated.

## What you built

- `kv_local_bytes` and `lin_state_bytes`: what every design stores.
- Windowed layers as attention over `min(context, W)` keys.
- Linear attention as a recurrent decode step priced by state traffic
  and a chunked prefill priced by math, with kernel constants fitted to
  the engine.
- Latent caches, absorbed in decode and materialized in prefill.
- Sparse and compressed attention: a capped attention and an indexer
  that grows with the context, skipped when there is nothing to select
  and not split by tp.
- Evidence: windows timed on an RTX A6000, and a hybrid served on a
  B200.

## Exercises

1. Sweep gpt-oss-20b's window from 128 to 4,096 tokens and its pattern
   from 1:1 to 5:1 windowed layers. At what context does each setting
   halve the decode step against the twin?
2. Batch the absorption GEMMs over heads, with every sequence's token
   as a row. How does Table 8.5's absorb column change?
3. Price latent attention reading the cache once. What does Table
   8.5's ratio become, and is latent decode attention at 96 heads still
   memory-bound?
4. Add the index keys to `kv_local_bytes` for latent models, and
   recompute GLM-5.3-Flash's largest batch at 128k tokens.
5. Context parallelism can split the indexer by giving each GPU a
   share of the queries. Sketch it for V4-Flash's prefill and
   estimate a million-token TTFT on 8, 16 and 32 GPUs.

---

*[← Chapter 7: Attention]({% post_url 2026-09-29-tinyperf-07-attention %}) · [Contents](/series/tinyperf/) · [Chapter 9: Mixture of experts →]({% post_url 2026-09-29-tinyperf-09-mixture-of-experts %})*
