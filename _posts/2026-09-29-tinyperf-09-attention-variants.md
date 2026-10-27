---
layout: post
title: "Building tinyperf, chapter 9: Attention variants"
date: 2026-09-29 12:09:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/09-attention-variants/
excerpt: "Chapter 7 found two costs that grow with every token a sequence holds: the KV cache, which takes memory, and its read in every decode step, which takes time. At long contexts they decide the step and how many sequences fit, so most recent models bound attention somehow. What does each design change in the price, and where does the cost go?"
redirect_from:
  - /tinyperf/perf-modeling/2026/08/29/building-tinyperf-m24.html
  - /tinyperf/perf-modeling/2026/08/31/building-tinyperf-m38.html
  - /tinyperf/perf-modeling/2026/08/31/building-tinyperf-m40.html
  - /tinyperf/perf-modeling/2026/08/31/building-tinyperf-m41.html
  - /tinyperf/perf-modeling/2026/09/09/building-tinyperf-m58.html
  - /tinyperf/perf-modeling/2026/09/10/building-tinyperf-m59.html
  - /tinyperf/book/08-attention-variants/
---

*[Building tinyperf](/series/tinyperf/) · Part II: One model on one GPU · Code: [`tinyperf/nets/transformer.py`](https://github.com/kaix-nv/tinyperf/blob/ca60d9a/tinyperf/nets/transformer.py), `TransformerParams.kv_local_bytes` and the attention branches of `build_llm_graph`, and [`tinyperf/gemm_model.py`](https://github.com/kaix-nv/tinyperf/blob/ca60d9a/tinyperf/gemm_model.py), `estimate_linear_attention`; [`tinyperf/nets/transformer.py`](https://github.com/kaix-nv/tinyperf/blob/65196fd/tinyperf/nets/transformer.py) at a later commit, the linear layers' input GEMMs in `build_llm_graph` · Every table and the plots in this chapter come from `python3 book/scripts/ch09_variants.py`.*

Chapter 7 found two costs that grow with every token a sequence holds:
the KV cache, which takes memory, and its read in every decode step,
which takes time. At long contexts they decide the step and how many
sequences fit, so most recent models bound attention somehow. What
does each design change in the price, and where does the cost go?

The short answer: each design bounds one term and pays somewhere else.

- A **sliding window** stops a layer looking back more than W tokens;
  the full layers carry the growth.
- **Linear attention** replaces the cache with a fixed-size state, but
  the deployed kernel that builds it runs, measured in vLLM's own
  steps, at about a hundredth of the tensor cores' rate.
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
  sequences that fit in memory; chapter 7) counts it;
- where each design moves the cost;
- how close the model gets where there are measurements, and what can
  only be checked on paper.

## Six ways to keep the past

![Six rows of token cells, one per design, showing what each layer
stores and what a query reads.](/assets/tinyperf-book/ch09-designs.svg)

*Figure 9.1. What one layer stores for one sequence under each design,
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
not what is stored. Table 9.1 puts four released models' configs into
these formulas.

```
Table 9.1  What one layer keeps for one sequence, by design: 16-bit cache, fp32 state
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
rising lines, two flat.](/assets/tinyperf-book/ch09-layer-cache.svg)

*Figure 9.2. The same designs from 1k to 1M tokens. The rising lines
are parallel wherever c/r outweighs the window: a latent is the same
fraction of per-head K and V at every context. The flat lines cross
them: the 3.2 MB linear-attention state outweighs a latent layer's
cache up to about 2,800 tokens.*

One function answers "how many bytes does one sequence's cache hold?"
for every design, and chapter 7's `max_batch` calls it (docstring and
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
20). The linear state is a separate property, `lin_state_bytes`, and
`max_batch` charges every request both. The script checks that Table
9.1's formulas add up to `kv_local_bytes` for each model.

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
each. Table 9.2 prices gpt-oss-20b, 12 full and 12 windowed layers,
against a twin whose layers are all full, on a B200 at the rates fitted
to it (chapter 5), kernels issued from a CUDA graph (launches recorded
once and replayed together).

```
Table 9.2  gpt-oss-20b against its full-attention twin, one B200 at its fitted rates
             KV MB/sequence    decode step ms       prefill ms        max batch   
   context   window     twin   window     twin    window      twin  window    twin
      2048     53.5    100.7     6.95     7.15      11.9      12.2    2229    1188
      8192    204.5    402.7     7.58     8.40      44.9      50.8     586     298
     32768    808.5   1610.6    10.10    13.44     249.7     346.3     148      74
    131072   3224.4   6442.5    20.16    33.57    2165.5    3723.2      37      18
  decode steps of 32 sequences; prefill of one prompt of that length
  at 131072: one full layer's attention 1132 us, one windowed layer's 15 us (the same at every context)
```

The window is first a capacity feature: from 8k tokens the cache is
half the twin's and twice as many sequences fit. In time it halves the
slope and no more. A windowed layer's attention costs 15 µs at any
context, while the 12 full layers' attention is 13.6 ms of the 20 ms
step at 131,072 tokens; the step still nearly triples from 2k to 128k.
(The B200's decode-attention constants were measured on Qwen3.8-27B's
kernel, chapter 16; gpt-oss inherits them, not measured.)
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
`fresh_state=(phase == "prefill")`), after two input projections run as
vLLM runs them (q, k, v and the output gate z in one GEMM, the inputs of
each value head's decay gate and correction strength in another) and a
short convolution, and before a gated output projection. A calibration (a
GPU's file of fitted constants; chapter 5) can supply the efficiency and
a per-layer floor, which applies only when the chunked kernel runs.
Table 9.3 prices the hybrid against a twin in which each linear layer is
replaced by the model's own full layer.

```
Table 9.3  Qwen3.8-27B (48 linear + 16 full layers) against an all-full-attention twin, one B200
            KV + state MB/seq    decode step ms    max batch  
   context    hybrid      twin   hybrid     twin  hybrid  twin
      2048       297       545    13.49    12.96     364   200
      8192       724      2181    15.17    19.67     149    50
     32768      2436      8724    21.88    46.52      44    12
    131072      9281     34897    48.72   153.89      11     3
  one linear layer: 3.15 MB of state + 0.06 MB of convolution history = 3.21 MB, a full layer's cache at 783 tokens
  decode, 32 x 2048: the 48 linear cores 2.36 ms against 2.35 ms of attention in the layers they replace;
    their projections and gating cost 0.52 ms more than those layers'
```

The architecture is a trade. At 2,048 tokens the hybrid's decode step
costs a little more, 13.49 ms against 12.96: the 48 linear cores cost
about what the attention they replace does, 2.36 ms against 2.35, and
the linear layers' wider projections and gating cost 0.52 ms more. The
gain comes with length: at 131,072 tokens the step is 3.2 times cheaper
and 11 sequences fit
where 3 do. The full layers keep the growth, 9.1 GB of cache per
sequence at that length.

Prefill is where the constants matter. Table 9.4 prices one layer's
attention in a prefill with the linear core two ways: at the asserted
efficiency, the 0.55 in the code, set by hand and never measured; and at
the constants fitted to vLLM's own kernels on a B200 (Table 9.10,
below).

```
Table 9.4  One layer's attention in a one-prompt prefill: Qwen3.8-27B on a B200, us per layer
   tokens  linear, asserted    linear, fitted  full attention
      512               5.4             132.3             5.3
     2048              13.3             367.6            50.0
     8192              44.7            1463.5           765.0
    32768             171.3            5847.1         12203.4
   131072             678.2           23381.5        195210.5
  asserted: 0.55 of the fitted tensor rate, no floor; fitted to the kernels of Table 9.10: 0.0159 and a 130 us floor (0.0119 of the datasheet peak)
  the fitted linear core costs more than a full layer's attention below 15,685 tokens
  whole prefill, hybrid / all-attention twin: 1.31 at 512, 1.23 at 2048, 1.13 at 8192, 0.85 at 32768, 0.51 at 131072
```

![Per-layer prefill time against prompt length for full attention and
the linear core at two sets of constants.](/assets/tinyperf-book/ch09-linear-prefill.svg)

*Figure 9.3. The scaling is as designed: the linear core grows with the
prompt, full attention with its square. The level is not: with the
constants fitted to the kernels the linear core costs more than full
attention below about 16,000 tokens, and the whole hybrid prefill costs
more than its twin's below some length between 8k and 32k tokens.*

At the asserted efficiency the linear core is almost free. At the
fitted one it runs at 1.2% of the datasheet peak, and the floor
dominates short prompts: the hybrid's 512-token prefill costs 1.3
times its twin's. Above the floor both columns grow in proportion to
the prompt, so no check of scaling alone could tell them apart.

> **Field note: a scaling check that could not fail.** The first
> silicon check timed a plain PyTorch version, a loop over chunks, on
> a B200. Doubling the prompt from 2,048 to 8,192 tokens multiplied its
> time by 1.98, then 1.94: linear, as claimed. But at 0.1% of the
> datasheet peak its time was the loop, linear by construction. It said
> nothing about the level: the asserted 0.55 is 35 times the efficiency
> the kernels show.

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
chapter 7's grouped-query attention (GQA) trick with one group, so the
shared latent is read once per sequence. The prefill branch emits
`mla_k_up` and `mla_v_up` GEMMs and per-head attention instead.
Table 9.5's twin has 96 KV heads at a head dim of 160 for both K and V,
the mean of K's 192 and V's 128, so its bytes and FLOPs match per-head
K and V. The last columns split the layer over eight GPUs by tensor
parallelism (tp=8: each GPU holds an eighth of every weight matrix and
of the heads; chapter 12).

```
Table 9.5  One Kimi-K3 latent layer against per-head K and V: decode, 32 sequences, B200
             cache MB/seq    attention us, one GPU   absorb   tp=8, per GPU  
   context  latent per-head     latent     per-head      us   latent per-head
      8192     9.4      503         92         2111     141       92      276
     32768    37.7     2013        360         8403     141      359     1062
    131072   151.0     8053       1429        33568     141     1429     4208
  absorption at batch 1: 9 us; the model charges its weights once per sequence
  prefill of 8192 tokens, one layer: latent path 2.05 ms (K and V up-projections 0.15), per-head attention 1.91 ms
```

- **The cache** shrinks 53 times.
- **Decode attention** is about 23 times cheaper, not 53: the fused-attention
  price reads the latent twice, as keys (576 values) and as values
  (512).
- **Absorption** adds two small GEMMs per layer. Their weights are the
  same at any batch, so at 32 sequences they should cost close to the
  9 µs of batch 1 (not measured), not the table's 141.
- **Tensor parallelism** splits per-head K and V eight ways but not the
  latent, which every head needs. Each GPU keeps and reads all of it, so
  per GPU the attention gain falls from 23 to about 3 times. Chapter 13's
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

The twin in Table 9.6 is the same layer without its indexer, reading
every token.

```
Table 9.6  One GLM-5.3-Flash sparse-attention layer in decode: 8 sequences, one B200, us per layer
   context  full-context twin  top-2048 attention   indexer    total  indexer share
      2048                7.7                 7.7       0.0      7.7            0%
      4096               13.0                 7.7      16.0     23.7           67%
      8192               23.4                 7.7      17.3     25.0           69%
     16384               44.4                 7.7      20.1     27.8           72%
     65536              170.2                 7.7      36.4     44.1           83%
    131072              338.0                 7.7      58.2     65.9           88%
   1048576             2686.8                 7.7     363.3    371.0           98%
  the sparse layer costs less than full-context attention above 8,923 tokens
  each token also caches a 256-byte index key beside its 1,024-byte latent
  at tp=8, per GPU at 131072: attention 7.6 us, indexer 58.2 us (at tp=1: 7.7 and 58.2)
```

![Per-layer decode time against context for full-context attention, a
sparse layer, and its attention and indexer apart.](/assets/tinyperf-book/ch09-sparse-decode.svg)

*Figure 9.4. The cap and what pays for it. Attention stops at 2,048
keys, and the indexer grows with the context in its place, more slowly
than full attention did.*

The cap is not free. Just past k, the indexer's projection and two
extra launches make the layer dearer than full attention (23.7 against
13.0 µs at 4,096 tokens). It wins above 8,923 tokens and at a million
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
compressor. Table 9.7's twin attends to its whole context through the
latent in every layer: no pooling, no indexer. The model needs eight
GPUs (tp=8, and ep=8: its mixture-of-experts layers spread their experts
over the eight GPUs; chapter 13); the tp=1 column prices its prefill as
if on one.

```
Table 9.7  DeepSeek-V4-Flash against its uncompressed twin, 8 B200s (tp=8, ep=8)
             cache GB/sequence     decode ms      prefill s    max batch   indexer share
   context     V4   twin  ratio     V4    twin     V4    twin    V4  twin    tp=1   tp=8
     32768   0.25   1.66  0.152   7.29   10.41   0.30    0.59   360    54      9%    24%
    131072   0.99   6.64  0.149   7.50   21.91   1.74    6.81    91    13     21%    47%
   1048576   7.88  53.15  0.148   9.41  129.22  54.84  387.37    11     1     58%    84%
  decode steps of 8 sequences; prefill of one prompt; indexer share of that prefill at tp=1 and tp=8
```

Compression buys a cache 0.15 of the twin's, 11 million-token sequences
where one fits, and a decode step that grows 29% from 32k to 1M tokens
where the twin's grows twelvefold. The prefill shows where the cost
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
Table 9.8  One image through the Qwen3.8 vision tower, B200 at its fitted rates
  pixels  patches  tokens for the LLM  encode ms  attention share  LLM prefill of those tokens ms
     512     1024                 256       1.51              12%                             9.0
    1024     4096                1024       5.20              38%                            41.7
    2048    16384                4096      42.03              74%                           170.2
  the tower: 27 layers, 16 heads of 72, 0.458B parameters; LLM prefill after a 512-token text prompt
```

Doubling the resolution quadruples the patches and multiplies the
attention's FLOPs by 16: from 12% of the encode to 74%. The image's
larger cost is still the language model's prefill of its tokens. None
of this has been measured.

## How close is it?

**Windows, timed.** Table 9.9 times gpt-oss-20b's two kinds of attention
layer alone in vLLM's Triton kernel on an RTX A6000, inside CUDA graphs.
The decode prices use chapter 8's decode kernel price (a fixed cost per
call plus the cache at a fitted rate), fitted on another kernel and
another model's heads, so the decode cells are held out; chapter 8's
Table 8.6 prices the full layers (0.90–1.16). Ratios are predicted ÷
measured: above 1 the model reads high. A set of ratios is summarized by
its range or its typical error, the geometric mean distance from 1
(chapter 4).

```
Table 9.9  gpt-oss-20b's two kinds of attention layer, timed alone in vLLM's Triton kernel, RTX A6000
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
kernel's prefill rate was fitted to these timings, chapter 8). The
model misses the windowed layer's level, 0.56–1.41: low at 8
sequences, high at 64, consistent with the kernel's switch between
splitting each context and not (chapter 8; not profiled). It is a small
term: at 6,144 tokens a windowed layer costs 12% of a full one at 8
sequences and 3% at 64.

**The hybrid, kernel by kernel.** The linear core's two constants come
from vLLM's own prefill steps. `tools/profile_gdn_steps.py` runs
Qwen3.8-27B in bf16 on one B200 under vLLM 0.22 with PyTorch's profiler,
compiled as served but with CUDA graphs off so each kernel can be tied
to the operation that launched it, and `tools/analyze_gdn_profile.py`
sums the gated-delta operation's kernels: the short convolution, the
gating, the chunked delta rule and the state copies. The model's
`lin_core` plus `lin_conv1d` are fitted to it on one-prompt prefills;
the batched ones are held out. vLLM has three kernels for the chunked
delta rule (Triton's, FlashInfer's, and one in CuTe DSL, CUTLASS's
Python kernel language) but runs Triton's for this model, because it
looks for the key head size where this checkpoint doesn't keep it;
setting that value (`hf_overrides={"linear_key_head_dim": 128}`) brought
in the other two. The constants these replace had been fitted to the
TTFT (time to first token) of served prefills (below).

```
Table 9.10  The gated-delta op in vLLM's prefill steps, picked out kernel by kernel: Qwen3.8-27B, one B200, us per layer
  batch x tokens              Triton    model model/meas  CuTe-DSL FlashInfer  full attn
  1 x 512        fitted        137.9    137.9      1.000      80.1       87.4       16.8
  1 x 2048       fitted        394.3    383.0      0.971     198.0      232.1       88.8
  1 x 8192       fitted       1467.6   1518.2      1.034     653.9      713.1      763.0
  1 x 32768      fitted       6060.5   6059.1      1.000    2417.5     2758.8    10221.5
  8 x 2048       held out     2904.5   3031.9      1.044    1290.0     1096.3      591.8
  4 x 8192       held out     5737.6   6059.1      1.056    2212.1     1934.4     2974.5
  model: lin_core + lin_conv1d at 0.0159 of the fitted tensor rate and a 130 us floor; typical error 1.6% fitted, 5.0% held out
  of which lin_conv1d, the short convolution: 5.6 us a layer at 512 tokens, 212.0 at 32,768
  of the datasheet peak, by the model's count over the whole op: Triton 0.8-1.2%, CuTe-DSL 1.4-3.2%, FlashInfer 1.2-3.6%
  full attn: a full layer's attention op, FlashInfer's kernel (vLLM's default attention here)
  the constants fitted to TTFT before these measurements, 0.0108 and a 550 us floor: 1.44-4.05x these kernels
  all six cells, Triton over CuTe-DSL: 2.44x; over FlashInfer: 2.45x
```

The kernel choice moves the level about 2.4 times, yet even the fastest
reaches only a few percent of peak. Table 9.4's fitted column is
`lin_core` alone; Table 9.10 adds the convolution, 6 µs a layer at 512
tokens and 212 at 32,768.

**The hybrid, served.** The first B200 run forced FlashAttention-2.
Predicted with the asserted 0.55, its prefill read 27–58% low on every
cell. Three constants were then fitted to its TTFT (the kernel's
efficiency, its floor, and an engine cost of 6 ms per prefill pass);
they matched it within 6% while pricing the kernels at 1.44–4.05 times
their time. With the kernels measured, predictions were committed
before a second run in vLLM's default configuration, the one profiled
(Table 9.11; TPOT, time per output token, as in chapter 7).

```
Table 9.11  Qwen3.8-27B served by vLLM on one B200, predictions written before the run: model/measured
  batch prompt            TTFT ms model/meas   TPOT ms model/meas
      1   1024  held out     55.8      0.991     11.14      0.978
      1   4096  held out    198.0      0.926     11.15      0.980
      1  12288  held out    578.2      0.939     11.25      0.978
      4   1024  held out    201.6      0.899     11.30      0.980
      4   4096  held out    739.5      0.944     11.47      0.977
      4  12288  held out   2283.2      0.937     11.96      0.965
     16   1024  held out    750.2      0.918     12.38      0.956
     16   4096  held out   2964.7      0.929     13.23      0.933
     16  12288  held out   9178.4      0.929     15.19      0.901
  nine new cells, held out: TTFT 0.899-0.991 (typical error 7.0%), TPOT 0.901-0.980 (4.1%)
  the first run's grid widened to nine, partly: TTFT 0.746-0.968 (typical error 10.5%), TPOT 0.875-0.980 (5.2%)
  partly held out: five of those nine were in the first run, and the 6 ms engine cost was fitted on three
  today's model, its decode step measured kernel class by kernel class (chapter 16): the nine new cells TTFT 0.894-0.970 (7.3%); all eighteen TPOT 0.895-1.005 (2.0%)
  the model's TTFT = its kernels + 6 ms of engine cost per prefill (fitted end to end); against the kernels profiled (profiler on, CUDA graphs off):
  1 x 512: model kernels 28.8 ms, profiled kernels 27.2, TTFT 48.0; the profiled step's GPU idle 63% of its span
  1 x 8192: model kernels 356.1 ms, profiled kernels 362.5, TTFT 382.7; the profiled step's GPU idle 2% of its span
  8 x 2048: model kernels 685.9 ms, profiled kernels 719.6, TTFT 746.9; the profiled step's GPU idle 2% of its span
```

```
Recorded  The first B200 run (FlashAttention-2 forced, five cells): model/measured, midpoint of both orders
  recorded, the predictions written before it (the asserted 0.55): TTFT 0.42-0.73, TPOT 0.882-0.958
  recomputed, with the constants once fitted to it (0.0108, 550 us): TTFT 0.946-1.001
  recomputed, today's model: TTFT 0.613-0.917, TPOT 0.924-0.977
  measured, first run minus the default-configuration run, TTFT: 1 x 512 +8.8 ms, 1 x 2048 +4.3 ms, 1 x 8192 +35.0 ms, 8 x 2048 +17.3 ms, 32 x 2048 +37.8 ms
  the first run also used PyTorch's sampler and no DeepGEMM, on another driver and transformers version
```

Decode, as predicted then, read 0.90–0.98: the step was bytes at a
memory rate fitted on GEMMs, and no linear-attention constant entered
it. Prefill reads
0.90–0.99, about 7% low, partly in the kernels (at 8 × 2,048 the model
prices them at 685.9 ms where they took 719.6) and partly outside them:
the TTFT exceeds the profiled kernels by 20–27 ms in the cells shown,
of which the model prices 6. At 512 tokens the outside time is most of
the gap, and the profiled step's GPU sat idle 63% of the time,
consistent with a short step waiting on the host (not profiled further).

The first run was 4–38 ms slower per cell than the second. Besides
FlashAttention-2 it ran PyTorch's sampler without DeepGEMM on another
driver, none of it profiled. The fitted constants had matched it by
absorbing what is not the linear kernel: time outside the kernels, and
whatever slowed that configuration. That is the chapter's lesson in
miniature: the bytes-over-bandwidth part of a price carried over to a
new architecture unchanged, but a kernel's efficiency has to be measured
on its own kernels in the engine that runs it, not fitted to the total.

**The decode step, kernel by kernel.** Decode read low, and lower as
the batch grew. Chapter 16 profiles this GPU's served decode step kernel
by kernel. It found a GEMM the model's graph lacked: vLLM runs each
linear layer's input as two GEMMs, q, k, v with the output gate z in
one, and the inputs of each value head's decay gate and correction
strength in the other, where the graph had q-k-v and z apart and no
second. It also measured the step's constants one kernel class at a
time. With both, TPOT on these eighteen cells reads 0.895–1.005, a
typical error of 2.0% (Table 9.11's last line). The grid was in view
when they were measured, so that is *in-sample*; chapter 16 has the
held-out runs.

**The rest, on paper.** No latent, sparse or compressed model has run
here; their prices rest on the mechanisms above, untested. What can be
checked is that the presets are read correctly:

```
Check  Presets against published figures, parameters in billions
  gpt-oss-120b         116.79 against 116.80
  qwen3.8-27b           26.90 against 27.25 with its token-prediction layer (0.37), which the count leaves out
  deepseek-v4-flash    283.66 against 284.00; active 14.2 against 13, 13.2 without the two embedding tables
  qwen3.8 vision        0.458 against 0.461
  glm-5.3-flash         313.4 from config arithmetic, no published total checked
  kimi-k3              5483.4 from config arithmetic, no published total checked
```

and that the cache formula is the published one: the DeepSeek-V2 paper
gives MLA's cache as `(d_c + d_R) · l` elements per token (latent
width, rope width, layers): 576 per layer at its dims and DeepSeek-V3's,
as `kv_local_bytes` charges.

## Where it breaks

- **The latent is read twice.** Fused attention charges keys and values
  as separate reads, but in a latent layer the values are the first 512
  of the 576 key values. A kernel that loads each latent once reads
  about half what Table 9.5 charges (not measured here).
- **Absorption per sequence.** `mla_q_absorb` and `mla_v_absorb` batch
  over sequence × head, so each head's matrix is read once per
  sequence: 141 µs a layer at 32 sequences, where the weights alone
  take about 9.
- **Index keys are not stored.** The latent path of `kv_local_bytes`
  counts no index keys, so GLM-5.3's sparse layers hold 25% more cache
  than capacity charges. The compressed path counts them.
- **The skip rule in prefill** compares the causal mean, `(s + 1) / 2`,
  with k. A prompt between k and 2k tokens skips the indexer, though
  its later queries need one.
- **Linear attention** is measured for one kernel, vLLM 0.22's Triton
  chunked delta rule, on one GPU; the faster CuTe-DSL and FlashInfer
  kernels would need constants of their own. Other gated linear layers
  (GLM-5.3's, Kimi-K3's) inherit Qwen's constants on the B200 and 0.55
  elsewhere. TTFT reads about 7% low, part in the kernels and part
  outside them, where the model prices only a fixed 6 ms per prefill.
  Decode's constants are measured on one GPU's served steps (chapter
  16); elsewhere the hybrid's decode step keeps the generic prices.
- **Windows** that also keep the first few tokens (sinks) pay for them
  unpriced; gpt-oss's sinks are a logit (chapter 8) and cost nothing.
- **Compressed attention:** the pooling and V4's residual stream, four
  copies wide (hyper-connections), are priced as passes over memory,
  and context parallelism (splitting a sequence's tokens over GPUs;
  chapter 13) is refused rather than approximated.

## What you built

- `kv_local_bytes` and `lin_state_bytes`: what every design stores.
- Windowed layers as attention over `min(context, W)` keys.
- Linear attention as a recurrent decode step priced by state traffic
  and a chunked prefill priced by math, with kernel constants measured
  in the engine's own steps.
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
   as a row. How does Table 9.5's absorb column change?
3. Price latent attention reading the cache once. What does Table
   9.5's ratio become, and is latent decode attention at 96 heads still
   memory-bound?
4. Add the index keys to `kv_local_bytes` for latent models, and
   recompute GLM-5.3-Flash's largest batch at 128k tokens.
5. Context parallelism can split the indexer by giving each GPU a
   share of the queries. Sketch it for V4-Flash's prefill and
   estimate a million-token TTFT on 8, 16 and 32 GPUs.

---

*[← Chapter 8: Attention]({% post_url 2026-09-29-tinyperf-08-attention %}) · [Contents](/series/tinyperf/) · [Chapter 10: Mixture of experts →]({% post_url 2026-09-29-tinyperf-10-mixture-of-experts %})*
