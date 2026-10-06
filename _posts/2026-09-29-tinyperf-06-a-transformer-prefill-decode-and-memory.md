---
layout: post
title: "Building tinyperf, chapter 6: A transformer: prefill, decode and memory"
date: 2026-09-29 12:06:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/06-a-transformer-prefill-decode-and-memory/
excerpt: "Chapter 3 priced one GEMM, and chapter 5 turned a list of operations into a time. This chapter writes the list for a whole language model and answers three questions. What does a forward pass cost when it reads a prompt (prefill)? What does it cost when it generates one token for each sequence in a batch (decode)? And how many requests fit on one GPU?"
redirect_from:
  - /tinyperf/perf-modeling/2026/08/18/building-tinyperf-m5.html
  - /tinyperf/perf-modeling/2026/08/18/building-tinyperf-m8.html
---

*[Building tinyperf](/series/tinyperf/) · Part II: One model on one GPU · Code: [`tinyperf/nets/transformer.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/nets/transformer.py), `build_llm_graph`, and [`tinyperf/capacity.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/capacity.py), `max_batch` · Every table and the plot in this chapter come from `python3 book/scripts/ch06_transformer.py`.*

Chapter 3 priced one GEMM, and chapter 5 turned a list of operations
into a time. This chapter writes the list for a whole language model
and answers three questions. What does a forward pass cost when it
reads a prompt (prefill)? What does it cost when it generates one token
for each sequence in a batch (decode)? And how many requests fit on one
GPU?

The short answer: a prefill costs about two FLOPs per parameter per
token, plus attention, which grows with the square of the prompt, and it
keeps the tensor cores busy. A decode step costs one read of the weights
plus one read of every sequence's KV cache, and it keeps the memory bus
busy. Adding sequences to a decode step is nearly free for the weights
until the batch nears the GPU's ridge point (where math, not memory,
starts to set the time; chapter 2), about 128 sequences on the GPUs
here. It is never free for the KV cache, which each sequence reads for
itself. And the requests that fit are the memory left after the weights,
divided by one request's cache.

By the end of this chapter you will know:

- how a model becomes a handful of integers, and how its parameters,
  FLOPs and bytes follow from them;
- the ops of one decoder layer, and the shapes prefill and decode give
  them;
- how big the KV cache is, and what grouped-query attention saves;
- where decode batching stops being free, and why above about 800
  tokens of context that is the KV cache, not the math;
- how many sequences fit on a GPU at a given context;
- how close the model gets to vLLM serving Qwen3-8B on an RTX A6000:
  within 5% on every cell of a batch-by-prompt grid.

## A model is a handful of integers

To a performance model, a dense transformer is its shape. tinyperf
describes one with a dataclass, shown here with only the fields a dense
model uses (the rest describe the attention variants of chapter 8 and
the experts of chapter 9):

```python
@dataclass
class TransformerParams:
    name: str
    hidden: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden: int
    vocab: int
    dtype: DType = DType.FP16
    ...
    head_dim_override: int = 0
    ...
    tied_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.head_dim_override or self.hidden // self.n_heads
```

A preset types in the numbers from the model's released config file
(docstring trimmed):

```python
def qwen3_8b() -> TransformerParams:
    return TransformerParams(
        "qwen3-8b", hidden=4096, n_layers=36, n_heads=32, n_kv_heads=8,
        ffn_hidden=12288, vocab=151936, head_dim_override=128,
    )
```

The parameter count follows from these. Each layer has four attention
projections (queries, keys, values and the output; keys and values may
have fewer heads than queries, below) and a SwiGLU feed-forward network
of three matrices (gate, up and down). The model adds an embedding
table and an LM head (the output projection that scores every token in
the vocabulary), each `vocab × hidden`. For a dense model,
`attn_param_count` and `param_count` compute:

```
attention  = hidden · (heads + 2 · kv_heads) · head_dim  +  heads · head_dim · hidden
ffn        = 3 · hidden · ffn_hidden
parameters = layers · (attention + ffn)  +  2 · vocab · hidden
```

The norms' weights are left out: a few million at most, under 0.01%.
Table 6.1 checks the arithmetic against published counts.

```
Table 6.1  Dense models from their released configs: parameters in billions
  model        layers hidden heads q/kv head dim    FFN   vocab stored public non-emb public
  llama2-7b        32   4096      32/32      128  11008   32000   6.74   6.74    6.48      -
  llama2-13b       40   5120      40/40      128  13824   32000  13.02  13.02   12.69      -
  llama3-70b       80   8192       64/8      128  28672  128256  70.55  70.55   68.45      -
  qwen3-0.6b       28   1024       16/8      128   3072  151936   0.60   0.60    0.44   0.44
  qwen3-4b         36   2560       32/8      128   9728  151936   4.02   4.00    3.63   3.60
  qwen3-8b         36   4096       32/8      128  12288  151936   8.19   8.20    6.95   6.95
  qwen3-14b        40   5120       40/8      128  17408  151936  14.77  14.80   13.21  13.20
  qwen3-4b ties its LM head to its embedding: counting both tables gives 4.41
```

The public columns are the parameter counts Hugging Face reports for
the released Llama checkpoints, and the Qwen3 model cards, which also
give the count without embeddings. Every row agrees to the precision
the public figure is given in. The "stored" column counts what sits in
memory. The two small Qwen3 models *tie* their LM head to the
embedding, one matrix serving both, so it is stored once; the formula
above, like `param_count`, counts two tables.

The embedding split matters more than it looks. Qwen3-8B's vocabulary
of 151,936 tokens makes its two tables 1.24 billion parameters, 15% of
the model, and the embedding does no math: it is a lookup.

## One layer, op by op

![One decoder layer: RMSNorm, QKV projection, RoPE, attention reading the
KV cache, output projection, residual; then RMSNorm, gate-up projection,
SwiGLU, down projection, residual.](/assets/tinyperf-book/ch06-layer.svg)

*Figure 6.1. A decoder layer as the model builds it. Blue boxes are
GEMMs, grey boxes are memory-bound operations, orange is attention and
its cache. Every GEMM runs T rows, one per token in the step: the whole
prompt in prefill, one token per sequence in decode.*

`build_llm_graph(p, phase, batch, seq_len)` turns this layer into a
graph (chapter 5): it emits one operator per box, each marked
`count=36` so the scheduler prices one layer and multiplies, then the
head's operators. This chapter needs only where the shapes come from.
The function also builds chapter 8's attention variants, chapter 9's
experts and the parallel layouts of chapters 11 and 12, so we quote only
the lines we explain:

```python
    nh_l = p.n_heads // tp                       # local query heads
    nkv_l = max(1, p.n_kv_heads // tp)           # local KV heads
    group = nh_l // nkv_l                        # query heads per KV head
    ffn_l = p.ffn_hidden // tp

    # tokens processed this step, and effective KV length seen by attention
    s_full = seq_len if phase == "prefill" else chunk
    if phase == "prefill":
        kv = math.ceil((seq_len + 1) / 2)
        s = -(-s_full // cp)                     # this rank's token shard
    else:
        kv = seq_len + (math.ceil((chunk + 1) / 2) if chunk > 1 else 0)
        kv = -(-kv // cp)                        # this rank's KV shard
        s = s_full                               # queries replicated over cp
    ...
    tokens = b_local * s                         # rows on this rank
```

On one GPU the parallel degrees `tp`, `cp` and `dp` (how many GPUs
split the weights, each sequence's tokens and the batch; chapters 11
and 12) are 1 and `b_local` is the batch; `chunk` is 1 in an ordinary
decode step (chapter 14's chunked prefill, which feeds a prompt into
the steps a chunk at a time, uses more). Two cases remain:

- **Prefill** of a prompt of s tokens: `tokens = batch · s`. Attention
  is causal, so token i attends to the i tokens up to itself, `(s+1)/2`
  on average. The model prices that average instead of a mask.
- **Decode**: `tokens = batch`, one new token per sequence, and `kv`
  is the context: every token already in the sequence.

`tokens` is the row count M of every GEMM in the layer. `kv` is the
length of the cache attention reads; the cache itself never appears as
a tensor, only as that length (below). The head adds one decision:

```python
        head_rows = b_local if (phase == "prefill" and logits == "last") else tokens
        y_head = Tensor("head_in", (head_rows, h), dt)
        g.RMSNorm("ln_final", y_head)
        ...
        head = g.Linear("lm_head", y_head, out_features=-(-p.vocab // tp))
```

A serving engine needs logits only for the next token, so in prefill the
LM head runs on one row per sequence. Training needs every position
(`logits="all"`, chapter 13). A pass (a rewrite of the graph; chapter 5)
fuses attention's three operators into one FlashAttention-style kernel
before pricing (`attn_fmha` below); chapter 7 prices it, and here we
take its price as given.

Table 6.2 prices one prompt and one decode step on an A100 at its
datasheet rates. Its bound columns name what set each price: math,
DRAM traffic or L2 traffic (chapter 5).

```
Table 6.2  One Qwen3-8B layer op by op, and the head: A100 SXM, datasheet rates
  prefill: one 2048-token prompt; decode: one sequence, context 2048
  op              prefill GFLOP      us  bound  decode MB     us  bound
  embed                     0.0    19.5  dram         0.0    3.0  dram
  ln_attn                   0.0    19.5  dram         0.0    3.0  dram
  qkv_proj                103.1   380.6  math        50.5   27.8  dram
  rope                      0.0    19.5  dram         0.0    3.0  dram
  attn_fmha                34.4   172.7  math         8.4    7.1  dram
  attn_out                 68.7   245.3  l2          33.7   19.5  dram
  residual_attn             0.0    27.7  dram         0.0    3.0  dram
  ln_ffn                    0.0    19.5  dram         0.0    3.0  dram
  ffn_gate_up             412.3  1419.2  math       201.5  101.8  dram
  swiglu                    0.0   101.7  dram         0.1    3.0  dram
  ffn_down                206.2   722.6  l2         101.1   52.6  dram
  residual_ffn              0.0    27.7  dram         0.0    3.0  dram
  ln_final                  0.0     3.0  dram         0.0    3.0  dram
  lm_head                   1.2   614.8  dram      1247.5  614.8  dram
  36 layers + head         29689  114248             15480   8789
  prefill: 114.2 ms, 260 TFLOP/s of 312; decode: 8.79 ms
  prefill with logits for every position: 122.0 ms (+7%)
```

The two columns are the same ops on opposite sides of the roofline (an
op's time as the larger of its math time and its memory time;
chapter 2). In prefill the GEMMs are bound by math, or by chapter 3's L2
traffic, and the prefill runs at 260 of the A100's 312 TFLOP/s. In
decode every op is bound by memory. A GEMM's time is its weights over
the bandwidth plus a launch, the fixed cost of starting a kernel: the
gate-up projection's 201.5 MB take 98.8 µs at 2,039 GB/s, plus 3 µs. The
small ops cost 3.0 µs each, their launch. The LM head costs the same 615
µs in both phases: in prefill it runs on one row, so reading its 1.25 GB
of weights is its whole cost. Scoring every position would make this
prefill 7% longer.

A decode step, then, is bytes over bandwidth plus launches:

```
Worked example  One decode step, batch 1, context 2048, A100: bytes over bandwidth plus launches
  weights read: 15.14 GB = 7.42 ms at 2039 GB/s
  KV cache read: 0.30 GB = 0.15 ms
  399 kernels x 3 us = 1.20 ms
  sum 8.77 ms; the model: 8.79 ms
```

The weights read are the parameters less the embedding, which a step
only looks up: 7.57 billion values of two bytes.

## Prefill: two FLOPs per parameter per token

In prefill every weight meets every token in one multiply-add, two
FLOPs. So a prompt of s tokens costs about `2 · parameters · s` FLOPs.
Two corrections make that rule accurate.

The first is which parameters. The embedding does no math, and the LM
head runs on one row per prompt, so the rule should use the
non-embedding parameters: 13.89 GFLOP per token for Qwen3-8B, where all
parameters would say 16.38. For a model with a small vocabulary, such
as Llama-2-7B, the difference is 4%; for Qwen3-8B it is 18%.

The second is attention. Token i's query meets i keys and i values in
each of the heads, `4 · heads · head_dim · i` FLOPs per layer. Summed
over a causal prompt:

```
prefill FLOPs ≈ 2 · non-embedding parameters · s  +  2 · layers · heads · head_dim · s²
```

The first term grows with the prompt; the second with its square.

```
Table 6.3  Prefill of one prompt by length: Qwen3-8B, A100 SXM, datasheet rates
  tokens  GFLOP/token: weights  attention        ms  TFLOP/s  attention share of time
     512                 13.89       0.15      31.5      228                     1.6%
    2048                 13.89       0.60     114.2      260                     5.4%
    8192                 13.89       2.42     507.6      263                    19.3%
   32768                 13.89       9.66    3159.3      244                    49.5%
  2 x all parameters: 16.38 GFLOP/token; 2 x non-embedding: 13.89
  attention equals the weights' FLOPs at 47,104 tokens
  16384 tokens as one prompt: 1191.7 ms; as 8 prompts of 2048: 850.0 ms
```

Per token, the weights cost the same at every length while attention
grows linearly. The two are equal at `non-embedding parameters /
(layers · heads · head_dim)`, 47,104 tokens. By 32,768 tokens attention
is half the time, more than its share of the FLOPs, because the model
runs fused attention at 0.65 of the tensor cores' peak (chapter 7).
Throughout, prefill runs at 73–84% of peak.

The last line is a trap worth remembering. Eight prompts of 2,048
tokens have the same GEMM work as one prompt of 16,384, but each
prompt's attention sees only its own tokens, so they cost 850 ms, not
1,192. A batch of prompts is not one long prompt.

## Decode: the weights, then the cache

In decode, M is the batch. A weight GEMM with M rows does `2·M·N·K`
FLOPs and reads `2·N·K` bytes of 16-bit weights, so its arithmetic
intensity is about M FLOPs per byte. Chapter 2's ridge point for the
A100 is 312 TFLOP/s ÷ 2,039 GB/s = 153 FLOPs per byte. Below that, a
weight GEMM is memory-bound: its time is the weight read, however many
rows share it. That is why serving engines batch decodes. Chapter 3
left one question open: how many sequences can share a weight read
before the math catches up? The cache decides part of the answer.

The KV cache is the other half of a decode step's bytes. Each layer
keeps every past token's key and value for each KV head:

```python
    def kv_bytes_per_token(self) -> float:
        return (2 * (self.n_full_layers + self.mtp_layers) * self.n_kv_heads
                * self.head_dim * self.dtype.nbytes)
```

(Docstring trimmed. For a plain transformer `n_full_layers` is the layer
count and `mtp_layers` is 0; chapter 19 adds prediction heads.)

![Thirty-two query heads in eight groups of four, each group sharing one
key and value head; the cache stores only the eight KV
heads.](/assets/tinyperf-book/ch06-gqa.svg)

*Figure 6.2. Grouped-query attention (GQA). Groups of query heads share
one key head and one value head, so the cache stores `kv_heads`, not
`heads`. The model batches attention per KV head, which reads each
cached head once for its whole group.*

The graph builder models GQA through the shape of attention's first
GEMM:

```python
    scores = None if skip_gqa else g.BatchedMatMul(
        "attn_qk", q, batch=b_local * nkv_l, m=group * s, n=kv_own, k=hd, count=L_ctx,
        out_dtype=DType.FP32, q_len=s,
    )
```

One batch entry per sequence and KV head, the group's query heads as
the rows, and the cache as the second operand, `n=kv_own` keys long
(`kv` for a plain transformer; `skip_gqa` is false and `L_ctx` the layer
count). The keys are read once per KV head, not once per query head.

Early transformers gave every query head its own key and value heads
(multi-head attention). GQA shares them, and the cache shrinks by the
ratio:

```
Table 6.4  The KV cache: bytes per token, 16-bit, and what grouped-query attention saves
  model           heads q/kv bytes/token/layer MB/token if q = kv saving GB at 32k tokens
  llama2-7b            32/32            16,384    0.524     0.524     1x             17.2
  qwen3-8b              32/8             4,096    0.147     0.590     4x              4.8
  llama3-70b            64/8             4,096    0.328     2.621     8x             10.7
```

Llama-2-7B gives every query head its own KV head and stores 0.52 MB
for every token of every sequence: one 32,768-token sequence needs 17.2
GB, more than its 13.5 GB of weights. Qwen3-8B stores a quarter of what
32 KV heads would need, and Llama-3-70B an eighth, which is why a
70-billion-parameter model caches less per token than a 7-billion one
without GQA.

A decode step reads every sequence's cache once, so its bytes are

```
decode bytes ≈ weights  +  batch · context · KV bytes per token
```

The first term is shared by the batch. The second is not: each sequence
brings its own. At 2,048 tokens of context, 50 Qwen3-8B sequences read
as much cache as weights.

## Where decode batching stops being free

Table 6.5 prices Qwen3-8B decode steps at 1,024 tokens of context, by
batch, on two GPUs: an A100 at its datasheet rates, and an RTX A6000 at
the rates chapter 4 fits to it, with its kernels issued from a CUDA
graph (launches recorded once and replayed as one; chapter 4) as a
serving engine issues them.

```
Table 6.5  Decode steps by batch: Qwen3-8B, context 1024
  A100 SXM at datasheet rates; RTX A6000 at its fitted rates, in a CUDA graph
                        A100                             RTX A6000             
  batch   step ms  GEMMs attention   tok/s    step ms  GEMMs attention   tok/s
      1      8.71   7.87      0.18     115      24.21  22.83      0.61      41 
      8      9.27   7.89      0.70     863      26.23  23.18      2.21     305 
     32     11.19   7.94      2.49    2859      32.49  23.71      7.69     985 
     64     13.76   8.01      4.87    4652      41.56  25.14     15.00    1540 
     96     17.41   9.17      7.24    5515      50.68  26.62     22.31    1894 
    128     20.07   9.35      9.62    6377      58.85  27.15     29.62    2175 
    129     24.54  13.73      9.70    5257      67.66  35.72     29.85    1907 
    192     29.53  13.83     14.38    6502      84.57  37.60     44.24    2270*
    256     36.58  15.90     19.14    6998     108.14  45.88     58.86    2367*
    512     70.67  30.06     38.17    7245*    196.90  73.53    117.34    2600*
  * more sequences of 1024 tokens than fit in the GPU's memory (Table 6.6)
  A100: ridge point 153 FLOP/byte
  RTX A6000: ridge point 168 FLOP/byte
  RTX A6000 GEMMs without the measured row curve: 128 rows 25.39 ms, 129 rows 25.95 ms
  the KV read equals the weight read at: context 512: 200, context 1024: 100, context 2048: 50, context 8192: 13
  129 sequences read as much cache as weights at context 796
```

Read the GEMM columns first. On the A100 the weight GEMMs cost 7.87 ms
for one sequence and 8.01 ms for 64, 2% more for 64 times the tokens. By
128 they cost 19% more, as chapter 3's tile effects start to show (a
kernel computes its output in fixed-size blocks, or tiles, and a partial
tile costs a whole one): padded math and L2 traffic begin to outlast
some GEMMs' weight read. At 129 the GEMMs jump by 47%. One row past a
128-row tile, a kernel computes 256 rows or streams its operands through
L2 far more: the math has caught up with the weight read. That is where
the ridge point says it should, a little early (129 against 153) because
tiles come in multiples of 128. On the A6000, whose fitted ridge point
is 168, the GEMM column above 32 rows carries cuBLAS's measured row
curve (a correction to the GEMM price, measured by row count;
chapter 3), so its 32% jump at 129 is measured, not predicted: without
the curve, the tile model (chapter 3's GEMM price alone) moves only from
25.39 to 25.95 ms.

Now read the attention columns. They grow from the first sequence,
because every sequence reads its own cache. The table's crossing line
gives the batch at which the cache read equals the weight read: 100 at
this context. So the cache doubles the A100's step by about 100
sequences (17.41 ms at 96, against 8.71 at 1), before the math catches
up at 129. The cache comes first at every context above 796 tokens,
where 129 sequences read as much cache as weights; at longer contexts
it comes much sooner.

![Microseconds per generated token against batch on an A100: the weight
GEMMs fall as one over the batch to about 128 sequences, then flatten;
whole steps flatten sooner at longer contexts.](/assets/tinyperf-book/ch06-decode-batch.svg)

*Figure 6.3. The cost of one generated token, Qwen3-8B on an A100 at
datasheet rates. The dashed line is the roofline of the weight GEMMs:
one weight read shared by the batch (falling as 1/batch) until it meets
the math of one token (48.5 µs, flat) at the ridge point. The tile
model (blue) turns just past 128. The whole step flattens at the tile
edge or where the cache read overtakes the weights, whichever comes
first: about 128 sequences at 512 tokens (the tile edge), 50 at 2,048,
13 at 8,192.*

So chapter 3's question has a two-part answer. For the weights, a
decode batch is nearly free up to about the ridge point, rounded down
to the last full 128-row tile: about 128 sequences on these GPUs. For
the cache, a decode batch is never free, and it stops being cheap at
`weights / (context · KV per token)` sequences. Past both, the cost per
token falls only slowly: the A100 makes 6,377 tokens/s at 128 sequences
of 1,024 tokens, 6,998 at 256 and 7,245 at 512 (more than fit). Each
doubling past 128 buys 10%, then 4%.

The asterisks point at the other limit, memory (Table 6.6). On the
A6000, 177 sequences of 1,024 tokens fit: past both the cache's
crossing (100) and the tile edge (129), so the rows beyond 177 are
hypothetical. At 2,048 tokens only 88 fit, before the math catches up
but after the cache has (50). At 2,048 tokens and beyond, capacity, not
arithmetic, decides how far a GPU with 48 GB can batch this model.

## How many requests fit

Memory holds three things: the weights, the KV cache of every running
sequence, and working memory for the step in flight. tinyperf's
capacity model is closed-form. Here is `max_batch` reduced to one GPU
(pseudocode: the real function also takes pipeline, context-parallel,
offloading and shared-prefix arguments, all neutral here):

```python
MEM_HEADROOM = 0.90  # usable fraction of HBM (fragmentation, workspace, context)

def max_batch(p, device, context_len):
    budget = MEM_HEADROOM * device.hbm_gb * 1e9 - _weights_local_bytes(p, 1)
    per_request = p.kv_local_bytes(context_len, 1) + _activation_bytes(p, 1, 1, 1)
    return max(0, math.floor(budget / per_request))
```

`kv_local_bytes` is `kv_bytes_per_token × context` for a plain
transformer. The working memory, `_activation_bytes`, is the live
activation and residual at the step's widest point plus one row of
logits per sequence: 0.01 GB for 32 decodes (Table 6.6). The 10%
headroom stands in for fragmentation, library workspaces and the CUDA
context, stated rather than hidden.

```
Table 6.6  The largest decode batch that fits, by context: weights + KV + working memory <= 90% of memory
  model      GPU              weights GB   1024   2048   4096   8192  32768
  llama2-7b  RTX A6000 48 GB        13.5     55     27     13      6      1
  llama2-7b  A100 80 GB             13.5    108     54     27     13      3
  qwen3-8b   RTX A6000 48 GB        16.4    177     88     44     22      5
  qwen3-8b   A100 80 GB             16.4    367    183     92     46     11
  qwen3-14b  RTX A6000 48 GB        29.5     81     40     20     10      2
  qwen3-14b  A100 80 GB             29.5    252    126     63     31      7
  qwen3-8b, RTX A6000, 32 x 2048: weights   16.4 + kv    9.7 + act  0.01 =   26.1 GB on 48 GB (fits)
  KV room on the RTX A6000: 181,879 tokens at 90% of 48 GB; vLLM's pool as chapter 17 prices it: 189,072 tokens
```

The KV cache is the capacity story: double the context and the batch
halves. GQA sets its size. Llama-2-7B has smaller weights than
Qwen3-8B but a cache 3.6 times larger per token, so a third as many of
its sequences fit at every context. Qwen3-14B's weights take most of an
A6000, leaving room for 81 sequences of 1,024 tokens.

The 90% rule is an estimate. A serving engine sizes its KV pool from
the memory it measures at start-up; on this GPU vLLM's pool holds 4%
more tokens than the rule (the table's last line). Chapter 17 prices
the pool the way vLLM allocates it.

## How close is it?

The evidence is Qwen3-8B, bf16, served by vLLM on one RTX A6000 at eight
cells: batches of 1, 8 and 32 sequences, prompts of 512, 2,048 and 8,192
tokens (32 × 8,192 would not fit: Table 6.6). Every prompt is distinct
random token ids, so the engine's prefix cache (which skips the prefill
of a prompt start it has already seen; chapter 17) cannot reuse
anything; decoding is greedy, with CUDA graphs on. Two numbers per cell,
both defined properly in chapter 14:

- **TTFT**, time to first token: here, the wall time for the batch of
  requests to produce one token each. The model prices it as the
  prefill of `batch` prompts of `prompt` tokens, each its own sequence.
- **TPOT**, time per output token: the wall time to generate 129
  tokens, less the TTFT, divided by 128. That is the mean decode step,
  which the model prices as one step of `batch` sequences at the mean
  context, `prompt + 64`.

The code follows the repository's test for this grid: `m.prefill_us(b *
s, n_seqs=b)` and `m.decode_us(b, s + 64)`, from the `StepLatencyModel`
of chapter 14 (the class that prices a serving step), on the calibrated
A6000 (its rates fitted to kernels timed on it; chapter 4). Each ratio
is predicted ÷ measured, so above 1 the model reads high; a set of
ratios is summarized by its range and its *typical error*, the geometric
mean distance from 1 (chapter 3).

```
Table 6.7  Qwen3-8B served by vLLM on one RTX A6000: model/measured, calibrated model
  batch prompt   TTFT ms model/meas   TPOT ms model/meas  attention share
      1    512      79.6      0.967     23.87      1.010               2%
      1   2048     291.3      1.002     24.30      1.007               3%
      1   8192    1282.1      1.049     25.75      1.003               9%
      8    512     564.5      0.978     25.12      1.013               6%
      8   2048    2280.8      1.005     27.80      1.013              15%
      8   8192   10420.5      1.029     39.12      0.999              39%
     32    512    2191.5      1.001     29.90      0.980              15%
     32   2048    9094.2      1.006     41.98      0.958              38%
  TTFT: 0.967-1.049, typical error 1.9%
  TPOT: 0.958-1.013, typical error 1.4%
```

Every cell is within 5%, with a typical error under 2%. The
measurements have this chapter's shape. TPOT at batch 1 barely moves
from 512 to 8,192 tokens of context (23.9 to 25.8 ms): the step is the
weights. At batch 8 it grows from 25.1 to 39.1 ms, and the model gives
39% of the larger step to attention. TTFT scales with the tokens in the
batch, as math does.

Which constants were fitted, and where? None to this grid. The GPU's
math and memory rates and its L2 bandwidth were fitted to 27 cuBLAS
GEMMs timed alone, and the per-kernel cost inside a CUDA graph was
measured separately (chapter 4). The decode attention kernel's
per-call cost and its rate of reading the cache were fitted to vLLM's
attention kernel timed alone (chapter 7). Chapter 3's measured row
curve, cuBLAS on this model's own layer GEMMs timed alone at 32 to
1,280 rows, touches only one cell: the 1 × 512 prefill, 512 rows. The
other prefills have 2,048 rows or more and are priced by the tile model;
no decode cell has more than 32 rows. Prefill attention runs at 0.65 of
the fitted math rate, a published FlashAttention figure, not a fitted
one.

So the TPOT column is held out (nothing in the model was fitted to it or
designed with it in view), and so is TTFT at batch 1, with one
qualification: TPOT's attention half rests on constants fitted to vLLM's
decode kernel at batch 1–64 and context 256–8,192, which covers the
grid's decode shapes. The five batched TTFT cells are in-sample: their
error is how the per-sequence attention price (`n_seqs`) was found
(field note below). One more caveat: this grid has been a test since it
was measured, and every later change to the model had to keep it within
0.92–1.10 for TTFT and 0.93–1.07 for TPOT. The strongest evidence is the
prediction committed before the measurement ran, which was recorded then
and is not recomputed:

```
Recorded  Predictions committed before the measurement, model/measured
  TPOT, all 8 cells: 0.969-1.014
  TTFT, batch 1: 0.954-1.049; batch 8 and 32: 1.079-2.793 (32 x 2048: 2.79)
```

> **Field note: the prompts laid end to end.** The committed
> predictions got decode right within 3.1% on every cell, and single
> prompts within 5%. Batched prompts read up to 2.79. The serving model
> had priced a batch's prefill as one sequence, so 32 prompts of 2,048
> tokens were priced as one of 65,536, with 32 times the attention (the
> trap of Table 6.3's last line). The fix was to pass the number of
> sequences (`n_seqs`), and the error's shape named the cause: exact for
> one sequence, worse with every sequence added, worst where attention's
> share was largest.

## Where it breaks

- **Fixed batches only.** Every price here is one step of a batch that
  starts together. A server admits requests as they come, mixes prompt
  chunks into decode steps and pads batches to CUDA-graph sizes;
  chapters 14–16 build those steps.
- **One kernel per op.** The model launches every op in Figure 6.1 as a
  kernel of its own. An engine can fuse some, and may run kernels the
  graph leaves out. Launches are 14% of the worked example's decode
  step, so a different fusion plan moves the price.
- **Prefill memory.** The working memory in `max_batch` is a decode
  step's; a long prefill needs far more (`llm_memory` with
  `phase="prefill"` prices it).
- **Datasheet rates.** Tables 6.2–6.5 on the A100 use no measurement.
  They rank and explain; chapter 4's calibrated tier is what predicts.
- **One validated model and GPU.** The grid is one dense model on one
  GPU. Long prompts rest on chapter 7's attention price, whose 1 ×
  8,192 cell already reads 5% high.

## What you built

- A transformer as a handful of integers from its released config, with
  parameter counts that match the published ones.
- `build_llm_graph`'s dense path: one layer replicated per layer, and an
  LM head that runs on one row per prompt in prefill.
- Prefill as `2 · non-embedding parameters · tokens` plus attention
  growing with the square of the prompt, math-bound.
- Decode as the weights once plus every sequence's cache, memory-bound;
  the cache per token, and GQA's saving.
- The two limits on decode batching: the ridge point for the weight
  GEMMs (about 128 sequences here) and `weights / (context · KV per
  token)` for the cache, whichever comes first.
- A closed-form capacity model: the largest batch at each context.
- Evidence: TTFT and TPOT within 5% of vLLM on every cell of an
  eight-cell Qwen3-8B grid on an RTX A6000, typical error under 2%.

## Exercises

1. Add a preset for a dense model not in the repository from its
   released config, and check `param_count` against its model card. If
   it ties its embeddings, which count does the card report?
2. Find the batch at which Qwen3-8B's weight GEMMs stop being flat on an
   H100 and a B200 (`data/devices/`). Compare each with its ridge point.
   Which GPU gains the most from batching, and at what context does the
   cache take that gain away?
3. Store the KV cache in fp8 (chapter 10's recipes set it). How do Table
   6.6's A6000 rows change? At what context does an fp8 cache let the
   math catch up with the weights before memory runs out?
4. Price Qwen3-8B and Llama-2-7B prefills of 2,048 tokens with
   `logits="all"`. Which grows more, and why?
5. Table 6.7's worst TPOT cell is 32 × 2,048, at 0.958, with 38% of the
   step in attention. Which half of the step would have to be priced
   low to explain the gap, and what would you time to find out?

---

*[← Chapter 5: Graphs and the scheduler]({% post_url 2026-09-29-tinyperf-05-graphs-and-the-scheduler %}) · [Contents](/series/tinyperf/) · [Chapter 7: Attention →]({% post_url 2026-09-29-tinyperf-07-attention %})*
