---
layout: post
title: "Building tinyperf, chapter 7: Attention"
date: 2026-09-29 12:07:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/07-attention/
excerpt: "Every other operation in a transformer layer costs the same whether a sequence has a hundred tokens behind it or a hundred thousand. Attention does not. In prefill its work grows with the square of the prompt; in decode, each new token reads the keys and values of every token before it. So how do you price the one part of a transformer whose cost grows with the context? And why does the answer depend on which kernel the serving engine runs?"
redirect_from:
  - /tinyperf/perf-modeling/2026/08/18/building-tinyperf-m7.html
  - /tinyperf/perf-modeling/2026/09/23/building-tinyperf-m68.html
  - /tinyperf/perf-modeling/2026/09/26/building-tinyperf-m75.html
---

*[Building tinyperf](/series/tinyperf/) · Part II: One model on one GPU · Code: [`tinyperf/passes.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/passes.py), `fuse_attention`; [`tinyperf/gemm_model.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/gemm_model.py), `estimate_fmha`; [`tinyperf/scheduler.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/scheduler.py), `_exec_fmha`; and [`tinyperf/serving.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/serving.py), `mixed_reread_us` · Every table and both plots in this chapter come from `python3 book/scripts/ch07_attention.py`.*

Every other operation in a transformer layer costs the same whether a
sequence has a hundred tokens behind it or a hundred thousand.
Attention does not. In prefill its work grows with the square of the
prompt; in decode, each new token reads the keys and values of every
token before it. So how do you price the one part of a transformer whose
cost grows with the context? And why does the answer depend on which
kernel the serving engine runs?

The short answer: written as math, attention is two matrix multiplies
with a softmax between them, and the score matrix they pass along is the
largest tensor in a long prefill. No engine runs it that way. A fused
kernel keeps the scores on chip, so it moves only the queries, keys,
values and output, and its math runs at an efficiency that belongs to
the kernel. In decode the price is the cache: a fixed cost per call plus
every sequence's cache at a measured rate, so a batch costs what its
mean context costs. Kernels differ by factors of two or three, so the
attention backend (the engine's library of attention kernels, such as
FlashAttention-2 or FlashInfer) is a parameter of the model.

By the end of this chapter you will know:

- what attention costs as three operations, and why the width of its
  scores matters;
- why a fusion pass, a graph rewrite that merges them into one kernel,
  needs a price of its own, and what fusion saves;
- how prefill attention is priced, and how far a kernel can sit from the
  model's efficiency constant;
- how decode attention is priced: a floor per call plus the cache, at
  the batch's mean context;
- GQA (grouped-query attention) packing: one read of a KV head's cache
  for all the query heads that share it;
- the step in which FlashAttention-2 loses it, and why that makes the
  backend a model parameter;
- how close it gets: 5.6% typical error (the geometric mean miss;
  chapter 3) on a decode kernel it was not fitted to, and whole decode
  steps within 1.5%.

## Attention as written

Take one layer and a step of b sequences, with h query heads sharing g
KV heads (grouped-query attention, chapter 6) of dimension d, and s new
tokens per sequence, each attending to kv keys:

```
S = Q · Kᵀ          scores: per KV head, (h/g)·s rows × kv columns, contracting over d
P = softmax(S)      row by row
O = P · V           (h/g)·s rows × d columns, contracting over kv
```

Chapter 6's graph builder writes exactly this: a `BatchedMatMul`, a
`Softmax` and a second `BatchedMatMul`.

S holds b·h·s·kv numbers. In prefill kv is the causal average,
(s + 1)/2 (chapter 6), so S grows with the square of the prompt. Tensor
cores accumulate in fp32, but a GEMM library can write its output in 16
bits. tinyperf assumes the unfused path keeps S in fp32, as a float32
softmax does (not measured); softmax then writes the probabilities in 16
bits for the second GEMM (chapter 6's builder: `out_dtype=DType.FP32`).
Per score, the three kernels move

```
QKᵀ writes 4 bytes, softmax reads 4 and writes 2, PV reads 2:   12 bytes per score
```

Table 7.1 prices one layer of Qwen3-8B, whose 32 query heads share 8 KV
heads, on an A100 at its datasheet rates.

```
Table 7.1  Attention as written, one layer: Qwen3-8B prefill, A100 SXM, datasheet rates, us
  tokens  scores GB      QK^T   softmax        PV  three ops     fused  fused bound
    2048       0.27     146.9     200.7      86.0      433.5     172.7  math
    8192       4.30    2263.4    3163.4    1165.6     6592.4    2715.6  math
   32768      68.72   36100.9   50559.9   17914.9   104575.7   43397.0  math
  bound of the three ops at every length: QK^T dram, softmax dram, PV dram
```

At 32,768 tokens one layer's scores are 68.7 GB, and the three
operations take 105 ms, every one of them bound by memory traffic, not
math. The last two columns are the same attention as one fused kernel,
which the rest of this chapter builds: 43 ms, bound by math. At every
length the three operations take 2.4–2.5 times as long.

## Fused attention: a rewrite and its price

FlashAttention (Dao and colleagues, 2022) computes the same O without S
ever leaving the SM, the streaming multiprocessor that computes it
(Figure 7.1). A CTA (a thread block, which runs on one SM; chapter 3)
keeps a block of query rows in registers and streams K and V through
shared memory, one block of keys at a time. For each block it computes
that block's scores on chip, updates a running maximum and running sum
for every row, rescales the output it has accumulated so far, and adds
the block's share of P·V. After the last block it divides by the sum and
writes O. The model charges Q, K, V and O once each.

![Three kernels passing the score matrix through DRAM, and one fused
kernel keeping it on chip.](/assets/tinyperf-book/ch07-fused.svg)

*Figure 7.1. Attention as written and as FlashAttention runs it. The
unfused path writes and reads every score twice (orange), 12 bytes per
score; the fused kernel keeps one block of scores at a time on the SM,
and the running maximum and sum let it rescale the output as it goes.*

tinyperf models this in two halves: a pass that rewrites the graph, and
a price for its result. The pass replaces each QKᵀ → softmax → PV chain
with one `FusedAttention` operator:

```python
    while i < len(ops):
        if i + 2 < len(ops):
            qk, sm, pv = ops[i], ops[i + 1], ops[i + 2]
            if (
                isinstance(qk, BatchedMatMul)
                and isinstance(sm, Softmax)
                and isinstance(pv, BatchedMatMul)
                and sm.inputs[0] is qk.out
                and pv.inputs[0] is sm.out
            ):
                op = FusedAttention(
                    qk.name.replace("_qk", "") + "_fmha",
                    [qk.inputs[0]],
                    batch=qk.batch, m=qk.m, kv=qk.n, k=qk.k, out_dim=pv.n,
                    count=qk.attrs.get("count", 1), q_len=qk.attrs.get("q_len"),
                )
                fused.append(op)
                n_fused += 1
                i += 3
                continue
        fused.append(ops[i])
        i += 1
```

(The set-up before the loop and the two lines after it are trimmed.) It
matches by dataflow, not by name: the softmax must consume the first
GEMM's output, and the second GEMM the softmax's. The fused operator
keeps both GEMMs' shapes, and `q_len`, the query tokens per sequence,
which decode needs.

Why does a rewrite need a price of its own? A pass that only deleted the
softmax would still charge the two GEMMs for writing and reading S,
which the fused kernel never stores. And the fused kernel's math is not
a GEMM's: between its matrix instructions it computes exponentials,
maxima and rescalings, so it runs further below the tensor cores' peak.
So the fused operator gets its own model, `estimate_fmha` (type hints,
the docstring and the result's bookkeeping trimmed):

```python
FMHA_MATH_EFFICIENCY = 0.65

def estimate_fmha(device, batch, m, kv, k, out_dim, dtype, math_efficiency=None):
    eff = math_efficiency if math_efficiency is not None else FMHA_MATH_EFFICIENCY
    flops = 2.0 * batch * m * kv * (k + out_dim)
    math_s = flops / (device.peak_tflops(dtype) * 1e12 * eff)

    dram_bytes = dtype.nbytes * batch * (m * k + kv * (k + out_dim) + m * out_dim)
    dram_s = device.dram_time_s(dram_bytes)

    time_us = max(math_s, dram_s) * 1e6 + device.kernel_launch_us
```

In words:

```
flops = 2 · batch · m · kv · (k + out_dim)                          both GEMMs
bytes = bytes per value · batch · (m·k + kv·(k + out_dim) + m·out_dim)    Q, K and V, O
time  = max(flops / (peak · efficiency), bytes / bandwidth) + launch
```

`batch` is sequences × KV heads and `m` is the group's query heads × the
step's tokens. The efficiency defaults to 0.65: FlashAttention-2's paper
reports its kernels at 50–73% of the A100's datasheet peak. On a
calibrated GPU, one whose rates were fitted to measurements, it
multiplies chapter 4's fitted GEMM rate, "0.65 of what the best GEMMs
reach": on the A6000, whose fitted rate is 0.75 of its datasheet peak,
0.65 × 0.75 = 0.49 of datasheet peak.

```
Table 7.2  What fusion saves: Qwen3-8B on an A100 SXM, datasheet rates, whole forward pass, ms
  step                    unfused    fused  saved   unfused, fp16 scores  saved
  prefill, 2048 tokens      123.6    114.2   7.6%                  119.5   4.4%
  prefill, 8192 tokens      647.1    507.6  21.6%                  577.0  12.0%
  prefill, 32768 tokens    5361.7   3159.3  41.1%                 4223.1  25.2%
  decode, 1 x 2048            9.1      8.8   3.8%                    9.1   3.8%
  decode, 32 x 4096          19.9     18.3   7.9%                   19.6   6.5%
```

Fusion saves 8% of a 2,048-token prefill and 41% of a 32,768-token one:
the scores grow with the square of the prompt, everything else with its
length. In decode fusion saves 4–8%: at batch 1, little more than two
launches per layer; at 32 × 4,096, the scores' traffic too.

> **Field note: the scores at half width.** The model first let every
> operator's output inherit its input's precision, which priced the
> scores at 16 bits: Table 7.2's last two columns. Against the fp32
> price, that reads 3% low for a 2,048-token prefill and 21% low for a
> 32,768-token one, and fusion saves 25% where the fp32 price says 41%.
> A difference that grows with context points at a term that does; how
> much fusion is worth depends on the width an unfused engine stores.

## Prefill: math at a kernel's efficiency

A prefill's attention does 4·h·d·s·kv FLOPs per layer and moves only Q,
K, V and O, so it is bound by math (Table 7.1's last column), and its
price rests on the kernel's efficiency. Is 0.65 right? Table 7.3 times
prefill attention kernels alone, one prompt with no earlier context, on
an RTX A6000 (84 SMs), and reads off the efficiency each reaches. Its
ratios, like every ratio in this book, are predicted ÷ measured: above
1 the model reads high.

```
Table 7.3  Prefill attention timed alone: one prompt, one layer, RTX A6000
  efficiency: achieved FLOP/s over the fitted math rate; model/measured with the fused price at 0.65 and at 0.215
  kernel, model (query/KV heads x head dim)     tokens measured us efficiency  at 0.65 at 0.215
  FlashAttention-2, Qwen3-8B (32/8 x 128)          512        52.5       0.35     0.61        -
  FlashAttention-2, Qwen3-8B (32/8 x 128)         1024       138.8       0.53     0.85        -
  FlashAttention-2, Qwen3-8B (32/8 x 128)         1920       396.5       0.66     1.02        -
  Triton, gpt-oss-20b full layers (64/8 x 64)      512        99.2       0.19     0.32     0.91
  Triton, gpt-oss-20b full layers (64/8 x 64)     1024       341.9       0.22     0.34     1.02
  Triton, gpt-oss-20b full layers (64/8 x 64)     2048      1322.9       0.22     0.35     1.05
  Triton, gpt-oss-20b full layers (64/8 x 64)     4096      5093.7       0.23     0.36     1.08
  Triton, gpt-oss-20b full layers (64/8 x 64)     8192     20047.6       0.24     0.36     1.10
  Triton, both layer kinds, 11 prompt lengths from 256 tokens: FlashAttention-2's price is 0.31-0.36 of measured, median 0.331; 0.65 x 0.331 = 0.215
```

vLLM's FlashAttention-2 (timed eagerly, each kernel launched on its own
from Python, a few microseconds of launch included) reaches the model's
constant at 1,920 tokens: 0.66 of the fitted rate, which is
0.66 × 0.75 = 0.50 of datasheet peak, the low end of the published
range. At 512 tokens it reaches 0.35, and the model reads 0.61,
consistent with a short prompt having too few blocks of query rows to
fill 84 SMs evenly (not profiled). One efficiency is right for long
prompts and optimistic for short ones.

The Triton rows are a different kernel. gpt-oss-20b gives each head a
learned *sink*, an extra logit in the softmax's denominator, and
because of it vLLM runs gpt-oss on this GPU with its own attention
kernel, written in Triton (a Python-based language for GPU kernels).
That kernel
reaches 0.19–0.24 at every length, about a third of FlashAttention-2's.
Each of its programs (Triton's CTAs) takes 16 query rows, all 8 query
heads of one KV head for 2 tokens, so each block of K and V it loads
feeds little math; that is consistent with its rate (not profiled), and
the model carries the measured value. The calibration stores this
backend's own efficiency, 0.215: 0.65 times the median ratio over both
of gpt-oss's layer kinds (it alternates full attention with 128-token
sliding windows, chapter 8). The last column is therefore fitted, and it
still drifts from 0.91 to 1.10. For a while a constant fitted to whole
gpt-oss prefills absorbed this factor of three; chapter 22 tells that
story.

## Decode: reading the cache

In decode each sequence brings one query token per head and attends to
its whole context. The math is negligible; the bytes are the sequence's
cache, context × 2·g·d values per layer. So the obvious price is the
bytes over the bandwidth plus a launch. Here it is for the smallest cell
of the measurement described below, run inside a CUDA graph (launches
recorded once and replayed together; chapter 4):

```
Worked example  One decode-attention call: Qwen3-8B, one sequence, context 256, RTX A6000 in a CUDA graph
  KV read: 256 tokens x 4,096 bytes = 1.05 MB = 1.52 us at the fitted 691 GB/s
  generic price: 3.5 us launch + 1.52 = 5.0 us
  measured: 11.7 us per layer
  with the fitted floor: 7.0 + 3.5 + 1.52 / 0.96 = 12.1 us
```

The kernel costs more than twice the generic price, and at short
contexts most of that doesn't grow with the cache. The model carries it
as a measured floor (not profiled).

The measurement calls vLLM's FlashAttention-2 decode kernel the way the
engine does: the cache paged in 16-token blocks (chapter 17), one query
token per sequence, inside a CUDA graph (chapter 4), at batches of 1 to
64 and contexts of 256 to 8,192 tokens, 36 cells. Two constants fitted
over all 36 price it: 7 µs per call, `fmha_decode_floor_us`, and the
cache streamed at 0.96 of the fitted DRAM rate,
`fmha_decode_dram_efficiency`. In `_exec_fmha`, the scheduler's price
for a fused-attention operator, they apply to any call with one query
token per sequence (the lines before, a comment and the result's
bookkeeping trimmed):

```python
    elif op.attrs.get("q_len") == 1 and ctx.fmha_decode_floor_us is not None:
        time_us = ctx.fmha_decode_floor_us + ctx.device.kernel_launch_us \
            + max(est.math_us, est.dram_us / (ctx.fmha_decode_dram_efficiency or 1.0))
```

`est` is `estimate_fmha`'s price for the same operator. So a decode call
costs `floor + launch + cache bytes / (bandwidth · 0.96)`: 10.5 µs plus
the cache here.

```
Table 7.4  The decode attention kernel timed alone: vLLM's FlashAttention-2, Qwen3-8B, RTX A6000 in a CUDA graph
  batch  context  measured us/layer  model/measured  generic/measured
      1      256               11.7           1.034             0.431
      1     1024               13.1           1.282             0.730
      1     2048               25.8           0.899             0.608
      1     8192               64.4           0.948             0.808
      8      256               28.0           0.834             0.565
      8     2048              113.1           0.988             0.891
      8     8192              397.1           1.046             0.987
     64      256              115.2           0.983             0.886
     64     2048              806.2           1.019             0.970
     64     8192             3173.6           1.024             0.981
  all 36 cells (batch 1-64, context 256-8192), fitted: 0.83-1.28, typical error 4.9%
  batch 8 and up: 0.83-1.06, typical error 3.8%
  generic price (no floor, the fitted DRAM rate): 0.43-1.01, typical error 18.3%
  the sweep repeated in a later session: every cell within 1.4% of the first; model 0.85-1.28, typical error 4.9%
  batch 64, context 8192: the cache read once in 3174 us per layer is 677 GB/s, 0.98 of the fitted DRAM rate
  FlashInfer's decode kernel, the same sweep, its own two constants (6.0 us, 1.02): 0.92-1.34, typical error 2.5%
```

![Decode attention per layer against context at batches 1, 8 and 64,
measured and modelled.](/assets/tinyperf-book/ch07-decode-kernel.svg)

*Figure 7.2. The decode kernel against its model, both axes
logarithmic. At batch 64 the cache dominates from the shortest context.
At batch 1 the kernel costs 12–13 µs a layer up to 1,024 tokens, then
jumps; the generic price (grey) misses the floor at every context.*

These ratios are fitted: the constants came from these 36 cells. The
generic price reads as low as 0.43, 18% typical; with the floor,
4.9%, and 3.8% from batch 8 up. A repeat of the sweep in a later
session reproduced every cell within 1.4%. The worst cell is at batch 1
and 1,024 tokens: from there to 2,048 the kernel's time doubles, 13.1 to
25.8 µs, where the extra cache takes about 6 µs to read. That is
consistent with the kernel changing how it splits a lone sequence's
context across CTAs (not profiled), and a straight line reads 1.28 on
one side of the jump and 0.90 on the other.

### A batch costs its mean context

Each sequence reads its own cache, so a batch's cache bytes are the sum
of its contexts times the bytes per token: the batch size times the
*mean* context. The floor is per call. So a step's attention depends on
its contexts only through their mean, and chapter 14's serving
simulator prices each step there:

```python
                kv_now = sum(r.prompt + r.generated for r in running) / len(running)
```

`decode_us` interpolates between prices it caches every 256 tokens.
Pricing at the longest context instead overcharges any batch whose
lengths differ; "How close is it?" measures by how much.

> **Field note: the maximum that paid for something else.** The serving
> simulator once priced every decode step at its batch's longest
> context, rounded up to 256 tokens. That overcharge had been cancelling
> a cost the model lacked, the sampler, which picks each next token
> (chapter 15), and pricing the mean alone moved seven validated results
> at once, one of them to 0.88. The fix was to price each piece of the
> step from its own kernel, timed alone, and only then change the
> context. An error you can't remove without making things worse is
> paying for another error; find that one first.

## GQA packing, and the step that breaks it

In a decode step a KV head serves h/g query heads, four for Qwen3-8B,
each with one query token. FlashAttention-2 packs them: when every
sequence in the call has exactly one query token, it treats the group's
query heads as the rows of one tile (the block of output one CTA
computes), and one CTA reads the KV head's cache once for all four
(Figure 7.3, left). Table 7.4's second-to-last line shows it: at batch
64 and 8,192 tokens, counting every byte of cache once, the kernel reads
at 0.98 of the fitted DRAM rate. If each query head read the cache for
itself, the time could be up to four times longer.

With *chunked prefill* (chapter 14) an engine runs chunks of a new
prompt in the same step as the running decodes: a *mixed step* (chapter
16). vLLM's FlashAttention-2 backend runs all of a step's rows through
one call per layer. Now some sequences have hundreds of query tokens,
the kernel doesn't pack, and each decode row's cache is read once per
query head, four times (Figure 7.3, right). The L2 cache catches some of
the repeats; the rest come from DRAM.

![One CTA reading a KV head's cache once for four query heads, and four
CTAs each reading it.](/assets/tinyperf-book/ch07-gqa-packing.svg)

*Figure 7.3. GQA packing. In a pure decode step the group shares one
read of its KV head's cache. Beside a prompt chunk, FlashAttention-2
runs each query head on its own and reads the cache once per head.*

To measure it, vLLM's kernel was timed on the decode rows alone, the
chunk alone, and both in one call: 220 cells of 1–63 decode rows ×
contexts of 278–4,600 × chunks of 128–1,920 tokens at Qwen3-8B's 32/8
heads, and 12 cells each at 64/8 and 16/8 heads, held out (nothing was
fitted to them). The *extra reads* are (together − decode alone − chunk
alone) ÷ decode alone: 0 if the call costs its parts, 3 if every one of
four query heads read the cache from DRAM.

```
Table 7.5  Decode rows beside a 512-token prompt chunk, context 1150: one layer, RTX A6000, us (timed eagerly)
        FlashAttention-2                                                     FlashInfer           
  rows  decode alone  chunk alone  together  extra reads  model extra reads  together  extra reads
     8          77.9         51.6     166.1         0.47               1.43     121.2         0.02
    16         133.4         53.7     374.8         1.41               1.60     168.4         0.01
    32         259.0         53.6     806.8         1.91               1.94     274.9         0.01
    48         343.1         53.0    1303.0         2.64               2.28     384.2         0.01
    63         443.3         53.0    1722.4         2.77               2.60     482.3         0.01
  extra reads = (together - decode alone - chunk alone) / decode alone; at most 3 for 4 query heads per KV head
  the model of the kernel, measured decode alone x (1 + modelled extra reads) + measured chunk alone, cells with 8-63 rows:
    32/8 heads, 160 cells, fitted  : 0.83-1.96, typical error 10.3%; without the re-read 0.26-1.15, typical error 120.4%
    64/8 heads,  12 cells, held out: 0.93-1.48, typical error 11.5%; without the re-read 0.13-0.71, typical error 259.7%
    16/8 heads,  12 cells, held out: 1.02-1.91, typical error 21.1%; without the re-read 0.55-1.49, typical error 36.4%
  median extra time over the two parts, all 220 cells: FlashAttention-2 85%, FlashInfer 0.7%
  Triton, gpt-oss-20b (64/8), 60 cells timed in a CUDA graph: together below the sum of the parts in 60
```

At 63 rows the call costs 1,722 µs where its parts cost 496: 2.77 extra
reads of a possible 3. At 8 rows it is 0.47. The share of repeats that
miss the L2 grows with the rows, consistent with more sequences' caches
competing for the 6 MB L2 at once (not profiled).

The model prices the extra reads as `(h/g − 1)` times the fraction that
misses the L2 (`m` in the code, not `estimate_fmha`'s rows), a straight
line in the rows, the log of the context and the log of the chunk,
clipped to [0, 1]. Its four coefficients, `mixed_decode_gqa_reread` in
the calibration, were fitted on the 32/8 grid's 160 cells with 8–63
rows; the 60 cells with 1–4 rows were measured but not fitted:

```python
    def mixed_reread_us(self, decode_batch: int, kv_len: float, chunk_new: int) -> float:
        coef = self.calibration.mixed_decode_gqa_reread if self.calibration else None
        group = self.p.n_heads / max(1, self.p.n_kv_heads)
        if not coef or group <= 1 or not decode_batch or not chunk_new:
            return 0.0
        a, b, c, d = coef
        m = a + b * decode_batch / 64 + c * math.log2(max(kv_len, 1) / 1000) + d * math.log2(max(chunk_new, 1) / 512)
        return (group - 1) * min(1.0, max(0.0, m)) * self.decode_attention_us(decode_batch, kv_len)
```

(Docstring trimmed.) The result is added to the mixed step's price;
chapter 16 composes the rest of that step. Table 7.5 checks the kernel
as measured decode alone × (1 + modelled extra reads) + measured chunk
alone. Without the term, that reads 120% typical error on the cells it
was fitted to; with it, 10%. Held out on 64/8 heads, where a decode row
can be read eight times, it reads 11.5% where no term reads 260%. On
16/8 it reads 1.02–1.91, all high: the line overestimates the misses for
a group of two, and, in Figure 7.4, at few rows.

![Extra reads of the decode rows' cache against decode rows, for
FlashAttention-2, its model and FlashInfer.](/assets/tinyperf-book/ch07-reread.svg)

*Figure 7.4. Extra reads beside a 512-token chunk at context 1,150,
Qwen3-8B. FlashAttention-2 approaches the limit of 3 as rows are added;
the fitted line runs high below 16 rows. FlashInfer pays nothing at any
row count. At 1–2 rows the extra reads are negative: the eager timing
saves a launch. The line was fitted from 8 rows up.*

## Attention backends

The re-read belongs to one kernel, not to attention. vLLM's FlashInfer
backend runs a step's decodes through FlashInfer's decode kernel, which
packs a KV head's query heads as rows whether or not the step carries a
chunk, and the chunk through a separate prefill call. Its two calls
together cost their parts: 0.01–0.02 extra reads in Table 7.5, 0.7%
extra at the median over all 220 cells, where FlashAttention-2's is 85%.
(Its decode kernel has two constants of its own, fitted to its own run
of Table 7.4's 36 cells.) vLLM's Triton kernel packs a KV head's query
heads into every program, and all 60 of its mixed cells cost less than
their parts.

So the attention backend is a parameter of the model. The calibration's
top-level attention fields are FlashAttention-2's, vLLM's default on
this GPU; other backends are overrides (reformatted):

```json
  "attention_backends": {
    "flashinfer": {"fmha_decode_floor_us": 6.0, "fmha_decode_dram_efficiency": 1.02,
                   "mixed_decode_gqa_reread": null},
    "triton_attn": {"mixed_decode_gqa_reread": null, "fmha_math_efficiency": 0.215}
  },
```

`Calibration.for_attention` swaps in the backend's fields. The serving
model takes the backend by name,
`StepLatencyModel(p, device, attn_backend="flashinfer")`, and applies it
after its software stack's launch cost (eager or CUDA graph; chapter 4),
`cal.for_stack(stack).for_attention(attn_backend)`. A `null` re-read
means packed. Triton's entry sets no decode fields, so it keeps
FlashAttention-2's floor and rate: an assumption the next section tests.

## How close is it?

Three tests, none of them fitted.

**A kernel the decode constants never saw.** vLLM's Triton decode kernel
on gpt-oss-20b's full-attention layers (64/8 heads of dimension 64),
timed alone in a CUDA graph, priced with FlashAttention-2's constants:

```
Table 7.6  Held out: vLLM's Triton decode kernel priced with FlashAttention-2's constants, gpt-oss-20b full-attention layers, model/measured
  batch     512    1024    2048    3072    4096    6144
      8    1.06    1.00    1.01    0.99    1.00    1.00
     16    0.90    0.92    0.97    0.99    1.00    1.02
     32    1.16    1.11    1.08    1.07    1.07    1.06
     64    1.11    1.09    1.07    1.07    1.07    1.07
  24 cells: 0.90-1.16, typical error 5.6%
```

A different kernel, head shape and model, and the typical error, 5.6%,
is close to the fitted 4.9%. vLLM runs this kernel two ways: while
sequences × KV heads ≤ 128, up to 16 sequences here, it splits each
context across programs, and above that it doesn't. The model reads
0.90–1.06 in the first regime and 1.06–1.16 in the second, where the
kernel reads the cache at more than the fitted 0.96 of the DRAM rate.

**Batches whose contexts differ.** Steady batches on vLLM, half
128-token prompts and half 1,920, 200 tokens generated, timed by the
engine's own clock (CUDA events around each step on the engine's stream;
chapter 15). The mean context is about half the maximum. The step
includes its GEMMs and the sampler (chapter 15):

```
Table 7.7  Steady batches, half 128-token and half 1920-token prompts: Qwen3-8B decode steps on the engine's clock, RTX A6000
  mean context 1124, maximum 2020
  batch  measured ms  at the mean: ms  model/meas  at the maximum: ms  model/meas  attention ms, mean / max
     16        28.36            28.59       1.008               31.77       1.120                 4.4 / 7.6
     32        32.90            33.29       1.012               39.66       1.205                8.4 / 14.8
     48        38.99            39.57       1.015               49.12       1.260               12.4 / 22.0
     64        43.38            43.15       0.995               55.89       1.288               16.4 / 29.2
```

At the mean the model lands within 1.5% at every batch. At the maximum
it reads 12–29% high, more as attention's share of the step grows.

**The backend in the engine.** Mixed steps under both backends on the
engine's clock: 8, 32 or 63 sequences decoding from prompts of 256,
1,024 or 2,048 tokens, beside an injected prompt of 128–1,024 tokens:

```
Table 7.8  The backend in the engine: Qwen3-8B mixed steps with 63 decodes, FlashInfer over FlashAttention-2, RTX A6000
  decodes prompt  chunk  FlashAttention-2 ms  FlashInfer ms  ratio measured  predicted  its re-read, model ms
             256    128                 56.9           45.3            0.80       0.79                   11.5
             256   1024                183.9          167.5            0.91       0.93                   12.2
            1024    128                 91.7           55.0            0.60       0.59                   37.0
            1024   1024                212.7          176.3            0.83       0.81                   39.7
            2048    128                146.3           69.1            0.47       0.46                   77.9
            2048   1024                267.0          187.6            0.70       0.69                   83.2
  all 36 cells (8, 32 or 63 decodes; chunks of 128-1024 tokens): measured ratio / predicted ratio 0.97-1.07
  model/measured, FlashInfer 0.96-1.02, FlashAttention-2 0.97-1.06
```

With 63 sequences decoding from 2,048-token prompts beside a 128-token
chunk, FlashInfer's step costs 0.47 of FlashAttention-2's; the model
predicted 0.46, and its re-read is 78 ms of the 146 ms FlashAttention-2
step. Over all 36 cells the measured ratio lands within 0.97–1.07 of the
predicted one. The backend alone can halve a mixed step.

## Where it breaks

- **One efficiency per kernel.** FlashAttention-2 reaches 0.65 at the
  longest prompt timed, 1,920 tokens; at 512 the model reads 0.61.
  Triton's rate drifts too (0.91–1.10).
- **K and V read once.** The kernel reads K and V once per block of
  query rows and relies on the L2 for the repeats; `estimate_fmha`
  reads them once per KV head and has no L2 term.
- **The decode kernel's own jumps.** The line through Table 7.4 can't
  follow the batch-1 jump between 1,024 and 2,048 tokens, nor Triton's
  switch at 16 sequences.
- **The re-read's form.** A straight line in rows reads 1.43 extra
  reads at 8 rows where the kernel pays 0.47, and runs high for a group
  of two. It multiplies the whole decode call, floor included, where
  only the cache reads repeat.
- **One GPU, one engine version.** Every kernel here was timed on an RTX
  A6000 with vLLM 0.15.1. Elsewhere the model has the published 0.65 and
  a generic decode price.
- **Dense attention only.** Sliding windows, sinks, latent caches and
  sparse attention change what is read; chapter 8 covers them.

## What you built

- Attention as three operators passing a score matrix through memory,
  12 bytes per score if it is stored in fp32.
- A fusion pass that rewrites by dataflow, and a price for what it
  produces: both GEMMs' math at a kernel's efficiency, traffic of Q, K,
  V and O. Fusion saves 8% of a 2,048-token prefill and 41% of a
  32,768-token one.
- Prefill attention at 0.65 of the GEMM rate for FlashAttention-2
  (0.66 measured at 1,920 tokens) and 0.215 for vLLM's Triton kernel.
- Decode attention as a floor per call plus the cache at a fitted rate,
  at the batch's mean context: 4.9% typical error where fitted, 5.6% on
  a kernel it never saw, decode steps within 1.5%.
- GQA packing, and FlashAttention-2's re-read beside a prompt chunk:
  10% where fitted, 11.5% and 21% held out.
- The attention backend as a parameter, `attn_backend`, which predicts
  a mixed step's backend ratio within 0.97–1.07.

## Exercises

1. Replace the single 0.65 with an efficiency that depends on the
   prompt's length, fitted to Table 7.3 and the chunk timings in
   `data/calibration/mixed_attention_rtx_a6000_32_8.json`. How much does
   a 512-token chunked prefill step change, and why?
2. Derive the re-read's miss fraction from an occupancy argument instead
   of a line: the bytes of cache that a wave of unpacked CTAs (one round
   of CTAs across the SMs; chapter 3) needs resident, against the
   A6000's 6 MB of L2. Test it on the 16/8 and 64/8 grids, where the
   line misses.
3. Time FlashAttention-2's decode kernel at batch 1 from 1,024 to 2,048
   tokens in steps of 128 (`tools/measure_decode_attention.py`). Where
   is the jump? Write a split-context term for `estimate_fmha` that
   produces it.
4. Using Table 7.2's method, find the prompt length at which fusion
   saves a quarter of a Qwen3-8B prefill on the H100 and on the B200
   (`data/devices/`). Why does it differ from the A100's?

---

*[← Chapter 6: A transformer: prefill, decode and memory]({% post_url 2026-09-29-tinyperf-06-a-transformer-prefill-decode-and-memory %}) · [Contents](/series/tinyperf/) · [Chapter 8: Attention variants →]({% post_url 2026-09-29-tinyperf-08-attention-variants %})*
