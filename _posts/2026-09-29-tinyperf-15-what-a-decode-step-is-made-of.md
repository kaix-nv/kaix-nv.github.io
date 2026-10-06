---
layout: post
title: "Building tinyperf, chapter 15: What a decode step is made of"
date: 2026-09-29 12:15:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/15-what-a-decode-step-is-made-of/
excerpt: "Serve Qwen3-8B with vLLM on an RTX A6000, start 64 requests together, and once their prompts are read the engine gives each of them a new token every 36.2 milliseconds. Send the same requests with greedy decoding and a step takes 31.5 ms; nothing else changed. A serving engine's decode step takes a measured number of milliseconds: what exactly is in it, and how do you price each part?"
redirect_from:
  - /tinyperf/perf-modeling/2026/09/08/building-tinyperf-m55.html
  - /tinyperf/perf-modeling/2026/09/08/building-tinyperf-m56.html
  - /tinyperf/perf-modeling/2026/09/09/building-tinyperf-m57.html
  - /tinyperf/perf-modeling/2026/09/22/building-tinyperf-m67.html
  - /tinyperf/perf-modeling/2026/09/24/building-tinyperf-m70.html
---

*[Building tinyperf](/series/tinyperf/) · Part IV: Serving · Code: [`tinyperf/serving.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/serving.py), `StepLatencyModel.decode_us`, `sampler_us` and `padded_batch`, and [`tinyperf/scheduler.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/scheduler.py), `_row_corrected` · Every table and the plot in this chapter come from `python3 book/scripts/ch15_decode_step.py`.*

Serve Qwen3-8B with vLLM on an RTX A6000, start 64 requests together,
and once their prompts are read the engine gives each of them a new
token every 36.2 milliseconds. Send the same requests with greedy
decoding and a step takes 31.5 ms; nothing else changed. A serving
engine's decode step takes a measured number of milliseconds: what
exactly is in it, and how do you price each part?

The short answer: a decode step is one forward pass, replayed from a
CUDA graph, then the sampler. The forward is mostly the weight GEMMs
and the LM head, which together read 15.14 GB of weights once: about
23 ms on this GPU for one sequence or 32, and 6–9% more above 32 rows,
where cuBLAS's time steps up. The LM head is under 2 ms of that.
Attention reads every sequence's own cache, so it grows with the batch
times the mean context. The small ops add 0.8–1.4 ms. The sampler
costs 0.17 ms for 64 greedy sequences and 4.68 ms when they ask for
top-p. And the step runs in the graph captured for the next batch size
up, so the GEMMs pay for padded rows that attention and the sampler
never see.

By the end of this chapter you will know:

- the five parts of a decode step, and what each costs on one GPU;
- why the weight GEMMs need cuBLAS's measured row curve above 32 rows;
- why attention is priced at the batch's mean context, and only for its
  real sequences;
- what the sampler costs, why top-p costs almost thirty times greedy,
  and why a load generator's defaults decide which one runs;
- how CUDA-graph padding changes the step;
- how to time a step on the engine's own clock, and what a client's
  inter-token gap does and doesn't measure;
- how close the model gets: whole steps within 0.98–1.01 for greedy,
  sampled and top-p decoding, and 2–5% low under top-k.

## A step, part by part

In steady decoding a serving engine repeats three things. Its scheduler
picks the running sequences (chapter 14). It runs one forward pass that
carries one new token per sequence through all 36 layers and the LM
head, replayed from a CUDA graph (chapter 4). And its sampler picks each
sequence's next token from that sequence's row of logits. The serving
simulator of chapter 14 charges each step five parts:

```
step = GEMMs + small ops + LM head      at the graph's rows
     + attention                        for the real sequences, at their mean context
     + sampler                          for the real sequences
```

Table 15.1 prices them for Qwen3-8B on the RTX A6000 at the rates
chapter 4 fitted, with kernels issued from a CUDA graph. The context,
278 tokens, is that of the measurement this chapter ends with: 128-token
prompts, halfway through 300 generated tokens.

```
Table 15.1  A decode step's parts as the model prices them: Qwen3-8B, RTX A6000 in a CUDA graph, context 278, ms
  batch   GEMMs  attention  LM head  small ops  forward  sampler: greedy  top-p  step: greedy  top-p
      1   21.00       0.44     1.83       0.77    24.05             0.01   0.48         24.06  24.53
      8   21.32       0.88     1.86       0.85    24.91             0.03   0.94         24.94  25.85
     16   21.36       1.38     1.86       0.93    25.53             0.05   1.48         25.58  27.00
     24   21.82       1.88     1.85       1.01    26.56             0.07   2.01         26.63  28.57
     32   21.85       2.38     1.86       1.09    27.18             0.09   2.54         27.27  29.72
     40   23.71       2.88     1.87       1.17    29.64             0.11   3.08         29.75  32.72
     48   23.90       3.39     1.88       1.26    30.42             0.13   3.61         30.54  34.02
     56   23.67       3.89     1.89       1.34    30.79             0.15   4.14         30.94  34.93
     64   23.25       4.39     1.89       1.42    30.95             0.17   4.68         31.11  35.62
  kernels per step: 399, each 3.5 us to launch from the graph: 1.40 ms, inside every column; 218 of them small ops: 0.76 ms
  weights read per step: 15.14 GB (13.89 in the layer GEMMs, 1.24 in the LM head) = 21.9 ms at the fitted 691 GB/s
  batch 64 at context 1024: attention 15.00 ms of a 41.56 ms forward (36%)
  batch 64 at context 2048: attention 29.56 ms of a 56.12 ms forward (53%)
```

![Stacked bars of a Qwen3-8B decode step by batch on an RTX A6000, split
into GEMMs, LM head, small ops, attention and sampler, with the measured
step as a dot.](/assets/tinyperf-book/ch15-step-parts.svg)

*Figure 15.1. The model's decode step by batch, and the step measured on
the engine's clock (dots; see Timing a step). Left: Table 15.1 with
top-p sampling. Right: steady batches of half 128-token and half
1,920-token prompts, mean context 1,124, greedy (chapter 7's Table
7.7). The GEMMs are at least half of every step and flat to 32
sequences; attention and the sampler grow with the batch. Every dot is
within 2% of its bar.*

Read the table by columns.

- **GEMMs.** The weights, 21.9 ms at the fitted 691 GB/s, are most of
  the GEMM and LM head columns (22.83 ms at one sequence). To 32
  sequences the GEMMs grow 4%: extra rows share one read of the weights
  (chapter 6). From 32 to 40 they jump 1.86 ms.
- **Attention** grows in a straight line with the batch: 53% of the
  forward at 64 sequences of 2,048 tokens.
- **Small ops** (norms, rotary embedding, activation, residual adds, the
  embedding lookup) are 218 of the 399 kernels. At one sequence they
  cost their launches, 0.76 ms.

## The weight GEMMs: cuBLAS's row curve

Chapter 3 found that the tile model can't follow a library's own
choices: one row past a 128-row tile, cuBLAS's time jumps by up to 73%,
and no first-principles argument says which kernel it will pick. The
calibrated tier stores a measured correction for this,
`dense_gemm_row_factor` (chapter 4, Table 4.9). In a decode step it
matters well before 128 rows.

`tools/measure_decode_gemms.py` times Qwen3-8B's four layer GEMMs the
way vLLM calls them: `F.linear` in bf16, captured in a CUDA graph, at
vLLM's graph sizes and beyond. Table 15.2 sums the 36 layers.

```
Table 15.2  The 36 layers' four weight GEMMs by rows, as vLLM calls them (bf16, CUDA graph), RTX A6000: sum of 144 GEMMs, ms
  rows  measured  tile model  model/meas  with the row curve  model/meas  row factor
     1     21.08       21.00        1.00               21.00        1.00           -
     8     20.94       21.32        1.02               21.32        1.02           -
    16     21.36       21.36        1.00               21.36        1.00           -
    24     22.30       21.82        0.98               21.82        0.98           -
    32     21.92       21.85        1.00               21.85        1.00       1.000
    40     23.72       21.23        0.89               23.71        1.00       1.117
    48     23.90       21.26        0.89               23.90        1.00       1.124
    56     23.68       22.08        0.93               23.67        1.00       1.072
    64     23.26       22.12        0.95               23.25        1.00       1.051
    96     24.69       22.23        0.90               24.70        1.00       1.111
   128     25.19       23.44        0.93               25.20        1.00       1.075
   129     33.73       23.95        0.71               33.72        1.00       1.408
  tile model alone: 1-32 rows 0.98-1.02 (each GEMM 0.91-1.08); 40-64 rows 0.89-0.95
  32 -> 40 rows, measured: qkv_proj x1.12, attn_out x1.01, ffn_gate_up x1.13, ffn_down x1.00
  each GEMM with the curve, 40-64 rows: 0.93-1.13 (one factor per row count for all four shapes)
  held out, tp=2 (each GPU's shapes), 40-64 rows: tile model 0.91-0.95 (typical error 7.0%), with the curve 0.96-1.07 (4.2%)
```

Up to 32 rows the tile model's sum is within 2%: the time is reading the
weights. At 40 rows cuBLAS takes 23.72 ms where 32 rows took 21.92,
while the tile model's price even falls, to 21.23 ms. The jump is in the
two wide GEMMs, the QKV projection (6,144 columns) and gate-up (24,576),
12–13%; the 4,096-wide ones don't move. That is consistent with
cuBLAS switching kernels for the wide shapes above 32 rows (not
profiled). Past 64 rows, which a pure decode step here never reaches,
the curve carries the tile edges that a mixed step of decodes and a
prompt chunk runs into (chapter 16): 129 rows cost 34% more than 128.

The correction is a table of factors: the measured sum over the tile
model's sum, at 33 row counts from 32 to 1,280, with points on both
sides of every 128-row edge. `_row_corrected` applies it (docstring
trimmed):

```python
def _row_corrected(table, m: int, price) -> float | None:
    if m <= table[0][0]:
        return None
    if m >= table[-1][0]:
        return price(-(-m // 128) * 128)
    for (a, fa), (b, fb) in zip(table, table[1:]):
        if a <= m <= b:
            ta, tb = price(a) * fa, price(b) * fb
            return ta + (m - a) / (b - a) * (tb - ta)
    return None
```

`price(rows)` is the tile model's time for this GEMM at that many rows.
Between measured points the corrected times are joined by a straight
line in rows, not the model's own tile steps. Beyond the table a GEMM
runs in whole 128-row tiles; at 32 rows or fewer nothing changes. The
scheduler applies it to dense layer GEMMs only (the arguments of
`estimate_gemm` and the lines around it trimmed):

```python
        if ctx.dense_gemm_row_factor and wo is None and not is_expert and op.batch == 1 and op.name != "lm_head":
            price = lambda rows: estimate_gemm(ctx.device, rows, op.n, op.k, ...).time_us
            t = _row_corrected(ctx.dense_gemm_row_factor, op.m, price)
            if t is not None:
                est = replace(est, time_us=t)
```

The curve's column reads 1.00 from 40 rows up by construction: its
factors came from those points (*fitted*). Two checks say more. The
factor is one number per row count for all four shapes, so each GEMM
alone reads 0.93–1.13 at 40–64 rows. And for tp=2 (each GPU holds half
of every weight matrix; chapter 11), whose per-GPU shapes the curve
never saw, it moves 40–64 rows from 0.91–0.95 to 0.96–1.07, a typical
error of 4.2% against 7.0%: *held out*.

## Attention at the mean context

Chapter 7 prices decode attention from its kernel and shows that a
batch's attention depends on its contexts only through their mean. The
simulator passes each step's mean context, and `decode_us` interpolates
between prices every 256 tokens.

## The LM head

```
Worked example  The LM head
  151,936 x 4,096 bf16 weights: 1.24 GB = 1801 us at 691 GB/s
  timed alone: 1755 us at 1 row, 1827 at 64; model 1832 and 1891
```

The head reads 1.24 GB at any batch, so its time barely moves. The model
prices it 4% high; the row curve, measured on the layer GEMMs, doesn't
apply. Under tensor parallelism vLLM splits the head by vocabulary and
all-gathers the logits for the sampler; chapter 11 prices both.

## The sampler

After the forward, every sequence that emits a token sends its row of
logits through vLLM's `Sampler`: a cast to fp32, whatever the request
asked for, then the draw. It runs eagerly, outside the graph, queued
while the graph runs, so the step pays its GPU time. What it does
depends on the request:

- **greedy** (temperature 0): an argmax;
- **sample** (a temperature alone): a softmax and a random draw;
- **top-k**: keep the k largest logits, found without a sort, and sample;
- **top-p**: sort every row's 151,936 logits, softmax, cumulative sum,
  mask, scatter back, and sample. Any top-p takes this path, with or
  without a top-k beside it.

`tools/measure_sampler.py` times vLLM's own `Sampler` on bf16 logits.
A sleeping kernel holds the GPU while the host queues the sampler
behind it, as a step's forward does, and CUDA events bracket only the
sampler.

```
Table 15.3  vLLM's sampler timed alone against its price: vocabulary 151,936, RTX A6000, ms of GPU time
  rows  greedy meas  price  sample meas  price   top-k meas  price   top-p meas  price
     1        0.015  0.012        0.069  0.036        0.370  0.160        0.262  0.477
     8        0.027  0.030        0.112  0.114        0.312  0.302        0.931  0.943
    16        0.048  0.049        0.207  0.204        0.474  0.465        1.505  1.476
    32        0.088  0.089        0.381  0.382        0.775  0.790        2.533  2.543
    64        0.164  0.168        0.741  0.740        1.460  1.440        4.681  4.675
  settings: sample temperature 0.6; top-k adds k = 20; top-p adds p = 0.95 to both (top-k 20)
  price = fixed + rows x passes x 151,936 x 4 bytes at the fitted 691 GB/s (0.879 us per pass)
  greedy 10 us + 2.8 passes = 2.5 us per row; sample 25 us + 12.7 passes = 11.2 us per row; top-k 140 us + 23.1 passes = 20.3 us per row; top-p 410 us + 75.8 passes = 66.6 us per row
  8-64 rows (8 row counts timed), where the constants were fitted: sample, top-k and top-p 0.97-1.02; greedy within 4 us
  1 row: top-p 1.82, top-k 0.43; top-k measured 0.370 ms at 1 row, 0.263 at 2
```

Measured at Qwen3's settings: sample is temperature 0.6; top-k adds
k = 20; top-p adds p = 0.95 to both. Each mode is a fixed cost plus a
cost per row. The price states the per-row cost as fp32 passes over the
vocabulary at the fitted DRAM rate,
so it scales with a model's vocabulary and, unmeasured, with another
GPU's bandwidth (docstring trimmed):

```python
SAMPLERS = {"greedy": (10.0, 2.8), "sample": (25.0, 12.7), "top_k": (140.0, 23.1), "top_p": (410.0, 75.8)}

    def sampler_us(self, rows: int) -> float:
        if rows <= 0:
            return 0.0
        fixed, passes = SAMPLERS[self.sampling]
        eff = self.calibration.dram_efficiency if self.calibration else 1.0
        rows_local = -(-rows // self.dp)
        return fixed + rows_local * self.p.vocab * 4 * passes / (self.device.dram_bw_gbps * 1e3 * eff)
```

With data-parallel attention (chapter 12) each of `dp` replicas samples
its own rows.

```
Worked example  The top-p sampler for 64 sequences
  one row's fp32 logits: 151,936 x 4 bytes = 0.61 MB
  75.8 passes over them at 691 GB/s: 66.6 us per row
  410 us + 64 x 66.6 us = 4.68 ms; greedy: 0.17 ms
```

Top-p's 75.8 passes per row are consistent with sorting 151,936
numbers (not profiled); greedy's argmax takes 2.8. The constants were
fitted to the kernel timings at 8–64 rows (Table 15.3 shows four of the
eight). There the other modes' price is within 3%, and greedy's within
4 µs. Below 8 rows the form fails: one row of top-p takes 0.262 ms where
the price says 0.477, and top-k's single row costs more than its two.
That is about 0.2 ms in a 24 ms step.

**The requests decide which sampler runs,** so `StepLatencyModel` takes
it as an input, `sampling=`, greedy by default. And the requests are
often decided by a tool's defaults. vLLM's load generator, `vllm bench
serve` (version 0.15.1, as measured here), sends no sampling parameters,
so the server applies the model's generation config, for Qwen3
temperature 0.6, top-k 20 and top-p 0.95: every step sorts. A tool that
sends `temperature: 0` gets greedy. Two runs of "the same" benchmark can
differ by 4.76 ms a step at 64 sequences (Table 15.6). If you serve with
a model's generation config, tell the model; chapter 14's engine preset,
`serving.VLLM`, says `"top_p"` for these checkpoints.

> **Field note: the constant that was the sampler.** Chapter 4's field
> note "three zeros" follows where a per-sequence cost went; this is
> what it was. A load sweep of 128-token prompts once left the model's
> decode steps 3.4–4.2 ms short at 64 running sequences, and a fit put
> the gap at 63 µs per sequence per step: the engine's bookkeeping, it
> was thought. Steps timed later with greedy decoding matched the model
> within 0.7 ms at batch 1–64, so the constant went to zero. But the
> sweep had not been greedy: its load generator sent no sampling
> parameters, and top-p costs 66.6 µs per row plus 410 µs, the
> constant's size and signature. Its absence stayed mostly hidden, for
> two reasons. The simulator then priced each step at its batch's
> longest context, rounded up to 256 tokens: 768 tokens on that sweep at
> saturation, against a mean near 384 (chapter 7's field note tells that
> side). And greedy steps looked exact by a second cancellation: priced
> with the same round-up they ran 1.1–1.3 ms high at 32 and 64
> sequences, while the forward at 48–64 ran 1.7–2.9 ms low, mostly in
> the GEMMs above 32 rows. A fitted constant can be right in size and
> wrong in name.

## CUDA-graph padding

A CUDA graph fixes every kernel's shapes when it is recorded
(chapter 4), so an engine records one graph per batch size and runs a
step in the smallest graph that holds it. vLLM's default sizes, capped
at the 64 sequences these measurements allow, are 1, 2, 4 and every
multiple of 8 up to 64. A step of 9 sequences runs the 16-sequence graph
(Figure 15.2).

![A decode step of 9 sequences in the 16-row CUDA graph: the GEMMs run
all 16 rows, attention reads the 9 real caches, the sampler runs on 9
rows.](/assets/tinyperf-book/ch15-padded-graph.svg)

*Figure 15.2. A padded step. The graph's GEMMs, LM head and small ops
run all 16 rows. Attention reads the 9 real sequences' caches, each at
its own length; a padded slot holds no KV. The sampler runs on the 9
real rows.*

The padded rows hold whatever the input buffer held (chapter 9 shows an
MoE router routing them). Below 32 rows they cost the GEMMs almost
nothing, since the weights are read once either way. `padded_batch`
finds the graph:

```python
DEFAULT_CUDAGRAPH_SIZES = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64)

def padded_batch(b: int, sizes=DEFAULT_CUDAGRAPH_SIZES) -> int:
    """The graph size a batch of b sequences actually runs at."""
    if not sizes or b <= 0:
        return b
    for sz in sizes:
        if sz >= b:
            return sz
    return b        # beyond the largest captured size the engine runs eagerly, unpadded
```

These are vLLM's sizes under a 64-sequence cap, the one measured here.
For a larger cap, `vllm_cudagraph_sizes(cap)` lists vLLM's default
sizes (1, 2, 4, then every 8 up to 256), the ones the engine preset
uses past 64 sequences, unmeasured.

`decode_us` prices the forward at the graph's size, then swaps its
attention for the real sequences' (docstring and the MoE routing lines
trimmed):

```python
    def decode_us(self, batch: int, kv_len: int, shared_prefix: int = 0, real: int | None = None,
                  reply_pos: float | None = None) -> float:
        lo = int(kv_len // KV_BUCKET) * KV_BUCKET
        f = (kv_len - lo) / KV_BUCKET
        ...
        p_lo = self._price("decode", batch, max(lo, 1), shared_prefix, moe_real, route)
        us = p_lo if f == 0 else p_lo + f * (self._price("decode", batch, lo + KV_BUCKET, shared_prefix, moe_real, route) - p_lo)
        if real is not None and 0 < real < batch and not shared_prefix and self.pp == 1 and not self.offload_kv:
            us += self.decode_attention_us(real, kv_len) - self.decode_attention_us(batch, kv_len)
        return us
```

`_price` builds, prices and caches the decode graph at `batch` rows and
one context; `KV_BUCKET` is 256 tokens. `decode_attention_us` is the
same graphs' attention kernels alone. A shared prefix, pipeline stages
and an offloaded cache keep the padded price. Chapter 14's loop prices a
pure decode step of n sequences as `decode_us(padded_batch(n), mean
context, real=n) + sampler_us(n)` (simplified: a shared-prefix argument
and a retired per-sequence constant, now zero, dropped).

Table 15.4 prices steps between graph sizes, at a longer context.

```
Table 15.4  CUDA-graph padding: B sequences run in the graph for the next size up. Qwen3-8B, RTX A6000, context 1024, greedy, ms
    B  graph  as the engine runs it  unpadded, B rows  padding costs  attention for every row  too high by
    3      4                  24.97             24.87           0.4%                    25.20         0.9%
    5      8                  25.55             25.49           0.2%                    26.23         2.7%
    9     16                  26.58             26.48           0.4%                    28.18         6.0%
   12     16                  27.27             27.21           0.2%                    28.18         3.4%
   17     24                  28.94             28.80           0.5%                    30.54         5.5%
   25     32                  30.89             30.78           0.3%                    32.49         5.2%
   33     40                  34.67             32.97           5.2%                    36.27         4.6%
   36     40                  35.36             34.39           2.8%                    36.27         2.6%
   41     48                  36.77             36.54           0.6%                    38.37         4.3%
   49     56                  38.47             38.59          -0.3%                    40.07         4.2%
   57     64                  39.96             40.26          -0.7%                    41.56         4.0%
   63     64                  41.33             41.37          -0.1%                    41.56         0.6%
  vLLM's capture sizes, up to a 64-sequence cap: 1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64
```

The model prices padding itself at 0.2–0.5% below 32 sequences. At 33 it
costs 5.2%: the step runs 40 rows, past cuBLAS's change at 32. At 49 and
57 it is slightly negative, because cuBLAS ran 56 and 64 rows faster
than 48 (Table 15.2) and between measured points the curve is a straight
line. The larger error is the other one: charging attention for every
row of the graph reads up to 6.0% high, at 9 sequences in the 16-row
graph. The engine's step logs showed it before the model priced it:

```
Recorded  Online decode forwards from two sweeps' step logs, model/engine median, as recorded at measurement time
  sequences in the step                     1-2    3-4    5-8   9-16  17-32  33-64
  steps                                   11694   6376   6214   3607   1678   1686
  attention for every row of the graph    1.008  1.024  1.038  1.048  1.030  1.003
  attention for the real sequences        1.008  1.018  1.017  0.999  0.986  0.998
```

From 3 to 32 sequences, attention for the graph's rows read 1.024–1.048;
for the real sequences, 0.986–1.018. These logs are where the
real-sequence swap came from (*in-sample*). The steady cells below all
run at graph sizes, so none is padded. The held-out online sweeps at the
end of the chapter contain padded steps, but only in aggregate.

## Timing a step on the engine's clock

A client sees a step as the gap between two streamed tokens, its
*inter-token latency*. That gap passes through detokenization and HTTP,
and can't say what a step held or how long its sampler took. So the
measurements here read each step off the GPU.
`tools/vllm_patches/step_timing` wraps vLLM's model runner: it records a
CUDA event on the runner's stream at the start of each forward, just
before the sampler and just after it, and reads them back later without
synchronizing, so timing doesn't slow the engine. For each step it
writes the *period* (GPU time from this forward's start to the next's,
idle included), the forward, the sampler, and what the step held:
requests, decodes, prompt tokens, contexts and chunks.

`tools/measure_decode_step.py` starts B requests together (128 random
token ids each, 300 tokens generated, end-of-sequence ignored) under one
sampling mode, and keeps the steps with exactly B requests and no
prompt, arrival or finish: 295–299 steps per cell
(`data/validation/vllm_qwen3_8b_rtx_a6000_decode_step_sampling.json`),
whose medians are Table 15.6's measurements.

When every step is alike, a client measures the step well (Table 15.6).
When steps differ, the two diverge: with a prompt chunk, an arrival or a
finish in a step, a client's gaps no longer map one to one onto steps,
and an engine that plans one step while the previous one runs can split
a long step's delay across two gaps (chapter 16). TPOT, the per-request
average of the gaps (chapter 14), mixes every kind of step a request
lived through. So a step's price is checked against the engine's clock,
and TPOT against the whole simulator.

## How close is it?

The steady cells can be checked part by part: GEMMs, LM head and
attention timed alone, the forward and sampler on the engine's clock.
What the kernels timed alone leave of the forward is *the rest*: the
small ops, launches, and every other part's error.

```
Table 15.5  The step part by part: each part timed alone or on the engine's clock, against the model. Context 278, ms
  batch         GEMMs      LM head    attention     the rest      forward  fwd ratio  sampler, top-p
          meas  model   meas model   meas model   meas model   meas model                 meas model
      1  21.08  21.00   1.75  1.83   0.42  0.44   0.55  0.77  23.80 24.05      1.010     0.29   0.48
     16  21.36  21.36   1.78  1.86   1.40  1.38   0.93  0.93  25.47 25.53      1.002     1.55   1.48
     32  21.92  21.85   1.80  1.86   2.47  2.38   1.16  1.09  27.36 27.18      0.994     2.58   2.54
     48  23.90  23.90   1.81  1.88   3.42  3.39   1.07  1.26  30.19 30.42      1.007     3.65   3.61
     64  23.26  23.25   1.83  1.89   4.45  4.39   1.77  1.42  31.30 30.95      0.989     4.71   4.68
  GEMMs and LM head: timed alone (Table 15.2's measurement); attention: timed alone at contexts 256 and 512 (chapter 7), read at 278
  forward: the engine's clock, greedy; the rest = forward - GEMMs - LM head - attention, all measured
  the rest against the model's small ops: within 0.35 ms at every batch
  forward, model/measured: 0.989-1.010
```

What kind of number each is: the GEMMs at 48 and 64 rows are the row
curve's own points, and chapter 7's attention constants came from these
kernel cells (*fitted*); the LM head, 3–5% high, and the sampler in the
engine, within 0.07 ms from 16 rows up, are *held out*. The rest is
within 0.35 ms of the model's small ops, about as close as a difference
of four measurements can show.

The whole step, for all four sampling modes:

```
Table 15.6  Whole decode steps on the engine's clock: Qwen3-8B, RTX A6000, 128-token prompts, 300 generated, model/measured
  batch  greedy ms  ratio  sample ratio  top-k ratio  top-p ms  ratio  sampler in the engine, top-p: meas  model
      1      23.82  1.010         1.004        0.959     24.29  1.010                                0.29   0.48
     16      25.53  1.002         0.994        0.950     27.27  0.990                                1.55   1.48
     32      27.45  0.993         0.987        0.953     30.09  0.988                                2.58   2.54
     48      30.33  1.007         1.001        0.977     34.08  0.998                                3.65   3.61
     64      31.48  0.988         0.984        0.957     36.23  0.983                                4.71   4.68
  all 20 cells: 0.95-1.01, typical error 1.7%; without top-k: 0.98-1.01, 0.8%
  the sampler in the engine against its price: within 0.19 ms in all 20 cells
  top-p minus greedy, ms measured / model, by batch: 1: 0.47 / 0.46, 16: 1.75 / 1.43, 32: 2.64 / 2.45, 48: 3.76 / 3.48, 64: 4.76 / 4.51
  top-p's forward minus greedy's, measured: 0.15-0.24 ms at every batch
  top-k's forward minus greedy's, measured: 0.81-0.98 ms at every batch
  period - forward - sampler (medians): under 0.01 ms for greedy, sample and top-p; up to 0.40 ms for top-k
  the client's median inter-token gap / the engine's period: 0.993-1.001 in all 20 cells
```

Greedy, sampled and top-p steps land within 0.98–1.01 from one sequence
to 64, a typical error of 0.8%. Top-p minus greedy is 4.76 ms measured
at 64 sequences against 4.51 priced: the sampler itself is within
0.03 ms, but top-p's forward measures 0.15–0.24 ms longer than greedy's.
Top-k is the miss, at 0.95–0.98. Its forward, which ends where the
sampler starts, measures 0.81–0.98 ms longer than greedy's, and up to
0.40 ms of its period falls outside both: something on the top-k path
runs outside the sampler's events (not profiled).

No constant was fitted to these 20 cells. But this measurement is where
the GEMM change above 32 rows was found (its 48- and 64-row forwards ran
1.7–2.9 ms above the model of that time;
`data/validation/decode_step_decomposition_qwen3_8b_rtx_a6000.json`),
so read its larger batches as *in-sample*. The predictions committed
before it ran:

```
Recorded  Predictions committed before the measurement (the model of that time), model/measured
  step period, 20 cells: 0.95-1.04; sampler in the engine within 0.19 ms
```

The stronger evidence is steps no one arranged. Three held-out online
sweeps' pure decode steps, each priced at the batch and context the
engine logged for it:

```
Recorded  Held-out online sweeps: pure decode steps priced at the batch and context the engine logged, model/engine median per cell
  14 cells (two prompt shapes, top-p): 0.987-1.016
```

## Where it breaks

- **Few-row sampling.** The sampler's form fits 8–64 rows; at one row it
  reads 1.82 for top-p and 0.43 for top-k.
- **Top-k steps**, 2–5% low: a cost on the top-k path outside the
  sampler (not profiled).
- **One engine version's sampler.** vLLM 0.15.1's default sorts for
  top-p; FlashInfer's sampler doesn't (Exercise 1). Other GPUs scale the
  constants by their DRAM rate, unmeasured.
- **GEMMs one at a time.** The sums are right; single GEMMs read
  0.91–1.08 at 1–32 rows and 0.93–1.13 at 40–64. Every dense model
  on this calibration gets Qwen3-8B's factors, checked on no shapes but
  tp=2's.
- **Graph sizes.** The list is vLLM's default up to a 64-sequence cap.
  Beyond its largest size the model prices a step unpadded but still at
  the graph's launch cost, where the engine runs without the graph (not
  measured). Pass the engine's own list as `cudagraph_sizes`.
- **The host.** Here the host keeps ahead of the GPU (Table 15.6). On a
  GPU whose step takes a few milliseconds (chapter 4 prices Qwen3-8B's
  batch-1 step on a B200 at 3.8 ms), the engine's work between steps
  could show; the model has no term for it.
- **One model, one GPU, one engine version.**

## What you built

- A decode step as five parts: the weight GEMMs and the LM head at the
  graph's rows, with cuBLAS's row curve above 32; attention for the real
  sequences at their mean context; small ops, mostly launches; the
  sampler for the real rows.
- The sampler as a fixed cost plus fp32 passes over the vocabulary per
  row, for four sampling modes, with sampling an input to the model.
- CUDA-graph padding, and the engine's own clock to time a step.
- Evidence: the forward within 0.989–1.010 of the engine's clock; whole
  steps within 0.98–1.01 for greedy, sampled and top-p decoding and
  0.95–0.98 for top-k; the sampler in the engine within 0.19 ms of a
  price fitted on the kernel alone; online decode steps 0.987–1.016
  (recorded).

## Exercises

1. **FlashInfer's sampler.** Run `tools/measure_sampler.py` with
   `VLLM_USE_FLASHINFER_SAMPLER=1` and fit its top-p constants. Does its
   cost per row depend on how peaked the logits are?
2. **Find top-k's millisecond.** Profile a top-k cell and a greedy one
   with Nsight Systems. Which kernels run between the forward's last
   GEMM and the sampler?
3. **Few rows.** Find a form for the sampler's price that fits all of
   Table 15.3's rows, 1 to 64. How much does it move a step of one
   sequence?
4. **Choose capture sizes.** Run `simulate` with `chunk_tokens` and a
   `trace` list (chapter 14) on a load of your choice, and histogram the
   decodes per step. How much step time does padding cost with vLLM's
   default sizes, and with a graph at every multiple of 4?
5. **Another model's row curve.** Time another dense model's decode
   GEMMs with `tools/measure_decode_gemms.py` (change its preset). Do
   its factors above 32 rows match Qwen3-8B's?

---

*[← Chapter 14: A serving simulator]({% post_url 2026-09-29-tinyperf-14-a-serving-simulator %}) · [Contents](/series/tinyperf/) · [Chapter 16: The mixed step →]({% post_url 2026-09-29-tinyperf-16-the-mixed-step %})*
