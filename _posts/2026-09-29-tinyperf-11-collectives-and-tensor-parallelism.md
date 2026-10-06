---
layout: post
title: "Building tinyperf, chapter 11: Collectives and tensor parallelism"
date: 2026-09-29 12:11:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/11-collectives-and-tensor-parallelism/
excerpt: "A model that doesn't fit on one GPU, or runs too slowly on one, can be split across several. The most common split for inference is tensor parallelism: every weight matrix is cut into pieces, one per GPU, and the GPUs work through every layer together. What do they exchange, what does it cost, and when does adding GPUs stop paying?"
redirect_from:
  - /tinyperf/perf-modeling/2026/08/18/building-tinyperf-m6.html
  - /tinyperf/perf-modeling/2026/08/31/building-tinyperf-m34.html
  - /tinyperf/perf-modeling/2026/09/05/building-tinyperf-m45.html
  - /tinyperf/perf-modeling/2026/09/08/building-tinyperf-m48.html
  - /tinyperf/perf-modeling/2026/09/25/building-tinyperf-m73.html
---

*[Building tinyperf](/series/tinyperf/) · Part III: Many GPUs · Code: [`tinyperf/comm_model.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/comm_model.py), the tensor-parallel lines of `build_llm_graph` in [`tinyperf/nets/transformer.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/nets/transformer.py), and `_exec_comm` in [`tinyperf/scheduler.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/scheduler.py) · Every table and the plot in this chapter come from `python3 book/scripts/ch11_collectives.py`.*

A model that doesn't fit on one GPU, or runs too slowly on one, can be
split across several. The most common split for inference is *tensor
parallelism*: every weight matrix is cut into pieces, one per GPU, and
the GPUs work through every layer together. What do they exchange, what
does it cost, and when does adding GPUs stop paying?

The short answer: with tensor parallelism over `tp` GPUs, each GPU holds
1/tp of every weight matrix, and two *all-reduces* per layer add up the
GPUs' partial sums, each of `tokens × hidden` values at any tp. A
ring all-reduce costs `2(n−1)/n · bytes / link bandwidth` plus a latency
per hop. Decode's messages are small, so they pay mostly latency, and
there a real link departs from the formula's straight line and must be
measured. Adding GPUs stops paying when a GPU's computation, shrinking
as 1/tp, no longer outweighs its communication, whose messages keep
their size and whose latency grows with the group. In decode that alone
ends the scaling; crossing a node's edge adds InfiniBand's lower rate,
which is what ends it for prefill.

By the end of this chapter you will know:

- how a layer is split so that it needs only two all-reduces;
- the four collectives and the ring algorithm's cost;
- why tensor parallelism stops paying, and what the node edge adds;
- why small messages must be measured, inside a CUDA graph;
- how tinyperf brackets communication that overlaps computation;
- how close the model gets to vLLM at tp=2.

## Splitting a layer

Take a GEMM `Y = X · A` and cut A by columns across two GPUs,
`A = [A₀ A₁]`. GPU i computes `X · Aᵢ`, half of Y's columns, from the
whole of X, with no communication. Feed the result into a second GEMM,
`Z = Y · B`, and cut B by rows, `B = [B₀; B₁]`. GPU i holds `Yᵢ`,
exactly the columns of Y that meet the rows `Bᵢ`, so it computes
`Yᵢ · Bᵢ`. That is not half of Z. It is a partial sum of all of Z:

```
Z = Y₀ · B₀ + Y₁ · B₁
```

One *all-reduce*, a collective that leaves the element-wise sum of every
GPU's buffer on every GPU, finishes the pair. This column-then-row
layout comes from the Megatron-LM paper; serving engines such as vLLM
use it.

![One Qwen3-8B layer split over two GPUs, with an all-reduce after
each block.](/assets/tinyperf-book/ch11-tp-layer.svg)

*Figure 11.1. One layer at tp=2. The norms and residual adds need the
whole hidden state, so every GPU runs them whole.*

In attention the column split is by head: each GPU projects queries,
keys and values for its own heads, attends over its share of the KV
cache, and applies its rows of the output projection. In the
feed-forward network, gate and up are split by columns, SwiGLU works on
each GPU's columns, and down is split by rows. So a layer costs two
all-reduces, each of `tokens × hidden` values.

The LM head is split by vocabulary: each GPU computes the logits of
1/tp of it, and an *all-gather* concatenates them for the sampler. It
runs once per step, but its message is large: Qwen3-8B's vocabulary is
37 times its hidden size.

In the graph builder, tensor parallelism is per-GPU shapes plus
explicit collectives. The lines from `build_llm_graph`, with the
attention and MoE variants between them trimmed:

```python
    nh_l = p.n_heads // tp                       # local query heads
    nkv_l = max(1, p.n_kv_heads // tp)           # local KV heads
    ffn_l = p.ffn_hidden // tp
    ...
        qkv = g.Linear("qkv_proj", y, out_features=q_width + 2 * nkv_l * hd, count=L_full)
    ...
        ctx = Tensor("ctx", (tokens, nh_l * hd), dt)
        attn_out = g.Linear("attn_out", ctx, out_features=h, count=L_full)
        if tp > 1:
            attn_out = g.AllReduce("attn_allreduce", attn_out, group_size=tp,
                                   count=L_full)
        x = g.Elementwise("residual_attn", attn_out, x, count=L_full)
    ...
        up = g.Linear("ffn_gate_up", y, out_features=2 * ffn_l, count=L)
        act = g.Elementwise("swiglu", up, count=L)
        act = Tensor("act", (tokens, ffn_l), dt)
        down = g.Linear("ffn_down", act, out_features=h, count=L)
        if tp > 1:
            down = g.AllReduce("ffn_allreduce", down, group_size=tp, count=L)
        x = g.Elementwise("residual_ffn", down, x, count=L)
    ...
        head = g.Linear("lm_head", y_head, out_features=-(-p.vocab // tp))
        if tp > 1:
            g.AllGather("logits_allgather", head, group_size=tp)
```

The column splits divide an output width by tp (`q_width` is `nh_l · hd`
for a plain transformer). The row splits need no code: a `Linear`'s K is
its input's width, and `ctx` and `act` are 1/tp as wide as on one GPU.
Every GPU runs the same shapes, so one GPU's graph is the step's time.
`nkv_l` floors at 1: with more GPUs than KV heads, each GPU keeps a
whole KV head, and its cache stops shrinking.

```
Table 11.1  A Qwen3-8B decode step split over two GPUs: batch 8, what each GPU runs
  op                tp=1: weight N x K    tp=2, each GPU            split
  qkv_proj          6144 x 4096           3072 x 4096               column-parallel
  attention         32/8 heads            16/4 heads                by head
  attn_out          4096 x 4096           4096 x 2048               row-parallel
  attn_allreduce    -                     8 x 4096 = 64 KB          all-reduce
  ffn_gate_up       24576 x 4096          12288 x 4096              column-parallel
  ffn_down          4096 x 12288          4096 x 6144               row-parallel
  ffn_allreduce     -                     8 x 4096 = 64 KB          all-reduce
  lm_head           151936 x 4096         75968 x 4096              vocabulary-parallel, once
  logits_allgather  -                     1.2 MB in, 2.3 MB out     all-gather, once
  weights per GPU: 16.4 GB -> 8.2 GB; KV cache of one 1024-token sequence: 151 MB -> 75 MB
  per decode step: 72 all-reduces and 1 all-gather
```

(Message sizes are binary, 1 KB being 1,024 bytes; memory is decimal.)
Each GPU streams half the weights and reads half the cache, a decode
step's two costs (chapter 6), and in exchange sends 72 messages of 64 KB
and one of 1.2 MB.

## Collectives and the ring

A *collective* is a communication call that every GPU in a group makes
together. Four cover this book:

- **all-reduce**: every GPU contributes a buffer; every GPU ends with the
  element-wise sum;
- **all-gather**: every GPU contributes a piece; every GPU ends with all
  the pieces, concatenated;
- **reduce-scatter**: every GPU contributes a whole buffer; each ends
  with the sum of one piece of it;
- **all-to-all**: every GPU sends a different piece to every other, as
  in chapter 12's expert dispatch.

On NVIDIA GPUs they are usually run by NCCL, NVIDIA's collective
communications library. The classic algorithm is the *ring*. Arrange n
GPUs in a ring, each sending to the next, and cut a buffer of B bytes
into n chunks. In each of n−1 steps, every GPU sends one chunk to its
neighbour and adds the chunk it receives into its own copy; afterwards
each GPU holds one chunk summed over all n, a reduce-scatter. In n−1
more steps the summed chunks travel round the ring, an all-gather. An
all-reduce is the two in turn. Each GPU sends 2(n−1) chunks of B/n bytes,
and all the links carry traffic at once. Each step also pays a fixed *hop
latency*, the cost of one hand-off however small the chunk, and the
collective pays one kernel launch. So:

```
all-reduce      t = 2(n−1)/n · B / link  +  2(n−1) · hop latency  +  launch
all-gather      t =  (n−1)/n · B / link  +   (n−1) · hop latency  +  launch    (B = the gathered result)
reduce-scatter  priced as an all-gather of its input
all-to-all      t =  (n−1)/n · B / link  +   (n−1) · hop latency  +  launch    (B = each GPU's send buffer)
```

Two properties matter for scaling. The bandwidth term tends to `2B /
link` as n grows, so more GPUs never make an all-reduce of a given
buffer cheaper, and the latency term grows with n.

Which bandwidth is "link"? NVIDIA quotes an H100's NVLink as 900 GB/s,
450 GB/s each way added. A ring sends on one side while it receives on
the other, so the formula takes one direction's 450 GB/s, as the H100
device file does. NCCL's benchmarks report an *algorithm bandwidth*, B ÷
t, and a *bus bandwidth*, which for an all-reduce multiplies it by
2(n−1)/n to undo the ring's factor: compare the bus bandwidth with a
link's per-direction rate.

Here is the all-reduce in `comm_model.py`, docstring trimmed:

```python
def ring_all_reduce_us(
    device: Device, nbytes: float, group_size: int, bandwidth_only: bool = False,
) -> float:
    if group_size <= 1:
        return 0.0
    nodes = _nodes(device, group_size)
    nv = device.nvlink_bw_gbps * 1e9
    if nodes == 1:
        steps = 2 * (group_size - 1)
        bw_term_s = (steps / group_size) * nbytes / nv
        lat_s = steps * device.nvlink_hop_latency_us * 1e-6
    else:
        gpn = device.nvl_domain_gpus
        ib = device.ib_bw_gbps * 1e9
        bw_term_s = (
            2 * (gpn - 1) / gpn * nbytes / nv                       # RS + AG
            + 2 * (nodes - 1) / nodes * (nbytes / gpn) / ib)        # inter AR
        lat_s = (2 * (gpn - 1) * device.nvlink_hop_latency_us
                 + 2 * (nodes - 1) * device.ib_latency_us) * 1e-6
    if bandwidth_only:
        return bw_term_s * 1e6
    return (bw_term_s + lat_s) * 1e6 + device.kernel_launch_us
```

`bandwidth_only` serves the speed-of-light tier (chapter 4). The other
collectives follow the same pattern.

```
Table 11.2  One all-reduce by size and group: the ring model on H100s at datasheet rates
  NVLink 450 GB/s per direction, 1 us per hop; InfiniBand 50 GB/s per GPU, 5 us; 3 us launch; 8 GPUs per node
      size     2 GPUs  latency     8 GPUs  latency    16 GPUs  latency
      8 KB        5.0     100%       17.0     100%       27.1     100%
    512 KB        6.2      81%       19.0      89%       30.3      89%
      4 MB       14.3      35%       33.3      51%       53.8      50%
     32 MB       79.6       6%      147.5      12%      241.4      11%
      1 GB     2391.1       0%     4192.7       0%     6887.0       0%
  us; latency = the share that is hop latency and launch. 16 GPUs span 2 nodes
```

A decode all-reduce is a few hundred kilobytes at most (512 KB is a
batch of 32 at a hidden size of 8192), and 81–100% of its price is
latency and launch. A prefill all-reduce is tens of megabytes and mostly
bandwidth. The 1 µs per hop is the device file's round number, not a
measurement.

## Crossing the node edge

Inside a node, GPUs share NVLink. Between nodes they have InfiniBand:
50 GB/s per GPU in the H100 file, a ninth of NVLink. One ring through
both nodes would run at InfiniBand's rate, so the model prices a group
that spans nodes with the standard hierarchical decomposition (Figure
11.2).

![An all-reduce across two nodes of four GPUs, in three
phases.](/assets/tinyperf-book/ch11-hierarchical.svg)

*Figure 11.2. A hierarchical all-reduce over n = g·N GPUs, g per node
and N nodes. Only one chunk per GPU, 1/g of the buffer, crosses
InfiniBand, and the g exchanges run side by side, one per network card.
Each phase is a ring, so the cost is the sum of their closed forms: the
`else` branch above.*

The InfiniBand phase moves an eighth of the bytes at a ninth of the
rate, and between two nodes its ring factor, 2(N−1)/N, is 1 against
NVLink's 1.75, so for a large message it costs 9/8 ÷ 1.75 ≈ 0.64 of the
NVLink phase:

```
Worked example  Crossing the node edge: one all-reduce on H100s, ring and hierarchical
  512 KB, a batch-32 decode (hidden 8192):
    8 GPUs:  NVLink 2.0 + 14 hops x 1 us + 3 us launch = 19.0 us
    16 GPUs, two nodes: NVLink 2.0 + InfiniBand 1.3 + 14 x 1 + 2 x 5 us + 3 us = 30.3 us
    16 GPUs, one flat NVLink ring: NVLink 2.2 + 30 hops x 1 us + 3 us = 35.2 us
  32 MB, a 2048-token prefill (hidden 8192):
    8 GPUs:  NVLink 130.5 + 14 hops x 1 us + 3 us launch = 147.5 us
    16 GPUs, two nodes: NVLink 130.5 + InfiniBand 83.9 + 14 x 1 + 2 x 5 us + 3 us = 241.4 us
    16 GPUs, one flat NVLink ring: NVLink 139.8 + 30 hops x 1 us + 3 us = 172.8 us
  Llama-3-70B, ms: tp=8; tp=16 over two nodes; tp=16 in one flat NVLink domain
    decode, batch 32, context 4096:  12.94; 12.31; 13.02
    prefill, one 2048-token prompt:  79.6; 72.3; 61.3
```

Against eight GPUs, each all-reduce gets about 1.6 times dearer. But
compare sixteen GPUs with no edge, one flat NVLink ring: for decode's
small message the flat ring is dearer, its 30 hops costing more than 14
NVLink hops and two InfiniBand ones; for prefill's large message the
hierarchical all-reduce is, by InfiniBand's rate. A device with no
inter-node fabric refuses to price a spanning group (`_nodes` raises).

## When adding GPUs stops paying

Table 11.3 runs Llama-3-70B decode on 1 to 32 H100s at datasheet
rates: a projection, with nothing measured.

```
Table 11.3  Tensor-parallel scaling, a projection: Llama-3-70B decode, batch 32, context 4096,
  H100 SXM at datasheet rates, nothing hidden
   tp nodes  GB/GPU fits   GEMMs attention  other   comm  step ms  speedup comm share
    1     1   184.1   no   43.44     13.08   1.77   0.00    58.29     1.00       0.0%
    2     1    92.0   no   21.86      6.66   1.67   1.00    31.19     1.87       3.2%
    4     1    46.0  yes   11.42      3.45   1.62   1.74    18.23     3.20       9.5%
    8     1    23.0  yes    6.42      1.84   1.60   3.07    12.94     4.51      23.7%
   16     2    14.2  yes    3.92      1.84   1.58   4.96    12.31     4.73      40.3%
   32     4     9.8  yes    2.65      1.84   1.58   8.31    14.38     4.05      57.8%
  every all-reduce carries 32 x 8192 x 2 bytes = 512 KB at every tp; 160 per step, plus the logits all-gather
  8 KV heads: from tp=8 up each GPU holds one, so attention stops shrinking
  GB/GPU: weights, KV cache and working memory; fits: within 90% of the H100's 80 GB
```

At this batch and context the model needs four H100s to fit (chapter
6); the first two rows couldn't run.

The columns show why the speedup falls away. The weight GEMMs shrink
from 43.4 to 6.4 ms at tp=8, 6.8 times rather than 8, mostly because
each GEMM's launch doesn't shrink. The "other" column (norms, residual
adds, SwiGLU, RoPE, the embedding) is mostly launches and barely
shrinks, 1.77 to 1.58 ms. Attention stops shrinking at tp=8. And
communication grows: every all-reduce carries the same 512 KB, while its
price grows with the group, from 6.2 µs at two GPUs to 19.0 at eight,
nearly all latency (Table 11.2). This is Amdahl's law, with a
serial part, launches and all-reduces, that grows.

At tp=16 the group crosses the node edge, each all-reduce costs 30.3 µs,
and the doubling buys 5%: the GEMMs save 2.5 ms and communication adds
1.9 ms. At tp=32 the step is slower than at eight. The edge is not what
stops decode: in one flat 16-GPU NVLink domain the step takes 13.02 ms,
slower than at tp=8 (the worked example). In decode the ring's latency
term, 2(n−1) hops, ends the scaling by itself (exercise 3). Crossing
the node adds InfiniBand's rate, which is what hurts prefill:
from 8 GPUs to 16 it gains 1.30 times in one domain and 1.10 across two
nodes. Hence the usual advice, "fill a node with tensor parallelism and
split across nodes some other way" (chapter 12).

## Hiding communication

Every number so far adds communication to computation, as chapter 5's
scheduler adds every operator. But a collective can run while an
independent kernel computes, and in a tensor-parallel step little is
independent: each all-reduce needs the GEMM before it, and the residual
add and norm after it need the all-reduce. Engines create independent
work by splitting a batch in two and running one half's all-reduce
during the other half's GEMMs, or by fusing the all-reduce into the
GEMM, tile by tile. Whether an engine does either, and how well, depends
on its kernels, not on the graph.

So tinyperf reports a bracket. The upper end is the serial sum,
`total_us`. The lower end assumes perfect hiding (`RunReport`, docstring
trimmed):

```python
    @property
    def total_us_overlapped(self) -> float:
        compute = self.total_us - self.comm_us
        return max(compute, self.comm_us)
```

A step can't be shorter than its computation, or than its communication.
Between the two ends, the user declares the fraction of the hideable
time (the smaller of computation and communication) that their engine
hides. `StepLatencyModel` (chapter 14's step pricer) takes it as
`comm_overlap`, as does a sweep configuration; the default is 0:

```python
    def _graph_us(self, phase, batch, seq, chunk=1, shared=0, **stage) -> float:
        rep = self._graph_report(phase, batch, seq, chunk, shared, **stage)
        hideable = rep.total_us - rep.total_us_overlapped
        return rep.total_us - self.comm_overlap * hideable
```

```
Table 11.4  Hidden or not: Llama-3-70B on H100s, projection, step ms with none, half or all
  of the hideable communication hidden; gain = step time at half the GPUs / this step time,
  with none / all hidden
           decode, batch 32, context 4096        prefill, one 2048-token prompt
   tp nodes    none   half    all         gain       none   half    all         gain
    4     1   18.23  17.36  16.49            -      113.6  104.0   94.3            -
    8     1   12.94  11.40   9.87  1.41 / 1.67       79.6   67.8   56.0  1.43 / 1.68
   16     2   12.31   9.83   7.35  1.05 / 1.34       72.3   55.5   38.6  1.10 / 1.45
   32     4   14.38  11.35   8.31  0.86 / 0.88       71.8   60.2   48.6  1.01 / 0.80
  prefill at tp=32: compute 23.2 ms, communication 48.6 ms: all hidden is the communication
```

Inside a node every column agrees that doubling pays. At the node edge
the assumption decides: from 8 GPUs to 16, decode gains 5% with nothing
hidden and 34% with everything hidden. At tp=32 a prefill's
communication outlasts its computation, so even perfect hiding can't
make 32 GPUs beat 16.

What fraction is right? On the one pair measured for this book, near
zero: the serial model lands within 7% of vLLM's prefill, where the
"all hidden" end reads 0.67–0.72 (Table 11.7). NVLink with an engine
that overlaps is unmeasured, so report both ends.

## Small messages on a real link

The ring model needs two numbers per link. A datasheet gives the
bandwidth; nothing gives the latency. This chapter's measurements come
from two RTX A6000s that talk through the PCIe host bridge, with no
NVLink bridge. The model loads the pair as
`Device.load("rtx_a6000_pcie_pair")`, the RTX A6000's device file with
the link as measured: `nvlink_bw_gbps` 4.0, `nvlink_hop_latency_us` 8.0
and `gpus_per_node` 2. The NVLink fields hold the PCIe link, which is
why a collective's bound column reads `nvlink`. The 4.0 GB/s is a 64 MB all-reduce's rate in a
first benchmark (`data/validation/nccl_rtx_a6000_pcie_2gpu.json`).

> **Field note: the link that measured the interpreter.** The first
> benchmark of this pair timed a Python loop of `all_reduce` calls.
> Messages from 8 KB to 256 KB all took 87–96 µs, which fitted a hop
> latency of 41 µs, and the predictions committed before the first tp=2
> runs carried it. Prefill landed within 5%; decode read up to 42% high
> (the Recorded block below). A flat time over a 32-fold range of sizes
> is the host's rate of issuing calls, not the wire's. Inside a CUDA
> graph, as an engine replays a decode step (chapter 4), small messages
> run about five times faster and large ones are unchanged (Table 11.5's
> last rows); the hop latency refits to 8 µs. Measure a link the way the
> engine drives it.

Even inside a CUDA graph, NCCL on this pair doesn't follow one line:

```
Table 11.5  The ring model against NCCL timed inside a CUDA graph: two RTX A6000s, PCIe host bridge
   per GPU  all-reduce us  ring/meas  all-gather us  ring/meas   Qwen3-8B at tp=2
      8 KB           17.2       1.25           15.9       0.85   all-reduce, 1 sequence
     16 KB           21.0       1.12           20.5       0.76
     32 KB           28.8       0.96           34.1       0.58
     64 KB           49.6       0.72           44.8       0.62   all-reduce, batch 8
    128 KB           69.9       0.75           65.8       0.67   all-reduce, batch 16
    256 KB          101.4       0.84          100.8       0.76   all-reduce, batch 32
    512 KB          164.8       0.91          178.7       0.80   all-reduce, batch 64
      1 MB          298.3       0.94          335.7       0.82
      4 MB         1130.7       0.94         1285.4       0.82
     16 MB         4375.8       0.96         5100.3       0.82   all-reduce, 2048-token prefill
     32 MB         8715.9       0.96        10170.1       0.83
  ring model: 4.0 GB/s, 8 us per hop, 3.5 us launch; 21 sizes, 8 KB-32 MB: all-reduce 0.72-1.25, all-gather 0.58-0.85
  largest messages, bytes per GPU / time: all-reduce 3.85 GB/s, all-gather 3.30 GB/s
  the curve is the mean of two runs; run 1 / run 2 over 63 points (21 sizes x 3 collectives): median 0.999, range 0.94-1.07
  earlier benchmarks, one run each, all-reduce us:    8 KB     64 KB    256 KB     16 MB
    timed from a Python loop                         87.1      95.6      95.1    4242.5
    timed inside a CUDA graph                        16.5      50.0      97.3    4245.1
```

![NCCL's achieved bandwidth by message size on two RTX A6000s,
measured and ring model.](/assets/tinyperf-book/ch11-link-curve.svg)

*Figure 11.3. Bytes per GPU divided by time, from Table 11.5. The ring
model's lines rise smoothly toward 4.0 GB/s; the all-gather's sits above
the all-reduce's because it pays half the hops. The measured all-gather
levels off at 3.3 GB/s.*

Between 64 KB and 512 KB the ring model reads 0.72–0.91: NCCL is slower
there than a line through its small and large messages says, and decode
lives in that range. The pattern is consistent with NCCL changing
protocol and channel count as messages grow (it has protocols tuned for
small, medium and large messages); we have not profiled which it picks
where.

So for this pair the calibration carries the measured curve, and the
scheduler reads it. `_curve_us` and the start of `_exec_comm`, with a
docstring, a comment and two lines trimmed:

```python
def _curve_us(table, x: float) -> float:
    if x <= table[0][0]:
        return table[0][1]
    for (a, ta), (b, tb) in zip(table, table[1:]):
        if x <= b:
            return ta + (x - a) / (b - a) * (tb - ta)
    (a, ta), (b, tb) = table[-2], table[-1]
    return tb + (x - b) * (tb - ta) / (b - a)


def _exec_comm(op: Operator, ctx: ExecContext) -> OpResult:
    nbytes = op.inputs[0].nbytes
    ...                                          # sol: the speed-of-light tier; kind: the op's class name
    curves = ctx.collective_curve_2gpu
    key = {"AllReduce": "all_reduce", "AllGather": "all_gather", "ReduceScatter": "reduce_scatter"}.get(kind)
    if curves and key in curves and op.group_size == 2 and not sol and ctx.device.nvl_domain_gpus >= 2:
        x = op.out.nbytes / 2 if kind == "AllGather" else nbytes
        return OpResult(name=op.name, op_type=op.op_type, time_us=_curve_us(curves[key], x),
                        bytes=nbytes, bound="nvlink", detail="group=2, measured curve")
    ...                                          # otherwise the ring model, by collective
```

The curve is linear between points, holds its first point below the
table and extends its last slope above it. An all-gather is keyed by
each GPU's input, as the benchmark sized it. The curve applies only to a
group of two, in the calibrated tier, on a device whose calibration has
one: the RTX A6000's. The benchmark also measured reduce-scatter, whose
curve chapter 12's expert layouts use.

## How close is it?

All the evidence is Qwen3-8B in bf16 under vLLM on the RTX A6000 pair,
with NCCL. (vLLM's default for small messages, its own all-reduce
kernel, took about 7 seconds per decode step on this PCIe pair, so every
run disables it.)

**The decode step on the engine's clock.** The cleanest test is a
decode step timed inside vLLM, with a batch of requests decoding
together and nothing else in the step:

```
Table 11.6  The tp=2 decode forward on the engine's clock: Qwen3-8B, two RTX A6000s, vLLM
  batch  measured ms  all-reduce  model comm ms  comm share  model/meas  ring model
      1        13.93        8 KB           1.31          9%       1.012       1.033
      4        15.11       32 KB           2.28         15%       1.026       1.018
      8        17.73       64 KB           3.96         22%       1.001       0.941
     16        20.83      128 KB           5.79         28%       0.995       0.927
     24        23.85      192 KB           7.44         32%       0.986       0.920
     32        26.60      256 KB           8.79         34%       0.977       0.923
     48        32.45      384 KB          11.92         36%       1.009       0.960
     64        38.58      512 KB          14.83         40%       0.963       0.923
  context about 1173 tokens; typical error: measured curve 1.6%, ring model 6.1%
```

In the model, communication is 9% of the step at batch 1 and 40% at
batch 64. With the measured curve every step lands within 0.96–1.03;
the ring model reads 6–8% low from batch 8 to 32, in Table 11.5's gap.
Nothing was fitted to these steps: the curve is a stand-alone
benchmark, and the GEMM and attention constants come from single-GPU
kernels (chapters 3, 4 and 7). But the curve was adopted to close this
gap, so these rows are in-sample for the mechanism.

**Fixed batches.** Chapter 6's grid of batches and prompts, at tp=2:

```
Table 11.7  Qwen3-8B at tp=2 on two RTX A6000s under vLLM: fixed batches, model/measured,
  with nothing hidden and with all communication hidden
  batch prompt   TTFT ms vs 1 GPU  model all hidden   TPOT ms vs 1 GPU  model all hidden
      1    512     122.1     1.53   1.00       0.67     13.90     0.58   1.01       0.91
      1   2048     453.7     1.56   1.03       0.69     14.04     0.58   1.01       0.92
      1   8192    1813.5     1.41   1.07       0.69     14.86     0.58   1.00       0.91
      8    512     880.8     1.56   1.04       0.71     17.50     0.70   0.98       0.76
      8   2048    3517.2     1.54   1.05       0.71     18.75     0.67   0.99       0.78
      8   8192   14563.5     1.40   1.07       0.69     24.42     0.62   0.98       0.82
     32    512    3470.4     1.58   1.05       0.72     24.77     0.83   0.96       0.61
     32   2048   13952.3     1.53   1.06       0.72     30.53     0.73   0.96       0.67
  model: TTFT 1.00-1.07, typical error 4.5%; TPOT 0.96-1.01, typical error 1.7%
  the model's own tp=2 / tp=1: TTFT 1.45-1.66, TPOT 0.58-0.81
  TTFT held out; TPOT in-sample: these cells exposed the host-loop latency, and the 8 us hop was refitted after
  communication in the model's prefill: 67%-68% of the step (an all-reduce of 16 MB per layer for a 2048-token prompt)
```

On this link, tensor parallelism makes prefill slower than one GPU,
1.40–1.58 times as long, because two thirds of a prefill step is
all-reduce at 4 GB/s. Decode is faster, 0.58–0.83 of the single-GPU
time. The model reproduces both directions, cell by cell. The TTFT
cells are held out: the link constants came from NCCL benchmarks, and
predictions committed before the run read 1.01–1.05. The TPOT cells are
in-sample: they exposed the field note's host-loop latency, and the 8 µs
hop was refitted after they were seen:

```
Recorded  Predictions committed before the tp=2 grid ran, with a 41 us hop from a Python-loop benchmark
  TTFT 1.01-1.05; TPOT batch 1 1.38-1.42, batch 8 1.15-1.22, batch 32 1.06-1.10
```

Prefill reads up to 7% high, consistent with the engine's 16 MB
all-reduces running a little faster than the benchmark's (not
profiled).

**Online.** Last, a server fed requests at random (Poisson arrivals,
chapter 14), 1 to 4 per second. Trace H1 has 1,024-token prompts and
128-token replies; H2 has 129–384-token prompts and 257–758-token
replies, keeping decode all-reduces mid-size. The predictions were
committed before either trace ran; the current model reproduces them:

```
Table 11.8  tp=2 online, predictions committed before the runs: Qwen3-8B, two RTX A6000s, vLLM, model/measured
  trace  req/s  TTFT p50  TTFT p95   TPOT  tokens/s  TPOT, ring model
  H1         1      1.04      1.07   1.02      1.00              0.98
  H1         2      1.01      1.03   1.01      1.00              0.91
  H1         3      1.03      1.05   1.05      1.00              0.91
  H1         4      1.05      1.05   1.02      0.99              1.00
  H2         1      1.09      1.06   0.99      1.00              0.91
  H2         2      1.06      1.10   0.99      1.00              0.89
  H2         3      1.03      0.91   0.99      1.01              0.93
  H2         4      0.94      0.95   0.99      1.01              0.94
  TTFT p50 and p95 0.91-1.10; TPOT 0.99-1.05, typical error 1.9%
  bars committed with the predictions: TPOT 0.95-1.05, TTFT 0.85-1.15 below saturation; past its bar: H1 at 3 req/s, TPOT 1.054
```

These are held out: TTFT 0.91–1.10, TPOT 0.99–1.05, throughput within
1%. One cell, H1 at 3 requests per second, sits just past the TPOT bar
committed beforehand (1.054 against 0.95–1.05). The ring model's TPOT
reads up to 11% low. Chapter 9's MoE model at tp=2 (Table 9.11) uses
the same curve.

## Where it breaks

- **Two GPUs, one link.** Every measurement is one PCIe pair running
  NCCL; the curve belongs to that pair and its NCCL version. Nothing
  above tp=2, over NVLink or over InfiniBand has been measured, and
  Tables 11.2–11.4 rest on a round-number hop latency.
- **Custom all-reduce kernels.** vLLM's own all-reduce kernel, its
  default for small messages, is unusable on this pair and unmeasured
  over NVLink.
- **The curve is keyed on the calibration, not the link.** Any
  calibrated RTX A6000 pair at tp=2, even one with an NVLink bridge,
  gets this PCIe pair's prices.
- **Topology and algorithm.** The model has one rate per level. Real
  systems have NVSwitch or PCIe switches, rail-optimized fabrics,
  rack-scale NVLink domains (`nvl_domain` sets one's size), and tree
  algorithms whose latency grows with log n rather than n, which would
  soften decode's limit; NCCL chooses by size and topology.
- **Overlap is declared, not derived,** and measured once, as near
  zero, on one pair.

## What you built

- Tensor parallelism as per-GPU shapes plus explicit collectives: two
  all-reduces per layer, one all-gather per step.
- The ring's closed forms for four collectives, and a hierarchical
  all-reduce across nodes.
- A measured curve for small messages, timed inside a CUDA graph.
- Overlap as a bracket with a declared fraction between its ends.
- Evidence at tp=2 on two RTX A6000s: decode steps within 0.96–1.03,
  fixed batches within 1.00–1.07 (TTFT) and 0.96–1.01 (TPOT), held-out
  online TPOT within 0.99–1.05.
- A projection of where tensor parallelism stops paying: in decode,
  where hop latencies outgrow the shrinking work; in prefill, at the
  node edge.

## Exercises

1. vLLM splits the input embedding by vocabulary too, which adds one
   all-reduce per step that the builder leaves out. Add it. How much
   does it move Table 11.6 and a 2,048-token prefill?
2. Give the ring model three protocols, each a latency and a bandwidth,
   and take the fastest at each size. How close do six constants get to
   the 21 points behind Table 11.5?
3. Load the H100 with `nvl_domain=72` and redo Table 11.3 to tp=64,
   then with a tree's `2·log2(n)` hops instead of `2(n−1)`. Where does
   decode stop paying?
4. On an NVLink pair, measure a tp=2 model under your engine and find
   the `comm_overlap` that matches it.

---

*[← Chapter 10: Precision and sparsity as passes]({% post_url 2026-09-29-tinyperf-10-precision-and-sparsity %}) · [Contents](/series/tinyperf/) · [Chapter 12: Parallel layouts →]({% post_url 2026-09-29-tinyperf-12-parallel-layouts %})*
