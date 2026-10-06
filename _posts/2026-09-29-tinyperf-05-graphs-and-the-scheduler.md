---
layout: post
title: "Building tinyperf, chapter 5: Graphs and the scheduler"
date: 2026-09-29 12:05:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/05-graphs-and-the-scheduler/
excerpt: "Chapter 3 priced one GEMM. A language model runs hundreds of kernels to produce one token: written out operator by operator, a decode step of Llama-2-7B is 419 of them, 193 GEMMs and the rest normalizations, softmax, RoPE (rotary position embeddings), activations, residual adds and an embedding lookup. How do you go from pricing one kernel to pricing a whole model, without running it?"
redirect_from:
  - /tinyperf/perf-modeling/2026/08/18/building-tinyperf-m3.html
  - /tinyperf/perf-modeling/2026/08/18/building-tinyperf-m4.html
---

*[Building tinyperf](/series/tinyperf/) · Part II: One model on one GPU · Code: [`tinyperf/graph.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/graph.py), [`tinyperf/operators.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/operators.py) and [`tinyperf/scheduler.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/scheduler.py), `execute` · Every table and the plot in this chapter come from `python3 book/scripts/ch05_graph.py`.*

Chapter 3 priced one GEMM. A language model runs hundreds of kernels to
produce one token: written out operator by operator, a decode step of
Llama-2-7B is 419 of them, 193 GEMMs and the rest normalizations,
softmax, RoPE (rotary position embeddings), activations, residual adds
and an embedding lookup. How do you go from pricing one kernel to
pricing a whole model, without running it?

Write the model down as a list of operators that carry shapes and data
types but no data, price each with the cost model for its kind, and add
the prices up. The addition is the easy part. The work is in making the
description complete, every FLOP and byte the model spends and none of
them twice, and in being plain about what adding assumes.

By the end of this chapter you will know:

- how a workload is described as a graph of operators, and why a
  performance model wants one;
- the operator families (kinds of operator that share a cost model) and
  how each is priced;
- what the scheduler (the loop that prices each operator) does, and the
  three assumptions hidden in adding up;
- how to read the report, a row of cost per operator, and especially
  its bound column, which names the limit that set each price;
- how to check that a graph is complete before any GPU is involved.

## Describing a model without running it

A performance model never runs the network. For each operation it needs
two answers, how much math and how many bytes, and both follow from
shapes and data types. So its description of a workload, its
*intermediate representation* or IR (as in a compiler), holds three
kinds of thing:

- a **tensor**: a name, a shape and a data type, and no data;
- an **operator**: a node with a type, input tensors and attributes, such
  as a layer's output width. When it is created it works out its outputs'
  shapes (*shape inference*);
- a **graph**: an ordered list of operators.

Building a graph reads like framework code. Here is a two-layer MLP, the
repository's graph example (`examples/03_graph.py`):

```python
g = Graph("mlp")
x = Tensor("input", (4096, 1024), DType.FP16)
h = g.Linear("fc1", x, out_features=4096)
h = g.Elementwise("gelu", h)
h = g.Linear("fc2", h, out_features=1024)
g.RMSNorm("norm", h)
```

Each call creates an operator, infers its output, appends it to the graph
and returns the output tensor for the next call:

```
Worked example  A two-layer MLP, described: shapes inferred, nothing run
  fc1   Linear       gemm  (4096, 1024) -> (4096, 4096) fp16
  gelu  Elementwise  rw    (4096, 4096) -> (4096, 4096) fp16
  fc2   Linear       gemm  (4096, 4096) -> (4096, 1024) fp16
  norm  RMSNorm      rw    (4096, 1024) -> (4096, 1024) fp16
```

Nothing ran: the shapes are all chapter 3's model needs. Weights are not
tensors in the graph; a `Linear` knows its weight is `K × N` from its
input's width and its `out_features`.

Two choices keep the IR small. **Order is the only structure the graph
records.** An operator can only consume tensors that already exist, so
the order of appending is a valid order of running (a *topological
order*), and the scheduler just walks the list. Each operator still
holds its input tensors, so a pass (code that rewrites the graph;
below) can follow producer to consumer.
**A description can take shortcuts.** The attention matmuls take batch,
M, N and K as attributes instead of a second input tensor, because that
operand is the KV cache: state the step reads, not a tensor the graph
produced.

Why an IR at all, rather than a function from a model to a time?

- **One description, many pricings.** The same graph is priced on any
  GPU, at any of chapter 4's tiers (from datasheet peaks alone to
  constants fitted to a real GPU), without being rebuilt (Table 5.5).
- **Passes can rewrite it.** A *pass* is a function that changes a graph.
  Attention runs as one fused kernel, not three (chapter 7); a precision
  recipe runs some GEMMs in FP8 or FP4 (chapter 10); training appends a
  backward pass (chapter 13). Each is a pass, not a second model.
- **Every cost has an address.** Each price belongs to a named operator,
  so a surprising total can be traced to its cause.

## One transformer layer as a graph

tinyperf's transformer builder, `build_llm_graph`, emits a decoder-only
transformer as such a list. Figure 5.1 shows what it holds.

![The sixteen operator records of a Llama-2-7B decode step in graph.ops
order, each with its family and attributes, and the tensor each one
consumes: solid if the operator above produced it, dashed if the builder
made it new.](/assets/tinyperf-book/ch05-layer-graph.svg)

*Figure 5.1. What the graph holds for a Llama-2-7B decode step at batch
1 with 2,048 tokens of context: operator records in `graph.ops` order,
each with its family, class and attributes, and the tensor each
consumes. A solid tensor was produced by the operator above it, a link a
pass can follow. A dashed one is a new tensor the builder made with only
a shape. The thirteen operators of a decoder layer carry `count=32`:
one record for all 32 layers. Chapter 6's Figure 6.1 draws the same
layer as a transformer.*

Every layer has the same shapes, so the builder emits one layer and gives
each of its operators a `count` attribute of 32. Table 5.1 is the whole
graph. Its last row is the LM head, the GEMM that turns the final hidden
state into a score for every token in the vocabulary.

```
Table 5.1  The graph of a decode step: Llama-2-7B, batch 1, context 2048, fp16
  op             operator       family count  shape (gemm: batch x M x N x K)        out
  embed          Embedding      rw         1  (1, 1) -> (1, 1, 4096)                 fp16
  ln_attn        RMSNorm        rw        32  (1, 4096) -> (1, 4096)                 fp16
  qkv_proj       Linear         gemm      32  1 x 1 x 12288 x 4096                   fp16
  rope           RoPE           rw        32  (1, 4096) -> (1, 4096)                 fp16
  attn_qk        BatchedMatMul  gemm      32  32 x 1 x 2048 x 128                    fp32
  attn_softmax   Softmax        rw        32  (32, 1, 2048) -> (32, 1, 2048)         fp16
  attn_pv        BatchedMatMul  gemm      32  32 x 1 x 128 x 2048                    fp16
  attn_out       Linear         gemm      32  1 x 1 x 4096 x 4096                    fp16
  residual_attn  Elementwise    rw        32  (1, 4096) + (1, 4096) -> (1, 4096)     fp16
  ln_ffn         RMSNorm        rw        32  (1, 4096) -> (1, 4096)                 fp16
  ffn_gate_up    Linear         gemm      32  1 x 1 x 22016 x 4096                   fp16
  swiglu         Elementwise    rw        32  (1, 22016) -> (1, 22016)               fp16
  ffn_down       Linear         gemm      32  1 x 1 x 4096 x 11008                   fp16
  residual_ffn   Elementwise    rw        32  (1, 4096) + (1, 4096) -> (1, 4096)     fp16
  ln_final       RMSNorm        rw         1  (1, 4096) -> (1, 4096)                 fp16
  lm_head        Linear         gemm       1  1 x 1 x 32000 x 4096                   fp16
  16 operators stand for 419 operator calls: 193 GEMMs and 226 memory-bound ops
  on the A100, 420 kernels: chapter 3's model splits the head's K and adds a reduction kernel
```

The GEMM rows read `batch × M × N × K` as in chapter 3; chapter 6
derives every shape from the batch, the context and the configuration.
The count is not an approximation: as long as each call is priced alone,
the second assumption below, 32 identical calls cost exactly 32 times
one. It also keeps pricing fast, at 16 operators instead of 419.

## Operator families

Every operator declares a *family* (its `op_type`), and the family
decides which cost model prices it. Adding an operator means choosing its
family, and its price comes with it.

```
Table 5.2  Operator families: which operators, and what prices them
  family   operators                                        priced by
  gemm     Linear, BatchedMatMul                            chapter 3's GEMM model: max(math, DRAM, L2) + launch
  rw       Elementwise, Softmax, RMSNorm, RoPE, Embedding   (bytes read + bytes written) / DRAM bandwidth + launch
  comm     AllReduce, AllGather, ReduceScatter, AllToAll    chapter 11's model: hop latency + bytes / link bandwidth + launch
  fmha     FusedAttention                                   chapter 7's fused-attention model
  linattn  LinearAttention                                  chapter 8's linear-attention model
```

**GEMMs.** A `BatchedMatMul` is many GEMMs of one shape; in attention
the batch is the sequences times the KV heads (key-value heads, which
several query heads share; chapter 6). It and `Linear` both go to
chapter 3's `estimate_gemm`.

**Memory-bound operations**, the `rw` (read-write) family: activations,
residual adds, normalizations, softmax, RoPE and the embedding lookup.
Their price is their traffic, and their math is ignored, for a reason.
An RMSNorm (the normalization these models use: each row divided by
its root mean square) does a handful of FLOPs per element and moves 4 bytes
for it in fp16: about one FLOP per byte. Such operations run on the
ordinary FP32 cores of each SM (streaming multiprocessor), and even
there the ridge point (the FLOPs per byte above which math, not memory,
sets the time; chapter 2) is far above one:

```
Worked example  Pricing memory-bound ops and a collective: A100, datasheet rates
  ridge point, fp16 tensor cores (312 TFLOPS): 153 FLOP per byte
  ridge point, FP32 cores (64 lanes per SM, 19.5 TFLOPS): 9.6 FLOP per byte
  ln_attn, decode, 1 token     :   0.016 MB / 2039 GB/s =   0.01 us + 3.0 us launch =   3.01 us
  ln_attn, prefill, 2048 tokens:  33.554 MB / 2039 GB/s =  16.46 us + 3.0 us launch =  19.46 us
  at tp=2: 64 all-reduces of 8.2 KB per decode step, 5.03 us each (2.0 us of hop latency, 3.0 us launch)
```

(The 64 FP32 lanes per SM are from NVIDIA's A100 whitepaper.) In decode
an RMSNorm's price is its launch, the fixed per-kernel cost of chapter 3;
in a 2,048-token prefill it moves 33.6 MB. An `extra_bytes` attribute
covers traffic the tensors don't show, such as the rows an embedding
lookup gathers.

**Collectives** move data between GPUs and appear only when a model is
split across them; chapter 11 prices them. The last line above is
Llama-2-7B's decode step on two A100s under tensor parallelism (tp=2:
each GPU holds half of every weight matrix; chapter 11): two small
all-reduces per layer (each leaves the sum of the GPUs' partial results
on both), priced almost entirely as launch and hop latency, the fixed
cost of each hand-off between GPUs.

**Fused kernels.** `FusedAttention` (chapter 7) is what the
attention-fusion pass makes of Figure 5.1's three attention operators, as
FlashAttention (the fused attention kernel engines run) does on the GPU.
`LinearAttention` (chapter 8) prices the recurrent layers of hybrid
models, which keep a fixed-size state instead of a KV cache in some of
their layers.

## The scheduler: dispatch and add

The scheduler walks the operators in order, sends each to its family's
cost model through a dispatch table, scales the result by the count, and
adds:

```
total = Σ over operators  count(op) · price[family(op)](op, device)
```

That is the whole algorithm, and it rests on three assumptions.

1. **Kernels run one after another,** with no overlap. For one stream of
   kernels on one GPU this is close to what happens. It fails where work
   runs concurrently, such as a collective beside computation, which
   chapter 11 brackets between this sum and perfect hiding.
2. **Each kernel is priced alone.** Nothing the previous kernel left in L2
   helps the next one, and a long run of GEMMs doesn't heat the GPU into a
   lower clock. This is what makes `count` exact.
3. **Every kernel pays its launch, in series:** chapter 3's per-kernel
   cost, 3 µs by default on every GPU in this chapter. Chapter 4 measures
   it for kernels issued one at a time and for kernels replayed from a
   CUDA graph (a recording of many launches replayed as one, unrelated
   to this chapter's graph).

The scheduler also takes one of chapter 4's three tiers: speed of light,
the projected model built so far, or constants fitted to a real GPU.

## The report

Design the report first: fix the shape of the answer before the models
that fill it in. Here it is one row per operator with its time, FLOPs,
bytes, achieved TFLOPS, the *bound* (what limits it) and a detail string
(the shape and kernel choice), then totals and a roll-up by family. Every
cost model has one job, to fill in a row, and reports the FLOPs and bytes
it charged; Table 5.6's FLOP rows come straight from the report.

Table 5.3 is the report for Table 5.1's graph on an A100 at its datasheet
rates. Time, GFLOP and MB are totals over the count; TFLOPS is per call.
The detail column is left out.

```
Table 5.3  The report: Llama-2-7B decode step, batch 1, context 2048, A100 at datasheet rates
  op             family count   time us   share     GFLOP        MB  TFLOPS  bound
  ffn_gate_up    gemm      32    2929.2   35.2%      5.77    5777.0     2.0  dram
  qkv_proj       gemm      32    1678.2   20.1%      3.22    3226.2     1.9  dram
  ffn_down       gemm      32    1516.9   18.2%      2.89    2897.2     1.9  dram
  attn_out       gemm      32     624.8    7.5%      1.07    1078.2     1.7  dram
  attn_pv        gemm      32     392.3    4.7%      0.54     604.2     1.4  dram
  attn_qk        gemm      32     365.5    4.4%      0.54     549.5     1.5  dram
  lm_head        gemm       1     138.5    1.7%      0.26     270.2     1.9  dram
  attn_softmax   rw        32     102.2    1.2%      0.00      12.6     0.0  dram
  8 more ops     rw               584.9    7.0%                 6.0          dram
  TOTAL                   419    8332.6  100.0%     14.29   14421.1
  by family: gemm 7645.5 us (91.8%), rw 687.1 us (8.2%)
  of which launches: 420 kernels x 3.0 us = 1.26 ms, 15% of the step
```

The layer's four weight GEMMs take 81% of the step, all bound by DRAM,
as chapter 6 explains for decode. The eight memory-bound operators at the
bottom move 6.0 MB between them, and their 584.9 µs is nearly all launch.
The table's last line counts every launch: 420 kernels paying the
default 3 µs each, in series. What a launch really costs depends on how
kernels are issued (chapter 4).

A prefill looks different:

```
Table 5.4  The report: Llama-2-7B prefill, one prompt of 2048 tokens, A100 at datasheet rates
  op             family count   time us   share     GFLOP        MB  TFLOPS  bound
  ffn_gate_up    gemm      32   39371.1   35.1%  11819.75   15971.9   300.2  math
  qkv_proj       gemm      32   23360.4   20.8%   6597.07   11475.6   282.4  l2
  ffn_down       gemm      32   20735.7   18.5%   5909.87   10005.5   285.0  l2
  attn_out       gemm      32    7850.8    7.0%   2199.02    4060.1   280.1  l2
  attn_softmax   rw        32    6421.4    5.7%      0.00   12897.5     0.0  dram
  attn_qk        gemm      32    4724.3    4.2%    550.29    9437.2   116.5  dram
  swiglu         rw        32    2926.5    2.6%      0.00    5771.4     0.0  dram
  attn_pv        gemm      32    2877.1    2.6%    550.29    5670.7   191.3  dram
  8 more ops     gemm/rw          3800.6    3.4%              6746.2          dram
  TOTAL                   419  112068.0  100.0%  27626.57   82036.1
  by family: gemm 99058.0 us (88.4%), rw 13010.0 us (11.6%)
  attention as built (QK^T, softmax, PV): 14.0 ms, 12.5% of the step
```

The same GEMMs lead, now near the tensor cores' peak (chapter 6 has
prefill's arithmetic). Attention takes 12.5% of the step, most of it
traffic, because the builder writes attention the way the math reads:
`QKᵀ` writes a score matrix of 32 heads × 2,048 queries × 1,025 keys (the
causal average; chapter 6) in fp32, softmax reads it and writes it back
in fp16, and the second GEMM reads it again. No modern engine runs
attention that way; chapter 7's fusion pass replaces the three operators
with one kernel.

Figure 5.2 splits both reports four ways, with each row's launch counted
apart. That is why the weight GEMMs, the head included, show 78% of the
decode step where Table 5.3's four layer GEMMs alone add to 81%: their
launches are in the grey segment.

![Stacked bars of the share of step time in weight GEMMs, attention,
other memory-bound operations and launch overhead, for a decode step and
a prefill.](/assets/tinyperf-book/ch05-time-breakdown.svg)

*Figure 5.2. Where the time goes, Llama-2-7B on the A100, with each
report row's launch overhead counted apart. A decode step at batch 1 is
weight streaming plus launches; its memory-bound operators cost almost
nothing beyond their launches. A prefill is the weight GEMMs' math, with
12% in attention as built.*

### Reading the bound column

The bound column names the term that set an operator's price: `math`,
`dram` or `l2`, chapter 3's three terms, or `nvlink` (the link between
GPUs) for a collective. It is the report's most useful column, because
it turns a time into a direction. A DRAM-bound decode step gets faster
with fewer bytes, such as smaller weights (chapter 10), not with more
FLOPS.

Read it with two cautions. It is the model's verdict, not a measurement:
a memory-bound operator says `dram` by construction. And when two terms
are close, the verdict is fragile. Here are the prefill GEMMs, each with
its tile, the output block one CTA (thread block) computes (chapter 3):

```
Worked example  The bound column up close: prefill GEMMs, one call each, A100
  op               tile  math us  DRAM us    L2 us  bound  runner-up
  qkv_proj      128x128    708.1    175.9    727.0  l2     math, 2.6% below
  attn_out      128x128    236.0     62.2    242.3  l2     math, 2.6% below
  ffn_gate_up   256x128   1227.3    244.8    981.9  math   l2, 20.0% below
  ffn_down      128x128    628.2    153.3    645.0  l2     math, 2.6% below
  the A100's L2 bandwidth: 4500 GB/s, which tinyperf/device.py marks as approximate
```

Three of the four say `l2`, with math 2.6% behind. A 3% higher L2
bandwidth, a number the device model marks as approximate, would flip
them to `math` and change their time by under 3%. Read `l2` here as
"math and L2 traffic both near their limits". Only `ffn_gate_up` is
clearly math-bound: on 256 × 128 tiles each CTA loads less per unit of
math, which puts its L2 time 20% below its math. When the bound matters
to a decision, look at the runner-up.

## One description, many pricings

Table 5.5 prices the same two graphs on three GPUs, at the projected
tier and at speed of light (every operator at its pure roofline, the
larger of its math time and its memory time, with no launch cost;
chapter 4), and after the attention-fusion pass.

```
Table 5.5  One graph, many pricings: Llama-2-7B, ms, datasheet rates
                                          A100      H100      B200
  decode, batch 1, context 2048           8.33      5.56      3.06
    at speed of light                     7.02      4.27      1.79
    after the attention-fusion pass       8.10      5.34      2.86
  prefill, 2048 tokens                  112.07     42.10     23.90
    at speed of light                   104.63     38.72     16.95
    after the attention-fusion pass     103.57     35.27     20.97
```

In decode the projected price sits about 1.3 ms above speed of light on
all three GPUs. Nearly all of that is the 1.26 ms of launches, which
don't get faster with the GPU; on the B200 they are 41% of the step.
Fusion takes 8% off an A100 prefill and 3% off a decode step. On the
A100 and H100 the fused prefill even comes in under speed of light:
speed of light prices the unfused graph, so fusion can beat it.

> **Field note: the graph knows its precision.** `execute` once took the
> math precision as an argument that defaulted to fp16. Build a graph in
> fp8, forget to say so when pricing it, and fp8 bytes were priced
> against fp16 tensor cores, silently. The current model puts a number on
> that mistake: an fp8 Llama-2-7B prefill of 2,048 tokens on the H100
> costs 24.2 ms at its own precision and 39.4 ms when told fp16, 62%
> high. Now `execute` reads the precision from the graph, and an argument
> only overrides it. Anything a price depends on belongs in the
> description, where it can't be forgotten.

## The code

### The graph

Here is `graph.py`, trimmed of its docstrings, of `numel` (the product of
the shape) and of small helpers (comments ours):

```python
@dataclass
class Tensor:
    name: str
    shape: tuple
    dtype: DType

    @property
    def nbytes(self) -> float:
        return self.numel * self.dtype.nbytes


class Operator:
    op_type = "base"

    def __init__(self, name: str, inputs: list, **attrs):
        self.name = name
        self.inputs: list[Tensor] = inputs
        self.attrs = attrs
        self.outputs: list[Tensor] = self.build_outputs()


class Graph:
    def __init__(self, name: str = "net"):
        self.name = name
        self.ops: list[Operator] = []

    @classmethod
    def register(cls, op_cls):
        def builder(self, name, *inputs, **attrs):
            op = op_cls(name, list(inputs), **attrs)
            self.add_op(op)                   # appends to self.ops
            return op.out                     # outputs[0]

        setattr(cls, op_cls.__name__, builder)
        return op_cls
```

`register` is the trick behind the builder calls: on `class Linear` it
creates `g.Linear(...)`, which builds the operator (running shape
inference), appends it and returns its output. Every attribute the caller
passes, `count` included, lands in `op.attrs`.

### Operators

A GEMM, and the base of the memory-bound family, from `operators.py`:

```python
@Graph.register
class Linear(Operator):
    op_type = "gemm"

    def build_outputs(self):
        (x,) = self.inputs
        n = self.attrs["out_features"]
        self.m = math.prod(x.shape[:-1])
        self.k = x.shape[-1]
        self.n = n
        self.batch = 1
        self.weight_bytes = self.k * self.n * x.dtype.nbytes
        return [Tensor(self.name + ".out", x.shape[:-1] + (n,), x.dtype)]


class GenericRW(Operator):
    op_type = "rw"
    rw_multiplier = 1.0

    def moved_bytes(self) -> float:
        rd = sum(t.nbytes for t in self.inputs)
        wr = sum(t.nbytes for t in self.outputs)
        extra = self.attrs.get("extra_bytes", 0.0)
        return (rd + wr) * self.rw_multiplier + extra
```

Each memory-bound operator subclasses `GenericRW` and differs only in its
output's shape and type; `rw_multiplier` is 1.0 for all of them. Nothing
in the scheduler reads `weight_bytes`, since chapter 3's model works out
a GEMM's traffic itself; it is what lets the last section check a graph
against the model's parameters.

### The builder

The feed-forward block of `build_llm_graph` in
`tinyperf/nets/transformer.py`, its dense branch (the experts' branch is
chapter 9's), shows the pattern of the whole builder:

```python
    y = g.RMSNorm("ln_ffn", x, count=L)
    if p.n_experts:
        ...
    else:
        up = g.Linear("ffn_gate_up", y, out_features=2 * ffn_l, count=L)
        act = g.Elementwise("swiglu", up, count=L)
        act = Tensor("act", (tokens, ffn_l), dt)
        down = g.Linear("ffn_down", act, out_features=h, count=L)
        if tp > 1:
            down = g.AllReduce("ffn_allreduce", down, group_size=tp, count=L)
        x = g.Elementwise("residual_ffn", down, x, count=L)
```

`tokens` is the step's rows, `h` the hidden width and `ffn_l` the
feed-forward width on this GPU. An `Elementwise` keeps its input's
shape, so the SwiGLU's output is `2·ffn_l` wide, where the real
activation halves it: it multiplies the gate half of its input,
activated, by the up half. `ffn_down` needs an input `ffn_l` wide, so
the builder replaces the SwiGLU's output with a new tensor of that
shape: one of Figure 5.1's dashed tensors. The shortcut has a price,
which "Where it breaks" counts. Chapter 6 walks through the rest of the
builder.

### The scheduler

A row of the report is an `OpResult`; a `RunReport` holds the rows, and
its `total_us` is the sum of their `total_us`. Trimmed of two fields the
energy model uses:

```python
@dataclass
class OpResult:
    name: str
    op_type: str
    time_us: float
    flops: float = 0.0
    bytes: float = 0.0
    bound: str = ""
    detail: str = ""
    count: int = 1  # replication factor (e.g. identical decoder layers)

    @property
    def total_us(self) -> float:
        return self.time_us * self.count
```

`execute`, trimmed of its docstring, its type annotations and the
constants a calibration, one GPU's record of fitted constants, carries
for particular kernels (chapter 4 and later; comments ours):

```python
EXEC_MODELS = {"gemm": _exec_gemm, "rw": _exec_rw, "comm": _exec_comm,
               "fmha": _exec_fmha, "linattn": _exec_linattn}


def execute(graph, device, dtype=None, methodology=Methodology.PROJ, calibration=None):
    if dtype is None:
        dtype = _graph_dtype(graph)
    if methodology is Methodology.CALIBRATED:
        cal = calibration or Calibration.load(device.name)
        ...                                   # no calibration: an error, never a silent fallback
        device = cal.apply(device)
    ctx = ExecContext(device=device, dtype=dtype, methodology=methodology, ...)
    report = RunReport(device=device.name, methodology=methodology.value)
    for op in graph:
        result = EXEC_MODELS[op.op_type](op, ctx)
        result.count = op.attrs.get("count", 1)
        report.rows.append(result)
    return report
```

`_graph_dtype`, the field note's fix, returns the precision of the input
of the first GEMM or fused-attention operator. Each family's price maps
an operator and a context to an `OpResult`. The memory-bound one, whole:

```python
def _exec_rw(op: Operator, ctx: ExecContext) -> OpResult:
    nbytes = op.moved_bytes()
    launch = 0.0 if ctx.methodology is Methodology.SOL else ctx.device.kernel_launch_us
    time_us = ctx.device.dram_time_s(nbytes) * 1e6 + launch
    return OpResult(
        name=op.name, op_type=op.op_type, time_us=time_us,
        bytes=nbytes, bound="dram",
    )
```

`_exec_gemm` is, at its core, one call,
`estimate_gemm(device, op.m, op.n, op.k, dtype, batch=op.batch, out_dtype=op.out.dtype)`,
whose result fills the row. It also carries branches for particular
kernels: scale factors for mixture-of-experts kernels (chapter 9) and
weight-only kernels (4-bit weights, 16-bit math; chapter 10), and
measured corrections, among them the row curve, cuBLAS's measured
correction by row count for a serving engine's decode-step GEMMs
(chapter 15). They sit in
lines 210–252 of `scheduler.py`, and those chapters explain them.

## How do we know the graph is complete?

Whether these totals match a real engine is for chapters 6 and 15. Two
earlier questions need only arithmetic, and every later comparison
depends on them: does the graph hold all of the model's work and no
more, and does the scheduler add what it says?

For the first, we compute FLOPs, weight bytes and KV-cache bytes from the
configuration alone. Every matmul weight does two FLOPs per token, and
attention does two matmuls against `heads · head_dim` numbers per key:

```
decode FLOPs per token  = 2 · (body weights + head weights) + 4 · layers · context · (heads · head_dim)
prefill FLOPs per token = 2 · body weights + 4 · layers · ceil((s+1)/2) · (heads · head_dim)
                          + 2 · head weights / s
```

The body weights, all layers together, are
`layers · (h · (q + 2·kv) + q · h + 3 · h · ffn)`, with q and kv the
widths of the query and KV projections. In a causal prompt of s tokens a
query sees (s+1)/2 keys on average, which the builder rounds up, and the
head runs once, for the last token. We check Llama-2-7B, with keys and
values for every head, and Qwen3-8B, whose 32 query heads share 8 KV
heads (grouped-query attention; chapter 6).

```
Table 5.6  Is the graph complete? The graph against arithmetic on the model's configuration
                                                     llama2-7b            qwen3-8b
                                                graph  by hand      graph  by hand
  GFLOP per token, decode at 2048              14.288   14.288     16.344   16.344
  GFLOP per token, prefill of 2048             13.490   13.490     14.497   14.497
  GB of weights the GEMMs stream               13.214   13.214     15.136   15.136
  GB of KV cache a decode step reads            1.074    1.074      0.302    0.302
  largest |graph / by hand - 1| in the table: 0.0e+00
```

The graph matches exactly on both models: every weight is in one GEMM,
and the cache is read once per step, a quarter as much per layer for
Qwen3-8B's 8 KV heads as for Llama-2-7B's 32.

For the second question, we compare each report's total with a
re-pricing that bypasses the scheduler (same GEMM model, bytes over
bandwidth for the rest) and with the sum of its printed rows:

```
Check  Six reports on the A100: two models; decode at batch 1 and 32, and prefill of 2048
  total minus a re-pricing that bypasses the scheduler: under 1e-9 us; minus its printed rows: at most 0.3 us
```

The re-pricing agrees to floating-point rounding, and the printed rows
add up to the total within their own rounding.

None of these numbers is a measurement, and nothing in them was fitted.
They are consistency checks: passing proves little, failing proves the
graph wrong. The repository keeps loose versions as its first anchor
tests (`tests/test_core.py`): prefill FLOPs per token within 0.8–1.5
times 2 × parameters, and a batch-1 A100 decode step between 1 and 2.5
times the time to stream the weights.

## Where it breaks

- **The serial sum.** Real GPUs overlap more than the sum admits:
  collectives run on their own stream beside computation (chapter 11).
  The sum also misses time no operator owns, such as a serving engine's
  CPU work between steps (chapter 16).
- **Operators priced alone.** A decode step's activations fit easily in
  the A100's 40 MB L2, so the next operator likely finds its input there
  (not profiled), but the model charges DRAM. In decode that is noise;
  between the large GEMMs of a prefill it need not be.
- **A fixed launch per kernel.** Every operator pays 3 µs in series, and
  3 µs is a default, not a measurement. On the RTX A6000
  (`data/calibration/rtx_a6000.json`) the measured per-kernel cost is
  25 µs when PyTorch issues kernels one at a time and 3.5 µs when a CUDA
  graph replays them (chapter 4).
- **The graph is the math, not the engine's launch list.** An engine
  fuses some operators, splits others, and runs kernels the math doesn't
  have, such as sampling. Chapter 15 builds a decode step from what an
  engine launches.
- **The checks don't see activations.** Table 5.6 checks FLOPs, weights
  and the KV cache, not activation traffic, and the builder's shortcuts
  err in both directions. SwiGLU keeps its input's shape (Table 5.1), so
  it is charged for writing 22,016 columns where the real activation
  writes 11,008: a third more traffic than the real kernel moves. RoPE
  is priced on the queries alone (`g.RoPE("rope", q)`), though the keys
  are rotated too: too little traffic.

## What you built

- A graph IR in three small classes, a builder API through
  `@Graph.register`, and a `count` that lets one layer stand for all.
- Five operator families, each with its cost model.
- A scheduler that dispatches by family and adds, on three stated
  assumptions: serial kernels, each priced alone, each paying its launch.
- A report with one row per operator, and the habit of reading the bound
  with its runner-up.
- Evidence: FLOPs, weight bytes and KV-cache bytes that match the
  configuration's arithmetic exactly on two models, and totals that
  equal a re-pricing that bypasses the scheduler.

## Exercises

1. Fuse memory-bound operators. Write a pass that merges each residual
   add with the RMSNorm after it into one operator, a kernel that reads
   the stream and the residual and writes the sum and the normalized
   output. How many kernels does it remove from Table 5.1, and how much of
   Table 5.3's total? Then fix SwiGLU's output width and RoPE's inputs,
   and see what each is worth in Table 5.4.
2. Diff two reports. Price Table 5.1's graph on the A100 and the H100 and
   print each operator's speedup. Which operators speed up by the ratio of
   DRAM bandwidths, which by the ratio of tensor-core peaks, and which
   barely at all? Explain each from its bound.
3. Relax "priced alone". Let an operator read its first input at L2
   bandwidth when the previous operator's output fits in the usable L2
   (chapter 3's 80%). How much does a batch-1 decode step change? A
   prefill of 256 tokens? What happens to `count`?
4. Check an MoE (mixture-of-experts) graph. Repeat Table 5.6 for
   Mixtral-8x7B (chapter 9). Does the FLOPs row match its total or its
   active parameters? Which do the weight bytes match at batch 1, and at
   batch 64?

---

*[← Chapter 4: Meeting a real GPU: calibration and its tiers]({% post_url 2026-09-29-tinyperf-04-meeting-a-real-gpu %}) · [Contents](/series/tinyperf/) · [Chapter 6: A transformer: prefill, decode and memory →]({% post_url 2026-09-29-tinyperf-06-a-transformer-prefill-decode-and-memory %})*
