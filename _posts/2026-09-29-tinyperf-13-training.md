---
layout: post
title: "Building tinyperf, chapter 13: Training"
date: 2026-09-29 12:13:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/13-training/
excerpt: "Every chapter so far priced inference. A training step does more with the same model: a forward pass over a batch of sequences, a loss, a backward pass that turns the loss into a gradient for every weight, and an optimizer update of every weight. It keeps more in memory, and at scale it is spread over hundreds of GPUs in three ways at once: copies of the model, stages of layers, and slices of every layer (chapter 11's tensor parallelism). What does one training step cost, in time and memory, and how do data, pipeline and ZeRO parallelism change it?"
redirect_from:
  - /tinyperf/perf-modeling/2026/08/21/building-tinyperf-m12.html
  - /tinyperf/perf-modeling/2026/08/28/building-tinyperf-m21.html
  - /tinyperf/perf-modeling/2026/09/08/building-tinyperf-m50.html
---

*[Building tinyperf](/series/tinyperf/) · Part III: Many GPUs · Code: [`tinyperf/passes.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/passes.py), `add_backward`, and [`tinyperf/training.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/training.py) · Every table and figure in this chapter comes from `python3 book/scripts/ch13_training.py`.*

Every chapter so far priced inference. A training step does more with
the same model: a forward pass over a batch of sequences, a loss, a
backward pass that turns the loss into a gradient for every weight, and
an optimizer update of every weight. It keeps more in memory, and at
scale it is spread over hundreds of GPUs in three ways at once: copies
of the model, stages of layers, and slices of every layer (chapter 11's
tensor parallelism). What does one training step cost, in time and
memory, and how do data, pipeline and ZeRO parallelism change it?

The short answer. A step costs about three forward passes, 6 FLOPs per
parameter per token. Mixed-precision Adam keeps 16 bytes per parameter,
2 of them the weights, plus activations that grow with the tokens in
flight. Copies of the model add a gradient all-reduce, hidden under the
backward once each GPU's micro-batch holds more tokens than its
achieved FLOP rate divided by its link bandwidth (about 1,400 on eight
H100s), whatever the model's size. ZeRO divides the 16 bytes among the
copies. With the model cut into pp stages and m micro-batches per step,
each GPU idles for a *bubble* of (pp − 1)/(m + pp − 1) of the step,
which newer schedules shrink by spending memory. No training step has
been measured for this repository: every time here is a projection, and
the evidence is arithmetic.

By the end of this chapter you will know:

- why the backward costs twice the forward, where the 6 FLOPs come
  from, and what else the graph adds;
- what mixed-precision Adam stores and moves per parameter, and what
  each ZeRO stage shards;
- how activations are counted, and what recomputation and sequence
  parallelism save;
- when data parallelism's gradient all-reduce is exposed;
- the pipeline bubble, and what each schedule buys and pays.

## A training step

A step processes a *global batch* of sequences, cut into *micro-batches*,
the sequences one GPU runs through the model at a time. Each micro-batch
runs forward, computes the loss and runs backward, adding its gradients
to the ones before (*gradient accumulation*). After the last, the copies
of the model average their gradients and the optimizer updates the
weights.

In tinyperf a micro-batch is a graph (chapter 5) with a prefill's shape
(chapter 6): every position at once, causal attention. One argument
differs: every position has a next token to predict, so every position
needs its logits for the loss, `build_llm_graph(..., logits="all")`.
`add_backward(g, optimizer_bytes=...)` then appends the backward, and
the rest of the step is arithmetic in `training.py`.

## The backward pass as a graph transform

A layer's GEMM computes `Y = X · W`: X holds the micro-batch's
activations, M tokens by K, and W is the weight, K by N. The backward
receives dY, the gradient of the loss with respect to Y, and produces
two things. The layer below needs `dX = dY · Wᵀ`; the optimizer needs
`dW = Xᵀ · dY`. Both are GEMMs over the forward's three dimensions,
permuted, so each costs the forward's `2·M·N·K` FLOPs.

![Three GEMMs side by side: the forward X times W, the input gradient dY
times W-transpose, and the weight gradient X-transpose times dY, each
with its shapes.](/assets/tinyperf-book/ch13-backward.svg)

*Figure 13.1. One GEMM forward, two backward. The input gradient (D)
flows to the layer below; the weight gradient (W) waits for the
optimizer. All three do `2·M·N·K` FLOPs. (D) and (W) name backward
chunks, not matrices.*

A weight does one multiply-add per token forward, 2 FLOPs, and two
backward, 4 more. A step over T tokens of a model with P parameters
costs about `6·P·T` FLOPs (the rule usually written 6N).

That rule counts only GEMMs. tinyperf's backward is a pass that walks
the forward graph in reverse and appends a backward operator for each
forward one, by family. Here it is without its docstring, its two
imports and the branch for chapter 8's linear attention:

```python
def add_backward(graph: Graph, optimizer_bytes: float = 0.0) -> int:
    fwd_ops = list(graph.ops)
    added = 0
    for op in reversed(fwd_ops):
        c = op.attrs.get("count", 1)
        if op.op_type == "gemm":
            graph.BatchedMatMul(op.name + "_dgrad", op.out, batch=op.batch,
                                m=op.m, n=op.k, k=op.n, count=c)
            graph.BatchedMatMul(op.name + "_wgrad", op.out, batch=op.batch,
                                m=op.k, n=op.n, k=op.m, count=c)
            added += 2
        elif op.op_type == "fmha":
            graph.FusedAttention(op.name + "_bwd", op.inputs[0], batch=op.batch,
                                 m=2 * op.m, kv=op.kv, k=op.k,
                                 out_dim=op.out_dim, count=c)
            added += 1
        elif op.op_type == "comm":
            builder = getattr(graph, type(op).__name__)
            builder(op.name + "_bwd", op.inputs[0], group_size=op.group_size, count=c)
            added += 1
        ...
        else:  # rw
            graph.Elementwise(op.name + "_bwd", op.out,
                              extra_bytes=0.5 * op.moved_bytes(), count=c)
            added += 1
    if optimizer_bytes:
        state = Tensor("opt_state", (1,), DType.FP32)
        graph.Elementwise("optimizer_step", state, extra_bytes=optimizer_bytes)
        added += 1
    return added
```

- **GEMMs** get Figure 13.1's two: `_dgrad` makes M × K from a
  contraction over N, `_wgrad` makes K × N over M.
- **Fused attention** (chapter 7) gets one backward kernel with twice
  the query rows, twice the forward's FLOPs. That is the model-FLOPs
  convention. The kernel itself recomputes the scores and then runs four
  more products (dV, dP, dQ, dK), five to the forward's two, so its time
  is likely a fifth low.
- **Collectives** are mirrored one for one, same type. Under tensor
  parallelism (chapter 11) the true backward swaps all-reduce and
  pass-through, but the count, two per layer, is the same.
- **Memory-bound operators** (chapter 5's `rw` family) get a mirror that
  moves 1.5 times the bytes: it reads the saved input and the incoming
  gradient and writes the outgoing one.
- **The optimizer** is one memory-bound operator whose bytes the caller
  passes in.

The names matter. Pipeline schedules need a micro-batch's time in
three parts: the forward F; the input-gradient backward D, which the
stage before waits for; and the weight-gradient backward W, which
nothing waits for. `split_fwd_bwd` sums a priced graph's rows by name:
`_wgrad` into W, `_dgrad` and `_bwd` into D, `optimizer_step` into O,
the rest into F.

Table 13.1 checks the graph's FLOPs, as chapter 5 did for a forward.

```
Table 13.1  A training step's FLOPs: the graph against arithmetic, one 4096-token sequence
                                                            llama2-7b   qwen3-8b
  parameters P, billions                                         6.74       8.19
  graph: forward GFLOP per token                                14.29      16.34
  graph: forward + backward GFLOP per token                     42.87      49.03
    forward + backward over forward                             3.000      3.000
  by hand: 6 x (P - embedding) + attention                      42.87      49.03
    6 x P                                                       40.43      49.14
    less 6 x embedding, which does no math                      -0.79      -3.73
    plus attention, 12 x layers x heads x head_dim x 2049        3.22       3.63
  with logits for the last position only                        42.08      45.30
  largest |graph / by hand - 1|, both models, both logits settings: under 1e-9
```

The ratio is exactly 3, and the count by hand matches: 6 FLOPs for
every weight that does math, plus attention's forward, 4 · heads ·
head_dim FLOPs per layer for each of the about 2,049 keys a query sees
(chapter 6), tripled by its backward. The 6·P rule errs differently on
each: for Llama-2-7B it is 6% low, mostly attention; for Qwen3-8B the
embedding, a lookup that does no math, nearly cancels attention.

The last row is the price of forgetting `logits="all"`. Qwen3-8B's LM
head does 3.73 of the step's 49.03 GFLOP per token, 8%, and a head
priced for one position per sequence drops them. Now time, on an H100
at datasheet rates, chapter 4's projected tier. *MFU*, model FLOPs
utilization, is the graph's FLOPs divided by what the GPU could do at
peak in the same time.

```
Table 13.2  One training micro-batch, one 4096-token sequence, one H100 at datasheet rates: ms
                                                llama2-7b   qwen3-8b
  forward (F)                                        73.9       84.8
  input-gradient backward (D)                        83.3       95.6
    GEMMs (dgrad)                                    58.6       67.1
    fused attention backward                         13.8       15.5
    memory-bound ops' backward                       10.9       13.0
  weight-gradient backward (W)                       59.3       67.9
  (D + W) / F                                        1.93       1.93
  optimizer step (O), once per step                  52.3       63.5
  step: F + D + W + O                               268.8      311.8
  MFU of F + D + W                                    82%        82%
  logits for every position cost, vs last only      +1.2%      +5.2%
```

The backward is 1.93 forwards, not 2, because the memory-bound mirrors
move 1.5 times their forward's bytes. D is 40% larger than W on both
models: W is GEMMs only, while D also holds attention's backward and the
memory-bound mirrors. The zero-bubble schedules (below) depend on that
split.

The 82% is the projected tier's, which is optimistic (chapter 4). The
optimizer's 52.3 ms is a fifth of this step, but it is paid once per
step, however many micro-batches the step accumulates.

## Memory: sixteen bytes per parameter, and the activations

Mixed-precision training computes in bf16 but can't update in bf16: an
update is often smaller than a bf16 weight's rounding step and would be
lost. So the optimizer keeps an fp32 *master* copy of each weight, and
Adam keeps two fp32 running averages per weight, of the gradient and of
its square. *Data parallelism* (dp, next section) runs dp copies of the
model, each on its own micro-batches, and holds all of this dp times.
*ZeRO* (the zero redundancy optimizer) removes those copies in stages.
Each rank updates only 1/dp of the weights, so it needs only that share
of the optimizer state (ZeRO-1). If the gradients are reduce-scattered
rather than all-reduced, it needs only its share of them (ZeRO-2).
ZeRO-3 shards the weights too, gathering each layer's just before use.

```
Table 13.3  Model state per parameter, bytes: mixed-precision Adam, sharded by ZeRO over dp ranks
  component                      bytes  sharded from
  bf16 weights                       2        ZeRO-3
  bf16 gradients                     2        ZeRO-2
  fp32 master weights                4        ZeRO-1
  fp32 Adam first moment             4        ZeRO-1
  fp32 Adam second moment            4        ZeRO-1
  total, replicated                 16
  per GPU         ZeRO-0  ZeRO-1  ZeRO-2  ZeRO-3
  dp=8             16.00    5.50    3.75    2.00
  dp=64            16.00    4.19    2.22    0.25
  optimizer step traffic: read and write master and both moments (24) + read gradients (2) + write weights (2) = 28 bytes; the code charges 26
  with fp32 gradients: 18 bytes of state, 30 of traffic; bf16 gradients and no master copy: 12 bytes of state
```

In the code, `train_state_bytes_per_param(dp, zero_stage)` divides
each term by dp from its stage on.

The traffic line is a different number: the bytes the optimizer step
*moves*. It reads the master weights, both moments and the gradient,
writes the first three back and writes the new bf16 weights: 28 bytes
per parameter. The code charges 26 (`ADAM_BYTES_PER_PARAM`), though its
comment lists the same terms, so the optimizer times in this chapter are
7% low. The last line is a common variant: Megatron-LM's bf16 recipe
keeps the gradients in fp32, 18 bytes of state and 30 of traffic. The
difference from 16 is the gradients' precision, not the master copy;
dropping the master copy would give 12.

The *activations* are what the backward needs from the forward: each
GEMM's input, attention's inputs and output, each norm's input. They
grow with the tokens in flight. `activation_bytes` counts them with one
constant, `ACT_H_MULT_FULL = 24.0`: tokens × hidden × layers × 24 bytes,
times `24/34/tp + 10/34` under tensor parallelism, or `1/tp` with
sequence parallelism (below). *Recomputation* (activation
checkpointing) keeps only each layer's 16-bit input and runs the layer's
forward again during the backward: 12 times less memory for one more
forward per micro-batch. In Megatron-LM's published count, whose split
the code takes, tensor parallelism shards the 24 of every 34 bytes that
live between each block's column- and row-parallel GEMMs.
The other 10, the norms' inputs and outputs (the outputs are the first
GEMMs' inputs) and two dropout masks, sit whole on every rank.
*Sequence parallelism* splits those along the sequence across the
tensor-parallel ranks, so everything shards.

![Stacked bars of weights, gradients, optimizer state and activations
per GPU for ZeRO stages 0 to 3; only ZeRO-0 crosses the 80 GB
line.](/assets/tinyperf-book/ch13-memory.svg)

*Figure 13.2. Memory per GPU for Llama-2-7B, data-parallel on eight
GPUs, one 4,096-token sequence each, nothing recomputed. The optimizer
state is two thirds of ZeRO-0's bar.*

```
Table 13.4  Memory per GPU, GB: model states plus activations (train_memory_gb)
  model, layout                ZeRO tokens recompute  states activations   total fits 80 GB
  llama2-7b, dp=8                 0   4096        no   107.8        12.9   120.7         no
                                  1   4096        no    37.1        12.9    49.9        yes
  gpt3-175b, dp=64                0   2048       yes  2802.9         4.8  2807.7         no
                                  1   2048       yes   733.6         4.8   738.4         no
                                  2   2048       yes   388.7         4.8   393.5         no
                                  3   2048       yes    43.8         4.8    48.6        yes
  gpt3-175b, tp=8 pp=8, 1F1B      0   2048        no    43.8        22.2    66.0        yes
    with sequence parallelism     0   2048        no    43.8         7.2    51.0        yes
    with recompute                0   2048       yes    43.8         1.8    45.6        yes
  not counted: fp32 logits for every position, e.g. qwen3-8b at 4096 tokens: 2.5 GB
  this chapter's estimate: without sequence parallelism the recompute checkpoint is whole on every tp rank, so the tp=8 recompute row holds 4.8 GB of activations, not 1.8
  train_memory_gb against the component sums, every row: under 1e-9 GB apart
```

Llama-2-7B on eight 80 GB H100s doesn't fit without ZeRO: 121 GB per
GPU, 81 of them optimizer state. GPT-3 175B is the classic case: 2.8 TB
of state if replicated, and on 64 data-parallel GPUs only ZeRO-3 fits.
Tensor and pipeline parallelism shard the same state from the other
side: at tp=8 and pp=8, 43.8 GB per GPU with no ZeRO. Its first stage
keeps eight micro-batches' activations (the pipeline section explains
why), 22.2 GB, which sequence parallelism cuts to 7.2 and recomputation
to the code's 1.8 (4.8 by the table's estimate; "Where it breaks").

## Data parallelism: when the gradient sync is exposed

A data-parallel copy may itself be a tp × pp shard of the model.
Before the optimizer runs, the copies all-reduce their gradients, 2
bytes per parameter (chapter 11). Frameworks overlap this with the
backward, reducing each bucket of gradients as soon as it is ready.
ZeRO-1 and -2 reduce-scatter the gradients and all-gather the updated
weights instead: the same bytes, since a ring all-reduce is those two
in turn. ZeRO-3 also gathers each layer's weights before its forward
and again before its backward, three passes where the all-reduce makes
two. `dp_grad_sync_us` prices the all-reduce with
chapter 11's ring model and multiplies by 1.5 for stage 3.

When does the backward hide it? Within one node, the ring moves
`2(n−1)/n · 2P` bytes over each GPU's link, with n = dp and P the
parameters on each GPU. The backward does `4·P·T` FLOPs for T tokens at
some fraction e of the tensor cores' peak rate R:

```
backward = 4·P·T / (e·R)        sync = 2(n−1)/n · 2P / link
hidden when  T ≥ (n−1)/n · e·R / link
```

P cancels, and so do the backward's 4 FLOPs per parameter per token
against the ring's 2 · 2 bytes per parameter, which leaves a FLOP rate
over a byte rate, counted in tokens. Whether the sync is exposed
depends on the tokens each GPU's backward covers, not on the model's
size.

```
Table 13.5  Where the gradient all-reduce hides: llama2-7b, bf16 gradients, one micro-batch per step, projected at datasheet rates
  GPU     dp  link GB/s  sync ms  hides from  backward there  (n-1)/n x peak/link
  A100     8        300     78.6         710     78% of peak                  910
  H100     8        450     52.4        1409     74% of peak                 1924
  B200     8        900     26.2        1153     53% of peak                 2157
  H100    64    50 (IB)    111.5        3073     76% of peak                    -
  hides from: the fewest tokens per micro-batch whose backward (D + W) takes as long as the sync
  ZeRO-3 on the H100s, dp=8: sync 78.6 ms (x1.5), hides from 2147 tokens
```

On eight H100s the backward covers the sync from 1,409 tokens per
micro-batch. At peak (e = 1) the formula gives the worst case: 910
tokens on A100s, 1,924 on H100s, 2,157 on B200s, as each
generation's FLOP rate has outgrown its NVLink. The model's counts are
lower because its backward runs below peak: 74% on the H100 there, and
53% on the B200, where chapter 3's GEMM model is far from peak at that
size. Across eight nodes, where the hierarchical all-reduce (chapter
11) crosses 50 GB/s of InfiniBand per GPU, the sync doubles and hiding
it takes 3,073 tokens.

Which backward? A gradient is final only after the step's last
micro-batch has added to it, so a framework can overlap the sync with
that micro-batch's backward alone. (The step function, below, is more
generous: it charges `max(0, sync − backward)` with the backward of
every micro-batch. With one micro-batch per step, as in Table 13.5, the
two agree.) That is the tension of the next section: pipelines want many
small micro-batches, and the gradient sync punishes small ones.

## Pipeline parallelism: the bubble

Chapter 12 cut a model into pp *stages* of consecutive layers. In
training, each stage runs a forward chunk F (one micro-batch through its
layers) and a backward chunk B = D + W per micro-batch. GPipe, the
simplest schedule, runs all m forwards, then all m backwards. Stage s
(stages counted from 0) waits s forwards for its first input, pp − 1 − s
forwards and backwards between its last forward and its first backward,
and s backwards at the end. Every stage idles for the same total:

```
bubble = (pp − 1) · (F + B)            F, B: one micro-batch through one stage
share of the step = (pp − 1) / (m + pp − 1)
```

More micro-batches cure it slowly: at pp = 8, 32 micro-batches still
leave 18% of the step idle.

![Timelines of four pipeline stages under GPipe, 1F1B and zero-bubble
H1, eight micro-batches each; the first two take 33 units, the third
27.](/assets/tinyperf-book/ch13-schedules.svg)

*Figure 13.3. Three schedules with F, D and W one unit each, drawn by
the event simulation of "What we can check". GPipe and 1F1B idle 9 units
on every stage, but GPipe's first stage holds all eight micro-batches'
activations, 1F1B's four. Zero-bubble H1 slots the weight-gradient
chunks (green) into holes and finishes in 27.*

*1F1B* starts each backward as soon as it can: after pp − 1 warm-up
forwards, a stage alternates one forward and one backward. The bubble is
GPipe's, but the first stage holds pp micro-batches, not m. A stage has
1/pp of the layers, so that is one whole micro-batch's activations,
whatever pp is. The other schedules attack the bubble itself:

- **Interleaved 1F1B** gives each GPU v non-adjacent blocks of layers.
  The bubble divides by v; the crossings between GPUs multiply by v.
- **Zero-bubble** schedules split the backward. The stage before waits
  for D; nothing waits for W but the optimizer. Deferring W into the
  holes gives **ZB-H1**: 1F1B's peak memory (pp micro-batches, now on
  every stage) and a bubble of (pp − 1)(F + D − W). **ZB-H2** holds up to
  2pp − 1 micro-batches for a bubble of (pp − 1)(F + D − 2W), zero when
  F + D ≤ 2W. The code prices it at zero.
- **DualPipe** runs two pipelines in opposite directions. Each GPU holds
  stages i and pp − 1 − i, so two copies of the weights, and up to
  pp + 1 micro-batches. Its published bubble is
  (pp/2 − 1)(F&B + B − 3W), where F&B is a forward and a backward chunk
  run together, overlapped.

Here is the code, docstring trimmed:

```python
def pipeline_bubble_us(F: float, D: float, W: float, pp: int, schedule: str = "1f1b",
                       virtual_stages: int = 1) -> float:
    assert schedule in SCHEDULES, f"schedule must be one of {SCHEDULES}"
    B = D + W
    if schedule in ("gpipe", "1f1b"):
        return (pp - 1) * (F + B)
    if schedule == "interleaved":
        assert virtual_stages >= 1
        return (pp - 1) * (F + B) / virtual_stages
    if schedule == "zb_h1":
        return (pp - 1) * max(0.0, F + B - 2 * W)
    if schedule == "zb_h2":
        return 0.0
    assert pp % 2 == 0, "dualpipe needs an even number of stages"
    return (pp // 2 - 1) * max(0.0, (F + B) + B - 3 * W)
```

Read two things carefully. ZB-H1's `F + B − 2W` is `F + D − W`, and
DualPipe's line prices F&B as F + B, with no overlap, so its bracket is
`(F + B) + B − 3W = F + 2D − W`. And **F, D and W here are per-stage
times**, one micro-batch through one stage's layers. `split_fwd_bwd`
returns whole-model times, and the step function divides them by pp
first (signature condensed, type hints and docstring trimmed):

```python
def train_step_us(fwd_us, dgrad_us, wgrad_us, pp, microbatches, schedule="1f1b",
                  virtual_stages=1, optimizer_us=0.0, sync_us=0.0, overlap_sync=True):
    F, D, W = fwd_us / pp, dgrad_us / pp, wgrad_us / pp
    compute = microbatches * (F + D + W) + pipeline_bubble_us(F, D, W, pp, schedule, virtual_stages)
    backward = microbatches * (D + W)
    exposed_sync = max(0.0, sync_us - backward) if overlap_sync else sync_us
    return compute + exposed_sync + optimizer_us
```

Each schedule pays for its bubble in memory. `pipeline_memory` returns
the micro-batches a stage holds at its peak and the copies of the
weights, as listed above (each count capped at m); `train_memory_gb`
multiplies the first by one micro-batch's activations for a stage's
layers and charges each extra copy 2 bytes per parameter.

Table 13.6 compares the schedules on GPT-3 175B on 64 H100s: tensor
parallelism of 8 in each node, 8 stages across nodes, 2,048-token
micro-batches. MFU counts the whole step, optimizer included.

```
Table 13.6  Pipeline schedules: gpt3-175b on 64 H100s, tp=8 x pp=8, 2048-token micro-batches, projected at datasheet rates
  whole model per micro-batch, one tp rank: F 164.9  D 169.1  W 101.1 ms; per stage: F 20.6  D 21.1  W 12.6 ms; optimizer 21.2 ms
  schedule         bubble ms     m=8 ms  MFU    m=32 ms  MFU    m=64 ms  MFU resident copies GB/GPU
  gpipe                380.7        837  33%       2142  51%       3883  57%       32      1  132.5
  1f1b                 380.7        837  33%       2142  51%       3883  57%        8      1   66.0
  interleaved v=2      190.4        647  42%       1952  56%       3693  60%        8      1   66.0
  zb_h1                203.8        660  42%       1966  56%       3706  59%        8      1   66.0
  zb_h2                  0.0        456  60%       1762  62%       3502  63%       15      1   85.4
  dualpipe             150.8        607  45%       1913  57%       3653  60%        9      2   74.2
  memory columns at m=32; 1F1B with recompute, charged as one more F inside D: m=32 2946 ms (+38%), 45.6 GB/GPU
  recompute saves 20.3 GB in the model; 17.3 GB with the checkpoint whole on every tp rank (this chapter's estimate)
  zb_h2 by the zero-bubble paper's (pp - 1)(F + D - 2W), per stage: bubble 115.4 ms; m=8 572 ms, MFU 48%
  logits all-gather and its mirror, which a vocabulary-parallel loss doesn't run: 0.8 ms per micro-batch, 0.2% of F + D + W
```

- **The schedule matters most when micro-batches are few.** Setting
  ZB-H2 aside, MFU runs from 33% to 45% at 8, and from 57% to 60% at 64.
- **GPipe doesn't fit:** it holds all 32 micro-batches, 132.5 GB per
  GPU, where 1F1B holds 8 in 66.
- **ZB-H1 is free in this model:** 1F1B's memory, and 21% off the
  8-micro-batch step.
- **ZB-H2's zero is a best case.** It needs F + D ≤ 2W, which these
  chunks miss (the paper's form gives 115.4 ms of bubble and 48% MFU at
  8), and it overlaps consecutive steps. It doesn't fit anyway, at
  85.4 GB.
- **DualPipe** sits between, paying a ninth micro-batch and a second
  copy of its weights.
- **Interleaving** halves the bubble for free: the model charges no
  crossings.

The footer prices recomputation, which `train_step_us` doesn't: one
more forward inside each backward chunk costs 38% of the step and saves
20.3 GB in the model, 17.3 by this chapter's estimate. DualPipe's
formula is easy to feed the wrong chunk times:

```
Worked example  DualPipe's bubble in per-stage chunk times: gpt3-175b, pp=8
  per stage: F 20.61, D 21.14, W 12.64 ms; B = D + W = 33.78 ms
  code: (pp/2 - 1)((F + B) + B - 3W) = 3 x (54.39 + 33.78 - 37.91) = 3 x (F + 2D - W) = 150.8 ms
  the code returns 150.8 ms; with whole-model chunks it would return 1206 ms
```

## What we can check

There is no "How close is it?" here: `data/validation` holds serving
and kernel measurements only, so there are no ratios. What can be
checked is the arithmetic. The graph's FLOPs match `6·(P − embedding) +
attention` exactly (Table 13.1). Every row of Table 13.4 equals the sum
of its components within 1e-9 GB, which shows the code implements its
accounting, not that the accounting is complete.

**Bubbles.** The chapter's script holds a small event simulation. Each
stage runs its chunks in a fixed order, each as soon as its input is
ready: a forward needs the stage before's forward of the same
micro-batch, a backward chunk the stage after's, and W its own stage's
D. The orders are GPipe's, 1F1B's, and ZB-H1's: 1F1B's with every
backward split, stage s (counted from 0, as in Figure 13.3) running W of
micro-batch j after D of micro-batch j + s.

```
Table 13.7  Bubbles: the code's closed forms against an event simulation of each schedule
  chunks F/D/W per stage   pp   m  schedule simulated closed form bubble share held: simulated code
  1/1/1                     4   8  gpipe         9.00        9.00        27.3%               8    8
  1/1/1                     4   8  1f1b          9.00        9.00        27.3%               4    4
  1/1/1                     4   8  zb_h1         3.00        3.00        11.1%               4    4
  20.6/21.1/12.6 ms         8  32  1f1b      380.7 ms    380.7 ms        17.9%               8    8
  20.6/21.1/12.6 ms         8  32  zb_h1     203.8 ms    203.8 ms        10.5%               8    8
  1/1/1                     4   2  zb_h1         5.00        3.00        45.5%               2    2
  1/1/2                     4   8  zb_h1         3.00        0.00         8.6%               4    4
  GPipe and 1F1B, 150 cases (pp 2-8, m 1-32, five chunk sets): simulated = (pp - 1)(F + D + W) within 1e-9 of a chunk
```

GPipe and 1F1B match their formula in all 150 cases, including fewer
micro-batches than stages. ZB-H1 matches with equal chunks and GPT-3's,
and the micro-batches held match `pipeline_memory` in every row. The
last two rows show where ZB-H1's formula stops holding. With fewer
micro-batches than stages the warm-up never fills: 5 units idle, not 3.
With W above D the formula's bracket falls to zero, but the last stage
can't start before the first micro-batch has crossed pp − 1 stages, (pp
− 1)·F, and it has m·(F + D + W) of work after that, so no order inside
one step idles less than 3 units. The model's chunks have D above W.

The same bound applies to ZB-H2, whose zero needs one step's last
weight-gradient chunks to overlap the next step's first forwards. That
means bypassing the optimizer's pipeline-wide checks (gradient-norm
clipping, inf and NaN checks), which the zero-bubble paper replaces with
a check after the update, "post-validation". ZB-H2, interleaving and
DualPipe are not in the simulation: unchecked.

## Where it breaks

- **No measurements.** 82% MFU is the projected tier's, not a
  framework's.
- **Activations are a round constant.** The code's 24 bytes per hidden
  unit isn't derived from the layer. The Megatron-LM count whose 24:10
  split it borrows (Korthikanti et al., 2022) totals 34 for a GPT layer,
  dropout masks included and the attention scores, which FlashAttention
  never stores, left out. The model is likely up to about 30% low
  without recomputation, and leaves out the fp32 logits and framework
  buffers.
- **The recompute checkpoint is sharded by tp.** The code applies the
  24:10 split to it too, but without sequence parallelism a layer's
  input sits whole on every tensor-parallel rank. By this chapter's
  estimate, Table 13.4's tp=8 recompute row holds 4.8 GB of activations,
  not the code's 1.8.
- **Undercounts:** attention's backward time and the optimizer's
  traffic (first two sections).
- **The sync window is too wide** (data-parallel section).
- **Equal stages, free crossings.** The step function splits the model
  evenly, though the embedding and LM head sit on the end stages
  (Qwen3-8B's head is 8% of its FLOPs), and charges nothing for passing
  activations between stages, or for interleaving's v times more.
- **Schedules as closed forms:** ZB-H1's and ZB-H2's limits are in
  "What we can check"; DualPipe's F&B is priced as F + B, so its
  purpose for MoE models, hiding communication under computation, earns
  nothing.
- **ZeRO-3's gathers** are only 1.5 times the sync: the memory the
  gathered layers occupy, and prefetching, are not modeled.
- **An inference builder.** At tp > 1 the graph all-gathers the
  logits, as vLLM does; a vocabulary-parallel training loss doesn't.
  On GPT-3 that is 0.2% of a micro-batch.

## What you built

- Training as a graph pass, and a micro-batch split into F, D and W by
  the names the pass writes.
- Memory accounting: 16 bytes of state per parameter (18 with fp32
  gradients), sharded by ZeRO and by tp × pp, plus activations with
  recomputation and sequence parallelism.
- The data-parallel sync condition, `T ≥ (n−1)/n · e·R / link`,
  independent of model size.
- Pipeline schedules as bubble formulas over per-stage F, D and W, each
  with its price in memory.
- Evidence: arithmetic only; no measurement.

## Exercises

1. Count the tensors a Llama layer saves for its backward, with
   FlashAttention and no dropout, in bytes per hidden unit per token.
   Compare with 24 and redraw Figure 13.2. Does ZeRO-1 still fit?
2. Make `train_step_us` hide the sync under the last micro-batch's
   backward only, and add dp=2 across nodes to Table 13.6. At which
   micro-batch count does the sync show?
3. Extend the simulation to interleaved 1F1B and check
   `(pp − 1)(F + B)/v`. Then charge each crossing chapter 12's hand-off
   cost. At what v does interleaving stop paying?
4. Put Qwen3-8B's embedding and LM head on the end stages of a
   four-stage pipeline and simulate 1F1B with each stage's own chunks.
   How far off is the equal-stage formula?
5. Time one training micro-batch of a small model on your GPU and
   compare it with Table 13.2's method: this chapter's first held-out
   evidence.

---

*[← Chapter 12: Parallel layouts]({% post_url 2026-09-29-tinyperf-12-parallel-layouts %}) · [Contents](/series/tinyperf/) · [Chapter 14: A serving simulator →]({% post_url 2026-09-29-tinyperf-14-a-serving-simulator %})*
