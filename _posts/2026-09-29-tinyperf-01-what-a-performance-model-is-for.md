---
layout: post
title: "Building tinyperf, chapter 1: What a performance model is for"
date: 2026-09-29 12:01:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/01-what-a-performance-model-is-for/
excerpt: "Before you buy a GPU, rent a thousand of them or promise a customer a response time, you want to know how fast your model will run there. Often you can't measure it: the hardware is on order, or not built yet, or there are ten thousand configurations to try. This book builds a program that answers such questions without running the model: an analytical performance model. What is such a model for, and how will we build one and check it?"
redirect_from:
  - /tinyperf/perf-modeling/2026/08/18/building-tinyperf-m0.html
---

*[Building tinyperf](/series/tinyperf/) · Part I: Pricing one kernel · Code: [`tinyperf/serving.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/serving.py), `StepLatencyModel` · Every table in this chapter comes from `python3 book/scripts/ch01_intro.py`.*

Before you buy a GPU, rent a thousand of them or promise a customer a
response time, you want to know how fast your model will run there.
Often you can't measure it: the hardware is on order, or not built yet,
or there are ten thousand configurations to try. This book builds a
program that answers such questions without running the model: an
*analytical performance model*. What is such a model for, and how will
we build one and check it?

The short answer: it replaces running the workload with arithmetic,
fast enough to answer questions no measurement campaign can afford, and
it is worth exactly what its checks against measurements say.

By the end of this chapter you will know:

- the questions an analytical model answers, and why they need answers
  in milliseconds;
- what it gives and costs next to measuring, or simulating cycle by
  cycle;
- how tinyperf, the model this book builds, is put together, and which
  chapter builds each piece;
- how tinyperf is designed, and why: what each design choice buys and
  what it costs;
- how to price a real model (predict its times) in a few lines, and how
  close it comes to a real server;
- the rules this book uses to say how far to trust a number.

## Four questions

**How fast will this run on a GPU I don't have?** A new generation, an
instance you haven't rented, a chip still being designed: the model
turns its datasheet into time.

**Which configuration?** Which GPU, how many, how the model is split
across them, the batch size, the precision, the settings of the serving
engine (the software that batches requests and runs the model). Five
options on each of these six axes make 15,625 configurations.

**What if?** What if the GPU had twice the memory bandwidth? An
architect asks this about a chip that doesn't exist; no measurement can
answer it.

**How much load can a server take?** How many requests per second
before its latency target breaks? That depends on queues, so it takes a
simulation of the serving engine in which every step, one forward pass
of the model over a batch, has a price.

Why milliseconds? Measuring takes the hardware and time on it: each
point of Table 1.4, below, is 240 requests, four minutes of a GPU at one
request per second. The second question needs thousands of answers. The
fourth needs thousands of priced steps per answer (that same point
simulates 7,777 engine steps), swept over rates and configurations.
Only a price in milliseconds makes that practical: tinyperf computes
the sixteen numbers of Table 1.1, starting from the model's
configuration, in tens of milliseconds on one CPU core.

## Three ways to get a number

- **Measure.** Run the workload and time it: the ground truth for that
  hardware, software and configuration. It needs the hardware and GPU
  time for every configuration, says what happened but not why, and can
  be wrong, as we will see.
- **Simulate cycle by cycle.** A cycle-level simulator models the GPU's
  pipelines, caches and memory controllers instruction by instruction.
  It can study hardware that doesn't exist, in detail, but needs a
  detailed description of the chip and every kernel, and runs far
  slower than the hardware it models: a tool for designing a memory
  system, not for pricing ten thousand serving configurations.
- **Model analytically.** Count the work and divide by rates. The answer
  takes milliseconds for any GPU you can describe, and says which
  resource limited each operation. But the model knows only the effects
  someone put into it; an effect it lacks shows up as error.

They work together: measurements check the model, and the model picks
the few configurations worth measuring.

## Rates and counts

The simplest analytical model is the *roofline* (chapter 2). An
operation is limited either by math or by traffic to the GPU's main
memory (DRAM), whichever takes longer:

```
time = max(flops / peak_flops, bytes / memory_bandwidth)
```

On the back of an envelope it already says a lot. Take Qwen3-8B, a
dense 8-billion-parameter model, on an NVIDIA RTX A6000. Producing one
token for each sequence in a batch, a *decode step*, reads every weight
once, except the input embedding table: its lookup reads only the rows
of the batch's tokens.

Throughout the book a ratio is **predicted ÷ measured**, and we say a
prediction *reads* its ratio: above 1 it reads high, below 1 it reads
low.

```
Worked example  A decode step on the back of an envelope: Qwen3-8B on an RTX A6000
  8.19 billion parameters at 2 bytes; a decode step reads all but the input embedding table: 15.1 GB
  15.1 GB / 768 GB/s = 19.7 ms per step; measured at batch 1: 23.9-25.8 ms, so the estimate reads 0.77-0.83
```

One datasheet number gets within a quarter of a real serving engine. The
rest of the book is about the other quarter, and cases where the
estimate is further off. Real kernels divide their work into tiles and
waves (the blocks a matrix multiply, or GEMM, is cut into, and the
rounds they run in; chapter 3), reuse data through caches, pay a cost to
launch, and run below the datasheet's peaks. Kernel by kernel, the
roofline can read under half the measured time: 0.45 for one GEMM in
chapter 3, even with this GPU's best measured rates. A model that prices
the wrong mechanism can be off by more still (Table 1.5).

The craft is adding the effects that matter one at a time, each a
mechanism with a formula (tile quantization, which pays for whole tiles
at a matrix's ragged edge; cache reuse; a fused attention kernel, one
kernel for all of attention that keeps the scores on chip; the experts
a step touches in a mixture-of-experts (MoE) model, where each token
uses a few of many feed-forward blocks), and checking each against a
measurement before adding the next. Constants stay few and physical.
The core of the *calibration*, a few constants fitted to kernels timed
on the real GPU (chapter 4), is four scalars, such as the fraction of
the peak of the tensor cores (the GPU's matrix-multiply units; see
chapter 2) that the best GEMMs sustain. Where a kernel's
behavior can't be derived, the model carries a measured table and says
so.

## The map

tinyperf is that model: a Python package that needs nothing outside the
standard library. Figure 1.1 shows how it is put together.

![The seven layers of tinyperf from the device up to sweeps, each with
what it takes in and gives out, its module files and the chapters that
build it, beside a calibration column and a validation
column.](/assets/tinyperf-book/ch01-map.svg)

*Figure 1.1. tinyperf, layer by layer, with what each layer takes in and
gives out. The five blue layers price one step; the two orange layers
use many step prices. Calibration attaches fitted constants to the lower
four layers. Validation checks every layer against measurements.*

Read it from the bottom: each layer uses the ones below it.

- **Device** (chapter 2). A GPU as about a dozen rates and sizes: peak
  math for each number format, memory bandwidth, the size and bandwidth
  of the on-chip L2 cache, a kernel's launch cost, the links between
  GPUs and the memory capacity. Most come from the datasheet, one JSON
  file per GPU in `data/devices/`. A what-if, such as twice the
  bandwidth, is a change to one number.
- **Kernel cost models** (chapters 3, 7 and 11). Each turns one
  operation's shapes into a time on a device: a GEMM by how a library
  cuts it into tiles and waves, fused attention by its math and the
  cache it reads, a collective (GPUs exchanging data, such as summing
  their partial results) by a link's bandwidth and per-hop latency.
- **Graph of operators** (chapters 5, 7, 10 and 13). One step written
  down as an ordered list of operations, each with its shapes, FLOPs
  and bytes. *Passes* rewrite the list the way a runtime would: fuse
  attention into one kernel, switch layers to 8- or 4-bit formats, add
  the gradient computation for training.
- **Scheduler and report** (chapter 5). `execute`, tinyperf's scheduler
  (not the serving engine's), prices every operation with its kernel
  model and adds them up. The report lists each operation's time and
  what limited it, math, DRAM, the L2 or a link, so every price comes
  with its reason.
- **Model builders** (chapters 6–9 and 11–13). They turn a model's
  configuration (layers, heads, experts) and a step's shape (the batch,
  the prompt or context length, and how the model is split across
  GPUs) into that step's graph, and say what fits in memory: the
  weights, the KV cache and, for training, the optimizer's state.
- **Serving simulator** (chapters 14–20). It plays a server forward in
  time: requests arrive, the engine's scheduler forms each step's
  batch, the layers below price the step, and the clock advances. Out
  come the latencies a user feels: the time to a request's first token
  and between its later ones (TTFT and TPOT; below). It also covers
  speculative decoding (a small draft model proposes tokens that the
  served model checks in one step) and separate prefill and decode
  servers.
- **Sweeps and planning** (appendix A). They run the step price or the
  simulator over thousands of configurations, keep those that no other
  beats on two metrics at once (the *Pareto front*), and turn the
  winners into GPUs, dollars and joules per million tokens.

Beside the stack stand two columns. **Calibration** (chapter 4)
attaches constants fitted to kernels timed on a real GPU to the lower
four layers; without it the model prices from the datasheet alone.
**Validation** (chapter 21) checks every layer against measurements:
`data/validation/` holds them, `tools/` the scripts that took them, and
`tests/` fails if a price leaves its validated range.

## How tinyperf is designed, and why

The map says what the layers are. This section says why they are drawn
that way. Each choice buys something and costs something, and the costs
come back in every chapter's "Where it breaks".

### An analytical model, built from mechanisms

tinyperf prices work by counting it and dividing by rates. It doesn't
simulate cycles, and it doesn't fit a curve to measured times. Each
behaviour of a real kernel that changes its time enters as a
*mechanism*, an effect with a formula: work padded to whole tiles, data
reused through the L2, a fixed cost per launch.

Three things follow. A price takes milliseconds, so sweeps of thousands
of configurations are practical. Any GPU with a datasheet can be priced,
including one that doesn't exist yet. And every price says which
resource set it, which a stopwatch can't. A fitted curve would do none
of this: it is only good near the points it was fitted to, and says
nothing about why.

The rule that keeps mechanisms honest: add one at a time, and check it
against a measurement before adding the next. Some behaviour can't be
derived, such as which tile cuBLAS, NVIDIA's matrix-multiply library,
picks for each row count, how NCCL, its library for communication
between GPUs, performs on one particular link, or which experts an MoE
model's router (the layer that sends each token to its experts)
chooses. There the model carries a measured table, says so, and prices
around it (chapters 3, 9, 11 and 15).

The cost: the model knows only the mechanisms someone put in. A missing
one shows up as error, which is why the book sets its prices beside
measurements wherever it can, and says when a number is unmeasured. And
a measured table holds only for the GPU and kernels it was measured on.

### A GPU is data, the model is code

A device is a frozen record of about a dozen rates and sizes, read from
a JSON file in `data/devices/`. Math is stored as factors, not peaks:
the SM count (streaming multiprocessors, the GPU's processors;
chapter 2), each SM's multiply-accumulates per clock for each number
format, and the clock. The peaks are derived. No GPU's rates are written
in the code; the few constants measured on one card, such as the
sampler's fixed cost, say so where they are defined.

So a new GPU is a new file, not new code. A what-if, such as twice the
bandwidth or 8 SMs disabled, is an override that returns a new device.
The record is frozen, so the original can't change by accident. Derived
numbers are methods, so none of them goes stale when a field changes.
And a field earns its place only if some price uses it (chapter 2).

The cost: what isn't a rate isn't modelled. How the SMs are wired, how
warps (groups of 32 threads that run in lockstep) are scheduled, and
how the caches replace lines enter only through mechanisms or measured
constants.

### One step, written down as a graph

A model builder turns a model's configuration and a step's shape into a
*graph*: an ordered list of operators, each with its tensor shapes, its
FLOPs, its bytes and a *family*, the kind of cost model that prices it.
A layer that repeats is stored once, with a count.

The graph separates what a step does from what it costs. The same graph
can be priced at any tier (how much of reality a price includes; below)
and on any device. *Passes* rewrite it the way a runtime would, each a
rewrite rather than a second model: they fuse attention into one kernel
(chapter 7), run GEMMs in 8- or 4-bit formats or mark weights as sparse
or stored in 4 bits (chapter 10), or add the gradient computation for
training (chapter 13). The graph's FLOPs and bytes can be checked
against counts derived by hand, so a missing operator shows (chapter 5).
And storing a repeated layer once keeps a step's graph small: 16
records stand for a decode step's 419 kernel calls (chapter 5).

The cost: a graph holds only work on tensors. The host's work, the
engine's scheduling and the time a request spends outside the GPU live
outside it, in the serving simulator.

### Price each operator, then add

`execute` sends each operator to its family's cost model. There are
five families: GEMMs, memory-bound operations, collectives, fused
attention and linear attention. Inside a GEMM, math, DRAM and L2
traffic overlap, so its time is the largest of the three, plus its
launch; a memory-bound operation costs its bytes over the DRAM rate,
plus its launch (chapter 5). Across operators, the times add.

Adding is right for a step on one GPU, because its kernels run one
after another, especially when an engine replays them as a CUDA graph
(the step's kernel launches recorded once and replayed together;
unrelated to tinyperf's graph). The report keeps one row per operator,
with the resource that bound it, so every price comes with its reason.
Where overlap between operators is real, the model doesn't simulate
concurrent streams. For communication hidden under computation it takes
the fraction hidden as an input, `comm_overlap`, between the serial sum
and the fully hidden case (chapter 11); offloaded weights streamed
during the forward are priced at one end or the other, fully hidden by
default (chapter 17).

The cost: kernels that run concurrently, and overlap that is only
partly hidden, are approximations.

### Three tiers, and calibration as data

The same mechanisms run at three tiers (chapter 4). The speed-of-light
tier keeps only the roofline. The projected tier adds every mechanism,
at datasheet rates. The calibrated tier also applies a record of
constants fitted to kernels timed on the real GPU. That record is data
too, a file in `data/calibration/`. It holds four core constants (the
fraction of peak math and of DRAM bandwidth that real kernels reach, the
L2 rate and the launch cost) and, for a GPU studied closely, further
fields for particular kernels and engine costs, with a note on how each
was measured (Table 4.9).

The tier tells you what kind of number you have: a projection for any
GPU, or a calibrated price where measurements exist. The gap between
the tiers measures what the datasheet misses. Keeping calibration as
data means the code is the same for every GPU, and adding one means
measuring, not programming. One rule protects the evidence: each field
is measured on its own kernel, never fitted to the end-to-end times it
will later be checked against, so comparing those stays a test. The few
fields that break the rule are marked, and are the first to distrust
(chapter 4).

The cost: a calibration describes one GPU under one software stack.
Launch cost alone differs several times between eager PyTorch, which
launches each kernel as Python reaches it, and a CUDA graph (chapter 4),
and a new library version can move the other constants. Using a
calibration with another stack is an unmeasured extrapolation.

### Serving: one engine's scheduler, played step by step

The serving simulator is a discrete-event simulation: its clock jumps
from one event to the next, and its events are the engine's step
boundaries and arrivals at an idle engine. At each step it forms the
batch the way one engine's scheduler does, that of vLLM 0.15.1, a
widely used open-source serving engine: its admission of waiting
requests, chunked prefill (a long prompt fed in pieces beside the
running decodes), paged KV cache (the cache held in fixed-size blocks)
and planning one step ahead. It then asks the step model for the step's
price and advances its clock. The step model, `StepLatencyModel`,
caches prices by step shape and prices contexts every 256 tokens,
interpolating between them.

Latency under load comes from the queue, and the queue depends on how
the engine forms its batches, so the simulator copies one engine's
rules rather than a generic "continuous batching" loop (requests
joining and leaving the batch every step). Simulating steps, not
kernels, reuses the step model unchanged and adds only scheduling. The
cache makes it cheap: one load sweep of 15,059 steps builds only 639 of
tinyperf's graphs (Table 14.1). The engine's settings travel together as
one preset, `serving.VLLM`, so a run gets the engine's scheduler, its
sampler (which picks each next token) and the right tier together.

The cost: another engine, or another version of this one, needs its
own rules. Work outside the steps (the HTTP request, handing it to the
engine, streaming the answer) enters as one constant per request, set
on an idle server (chapter 16).

### Small, plain Python

tinyperf is pure Python using only the standard library: no NumPy, and
nothing to install. It runs wherever Python runs, the book can quote
its functions whole, and a reader can follow any price to the line that
computes it. Plain Python is slow at arithmetic, but a step's graph is
tens of operator records, and the caches remove the repetition across a
sweep.

The cost: no vectorized sweeps. Very large sweeps depend on the caches,
and on pricing a repeated layer once.

### Evidence kept with the code

The measurements live in the repository, in `data/validation/` and
`data/calibration/`. Predictions for new runs are committed before the
runs, every number says what the model had seen, and tests pin
validated results so that a change which breaks one fails loudly
(chapter 21). This book's tables are printed by scripts and checked
against the text on every change.

A performance model is worth only as much as the trust in its error
bars, and that trust has to survive the next change to the code. The
cost is deliberate friction: a new mechanism must keep every pinned
result inside its bounds.

### Left out on purpose

tinyperf has no cycle-level behaviour, no kernel source, and no
autotuning on hardware (the GEMM model searches its own menu of tiles;
chapter 3), and it validates only one engine's scheduler. It prices one
GPU type per deployment, and a clock as one number. Each omission keeps
the model small enough to check against measurements piece by piece;
each chapter's "Where it breaks" says what it costs there.

## How to read this book

Part I (chapters 1–4) prices one kernel; Part II (5–10) one model on one
GPU; Part III (11–13) many GPUs, for inference and training; Part IV
(14–20) builds the serving simulator; Part V (21–22) is about knowing
the model is right; appendix A applies it to sweeps and capacity
planning, and appendix B is a glossary. The book states the current
understanding, and its field notes tell the detours that reached it.

## A first taste

The lines below are the script's `first_taste` with b = 8 prompts of
s = 2,048 tokens written in, and the imports it needs:

```python
from tinyperf import Device, Methodology
from tinyperf.nets import qwen3_8b
from tinyperf.serving import StepLatencyModel

a6000 = Device.load("rtx_a6000")                   # data/devices/rtx_a6000.json
model = StepLatencyModel(qwen3_8b(), a6000, methodology=Methodology.CALIBRATED)

ttft_ms = model.prefill_us(8 * 2048, n_seqs=8) / 1e3   # eight 2,048-token prompts at once
tpot_ms = model.decode_us(8, 2048 + 64) / 1e3          # one decode step of the eight
```

`qwen3_8b()` is the configuration from the released checkpoint: 36
layers, hidden size 4,096, 32 query heads sharing 8 key-value heads.
`Methodology.CALIBRATED` selects the *calibrated tier*: the full
mechanism with the calibration's constants. It is one of three tiers,
ways the model can price, which Table 1.2 sets side by side. The two
numbers are the two an LLM server's user feels:

- **TTFT**, time to first token: from sending a request to its first
  output token, dominated by the *prefill*, one pass over the prompt;
- **TPOT**, time per output token: the mean gap between the later
  tokens. In a batch that decodes together, it is one decode step.

Chapter 14 is their home. Here they are measured through the offline
batch API of vLLM, with no server in front: b prompts go to the engine
together, TTFT is the time until all b have their first token, and TPOT
is the mean of the 128 decode steps that follow. Those steps see
contexts from about s to s + 128 tokens, so the model prices one step at
the mean, s + 64.

Inside, `StepLatencyModel`, the object that prices steps, runs the
whole map for each step shape. Here is the function that does it,
trimmed of two more optional passes and of the builder's arguments for
MoE models, for a prompt prefix already in the engine's cache and for a
model whose layers are split across GPUs (chapters 9, 17 and 12):

```python
    def _graph_report(self, phase, batch, seq, chunk=1, shared=0, **stage):
        g = build_llm_graph(self.p, phase, batch=batch, seq_len=max(seq, 1),
                            tp=self.tp, ep=self.ep, dp=self.dp, cp=self.cp, chunk=chunk, ...)
        fuse_attention(g)
        if self.recipe:
            apply_recipe(g, self.recipe)
        ...                                  # sparsity and weight-only passes, the same way
        return execute(g, self.device, methodology=self.methodology, calibration=self.calibration)
```

The builder emits one step's graph, any parallel layout (how the model
is split across GPUs) baked into its shapes (chapters 6 and 12). Passes
fuse attention into the kernel an engine runs (chapter 7) and change
precision (chapter 10). `execute` prices the graph at the chosen tier
(chapters 4 and 5). A cache keyed by step shape lets the serving
simulator price thousands of steps (chapter 14).

## How close is it?

We compare with vLLM running Qwen3-8B in bf16 on one RTX A6000 over a
grid of batch sizes and prompt lengths
(`data/validation/vllm_qwen3_8b_rtx_a6000.json`). Ratios are
predicted ÷ measured, as above. And every number says what kind it is:

- *held out*: nothing in the model was fitted to it, and no mechanism
  was designed with it in view;
- *fitted*: it supplied a constant;
- *in-sample*: it was used to build a mechanism;
- *recorded*: it was stored when measured, and is not recomputed by the
  current model. This one combines with the others: a number can be
  recorded and held out.

A number with no measurement behind it is a *projection*.

Table 1.1 sums up each column of ratios by its range and its **typical
error**: their geometric mean distance from 1 in either direction,
`exp(mean |ln ratio|) − 1`.

```
Table 1.1  Qwen3-8B served by vLLM on one RTX A6000: the calibrated tier against measurement (in-sample)
  batch  prompt   TTFT ms  model ms  model/meas  TPOT ms  model ms  model/meas
      1     512      79.6      76.9       0.967    23.87     24.11       1.010
      1    2048     291.3     292.0       1.002    24.30     24.46       1.007
      1    8192    1282.1    1345.3       1.049    25.75     25.82       1.003
      8     512     564.5     551.8       0.978    25.12     25.44       1.013
      8    2048    2280.8    2292.3       1.005    27.80     28.17       1.013
      8    8192   10420.5   10719.5       1.029    39.12     39.09       0.999
     32     512    2191.5    2194.0       1.001    29.90     29.30       0.980
     32    2048    9094.2    9146.3       1.006    41.98     40.22       0.958
  TTFT 0.97-1.05, typical error 1.9%; TPOT 0.96-1.01, typical error 1.4%
```

Every TTFT is within 5%, from 80 ms to 10.4 s. TPOT is within 1.3% up
to batch 8, and at batch 32 reads 2.0% and 4.2% low.

Nothing in the model was fitted to these times: its constants come from
kernels timed on their own (GEMMs from cuBLAS; launch costs; the
engine's decode-attention kernel; this model's layer GEMMs by row
count). But a mechanism was designed with this grid in view. It exposed
a flaw in how the model priced a batch of prompts (Table 1.5), and every
change since has had to keep it inside fixed bounds. So Table 1.1 is
in-sample, for one dense model on one GPU and one engine.

How much of the agreement is mechanism and how much fitted constants?
Table 1.2 prices the same grid at the model's three tiers: the *speed of
light*, a roofline per operation at datasheet rates; *projected*, the
full mechanism at datasheet rates; and *calibrated*.

```
Table 1.2  The same eight cells at the model's three tiers: model/measured
  tier              TTFT range  typical error  TPOT range  typical error
  speed of light     0.66-0.75          37.9%   0.79-0.83          22.0%
  projected          0.77-0.82          26.6%   0.84-0.90          13.3%
  calibrated         0.97-1.05           1.9%   0.96-1.01           1.4%
  fitted to measured cuBLAS GEMMs: tensor math at 0.75 of peak, DRAM at 0.90 of its datasheet rate
```

Summed over a whole model, the roofline reads only 17–34% low: most of
the time goes to large kernels deep inside one regime, clearly limited
by math or clearly by memory. Much of the rest is the datasheet itself:
this GPU's best GEMMs sustain 0.75 of the tensor cores' peak, and the
roofline's TTFT reads 0.66–0.75. The mechanism closes part of the gap,
the measured rates most of the rest.

## Asking it questions

Table 1.3 asks the architect's what-if at the projected tier, which uses
only the device file (the GPU's dozen rates, mostly from its datasheet;
chapter 2) and so works for any device you can write down. Its times
read 10–23% low on this GPU (Table 1.2), so compare the rows with each
other, not with a stopwatch. The last column names what limits the
prefill's first feed-forward network (FFN) GEMM: math, or the bandwidth
of the L2, which all DRAM traffic passes through.

```
Table 1.3  What if? Qwen3-8B on an RTX A6000 with one rate doubled, projected tier
  device                   TTFT, 8 x 2048 ms  TPOT, batch 8 ms  prefill FFN GEMM limited by
  as shipped                          1760.8             24.93  math
  2x DRAM bandwidth                   1684.3             13.06  math
  2x tensor math                      1560.0             24.93  L2 bandwidth
  2x tensor math and L2                957.5             24.93  math
  L2 bandwidth: 2000 GB/s in the device file (an estimate), 4000 GB/s fitted to measured GEMMs
```

Twice the memory bandwidth nearly halves the decode step and barely
moves prefill, as the roofline expects: decode streams weights, prefill
is math. Twice the tensor math moves prefill only from 1,761 to
1,560 ms: the feed-forward GEMM becomes limited by the L2 cache's
bandwidth instead. Removing one bottleneck exposes the next; double the
L2 bandwidth too and prefill falls to 958 ms. Yet the device file's L2
bandwidth is only an estimate, half the fitted value: a what-if is only
as good as the rates you left alone.

Table 1.4 asks the capacity question. vLLM's benchmark sent 240
requests, each a 1,024-token prompt asking for 128 tokens, at random
times at a fixed average rate. The simulator replays the same arrivals
through a model of vLLM's scheduler, the part of the engine that decides
which requests join each step, and prices every step at the calibrated
tier (`serving.VLLM`, vLLM's settings as one preset; chapter 14). The *p95 TTFT* is the TTFT that 95% of requests beat.

```
Table 1.4  How much load can one RTX A6000 take? Qwen3-8B under vLLM, 240 requests of 1024 tokens in, 128 out (in-sample)
  req/s  p95 TTFT ms  model ms  model/meas  model, steps 2% faster / slower  engine steps
      1          367       370        1.01                        367 / 384          7777
      2          501       499        1.00                        500 / 519          3202
      3          769       787        1.02                        726 / 842          1543
      4         2693      3126        1.16                      2442 / 3862           731
      5        12250     12896        1.05                    11866 / 14018           647
      6        18671     19263        1.03                    18324 / 20112           602
      8        26358     26748        1.01                    25772 / 27782           557
  measured, requests completed per second: 0.99 at 1, 1.94 at 2, 2.85 at 3, 3.56 at 4, 3.65 at 5, 3.73 at 6, 3.85 at 8
```

The fifth column reprices each run with every step 2% faster, then 2%
slower, to show how much a small step-price error matters. The
last column counts the engine steps simulated; it falls as the rate
rises because more requests share each step.

At a p95 target of one second this GPU takes 3 requests per second, not
4, and the model agrees. Offered 4 per second, the server completes
only 3.56: it can't keep up, and requests queue. This table is
in-sample too: the engine's behaviors were modeled with these runs in
view.

## How far to trust a number

Chapter 21 is about how a performance model earns trust. Here are its
rules.

**Predictions are written down before the measurement,** committed to
the repository and never edited. Table 1.5 shows those made for Table
1.1's grid.

```
Table 1.5  Recorded: the calibrated tier's predictions, written down before the measurement ran
  batch  prompt  TTFT predicted ms  model/meas  TPOT predicted ms  model/meas
      1     512               75.9       0.954              23.90       1.001
      1    2048              292.0       1.002              24.23       0.997
      1    8192             1345.3       1.049              25.54       0.992
      8     512              609.2       1.079              25.47       1.014
      8    2048             3210.0       1.407              28.09       1.010
      8    8192            25403.2       2.438              38.58       0.986
     32     512             3210.0       1.465              30.19       1.010
     32    2048            25403.2       2.793              40.68       0.969
  TPOT: every cell within 3.1% (0.97-1.01); TTFT: one prompt 0.95-1.05, a batch of prompts 1.08-2.79
  data/validation holds 37 files of predictions written before their measurements
```

Both columns are recorded and held out: they were written before the
grid was measured, when nothing in the model had seen it.

Decode came back within 3.1% on every cell. Prefill was right for
single prompts and up to 2.79 times high for batches: the model had
priced a batch of prompts as one long prompt, so every token attended to
the other prompts' tokens too. Note the identical predictions for 8 ×
2,048 and 32 × 512: 16,384 tokens either way. Chapter 6 tells how the
error's shape named the cause, and the fix.

**Every number carries a label**, as defined above.

**Error bars are pinned by tests.** The test suite asserts that every
cell of Table 1.1 stays within 0.92–1.10 on TTFT and 0.93–1.07 on TPOT
(`test_silicon_envelope` in `tests/test_core.py`). A change that breaks
agreement with the hardware fails.

**Measurements get checked too.**

> **Field note: the benchmark that lied.** An early version of
> `tools/measure_vllm.py` sent identical prompts. The engine's prefix
> cache recognized the repeats and skipped most of each prefill, and the
> measured TTFT stayed flat across a 128-fold range of prompt lengths:
> it measured the cache, not the GPU. The tool now gives every request
> its own random prompt.

## Where it breaks

An analytical model prices what it can count. Four kinds of thing it
can't:

- **Kernels competing for the GPU.** `execute` adds up operation times
  one after another. When communication overlaps computation, or two
  processes share a GPU, it has no schedule of who waits for whom. It
  adds communication to computation, or hides a fraction of it that you
  declare (chapter 11).
- **Timing.** In Table 1.4, making every step 2% faster or slower moves
  the p95 TTFT by a few percent at 1 request per second. At 4, where the
  server tips over, it moves it between 2,442 and 3,862 ms, either side
  of the measured 2,693. Near capacity the answer depends on which
  request lands in which step, and small errors grow. Chapter 18 turns
  this into prediction intervals, a range of answers instead of one.
- **A library's choices you haven't measured.** Kernel libraries pick
  kernels by heuristics a model can't derive. Chapter 3 measures a GEMM
  that takes 73% longer at 129 rows than at 128, consistent with cuBLAS
  changing its tiling there (not profiled). The model carries such edges
  only once they are measured.
- **Anything never measured.** Which experts an MoE router picks depends
  on the text. Without a measurement the model assumes uniform routing,
  which reads 28–44% high on one model's decode steps from batch 8 up
  (chapter 9). In speculative decoding, how often the served model
  accepts the draft's tokens, the acceptance rate, is an input you
  supply (chapter 19). A GPU without public rates can't be described at
  all.

## What you built

- A map of tinyperf, and the chapter that builds each layer.
- A first priced model: Qwen3-8B on an RTX A6000 within 0.96–1.05 of
  vLLM (in-sample); predictions written before the measurement held
  decode within 3.1% (recorded, held out).
- A what-if and a capacity question, answered.
- Rules for trusting a number: predictions first, every number labeled,
  error bars pinned by tests, limits stated.

## Exercises

1. Price Table 1.1's grid on an H100 (`Device.load("h100_sxm")`) at the
   projected tier. Which speeds up more, TTFT or TPOT? Why?
2. In Table 1.3, doubling the tensor math made the L2 bandwidth the
   limit. Find the smallest L2 bandwidth at which the prefill FFN GEMM
   is limited by math again.
3. Do one 2,048-token prefill on the back of an envelope: 2 FLOPs per
   weight per token in the 36 layers, plus attention, divided by
   `Device.load("rtx_a6000").peak_tflops(DType.BF16)`. Compare it with
   Table 1.1's measured TTFT.
4. Write a prediction down before you measure. Pick a model and a GPU
   you have, commit the calibrated tier's TTFT and TPOT for a few cells
   to a file, then run `tools/measure_vllm.py`. Label each number you
   report by its kind.

---

*[Contents](/series/tinyperf/) · [Chapter 2: A GPU as a handful of rates →]({% post_url 2026-09-29-tinyperf-02-a-gpu-as-a-handful-of-rates %})*
