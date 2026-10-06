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
- how to price a real model in a few lines, and how close it comes to a
  real server;
- the rules this book uses to say how far to trust a number.

## Four questions

**How fast will this run on a GPU I don't have?** A new generation, an
instance you haven't rented, a chip still being designed: the model
turns its datasheet into time.

**Which configuration?** Which GPU, how many, how the model is split
across them, the batch size, the precision, the serving engine's
settings. Five options on each of these six axes make 15,625
configurations.

**What if?** What if the GPU had twice the memory bandwidth? An
architect asks this about a chip that doesn't exist; no measurement can
answer it.

**How much load can a server take?** How many requests per second
before its latency target breaks? That depends on queues, so it takes a
simulation of the serving engine in which every step has a price.

Why milliseconds? Measuring takes the hardware and time on it: each
point of Table 1.4, below, is 240 requests, four minutes of a GPU at one
request per second. The second question needs thousands of answers. The
fourth needs thousands of priced steps per answer (that same point
simulates 7,777 engine steps), swept over rates and configurations.
Only a price in milliseconds makes that practical: tinyperf computes
the sixteen numbers of Table 1.1, graphs included, in tens of
milliseconds on one CPU core.

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
operation is limited either by math or by memory traffic, whichever
takes longer:

```
time = max(flops / peak_flops, bytes / memory_bandwidth)
```

On the back of an envelope it already says a lot. Take Qwen3-8B, a
dense 8-billion-parameter model, on an NVIDIA RTX A6000. Producing one
token for each sequence in a batch, a *decode step*, reads every weight
once, except the input embedding table: its lookup reads only the rows
of the batch's tokens.

```
Worked example  A decode step on the back of an envelope: Qwen3-8B on an RTX A6000
  8.19 billion parameters at 2 bytes; a decode step reads all but the input embedding table: 15.1 GB
  15.1 GB / 768 GB/s = 19.7 ms per step; measured at batch 1: 23.9-25.8 ms, so the envelope reads 0.77-0.83
```

One datasheet number gets within a quarter of a real serving engine.
The rest of the book is about the other quarter, and cases where the
envelope is further off. Real kernels divide their work into
tiles and waves (the blocks a GEMM is cut into, and the rounds they run
in; chapter 3), reuse data through caches, pay a cost to launch, and run
below the datasheet's peaks. Kernel by kernel, the roofline can read
under half the measured time: 0.45 for one GEMM in chapter 3, even with
this GPU's best measured rates. A model that prices the wrong mechanism
can be off by more still (Table 1.5).

The craft is adding the effects that matter one at a time, each a
mechanism with a formula (tile quantization, which pays for whole tiles
at a matrix's ragged edge; cache reuse; a fused attention kernel; the
experts an MoE step touches), and checking each against a measurement
before adding the next. Constants stay few and physical. The core of the
*calibration*, a few constants fitted to kernels timed on the real GPU
(chapter 4), is four scalars, such as the fraction of the tensor cores'
peak that the best GEMMs sustain. Where a kernel's behavior can't be
derived, the model carries a measured table and says so.

## The map

tinyperf is that model: a Python package that needs nothing outside the
standard library. Figure 1.1 shows how it is put together.

![The seven layers of tinyperf from the device up to sweeps, with the
module files of each layer and the chapters that build them, beside a
calibration column and a validation column.](/assets/tinyperf-book/ch01-map.svg)

*Figure 1.1. tinyperf, layer by layer. The five blue layers price one
step, one forward pass of the model over a batch; the two orange layers
use many step prices. Calibration attaches fitted constants to the lower
layers. Validation checks every layer against measurements.*

Read it from the bottom. The device is a GPU as a dozen rates and
sizes, most from its datasheet. Kernel cost models turn one operation's
shapes into time. A graph lists one step's operations, and passes
rewrite it the way a runtime does. `execute`, tinyperf's scheduler (not
the serving engine's), prices every operation, adds them up and reports
which resource limited each. Model builders turn a model's configuration
into one step's graph. The serving simulator and sweeps use many step
prices.

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
`Methodology.CALIBRATED` selects the *calibrated tier*, the full
mechanism with the calibration's constants. The two numbers are the two
an LLM server's user feels:

- **TTFT**, time to first token: from sending a request to its first
  output token, dominated by the *prefill*, one pass over the prompt;
- **TPOT**, time per output token: the mean gap between the later
  tokens. In a batch that decodes together, it is one decode step.

Chapter 14 is their home. Here they are measured through vLLM's offline
batch API, with no server in front: b prompts go to the engine together,
TTFT is the time until all b have their first token, and TPOT is the
mean of the 128 decode steps that follow. Those steps see contexts from
about s to s + 128 tokens, so the model prices one step at the mean,
s + 64.

Inside, `StepLatencyModel` runs the whole map for each step shape. Here
is the function that does it, with the builder's MoE, prefix-cache and
pipeline-stage arguments and two more optional passes trimmed:

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

The builder emits one step's graph, any parallel layout baked into its
shapes (chapters 6 and 12). Passes fuse attention into the kernel an
engine runs (chapter 7) and change precision (chapter 10). `execute`
prices the graph at the chosen tier (chapters 4 and 5). A cache keyed
by step shape lets the serving simulator price thousands of steps
(chapter 14).

## How close is it?

We compare with vLLM, a widely used serving engine, running Qwen3-8B in
bf16 on one RTX A6000 over a grid of batch sizes and prompt lengths
(`data/validation/vllm_qwen3_8b_rtx_a6000.json`). Throughout the book a
ratio is **predicted ÷ measured**: above 1 the model reads high, below 1
it reads low. And every number says what kind it is:

- *held out*: nothing in the model was fitted to it, and no mechanism
  was designed with it in view;
- *fitted*: it supplied a constant;
- *in-sample*: it was used to build a mechanism;
- *recorded*: it was stored when measured, and is not recomputed by the
  current model. This one combines with the others: a number can be
  recorded and held out.

A number with no measurement behind it is a *projection*.

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

The **typical error** of a set of ratios is their geometric mean
distance from 1 in either direction, `exp(mean |ln ratio|) − 1`. Every TTFT is within
5%, from 80 ms to 10.4 s. TPOT is within 1.3% up to batch 8, and at
batch 32 reads 2.0% and 4.2% low.

Nothing in the model was fitted to these times: its constants come from
kernels timed on their own (cuBLAS GEMMs, launch costs, the engine's
decode-attention kernel, this model's layer GEMMs by row count). But a
mechanism was designed with this grid in view. It exposed a flaw in how
the model priced a batch of prompts (Table 1.5), and every change since
has had to keep it inside fixed bounds. So Table 1.1 is in-sample, for
one dense model on one GPU and one engine.

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
the time goes to large kernels deep inside one regime. Much of the rest
is the datasheet itself: this GPU's best GEMMs sustain 0.75 of the
tensor cores' peak, and the roofline's TTFT reads 0.66–0.75. The
mechanism closes part of the gap, the measured rates most of the rest.

## Asking it questions

Table 1.3 asks the architect's what-if at the projected tier, which uses
only the device file and so works for any device you can write down.
Its times read 10–23% low on this GPU (Table 1.2), so compare the rows
with each other, not with a stopwatch.

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
  this into prediction intervals.
- **A library's choices you haven't measured.** Kernel libraries pick
  kernels by heuristics a model can't derive. Chapter 3 measures a GEMM
  that takes 73% longer at 129 rows than at 128, consistent with cuBLAS
  changing its tiling there (not profiled). The model carries such edges
  only once they are measured.
- **Anything never measured.** Which experts an MoE router picks depends
  on the text. Without a measurement the model assumes uniform routing,
  which reads 28–44% high on one model's decode steps from batch 8 up
  (chapter 9). The acceptance rate of speculative decoding is an input
  you supply (chapter 19). A GPU without public rates can't be
  described at all.

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
