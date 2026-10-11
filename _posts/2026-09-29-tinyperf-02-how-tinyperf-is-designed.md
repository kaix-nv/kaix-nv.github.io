---
layout: post
title: "Building tinyperf, chapter 2: How tinyperf is designed"
date: 2026-09-29 12:02:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/02-how-tinyperf-is-designed/
excerpt: "Chapter 1 drew tinyperf, the performance model this book builds, as a stack of layers. This chapter opens the stack. How does a price travel from a GPU's datasheet and a model's configuration to a latency? What crosses each boundary between the layers, why are the boundaries drawn where they are, and what does each choice cost? The rest of the book builds the layers one at a time. This chapter is the plan they fit into, and the place to come back to when you want to change something."
---

*[Building tinyperf](/series/tinyperf/) · Part I: Pricing one kernel · Code: [`tinyperf/`](https://github.com/kaix-nv/tinyperf/tree/dd27bf0/tinyperf) at commit `dd27bf0`, chiefly [`scheduler.py`](https://github.com/kaix-nv/tinyperf/blob/dd27bf0/tinyperf/scheduler.py), `execute`, and [`serving.py`](https://github.com/kaix-nv/tinyperf/blob/dd27bf0/tinyperf/serving.py), `StepLatencyModel` and `simulate` · Every table in this chapter comes from `python3 book/scripts/ch02_design.py`.*

Chapter 1 drew tinyperf, the performance model this book builds, as a
stack of layers. This chapter opens the stack. How does a price travel
from a GPU's datasheet and a model's configuration to a latency? What
crosses each boundary between the layers, why are the boundaries drawn
where they are, and what does each choice cost? The rest of the book
builds the layers one at a time. This chapter is the plan they fit
into, and the place to come back to when you want to change something.

The short answer: tinyperf is two streams that meet in one function.
One stream turns a file of GPU rates into a record that can't change
once made, adjusted by constants fitted on the real GPU. The other
turns a model's configuration and the shape of one step (one forward
pass of the model over a batch) into a list of operations, rewritten
the way a runtime would run them. A function called `execute` prices
each operation with a cost model chosen by its kind, and adds the
times. Above it, a step model remembers those prices, and a simulator
plays a serving engine's scheduler over them. Each boundary passes data
with known fields, so each layer can be changed, questioned and checked
largely on its own.

By the end of this chapter you will know:

- how one price is made, layer by layer, and where each of its numbers
  comes from;
- the contracts between the layers, in the code;
- why the GPU's rates and the fitted constants are data, while the
  model is code;
- where each kind of question enters the model;
- what each piece is worth, measured by taking it away;
- the designs tinyperf turned down, and what its own design costs.

## Two streams and a meeting point

A few words first, each built properly in a later chapter:

- a *step* is one forward pass of the model over a batch: a *prefill*
  step reads prompts, a *decode* step produces one token for each
  sequence in the batch (chapter 7);
- a *step shape* is what a step's price depends on, given the model and
  how it is split across GPUs: the phase, the batch, and the prompt
  length of a prefill or the context of a decode (the tokens already in
  the KV cache);
- the *device* is a GPU described as about a dozen rates and sizes
  (chapter 3);
- the *calibration* is a set of constants fitted to kernels timed on
  the real GPU (chapter 5);
- the *tier* says how much of reality a price includes: the *speed of
  light* is a roofline, max(math time, memory time), for every
  operation; *projected* adds the kernels' mechanisms at datasheet
  rates;
  *calibrated* also applies the calibration (chapter 5);
- a *graph* is one step written down as an ordered list of *operator
  records*, one per kind of kernel call, each with its shapes, FLOPs,
  bytes and *family*, the kind of cost model that prices it (chapter 6);
  a *model builder* writes the graph for a model and a step shape
  (chapter 7);
- a *pass* rewrites a graph the way a runtime would run it, for example
  fusing attention's three kernels into one (chapter 8).

Figure 2.1 shows how they connect.

![A data-flow diagram: on the left a device file becomes a Device
record and the GPU's calibration is loaded; on the right a model preset
and a step shape become a graph that passes rewrite; both meet in
execute, whose report feeds StepLatencyModel, whose step prices feed
simulate.](/assets/tinyperf-book/ch02-flow.svg)

*Figure 2.1. How a price is made. The GPU stream (left) and the work
stream (right) meet in `execute`, which applies the calibration and
prices the graph. Above it, the step model caches step prices and the
simulator plays requests through an engine's scheduler. Each arrow is
labeled with what crosses it.*

The two streams know nothing of each other: the device doesn't know
which model will run on it, and the graph doesn't know which GPU will
price it. That is the first design decision. A question about the GPU
changes the left stream; one about the model or a rewrite, the right
one; one about the software stack, the calibration or the step model.
The code that prices a graph doesn't move.

## One price, followed through

Table 2.1 follows one price through every layer: chapter 1's decode
step of 8 sequences, each with a 2,048-token prompt, priced at its mean
context of 2,112 tokens (the prompt plus half of the 128 tokens
generated), on an NVIDIA RTX A6000 at the calibrated tier. The device
row uses the GPU's SMs (streaming multiprocessors, its processors),
MACs (multiply-accumulates), DRAM (its main memory) and L2 (the on-chip
cache that all DRAM traffic passes through). The last row compares the
price with a measurement: a ratio is predicted ÷ measured, and above 1
the model *reads* high. *In-sample* means the measurement was used
while the mechanisms were built (chapter 1).

```
Table 2.1  One price through the layers: Qwen3-8B, 8 sequences decoding at mean context 2112, RTX A6000, calibrated tier
  layer         result
  device        84 SMs x 512 bf16 MACs per clock x 2 FLOPs x 1.8 GHz = 154.8 TFLOPS; DRAM 768 GB/s; L2 2000 GB/s
  calibration   math x 0.75 = 116.1 TFLOPS; DRAM x 0.90 = 691.2 GB/s; L2 set to 4000 GB/s; launch 3.5 us
  step model    context 2112 lies between the 256-token buckets 2048 and 2304: two graphs to price
  builder       decode, batch 8, KV 2048: 16 operator records standing for 471 kernel calls
  pass          fuse_attention: 14 records, 399 calls (scores, softmax and output become one kernel)
  execute       GEMMs 23.184 + attention 4.026 + memory-bound 0.845 = 28.055 ms; GEMM and attention rows bound by DRAM: 6 of 6
  step model    KV 2304 prices at 28.510 ms; 2112 is 0.25 of the way: 28.055 + 0.25 x 0.455 = 28.169 ms
  measured      vLLM, the mean of 128 decode steps: 27.80 ms; the model reads 1.013 (in-sample)
```

Row by row:

- **Device.** The file stores factors, not peaks; the peak and the
  roofline are methods computed from them.
- **Calibration.** The calibrated tier scales the math and DRAM rates
  by fitted efficiencies (0.75 and 0.90), sets the L2 bandwidth and the
  launch cost outright, and prices on the resulting device. The launch
  cost, 3.5 µs a kernel, is the cost inside a CUDA graph, a step's
  kernel launches recorded once and replayed together, as a serving
  engine runs them (chapter 5).
- **Step model.** It prices decode contexts every 256 tokens and
  interpolates between them, so this step needs two graphs.
- **Builder.** Qwen3-8B's 36 layers are identical, so the builder
  writes each kind of kernel once with a count of 36: 16 records stand
  for 471 calls (13 per layer, plus the embedding, the final norm and
  the LM head, the GEMM that turns the last hidden state into scores
  over the vocabulary).
- **Pass.** `fuse_attention` replaces the score GEMM, the softmax and
  the output GEMM with one fused attention kernel, as FlashAttention
  does: 14 records, 399 calls.
- **Execute.** Each record is priced by its family's cost model and
  multiplied by its count. All six GEMM and attention rows are bound by
  DRAM (memory-bound operations are, by definition): with 8 sequences,
  a decode step is reading weights and cache.
- **Step model, again.** 2,112 lies a quarter of the way from 2,048 to
  2,304, and the step reads 1.013 of the measured 27.80 ms.

## Where the numbers live

Table 2.2 says where each number of Table 2.1 came from, and what kind
of number it is.

```
Table 2.2  Where Table 2.1's numbers come from
  number                                        where                                         kind
  SMs, MACs per SM per clock, clock, DRAM       data/devices/rtx_a6000.json                   datasheet
  L2 bandwidth 2000 GB/s                        data/devices/rtx_a6000.json                   estimate (no datasheet figure)
  math 0.75, DRAM 0.90, L2 4000 GB/s            data/calibration/rtx_a6000.json               fitted to 27 GEMMs timed on the GPU
  launch 3.5 us a kernel in a CUDA graph        data/calibration/rtx_a6000.json               measured on a B200, carried over
  decode attention: 7 us a call, KV at 0.96     data/calibration/rtx_a6000.json               fitted to the engine's kernel timed alone
  36 layers, hidden 4096, 32 and 8 heads        tinyperf/nets/transformer.py, qwen3_8b()      the checkpoint's configuration
  tiles, waves, L2 reuse, launch per kernel     tinyperf/gemm_model.py                        mechanism (chapter 4)
  256-token context buckets                     tinyperf/serving.py, KV_BUCKET                a modeling choice
  27.80 ms                                      data/validation/vllm_qwen3_8b_rtx_a6000.json  measured, tools/measure_vllm.py
  imports in tinyperf/ from outside Python's standard library: none
```

The rule is: **a GPU's rates and the constants measured on it are data;
a mechanism is code.** A GPU is a JSON file in `data/devices/`. A
calibration is a JSON file in `data/calibration/`, and every one carries
a `source` field that says what was timed, with which software, and how
well the fit did. The measurements that check the model are files in
`data/validation/`, beside the predictions that were written down before
their runs, and the scripts that took them are in `tools/`.
`tests/test_core.py` fails if a validated price leaves its bounds
(chapter 22), and `book/scripts/check_chapters.py` fails if a chapter's
table no longer matches its script.

So a new GPU is a new file, not new code; a mechanism checked on one
GPU is the same code on the next; and a price can be traced to what
it rests on, down to a constant carried over from another GPU, as the
launch cost here is (chapter 5 says how loosely the A6000 bounds it).

The calibration has a rule of its own: each field is measured on its
own kernel, never fitted to the end-to-end times it will later be
checked against, so comparing with those stays a test. The few fields
that break the rule, fitted to an engine's step times or TTFT (time to
first token), say so, and are the first to distrust (chapter 5).

Some numbers do live in the code, each with a comment saying where it
came from:

- model configurations: a preset function such as `qwen3_8b()` is short
  and reads like the checkpoint's `config.json`;
- default efficiencies from published reports, used where a GPU's
  calibration has none, such as FlashAttention's 0.65 of peak math in
  `gemm_model.py`;
- measured tables of how many experts a step of a mixture-of-experts
  (MoE) model touches, by its tokens, in `tinyperf/nets/routing.py`
  (chapter 10);
- constants taken from one engine, vLLM: the cost of its sampler, which
  picks each next token, and its memory accounting, each measured on
  one card (chapters 16 and 18), and its list of CUDA-graph sizes.

The last line of Table 2.2 checks a different promise: the package
needs nothing but Python.

## The contracts between layers

Each arrow in Figure 2.1 is a contract: an object with known fields.
This section quotes each from the code at commit `dd27bf0`, trimmed
where marked.

### A device is a frozen record

```python
@dataclass(frozen=True)
class Device:
    name: str
    sm_count: int
    boost_clock_ghz: float
    # dense tensor-core MACs (multiply-accumulates) per SM per clock, by dtype key
    tensor_macs_per_sm_clk: dict = field(default_factory=dict)
    dram_bw_gbps: float = 0.0          # GB/s
    ...                                # L2, launch, links, memory, power, price; two small properties

    def peak_tflops(self, dtype: DType) -> float:
        """Dense peak at the sustained clock, TFLOPS. 1 MAC = 2 FLOPs."""
        macs = self.tensor_macs_per_sm_clk.get(dtype.key)
        ...                            # an error if this GPU has no rate for dtype
        return self.sm_count * macs * 2 * self.boost_clock_ghz * self.sustained_clock_fraction / 1e3
```

`frozen=True` means no field can be reassigned after the record is made
(the per-SM rates sit in a dictionary, whose contents a careless caller
could still change). A what-if is a new record:
`Device.load("rtx_a6000", dram_bw_gbps=1536)` reads the file and returns
a copy with one field replaced, and remembers which fields it replaced,
so a calibration that would overwrite one of them can warn you. Derived
numbers are methods, not fields, so none can go stale when a field
changes. `sustained_clock_fraction`, the clock a power-limited GPU
holds, is 1.0 unless a what-if sets it.

### An operator is a record with a family

```python
class Operator:
    """Base class. Subclasses set op_type and implement build_outputs()."""

    op_type = "base"

    def __init__(self, name: str, inputs: list, **attrs):
        self.name = name
        self.inputs: list[Tensor] = inputs
        self.attrs = attrs
        self.outputs: list[Tensor] = self.build_outputs()
```

A tensor here is only a name, a shape and a number format: it carries
no data, because nothing is computed. Each operator class sets
`op_type`, its family, and works out its output shapes from its inputs.
`Linear`, for example, is family `"gemm"` and records its row count M,
its K and N, and its weight bytes. Anything else rides in `attrs`, an
open dictionary: the count, a number format set by a pass, a sparsity
tag. A decorator, `Graph.register`, gives the graph one builder method
per operator class, so a model builder reads like the model. Here is
Qwen3-8B's feed-forward block, from `build_llm_graph`:

```python
        up = g.Linear("ffn_gate_up", y, out_features=2 * ffn_l, count=L)
        act = g.Elementwise("swiglu", up, count=L)
        act = Tensor("act", (tokens, ffn_l), dt)
        down = g.Linear("ffn_down", act, out_features=h, count=L)
        if tp > 1:
            down = g.AllReduce("ffn_allreduce", down, group_size=tp, count=L)
        x = g.Elementwise("residual_ffn", down, x, count=L)
```

The gate-up GEMM produces two halves that SwiGLU, the activation,
multiplies together, so its output is half as wide; the third line
writes that half-width tensor by hand, because `Elementwise` keeps its
input's shape. Three design choices show here. First, `count=L`: a
repeated layer is stored once with its count, which keeps a graph tens
of records long. Second, the layout is baked into the shapes: with
tensor parallelism (tp, each GPU holding a slice of every weight
matrix; chapter 12) `ffn_l` is each GPU's share of the width, and the
builder adds the all-reduce that sums the GPUs' partial results. The
graph is always one GPU's view of the step. Third, the graph is a list,
not a web of dependencies. The builder writes operators in program
order, so the list is an order in which they can run, and one GPU runs
a step's kernels one after another: a list is all the pricing needs
(chapter 6).

### A cost function takes a record and a context

Every family has one cost function, and every cost function has the
same shape: an operator and a context in, an `OpResult` out. The
memory-bound family, `"rw"` (norms, element-wise operations, the
embedding lookup), is the simplest:

```python
def _exec_rw(op: Operator, ctx: ExecContext) -> OpResult:
    nbytes = op.moved_bytes()
    launch = 0.0 if ctx.methodology is Methodology.SOL else ctx.device.kernel_launch_us
    time_us = ctx.device.dram_time_s(nbytes) * 1e6 + launch
    return OpResult(
        name=op.name, op_type=op.op_type, time_us=time_us,
        bytes=nbytes, bound="dram",
    )

...                                    # three more cost functions and a helper

EXEC_MODELS = {"gemm": _exec_gemm, "rw": _exec_rw, "comm": _exec_comm,
               "fmha": _exec_fmha, "linattn": _exec_linattn}
```

The five families are GEMMs (chapter 4), memory-bound operations,
collectives between GPUs (chapter 12), fused attention (chapter 8) and
linear attention, whose prefill cost grows linearly with the sequence
and whose decode updates a fixed-size state (chapter 9). `execute`
loops over the records, here trimmed of the set-up before the loop and
the notes after it:

```python
    for op in graph:
        result = EXEC_MODELS[op.op_type](op, ctx)
        result.count = op.attrs.get("count", 1)
        report.rows.append(result)
```

The set-up is where the tier takes effect. At the calibrated tier,
`execute` loads the GPU's calibration if it wasn't given one, picks the
launch cost for the software stack, applies the calibration to the
device (below), and copies the calibration's per-kernel fields into
the context, `ExecContext`. That is a frozen record of 17 fields: the
device, the number format, the tier, and 14 fields for particular
kernels, such as the decode attention kernel's 7 µs per call. A cost
function reads only its operator, its context and the constants of its
own code, so no file or global switch can change a price behind the
call's back.

The report, `RunReport`, holds the rows, each with what bound it
(math, DRAM, the L2 or a link), and how they were priced: the tier, the
launch cost, and notes where a reader might guess wrong, such as
attention left unfused. Its total is the sum of every row's time times
its count, the claim that kernels run one after another. For a step on
one GPU, replayed as a CUDA graph, that prices decode steps within a
few percent (chapter 1). Where overlap is real, the report also gives
the other end of the bracket, every collective hidden behind compute,
and the step model takes the fraction hidden, `comm_overlap`, as an
input (chapter 12).

### Calibration transforms the device

```python
    def apply(self, device: Device) -> Device:
        ...                            # the docstring; a warning if this replaces a what-if override
        macs = {
            k: v * self.math_efficiency
            for k, v in device.tensor_macs_per_sm_clk.items()
        }
        fields = dict(
            tensor_macs_per_sm_clk=macs,
            dram_bw_gbps=device.dram_bw_gbps * self.dram_efficiency,
        )
        if self.l2_bw_gbps is not None:
            fields["l2_bw_gbps"] = self.l2_bw_gbps
        if self.kernel_launch_us is not None:
            fields["kernel_launch_us"] = self.kernel_launch_us
        return replace(device, **fields)
```

Inside `execute`, the projected and calibrated tiers are not two
models: they run the same cost functions on two devices, the
datasheet's and the one `apply` returns, the calibrated tier with its
per-kernel fields in the context. (The speed of light drops the
mechanisms and keeps each operation's roofline.) Some mechanisms exist
only where something was measured, such as the decode-attention floor
or the per-request overhead a server adds; at the projected tier they
are absent, not projected. Because the efficiencies multiply, a what-if
keeps its calibration: twice the DRAM bandwidth is twice the datasheet
rate, still at 0.90 of it.

`for_stack` picks the launch cost for how kernels are issued: "eager"
(PyTorch issuing each as Python reaches it, 25 µs here) or "graph" (a
CUDA graph, 3.5 µs). `execute` alone defaults to eager, and its report
notes it; `StepLatencyModel`, built for serving, defaults to the
graph.

### A step price is a cached function of a step shape

The serving simulator prices thousands of steps, but their shapes
repeat. `StepLatencyModel` builds, rewrites and executes a graph once
per step shape and remembers the price. Two parts of the key, trimmed
here, matter only for MoE models:

```python
    def _price(self, phase: str, batch: int, seq: int, shared: int = 0, moe_real: int | None = None,
               route: float | None = None) -> float:
        key = (phase, batch, seq, shared) + ...   # two MoE-only parts of the key
        if key not in self._memo:
            ...                                   # the MoE arguments for the builder
            self._memo[key] = self._traverse_us(phase, batch, seq, shared=shared, **graph)
        return self._memo[key]
```

`decode_us` rounds the context down to a multiple of `KV_BUCKET`, 256,
and interpolates linearly to the next, as Table 2.1 did; a prefill is
priced at its exact length. Under the cache sits `_graph_report`,
chapter 1's quoted function: build the graph, run the passes, call
`execute`. Between them, `_traverse_us` adds what isn't in one graph:
the hand-offs between pipeline stages when the layers are split across
GPUs (chapter 13), and weights streamed from host memory, hidden under
compute by default (chapter 18). The step model also prices the
engine's sampler, which picks each next token (`sampler_us`,
chapter 16).

### The simulator sees only step prices

`simulate` plays the engine forward in time: requests arrive, the
engine's scheduler forms each step's batch, the step is priced, and the
clock moves on (chapter 15). It never sees a graph. Here is how the
scheduler that `serving.VLLM` runs charges a step, trimmed:

```python
            n_dec = padded_batch(len(decoders), cudagraph_sizes) if decoders else 0
            if chunked:
                ...                    # a mixed step: decodes beside chunks of prompts
                step_us = lat.mixed_step_us(len(decoders), kv_now, pre_kv,
                                            max(1, new_tokens // len(chunked)), len(chunked))
            elif decoders and p.routing_by_position:
                ...                    # an MoE model whose experts follow the replies' positions
            else:
                step_us = lat.decode_us(n_dec, kv_now, real=len(decoders)) if decoders else 0.0
            ...                        # whether the step leaves the CUDA graph; which prompts finish; the trace
            t += step_us + (engine_us(len(decoders) + len(chunked), carries_prefill=leaves_graph)
                            + sampler_us(len(decoders) + len(chunked)) + producer_us(finishing) if step_us else 0.0)
```

`lat` is the step model. A pure decode step is padded to the next
recorded CUDA-graph size (`n_dec`); a *mixed step*, decodes beside
chunks of prompts, runs outside the graph at its real rows (chapters 15
and 17). The simulator asks for the step's price and adds the engine's
own costs: its bookkeeping (`engine_us`, from the calibration, measured
at zero on this stack; chapter 5), the sampler for every row, and, on
a disaggregated prefill server (one that only runs prompts and hands
their KV cache to another; chapter 21), the cost of saving the cache
to send (`producer_us`). It
calls only a few of the step model's methods (`prefill_us`, `decode_us`,
`mixed_step_us`, `sampler_us`, `kv_producer_us`) and reads its
calibration. So anything that answers those can stand in. The code uses
this: to ask how far a 2% error in every step moves the latency, it
wraps the step model in a class that multiplies each answer by a
factor, 0.98 or 1.02, and changes nothing else (chapter 19).

The engine's settings travel as one record, `EngineConfig`, under the
engine's own names (`max_num_seqs`, the most sequences a step may run;
`max_num_batched_tokens`, the most tokens). The preset `serving.VLLM`
holds vLLM 0.15.1's defaults plus the calibrated tier, the CUDA-graph
launch cost and its sampler; `VLLM.step_model(...)` and
`VLLM.simulate(...)` build the step model and the run from it.

## Where a question enters

Because each layer hands data up, a question enters at one layer and
everything above it runs unchanged. Table 2.3 asks one question at each
layer about chapter 1's step and server. FFN is the feed-forward
network; 2:4 sparsity keeps two of every four weights; p95 TTFT is the
TTFT that 95% of requests beat. vLLM's preset samples *top-p*, drawing
each token from the most likely ones, which costs a sort; chapter 1's
grid ran *greedy*, always taking the most likely token.

```
Table 2.3  One question per layer: Qwen3-8B on an RTX A6000, calibrated tier
  layer       question                           change                                         answer
  device      twice the DRAM bandwidth?          Device.load("rtx_a6000", dram_bw_gbps=1536)    step 28.17 -> 14.92 ms
  calibration datasheet rates, nothing fitted?   methodology="proj"                             step 28.17 -> 24.93 ms
  pass        2:4-sparse FFN weights?            sparsity=SPARSITY_2_4_FFN                      step 28.17 -> 21.22 ms
  builder     split over two GPUs?               tp=2, on the pair's device file (*)            step 28.17 -> 18.59 ms
  builder     Qwen3-14B instead?                 qwen3_14b()                                    step 28.17 -> 47.31 ms
  step model  vLLM's step, with its sampler?     VLLM.step_model(...).decode_step_us            step 28.17 -> 29.11 ms
  simulator   half the running sequences?        VLLM.with_(max_num_seqs=32)                    p95 TTFT 787 -> 1184 ms
  simulator   512 tokens a step, not 2048?       max_num_batched_tokens=512                     p95 TTFT 787 -> 1095 ms
  step: 8 sequences decoding at mean context 2112; the preset samples top-p, chapter 1's grid ran greedy
  (*) Device.load("rtx_a6000_pcie_pair"): two RTX A6000s and the PCIe link between them
  p95 TTFT: vLLM with max_num_seqs=64 at 3 req/s, 240 requests of 1024 tokens in, 128 out (chapter 1's Table 1.4)
```

Each answer is a model's answer, not a measurement:

- Twice the DRAM bandwidth nearly halves the step, which reads weights
  and cache. The projected tier reads 24.93 / 27.80 = 0.90.
- The sparsity pass marks the FFN weights 2:4-sparse, the zeros not
  stored (chapter 11), so those GEMMs stream about half their bytes. No
  sparse kernel was timed on this GPU: it is a projection.
- Splitting over two GPUs (tp=2) halves each GPU's weights but adds
  all-reduces over their PCIe link, priced from a measured curve:
  18.59 ms, not 14. It needs the pair's device file too: a link is part
  of the device. A bigger model is only a different configuration.
- vLLM's step adds its sampler, 0.94 ms for 8 rows on the top-p path
  that the instruct checkpoint's settings select (chapter 16). The
  greedy grid's 27.80 ms compares with the forward alone.
- The simulator rows change only the engine's settings. No step's price
  changes; the queue does. The p95 TTFT goes from 787 to 1,184 ms when
  the engine may run 32 sequences at once instead of 64, and to 1,095 ms
  when a step may carry 512 tokens, decodes included (chapter 15).

Each question has one place to go, but not every answer is equally
good: a question about a kernel nobody has timed gets a projection.

## What each piece is worth

Is each piece of the design earning its place? One way to tell is to
take it away and price chapter 1's eight measured cells again (batches
1 to 32, prompts of 512 to 8,192 tokens; TTFT is the prefill, and TPOT,
time per output token, is one decode step at the mean context). Table
2.4 summarizes each set of ratios by its range and its typical error,
`exp(mean |ln ratio|) − 1`, the geometric-mean distance from 1. Two of
its rows remove a measured field: the *row curve*, cuBLAS's measured
time ÷ the model's for a dense GEMM by row count (cuBLAS is NVIDIA's
GEMM library; chapter 16), and the decode-attention kernel's floor and
rate (chapter 8).

```
Table 2.4  Take one piece away: chapter 1's eight cells, Qwen3-8B under vLLM on one RTX A6000, model/measured
  piece removed                              layer        TTFT range  typical error  TPOT range  typical error
  nothing: the calibrated tier               -             0.97-1.05           1.9%   0.96-1.01           1.4%
  the fusion pass (scores through DRAM)      pass          1.00-1.39          13.1%   0.99-1.04           1.7%
  separate prompts (a batch as one prompt)   step model    0.97-2.79          42.0%   0.96-1.01           1.4%
  CUDA-graph launch (eager's 25 us instead)  calibration   0.99-1.08           2.7%   1.16-1.38          30.1%
  the fitted L2 rate (the device file's)     calibration   1.00-1.07           2.7%   0.96-1.01           1.4%
  the measured row curve for dense GEMMs     calibration   0.95-1.05           2.0%   0.96-1.01           1.4%
  the measured decode-attention kernel       calibration   0.97-1.05           1.9%   0.94-1.00           1.7%
  every fitted constant (projected tier)     calibration   0.77-0.82          26.6%   0.84-0.90          13.3%
  all but the roofline (speed of light)      all           0.66-0.75          37.9%   0.79-0.83          22.0%
  the row curve corrects dense GEMMs of 33-1280 rows; the cells' decode steps have 1-32, and only the 512-row prefill (factor 1.016) falls inside
  decode attention, measured price / generic price: 1.06 where attention is 39% of the step, 2.03 where it is 2%
```

The large rows each name one mistake a performance model can make:

- **Without the fusion pass**, attention's scores go through DRAM, and
  long prompts read up to 39% high.
- **Without separate prompts**, a batch of prompts is priced as one long
  prompt, every token attending to every other prompt's tokens: up to
  2.79 times high. That was the error in chapter 1's Table 1.5, the
  predictions written before this grid was measured.
- **With eager PyTorch's launch cost**, decode reads 16–38% high: the
  right constants for the wrong software.
- **Without the fitted constants**, everything reads low (the projected
  and speed-of-light rows, as in chapter 1's Table 1.2).

The small rows are as instructive. The row curve and the measured
decode-attention kernel barely move this grid, because its shapes sit
where the corrections are small. Its decode steps have too few rows for
the row curve, which touches only the one 512-row prefill. Where
attention is a large part of a step, the measured price is within 6% of
the generic one; where they differ twofold, attention is 2% of the step.
Each field was added for steps where it matters, and checked there: on
GEMMs held out from its fit, each GPU's shapes at tp=2, the row curve
takes GEMMs of 40 to 64 rows from a typical error of 7.0% to 4.2% (Table
16.2). A grid can only vouch for the mechanisms it exercises, which is
why each mechanism gets its own measurement.

## Choices turned down

Each alternative would have bought something; here is why tinyperf
does without it.

- **Simulating cycles.** A cycle-level simulator can study a memory
  system in detail, but needs a detailed description of the chip and
  every kernel's code, and runs slower than the hardware. tinyperf
  answers in milliseconds, for any GPU with a datasheet.
- **Fitting a curve to measured times.** A regression on measured steps
  is accurate near its points, but can't price an unbuilt GPU, and says
  nothing about why. tinyperf fits four physical constants per GPU, and
  adds a measured table only where a library's or a model's behavior
  can't be derived, such as the row curve, NCCL's collectives on one
  link (NCCL is NVIDIA's library for communication between GPUs;
  chapter 12) or how many experts an MoE step touches (chapter 10).
- **A dependency graph with concurrent streams.** One GPU runs a step's
  kernels in order. The overlaps that matter, communication behind
  computation and weights streamed from the host, enter as declared
  assumptions.
- **A generic batching loop.** Latency under load comes from the queue,
  and the queue depends on the engine's exact rules: when it admits a
  request, how many tokens a step may carry, what it does when the
  cache is full. So the simulator copies one engine's rules, vLLM
  0.15.1's (chapter 15).
- **NumPy, or a compiled core.** Plain Python is slow at arithmetic,
  but a graph is tens of records and the cache removes repetition. In
  exchange there is nothing to install.

## Where it breaks

The design's costs show in the code:

- **Names and `attrs` are part of the contract.** `apply_sparsity`
  marks GEMMs whose names start with the prefixes it is given
  (`"ffn_"`, `"moe_gate_up"` and `"moe_down"` for FFN sparsity), and
  refuses names outside a fixed list of weight GEMMs; precision recipes
  match by prefix too (chapter 11). The GEMM cost function gives expert
  GEMMs their measured rates by name, and leaves `"lm_head"` out of the
  row curve. Passes leave their tags in `attrs`, an open dictionary the
  cost functions read by key. A builder that renames an operator, or a
  pass that misspells a key, silently loses a price. (`fuse_attention`
  matches by type and data flow instead.) And passes change a graph in
  place: to price it before and after one, build it twice.
- **The context grows with the calibration.** Each per-kernel field a
  cost function reads adds a field to `Calibration`, one to
  `ExecContext` and a line in `execute` that copies one to the other:
  14 at this commit. The step model reads others directly.
- **Many knobs in one place.** `StepLatencyModel` takes 24 arguments;
  `EngineConfig` gathers only the engine's.
- **The layers are not strict.** `simulate` reaches past the step model
  to the model and the device to size the KV cache, and the engine's
  own costs live in a GPU's calibration file.
- **Adding is an approximation.** Kernels that truly run at once, or a
  link shared by two jobs, are outside the model; overlap fractions are
  declared, not derived.
- **A calibration belongs to one GPU and one software stack**, and
  nothing in the code checks that it matches the software you run.
  Table 2.4's eager row shows what a mismatch costs. Another engine, or
  another version of this one, needs its own rules.
- **What isn't a rate isn't modeled.** How the SMs are wired, how warps
  (groups of 32 threads in lockstep) are scheduled, how caches replace
  lines: these enter only through mechanisms or measured constants. A
  clock is one number, and a deployment has one GPU type. Work outside
  the steps (the HTTP request, streaming the answer) is one constant per
  request, set on an idle server (chapter 17).

## What you built

- A map of how a price is made: two streams, a device and a graph,
  meeting in `execute`, with a step model and a simulator above.
- The contracts at each boundary, from a frozen device to step prices
  by shape, and the rule that measured numbers are data.
- A way to judge a piece: take it away, and see what moves.

## Exercises

1. Price Table 2.1's step on an H100 (`Device.load("h100_sxm")`) with
   `StepLatencyModel` at the calibrated tier. Which rows of Table 2.1
   change, and which can't?
2. `count` is only compression. Write a function that replaces each
   record of a graph with `count` copies of count 1, price both graphs
   with `execute` (pass `stack="graph"`), and check that the totals
   agree to within 1e-6 µs.
3. Write a step model of your own: a class whose `decode_us`,
   `prefill_us`, `mixed_step_us` and `sampler_us` read chapter 1's
   measured TTFT and TPOT, and pass it to `simulate` as
   `latency_model`. What does the simulator ask for that the
   measurements don't contain?
4. Change a field of a what-if device, say twice `l2_size_mb`, and find
   the function that reads it. Which questions can the L2 size answer?

---

*[← Chapter 1: What a performance model is for]({% post_url 2026-09-29-tinyperf-01-what-a-performance-model-is-for %}) · [Contents](/series/tinyperf/) · [Chapter 3: A GPU as a handful of rates →]({% post_url 2026-09-29-tinyperf-03-a-gpu-as-a-handful-of-rates %})*
