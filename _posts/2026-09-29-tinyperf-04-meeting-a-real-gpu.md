---
layout: post
title: "Building tinyperf, chapter 4: Meeting a real GPU: calibration and its tiers"
date: 2026-09-29 12:04:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/04-meeting-a-real-gpu/
excerpt: "Chapters 2 and 3 priced a kernel from a GPU's datasheet. Put that model next to a real GPU and it is wrong: against 27 GEMMs timed with cuBLAS on an RTX A6000, it predicts between 0.11 and 0.89 of the measured time, and is typically 48.5% off. It assumes a GPU at its boost clock, memory that delivers every byte it is rated for, and 3 µs to launch a kernel."
redirect_from:
  - /tinyperf/perf-modeling/2026/08/29/building-tinyperf-m26.html
  - /tinyperf/perf-modeling/2026/08/31/building-tinyperf-m39.html
---

*[Building tinyperf](/series/tinyperf/) · Part I: Pricing one kernel · Code: [`tinyperf/methodology.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/methodology.py), `Calibration`, and [`tools/fit_calibration.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tools/fit_calibration.py), `fit` · Every table and the plot in this chapter come from `python3 book/scripts/ch04_calibration.py`.*

Chapters 2 and 3 priced a kernel from a GPU's datasheet. Put that model
next to a real GPU and it is wrong: against 27 GEMMs timed with cuBLAS
on an RTX A6000, it predicts between 0.11 and 0.89 of the measured time,
and is typically 48.5% off. It assumes a GPU at its boost clock, memory
that delivers every byte it is rated for, and 3 µs to launch a kernel.

The obvious fix is to fit something to the measurements. The danger is
just as obvious: give a model enough knobs and it matches any benchmark,
and then it predicts nothing it hasn't seen. How do you make the model
match a real GPU without turning it into a curve fit? And how do you
know you haven't?

The short answer: keep the mechanism and fit only a few constants, each
with a physical meaning: the tensor cores' sustained rate, the DRAM
(main memory) bandwidth a kernel gets, the bandwidth of the L2 (the
on-chip cache) and the cost of a launch. Test them on shapes they were
not fitted to, measure them a second way, and read the errors that
remain. And report the fitted prediction beside the unfitted ones: the
spread is the error bar.

By the end of this chapter you will know:

- the three tiers (speed of light, projected, calibrated), three ways
  to price that go from datasheet rates alone to constants fitted to a
  real GPU, and why tinyperf reports them side by side;
- how four constants are fitted on three GPUs, and three ways to check
  they are not just a curve fit;
- what the errors left after a fit say about the model;
- why launch cost belongs to the software stack (how kernels are
  issued), and what a CUDA graph (launches recorded once, then replayed)
  changes;
- how the calibrated tier grows into measured per-kernel tables, and
  the rule that keeps it honest.

## Three tiers

tinyperf can price the same model three ways. The code calls each a
*methodology* (Figure 4.1).

![Three tiers: speed of light, projected and calibrated, with the model
and the rates each one uses.](/assets/tinyperf-book/ch04-tiers.svg)

*Figure 4.1. The three tiers. Speed of light and projected both run on
datasheet rates; projected and calibrated both run chapter 3's model.
The gap between the first two is what the mechanism adds; the gap
between the last two is what the measured constants add.*

- **Speed of light (SOL)** prices each operation with chapter 2's
  roofline: operands read once, the full datasheet peak, no tiles, no
  waves, no launch cost. No real kernel is faster.
- **Projected (PROJ)** runs chapter 3's model (tiles, waves, L2 reuse,
  pipeline fill, a 3 µs launch) on datasheet rates. It is the best guess
  for a GPU nobody has measured.
- **Calibrated (CAL)** runs the same model on constants fitted to a real
  GPU's measured kernels. It predicts that GPU, running that software.

A calibration is a small frozen record per GPU, stored in
`data/calibration/<gpu>.json`. Trimmed below: the comments on its
fields, 24 fields that later chapters introduce, `apply`'s docstring,
and the methods `for_stack` (quoted later), `for_attention`, `save` and
`load`.

```python
class Methodology(Enum):
    SOL = "sol"
    PROJ = "proj"
    CALIBRATED = "calibrated"


@dataclass(frozen=True)
class Calibration:
    device_name: str
    math_efficiency: float = 1.0
    dram_efficiency: float = 1.0
    l2_bw_gbps: float | None = None
    kernel_launch_us: float | None = None
    kernel_launch_us_graph: float | None = None
    ...                                         # per-kernel fields of later chapters
    source: str = ""  # provenance: hardware, library, date, fit error

    def apply(self, device: Device) -> Device:
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

For the four constants, `apply` is the whole trick; the per-kernel
tables at the chapter's end are the exception. It hands the model a
different GPU, with slower tensor cores and memory, a fitted L2
bandwidth and a costlier launch, and every mechanism of chapter 3 runs
on it unchanged. `source` records the GPU, library versions, kernels
timed and fit error. Asking for CALIBRATED without a file is an error,
not a silent fallback to PROJ: the labels exist so a reader
can tell the tiers apart, and side by side they show how much of a
number rests on the mechanism and how much on the fit.

## Measuring kernels

`tools/measure_gemm.py` times 27 GEMMs in five regimes, chosen so each
constant's term dominates somewhere: **square** (1024³ to 8192³,
math-bound), **decode-like** (1 to 128 rows against 4096 × 4096 to
8192 × 8192 weights, memory-bound), **tall** (2048 rows against a
weight 32,768 wide or deep, like an output head in prefill), **batched**
(many small attention-like GEMMs in one call) and **tiny** (128³ and
256³). Each is timed like this:

```python
def time_gemm(batch, m, n, k, dtype, iters=30, warmup=5):
    a = torch.randn(batch, m, k, dtype=dtype, device="cuda")
    b = torch.randn(batch, k, n, dtype=dtype, device="cuda")
    fn = (lambda: torch.bmm(a, b)) if batch > 1 else (lambda: a[0] @ b[0])
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3)  # ms -> us
    return statistics.median(times)
```

PyTorch's `@` and `bmm` call cuBLAS. A CUDA event is a timestamp the GPU
writes when its stream reaches it, so the time between the two is what
the GPU saw: the kernel, plus any wait for the host to issue it. The
timings are warm: repeated calls find their inputs in L2 where they fit.

Table 4.1 compares PROJ with these timings as ratios, predicted ÷
measured: a ratio of 0.8 reads 20% low. It sums up a set of ratios by
their range and by their typical error, `exp(mean |ln ratio|) − 1`, the
geometric mean distance from 1 (chapter 3).

```
Table 4.1  The model on datasheet rates (PROJ) against cuBLAS: RTX A6000, 27 fp16 GEMMs
  regime                    shapes  measured us  model/measured
  square, 1024-8192              4     44-10299       0.51-0.84
  decode-like, 1-128 rows       16       72-227       0.65-0.89
  tall, 2048 x 32768             2    4710-4780       0.79-0.84
  batched, attention-like        3       96-530       0.77-0.86
  tiny, 128 and 256              2        30-34       0.11-0.15
  decode-like, 4096 x 4096 weights: 0.65-0.73; larger weights: 0.78-0.89
  square: 1024^3 0.51, 2048^3 0.84, 4096^3 0.84, 8192^3 0.71
  tiny: 128^3 measures 34.3 us, 256^3 29.7 us
  all 27: typical error 48.5%
```

Every ratio is below 1, but the errors are not random. Where one term
dominates, the ratio is nearly level: the tall GEMMs read 0.79–0.84. The
shortest kernels read lowest, because a fixed cost per call is missing:
a 128³ GEMM, a sliver of a microsecond of tensor-core work, measures
34.3 µs against PROJ's 3 µs launch, and the 4096 × 4096 decode-like
weights read 0.65–0.73 against up to 0.89 for larger ones. (The 8192³
GEMM's 0.71 has another cause, below.)

## Fitting four constants

The five regimes cover four terms of chapter 3's model:

```
tensor-core rate  = datasheet rate × math_efficiency      square, tall
DRAM bandwidth    = datasheet rate × dram_efficiency      decode-like, batched
L2 bandwidth      = device estimate × l2_bw_scale         square and tall, beside math
launch cost       = fitted outright                       tiny, and every short kernel
```

`tools/fit_calibration.py` minimizes the mean of
`|ln(predicted / measured)|` over the shapes. The logarithm makes the
error symmetric (20% high counts as much as 1/1.2 low) and scale-free (a
30 µs GEMM counts as much as a 10 ms one), and minimizing it minimizes
the typical error. The search is coordinate descent on a grid: for each
constant in turn, try every value on its grid with the others held,
keep the best, and sweep three times.

```python
GRIDS = {
    "math_efficiency": [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
    "dram_efficiency": [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
    "l2_bw_scale": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0],   # x the device estimate
    "kernel_launch_us": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 25.0],
}


def fit(device, gemms, sweeps=3):
    params = {"math_efficiency": 0.90, "dram_efficiency": 0.85,
              "l2_bw_scale": 1.0, "kernel_launch_us": 3.0}
    for _ in range(sweeps):
        for name, grid in GRIDS.items():
            best_v, best_e = params[name], float("inf")
            for v in grid:
                e = _error(device, {**params, name: v}, gemms)
                if e < best_e:
                    best_v, best_e = v, e
            params[name] = best_v
    ...                   # warn about any constant on the edge of its grid
    return params, _error(device, params, gemms)
```

`_error` applies the four values to the device and returns the mean
`|ln(estimate_gemm / measured)|`. A grid, not a gradient, because the
model's `max`, `min` and `ceil` make its error piecewise in the
constants, with jumps, and because a coarse grid is honest about how
finely 27 shapes can determine a constant.

```
Table 4.2  Four constants fitted to 27 cuBLAS GEMMs, and the typical error before (PROJ) and after (CAL, fitted)
  GPU        math eff  DRAM eff (GB/s)  L2 GB/s (x est.)  launch us    PROJ    CAL
  RTX A6000      0.75       0.90 (691)         4000 (2x)       25.0   48.5%   4.9%
  H100 SXM       0.75      0.80 (2682)      8250 (0.75x)       12.0   71.6%   8.5%
  B200           0.75      0.80 (6400)      27000 (1.5x)       16.0  108.8%   9.1%
  a fresh fit with the current code returns these constants on all three GPUs: yes
  L2-bound shapes at the fitted rates: RTX A6000 none; H100 SXM 3 square, 2 tall; B200 1 square
  RTX A6000: the launch constant sits on its grid's top edge (25 us); with the grid extended to 40 us the fit picks 25
```

The constants read like physics, which is the point of fitting only a
few. The tensor cores sustain 0.75 of their datasheet rate on all three
GPUs (on the A6000, mostly the power limit; see below). Memory delivers
0.80–0.90 of its rated bandwidth. The launch cost, 12–25 µs, is four to
eight times the model's 3 µs default: it is the cost of issuing a kernel
from PyTorch. The L2 bandwidth moves most, 0.75 to 2 times the device
file's estimate, and binds mostly on the H100, where it prices the large
square and tall GEMMs.

## Did we just fit a curve?

**Hold out half the shapes.** Split each GPU's shapes into alternate
halves of 14 and 13, so each half spans every regime. Fit on one half
with the unmodified `fit`, measure the other, then swap. The halves are
held out from the fit, not from the mechanism: chapter 3's L2 rule came
from the error left on one of these shapes.

```
Table 4.3  Fit on half the shapes, test on the other half: held out from the fit, not from the mechanism
  GPU        fitted on  math   DRAM L2 x est. launch us   error, fitted half  held-out half
  RTX A6000  14 even    0.75   0.85      3.0x*     20.0                 5.3%           7.4%
             13 odd     0.80   0.90      1.5x      25.0*                5.0%           5.3%
  H100 SXM   14 even    0.75   0.80     0.75x      12.0                 8.2%           8.7%
             13 odd     0.80   0.85      2.0x      16.0                 5.4%           8.0%
  B200       14 even    0.80   0.75      1.5x      16.0                10.7%           8.3%
             13 odd     0.70*  0.80      2.0x      16.0                 6.1%          12.3%
  * on the edge of its grid
  within a GPU, between halves: math 0.05-0.10, DRAM 0.05, L2 x1.3-2.7, launch 0-5 us
  H100, all 27 shapes, the other constants as fitted: mean |ln error| 0.081-0.086 with the L2 bandwidth anywhere from 0.75x to 3x the estimate
```

The held-out halves are 5.3–12.3% off, against 5.0–10.7% for the halves
the constants were fitted on: within a few points of their own, at worst
double (the B200's odd half). What the constants learned is mostly about
the GPU, not the shapes.

How far a constant moves between halves says how well the data
determines it. Within a GPU, math efficiency moves 0.05–0.10, DRAM
efficiency 0.05 and launch cost 0–5 µs, but L2 bandwidth 1.3–2.7 times,
and on the H100 the error barely changes anywhere from 0.75 to 3 times
the estimate. L2 trades against math on the same GEMMs, so read its
value as a fit, not a measurement of the L2.

**Measure it another way.** A decode-like GEMM streams its weights, so
its time should be `overhead + bytes / bandwidth`. A least-squares line
through the 12 decode-like GEMMs of 1 to 32 rows reads both constants
off with no tile model and no grid.

```
Table 4.4  Two constants read off a straight line through 12 decode-like GEMMs (1-32 rows)
  GPU        line: GB/s  of datasheet  intercept us   fit: DRAM eff  launch us
  RTX A6000         698          0.91          23.8            0.90       25.0
  H100 SXM         2693          0.80          12.4            0.80       12.0
  B200             7743          0.97          18.6            0.80       16.0
  B200: decode-like GEMMs take 22-42 us; the largest weight, 134 MB, streams in 16.8 us at the datasheet rate
  B200: L2 126 MB; the decode-like weights are 33.6, 90.2, 134.2 MB (timed warm)
```

On the A6000 and the H100 a ruler and the whole model find the same two
numbers: 0.91 against 0.90 and 23.8 against 25 µs; 0.80 against 0.80
and 12.4 against 12 µs. On the B200 they disagree about bandwidth, 0.97
against 0.80. Its largest decode-like weight streams in 16.8 µs at the
datasheet rate, so launch cost is a large part of every measurement and
the bandwidth hides in small differences. Shapes sized for the A6000's
bandwidth are too small for ten times it; timed warm, all but the
largest also fit the B200's 126 MB L2 (not profiled). The check found a
constant not to trust.

## Reading the residuals

The third check is the residuals, the errors left after a fit.

![Predicted over measured for 27 GEMMs on the RTX A6000, grouped by
regime: on datasheet rates every ratio is below 1, lowest for the
shortest kernels; with the fitted constants the points gather around
1.](/assets/tinyperf-book/ch04-residuals.svg)

*Figure 4.2. Predicted ÷ measured for the A6000's 27 shapes. On
datasheet rates (orange) each error tracks one term's share of the time;
four fitted constants (blue), each set mostly by one regime, move them
to 1. The lowest orange decode-like points are the 4096 × 4096 weights,
where the missing launch cost weighs most.*

The rule, which Figure 4.2 illustrates:

- **An error that tracks one term's share of the time points at a
  constant.** It is uniform where that term dominates, and deepens as
  kernels shorten when the missing term is a fixed cost: 1024³, at
  44 µs, reads 0.51; 2048³ reads 0.84. A fit finds the constant.
- **A shape-dependent error the constants can't absorb points at a
  mechanism.** A fit can only trade it between shapes, and the constants
  that come out absorb the mistake.

Chapter 3's field note is the second kind. The earlier L2 model read the
A6000's largest square GEMM high (17 ms against 10.3 ms, chapter 3)
while it read the others low, so any constant that brought one toward 1
pushed the others away. The fix was a mechanism, judging L2 reuse one K
slice at a time.

```
Table 4.5  What the fit leaves: calibrated model/measured by regime
  regime                      RTX A6000   H100 SXM       B200
  square, 1024-8192           0.93-1.15  0.93-1.12  0.91-1.03
  decode-like, 1-128 rows     0.98-1.08  0.89-1.06  0.84-1.22
  tall, 2048 x 32768          1.02-1.03  1.01-1.09  1.04-1.07
  batched, attention-like     0.89-1.08  0.96-1.05  0.84-1.29
  tiny, 128 and 256           0.76-0.91       0.62       0.87
  all: typical error               4.9%       8.5%       9.1%
  H100 squares: 1024 0.93, 2048 1.00, 4096 1.12, 8192 1.10
```

In Table 4.5, on the A6000, every shape but the tiny two lands within
0.89–1.15. The tiny ones read 0.76 and 0.91, and Table 4.1 shows why:
the 128³ GEMM measures 34.3 µs, the larger 256³ one 29.7 µs. That is the
host's time, and it varies more than a GPU model can follow. On the H100
the squares read 0.93 at 1024 and 1.10–1.12 at the two largest sizes: an
error that changes with shape, so a mechanism, possibly Hopper-class
tile loading, which chapter 3 lists as a limit (not profiled). On the
B200 the decode-like and batched shapes spread from 0.84 to 1.29, where
launch cost dominates (Table 4.4).

## Clocks and power

Chapter 3's field note left the A6000's largest GEMM at 0.71 of its
measurement and put the gap down to the clocks this GPU sustains. The
worked example below starts from the datasheet peak: streaming
multiprocessors (SMs) × multiply-accumulates (MACs) per SM per clock ×
2 FLOPs per MAC × clock (chapter 2).

```
Worked example  The 8192^3 fp16 GEMM on the RTX A6000: clocks and power
  datasheet peak: 84 SMs x 512 MACs x 2 x 1.8 GHz = 154.8 TFLOPS
  calibration run (median of 30): 10.30 ms = 106.8 TFLOPS, 0.69 of peak
  power run (22.6 s of back-to-back GEMMs): 106.1 TFLOPS at 299 W; the card's limit is 300 W
  energy model at boost clock: demand 391 W; held to 300 W, 114.9 TFLOPS, 0.74 of peak
  of that demand, DRAM bytes already inside the fitted energy per FLOP: 28 W; without them 363 W, 0.80 of peak
  square and tall GEMMs from 4096 up: 0.69-0.75 of peak; fitted math efficiency 0.75
```

Run back to back for 22.6 seconds, the same GEMM drew 299 W against the
card's 300 W limit, and a GPU at its power limit lowers its clock (that
run recorded no usable clock, so this is inferred). tinyperf's energy
model (Appendix A) puts the GEMM's demand at the boost clock at 391 W;
held to 300 W, it would run at 0.74 of peak. That demand counts the
GEMM's DRAM traffic twice, about 28 W: the energy per FLOP was fitted
on this GEMM's whole draw, and the model adds its bytes again. Without
them the demand is 363 W and the capped GEMM runs at 0.80 of peak. The
energy model was fitted to that same run, so this is a consistency
check, not independent evidence, but either way it agrees with the fit:
most of the gap between peak and the A6000's 0.75 math efficiency is
the power limit.

So the constant belongs to this card at its power limit; the same GPU
at another limit would want another number. For a GPU nobody has
measured, the device file's `sustained_clock_fraction` sets a clock by
hand for PROJ what-ifs; calibrated devices leave it at 1.0.

## Launch cost belongs to the software stack

The events in `time_gemm` bracket one *eager* call: PyTorch running each
operation as Python reaches it. The GPU records the first event, then
waits while Python, PyTorch, cuBLAS and the driver issue the kernel.
Beside a big GEMM that wait is negligible; for a small one it *is* the
time. So the fitted 12–25 µs is mostly the cost of issuing a kernel
from eager PyTorch, not a property of the GPU.

Serving engines avoid most of it with a **CUDA graph**: a recorded
sequence of kernel launches that the GPU replays as one. The engine runs
its step once while CUDA records every kernel and its arguments, then
replays the recording at each step with a single launch (Figure 4.3).
The price is rigidity: shapes and memory addresses are fixed when the
graph is recorded, so an engine records one graph for each of a set of
batch sizes (chapter 15).

![Two timelines of five small kernels: eagerly, the host issues each
kernel and the GPU waits between them; from a CUDA graph, the host
issues one replay and the kernels run back to
back.](/assets/tinyperf-book/ch04-cuda-graph.svg)

*Figure 4.3. Eager launches against a CUDA graph. Eagerly, each small
kernel waits for the host; from a graph, the kernels run back to back.*

`tools/measure_launch.py` measures both costs. It builds a chain of 100
or 400 tiny dependent `addmm` kernels, 64 to 256 wide, so small that
launching is nearly all the work. It times the chain eagerly, then
records it once inside `torch.cuda.graph(g)` and times `g.replay()`
between the same two events, and divides by the count.

```
Table 4.6  Per-kernel cost of a chain of tiny addmm kernels on a B200: eager against a CUDA graph
  kernel     chain  eager us/kernel  graph us/kernel  eager/graph
  addmm64      100            13.86             2.78          5.0
  addmm64      400            13.16             3.17          4.2
  addmm128     100            13.52             3.52          3.8
  addmm128     400            13.22             3.48          3.8
  addmm256     100            17.51             3.74          4.7
  addmm256     400            16.56             3.71          4.5
  median                      13.69             3.50          3.9
  the B200's launch constant fitted from eager cuBLAS GEMMs (Table 4.2): 16 us; its graph constant: 3.5 us
```

From a graph, a kernel costs about what chapter 3's 3 µs default
assumed for the GPU's side alone: the host's work is gone. The eager
median also checks the fit: a different workload, timed a different
way, lands near Table 4.2's 16 µs.

A calibration carries both launch constants, and `for_stack` picks one
(docstring trimmed):

```python
    def for_stack(self, stack: str) -> "Calibration":
        assert stack in ("eager", "graph")
        if stack == "eager" or not self.kernel_launch_us_graph:
            return self
        return replace(self, kernel_launch_us=self.kernel_launch_us_graph,
                       source=self.source + " [graph stack]")
```

Only the launch constant changes: math and memory efficiency belong to
the silicon, the cost per kernel to how kernels are issued, so calibrate
against the stack you mean to predict. It matters most where kernels are
many and short. Table 4.7 prices a decode step from the model's list of
operations (chapter 5), at batch 1 and a context of 2,112 tokens (a
2,048-token prompt plus 64 generated):

```
Table 4.7  Launch cost in one decode step: Qwen3-8B at batch 1, context 2112, the model's list of operations
  GPU        kernels   weights     launch, eager     launch, graph  step, eager  step, graph
  RTX A6000      399    21.9ms   10.0 ms (25 us)   1.4 ms (3.5 us)       33.3ms       24.5ms
  B200           399     2.4ms    6.4 ms (16 us)   1.4 ms (3.5 us)        8.8ms        3.8ms
  kernels: 11 per layer x 36 layers (4 GEMMs, 1 attention, 6 norms and elementwise) + 3 (embed, ln_final, lm_head)
  weights: 15.1 GB of 16-bit GEMM weights, at each GPU's fitted DRAM rate
  on the RTX A6000, each 1 us per kernel moves the step by 0.40 ms (1.6% of the graph step)
```

Six of each layer's eleven kernels (norms, rotary embedding, activation,
residual adds) cost little more than their launch at batch 1. On the
A6000, eager launches add 10.0 ms to 21.9 ms of weight streaming. On a
B200 the balance tips, 6.4 ms of launches against 2.4 ms of weights: the
model predicts an eager decode step there is mostly the GPU waiting for
the host.

The graph constant was measured only on the B200; the A6000 and H100
files carry its 3.5 µs. A6000 decode steps priced with it match the
measurement (next section), but that bounds it only loosely:
1 µs per kernel is 1.6% of a step, the size of the model's other errors.

## The ladder on a whole model

Table 4.8 adds the eager rung to chapter 1's Table 1.2, which climbs
the tiers like a ladder on the same eight cells of Qwen3-8B served by
vLLM. Prefill sets the TTFT (time to first token), and a decode step
the TPOT (time per output token). Decode is priced at the prompt length
plus 64, the mean context over the 128 tokens each request generated
after the first. The cells are in-sample: a mechanism was designed with
them in view (below).

```
Table 4.8  The ladder on a whole model: Qwen3-8B bf16 on one RTX A6000, against vLLM (in-sample)
  milliseconds                      SOL       PROJ  CAL eager  CAL graph   measured
  prefill, 1 x 2048 tokens        212.3      239.8      300.6      292.0      291.3
  decode, batch 1                  20.1       21.7       33.3       24.5       24.3
  decode, batch 32                 33.2       35.4       48.8       40.2       42.0
  tier/measured, all 8 cells (batch 1-32 x prompt 512-8192):
  prefill (TTFT)              0.66-0.75  0.77-0.82  0.99-1.08  0.97-1.05
  decode (TPOT)               0.79-0.83  0.84-0.90  1.16-1.38  0.96-1.01
  CAL graph typical error: prefill 1.9%, decode 1.4%
```

SOL is a floor everywhere, at 0.66–0.83 of the measurement. The
calibrated tier on the right stack lands on it, within 0.96–1.05, a
typical error under 2%. On the wrong stack, its prefill is still fine,
because prefill kernels are long, but its decode reads 16–38% high,
because each of 399 short kernels pays 25 µs instead of 3.5. The right
mechanism on the wrong software stack is still wrong. Here the truth sat
11–30% above PROJ, the kind of gap a projected number can carry.

Nothing in the calibration was fitted to these measurements: the four
constants came from cuBLAS GEMMs, and the per-kernel tables it uses
here came from kernels timed alone. But this grid was in view while the
model was built: its batched prefills are where the model learned to
price attention per prompt in a batch, not over one concatenated
sequence (chapter 6). So the numbers are in-sample (chapter 21 draws
that line). The measured times also include work outside the model's
operations, such as the engine's sampler, which picks each next token;
chapters 14 and 15 add it.

## Beyond four constants

A served model runs kernels that do things no first-principles model
derives: cuBLAS's tile edges, a decode-attention kernel's fixed cost per
call, a mixture-of-experts kernel that switches block size, NCCL
(NVIDIA's library for exchanges between GPUs) changing protocol with
message size. For each, the calibrated tier stores a measurement.
Table 4.9's first row is the row curve, a measured correction to the
tile model by row count for the edges of chapter 3's Figure 3.4. Most
rows price mechanisms of later chapters, named in the chapter column:
MXFP4 is a 4-bit weight format (chapter 10), and the pair is two RTX
A6000s linked over PCIe (chapter 11).

```
Table 4.9  Beyond four constants: the other fields of the RTX A6000's calibration
  field                          what it prices                           chapter  how it was obtained
  dense_gemm_row_factor          cuBLAS time over the tile model, by rows   3, 15  kernel alone
  kernel_launch_us_graph         per-kernel cost under CUDA-graph replay        4  tiny-kernel chain, B200
  fmha_decode_floor_us           decode attention: cost per call                7  kernel alone
  fmha_decode_dram_efficiency    decode attention: KV streaming rate            7  kernel alone
  attention_backends             the same, for FlashInfer and Triton            7  kernel alone
  moe_dram_efficiency            expert kernel: weight streaming rate           9  kernel, in the engine
  moe_dram_efficiency_big_block  the same, past the block-size switch           9  kernel, in the engine
  weight_only_dram_efficiency    MXFP4 expert kernel: streaming rate            9  kernel, in the engine
  weight_only_expert_tflops      MXFP4 expert kernel: rate by tokens            9  kernel alone
  moe_math_efficiency            bf16 expert kernel: prefill math               9  fitted end to end
  weight_only_math_efficiency    MXFP4 expert GEMMs: math                  10, 22  fitted end to end
  collective_curve_2gpu          NCCL collectives on two GPUs                  11  kernel alone
  collective_curve_link_gbps     the link that curve was measured on           11  the pair's ring rate
  pp_step_overhead_us            pipeline-parallel engine: cost per step       12  fitted end to end
  per_seq_step_overhead_us       cost per sequence per step                    15  retired: 0
  mixed_decode_gqa_reread        decode rows' KV re-read in a mixed step       16  kernel alone
  mixed_step_fixed_us            mixed step: fixed cost                        16  retired: 0
  mixed_step_per_seq_us          mixed step: cost per sequence                 16  retired: 0
  online_overhead_us             a request's cost outside the steps            16  idle server
  serving_error_band             the serving model's validated error           18  held-out cells
  kv_producer_layer_us           KV producer: cost per layer                   20  fitted end to end
  weight_only_expert_mixed_us    MXFP4 expert kernel beside decodes            22  kernel alone
  28 fields in all: the device name, the four fitted constants, the provenance string and these 22
```

The rule: **each field is measured on its own kernel, timed alone or
picked out of real engine steps, in the engine's configuration, and
never fitted to the end-to-end numbers it will be checked against.** The
engine's configuration means its shapes, kernels and settings, usually
inside a CUDA graph: the row curve is Qwen3-8B's own GEMMs called as
vLLM calls them. Such a field can be wrong about its kernel, but it
can't quietly absorb another part of the model's error, because it
never saw that error.

Two fields are not kernel prices: a request's overhead on an idle
server, and the model's measured error band. Four fields break the rule,
and the table says so: they were fitted end to end, to an engine's step
times or TTFT, with some cells held out. The B200's file adds three
more, fitted on the prefill of a hybrid model, which mixes attention
layers with linear-attention ones (chapter 8). These are the fields to
distrust. A constant fitted to an end-to-end number absorbs everything
the model gets wrong about it, in every prediction it touches. Chapter
22 follows one, `weight_only_math_efficiency`, through the three
mechanisms it was hiding.

> **Field note: three zeros.** Three fields in Table 4.9 are zero. Each
> was once fitted to a serving engine's measurements and brought the
> model closer to them, first as 63 µs per running sequence per step.
> When pure decode steps turned out to need nothing, that became 3.3 ms
> plus 213 µs per sequence for steps carrying part of a prompt. Measured
> one kernel at a time, both costs were mechanisms: decode rows
> re-reading their KV cache once per query head beside a prompt chunk
> (chapter 16), and mostly cuBLAS's tile edge just past 1,024 rows
> (chapter 3's row curve). Priced as mechanisms, the constants went to
> zero; they stay in the file as a record of what a good fit can hide.

## Where it breaks

- **One library, one precision, one way of calling it.** The constants
  come from fp16 cuBLAS GEMMs called from eager PyTorch. FP8, FP4, other
  libraries and fused kernels get the same efficiencies, unmeasured.
- **Constants the data barely pins:** the L2 bandwidth everywhere, and
  the B200's DRAM efficiency, whose shapes are too small for it (0.80
  fitted, 0.97 from the line).
- **One number, several causes.** Math efficiency folds the power limit
  and whatever the tile menu misses on newer architectures into one
  factor.
- **Launches in series.** The model adds a launch cost to every kernel
  in turn. The eager constant was fitted with the GPU idle, waiting on
  each call; a real eager host can run ahead of long kernels and hide
  its issue cost (not measured).
- **End-to-end fits.** Those fields have far less data behind them than
  the four, and are the first suspects when a measurement disagrees. So
  is the graph launch cost, measured on one GPU.

## What you built

- Three tiers from one model, reported side by side; CAL refuses to run
  without a calibration.
- A measurement of 27 cuBLAS GEMMs in five regimes and a fit of four
  physical constants by coordinate descent on the mean `|log error|`.
- Evidence: typical error from 48.5–108.8% to 4.9–9.1% (fitted),
  5.3–12.3% on held-out halves, and two constants reproduced by a
  straight line on two GPUs of three.
- The software stack as a calibration axis: 12–25 µs per kernel issued
  eagerly, 3.5 µs from a CUDA graph.
- Qwen3-8B on vLLM within 0.96–1.05 on the graph stack (in-sample), and
  decode 16–38% high on the eager one.
- Measured per-kernel tables, and the rule that keeps them honest.

## Exercises

1. **Calibrate your GPU.** Run `tools/measure_gemm.py` and
   `tools/fit_calibration.py` on a GPU you have, in a copy of the
   repository: the fit rewrites `data/calibration/<gpu>.json`, keeping
   only the graph launch cost and the attention efficiency, so here it
   would erase every per-kernel table. Which constants land near Table
   4.2's?
2. **Is 3.5 µs a GPU number?** Run `tools/measure_launch.py` on a second
   GPU generation. Does graph replay cost the same?
3. **Scale the shapes.** Add decode-like GEMMs whose weights take 100 µs
   or more to stream on a B200, too big for its L2, and refit. Does the
   DRAM efficiency move toward the line's 0.97?
4. **Leave a regime out.** Fit without the tiny GEMMs and predict them;
   then without the decode-like ones. Which constant does each regime
   pin, and how wrong is the fit on the regime it never saw?
5. **Separate the power limit.** Price each A6000 GEMM with the energy
   model's power-capped time and refit. How far does the math efficiency
   rise toward 1, and what is left?

---

*[← Chapter 3: Pricing a GEMM]({% post_url 2026-09-29-tinyperf-03-pricing-a-gemm %}) · [Contents](/series/tinyperf/) · [Chapter 5: Graphs and the scheduler →]({% post_url 2026-09-29-tinyperf-05-graphs-and-the-scheduler %})*
