---
layout: post
title: "Building tinyperf, chapter 21: Validating a performance model"
date: 2026-09-29 12:21:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/21-validating-a-performance-model/
excerpt: "tinyperf has a knob for almost everything: an efficiency per kernel class, a fixed cost per call, a measured table wherever a library does something no formula derives. Show it a measurement it gets wrong and some knob will fix it. An analytical model can be made to match any measurement after the fact. So when twenty chapters report agreement within a few percent, how do you know the model predicts, and how do you keep yourself honest while building it?"
redirect_from:
  - /tinyperf/perf-modeling/2026/08/31/building-tinyperf-m30.html
---

*[Building tinyperf](/series/tinyperf/) · Part V: Knowing it's right · Code: [`tests/test_core.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tests/test_core.py), `test_silicon_envelope`, and [`tinyperf/serving.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/serving.py), `prediction_interval` · Every table and the plot in this chapter come from `python3 book/scripts/ch21_validation.py`.*

tinyperf has a knob for almost everything: an efficiency per kernel
class, a fixed cost per call, a measured table wherever a library does
something no formula derives. Show it a measurement it gets wrong and
some knob will fix it. An analytical model can be made to match any
measurement after the fact. So when twenty chapters report agreement
within a few percent, how do you know the model predicts, and how do you
keep yourself honest while building it?

The short answer: treat every change to the model as an experiment, and
keep a record you can't edit. Write the prediction and its pass/fail
criteria down, and commit them, before the measurement runs. Label every
number by what the model had seen. Fit few constants and read what they
leave. Check the measurement as hard as the model: run controls, measure
the way the engine runs, and watch the machine. Then turn each validated
grid into a test. None of this proves the model right. It makes it hard
to fool yourself about how wrong it is.

By the end of this chapter you will know:

- why agreement after the fact is cheap, and what a prediction
  committed before the measurement buys;
- the labels every chapter uses, and how a number moves between them;
- how a residual points at a constant or at a mechanism;
- the controls, instruments and machine checks that caught wrong
  measurements in this project;
- how errors cancel, and what to do when a correct fix makes things
  worse;
- how tests pin error bars, and why near saturation a prediction is an
  interval;
- what the repository's record holds, and where the method stops.

## Agreement after the fact is cheap

Give a model k free constants and it can match k measurements exactly,
whatever its mechanism. Only the rest test it. Roughly:

```
evidence ≈ measurements − constants fitted to them − measurements a mechanism was designed around
```

The constants are easy to count. Chapter 4 fitted four per GPU to 27
GEMMs; chapter 8 fitted three prefill constants on three cells of five,
leaving two to test them. The last term is the hard one. A mechanism
designed after seeing a grid fits it as well as a constant does, and
shows up in no count: chapter 6's per-prompt attention was found by
looking at Table 1.1's grid, which is why chapter 1 calls that grid
in-sample. And constants can fit well and be wrong: chapter 4's field
note "three zeros" tells of three that turned out to be mechanisms the
model lacked.

The defense is a loop that records what the model said before each
measurement (Figure 21.1).

![A loop of six boxes: predict, freeze, measure, compare, diagnose the
residual as a constant or a mechanism, and pin the grid as a test, with
a path from compare straight to pin when the cells pass, and from pin
back to predict for the next question.](/assets/tinyperf-book/ch21-loop.svg)

*Figure 21.1. The validation loop. The prediction is written and
committed before the run (1, 2); the run is measured the way the engine
runs, with the machine logged and a control beside it (3); a miss is
diagnosed as a constant or a mechanism (4, 5); and every grid, passed or
fixed, becomes a test (6). The footer gives the labels.*

## Freeze the prediction

A prediction is **frozen** when its numbers and pass/fail criteria are
committed before the measurement exists and never edited afterwards; a
revision is a new file that says what it has seen. Here the numbers go
into a text file in `data/validation`, named
`predictions_*_before_measurement.txt`, committed and pushed before the
run: a commit's date can be set by hand, a push to a shared repository
is harder to fake. A frozen file holds:

1. the configuration, down to the engine's version and flags;
2. the model's numbers, cell by cell, with a band where small errors
   grow;
3. the pass/fail criteria;
4. the residuals you expect, stated before anyone has seen one;
5. the old model or an alternative, priced on the same cells.

Table 21.1 shows parts of one, for Qwen3-8B at tp=2 (each GPU holds half
of every weight matrix; chapter 11) serving a stream of requests. In it,
"curve" is the new model, which prices the collectives from the GPU
pair's measured curve, and "ring" the old one, which used the ring
formula (chapter 11); "topology NODE" means the two GPUs talk through
the host's PCIe bridges. The band is the same simulation with every step
1% faster and slower: near the *knee*, where a server tips into
queueing, a small step error moves latency a lot (chapter 18). A step's
"recorded composition" is what the engine's step log says it carried
(chapter 15).

```
Table 21.1  A frozen prediction file, excerpted: Qwen3-8B at tp=2 served online, prompts of 129-384 tokens, replies of 257-758
  data/validation/predictions_qwen3_8b_rtx_a6000_tp2_serving_before_measurement.txt
  Qwen3-8B (bf16), vLLM 0.15.1, TENSOR PARALLEL 2 on two RTX A6000s (PCIe host bridge, topology NODE), --disable-custom-all-reduce,
  ...
  EXPECTED, stated now: prefill-carrying steps price ~1.5% high with the curve, so saturated cells may read a few % HIGH on TTFT.
  ...
  Each cell: TTFT p50 and p95 (ms), mean TPOT (ms), output tok/s. band = the curve model's TTFT p50 with every step 1% faster / slower.
  Criteria: (1) where the band spans less than +-15%: TTFT p50 and p95 within 0.85-1.15, TPOT within 0.95-1.05; (2) where it spans
  more (the knee): the measured TTFT p50 inside the band; (3) throughput within 3%; (4) decode steps and prefill-carrying steps at
  their recorded composition within 0.95-1.05 at the median.
  ...
     rate | curve: TTFT p50 [band -1%, +1%]       p95   TPOT  tok/s || ring: TTFT p50      p95   TPOT  tok/s
        1 |             135 [    135,     139]       177   21.6    500 ||            130      167   19.9    501
        2 |             155 [    151,     157]       233   30.2    941 ||            146      222   27.2    949
        3 |             227 [    220,     229]      4918   40.9   1204 ||            198     3315   38.5   1246
        4 |            6221 [   6041,    6599]     16420   42.1   1250 ||           5138    14127   40.2   1301
  measured, frozen prediction over measured (chapter 11's Table 11.8 has the rows):
  TTFT p50 0.94-1.09, p95 0.91-1.10, TPOT 0.99, throughput 1.00-1.02; the old (ring) model's TPOT 0.89-0.94
  steps at the composition the engine's step log recorded: decode 0.976-0.993, prefill-carrying 1.008-1.023
  within the criteria: this sweep 4 of 4 cells; the file's other sweep 3 of 4, its miss TPOT 1.054 against 1.05; together 7 of 8
  data/validation holds 37 files of predictions written before their measurements, and 2 revised mid-campaign that say what they had seen
```

The file's opening claims and its other sweep are trimmed. This sweep
met all four criteria in all four cells, and the old model read TPOT
0.89–0.94 on the same arrivals. The file's other sweep missed one cell,
TPOT 1.054 against 1.05: 7 of 8. The expected residual showed in the
steps, prefill-carrying ones at 1.008–1.023, though not in this sweep's
TTFT. Either way it was written first, so it couldn't be invented
afterwards.

What does freezing protect against?

- **Tuning on the test.** A model adjusted after seeing the data agrees
  with it; the file says what the model said before.
- **Moving the goalposts.** The criteria are written down, so a miss
  stays a miss: chapter 8's hybrid prefill came back 27–58% low, and
  chapter 1's batched prefills up to 2.79 high (Table 21.5).
- **A flattering baseline.** The old model is priced on the same cells,
  in the same file.
- **Quiet selection.** Every cell is listed; the bad one can't be
  dropped.

A revision shows what "says what it has seen" means: one opens with
"STATUS OF MEASUREMENTS AT THIS MOMENT", marks one run "EXISTS (seen)"
and another "EXISTS (NOT opened)", and calls itself post hoc for the
first and held out for the rest
(`predictions_qwen3_8b_rtx_a6000_pp2_v2_post_serial_pre_steady.txt`).

## What kind of number is it?

Three labels say what the model had seen (chapter 1):

- *held out*: nothing in the model has seen it. Qwen3-30B-A3B's decode
  steps (chapter 9) took their routing from a separate routing-only run
  and their kernel constants from another model, and read 0.935–1.073
  (Table 21.4).
- *fitted*: a constant came from it. Chapter 4's four constants leave a
  typical error of 4.9–9.1% on the GEMMs that set them.
- *in-sample*: a mechanism was built with it in view, as in Table 1.1's
  batched prefills.

*Recorded* says the number was stored when measured and not recomputed
by the current model, and it combines with the others: Table 1.5 is
recorded and held out. A *projection* has no measurement at all.

Held out along what? A number can be held out along one axis and not
another, and a new seed tests less than a new shape, a new model or a
new GPU. **Seeds**: chapter 14's sweeps repeat a workload on new random
arrivals, which tests the queue more than the step prices. **Shapes**:
chapter 15's row curve cut the error on tp=2 GEMM shapes it never saw
from 7.0% to 4.2%; chapter 4's alternate halves are held out from the
fit but not from the mechanism, since chapter 3's L2 rule came from one
of those shapes' residuals. **Models**: chapter 17's KV-pool accounting,
fitted on Qwen3-8B, predicted eight held-out dense configurations,
mostly of other models, within 0.9999–1.0097; its two MoE configurations
missed, at 1.020 and 1.055, until a term for the expert kernel's
workspace was added, and stay pinned as misses (Table 21.4). **GPUs**:
chapter 8's hybrid decode on a B200, nothing fitted to it, read
0.88–0.96.

A label belongs to a number and to a version of the model, and it moves
one way. The first time the model is changed with a grid in view, the
grid is in-sample for good. Held-out data is spent by looking at it,
which is why the record keeps growing: each new claim needs data nobody
has seen yet.

## Fit few constants, and read what they leave

Chapter 4 gave the rule for the *residuals*, the errors a fit leaves:

- an error that tracks one term's share of the time points at a
  **constant**;
- an error that changes with the shape, which no constant can absorb,
  points at a **mechanism**.

Three cases from the record show it working, and a fourth its limit.

- **A constant.** Chapter 4's GEMMs on datasheet rates read low
  everywhere, lowest for the shortest kernels: a missing fixed cost per
  call.
- **A mechanism.** Chapter 6's batched prefills were exact for one
  prompt, worse with each prompt added, worst where attention's share
  was largest. No constant does that: the model had priced a batch of
  prompts as one long prompt.
- **Constants, labeled.** The hybrid on the B200 read low on every
  prefill cell, the missing time growing with the tokens (chapter 8).
  With no mechanism to fix, three constants were fitted on three cells,
  checked on two, and labeled fitted; the floor's cause was never
  profiled.
- **The limit.** The first tp=2 decode predictions read up to 1.42,
  worst at batch 1, whose step is mostly tiny all-reduces. The residual
  pointed at the right term, a per-message latency, but that constant's
  value came from an instrument that measured something else (below).
  The rule finds the term; it can't vouch for the measurement.

When you do fit, hold out part of the data; fit each constant where its
term dominates; report a constant that lands on the edge of the range
its fit searched (Table 4.3's asterisks); and never fit a constant to
the end-to-end number it will be checked against. Chapter 4's Table 4.9
marks the few fields that break that rule: the first suspects when a
measurement disagrees.

## Controls

A *control* is the same experiment with one thing changed on purpose,
run beside it, whose result you can predict.

On the model's side, the control is the model you are replacing,
priced on the same trace: Table 21.1's ring column, or chapter 9's
uniform routing, which reads 1.275–1.435 on the MoE decode steps where
the measured router's table reads 0.935–1.073 (Table 21.4). A new
mechanism has to beat the old one on the same data. Write the control's
prediction down too, so a surprising control is a finding: chapter 20
turns one engine setting off on a disaggregated server and writes the
control's answer times down from the model before reading its data
(Table 20.4).

On the measurement's side, a control is the same cells measured again.

```
Table 21.2  Controls on the measurement: the same cells measured twice
  Qwen3-8B, one RTX A6000: three cells of Table 1.1 re-measured on the day of the two-GPU runs (re-run over first run)
  batch  prompt   TTFT   TPOT
      1     512   1.02   1.00
      8    8192   1.00   1.01
     32    2048   1.00   1.00
  Qwen3.8-27B, one B200: the five cells run in forward order, then in reverse (forward over reverse)
  batch  prompt   TTFT   TPOT
      1     512   1.04   1.03
      1    2048   1.04   1.04
      1    8192   1.02   1.04
      8    2048   1.01   1.04
     32    2048   1.01   1.00
  largest difference between the two orders: 4.2% of their midpoint, which chapter 8's table uses
```

Three cells of Table 1.1, re-measured on the day of the two-GPU runs,
agree within 2%, so the comparisons of two GPUs with one (chapters 11
and 12) are not drift. On the B200 the forward run was slower on nine of
ten numbers, by up to 4.2%, for a reason not profiled. That is the
measurement's resolution: a model difference of 3% on that GPU is not a
finding.

## Instruments lie

Every measurement is an instrument, and an instrument can measure
something other than what you meant. Table 21.3 collects those that did,
each against the way the engine runs.

```
Table 21.3  Instruments against the engine's way of running, from the record
  what was measured                               the instrument                       read   the way the engine runs         read
  NCCL all-reduce of 8 KB, two RTX A6000s         a Python loop of calls            87.1 us   replayed in a CUDA graph     16.5 us
  NCCL all-reduce of 64 KB, two RTX A6000s        a Python loop of calls            95.6 us   replayed in a CUDA graph     50.0 us
  NCCL all-reduce of 256 KB, two RTX A6000s       a Python loop of calls            95.1 us   replayed in a CUDA graph     97.3 us
  experts touched, gpt-oss-20b, 8 random prompts  at the last prompt position          9.65   over the 128 decode steps      13.10
  a step of 8 decodes and a 256-token chunk       the client, largest token gap     51.5 ms   the engine clock             60.1 ms
  gpt-oss-20b experts, batch 8, replayed alone    under the torch profiler          19.7 ms   no profiler                  16.1 ms
  the client's largest gap over the engine clock, 20 mixed-step cells: 0.83-1.01, median 0.91 (chapter 16)
  the torch profiler inside the engine: about nothing at batch 8, 7-15% long from batch 16 (recorded, not re-measured)
```

- **A host loop measured the interpreter.** 87–96 µs from 8 KB to
  256 KB: a time that doesn't move over a 32-fold range of sizes is the
  host issuing calls. It made the frozen tp=2 decode predictions up to
  42% high (chapter 11).
- **Counting at the wrong position.** See chapter 9's field note
  "count over the steps you time", and chapter 22.
- **The client's clock.** A client's largest gap reads low against the
  engine's clock, consistent with part of a long step's delay landing in
  a neighbouring gap under asynchronous scheduling (chapter 16; not
  profiled). Step prices are checked on the engine's clock (chapter 15),
  TPOT against the whole simulator.
- **A profiler.** At batch 8 it inflated a kernel replayed alone by 22%
  and the same kernel in the engine by about nothing; constants from
  profiled kernels would carry the difference.
- **A cache.** Identical prompts hit the engine's prefix cache, and
  TTFT stayed flat across a 128-fold range of prompt lengths (chapter
  1).

Measure the way the engine runs, with its graphs, its routing and its
clock. And distrust a reading that doesn't move when it should.

> **Field note: measured right, concluded wrong.** gpt-oss-20b's expert
> kernel once ran about 1.4 times slower inside the engine's decode step
> than replayed alone. Two profilers and the GPU's counters agreed that
> both ran identical launches at the same clocks, at the memory roof.
> The replay ruled out, one by one, neighbouring kernels and cache
> flushes, a thermal soak, busy host threads, the process's history, the
> L2's reserved set-aside and every launch argument. Every measurement
> stood, and the conclusion drawn from them was wrong; chapter 22 says
> why.

## Errors that cancel

A model with several errors can agree with a measurement because two of
them cancel. You find out when a correct fix makes validated numbers
worse.

Chapter 15's field note is the clearest case: a fitted 63 µs per
sequence per step was the top-p sampler, right in size and wrong in
name, and once zeroed its absence hid behind two other errors. Chapter
7's field note is the same cancellation seen from the other side:
pricing the mean context alone moved seven validated results at once.
Chapter 22 follows a fitted constant that stood in for three
mechanisms, and a correct fix that removed a cancellation and made a
sweep's agreement worse.

So when a correct change makes things worse, don't revert it; find the
error it uncovered. And price each piece from its own kernel before
changing the whole: a sum that agrees says nothing about its parts.

## Watch the machine

A GPU is not a constant. Chapter 4's A6000 ran its largest GEMM at 299 W
against a 300 W limit, and a GPU at its power limit lowers its clock
(chapter 2). Other processes share it. So the later online runs log the
GPU's clocks, power and temperature every second, every process on each
GPU every 2 seconds, and the engine's own step timer (Table 21.6 counts
the logs).

The model is an instrument too. Its pure decode steps land within a few
percent almost everywhere, so a stretch where they don't says the GPU
changed, not the model (Figure 21.2).

![Model over measured for pure decode steps in 30-second windows of two
runs of the same sweep: both near 1 throughout, except three windows of
the first run after 720 seconds that fall to about
0.5.](/assets/tinyperf-book/ch21-windows.svg)

*Figure 21.2. Two runs of the same long-prompt sweep of Qwen3-8B on one
RTX A6000: the pure decode steps' model/measured per 30-second window,
from the engine's step log. Every window reads 0.99–1.05 except
three in the first run, at 0.47–0.52, while another job ran on the
neighbouring GPU. The first run was set aside; the re-run logged the
GPU's clocks.*

A slowed GPU can also pass for a scheduling effect: chapter 18's knee
sweep looked bistable until a re-run with the step clock showed the
first run's TPOT 30–67% longer (Table 18.8). And steady clocks don't
clear a GPU: steps that run long in bursts while the clocks hold point
at another process, which is why every process is now logged.

## Pinning the error bars

The last step of the loop turns a validated grid into a test. Here is
chapter 1's, `test_silicon_envelope`, with its docstring and imports
trimmed:

```python
def test_silicon_envelope():
    meas = json.loads((_P(__file__).resolve().parent.parent /
                       "data/validation/vllm_qwen3_8b_rtx_a6000.json").read_text())
    dev = Device.load("rtx_a6000")
    m = StepLatencyModel(qwen3_8b(), dev, methodology=Methodology.CALIBRATED)
    for r in meas["serving"]:
        b, s = r["batch"], r["prompt"]
        ttft = m.prefill_us(b * s, n_seqs=b) / 1e3
        tpot = m.decode_us(b, s + 64) / 1e3
        assert 0.92 < ttft / r["ttft_ms"] < 1.10, ("ttft", b, s, ttft, r["ttft_ms"])
        assert 0.93 < tpot / r["tpot_ms"] < 1.07, ("tpot", b, s, tpot, r["tpot_ms"])
```

It prices chapter 1's eight cells as chapter 1 does and asserts every
ratio inside a band. The margin is the design choice: today TTFT reads
0.967–1.049 against 0.92–1.10, and TPOT 0.958–1.013 against 0.93–1.07
(Table 21.4). That leaves 3–6% of headroom: enough that an unrelated
fix doesn't force the bound open, but a smaller regression passes this
test and has to be caught by a tighter one, or not at all. A change
that moves a bound says why.

Tests pin more than agreement.

- **Directions.** The tp=2 test also asserts that on this link tp=2
  prefill is slower than one GPU and decode faster, in the model and in
  the measurement:

  ```python
          # the direction of the tp=2 trade-off, model and silicon agree
          assert ttft > m1.prefill_us(b * s, n_seqs=b) / 1e3 and r["ttft_ms"] > one[(b, s)]["ttft_ms"]
          assert tpot < m1.decode_us(b, s + 64) / 1e3 and r["tpot_ms"] < one[(b, s)]["tpot_ms"]
  ```
- **Controls.** The MoE test asserts that uniform routing stays above
  1.25.
- **Known misses.** A prefix-cached prompt with a 128-token suffix
  reads 0.710, and its test asserts 0.6–0.75. A change that fixes it
  fails the test too, and has to claim the fix.

Table 21.4 re-prices a curated set of pinned grids with each test's own
set-up, beside the bound the test asserts, read from the test's source.
Its labels are those each chapter gives, for the model that first priced
the grid.

```
Table 21.4  Grids pinned by tests/test_core.py: the current model against the record, and the bound each test asserts
  grid and quantity                                   cells  label                  model/measured        bound
  test_silicon_envelope
    Qwen3-8B, one RTX A6000: TTFT, batch 1                3  held out                  0.967-1.049    0.92-1.10
    the same: TTFT, batch 8 and 32                        5  in-sample                 0.978-1.029    0.92-1.10
    the same: TPOT                                        8  held out                  0.958-1.013    0.93-1.07
  test_tensor_parallel_silicon_envelope
    Qwen3-8B, tp=2 on two RTX A6000s: TTFT                8  held out                  1.002-1.073    0.95-1.08
    the same: TPOT                                        8  in-sample                 0.961-1.011    0.89-1.10
  test_pipeline_silicon_envelope
    Qwen3-8B, pp=2 on two RTX A6000s: TTFT                8  in-sample                 0.913-1.062    0.90-1.07
    the same: TPOT, 4 cells the step cost came from       4  fitted                    0.983-1.018    0.94-1.03
    the same: TPOT, the other 4                           4  held out from the fit     0.957-1.018    0.94-1.03
  test_hybrid_silicon_envelope
    Qwen3.8-27B, one B200, forward run: TTFT, 3 cells     3  fitted                    0.967-0.992    0.90-1.08
    the same: TTFT, the other 2                           2  held out                  0.937-0.995    0.90-1.08
    the same: TPOT                                        5  held out                  0.880-0.940    0.85-1.10
  test_moe_online_on_silicon
    Qwen3-30B-A3B decode steps, tp=2: batch 1             1  held out                        1.026    0.95-1.05
    the same, batch 8-64                                  5  held out                  0.935-1.073    0.90-1.10
    the same, uniform routing (the control)               5  held out                  1.275-1.435   above 1.25
  test_sampler_on_silicon
    Qwen3-8B decode steps, four samplers: batch 1-32     12  held out                  0.950-1.010    0.94-1.05
    the same: batch 48 and 64                             8  in-sample                 0.957-1.007    0.94-1.05
  test_kv_pool_on_silicon
    KV pool at start, dense models and layouts            8  held out                0.9999-1.0097  0.999-1.010
    the same, two MoE layouts as frozen: pinned misses    2  held out                  1.020-1.055    1.02-1.06
  test_prefix_cache_silicon_envelope
    prefix-cached TTFT, suffix of 512 tokens or more      6  held out                  0.977-1.104    0.94-1.12
    128-token suffix at batch 1: a pinned miss            1  held out                        0.710     0.6-0.75
  labels as each chapter gives them, for the model that first priced the grid; each grid has been a test since, in view of every later change
  tests/test_core.py: 92 tests, 31 of them read measurements from data/validation
```

The tests also keep the repository's scope statement true. And pinning
has a price: every pinned grid is in view of every later change, so it
is in-sample for all of them. Only frozen files bring in new held-out
data.

## Intervals near saturation

Near a server's knee, a 1–2% error in step time moves TTFT by tens of
percent, because the queue amplifies it. A point prediction there can't
be trusted or pinned, so `prediction_interval` reruns the simulation
across the model's validated step error and quotes a range: narrow away
from a knee, wide at one. It covers the model's typical step error, not
a mechanism it lacks, so a cell outside it is a finding (chapter 18,
Table 18.7).

## How good is the record?

Table 21.5 puts four grids' frozen predictions beside the current model.

```
Table 21.5  What the frozen predictions said, and what the model says now: model/measured
  grid                                  quantity  cells     frozen        now  label                    what changed
  Qwen3-8B, one RTX A6000               TTFT          8  0.95-2.79  0.97-1.05  3 held out, 5 in-sample  a batch's prompts priced one by one (ch. 6)
                                        TPOT          8  0.97-1.01  0.96-1.01  held out
  Qwen3-8B, tp=2 on two RTX A6000s      TTFT          8  1.01-1.05  1.00-1.07  held out
                                        TPOT          8  1.06-1.42  0.96-1.01  in-sample                the link re-measured in a CUDA graph (ch. 11)
  Qwen3.8-27B, one B200, midpoint of    TTFT          5  0.42-0.73  0.95-1.01  3 fitted, 2 held out     three prefill constants (ch. 8)
    both orders, as chapter 8           TPOT          5  0.88-0.96  0.88-0.96  held out                 nothing (within 0.5%)
  Qwen3-30B-A3B decode steps, tp=2      step          6  0.93-1.07  0.93-1.07  held out                 nothing
  the Qwen3-30B-A3B file's other section, online at tp=2: 2 of 4 cells within its criteria (recorded)
  labels as each chapter gives them, for the model that first priced the grid
```

The frozen column is the honest one, misses included: the MoE file's
online section met its criteria in 2 of 4 cells. The "now" column is
what the book reports.

```
Table 21.6  The validation record: files in data/validation
  predictions written before their measurements          37
  comparisons of a run with its predictions              30
  engine runs: offline grids, online sweeps, steps       77
  the engine's own step logs                             15
  GPU telemetry logs (clocks, power, temperature)        19
  GPU process logs                                       13
  arrival traces the load generator sent                 14
```

## Where it breaks

- **One GPU type, mostly.** Nearly every end-to-end measurement is one
  or two RTX A6000s over PCIe. The B200 has five cells of one hybrid
  model; the H100 only kernel calibrations. No NVLink pair, and no
  tensor parallelism beyond 2.
- **Synthetic prompts.** Most online sweeps send random tokens, and
  MoE routing depends on the text (chapter 22).
- **One engine version.** vLLM 0.15.1 (0.22 on the B200), with its
  kernels, sampler and graph sizes. The tests check the model against
  recorded measurements; if the engine changes, nothing notices until
  someone measures again.
- **Few repeats.** Most cells come from one run. The spread between
  runs is known only where one was repeated, as in Table 21.2, so the
  smallest error the record can resolve is known on few grids.
- **The same hands.** The same people predict, set the criteria and
  measure. Freezing stops edits after the fact, not generous criteria: a
  ±15% band is easy to meet away from the knee.
- **Numbers, not code.** A frozen file names its commit; re-deriving a
  recorded number needs that commit.

## What you built

- A loop: predict, freeze, measure, compare, diagnose, pin.
- The labels, and the rule that held-out data is spent by looking at
  it.
- Residuals read as constants or mechanisms; controls on the model and
  on the measurement; instruments checked against the way the engine
  runs; the machine logged.
- Tests that pin agreement, directions, controls and known misses, and
  intervals where a point can't be trusted.
- Evidence: 37 frozen predictions and 31 tests reading measurements; on
  four grids, frozen predictions from 0.42 to 2.79 and the current model
  within 0.88–1.07.

## Exercises

1. **Freeze one.** Pick a cell you can measure. Write the calibrated
   tier's numbers, the criteria and one expected residual into a file,
   commit it, then run `tools/measure_vllm.py`. Write the comparison
   without editing the file.
2. **Measure your noise.** Run three cells of Table 1.1 in forward and
   reverse order, twice. What is the smallest model error you could
   claim to see?
3. **Tighten a bound.** Find the tightest bounds `test_silicon_envelope`
   passes today. Then, in a copy of the repository, change the A6000's
   `dram_efficiency` by 2%. Which of Table 21.4's tests fail?
4. **Knock out a mechanism.** Set `dense_gemm_row_factor` or
   `mixed_decode_gqa_reread` to None in a model's calibration
   (`dataclasses.replace`, as chapter 16's script does) and re-price
   Table 21.4's grids. Which move, which way, and does any move toward
   1?
5. **Check an instrument.** Time an 8 KB all-reduce between two GPUs
   you have, in a Python loop and with `tools/measure_nccl.py --mode
   graph`. How far apart are they?

---

*[← Chapter 20: Disaggregated prefill and decode]({% post_url 2026-09-29-tinyperf-20-disaggregated-serving %}) · [Contents](/series/tinyperf/) · [Chapter 22: Case study: one constant, three hidden mechanisms →]({% post_url 2026-09-29-tinyperf-22-case-study %})*
