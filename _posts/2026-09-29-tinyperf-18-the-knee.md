---
layout: post
title: "Building tinyperf, chapter 18: The knee"
date: 2026-09-29 12:18:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/18-the-knee/
excerpt: "Chapter 14's load curve is flat, then it isn't. Serving Qwen3-8B on one RTX A6000, the median time to first token (TTFT) is 0.44 s at 3.5 requests per second and 1.15 s at 4. That bend is the knee. Near it, three things go wrong for anyone predicting latency: the answer depends on exactly when the requests arrived, a step price 2% off moves the tail by tens of percent, and a single number stops being an honest prediction. Why, and how should a model report what it predicts there?"
redirect_from:
  - /tinyperf/perf-modeling/2026/09/27/building-tinyperf-m79.html
---

*[Building tinyperf](/series/tinyperf/) · Part IV: Serving · Code: [`tinyperf/serving.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/serving.py), `prediction_interval`, `simulate(step_scale=)` and `bench_requests`, and `Calibration.serving_error_band` in [`tinyperf/methodology.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/methodology.py) · Every table and both plots in this chapter come from `python3 book/scripts/ch18_knee.py`.*

Chapter 14's load curve is flat, then it isn't. Serving Qwen3-8B on one
RTX A6000, the median time to first token (TTFT) is 0.44 s at 3.5
requests per second and 1.15 s at 4. That bend is the *knee*. Near it,
three things go wrong for anyone predicting latency: the answer depends
on exactly when the requests arrived, a step price 2% off moves the tail
by tens of percent, and a single number stops being an honest
prediction. Why, and how should a model report what it predicts there?

The short answer: the knee is the rate at which the requests a server
holds fill its seats (the most requests the engine runs at once). Past
it a queue forms, whose growth is the difference between arrivals and
completions: two nearly equal numbers, so a 2% error in one is a large
error in their difference. The model therefore replays the arrivals the
benchmark actually sent, and it reports an interval: the simulator run
across the model's measured step error, widened by what that error
doesn't explain. On two held-out sweeps planned around their knees, 35
of 36 measured values landed inside intervals committed before the runs.

By the end of this chapter you will know:

- where the knee is, from three numbers per step and Little's law;
- why the tail is so sensitive there, in the simulator and on a server;
- why a prediction must use the benchmark's own arrivals, and what one
  simulation of random arrivals is worth;
- how tinyperf turns its measured step error into a calibrated
  prediction interval;
- how the intervals did on held-out runs, and what they can't cover.

## Where the knee is

Price a serving step (chapters 15 and 16) in three parts:

- **a**, a step with one sequence in it: mostly the weight read, which
  every step pays once for everyone;
- **b**, what each further decoding sequence adds: its cache read, its
  sampler row;
- **c_p**, what each prompt token riding in the step adds (priced as a
  whole prompt beside 32 decodes).

A request brings P prompt tokens and needs G more decode steps after
its first token. *Little's law* says the number of requests in a system
is the arrival rate times the time each spends in it; a request spends
G steps decoding, so at arrival rate λ and step time t, B = λ·G·t
requests decode at once. And the step pays for everyone in it:
t = a + b·B + λ·t·P·c_p, the last term being the prompt tokens that
arrive per step. Solving the pair:

```
rho = λ · (P·c_p + G·b)       the share of GPU time spent on requests' own work
t   = a / (1 − rho)           the step stretches with the load
B   = λ · G · t               the requests decoding at once (Little's law)
knee: B reaches the seats S   λ_knee = S / (G·a + S·(P·c_p + G·b))
```

These four lines are the *envelope*, a steady-state model of the server.
The weight read a costs no capacity: every step pays it whatever the
batch. Every other microsecond is a request's own work, and ρ is its
share of the time. As ρ grows the step stretches, each request stays
longer, and more are in flight. An engine caps that number: vLLM's
`max_num_seqs`, 64 in every run here (`VLLM.with_(max_num_seqs=64)`,
chapter 14), which we call its *seats*. When B
reaches the seats, a new request waits for one. That is the knee.

Table 18.1 takes Table 1.4's workload, 1,024-token prompts and
128-token replies, derives the constants from the step model, and puts
the envelope beside the simulator on the benchmark's trace:

```
Table 18.1  The step stretches with the load: the envelope against the simulator, Qwen3-8B on one RTX A6000, 1024 tokens in, 128 out (measured: in-sample)
  req/s   rho  TPOT ms: envelope  simulator  measured  decodes: envelope  simulator  seat queue
    1.0  0.20               30.9       31.1      30.6                3.9        4.0         0.0
    2.0  0.40               41.2       39.9      39.5               10.5        9.9         0.0
    3.0  0.60               62.0       59.6      59.4               23.6       21.6         0.0
    3.5  0.70               82.9       81.1      80.0               36.8       33.7         0.0
    4.0  0.80              124.9      110.6     106.4               63.4       49.6         2.9
    5.0  1.00              125.8      115.6     112.9           64, full       53.2        18.7
    8.0  1.60              125.8      113.8     111.4           64, full       55.4        45.6
  a = 24.71 ms a step for the weights, shared; b = 0.356 ms per sequence; c_p = 152 us per prompt token
  one request's own work: prefill 1024 x 152 us = 155 ms, its share of 127 steps 127 x 0.356 = 45 ms; rho = 1 at 4.99 req/s
  simulator, all 240 requests at once: 3.97 completed per second; measured TPOT at 3.5 req/s from Table 18.8's re-run
  the envelope's knee, where the batch fills 64 seats (measured: p95 / median TTFT first passes 3x its lightest load between): 1024/128 4.01 (3.5-3.75 / 3.75-4); 512/384 3.05 (2.75-3 / 3.25-3.5); 1536/192 2.52 (2.25-2.5 / 2.25-2.5)
  simulator at 4, 5 and 8 req/s, while requests queue for a seat: 62.2-62.8 decoding, 1.2-1.8 prefilling; with no queue (23-48% of the run, its start and drain): 30-35 decoding
```

The simulator's decodes and seat queue are time-averages over the run;
the seat queue counts the requests in the engine beyond its 64 seats.
Up to 3.5 requests per second the envelope's step is within about 4% of
the simulator's TPOT, and nobody waits for a seat; the step roughly
doubles from 1 to 3, the 1/(1 − ρ) at work. At 4 the seats fill and a
queue appears. The tail tips first: the measured p95 between 3.5 and
3.75, the median between 3.75 and 4. Two more things stand out.

**The seats run out before the GPU's time does.** At the knee ρ is
0.80: a fifth of every step is still the shared weight read. More seats
would put more sequences on each read and move the knee toward ρ = 1
(Exercise 1). Here a scheduler setting decides the knee.

**The envelope's knee comes late:** at or past the rate where the
measured tail tips, in all three workloads of this chapter. The
envelope is a steady state; real arrivals come in bursts, and a burst
fills the seats before the average does. Past the knee the simulator's
53–55 decodes are an average: while requests queue, all 64 seats are
busy, one or two of them prefilling; the average counts the run's start
and drain.

This is why TTFT is flat, then steep. Below the knee a request waits a
step or two (chapter 16), then for its prefill, all growing smoothly
with the step. Past it, it first waits for a seat, for as long as
requests arrive faster than they finish.

## Queue amplification

Past the knee, arrivals at λ exceed completions at μ, the most the
server finishes per second with every seat full (3.97, Table 18.1), and
the queue grows at λ − μ. Near the knee the two are nearly equal. A
step error ε changes μ by a fraction ε, and λ − μ by a fraction
ε·μ/(λ − μ): at λ = 1.02·μ, a 2% error either doubles the excess or
removes it. Below the knee, the textbook single-server queue has the
same shape from the other side, with u = λ/μ the server's load (not ρ,
which leaves out the shared weight read):

```
below the knee:  wait ∝ u / (1 − u)                  a step error ε moves it by about ε / (1 − u)
past the knee:   the last of N requests waits about N·(1/μ − 1/λ)    moved by about ε · λ / (λ − μ)
```

Both factors blow up at the knee; the finite run and the bursts keep
them finite. Far past it the wait is long whatever the step: at 8
requests per second λ/(λ − μ) is about 2, and a 2% error moves the
wait about 4%.

![Three linked boxes: every step 2% longer, each burst drains later, the
queue's waits add up in the tail. Below, requests queued or prefilling
over a 70-second run for three step prices, and every request's TTFT
sorted, with p95 at 2.4, 3.1 and 3.9 s.](/assets/tinyperf-book/ch18-amplification.svg)

*Figure 18.1. Queue amplification at 4 requests per second in the
simulator: the same arrivals, every step 2% faster (blue), at the
model's price (grey) and 2% slower (orange). Left: requests without
their first token (queued or prefilling). Right: the fastest fifth of
requests barely move; the tail moves by about a quarter each way.*

`simulate(step_scale=s)` multiplies every step's GPU time by s. Table 18.2
runs it across the knee on the benchmark's trace, extending Table 1.4
to the median and to the queue itself:

```
Table 18.2  Every step 2% faster or slower: the simulator on the same trace, 1024 tokens in, 128 out
  req/s  TTFT p50 ms  x0.98  x1.02  TTFT p95 ms  x0.98  x1.02  longest seat queue at 0.98 / 1.00 / 1.02
   3.00          355   0.92   1.03          787   0.92   1.07              0 / 0 / 0
   3.50          437   0.94   1.08         1076   0.94   1.06              0 / 0 / 1
   3.75          587   0.89   1.14         1423   0.82   1.32              3 / 6 / 7
   4.00         1378   0.67   1.41         3126   0.78   1.24             11 / 12 / 15
   4.25         2937   0.81   1.17         5650   0.81   1.18             20 / 22 / 26
   4.50         4231   0.86   1.13         8305   0.86   1.15             28 / 30 / 33
   5.00         6162   0.94   1.09        12896   0.92   1.09             43 / 45 / 49
   6.00         8533   0.96   1.03        19263   0.95   1.04             74 / 78 / 80
   8.00        11238   0.96   1.04        26748   0.96   1.04            112 / 112 / 112
  steps 2% slower: the median rises most at 4 req/s (x1.41), the 95th percentile at 3.75 (x1.32)
```

A 2% step error is worth 3–8% of TTFT at 3–3.5 requests per second and
3–5% at 6–8. At 4 it moves the median 41% one way and 33% the other,
and the 95th percentile by about a quarter, as the longest queue for a
seat grows from 11 to 15.

The server shows it too. The server sampled with top-p, whose sampler
sorts every row's logits (chapter 15); a second run of the same trace
asked for greedy decoding. Both runs are in-sample.

```
Table 18.3  Steps 2-4% longer, measured: top-p against greedy sampling on one trace, 1024 tokens in, 128 out (in-sample)
         top-p adds  top-p over greedy:
  req/s     (model)  TTFT p50 meas  model  TTFT p95 meas  model  TPOT meas  model
      1        2.2%           1.00   1.01           1.01   1.00      1.027  1.029
      2        2.7%           1.01   1.02           1.04   1.00      1.042  1.046
      3        3.2%           1.03   1.05           1.03   1.07      1.099  1.104
      4        3.5%           1.55   1.86           1.40   1.51      1.104  1.122
      5        3.5%           1.11   1.10           1.14   1.14      1.037  1.037
      6        3.5%           1.07   1.08           1.09   1.10      1.032  1.036
      8        3.6%           1.06   1.06           1.07   1.07      1.037  1.039
  top-p adds (model): its sampler's price over greedy's in the greedy run's steps, over their GPU time
  the greedy run itself, model/measured: TTFT p50 1.01-1.08, p95 0.98-1.08, TPOT 0.998-1.023
```

In the model the sampler adds 2–4% to the steps. At every rate but
one, TTFT moves by 14% or less. At 4 requests per second, where the
median has just tipped, top-p's median TTFT is 1.55 times greedy's on
the server and 1.86 times in the model.

## The trace decides

Benchmarks send Poisson arrivals (chapter 14): random, at a fixed
average rate, so they come in bursts and lulls. Near the knee a burst
is a moment when λ runs above μ, and the queue it leaves depends on its
size and on how soon the next one comes. Chapter 14's Table 14.5
showed the effect on the median; Table 18.4 follows the tail. `vllm
bench serve` draws its arrivals from a seed, and traces 0, 1 and 2 are
its seeds, rebuilt request for request by `tools/bench_trace.py`.
Trace 0 was run several times.

```
Table 18.4  One process, many traces: TTFT p95 in ms, Qwen3-8B, RTX A6000, 240 requests of 1024 in, 128 out (trace 0 in-sample; traces 1 and 2 held out)
                       measured on trace        model on trace  model, 12 other traces  model, trace 0
  req/s         0 (runs)       1       2       0       1       2           range  tipped    steps +-1.7%
    3.0      769-790 (3)     806     880     787     833     868     560 to 1868    3/12         722-834
    3.5         1053 (1)     967    1087    1076     970    1106     656 to 5527    7/12       1022-1138
    4.0    2693-2830 (4)    2468    3609    3126    2764    3850    880 to 11097   10/12       2546-3744
    4.5         7728 (1)    6465    7206    8305    6927    7714   3343 to 16197   12/12       7361-9539
    5.0  12250-12309 (3)   10687   11594   12896   11140   11888   8177 to 20066   12/12     11967-13916
  tipped: p95 above 1100 ms (3x the measured p95 at 1 req/s); other traces: poisson_requests seeds 1000-1011; trace 0's runs on a slowed GPU left out (Table 18.8)
```

Read the 4-request row. The same trace, run four times, gave a p95 of
2,693–2,830 ms: the server repeats itself within 5%. Other traces of the
same process gave 2,468 and 3,609. On each run's own trace the model
reads 7–16% high, and ranks the three correctly. Twelve other samples of
the same process span 880 to 11,097 ms, over a factor of 12. The step
band, trace 0 with every step scaled between 0.983 and 1.017 (the band
of the next section), spans 2,546–3,744. At 3 requests per second, below
trace 0's knee, 3 of the 12 other traces already pass the 1.1-second
line.

Two rules follow.

- **To predict a measured run, replay its arrivals.** `bench_requests`
  (chapter 14) does, and every comparison here uses it.
- **One simulation of random arrivals is one draw.** For unrecorded
  traffic, a single seed's p95 is an ensemble member, not the answer.
  Report the spread over many traces, or the share that tip
  (`tinyperf.sweep.tip_fraction`). A process's knee is a band of rates.

## An interval

With the trace fixed, what is left is the model's own error. Its step
prices land within a few percent of the engine's clock (−1.9% to +2.6%
in Table 18.5), which Table 18.2 prices at the knee. So a prediction
becomes an interval: the simulator with every step scaled across the
model's step error, widened by what that error doesn't explain.

The band lives in the calibration, since it describes one stack:

```
"serving_error_band": {"step": 0.017, "ttft_p50": 0.052, "ttft_p95": 0.072, "tpot": 0.013}
```

It was fitted on the 13 online runs before Table 18.7's sweeps, 52
cells (a cell is one rate of one run), each held out from the step
model, among them Table 18.4's traces 1–2 and both runs of Table 18.6:
Qwen3-8B on one and two GPUs, and the MoE Qwen3-30B-A3B on two. The
engine logged every step (chapter 15's step timing), and a cell's *step
bias* is the model's price for its steps, at their logged composition,
over the engine's clock. `step` is the 90th percentile of `|bias − 1|`.
Scaling each cell's steps by its own bias removes the step error; the
three residuals are the 90th percentiles of the error that remains.

The scaling wraps the step model (docstring and the other price
methods, wrapped the same way, trimmed):

```python
class _ScaledSteps:
    def __init__(self, lat, scale: float):
        self._lat, self._scale = lat, scale

    def __getattr__(self, name):
        return getattr(self._lat, name)

    def decode_us(self, *a, **k):
        return self._scale * self._lat.decode_us(*a, **k)
    ...
```

`simulate` wraps its step model in it when `step_scale != 1.0`: decode,
mixed and prefill steps and the sampler scale; the per-request overhead
of chapter 16, not a GPU step, doesn't. The interval (docstring
trimmed):

```python
def prediction_interval(run, band: dict, points: int = 5) -> dict:
    def metrics(rep):
        return dict(ttft_p50=rep.ttft_ms(0.5), ttft_p95=rep.ttft_ms(0.95),
                    tpot=statistics.mean(q.tpot_us for q in rep.requests if q.gen > 1) / 1e3)
    d = band["step"]
    scales = [1 - d + 2 * d * i / (points - 1) for i in range(points)] if points > 1 else [1.0]
    runs = {s: metrics(run(s)) for s in scales}
    point = runs.get(1.0) or metrics(run(1.0))
    out = {}
    for m, v in point.items():
        vals = [r[m] for r in runs.values()]
        out[m] = (min(vals) * (1 - band[m]), v, max(vals) * (1 + band[m]))
    return out
```

`run(s)` simulates the benchmark's own requests with `step_scale=s`.
Five runs span 1 − d to 1 + d, the middle one the model itself. The
interval takes the smallest and largest of all five, not the two ends:
on one trace TTFT need not rise with the step price, consistent with
arrivals landing in different steps as the steps shift, and in 7 of
Table 18.7's 24 TTFT intervals the five runs are out of order. Each end
is then widened by the metric's residual. Off the knee the five runs
barely differ and the interval is little wider than the residual; at
the knee they spread, and it opens.

The fit itself used all 52 cells; the coverage was tested honestly,
each run's cells getting intervals from a band estimated on the other
twelve runs:

```
Table 18.5  The model's step error on 52 held-out online cells (13 runs), and the intervals it gives, one run left out at a time
  step bias, the model's steps over the engine's per cell: 0.981-1.026; 90th percentile of |bias - 1|: 0.017
             90th percentile error                              interval, one run left out  flat +-15%
  metric                 the model   own bias applied    inside  median width       widest      / +-5%
  TTFT p50                    9.0%               5.2%     50/52         1.21x        2.23x       50/52
  TTFT p95                   12.0%               7.2%     49/52         1.27x        2.76x       48/52
  TPOT                        3.9%               1.3%     49/52         1.08x        1.18x       48/52
  misses: 8 of 156; 6 in two runs whose bias sat at or past the edge of the others': an MoE on long prompts (1.015-1.026, mostly past the others' largest, 1.017), FlashInfer (0.981-0.983, past the others' smallest, 0.986); 2 residual
  step spread read at the recorded scales within the band: +-1.5%; recorded: each cell's step bias and its predictions at step scales 0.97-1.03; the statistics are recomputed from them
```

- **Most of the error is a steady step bias.** Scaling each cell by its
  own bias cuts the 90th-percentile TTFT error from 9–12% to 5–7%, and
  TPOT's from 3.9% to 1.3%.
- **Coverage is 94–96%.** A flat band on the point prediction, ±15% on
  TTFT and ±5% on TPOT, covers about as much. The difference is where
  each is wrong: the flat TTFT band is 1.35 times wide everywhere, too
  wide off the knee and too narrow on it, while the interval's median
  width is 1.21–1.27 and it opens to 2.2–2.8 near knees.
- **The misses are mostly a model gap.** Six of the eight are in two
  runs whose bias sat at or past the edge of what the others saw (the
  MoE, 1.015–1.026; FlashInfer, 0.981–0.983); two are residual misses.
  The recorded scales reach only ±1.5%, inside the band, so the coverage
  is if anything conservative.

Chapter 16's backends make a good test. vLLM's FlashInfer reads a mixed
step's decode caches once, where FlashAttention-2 re-reads them per
query head, so beside many long decodes its mixed step can cost half as
much. The same trace ran under both:

```
Table 18.6  One trace, two attention backends: Qwen3-8B, RTX A6000, 1024 tokens in (+-50%), 256 out (held out)
                 FlashAttention-2                               FlashInfer                     
            measured, ms   model/measured     measured, ms   model/measured    step    p95 at
  req/s      p50     p95      p50     p95      p50     p95      p50     p95    bias  own bias
    1.5      251     508     1.03    0.91      248     455     0.99    0.98   0.983      1.00
    2.5      348     728     0.95    1.05      301     645     0.96    0.91   0.982      0.93
    3.0     2579    4074     0.90    0.94      373    1186     0.93    0.72   0.981      1.02
    4.0     8341   17023     0.98    0.98     6312   11149     0.93    0.92   0.982      1.00
  at 3 req/s, FlashInfer's median TTFT over FlashAttention-2's: measured 0.14, model 0.15
  FlashInfer at 3 req/s, every step 2% slower: p95 852 -> 1218 ms (x1.43)
  the model's knee on this trace (p95 past 3x its 1.5 req/s value, rates 0.25 apart): FlashAttention-2 2.75-3, FlashInfer 3-3.25; the envelope's: 2.86 and 3.05
  step bias (recorded): the model's steps over the engine's clock; own bias: every step scaled by 1/bias
  the current model against the prices recorded for these cells: relative difference under 1e-09
```

The backend moves the knee. At 3 requests per second FlashAttention-2
has tipped, at a median TTFT of 2.6 s, and FlashInfer hasn't, at
0.37 s: 0.14 of it, where the model says 0.15. The model's knee moves
from between 2.75 and 3 requests per second to between 3 and 3.25, the
envelope's from 2.86 to 3.05, and both backends' medians land within
0.90–1.03.

One cell stands out: FlashInfer's p95 at 3 requests per second reads
0.72. The model's FlashInfer steps run 1.7–1.9% short of the engine's
clock in every cell, and at the knee that is worth a lot: with every
step 2% slower, the model's p95 moves from 852 to 1,218 ms, 43%. Scaled
by the cell's own bias it reads 1.02. It is also one of Table 18.5's
misses: left out, its run's bias lies past the others' smallest.

## How close is it?

The band was then tested as chapter 21 tests everything: on runs nobody
had seen, with the predictions *frozen*, committed to the repository
before the measurement ran. Two sweeps put most of their rates around
their knees: a decode-heavy one (512 tokens in, 384 out, each ±50%) and
a prefill-heavy one (1,536 in, 192 out). Both ran on an RTX A6000 that
carried nothing else. The criteria were written down with the
predictions: at least 80% of the 36 values (12 rates, three metrics)
inside their intervals, and TTFT intervals narrower than a flat ±15%
band, 1.35 times, in at least half the cells, but wider at the knee.

```
Table 18.7  Frozen intervals on two sweeps planned around their knees: Qwen3-8B on one RTX A6000 (held out)
  measured, and the interval [low point high] as ratios to it; * outside
  512 in, 384 out
  req/s                TTFT p50 ms                 TTFT p95 ms                   TPOT ms  step bias
   2.00      185 [0.92 0.99 1.08]        283 [0.87 0.96 1.07]     37.8 [0.94 0.98 1.02]       0.990
   2.75      224 [0.89 0.96 1.03]        523 [0.62 0.72 1.11]     48.1 [0.93 0.98 1.02]       0.988
   3.00      292 [0.81 0.91 1.02]       3161 [0.52 0.76 1.06]     50.2 [0.95 0.98 1.02]       0.987
   3.25      330 [0.81 0.91 0.99]*      5608 [0.74 0.88 1.09]     50.9 [0.95 0.98 1.02]       0.987
   3.50     1538 [0.46 0.79 1.14]       8900 [0.74 0.92 1.08]     51.5 [0.95 0.99 1.02]       0.987
   4.50     6775 [0.84 0.94 1.07]      18518 [0.84 0.95 1.07]     52.7 [0.96 0.99 1.02]       0.986
  1536 in, 192 out
  req/s                TTFT p50 ms                 TTFT p95 ms                   TPOT ms  step bias
   1.50      347 [0.93 1.02 1.07]        953 [0.89 1.00 1.11]     45.4 [0.94 0.99 1.03]       0.996
   2.00      451 [0.85 0.93 1.02]       1105 [0.89 0.97 1.08]     66.3 [0.90 0.96 1.03]       0.987
   2.25      579 [0.86 0.94 1.07]       1390 [0.91 0.99 1.13]     91.4 [0.89 0.98 1.07]       0.993
   2.50     1117 [0.83 1.01 1.35]       3445 [0.84 1.09 1.36]    116.5 [0.97 1.01 1.06]       1.003
   2.75     3673 [0.79 1.07 1.29]       6547 [0.87 1.04 1.22]    122.0 [0.98 1.01 1.04]       1.004
   3.50     7792 [0.94 1.05 1.14]      14462 [0.92 1.04 1.19]    121.8 [0.99 1.02 1.05]       1.007
  inside: 35 of 36; the point with a flat +-15% (TTFT) / +-5% (TPOT) band: 33 of 36
  TTFT widths, high/low: 15 of 24 under 1.35x (the flat band's), median 1.26x, widest 2.50x (512 in, 384 out, 3.5 req/s, TTFT p50)
  the five runs of a TTFT interval out of the order of their step scales: 7 of 24
  the cells' step bias (recorded): 0.986-1.007; the current model reproduces every frozen interval (relative difference under 1e-09)
```

![Two panels of TTFT against request rate on a log scale, median and
95th percentile as shaded interval bands with dashed point predictions
and measured dots; the bands open where the curves turn upward, and one
measured median sits just above its band.](/assets/tinyperf-book/ch18-intervals.svg)

*Figure 18.2. Table 18.7 as curves. The bands are narrow on the flat
part, widest where TTFT turns upward, and narrow again far past it. The
open circle is the one value outside.*

Both criteria were met: 35 of 36 values inside, where a flat band on
the point prediction holds 33. The flat band misses exactly the
decode-heavy sweep's knee cells, chapter 14's worst held-out cells: the
p95 at 2.75 requests per second, where the median is still a calm
224 ms but the tail has begun to queue, reads 0.72; the p95 at 3 reads
0.76; the median at 3.5, where the run tipped into saturation, 0.79.
Their intervals are 1.8–2.5 times wide and hold them. The one value
outside is the decode-heavy median at 3.25 requests per second, 330 ms
against an interval of 267–327.

## What intervals don't cover

An interval claims that the model's step error on the new run lies
within the band. Two things break that claim.

**A missing mechanism.** When the model lacks something the engine
does, its step bias leaves the band, and the interval misses by as much
as the queue amplifies. Two held-out sweeps of gpt-oss-20b, an MoE
with 4-bit experts, frozen after a first had exposed three missing
mechanisms (chapter 22), show it:

```
Recorded  gpt-oss-20b served online on one RTX A6000: two held-out sweeps with frozen intervals, as recorded
  short prompts (256-765 tokens): 10 of 18 values inside; the cells' step bias 0.965-0.983
  long prompts (1025-3068 tokens): 17 of 18 values inside; the cells' step bias 1.000-1.020
  together: 27 of 36 (75%), against a criterion of 80%
```

With the long prompts the steps were within 2% of the engine's, and 17
of 18 values landed inside. With the short prompts they ran 1.7–3.5%
short, mostly outside the band, and 8 of 18 values missed. Chapter 22
tells what the model was missing. Measure the step bias on every run;
when it leaves the band, look for a mechanism before trusting an
interval.

**A slowed GPU.** A GPU held back by its power or temperature limits,
or shared with another job, lengthens every step, and at the knee that
looks like a strange queue.

> **Field note: the knee that looked bistable.** The first sweep
> through this knee, in steps of 0.25 requests per second, measured a
> median TTFT of 4.6 s at 3.5 and 1.1 s at 4.0, on the same arrivals
> rescaled. A queue that shrinks as the load grows looked like a
> bistable server, quiet or queued at one load depending on its first
> bursts, and no plausible step price reproduced it. Run again on the
> same trace with the engine's step clock and GPU telemetry (Table
> 18.8), the queue grew with the rate, as the model's does. The first
> run's TPOT at 3.25–3.75 requests per second had been 30–67% above the
> re-run's, and at 3.5 and 3.75 its 133 and 121 ms exceed a full
> 64-seat batch on a healthy GPU (107–113 ms): its steps were slow,
> consistent with a throttled or shared GPU. It logged no clocks, so the
> cause is not known.

```
Table 18.8  One trace, run twice: the first sweep through the knee and its re-run, Qwen3-8B, RTX A6000, 1024 tokens in, 128 out (trace 0, in-sample)
  req/s  TTFT p50 ms: first run  re-run   model  TPOT ms: first run  re-run   model
   3.25                     415     360     391                84.6    64.7    68.2
   3.50                    4602     437     437               133.4    80.0    81.1
   3.75                    2654     596     587               121.1    92.9    94.5
   4.00                    1144    1146    1378               108.3   107.1   110.6
  measured TPOT with every seat busy on a healthy GPU (the re-run at 4 req/s, the in-sample sweep at 5-8): 107-113 ms
  the re-run logged the engine's step clock and GPU telemetry; the first run logged neither
```

A low clock alone is not a slowed GPU: at 1,500–1,530 MHz, in the
prefill-heavy sweep's cells from 2.25 requests per second up
(`data/validation/comparison_qwen3_8b_rtx_a6000_serving_intervals.txt`),
the step bias stayed 0.993–1.007. The step bias is the check.

## Where it breaks

- **One band per stack.** One number for every load and step
  composition, read off one GPU type, one engine version and two
  models. A new stack needs its own held-out runs first.
- **Step error only.** Not a missing mechanism, a slowed GPU or another
  trace (Table 18.4).
- **Few runs.** 52 cells in 13 runs, and a run's cells share its bias,
  so a run-level miss takes several cells at once.

## What you built

- An envelope for the knee: the step stretches as a/(1 − ρ), Little's
  law gives the batch, and the knee is where the batch fills the seats;
  within about 4% of the simulator's TPOT below it.
- Queue amplification, simulated (a 2% step error moves the median TTFT
  41% at the knee) and measured (top-p's sampler, 3.5% of each step in
  the model, moved it 55%).
- Two rules: replay the run's own arrivals; treat one simulation of
  random arrivals as one draw from a wide ensemble.
- Prediction intervals: the simulator across the measured step bias,
  widened by the residual; one run left out at a time, they covered
  94–96% of 52 held-out cells, narrow off the knee and wide on it.
- Evidence: 35 of 36 values inside intervals frozen before two sweeps
  around their knees; the attention backend moving the knee as predicted.

## Exercises

1. **More seats.** Rerun Table 18.2 with 256 seats, vLLM's own default
   (`VLLM` as shipped). Where does the knee move, and what limits it
   now: ρ, or chapter 17's KV pool?
   Compare with the envelope at S = 256.
2. **A band by composition.** Fit the step bias separately for steps
   that carry prefill and steps that don't, from the recorded cells of
   the serving-error retrospective in `data/validation`. Do Table
   18.7's knee intervals narrow, and does coverage hold?
3. **The knee as a probability.** Plot `tip_fraction` over 48 traces
   from 2.5 to 5 requests per second. How does the transition's width
   change with the number of requests per run?
4. **Unrecorded traffic.** Build an interval that also covers the trace:
   the step band's range over 12 Poisson traces. How wide is it at 4
   requests per second, and does it hold traces 1 and 2 of Table 18.4?

---

*[← Chapter 17: Memory under load]({% post_url 2026-09-29-tinyperf-17-memory-under-load %}) · [Contents](/series/tinyperf/) · [Chapter 19: Speculative decoding →]({% post_url 2026-09-29-tinyperf-19-speculative-decoding %})*
