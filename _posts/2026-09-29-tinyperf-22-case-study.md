---
layout: post
title: "Building tinyperf, chapter 22: Case study: one constant, three hidden mechanisms"
date: 2026-09-29 12:22:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/22-case-study/
excerpt: "Chapter 21 set out a method for keeping a performance model honest. This chapter follows it through one case, in order, wrong turns included, because the wrong turns are what the method is for. The question: what does it look like when a fitted constant stands in for mechanisms nobody has measured? And how do you get them out?"
redirect_from:
  - /tinyperf/perf-modeling/2026/09/12/building-tinyperf-m60.html
  - /tinyperf/perf-modeling/2026/09/14/building-tinyperf-m61.html
  - /tinyperf/perf-modeling/2026/09/20/building-tinyperf-m63.html
  - /tinyperf/perf-modeling/2026/09/20/building-tinyperf-m64.html
  - /tinyperf/perf-modeling/2026/09/28/building-tinyperf-m80.html
  - /tinyperf/perf-modeling/2026/09/28/building-tinyperf-m81.html
  - /tinyperf/perf-modeling/2026/09/28/building-tinyperf-m82.html
  - /tinyperf/perf-modeling/2026/09/29/building-tinyperf-m83.html
---

*[Building tinyperf](/series/tinyperf/) · Part V: Knowing it's right · Code: [`tinyperf/scheduler.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/scheduler.py), the weight-only branch of `_exec_gemm`, and [`data/calibration/rtx_a6000.json`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/data/calibration/rtx_a6000.json) · Every table and the plot in this chapter come from `python3 book/scripts/ch22_case_study.py`.*

Chapter 21 set out a method for keeping a performance model honest.
This chapter follows it through one case, in order, wrong turns
included, because the wrong turns are what the method is for. The
question: what does it look like when a fitted constant stands in for
mechanisms nobody has measured? And how do you get them out?

The short answer: it looks like success. A constant fitted to a whole
step's time makes its cells agree, and keeps agreeing on held-out cells
built the same way. It fails when the mix of work inside the step
changes, and its errors surface elsewhere, looking like other problems.
You get the mechanisms out by pricing the step part by part, each kernel
timed alone, in the engine's configuration, on the engine's own inputs,
until the constant stops mattering. And a correct fix may make
agreement worse.

The constant is `weight_only_math_efficiency`, 0.66, which priced the
math of gpt-oss-20b's 4-bit expert GEMMs on an RTX A6000. Decode
constants that grew beside it teach the same lesson from another side.

By the end of this chapter you will know:

- how a constant fitted end to end absorbs other parts' errors;
- how one routing count, taken at the wrong steps, bent three kernel
  constants;
- what serving under load showed that fixed batches never did;
- how 0.66 was taken apart into three measured mechanisms, until it
  priced no prompt of more than 24 tokens;
- why a correct fix cost a validation sweep, and how frozen predictions
  made every failure visible.

## The case

gpt-oss-20b is a mixture-of-experts model (chapter 9): 24 layers of 32
experts, 4 per token. Its experts ship in MXFP4, 4.25 bits per weight
(chapter 10), which vLLM runs on the RTX A6000, a GPU without a 4-bit
math rate, through the Marlin kernel: it unpacks the weights to bf16
inside the GEMM. Its attention gives each head a learned sink, so vLLM
runs it with its own Triton kernel, not FlashAttention-2 (chapter 7).
Every measurement here is vLLM on one RTX A6000. Figure 22.1 is the map.

![A timeline of nine measurement runs of gpt-oss-20b, each with what it
showed, what was fitted or assumed, and what was measured or derived
instead.](/assets/tinyperf-book/ch22-timeline.svg)

*Figure 22.1. How the constant came apart. In runs 1 to 4 each surprise
was answered with a fitted or bent constant; from run 5 each answer was
a measurement. Two constants fitted to gpt-oss remain in the
calibration: 0.66, which now prices no prompt of more than 24 tokens,
and 0.44, which nobody has taken apart.*

## How a fitted constant hides mechanisms

A step's time is a sum of kernels. Group them into the part a constant
scales, E (here the expert GEMMs), and everything else: attention A and
the rest R. The model prices each part, and its error on the whole step
is the sum of the parts' errors, each over the measured time T:

```
model / measured  =  1  +  (Ê − E) / T  +  (Â − A) / T  +  (R̂ − R) / T
```

Fit a constant c inside Ê until the fitted cells read 1, and on those
cells

```
Ê(c) − E  =  −(Â − A)  −  (R̂ − R)
```

The fitted part carries every other part's error, with its sign
flipped. Three things follow.

- **On the fitted cells the sum is right**, by construction.
- **On held-out cells built the same way**, with similar shares of
  time, it stays right: they carry the same absorbed error in the same
  proportion.
- **On a step whose mix differs**, the absorbed error is the wrong
  size, and the residual follows the parts' shares: chapter 4's rule,
  seen from the other side.

And the story told about a fitted constant is a guess: the fit knows
the sum was short, not which part.

## The code: where the constant enters

The calibrated tier (chapter 4) prices a GEMM with chapter 3's tile
model, scaled by a math and a memory efficiency. This is the branch of
`_exec_gemm` for GEMMs with 4-bit weights, comments, the bf16 expert
branch and a detail string trimmed:

```python
        wo = op.attrs.get("weight_nbytes")
        is_expert = op.name.startswith(("moe_gate_up", "moe_down"))
        ds = 1.0
        if wo is not None:
            ms = ctx.weight_only_efficiency or 1.0
            if is_expert:
                ds = ctx.weight_only_dram_efficiency or 1.0
        ...
        est = estimate_gemm(ctx.device, op.m, op.n, op.k, _op_dtype(op, ctx), batch=op.batch,
                            out_dtype=op.out.dtype, sparse=op.attrs.get("sparse", False),
                            weight_nbytes=wo, math_scale=ms, dram_scale=ds)
        rate, tok = ctx.weight_only_expert_tflops, op.attrs.get("moe_launch_tokens")
        mixed, dec = ctx.weight_only_expert_mixed_us, op.attrs.get("moe_decode_rows")
        if wo is not None and is_expert and rate and mixed and dec and tok is not None and tok - dec >= mixed["chunks"][0]:
            ...                                  # a chunk beside decodes: the kernel's table by composition
        elif wo is not None and is_expert and rate and tok is not None and tok >= rate[0][0]:
            math_us = est.flops / (_log2_interp(rate, tok) * 1e6)
            t = max(math_us, est.dram_us, est.l2_us)
            est = replace(est, time_us=t + ctx.device.kernel_launch_us, math_us=math_us,
                          bound="math" if t == math_us else ("dram" if t == est.dram_us else "l2"))
```

`ms` and `ds` scale the tile model's math and memory rates. For a
4-bit GEMM, `ms` is the calibration's `weight_only_math_efficiency`,
0.66; for an expert, `ds` is `weight_only_dram_efficiency`, 0.80,
Marlin's weight-streaming rate timed inside decode steps (chapter 10's
Table 10.6). 0.66 once priced every expert GEMM of every gpt-oss
prefill. The two branches below replaced it: an expert launch carrying
a chunk beside decodes reads a table of the kernel timed on such steps,
and any other launch of 128 tokens or more takes its math time from
`weight_only_expert_tflops`, Marlin's rate by tokens per launch. Only
smaller launches, mostly decode, still see 0.66.

## The first run

The first run was an eight-cell grid: batches of 1, 8 and 32 prompts
of 512 to 8,192 random tokens, timed for TTFT (time to first token: the
prefill) and TPOT (time per output token: the mean decode step), as in
chapter 6. The predictions were *frozen*, committed before the run was
opened (chapter 21).

```
Table 22.1  The first run on real hardware: gpt-oss-20b with 4-bit experts, one RTX A6000 (vLLM), TTFT model/measured
  batch  prompt  measured ms  frozen (recorded)  as fitted, 0.66
      1     512         59.4               0.60             0.80
      1    2048        183.7               0.73             0.98
      1    8192        850.0               0.70             0.91
      8     512        344.0               0.75             1.01
      8    2048       1374.7               0.76             1.02
      8    8192       6925.8               0.68             0.89
     32     512       1299.9               0.78             1.06
     32    2048       5494.6               0.76             1.02
  as fitted (recomputed) against the ratios recorded after the fit: equal to two decimals in 8 of 8 cells
  refit on the five batch-8 and batch-32 cells, grid 0.01: 0.66 centring them (geometric mean nearest 1), 0.68 by least mean |ln ratio|
  TPOT, frozen (recorded): batch 1 0.91-0.96, batch 8 1.49-1.59, batch 32 1.13-1.17
  TPOT after the fair-router count (recorded): batch 1 0.91-0.96, batch 8 and 32 1.11-1.17
```

Two misses, with two shapes. Decode priced 49–59% high at batch 8,
within 9% at batch 1 and 13–17% high at batch 32. The model assumed
every pick was distinct, min(32, 4 × batch): all 32 experts at batch 8,
where a fair router touches 21 (chapter 9's Table 9.4); at batch 1 and
32 the two agree. A fair router's count replaced it (chapter 9's field
note), leaving batch 8 and 32 11–17% high.

Prefill priced 22–40% low at every cell. The suspect was obvious: the
expert GEMMs are most of a prefill's work, and Marlin unpacks 4-bit
weights in its inner loop. So the model gained a constant for
weight-only GEMMs, 0.66 of dense math efficiency, fitted on the five
batch-8 and batch-32 cells and checked on the batch-1 cells: 0.98 and
0.91 at 2,048 and 8,192 tokens, with a small-prompt miss at 512 left for
later. A textbook calibration.

The last column is recomputed: today's code with two later
measurements switched off reproduces the ratios recorded after the fit,
so the rest of the model has moved these cells by no more than
rounding. A refit returns 0.66 if it centres the five cells and 0.68 by
chapter 4's least mean `|ln ratio|`; the record doesn't say which
criterion the original fit used.

## The routing count

The next run was like for like: the same grid with every weight in
bf16, which vLLM runs through its Triton fused-MoE kernel on a default
configuration (no tuned entry for this GPU). With no 4-bit unpacking at
all, its prefill priced 45–48% low, further off than the 4-bit run
(Table 22.2's footer), and got a constant of its own,
`moe_math_efficiency` = 0.44. The number 0.66 stayed, but its story
fell: most of what it measured was not unpacking.

Decode at batch 8 and 32 priced 30–42% high. A router with favourites
touches fewer experts than a fair one (chapter 9), so a Zipf skew for
expert popularity was fitted on those cells: 1.4.

```
Table 22.2  Four counts of the experts a gpt-oss-20b decode step touches (of 32), random-token prompts
  batch                                        1     2     4     8    16    32    64
  uniform (a fair router)                    4.0   7.5  13.2  21.0  28.2  31.6  32.0
  Zipf skew 1.4, fitted to step times        4.0   6.0   8.9  12.9  17.9  23.4  28.2
  at the prompt's last position              4.0   5.5   8.4   9.7  12.5  15.9  18.7
  Zipf skew 2.15, fitted to that count       4.0   5.3   7.2   9.6  12.8  16.7  21.2
  averaged over the 128 decode steps timed   4.0   6.1   9.4  13.1  17.4  21.3  24.4
  bf16 run, frozen (recorded): TTFT 0.52-0.55 (all but 1 x 512), TPOT at batch 8 and 32 1.30-1.42
  after fitting 0.44 to its batch-8 and batch-32 prefills and the skew to its decodes (recorded): TTFT 0.91-1.03, TPOT at batch 8 and 32 0.90-1.07
```

Then the router was read directly, counting the distinct experts
chosen at each prompt's last position: 9.7 at batch 8, not the skew's
12.9. A direct measurement beats a fit, so the preset took a skew of
2.15. But the step times had fitted, so if fewer experts were read, each
was read more slowly: over the new count's bytes, the profiled kernels
streamed at 0.65–0.66 of the fitted DRAM rate (bf16) and 0.51–0.55
(Marlin). The calibration took 0.70 and 0.53 (the record doesn't say
why 0.70). A sweep of batches 2 to 64 turned them into curves read off
step times less the model's other parts: Table 22.3's first rows, down
to 0.53–0.55 at batch 64.

```
Table 22.3  The bf16 expert kernel's weight-streaming rate, as a fraction of the fitted 691 GB/s, read three ways
  batch                                                       1     2     4     8    16    32    64
  512-token prompts, last-position count (recorded)        1.00  0.77  0.81  0.65  0.75  0.74  0.53
    the same times, its decode-step count                  1.00  0.88  0.88  0.89  1.00  0.98  0.68
  2048-token prompts, last-position count (recorded)       0.99  0.82  0.80  0.67  0.67  0.72  0.55
    the same times, its decode-step count                  0.99  0.92  0.91  0.99  0.96  0.96  0.71
  profiled kernels over the last-position count's bytes, batch 8 and 32 (recorded): bf16 0.65-0.66, 4-bit 0.51-0.55; adopted: 0.70, 0.53
  timed launch by launch on its own step's routing: batch 1 1.00, 8 1.02, 32 0.96, 33 0.71, 64 0.71; 4-bit Marlin, in the calibration: 0.80
  batch 8, a gate-up launch replayed alone / in the engine's step (recorded): 435.0 / 608.6 us, 1.40x
  at the paired 47.2 us per expert: 9.2 and 12.9 experts' weights; the replay's step touched 9.9 (recorded), the engine's steps 12.7
```

The obvious check was to time the engine's expert layers alone,
replayed on inputs captured from a decode step. At batch 8 a gate-up
launch took 435.0 µs alone and 608.6 µs in the step. A kernel
1.4 times slower in company was investigated hard (chapter 21's field
note lists what was ruled out). It ran at the memory roof in both
places, so the conclusion was that the step read bytes it didn't need.

It didn't. Hooks on the routers inside the engine found that routing
spreads as a reply goes on (chapter 9's Table 9.6). A count at the
prompt's last position describes the first decode step at best; a TPOT
averages 128. Counted over the steps timed, batch 8 touches 13.1
experts: the fitted skew had been closer than the "direct" count.
Paired launch by launch with its own step's routing, the bf16 kernel
streams at the full rate up to 32 tokens per launch (chapter 9's Table
9.3). The extra time was uncounted experts: at 47.2 µs per expert, the
replay's launch is 9.2 experts' weights and the in-step one 12.9,
matching the replay's step (9.9) and the engine's (12.7: that run's 32
steps after 64-token prompts).

Table 22.3's second rows recount the curve's own step times over each
cell's decode steps: it flattens to 0.88–1.00 up to batch 32. One real
effect survives at batch 64, 0.68–0.71: vLLM's default configuration
switches to 64-row blocks once a launch carries more tokens than there
are experts. The skews, bent constants and curve were withdrawn; the
calibration streams bf16 experts at the full rate below that switch and
0.70 above it, and Marlin's at 0.80.

## Serving it online

A server's requests arrive at random, prompts ride in chunks beside
other requests' decodes, and every step's mix differs (chapters 14–16).
The first online sweep sent prompts of about 1,024 random tokens with
replies of about 256, each prediction frozen as an interval (the range
the model gives once its validated step error runs through the queue;
chapter 18): median and p95 TTFT and mean TPOT at six rates, 18 values.

```
Table 22.4  The first online sweep (prompts about 1,024 tokens, replies about 256): what fixed batches never showed
  values inside their frozen intervals, the model as it stood (recorded): 9 of 18
  decode steps of 17-64 decodes, model/engine (recorded): as it stood 0.88-0.90; with the three changes below 1.01-1.02
  1. the KV pool, tokens: the experts at bf16 67,312; at their 4.25 bits 638,432; the engine logged 607,344
  2. experts an online decode step touches over a batch decoded in step, 8-64 decodes: 1.11-1.19 times
  3. Triton's attention, decodes and a chunk in one call against the two run apart: cheaper in 60 of 60 cells
  the sweep re-priced with all three (in-sample, recorded): 18 of 18 inside
  the next two sweeps, frozen with the three (recorded): 10 and 17 of 18 inside
  stated before the next sweeps ran (recorded), prompt-carrying steps: 513-768 tokens 0.91, 769-1024 0.96, 1025-1280 0.97, 1281-1536 1.01, 1537-2048 1.02-1.07
```

Three things read off that run account for the misses. The capacity
model held the experts at bf16, so the simulated KV pool was a ninth of
the engine's (chapter 10's field note). Sequences out of step share
fewer experts, so the model now reads a count measured on the engine as
it serves (chapter 9's Table 9.7, the *online table*). And Triton's
attention pays no re-read in a mixed step (chapter 7).

With the three, the sweep came in, in-sample. What it left was written
into the next prediction file with its suspect: steps carrying a fresh
prompt priced low when small and high when large, and the file blamed
the constant fitted on prefills of thousands of tokens.
The next two sweeps, frozen, came in at 10 and 17 of 18; the one that
missed had the shorter prompts.

## Taking the constant apart

To see the residual without a queue in the way, one fresh prompt at a
time went to an idle server, and the engine's clock timed its forward.
Table 22.5's first row, the model with 0.66, is the frozen record.

```
Table 22.5  One fresh prompt on an idle server: the engine's forward, model/measured, gpt-oss-20b with 4-bit experts
  prompt tokens                                128   256   384   512   640   768  1024  1280  1536  2048
  engine ms                                   23.8  33.1  44.6  55.3  66.5  74.6  92.5 117.8 136.1 179.7
  0.66, FlashAttention-2's price              1.01  0.95  0.91  0.93  0.90  0.96  1.05  0.96  1.05  1.02
  Marlin measured alone                       1.01  1.00  1.01  0.98  0.96  0.98  1.00  0.94  0.95  0.93
  Triton's attention measured alone           1.02  0.96  0.92  0.94  0.92  0.99  1.09  1.01  1.10  1.09
  both measured (the model now, in-sample)    1.02  1.01  1.02  1.00  0.98  1.01  1.03  0.98  1.00  1.00
  the first row against the frozen record: within 0.001 at every length: yes
  the model now with 0.66 set to 1.0: no step here moves 1e-9 ms; of prompts of 1-128 tokens 10 move, none over 24 tokens, at most 0.13 ms
  the tile model x 0.66 against Marlin timed alone under uniform routing, per layer (recorded): 256-1024 tokens 0.83-0.97, 1536-2048 tokens 1.03-1.05
```

0.90 to 1.05, and not smoothly. Three mechanisms were inside it.

**The kernel's own rate.** A tool timed one layer's Marlin call on the
engine's repacked weights in a CUDA graph. Against it, the tile model
times 0.66 read 0.83–0.97 at 256–1,024 tokens and 1.03–1.05 at
1,536–2,048 (Table 22.5's last line). The model's tile choice jumps
between sizes as the rows grow; the kernel's rate rises smoothly. A
constant times a staircase can only average a ramp.

**The hot expert.** That kernel was timed with the tokens spread
uniformly over the experts. The engine's router doesn't spread them:

```
Table 22.6  Marlin under the engine's own routing: one prompt, every layer's experts, timed alone in a CUDA graph
  prompt tokens                                128   256   384   512  1024  2048  8192
  busiest expert's share, mean over layers    0.94  0.94  0.95  0.94  0.95  0.95  0.95
  time, engine routing / uniform              0.89  1.16  1.21  1.05  1.06  1.04  1.06
  median over layers: random-id prompts 0.96-0.98; vllm bench's prompts 0.98-0.99, timed at 1.00-1.02 of the random-id ones
  the first run's 8,192-token prompts with Marlin measured, attention at FlashAttention-2's price: 0.81, 0.79 (as fitted 0.91, 0.89)
```

On average over layers one expert takes 94–95% of a prompt's tokens
(chapter 9's 99% is the median on the benchmark's prompts, which time
within 2% of these). vLLM sizes Marlin's row blocks from the mean rows
over all 32 experts, which fits none when one expert holds nearly every
token, and from 256 tokens the kernel takes 4–21% longer than under
uniform routing. So it was timed again on each layer's own
router choices, and its rate stored by tokens per launch:
`weight_only_expert_tflops`, the table the code above reads.

**The attention.** With Marlin measured, the first run's own grid fell:
its 8,192-token prompts dropped from 0.91 and 0.89 to 0.81 and 0.79
(Table 22.6's last line). The constant had been covering something
else. The model had priced Triton's prefill attention as if it were
FlashAttention-2's. Timed alone, it runs at about a third of that rate,
consistent with each of its programs taking only two query tokens
(chapter 7's Table 7.3; not profiled). The attention backend now carries
0.215.

![Lone-prompt steps model over measured against prompt length, for the
model with 0.66, with Marlin measured alone, with Triton's attention
measured alone, and with both.](/assets/tinyperf-book/ch22-prefill-steps.svg)

*Figure 22.2. Table 22.5 as a plot. At 1,536 and 2,048 tokens either
measurement alone makes the step worse, in opposite directions;
elsewhere one alone can help. Only both bring every length within 3%
(in-sample).*

Both models price the rest of the step alike, so a step's ratio as
fitted is exactly its ratio now plus one shift per part:

```
Worked example  Where the fitted constant's errors went: a step's ratio as fitted = its ratio now + each part's shift
  step                     now   experts ms, fitted vs alone  shift   attention ms, fitted vs alone  shift  as fitted
  one prompt of 512       1.00                  37.6 vs 40.6  -0.05                      0.6 vs 1.6  -0.02       0.93
  one prompt of 2,048     1.00                134.7 vs 119.9  +0.08                     6.2 vs 18.7  -0.07       1.02
  8 prompts of 8,192      1.00              4177.0 vs 3486.8  +0.10                 721.3 vs 2180.6  -0.21       0.89
  alone: from the kernel measured alone; shift: (fitted - alone) / measured step
```

The constant priced the experts too cheap on small launches and too
dear on large ones. It priced attention at a third of its cost always,
a shift that grows with the prompt because attention's share does. At
2,048 tokens the two nearly cancel. The fit balanced errors of both
signs across its five cells; that is why they agreed.

So redo the fit:

```
Table 22.7  The fit redone on the same five cells, as each hidden part is measured: weight_only_math_efficiency and TTFT model/measured
  expert GEMMs                attention                  fitted  all 8 cells at the fit  8,192-token prompts
  tile model x the constant   FlashAttention-2 (0.65)      0.66               0.80-1.06           0.91, 0.89
  Marlin measured alone       FlashAttention-2 (0.65)      none               0.79-0.92           0.81, 0.79
  tile model x the constant   Triton, measured (0.215)     0.74               0.75-1.05           1.05, 1.03
  Marlin measured alone       Triton, measured (0.215)     none               0.92-1.02           1.02, 1.00
  fitted: grid 0.01, centring the batch-8 and batch-32 cells; by least mean |ln ratio|: 0.68 and 0.75
  none: with Marlin measured, any constant from 0.40 to 1.50 moves no cell by more than 1e-9 ms; the last row is in-sample (this grid is a pinned test)
  Marlin's measured rate, weight_only_expert_tflops: 39.2 TFLOP/s at 128 tokens per launch to 89.8 at 16,384
```

Price the attention right and the same fit returns 0.74 (0.75 by
chapter 4's criterion): the experts' math rate had been set 9–11% lower
to pay for attention priced at a third of its cost. Even 0.74 describes
no kernel: Marlin's measured rate runs from 39 TFLOP/s at 128 tokens per
launch to 90 at 16,384, which no single factor on a staircase follows.
With Marlin and the attention both measured, the grid reads 0.92–1.02
(in-sample: it is a pinned test) and the constant prices nothing.

## A correction that cost a sweep

A new sweep of long prompts, about 3,072 tokens with replies of about
64, was frozen with Marlin and the attention measured: 17 of 18 inside.
It could not tell the new model from the old: most of its prompt steps
carry more than 1,536 tokens, near where the two errors cancel (the
worked example), and both models price its steps within a few percent
(Table 22.8). Re-priced with nothing fitted to them, the three earlier
sweeps read 45 of 54, no better than before: steps with a chunk beside
many decodes still priced 6–11% low. The prediction file guessed
routing: decodes spread over more experts than the lone prompts behind
the Marlin table, so the kernel should run 10–25% above the model. The
engine was stepped by hand to record that routing, and the kernel timed
on it.

```
Table 22.8  A correction that cost a sweep: Marlin's table re-derived, and the long-prompt sweep (prompts about 3,072 tokens, replies about 64)
  frozen first (recorded): steps with a chunk beside many decodes 6-11% low; expected, the kernel 10-25% above the model at 16+ decodes
  Marlin on a mixed step's routing, 128-1024-token chunks (recorded): the frozen price over the kernel, 16-64 decodes 0.96-1.11, a lone chunk 1.05-1.24; each decode adds 4.0-9.2 us per layer, the model at most 3.7
  Marlin, TFLOP/s by tokens per launch                          128    512   2048   8192
    activation and sum subtracted at eager launch cost         42.0   62.8   82.7   89.3
    the same at CUDA-graph launch cost, as served              39.2   61.2   81.9   89.1
  the difference per layer, at every size: 43.0 us = 2 x (25.0 - 3.5); the graph-cost row is the calibration's table: yes
  frozen after the correction (recorded): its decode steps, contexts 1.5k-4.6k tokens, 1.03-1.10 by batch
  the long-prompt sweep as frozen (recorded): 1065 of 1264 prompt-carrying steps over 1,536 tokens; step bias by rate, the model before Marlin and attention 0.99-1.02, after 1.01-1.03
  the long-prompt sweep, priced by                           inside, of 18  TPOT point/measured
  the eager-derived table (frozen before the sweep ran)                 17            1.02-1.10
  the graph-derived table and the mixed-step table                      10            1.07-1.14
  the model now: routing by reply position as well                      16            1.01-1.11
  decode steps started together, online table, 32-64 decodes, prompts 512-4096 (recorded): 0.97-1.03; in each reply's first 16 tokens 1.03-1.17
  the sweep's decode steps of 2-15 decodes touch 0.76-0.94 of the online table's experts (recorded)
  held out, frozen with both changes: a new short-prompt sweep (512/384 tokens), 16 of 18 inside (recorded)
```

The expectation failed: at 16–64 decodes the frozen price sat at
0.96–1.11 of the kernel, never more than 4% below it. The kernel showed
two errors pulling against each other instead: a lone chunk priced high
(1.05–1.24), and each decode beside it priced low, 4–9 µs per layer
against at most 3.7 in the model. So the kernel's time went into a
table by decodes and chunk size, `weight_only_expert_mixed_us` (chapter
4's Table 4.9).

Building it turned up an error. Marlin's call also runs the activation
and the top-k sum, which the model prices as separate operations, so
its table was derived as the kernel's time less the model's price for
those two. That price used the eager launch cost, 25 µs each, where a
served step replays a CUDA graph at 3.5 µs (chapter 4). The table was
43 µs per layer too cheap. The correction is arithmetic, and it
reproduces the calibration's table exactly.

It cost a sweep. A new short-prompt sweep, frozen with the correction
and the mixed-step table, came in at 16 of 18. But the long-prompt
sweep fell from 17 to 10: its decode steps priced 3–10% high, and the
too-cheap Marlin price had been cancelling that. The fix took the
cancellation away. It stayed, because it was right, and the loss became
the next thing to measure.

The next prediction file blamed the long contexts, which were
innocent: decode steps started together barely move from 512 to 4,096
tokens. The replies were to blame. This sweep's were short, and early
in their replies sequences crowd onto the same experts (chapter 9's
Table 9.8), so its steps touched fewer experts than the online table,
which came from replies of 128 to 384 tokens (Table 22.8's last lines).
The model now scales a step's decodes beyond the first by a factor read
off their median position in their replies.

```
Table 22.9  Six online sweeps, 18 values each: how many land inside their intervals, by model (recorded)
  sweep  prompts/replies     frozen  Marlin, attention  + mixed-step table  + reply position (now)
  1             1024/256          9                 17                  18                      18
  2              512/384         10                 11                  18                      17
  3             2048/128         17                 17                  16                      18
  4              3072/64         17                 17                  10                      16
  5              512/384         16                 14                  16                      18
  6              1024/48         15                  -                  14                      16
  total                   84 of 108           76 of 90           92 of 108              103 of 108
  prompts/replies: mean tokens, +-50%; frozen: the intervals of the model of the day, committed first
  sweep 1's frozen 9 predates the online changes of Table 22.4; with them, in-sample, 18. '-': that model never priced sweep 6
  sweeps 1-3, before Marlin and attention were measured: 18 + 10 + 17 = 45; after: 17 + 11 + 17 = 45
  sweep 6, the model now (in-sample for its form), decode steps of up to 4, 8 and 16 decodes: 0.92, 0.96, 0.94; padded steps 0.93
```

The frozen column is the honest record; the others re-price old sweeps
with later models. Sweep 6 was frozen with a first form of the position
factor that scaled every decode, pricing a lone decode below the four
experts a token always touches; with a lone decode unscaled it reads 16
of 18, in-sample for that fix.

## Where it breaks

- **Short replies.** Under the model now, sweep 6's decode steps of up
  to 4, 8 and 16 decodes read 0.92–0.96, its padded steps 0.93. The
  position factor is one curve for every batch size.
- **Real text, online.** Every online sweep used random-token prompts;
  real text was measured only in fixed batches (chapter 9's Table 9.12).
- **0.44 is untouched.** The bf16 kernel's constant was fitted with
  FlashAttention-2's attention price on runs where the engine used
  Triton's. Priced with Triton's, bf16 TTFT reads 0.99–1.08 (chapter
  10's Table 10.9): consistent with attention living inside 0.44 too
  (not measured).
- **0.66 still prices** gpt-oss's decode launches, prompts of up to 24
  tokens (by at most 0.13 ms) and every other weight-only GEMM. No other
  4-bit kernel has been timed.
- **One engine version, one GPU, untuned configurations** own every
  table here.
- **Re-pricing is not holding out.** Only Table 22.9's frozen column is
  held out (chapter 21).

## What you built

- **The absorption rule.** A constant fitted end to end carries every
  other part's error, sign flipped: right for the mix of work it was
  fitted on, wrong elsewhere by an error that follows the parts' shares.
- **A method for taking one apart.** Isolate the step; price it part
  by part; time each kernel alone, in the engine's configuration (CUDA
  graphs, launch cost, attention backend) and on its own inputs (its
  routing, over the steps you time); refit to see what the constant
  carried.
- **Three mechanisms out of one number.** Marlin's rate by tokens per
  launch, the hot expert's effect on its block size, and Triton's
  prefill attention at 0.215. With them 0.66 prices no prompt over 24
  tokens, and the first run's grid reads 0.92–1.02 (in-sample).
- **Two rules from the decode side.** Count over the steps you time. A
  measurement of the wrong thing is more dangerous than a fit: it is
  trusted, and whatever is fitted around it absorbs its error.

Chapter 21 draws the general rules from this case.

## Exercises

1. **Take 0.44 apart.** Refit `moe_math_efficiency` on the bf16 grid's
   batch-8 and batch-32 prefill cells with `attn_backend="triton_attn"`,
   as Table 22.7 did for 0.66. How much of 0.44 was attention? What
   would you time to retire it?
2. **Where they cancel.** Using the worked example's method, find the
   prompt length at which the fitted model's expert and attention
   errors cancel. Would a grid of only that length have exposed either?
3. **A control for the count.** Design one measurement that would have
   caught the last-position count before any kernel constant was bent.
   (Batch 1 is exact under any count; what else is?)

---

*[← Chapter 21: Validating a performance model]({% post_url 2026-09-29-tinyperf-21-validating-a-performance-model %}) · [Contents](/series/tinyperf/) · [Appendix a: Using the estimator →]({% post_url 2026-09-29-tinyperf-appendix-a-using-the-estimator %})*
