---
layout: post
math: true
title: "Building tinyperf M52: Joules per token, and the cap the clock lives under"
date: 2026-09-08 17:02:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "A first-order energy model needs four coefficients no datasheet prints. Measuring them on a real GPU taught the model that a decode GEMV burns power on padded MACs, and that most of what a calibration called efficiency was the power cap."
---

*Milestone 52 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tinyperf/energy.py`, `tools/measure_power.py` · Example: `examples/35_energy.py`.*

Every number in this series has been a time. A fleet is also billed in
joules, cooled in watts, and clocked by a power controller that does not
care what the datasheet's boost frequency says. This milestone adds a
first-order energy model, and — because the coefficients it needs are the
ones datasheets never print — a tool to measure them on whatever GPU you
have, plus the measurements from this box's RTX A6000.

## Four terms, one cap

```
  E_op = executed_flops x pJ/flop x dtype_scale    the MACs the tensor cores issue
       + bytes x pJ/byte                            DRAM traffic
       + active_w x time                            the SM array running at clock
  E_step = sum(E_op) + static_w x total_time        the card holding a context
```

An op whose demand exceeds `tdp_w` cannot draw it; the controller drops
the clock until it fits. First order — power scales with frequency at a
fixed voltage, so the op stretches by `demand / tdp` — which is the
mechanism behind milestone 36's `sustained_clock_fraction`, now with a
cause.

The word *executed* is the part that took a measurement to learn.

## Measuring the coefficients

`tools/measure_power.py` runs four loads while sampling power: idle with
a context (static), a 2 MB in-place scale that keeps the SMs at clock
without touching DRAM (active), a 1 GB copy (bytes), and an 8192³ fp16
GEMM (flops). On the A6000:

```
  idle          78 W                     static
  l2_resident  145 W                     active = 67 W
  copy_1gb     241 W   683 GB/s          140 pJ/byte
  gemm_fp16    299 W   106 TFLOPS        1.45 pJ/flop  (at the cap: a lower bound)
```

Then a fifth load the fit had not seen: a GEMV, one row against the same
8192² weight — a decode step's shape. It moves 695 GB/s of weights and
does almost no useful arithmetic, so the first version of the model
predicted 239 W. It measured 299 W, pinned at the cap.

The missing 60 W is the tensor cores doing useless work. A GEMM kernel
tiles its output; at `m = 1` the tile still spans 64 rows, and 63 of them
are padding that the MMA units execute anyway, at full energy. Charge
energy on *executed* MACs — the GEMM model already knows its tiles, so
`GemmEstimate` now reports them — and the GEMV's demand comes out at
307 W: above the cap, capped, as measured. The model's own tile choice
for `m = 1` is 16 rows rather than the 64 cuBLAS used, so its prediction
is 268 W, 10% under, and the test pins exactly that.

The cap also closes an old account. The GEMM model at boost clock
predicts 150 TFLOPS for that 8192³ GEMM; silicon gave 106, and milestone
9's calibration absorbed the gap into a 0.75 math efficiency. The energy
model says the GEMM demands 391 W on a 300 W card: capped, it runs at
115 TFLOPS. Most of what the calibration called "efficiency" was the
power limit.

## A 70B model on two generations

LLaMA3-70B on eight GPUs, PROJ tier, with the H100 and B200 coefficients
*derived* from public TDP and peak rates (the device files say so — the
tool exists to replace them):

```
  device    phase     b   step ms  capped ms  avg W  J/token   tok/kWh
  H100_SXM  decode    8     11.88      11.88    277    3.287     1.1 M
  H100_SXM  decode  128     19.92      19.92    346    0.431     8.4 M
  H100_SXM  prefill   8    563.25     567.81    499    0.137    26.2 M
  B200_SXM  decode    8      8.16       8.16    376    3.066     1.2 M
  B200_SXM  decode  128     11.89      11.89    472    0.350    10.3 M
  B200_SXM  prefill   8    311.40     311.40    640    0.097    37.0 M
```

Two things the table says. **Batch is the energy lever for decode**: at
batch 8 the card spends most of its power being on — static and active
terms for a handful of tokens — and a token costs 3.3 J; at batch 128 the
same silicon costs 0.43 J per token, 7.6× less, for 1.7× the step time.
And **the cap is a prefill phenomenon**: only the math-bound prefill
approaches TDP (the H100 clips by 1%); decode never gets near it. Which
is why power-limited datacenter fleets throttle training and prefill,
and why a `sustained_clock_fraction` fitted on decode would be wrong for
them.

## What is not modeled

Voltage scaling (power ∝ f³ under DVFS proper, so the cap costs less time
than the linear rule says); communication energy (a collective is charged
its bytes and active time, not its link); thermal throttling separate
from the power cap; and every coefficient on every device except this
A6000 — the H100 and B200 numbers are derived and flagged, and the tool
is thirty seconds on a real card.

## Exercises

1. Run `tools/measure_power.py` on a datacenter GPU and replace the
   derived coefficients. Does the derived pJ/byte for HBM3 survive?
2. Add `dtype` to the measurement (an fp8 GEMM) and check the
   bits-proportional energy scale.
3. Combine with the sweep pricer: joules per million tokens next to
   dollars per million tokens, and find where they disagree about the
   best batch.
