---
layout: post
math: true
title: "Building tinyperf M1: A GPU is a bag of numbers"
date: 2026-08-18 16:40:05 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Describe a GPU as a dozen priceable rates and capacities, derive everything else, and read ridge points like a hardware roadmap."
---

*Milestone 1 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) —
`tinyperf/device.py`, `data/devices/*.json` · Demo: `examples/01_roofline.py`.*

## Choosing the numbers

For an analytical model, a device description must satisfy one criterion:
every parameter must be *priceable* — it must appear in some time equation.
Our `Device` has about a dozen fields:

| Group | Fields | Appears in |
|-------|--------|-----------|
| Compute | `sm_count`, `boost_clock_ghz`, `tensor_macs_per_sm_clk[dtype]` | math time |
| Memory | `dram_bw_gbps`, `l2_size_mb`, `l2_bw_gbps`, `smem_kb_per_sm` | memory time, tile feasibility |
| Overhead | `kernel_launch_us` | every kernel |
| Scale-out | `nvlink_bw_gbps`, `nvlink_hop_latency_us` | collectives |

Note what's *derived*, not stored: peak TFLOPS is
`sm_count x macs/SM/clk x 2 x clock`. Storing rates per-SM-per-clock instead
of aggregate TFLOPS means floorsweeping ("what if 8 SMs are disabled?") and
clock changes are one-field edits. That is exactly why production device specs
are layered the same way — architecture files carry per-SM microarchitecture,
product files carry SM counts, clocks, and memory config, and everything
downstream is derived.

Two sanity anchors (from `examples/01_roofline.py`):

```
A100_SXM_80GB : peak fp16 311.9 TFLOPS   (datasheet: 312)
H100_SXM      : peak fp16 989.4 TFLOPS   (datasheet: 989.4)
                peak fp8  1978.9 TFLOPS  (datasheet: 1979)
```

A subtlety worth internalizing: datasheet TFLOPS numbers imply a specific
clock. H100's 989 TFLOPS corresponds to 132 SMs x 2048 MACs x 2 x **1.83 GHz**
— not the max boost clock. Analytical models must pick a sustained clock and
say so; production models handle frequency properly with a GPU energy model
(power determines sustainable clocks).

## The roofline, and the ridge point

With a device in hand, the zeroth-order model is one line:

```python
def roofline_time_s(self, flops, nbytes, dtype):
    return max(self.math_time_s(flops, dtype), self.dram_time_s(nbytes))
```

The **ridge point** — `peak_flops / dram_bw` — is the arithmetic intensity
(FLOP/byte) where a kernel transitions from memory-bound to math-bound:

```
A100 fp16: 153 FLOP/byte     H100 fp16: 295 FLOP/byte     H100 fp8: 590
```

Read that table twice: every generation the ridge moves *right* (compute
grows faster than bandwidth), so ever more kernels are memory-bound. This
single number explains most of modern inference engineering — an LLM
decoding one token has intensity ≈ 2 FLOPs per weight byte, two orders of
magnitude below the ridge.

## What-if devices

Design-space exploration is the reason these models exist, so overriding
parameters is first-class:

```python
fat = Device.load("h100_sxm", dram_bw_gbps=6704)   # 2x bandwidth H100
```

The demo shows a 4096^3 GEMM doesn't care (math-bound, 138.9 µs either
way) — bandwidth upgrades are invisible to compute-bound work. Production
tools expose the same operation as device-override flags, and their batch
sweeps run millions of such what-ifs.

## In production-scale models

- Devices are several layers of spec files (product / architecture /
  derived-parameter schemas) with property groups so related fields
  override together.
- Instruction-throughput tables map dozens of op precisions to per-SM
  rates, often mirrored into a compiled backend for speed.
- Accelerators from any vendor fit the same schema — a device model is
  vendor-neutral arithmetic.

## Exercises

1. Add a B200-class device file from public figures and re-derive its ridge points.
2. Add `disabled_sms` floorsweeping as a derived reduction of `sm_count`.
3. Model sustained vs boost clock as a utilization-dependent function.

*Next up, milestone 2: the analytical GEMM model — tile search, wave
quantization, and an L2 reuse model that decides whether an 8192³ matmul is
math-bound or memory-bound.*
