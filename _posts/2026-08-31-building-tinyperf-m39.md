---
layout: post
math: true
title: "Building tinyperf M39: The hybrid meets a datacenter GPU"
date: 2026-08-31 17:30:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Decode passed as-is; hybrid prefill failed 27-58% everywhere and decomposed into three measured constants — the asserted GDN efficiency was 50x optimistic. Held-out validation: 5.3% and 0.0%."
---

*Milestone 39 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — three measured constants in `methodology.py` / `gemm_model.py` · Data: `data/validation/comparison_qwen3_8_27b_b200.txt` (frozen predictions, forward/reverse order control, full provenance).*

The A6000 validation (milestone 32) anchored the dense model. The hybrid
architecture — 48 gated-delta layers of the 64 — had only ever been
checked against another analytical model. This milestone puts it on a
real B200 under vLLM 0.22, predictions frozen first.

## Decode passed. Prefill failed honestly.

TPOT came back within 12% on the worst cell (6.5% geometric) with no
changes. TTFT came back **27-58% under-predicted on every cell** — a
mechanism miss the dense validation could never have seen, because it
lives entirely in the GDN prefill path.

The miss decomposed cleanly into three physical pieces, each of which
became a measured constant:

1. **The GDN kernel level.** Milestone 24 asserted a 0.55 math efficiency
   for the delta-rule core and milestone 25, unable to reach a fused
   kernel, could only verify the *scaling*. Silicon has now measured the
   level: the deployed Triton GDN-prefill path runs at ~**0.011** of
   tensor peak — the assertion was ~50x optimistic. (Decode never cared:
   it is state-traffic-bound, which is why it validated anyway.)
2. **A per-layer kernel floor (550 us).** The shortest prompt was missing
   ~21 ms that no token-scaled term could explain: 512 tokens is 8
   chunks, which cannot occupy 148 SMs — the same physics family as
   milestone 23's pipeline fill. The floor applies only when the chunked
   kernel actually runs; a few-token speculative verify takes the
   recurrent path and must not inherit it (the MTP economics test caught
   exactly that leak).
3. **6 ms of engine overhead per prefill pass** — scheduler, sampler,
   output processing; a software-stack constant in the milestone-26
   sense.

Fit on three of five cells, validated on the two held out: **-5.3% and
0.0%**. The test suite now pins all five against the committed
measurement.

## The three-way scoreboard

For the same cells, against silicon: tinyperf CALIBRATED-graph decode
lands at 6.1% MAPE; the production reference model's calibrated tier
lands at ~16%. On prefill, the reference agrees with tinyperf more
closely than either agrees with the machine — a reminder that two
analytical models agreeing is comfort, not truth, and part of that
earlier agreement was error cancellation between components.

## What this makes the envelope say

The hybrid architecture on B200 is now silicon-validated end to end:
TTFT within 5.3% held-out, TPOT within 12% worst-cell. The 0.55 default
still applies to uncalibrated devices and the envelope now says plainly
what silicon thinks of it.

## Exercises

1. The 0.011 efficiency and 550 us floor are per-stack constants —
   measure them under a different GDN kernel (fla upstream, or a future
   fused CUDA path) and watch them move while the mechanism stands.
2. The 6 ms engine overhead was fit, not traced — profile a vLLM prefill
   step and attribute it.
3. Re-run the milestone-37 disagg regime map for the hybrid with the
   corrected prefill: heavier prefill moves the P:D optimum — where?
