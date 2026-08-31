---
layout: post
math: true
title: "Building tinyperf M30-33: From teaching tool to instrument"
date: 2026-08-31 13:45:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "A full-codebase review, an accuracy pass, silicon validation against a real serving engine (decode within 3.1%), and a support envelope pinned by CI. What it takes to make a model's numbers usable by someone who didn't write it."
---

*Milestones 30-33 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — everywhere · The contract: `ENVELOPE.md` · Validation data: `data/validation/`.*

The series changed its goal: tinyperf is now a **lightweight performance
estimator for LLMs, for research and production planning** — not just a
teaching artifact. That sentence obligates things a teaching artifact can
shrug off, and these four milestones paid the debt.

## M30: the review, and what it found

Three independent review passes over all ~4,000 lines, every finding
verified by reproduction. Six criticals — and every single one was a
**cross-milestone composition**: the training pass meeting the hybrid op,
the chunk step meeting the recurrent state, the sweep cache meeting
modified devices, the idle loop meeting in-flight prefills. Per-feature
tests had passed 33 consecutive CI runs while all six sat there. The
regression tests added now pin the compositions, not the features.

## M31: the accuracy pass

Everything that changed modeled numbers, validated before/after against
the cached cross-model comparison: logits projected for the last position
in inference prefill (was +14% TTFT on large-vocab models), L2 traffic
reformulated as window visits, the calibration grid extended past a rail
it had been silently sitting on, and serving switched to the measured
CUDA-graph launch constant by default. The anchors held; the hybrid
serving tier tightened from 1.15 to 1.04 against the reference model.

## M32: the instrument meets the world

Everything before this rested on cuBLAS microbenchmarks or agreement with
another analytical model. So: vLLM, a real 8B checkpoint, a real GPU,
eight batch-x-prompt cells, and tinyperf's predictions committed to the
repository **before** the measurement ran.

Decode came back inside **3.1% on every cell** — geomean 0.997. Prefill
came back exact for single sequences and up to **2.8x over-predicted**
for batched ones, and the error's shape named the culprit: the serving
engine had priced each admitted batch as one concatenated sequence since
milestone 13, quadratic attention and all. One `n_seqs` parameter later,
the same measurements read geomean 1.003, every cell within 5%.

The measurement also tried to lie to us first: identical prompts hit the
engine's prefix cache and produced TTFTs flat across a 128x token range.
An instrument that can be fooled by its own benchmark is not an
instrument; the tool now generates distinct random prompts per request
and the trap is documented in its docstring.

## M33: the envelope

Accuracy without a boundary is a rumor. `ENVELOPE.md` now states the
supported scope, the tier to use for each question, every measured error
bar with its provenance, every assumption you inherit (speculation
acceptance is *your* input; one constant is asserted, not measured; one
sits on a grid rail and says so), and an explicit out-of-scope list that
names what a production-grade model should do instead. The error bars are
pinned by a test: a model change that degrades silicon agreement now
fails CI. Two published posts got errata (milestones 19 and 29) because
the corrections above changed their headline numbers — the qualitative
conclusions survived, the magnitudes moved, and the posts now say so.

## What this arc teaches

Three lessons earned rather than asserted. **Bugs live in compositions**:
every critical was two correct milestones meeting incorrectly, which is
an argument for testing the products of features, not the features.
**Validation is a debugger**: the flattened-prefill bug was invisible to
23 cells of cross-model comparison and fell out of one afternoon against
real silicon, because the reference model never exercised the engine's
aggregation path. And **the instrument's honesty is the product**: the
error bar, the rail warning, the asserted-not-measured label, the errata
— that is what makes a number from a 4,000-line model usable by someone
who didn't write it.

## Exercises

1. Run `tools/measure_vllm.py` on a datacenter GPU and extend the
   envelope table — do the A6000 error bars transfer?
2. The out-of-scope list names multi-node topology and mixed-precision
   recipes as planned. Sketch the minimal mechanism for each before
   reading the milestones that add them.
3. Find the next composition bug: write one test that composes three
   or more milestones (say: MoE x hybrid x speculation x paged KV) and
   see what falls out.
