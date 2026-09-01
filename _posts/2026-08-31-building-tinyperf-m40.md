---
layout: post
math: true
title: "Building tinyperf M40: The window: gpt-oss and bounded attention"
date: 2026-08-31 17:10:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "gpt-oss alternates full and 128-token-window attention: half the layers' KV is bounded, halving cache at 131k and making windowed attention flat in context."
---

*Milestone 40 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `sliding_window` in `nets/transformer.py` · Preset: `gpt_oss_120b`.*

The SOTA comparison campaign starts by closing capability gaps, and
gpt-oss-120b's is sliding-window attention: 36 layers alternating between
full context and a 128-token window. Dims from the released config; the
modeled parameter count lands on the card's 116.8B exactly, and the
checkpoint's MXFP4-experts-only quantization is a milestone-35 recipe
(attention, router, embeddings stay bf16) rather than a model dtype.

The mechanism is one field and one extra attention block: windowed
layers attend to `min(context, 128)` tokens, so their KV is **bounded**.
Two consequences, both now priced and pinned:

- at 131k context the model's KV is **half** an unbounded twin's, and
  `max_batch` grows accordingly — the window is a capacity feature at
  least as much as a compute one;
- windowed attention time is *flat* in context (pinned exactly equal at
  2k and 64k), so long-context decode grows only through the 18
  full-context layers.

Sinks (a few tokens of always-attended KV) are noted and not priced.

## Exercises

1. Sweep the window size: where does 128 vs 1024 vs 4096 actually bite,
   for compute and for capacity?
2. gpt-oss's pattern is 1:1; Gemma-class models ship 5:1. Generalize the
   pattern and find the capacity/quality frontier the vendors are
   walking.
