---
layout: post
math: true
title: "Building tinyperf M38: What an image costs"
date: 2026-08-31 15:38:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "A 0.46B vision tower costs as much as the 8.2B trunk per image — bidirectional patch attention has no causal halving. One image is 4.4x a text-only TTFT."
---

*Milestone 38 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tinyperf/nets/vision.py` · Demo: `examples/27_multimodal.py`.*

The Qwen3.5-class checkpoint ships a vision tower this series has been
stepping around since milestone 24. It is the last unsupported piece of
the model file: a 27-layer ViT (dims read from the checkpoint — fused qkv
[3456,1152], merger 4608 -> 5120; modeled parameters land on the
checkpoint's 0.461B to within 1%), a 2x2 spatial merger, and then the
merged tokens join the text prompt for ordinary prefill.

To a performance model it is nothing new — GEMMs plus attention — which
is why the builder is ~70 lines. The one structural difference matters
though: patch attention is **bidirectional**. Every patch attends to
every patch of its image, no causal halving, so a 1024px image is a full
4096 x 4096 attention problem, 27 times.

## The table that corrected the author

Qwen3-8B + tower on a B200, 512 text tokens plus N 1024px images:

```
 images  encode ms  prefill ms  TTFT ms  vision %  vs text-only
      0       0.00        7.40      7.4      0.0%         1.0x
      1      14.93       17.87     32.8     45.5%         4.4x
      4      57.00       52.11    109.1     52.2%        14.7x
      8     113.09      106.79    219.9     51.4%        29.7x
```

I wrote the first draft of this post assuming the 0.46B tower would be
noise next to the 8.2B trunk, and the model said otherwise: **the image
tax splits roughly half and half**. The tower's quadratic, un-halved
attention over 4096 patches rivals the trunk's prefill over the 1024
merged tokens the image injects. One image makes TTFT 4.4x a text-only
request; a capacity plan that only counts the extra prompt tokens — the
common shortcut — is off by 2x.

That is the recurring lesson of this series in miniature: intuition
about which term dominates is exactly what an analytical model exists to
replace.

## Exercises

1. Resolution sweep: encode cost is quadratic in (px/patch)^2 — find the
   resolution where the tower overtakes the trunk outright.
2. Video: `temporal_patch=2` means frame pairs; price a 32-frame clip
   and find what dominates.
3. Compose with M37: images make prefill heavier and decode unchanged —
   how does the optimal P:D ratio move for a multimodal workload?
