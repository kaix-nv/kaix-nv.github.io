---
layout: post
math: true
title: "Building tinyperf M41: The latent cache: MLA and sparse attention"
date: 2026-08-31 17:40:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Multi-latent attention caches one 576-byte latent per token — 40x smaller than per-head KV — and DeepSeek sparse attention caps attention at top-2048 while its indexer pays for the whole context."
---

*Milestone 41 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — MLA + DSA in `nets/transformer.py` · Presets: `glm5_3_flash`, `kimi_k3`.*

Two of the four SOTA targets in the comparison campaign — GLM-5.3-Flash
and Kimi-K3 — use multi-latent attention, the mechanism this series had
deliberately not improvised. Now it is built, read from the released
configs.

**MLA in one sentence**: instead of caching per-head K/V, cache one
latent vector per token (kv_lora 512 + shared rope dims) and absorb the
up-projections into the query and output paths at decode. The cache is
**~40x smaller** than the per-head equivalent at Kimi's dims — the
entire reason trillion-parameter MoEs can serve long context at all.
The builder prices both real formulations: materialized per-head K/V in
prefill, absorbed-latent scores in decode, exactly as engines do.

**DSA (GLM-5-class) in one sentence**: a cheap 32-head indexer scores the
whole context, the main attention reads only the top-2048 tokens — and
the model now shows the trade honestly: attention time is *flat* in
context (pinned: 4k == 64k within 1%) while the indexer grows 14.7x over
the same range. The cap is real, and so is what pays for it.

Both presets carry their approximations on the label: shared experts
folded into top_k (perf-equivalent, ~0.3% params), layer patterns as
1-in-4 intervals (±1 layer vs the config lists), and param counts from
config arithmetic (GLM ~313B; Kimi ~5.5T) pending exact anchoring against
the checkpoint indexes — the local shards are LFS stubs.

## Exercises

1. The absorbed-vs-materialized crossover: at what chunk size should a
   chunked prefill switch formulations? Price both and find it.
2. DSA's indexer is itself cacheable (index keys per token). Add its
   cache to the capacity model and re-derive max context.
3. Kimi's KDA layers reuse the milestone-24 delta-rule machinery — but
   its gating differs. Read the released kernel and check whether the
   0.011 efficiency measured for Qwen's GDN transfers.
