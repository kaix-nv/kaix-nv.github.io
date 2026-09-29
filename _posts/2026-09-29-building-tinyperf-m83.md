---
layout: post
math: true
title: "Building tinyperf M83: Not the context: a decode's routing follows its reply"
date: 2026-09-29 11:30:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Two milestones blamed long contexts for gpt-oss's decode steps pricing high. Frozen and measured, the context barely moved them. What moved them was where the batch sat in its replies: early on, sequences crowd onto the same experts. A factor by reply position brings the six gpt-oss sweeps to 103 of 108 values inside, and the held-out sweep showed where the frozen form was wrong."
---

*Milestone 83 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `GPT_OSS_20B_ROUTING_BY_POSITION` in `nets/transformer.py`, `decode_us(reply_pos=)` in `serving.py` · Data: `data/validation/comparison_gpt_oss_20b_rtx_a6000_reply_position.txt`.*

Milestone 82 lost a sweep. Its long prompts, 1,536–4,608 tokens, had
decode steps that priced 3–10% high. Two milestones had blamed the long
contexts and Triton's decode kernel there, which nobody had timed. This
milestone froze that claim and tested it.

## Frozen, then measured

Sequences started together at 512, 2,048 and 4,096-token prompts, and the
engine's clock timed their pure decode steps:

```
                      the fixed-batch table        the online table
  prompt tokens      8    16    32    64         8    16    32    64
    512            0.91  0.83  0.89  0.89      1.00  0.93  0.97  1.00
   2048            0.90  0.83  0.90  0.92      0.99  0.92  0.98  1.03
   4096            0.89  0.85  0.91  0.94      0.97  0.93  0.98  1.03
```

The frozen table read low everywhere. It came from greedy decodes, and
these sampled ones spread wider. Priced with the online table, the steps
barely moved from 512 to 4,096 tokens. Triton's decode kernel, timed
alone out to 6,144 tokens, sat within 0.7 ms of the model's price on a
step. The context was not it.

## What was

The steps within each reply's first 16 tokens read 1.03–1.17. Sweep C's
replies had been 32–96 tokens, against the online table's 128–384.
Traced online, its decode steps touched 6–24% fewer experts than the
table has.

The patch that reads the routers now records each decode's position in
its reply too. Two workloads ran with it, one with short replies and one
with long ones. At equal decodes, a step's experts grow with where the
batch sits in its replies:

```
  decodes      median position under 64     median position 256 or more
     4                8.4-9.4                        12.4
    12               12.0-16.0                       22.6
    16               15.0-18.5                       24.9
```

Early in their replies, sequences crowd onto the same experts; later they
spread. At equal positions the two workloads route alike. The simulator
knows every sequence's position. So `gpt_oss_20b("online")` carries a
factor on the decodes beyond the first, by the batch's median position,
equal to 1 at 128 tokens, where the online table's own workload sits.

## Frozen again

Sweep E had short replies (24–72 tokens) on a new seed. 15 of its 18
values landed inside, where M82's model would have held 14, with
throughput at 1.00. Its TPOT exposed the mistake in the frozen form. At
2–6 req/s it read 6–9% low, and decode steps of up to 16 sequences read
0.89–0.95. That form had scaled every decode, so a lone decode priced
below the four experts a token always touches. Scaling only the decodes
beyond the first is the correct form; a single sequence has nothing to
crowd with. E is in-sample for that fix. With it E reads 16 of 18.

Across the six gpt-oss sweeps, 103 of 108 values now land inside, against
92 for M82's model. M81's long-prompt sweep is back to 16 of 18. Short
replies still leave early, mid-size batches and padded steps 4–9% low.
The position traces shared seeds with sweeps C and D, though they
recorded routing, not time.

Everything ran on GPU 1. Only this milestone's own processes used it.

## Errata

- **M81, M82:** the residual they named, decode steps at long contexts,
  was neither the context nor Triton's kernel. It was short replies.

## Exercises

1. Early, mid-size batches still read 4–9% low. Fit the factor by batch
   size as well as position. Do a handful of cells hold it?
2. Padded steps are scaled by position too, but their stale rows are not
   replies. Leave them unscaled and re-price sweep E.
3. The traces were random-token prompts. Does real text crowd early in a
   reply the same way?
