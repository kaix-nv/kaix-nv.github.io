---
layout: post
math: true
title: "Building tinyperf M27: MTP — the draft that comes with the model"
date: 2026-08-29 23:55:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Current checkpoints ship a one-layer multi-token-prediction head. As a speculative draft it costs 5% of a target pass, and break-even acceptance falls from 50% to 5%."
---

*Milestone 27 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tinyperf/mtp.py` · Demo: `examples/21_mtp_speculation.py`.*

Milestone 17 priced speculative decoding with a *separate* draft model —
a 1.1B TinyLlama proposing for a 7B target — and found a crossover: below
roughly 50% acceptance, speculation is pure overhead. That number is a
property of the draft's cost, and modern checkpoints changed the draft.

Look at what a current release actually ships:

```
mtp.fc.weight                       # fuse hidden state + next embedding
mtp.layers.0.self_attn.{q,k,v,o}    # ONE decoder layer
mtp.layers.0.mlp.{gate,up,down}
mtp.norm.weight
```

One layer. No separate vocabulary, no separate embeddings — it reuses the
trunk's. That is a multi-token-prediction head, and it is a draft model
that costs almost nothing.

## What it costs

The MTP head runs one decoder layer plus the LM head projection (which is
the expensive part, because vocabularies are large). On a 64-layer 27B
hybrid, per proposed token:

```
 batch      kv  target ms  draft ms  draft %  break-even alpha
     1    4096       8.90     0.460     5.2%             0.050
     8    4096       9.43     0.481     5.1%             0.038
    32   32768      18.76     1.004     5.4%             0.029
```

**Five percent of a target pass, and a break-even acceptance rate of about
five percent.** Compare the separate-draft arrangement on the same
hardware: a 1.1B draft for a 7B target costs 34% of a target pass and has
to be right about half the time.

That is a change of kind, not degree. With a separate draft you must ask
"is my draft good enough to be worth running?" With an MTP head the
question barely arises — the head has to beat *nothing*, and almost any
signal does.

## The speedup curve

```
  alpha  E[tokens]   TPOT ms  speedup
   0.00       1.00      9.34     0.95x     <- pure overhead, and only 5%
   0.05       1.05      8.90     1.00x     <- break-even
   0.40       1.40      6.67     1.33x
   0.80       1.80      5.19     1.71x
   0.95       1.95      4.79     1.86x
```

The ceiling is set by `E[tokens] = 1 + alpha` for a single MTP layer:
even perfect acceptance only doubles the tokens per verify pass, and the
verify pass is not free, so 1.86x is the asymptote. Deeper MTP stacks
(`mtp_layers > 1`) raise the ceiling and cost proportionally more per
cycle — the same k-sweep tradeoff milestone 17 left as an exercise, now
with a much cheaper draft.

## Why the model could answer this at all

`MTPStepLatencyModel` implements the same interface as milestone 13's
`StepLatencyModel`, exactly like milestone 17's separate-draft version.
So it drops into the serving engine, the knee finder, and the sweep
harness without changing any of them. And the draft's cost is computed by
*building a graph* — the MTP head is just the target's parameters with
`n_layers=1` — so all of the fusion, quantization, tiling and hybrid work
from previous milestones applies to the draft automatically.

Composability was the design bet made back at milestone 4, when the
scheduler became a dispatch table. Nine milestones later it is still
paying: a genuinely new inference technique needed 80 lines and no
changes anywhere else.

## Exercises

1. Sweep `mtp_layers` (1, 2, 4) against acceptance: deeper heads raise
   the token ceiling but cost more per cycle. Where is the optimum?
2. Acceptance is not constant across positions — the second proposed
   token is accepted less often than the first. Replace the i.i.d. alpha
   with a per-position curve and redo the ceiling.
3. Serve it: run the milestone-13 engine with an MTP latency model and
   measure what the SLO price curve (milestone 16) does when the wall
   moves this cheaply.
