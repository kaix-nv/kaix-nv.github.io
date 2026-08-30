---
layout: post
math: true
title: "Building tinyperf M28: How far ahead should you speculate?"
date: 2026-08-30 10:35:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Acceptance decays with position, so expected tokens saturate while cost grows linearly. The i.i.d. formula does not just overstate the payoff — it recommends the wrong depth."
---

*Milestone 28 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `expected_tokens(alpha, k, decay)` in `spec_decode.py` · Demo: `examples/22_speculation_depth.py`.*

Milestones 17 and 27 both used the same acceptance model: every proposed
token is accepted independently with probability `alpha`, so a cycle of
depth k emits `(1 - alpha^(k+1)) / (1 - alpha)` tokens. It is the formula
everyone writes down first, and it is wrong in a way that matters.

A draft runs forward from the last verified token. Its second proposal is
conditioned on its own first guess, its third on two guesses, and errors
compound. Acceptance therefore *decays with position* — every published
measurement of speculative decoding shows this — and the i.i.d. formula
quietly assumes it does not.

## What decay does to the arithmetic

Model position i as accepted with `alpha * decay^(i-1)`. At alpha = 0.8:

```
 depth k      iid  decay .9  decay .8  decay .6
       2     2.44      2.38      2.31      2.18
       4     3.36      2.97      2.68      2.31
       8     4.33      3.17      2.73      2.32
      16     4.89      3.17      2.73      2.32
```

The i.i.d. column keeps climbing. Every other column **saturates** — by
depth 8 at decay 0.8, more depth buys nothing at all. That is the
qualitative difference: a geometric sum grows without bound in k, a
decaying one converges.

## What it does to the recommendation

Cycle cost, meanwhile, grows linearly in k: every extra proposal is
another draft pass. Divide saturating tokens by linear cost and the
optimum becomes finite and shallow (27B hybrid, B200, b=1, kv=4096):

```
 depth k    decay 1.0    decay 0.9    decay 0.8    decay 0.6
       3         3.48         3.74         3.99         4.48   ms/token
       8         2.91         3.98         4.62         5.44
      16         3.34         5.15         5.98         7.05
 optimum:          k=8          k=4          k=3          k=3
 speedup:        3.06x        2.46x        2.23x        1.99x
```

The i.i.d. model does not merely overstate the payoff (3.06x against a
realistic 2.2x) — **it recommends the wrong configuration**, k=8 where
the honest answer is k=3. An over-optimistic number is a nuisance; a
wrong recommendation is a bug. And it is the kind of bug an analytical
model exists to prevent, which is why it was worth going back to fix an
assumption two milestones after shipping it.

Worth noting where the released checkpoint sits: it ships **one** MTP
layer. Shallow.

## On honest defaults

`decay` defaults to 1.0, so every existing call reproduces the old
numbers exactly and the change is opt-in. That is deliberate but
uncomfortable: the default is the assumption we just argued is wrong.
The alternative — picking a decay factor out of the air — would trade a
visible wrong assumption for an invisible one. The right resolution is a
measurement: acceptance-by-position from a real draft/target pair, which
is a straightforward thing to log and something we do not have.

Until then the model can at least show what the assumption is worth,
which is the difference between 3.06x at k=8 and 2.23x at k=3.

## Exercises

1. Fit `decay` from a published acceptance-by-position curve (EAGLE and
   Medusa both report one) and redo the optimum.
2. Tree speculation changes the geometry: n parallel chains mean position
   i is reached if *any* chain survives. Does the optimum deepen?
3. Compose with milestone 16: plot the SLO price curve at k=1 vs the
   decay-optimal k. How much of the interactivity wall does honest
   speculation actually move?
