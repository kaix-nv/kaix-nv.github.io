---
layout: post
math: true
title: "Building tinyperf M29: What a cheap draft does to the price of interactivity"
date: 2026-08-30 21:40:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "A one-layer MTP head moves the SLO wall from 16 ms to 10 ms — and running beside a wall turns out to cost 25x, far more than the speedup itself."
---

*Milestone 29 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — the `mtp` path in `sweep.py` · Demo: `examples/23_mtp_economics.py`.*

Milestone 16 found that every deployment has a **wall**: an SLO below its
single-request decode time that no amount of money can buy. Milestone 17
moved that wall with a separate draft model, at the price of needing
roughly half its guesses accepted. Milestone 27 made the draft nearly
free. This milestone asks what that does to the bill.

Qwen3.5-class 27B on 8 B200s, prompt 2048 / gen 256, $/Mtok at the SLO
knee:

```
 TPOT SLO       plain     MTP k=1   MTP k=4 iid  MTP k=4 decay.8
      8ms    — wall —    — wall —        11.42$           28.73$
     10ms    — wall —      42.80$         6.32$           11.72$
     12ms    — wall —      12.16$         6.32$            6.62$
     16ms      85.04$       3.43$         2.75$            3.10$
     40ms       4.68$       3.43$         2.75$            3.10$
```

Three readings, in order of how much they should change your plans.

**The wall moves, and cheaply.** Plain decode cannot serve a 12 ms SLO at
any price; with the model's own MTP head it costs $12.16/Mtok, and the
lowest servable SLO drops from 16 ms to 10 ms. The head is one layer —
about 5% of a target pass — so this is close to free in a way a separate
draft model never was.

**Near the wall, price explodes.** At a 16 ms SLO the plain deployment is
*just* servable and pays $85.04/Mtok for the privilege, serving 157 tok/s
of goodput; the same deployment with MTP pays $3.43 and serves 3880. A
25x price difference is not the speculation speedup — the speedup is
under 2x. It is what happens when a deployment is forced to run at batch
sizes that keep it inside an SLO it can barely meet. Operating next to a
wall is expensive, and moving the wall is worth far more than the raw
token-rate gain suggests.

**And the assumption still costs money.** The last two columns are the
same model at the same acceptance rate and the same depth, differing only
in whether acceptance decays with position — milestone 28's correction.
At a loose SLO they agree within 13%; at 8 ms the i.i.d. version quotes
$11.42 against an honest $28.73. That 2.5x is not physics. It is the
error a capacity plan inherits from the formula everyone writes down
first, and it is largest exactly where the decision is tightest.

## The shipped configuration is the robust one

Notice what the k=1 column is immune to. With a single proposal there is
no second position for acceptance to decay at, so `decay` drops out of
the arithmetic entirely — the depth the checkpoint actually ships is
exactly the one where milestone 28's unmeasured assumption cannot hurt
you. Whether that was reasoned or found empirically, it is a robust
choice, and a model that keeps "measured" and "assumed" in separate
columns is what lets you notice it.

## Where the series has got to

This is the whole stack answering one question at once: a GEMM model
(milestone 2) under a calibrated device (9, 22, 26), running a hybrid
architecture (24) with a speculative head (27) under an honest acceptance
model (28), inside an event-driven serving engine (13), swept by a
harness (14) into an economic frontier (15, 16). Every layer is doing
work in that table, and each was small enough to check on its own.

## Exercises

1. Sweep k against the SLO: the optimal depth is not the same at 8 ms as
   at 40 ms. Where does it move, and why?
2. Add the MTP head's own KV to the capacity model and re-derive max
   batch — how much of the goodput gain survives?
3. Put price on the other axis: at what $/GPU-hour does a second replica
   beat speculation for the same SLO?
