---
layout: post
math: true
title: "Building tinyperf M48: Communication you cannot see: overlap as a declared fraction"
date: 2026-09-08 16:31:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Every collective in the model was priced in series. Serving engines hide some of it — how much depends on the stack. A declared overlap fraction makes the assumption explicit, and shows the tensor-parallel cliff at the node edge rests on it."
---

*Milestone 48 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `comm_overlap` on `StepLatencyModel`, `simulate`, sweep configs · Example: `examples/31_comm_overlap.py`.*

Every collective in this model is priced in series with the compute
around it. That was a deliberate choice back in milestone 5 and it has
survived forty milestones because it is honest about what the model
knows: an analytical graph has no schedule, so it cannot say which
all-reduce hides behind which GEMM. Milestone 34 added the other end of
the range — `total_us_overlapped`, every collective perfectly hidden —
and called the truth "somewhere between".

Serving engines do overlap communication, some of it, on some links.
Production models handle this with a mode switch ("full overlap", "no
overlap") that the user picks. That is what this milestone adds, in the
same spirit as `moe_imbalance` and the speculation acceptance rate: a
number the model prices but cannot know.

## The knob

`comm_overlap` is the fraction of *hideable* collective time that the
engine hides. Hideable is `total − max(compute, comm)`: communication can
disappear behind compute only up to the point where it becomes the longer
of the two.

```
step = total − comm_overlap × (total − total_overlapped)
```

`0` is the serial sum every number in this series has used. `1` is the
milestone-34 bracket. `0.5` is halfway. A single GPU has nothing to hide
and the knob does nothing to it, which the test checks.

## What we know about the right value

One measurement: milestone 45 ran tensor parallelism across two GPUs on
a PCIe host bridge under a real engine, and the all-reduces were fully
exposed — the frozen serial predictions matched TTFT to 1–5% with no
overlap at all. So on that stack and that link, `comm_overlap = 0` is not
a conservative default, it is the measurement. On NVLink, with an engine
that fuses communication into GEMM epilogues, it will be higher; nobody
has measured it here yet, and the envelope says so.

## Where the assumption bites: the TP cliff, revisited

LLaMA3-70B on H100s, decode at batch 32, step time in ms and the speedup
from each doubling of `tp`:

```
  tp  GPUs   serial     x   50% hidden     x   all hidden     x
   1     1    72.64  1.00        72.64  1.00        72.64  1.00
   2     2    38.98  1.86        38.44  1.89        37.91  1.92
   4     4    22.81  1.71        21.91  1.75        21.01  1.80
   8     8    15.99  1.43        14.43  1.52        12.86  1.63
  16    16    14.64  1.09        12.18  1.18         9.71  1.32
  32    32    16.43  0.89        12.34  0.99         8.24  1.18
```

Inside the node the three columns tell the same story: doublings pay,
less each time. The columns part at the node boundary. Under the serial
assumption `tp=16` is barely worth having (1.09×) and `tp=32` is a
regression; with everything hidden, 16 is a 1.32× step and 32 still
gains. The decision that milestone 34 made confidently — the cliff is at
the node edge — turns out to rest on the overlap assumption, and this is
the honest way to say so: the cliff is at the node edge *for an engine
that does not hide its all-reduces*. Prefill moves the same way; 8×2048
tokens on `tp=16` costs 586 ms serial and 308 ms fully hidden.

## Why a fraction and not a schedule

A schedule would need to know kernel ordering, stream assignment and
which engine fuses what — the kind of knowledge a kernel-level simulator
has and a 5000-line analytical model does not, by design. A fraction
lets the user carry the one number they can measure on their own stack
(run two configurations, take the ratio) and lets the envelope say
plainly what has and has not been measured. When an NVLink measurement
lands, this knob is where its result goes.

## Exercises

1. Measure it: run any tensor-parallel model under your engine on an
   NVLink pair, predict with `comm_overlap` 0 and 1, and bracket the truth.
2. Overlap for the expert all-to-all is a different mechanism (dual-pipe
   schedules); give it a separate fraction and see which one the
   64-GPU MoE layouts of milestone 44 care about.
3. Sweep `comm_overlap` in the knee finder (`examples/14`): how much does
   the price of a 20 ms token move between the two ends of the bracket?
