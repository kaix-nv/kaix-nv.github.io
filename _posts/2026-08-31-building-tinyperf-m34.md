---
layout: post
math: true
title: "Building tinyperf M34: Crossing the node boundary"
date: 2026-08-31 14:00:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Hierarchical NVLink+IB collectives from two bandwidth numbers: TP doubling buys 1.4-1.7x inside a node and 1.10x across it — the folk theorem, derived."
---

*Milestone 34 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — hierarchical collectives in `comm_model.py`, fabric fields in `device.py`, the overlap bracket on `RunReport` · Demo: `examples/24_multinode.py`.*

The envelope said it plainly: estimates were trustworthy up to eight
GPUs, because the communication model knew one link — intra-node NVLink.
Every deployment question above a node was outside the physics. This
milestone adds the second link.

## The mechanism

A device gains three public numbers: `gpus_per_node`, per-GPU inter-node
bandwidth (`ib_bw_gbps` — 25/50/100 GB/s for the shipped A100/H100/B200
HGX specs), and an IB hop latency. Collectives that fit in a node keep
the milestone-6 closed forms exactly. Collectives that span nodes run the
NCCL-style hierarchical decomposition:

    all-reduce  = intra-node reduce-scatter (NVLink)
                + inter-node all-reduce of the 1/gpn shard (IB)
                + intra-node all-gather (NVLink)

with all-gather and all-to-all decomposed the same way — for all-to-all,
`(g - gpn)/g` of every buffer rides the fabric, which is where expert
parallelism lives. A device with no fabric fields refuses to price a
spanning group at all: a number computed over a link that does not exist
is worse than no number.

## What it shows: the cliff

LLaMA3-70B prefill on H100s (NVLink 450, IB 50 GB/s per GPU):

```
  tp  nodes  serial ms  overlap ms  comm %  speedup vs tp/2
   2      1     188.10      175.37    6.8%        —
   4      1     114.10       94.76   16.9%      1.65x
   8      1      80.12       56.52   29.5%      1.42x
  16      2      72.85       38.62   53.0%      1.10x
  32      4      72.39       48.53   67.0%      1.01x
```

Inside the node, doubling TP buys 1.4-1.7x. The first doubling that
crosses the boundary buys **1.10x**, and the next buys nothing — at tp=32
two-thirds of the step is communication. This is the entire folk theorem
"fill a node with TP, go across nodes with PP or EP," derived from two
bandwidth numbers.

The overlap column is the second half of the milestone: `RunReport` now
brackets every graph between serial execution (`total_us`) and perfect
comm/compute hiding (`total_us_overlapped`) — the same
serial-vs-overlapped bracket milestone 21 used for DP sync. Production
TP-overlap lands between the columns; the model reports the interval
instead of guessing a point.

## Expert parallelism pays in fabric

MoE decode (35B-A3B class, batch 64), sweeping experts across nodes:

```
  ep  nodes  step ms   a2a %
   8      1    10.28    8.2%
  16      2     9.53   13.8%
  32      4     9.68   21.5%
  64      8    10.94   33.3%
```

Each doubling of `ep` halves the expert weights a GPU must stream — and
routes more of every token over InfiniBand. The step time bottoms out at
two nodes and comes back up; by ep=64 a third of the step is all-to-all.
The knob buys memory relief priced in interconnect, and now the price is
on the label.

## What this does NOT model

Rail-optimized topologies, NVSwitch domains larger than a node (NVL72),
multiple algorithms per collective, congestion, and actual overlap
scheduling. The envelope's out-of-scope list shrinks by one line but the
kernel of it stands: this is the two-bandwidth first-order model, honest
about being one.

## Exercises

1. Add an `nvl_domain` field for rack-scale NVLink (72 GPUs) and re-derive
   the TP table — where does the cliff move?
2. Model PP across nodes: microbatch activations ride IB point-to-point.
   Combine with milestone 12's bubble algebra for a 2-node 70B trainer.
3. The a2a latency term uses one IB hop per node — measure a real
   multi-node all-to-all and check the latency floor at small batch.
