---
layout: post
math: true
title: "Building tinyperf M4: The scheduler — where graph meets device"
date: 2026-08-18 16:41:20 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Dispatch each op family to its cost model, sum, and report — with the binding bottleneck attached to every row."
---

*Milestone 4 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tinyperf/scheduler.py` · Demo: `examples/03_graph.py`.*

## Execution = dispatch + sum

With workload (graph) and machine (device) described, "simulation" is
almost anticlimactic:

```python
EXEC_MODELS = {"gemm": _exec_gemm, "rw": _exec_rw, "comm": _exec_comm}

def execute(graph, device, dtype=DType.FP16):
    report = RunReport(device=device.name)
    for op in graph:
        result = EXEC_MODELS[op.op_type](op, device, dtype)
        result.count = op.attrs.get("count", 1)
        report.rows.append(result)
    return report
```

Walk ops in topological order, dispatch each to the cost model for its
family, sum. The dispatch table is a miniature of a production layer-model
factory, which maps every op type to a layer-model class
that itself searches over *implementations* (direct kernel vs implicit GEMM
vs event-simulated custom kernel). Our third layer of that hierarchy is a
function pointer.

Two assumptions are hiding in that loop, and both are worth making explicit:

1. **Serialization**: ops run back-to-back on one GPU with no overlap. For
   steady-state, kernel-granular workloads this is nearly true; frameworks
   do overlap communication with compute, which is why production models add
   dedicated comm-overlap models on top of a sequential scheduler.
2. **Statelessness**: each op is priced independently; op N's cache
   residue doesn't help op N+1. Analytical models live at L2-working-set
   granularity, not cache-line granularity.

## The `count` trick

`OpResult.count` lets a builder emit one op and declare "this happens 80
times". All decoder layers of an LLM see identical shapes, so building one
layer and multiplying is *exact*, and 80x cheaper to evaluate. Production models do the same thing one level up (detecting repeated
decode timesteps). Simulation speed is a feature: production fleets run
tens of thousands of simulations a day, and batch sweeps price millions of
configs.

## The report is the product

Everything upstream exists to fill in this table (`examples/03_graph.py`):

```
op        type  count   time_us   GFLOP     MB   TFLOPS  bound  detail
fc1       gemm      1      42.7   34.36   69.2    805.5  l2     m=4096 n=4096 k=1024 tile=256x128
fc2       gemm      1      40.4   34.36   50.3    851.2  l2     m=4096 n=1024 k=4096 tile=256x128
gelu      rw        1      23.0    0.00   67.1      0.0  dram
norm      rw        1       8.0    0.00   16.8      0.0  dram
TOTAL                     114.0 us        gemm 72.8% | rw 27.2%
```

Design the row schema before the models: *time, work, traffic, achieved
rate, binding bottleneck, provenance detail*. The `bound` column is the most
valuable — it converts a latency estimate into an engineering direction
("this MLP spends 27% in pointwise ops; fuse them"). Roll-ups by op type
answer the architect's first question (where does time go?); the CSV export
(`report.to_csv`) is the seed of every downstream pipeline — production
tools grew spreadsheet reports, then databases and dashboards, all
downstream of the same per-layer row.

## Exercises

1. Add an `energy_j` column: `flops * pJ_per_flop + bytes * pJ_per_byte`
   with public energy-per-op estimates. (Production models carry full
   energy models.)
2. Add a `--compare` mode that diffs two reports (device A vs device B) and
   prints per-op speedups — the germ of design-space exploration.
3. Implement comm/compute overlap: let a `comm` op's time hide under the
   *next* gemm op, and measure the effect on milestone 6's TP scaling.

*Next: milestone 5 builds a real LLM as a graph and reproduces the economics of prefill and decode.*
