---
layout: post
math: true
title: "Building tinyperf M3: A graph IR in 90 lines"
date: 2026-08-18 16:41:10 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Three tiny classes and one registration trick make workload description read like PyTorch — shapes and bytes, no data, no kernels."
---

*Milestone 3 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tinyperf/graph.py`, `tinyperf/operators.py` · Demo: `examples/03_graph.py`.*

## Description, not execution

A performance model never runs the network, so its IR needs no data, no
kernels, no autograd — just enough structure to answer two questions per op:
**how much math?** and **how many bytes?** That reduces the IR to three tiny
classes:

- `Tensor` — name, shape, dtype. Its `nbytes` property is the whole point.
- `Operator` — typed node; `build_outputs()` does shape inference.
- `Graph` — an ordered list of operators plus one good trick.

## The registration trick

The trick (standard in production graph IRs) makes graph-building read
like PyTorch:

```python
@Graph.register
class Linear(Operator):
    op_type = "gemm"
    def build_outputs(self):
        (x,) = self.inputs
        self.m = math.prod(x.shape[:-1]); self.k = x.shape[-1]
        self.n = self.attrs["out_features"]
        return [Tensor(self.name + ".out", x.shape[:-1] + (self.n,), x.dtype)]
```

`@Graph.register` synthesizes a `Graph.Linear(...)` builder method that
constructs the op, runs shape inference, appends it, and returns the output
tensor. Model code becomes:

```python
g = Graph("mlp")
x = Tensor("input", (4096, 1024), DType.FP16)
h = g.Linear("fc1", x, out_features=4096)
h = g.Elementwise("gelu", h)
h = g.Linear("fc2", h, out_features=1024)
```

Because ops can only consume tensors that already exist, insertion order is
a valid topological order — the scheduler just walks the list. (Production IRs keep a full dependency digraph because fusion passes need
to rewrite structure; we don't fuse yet, so we don't pay for it.)

## Three op families = three cost models

Every operator declares an `op_type`, which selects its pricing model
downstream. This is the load-bearing design decision:

| family | ops | cost model |
|--------|-----|-----------|
| `gemm` | Linear, BatchedMatMul | analytical GEMM model (milestone 2) |
| `rw`   | Softmax, RMSNorm, RoPE, Elementwise, Embedding | `bytes_moved / dram_bw` |
| `comm` | AllReduce, AllGather | ring collective model (milestone 6) |

The `rw` family encodes a deep truth: on a modern
GPU, *anything that isn't a matmul is a memory op*. A softmax does exps and
sums, but its time is `(bytes_in + bytes_out) / bandwidth` — arithmetic is
free next to HBM. So non-GEMM ops don't model math at all, just traffic
(with an `rw_multiplier` for multi-pass ops).

One modeling choice worth noticing: `BatchedMatMul` accepts explicit
`batch/m/n/k` attributes instead of requiring a second input tensor. In
attention, the "B matrix" is the KV cache — state, not a produced tensor.
An IR for *description* can afford such shortcuts; an IR for execution cannot.

## What a production graph IR adds

Beyond this skeleton: ~38 operator files; loaders from ONNX/prototxt/YAML;
graph passes that *rewrite* the IR (kernel fusion emulating TRT/cuDNN, and —
elegantly — training support implemented as a pass that appends backward-pass
Gradient ops to an inference graph); per-op annotation for structured
reports; and a memory allocator that walks the graph to estimate peak usage.

## Exercises

1. Add a `Concat` op and check shape inference composes.
2. Add an FMHA "fused attention" op whose traffic skips the S x S score
   matrix (the flash-attention effect), and diff prefill time vs milestone 5's
   unfused attention. (We do exactly this in milestone 7.)
3. Write a fusion pass: merge consecutive `rw` ops into one (saving one
   read+write of the intermediate), the first-order effect of pointwise fusion.

*Next: milestone 4 wires graph to device — a scheduler that dispatches each op family to its cost model and emits the per-layer report.*
