---
layout: post
math: true
title: "Building tinyserve M6a: Tensor parallelism: split every layer"
date: 2026-10-05 08:00:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Split attention heads and MLP channels across two GPUs, then follow the collectives, KV shards, and measured cost."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m6a-tensor-parallelism.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M5b — Prefix caching: the same history, computed once]({% include tinyserve-post-url.html slug="building-tinyserve-m5b" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m5b-prefix-caching.md" %}) · Next: [M6b — Pipeline parallelism: split the layer stack]({% include tinyserve-post-url.html slug="building-tinyserve-m6b" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m6b-pipeline-parallelism.md" %})

M0–M5 made one GPU useful: avoid repeated computation, batch requests, page
KV memory, schedule continuously, bound prompt interference, reuse prefixes,
and replay decode work. M6 asks a different question:

> What changes when one GPU cannot hold or execute the model by itself?

There are several valid partition axes. Pipeline parallelism gives each GPU
different layers. Expert parallelism gives each GPU different MoE experts.
Context parallelism gives each GPU different parts of the sequence. Tensor
parallelism, the subject of M6a, splits the matrix work *inside every layer*.

This post builds the smallest complete TP=2 path: two processes, two GPUs,
rank-local weights and KV heads, two visible all-reduces per decoder layer,
one scheduler and sampler, FlashInfer decode, CUDA-graph replay, correctness
tests, and a matched measurement.

## The problem: a layer is larger than one device

Qwen3-0.6B easily fits on one RTX A6000, so it is not a compelling deployment
target for tensor parallelism. That is useful for learning: a small model lets
us compare one GPU and two GPUs directly and makes communication overhead hard
to hide.

The mechanism is meant for the case where weights, KV, or compute no longer
fit comfortably on one device. Simply copying the full model to two GPUs is
data parallelism: it serves more independent work, but one request still needs
a complete model on one GPU. TP instead makes both GPUs cooperate on every
transformer layer of one logical model.

A correct teaching implementation must prove more than “both GPUs are busy”:

- each rank retains different weight shards;
- each rank stores only its own KV heads;
- partial layer outputs are combined at the correct boundary;
- only one rank advances request state and samples tokens; and
- optimized attention and graph replay preserve the eager reference behavior.

## The split: column first, row second

Consider a linear layer written as `Y = X Wᵀ`.

A **column-parallel** linear splits the output features of `W`. Rank 0 computes
one slice of `Y`; rank 1 computes the other. No communication is required yet
because the next operation can consume the corresponding local feature slice.

A following **row-parallel** linear splits its input features. Each rank
multiplies its local input slice by the matching weight columns and produces a
partial result with the full residual width. The mathematical result is the
sum of those partials, so the ranks run an all-reduce:

```text
rank 0: partial_0 = local_features_0 × weight_columns_0
rank 1: partial_1 = local_features_1 × weight_columns_1

replicated result = all_reduce_sum(partial_0, partial_1)
```

Qwen3 uses this pair twice per decoder layer:

1. q/k/v are column-parallel; attention is local by complete heads;
   `o_proj` is row-parallel and ends with an all-reduce.
2. MLP gate/up are column-parallel; `down_proj` is row-parallel and ends
   with another all-reduce.

The residual stream is therefore replicated at every layer boundary. This is
the invariant that lets the next layer begin without an all-gather.

[![Two ranks compute local attention heads and MLP channels, then reconstruct the replicated residual stream with two all-reduces](/assets/tinyserve/m6a-tensor-parallel-dataflow.svg)](/assets/tinyserve/m6a-tensor-parallel-dataflow.svg)

*Figure 1. One Qwen3 decoder layer at TP=2. The blue dashed arrows are the
rank-0 control plane; the solid path is tensor work. Each rank owns different
attention/KV heads and MLP channels. The two orange all-reduces reconstruct
the full residual-width tensor on both ranks.*

## A concrete Qwen3-0.6B layer

The model has hidden width 1024, 16 query heads, 8 KV heads, head dimension
128, MLP width 3072, and 28 decoder layers. For a decode batch `[B, 1]`, both
ranks begin with the same residual tensor:

```text
x on each rank                 [B, 1, 1024]

rank-local q                   [B, 1, 8, 128]
rank-local k/v                 [B, 1, 4, 128]
rank-local attention output    [B, 1, 8, 128]
o_proj partial                 [B, 1, 1024]
all-reduce 1 output            [B, 1, 1024]  replicated

rank-local gate/up             [B, 1, 1536]
down_proj partial              [B, 1, 1024]
all-reduce 2 output            [B, 1, 1024]  replicated
```

The projection dimensions must divide evenly by the TP degree. Tinyserve
checks query heads, KV heads, and MLP width before useful distributed work
begins. A rank may read the full CPU checkpoint in this educational loader,
but only its selected q/k/v, o, gate/up, and down shards move to that GPU.
Embeddings, normalization weights, and the tied embedding/LM-head tensor stay
replicated.

The two all-reduces per layer mean one Qwen3-0.6B forward executes 56
collectives. This cost is the reason TP is not automatically a speedup.

## KV pages are logical peers, physical shards

The scheduler and block manager remain mirrored across ranks. If a sequence's
block table is `[3, 9]`, both processes use those logical block IDs and write
the same token offsets. The underlying tensors differ:

```text
rank 0 block 3: KV heads 0..3
rank 1 block 3: KV heads 4..7
```

No rank owns a complete KV block for the model. Each FlashInfer wrapper plans
decode with the local counts—8 query heads and 4 KV heads—not the global
16 and 8. It then reads only that rank's paged pool. Attention stays local;
the communication boundary is after `o_proj`, not inside paged attention.

This preserves the M3 page-table abstraction: scheduling reasons about one
logical cache layout while the model stores rank-local head shards behind it.

## One scheduler, two tensor workers

Running the scheduler independently on both ranks would be fragile. A timing
difference could change admission or preemption order, and one rank would
enter a collective for a request its peer did not select.

Tinyserve instead has one control-plane owner:

1. Rank 0 tokenizes prompts and owns arrivals, FCFS admission, preemption,
   timing, sampling, and returned output.
2. It broadcasts immutable request inputs once.
3. Each engine step, rank 0 broadcasts the exact arrived/admitted/preempted
   request IDs and the decode membership.
4. Worker ranks apply that plan to mirrored `Request`, `Sequence`, and block
   metadata, then execute their local weight and KV shards.
5. Only rank 0 runs the final LM head and samples. The selected token IDs are
   broadcast before every rank appends them and begins the next forward.

Chunked prefill uses the same rule. Rank 0 chooses either one packed batch or
one bounded chunk; workers receive the action and verify that their local
`num_cached` cursor matches its advertised start. The scheduler remains one
policy replicated as explicit state transitions, not two policies expected to
make the same decision by luck.

## FlashInfer and CUDA graphs under TP

The first M6a slice deliberately used fp32 gather attention and eager
execution. That separated sharding correctness from two optimized systems
with their own constraints. The completion slice adds them one at a time.

For eager FlashInfer, every rank builds a wrapper over its local KV pool and
plans with local head counts. The step still performs the same two NCCL
all-reduces per layer.

CUDA graphs then capture the complete decode forward, including those NCCL
collectives. Capture and replay must occur in the same bucket order on both
ranks. Persistent token, position, slot, and page-table buffers keep their
addresses fixed exactly as in M5c. Worker graphs execute every transformer
operation but return no logits; rank 0's graph also captures the final norm
and LM head.

One lifecycle detail was easy to miss. NCCL retains resources for collective
operations captured by CUDA graphs. Calling `destroy_process_group()` while
captured graph objects still exist can hang in communicator destruction. This
is a documented open [PyTorch issue](https://github.com/pytorch/pytorch/issues/115388).
`GraphRunner.close()` therefore synchronizes and explicitly resets every
captured graph before the distributed context destroys its process group.
Relying on eventual Python garbage collection was not sufficient.

The dispatch ladder is now the same on TP1 and TP2:

```text
graph bucket fits  → FlashInfer CUDA-graph replay
otherwise          → eager FlashInfer
fp32/reference     → eager PyTorch gather attention
```

## Where the code lives

The implementation stays intentionally small and visible:

- [`distributed.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/distributed.py) owns process-group lifetime,
  broadcasts, all-reduces, row/column linear layers, and optional collective
  accounting.
- [`loader.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/loader.py) maps checkpoint names to replicated,
  output-sharded, or input-sharded tensors.
- [`qwen3.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/models/qwen3.py) derives local head/channel counts,
  runs local attention and MLP work, and skips worker LM heads.
- [`engine.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/engine.py) keeps rank 0 as the policy owner and
  mirrors plans, sampled tokens, block tables, and request transitions.
- [`graph_runner.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/graph_runner.py) captures local FlashInfer
  plus NCCL, counts real replays, and explicitly releases graph resources.
- [`bench_tp.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/examples/bench_tp.py) runs the matched TP1/TP2 workload
  and the separately labelled collective attribution pass.

The CLI exposes the optimized path directly:

```bash
.venv/bin/torchrun --standalone --nproc-per-node=2 examples/generate.py \
  --tp-size 2 --backend flashinfer --serve \
  --prompts "2+2=" "Capital of Japan?" \
  --max-new-tokens 16
```

`--backend gather --no-cuda-graphs` keeps the readable eager reference
available while debugging.

## The proof—and its boundary

[`tests/test_tensor_parallel.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tests/test_tensor_parallel.py) launches
real two-process tests on the two A6000s and checks three layers of evidence:

1. CPU tests reassemble every checkpoint shard exactly and verify collective
   byte accounting.
2. A two-GPU fp32 test compares one decoder layer within `2e-4`, observes
   exactly two all-reduces, verifies half-sized KV-head pools, and matches
   greedy TP1 tokens through paged generation, continuous serving, chunked
   prefill, warm prefix adoption, and forced preemption.
3. A bf16 optimized test compares TP eager FlashInfer with TP graph replay,
   matches controlled TP1 prompts, exercises `serve()`, proves that graph
   replay actually occurred, and verifies that worker ranks return no logits
   or output objects.

The exact-token claim is intentionally scoped. All-reduce changes floating
point accumulation order. On stable controlled prompts the bf16 optimized
test is token-identical; a near-tied argmax can still flip on another prompt.
The matched benchmark below recorded 12 of 16 rows identical to TP1. That is
consistent with changed bf16 accumulation order, but the run did not retain
the logits needed to prove why the other four rows diverged. Every request
still generated the fixed 32-token workload, so the performance tensor shapes
remained matched. The fp32 test is the stronger semantic reference; the bf16
test proves the optimized execution path, not bitwise logit equivalence for
every prompt.

## Matched measurement

The retained run used Qwen3-0.6B, bf16, FlashInfer 0.6.14, PyTorch
2.13.0+cu130, CUDA 13.0, and two RTX A6000 GPUs. Both headline conditions
used CUDA graphs, prefix caching was disabled, each request produced 32
tokens, and the same 16 prompts arrived together. The table reports the
median of three runs after warmup; ranges show the three retained values.

```bash
PYTHONPATH=$PWD .venv/bin/torchrun --standalone --nproc-per-node=2 \
  examples/bench_tp.py --requests 16 --max-new-tokens 32 \
  --pool-tokens 8192 --repeats 3
```

The machine-readable result is retained in
[`m6a-tp2-a6000-2026-08-28.json`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m6a-tp2-a6000-2026-08-28.json).

| condition | goodput | TTFT p50/p99 | ITL p50 | ITL p99 |
|---|---:|---:|---:|---:|
| TP1 | 2487 tok/s (2434–2504) | 27.0 / 27.0 ms | 5.74 ms | 6.58 ms (5.89–7.14) |
| TP2 | 1945 tok/s (1851–1957) | 35.1 / 35.1 ms | 7.34 ms | 7.71 ms (7.54–9.12) |

TP2 was 22% slower in goodput, with 30% higher median TTFT and 28% higher
median ITL. For this small model, 56 all-reduces per forward cost more than
halving the sharded matrix work saves. A slower result is the expected lesson,
not a failed milestone.

Peak PyTorch memory was measured after graph warmup so it includes weights,
the paged pool, FlashInfer workspaces, persistent graph buffers, and captured
graph allocations:

| condition | rank 0 reserved | rank 1 reserved | total reserved |
|---|---:|---:|---:|
| TP1 | 3568 MiB | — | 3568 MiB |
| TP2 | 2712 MiB | 2672 MiB | 5384 MiB |

TP reduced per-rank reserved memory by about 24–25%, not 50%. Q/K/V, o,
gate/up, down, and KV heads are sharded, but embeddings, the tied LM head,
norms, graph buckets, and runtime workspaces are replicated. Total reserved
memory rose by 51% because two processes duplicate that fixed state. TP adds
*per-device capacity* by partitioning selected tensors; it does not promise
lower cluster-wide memory.

The script also runs a separate eager attribution case with two requests and
four generated tokens. It observed 224 all-reduces on each rank—four forwards
times 28 layers times two collectives—with 1.531 MiB of logical tensor payload
per rank. CUDA-event sums were 19.2 ms on rank 0 and 20.3 ms on rank 1. Those
times intentionally synchronize every collective and therefore must not be
subtracted from the graph-replay headline latency. “Payload” means bytes in
the tensors handed to all-reduce per rank; actual link traffic depends on the
collective algorithm.

## What M6a teaches

Tensor parallelism is a placement contract, not a flag that makes inference
faster:

- column-parallel projections create independent local features;
- row-parallel projections define the exact sum boundary;
- the residual stream stays replicated while weights and KV heads shard;
- one control-plane owner prevents scheduler divergence;
- optimized kernels must plan from local, not global, tensor shapes;
- CUDA graphs may capture NCCL, but graph lifetime must end before process-
  group lifetime; and
- per-rank memory can fall while total memory and latency both rise.

M6b changes the partition axis. Pipeline parallelism gives each rank whole
consecutive layers and sends activations between stages. That removes the two
all-reduces per layer, but introduces pipeline bubbles and request-microbatch
ordering—the next set of invariants to make visible.
