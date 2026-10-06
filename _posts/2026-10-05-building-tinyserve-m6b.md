---
layout: post
math: true
title: "Building tinyserve M6b: Pipeline parallelism: split the layer stack"
date: 2026-10-05 08:01:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Split the layer stack, move microbatch activations, and account for pipeline bubbles and tied weights."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m6b-pipeline-parallelism.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M6a — Tensor parallelism: split every layer]({% include tinyserve-post-url.html slug="building-tinyserve-m6a" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m6a-tensor-parallelism.md" %}) · Next: [M6c — Expert parallelism: route the token, not the whole model]({% include tinyserve-post-url.html slug="building-tinyserve-m6c" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m6c-expert-parallelism.md" %})

Tensor parallelism made both GPUs cooperate inside every transformer layer.
That is useful when one layer is too large, but it also paid 56 all-reduces per
Qwen3-0.6B forward. Pipeline parallelism chooses a different boundary:

> Give each GPU consecutive layers, and move the hidden state only when it
> crosses from one stage to the next.

This milestone builds PP=2 as a separate mode. Rank 0 owns the embedding and
layers 0–13. Rank 1 owns layers 14–27, the final normalization, and the LM
head. Each GPU holds only its stage's weights and KV layers. Rank 0 remains the
only scheduler, sampler, and output owner.

The split adds model capacity, but it creates a utilization problem. One input
first uses stage 0 while stage 1 waits, then uses stage 1 while stage 0 waits.
Request microbatches can overlap those two periods. The important result from
this small model is that overlap alone is not a speedup: making microbatches
too small kept both stages busier while making the request slower.

## One model, two stage owners

Qwen3-0.6B has 28 decoder layers. The PP=2 partition is deliberately concrete:

```text
rank 0 / stage 0                    rank 1 / stage 1
----------------                    ----------------
token embedding                     layers 14..27
layers 0..13                        final RMSNorm
rank-0 scheduler and sampler        LM head
KV for local layers 0..13           KV for local layers 14..27
```

The model's hidden width is 1024. For an input tensor `[B,T]`, stage 0 produces
an activation `[B,T,1024]` and sends it to rank 1. Stage 1 finishes the model
and returns only the last position's logits `[B,1,151936]` to rank 0.

[![Rank ownership, activation and logit transfers, and one-versus-four-microbatch timelines](/assets/tinyserve/m6b-pipeline-parallelism.svg)](/assets/tinyserve/m6b-pipeline-parallelism.svg)

*Figure 1. The top panel shows stage-owned weights and KV. The bottom compares
the fill/drain bubbles for one microbatch with the overlapped steady portion
for four. More colored stage slots mean more overlap, not necessarily more
useful work per second.*

This is not two replicas. A parameter inspection shows that rank 0 has no
layer-14 weights or LM head, while rank 1 has no embedding or layer-0 weights.
The module list keeps the global layer numbers, however, so checkpoint names
remain `model.layers.0...model.layers.27`. Non-owned entries are parameterless
identities that the stage forward never visits.

### The tied-weight corner case

Qwen3-0.6B ties the token embedding and LM-head weights. Its checkpoint stores
the tensor under `model.embed_tokens.weight`; the ordinary single-GPU loader
aliases that tensor as `lm_head.weight`.

PP places those two consumers on different GPUs, so a cross-device parameter
alias is impossible. The loader gives rank 0 the embedding and loads a second
copy of the same checkpoint tensor as rank 1's LM head. Every other parameter
belongs to exactly one stage. This endpoint duplication is part of the chosen
partition, not accidental full-model replication.

## Global layer names, local KV indices

The scheduler still reasons about one logical sequence and one block table.
Both ranks therefore allocate the same physical block IDs in mirrored local
pools. What differs is the leading layer dimension:

```text
single GPU pool: [28, blocks+1, 2, 16, 8, 128]
PP rank 0 pool:  [14, blocks+1, 2, 16, 8, 128]
PP rank 1 pool:  [14, blocks+1, 2, 16, 8, 128]
```

Rank 1's global layer 14 writes local cache layer 0; global layer 27 writes
local cache layer 13. Attention therefore carries both identities:

- `layer_idx` keeps the checkpoint/model layer number;
- `cache_layer_idx` selects a stage-local KV slice.

For a sequence block table `[3,9]`, both stages address blocks 3 and 9. Rank 0
finds the first half of the model's KV there; rank 1 finds the second half.
Allocation, prefix adoption, preemption, and release remain mirrored because
rank 0 broadcasts the exact scheduler plan before either stage runs tensor
work.

## A microbatch is a group of request rows

Pipeline microbatching here does not mean cutting one prompt into token
chunks. Chunked prefill already owns that job in M5a. M6b splits the **rows of
one scheduled forward**.

Suppose decode selected four requests and `pp_microbatch_size=1`:

```text
scheduled batch rows: [request 7, request 2, request 9, request 4]

MB0 = [request 7]
MB1 = [request 2]
MB2 = [request 9]
MB3 = [request 4]
```

Rank 0 executes stage 0 for MB0 and sends its hidden state. While rank 1 runs
stage 1 for MB0, rank 0 can run stage 0 for MB1. Each returned logits buffer is
stored in its microbatch slot and concatenated in the original row order. The
scheduler still sees `[7,2,9,4]`; stages never admit, reorder, or finish a
request independently.

Packed prefill uses the same row split. A single long prompt chunk has one row,
so it still has a fill and drain bubble. Pipelining token chunks from one
request is possible, but it is a different dependency schedule and is outside
this milestone.

## The point-to-point order is part of correctness

Two ranks cannot issue sends and receives in arbitrary order. If both wait to
receive first, neither sends. If a rank reuses an activation buffer before its
asynchronous send completes, its peer may read changed data.

Tinyserve uses this alternating protocol for every microbatch `i`:

```text
rank 0                                      rank 1
------                                      ------
run stage 0 for MB_i
isend activation_i  ----------------------> irecv activation_i; wait
post irecv logits_i                         run stage 1 for MB_i
run stage 0 for MB_i+1                      isend logits_i -------------->
```

The send tensors and receive buffers stay alive until every NCCL work handle
completes. Because the two processes drive different GPUs, stage 0 of the next
microbatch can overlap stage 1 of the current one. P2P counters make the data
movement observable instead of hiding it inside a framework runtime.

## The surprisingly large return message

The forward activation sounds like the important transfer, but decode reverses
that intuition for this model. In bf16, per request row:

```text
stage-boundary activation: 1 × 1024 × 2 bytes    =   2.00 KiB
last-token logits:         1 × 151936 × 2 bytes = 296.75 KiB
```

Returning full logits costs about 148 times more bytes than sending the decode
activation. We keep it because the M6 control-plane contract says rank 0 owns
sampling. A more performance-oriented design could sample on the last stage
and return one token ID, but that would move a policy boundary and hide this
tradeoff.

Prefill is different: its activation transfer grows with `T`, while
`last_token_only=True` still returns one logits row. The benchmark records both
directions rather than calling all P2P bytes "activation communication."

## FlashInfer stays local; CUDA graphs do not

Each PP stage has a local FlashInfer wrapper planned against its own 14-layer
KV tensor. Decode block tables are sliced by microbatch and replanned before
that stage processes the rows. No attention communication crosses the stage
boundary.

CUDA graphs are intentionally disabled for PP. The M5c/M6a graph runner
captures one end-to-end model forward, whereas PP needs asynchronous P2P and a
variable number of row microbatches. Capturing that schedule is useful later,
but it is not required to teach stage ownership and bubbles. The matched PP
measurement therefore uses eager FlashInfer for both the one-GPU and PP2
conditions.

## Correctness checks

The acceptance ladder separates partition correctness from optimized kernels:

1. CPU tests split a four-layer toy model, inspect exact stage-owned parameter
   names, check local cache indices, and prove that the final stage restores
   the tied LM-head tensor from the embedding checkpoint key.
2. A two-GPU fp32 test compares rank 0's output after layer 13 with the same
   stage cut through an unpartitioned model.
3. The same test compares greedy `generate_paged()` and `serve()` tokens, then
   combines chunked prefill, shared-prefix adoption, and forced recompute
   preemption against the single-GPU reference.
4. A bf16 FlashInfer test compares one large microbatch with one-row
   microbatches on controlled prompts and observes real P2P traffic and stage
   compute events.

The strict tests also check that each rank owns 14 transformer layers and 14
KV slices, rank 1 returns no user output, and the microbatch counter exceeds
the forward counter when rows are split.

## Matched A6000 measurement

The retained run used two RTX A6000 GPUs, PyTorch 2.13.0+cu130, bf16,
FlashInfer 0.6.14, eager execution, prefix caching off, eight requests, 16
generated tokens per request, and three repetitions. The one-GPU baseline and
every PP2 row used the same prompts and serving path. Values are medians; stage
busy fractions come from a separate instrumented pass because CUDA-event
profiling perturbs the headline latency. With only eight requests, the p99
columns describe this workload rather than estimating a production tail.

| condition | rows / microbatch | goodput tok/s | TTFT p50 ms | ITL p50 / p99 ms | stage busy r0 / r1 |
|---|---:|---:|---:|---:|---:|
| one GPU | 8 | 320.4 | 26.5 | 24.8 / 25.5 | n/a |
| PP2, 1 microbatch | 8 | 293.3 | 19.6 | 26.8 / 28.5 | 46.4% / 45.2% |
| PP2, 2 microbatches | 4 | 200.8 | 32.9 | 40.0 / 42.3 | 57.5% / 56.1% |
| PP2, 4 microbatches | 2 | 112.0 | 57.2 | 66.2 / 115.0 | 66.4% / 64.3% |
| PP2, 8 microbatches | 1 | 64.2 | 112.1 | 114.1 / 200.1 | 82.3% / 80.5% |

The one-microbatch PP2 path reduced per-rank peak reserved memory from 2098
MiB to 1156 MiB, about 45%. Total reserved memory rose to 2312 MiB because two
CUDA contexts, stage workspaces, and duplicated endpoint state cost memory.
That is the PP capacity trade: lower maximum memory per device, not lower
cluster-wide memory.

The microbatch sweep is the more important result. From one to eight
microbatches, measured stage busy fractions rose from roughly 46%/45% to
82%/81%, yet goodput fell from 293 to 64 tokens/s. Smaller row batches made
the layer kernels less efficient and fragmented the same 37.48 MiB logical P2P
payload into 128 sends per direction instead of 16. "GPU busy" is not the same
as "GPU doing efficient work."

All eight one-microbatch rows matched the one-GPU greedy tokens. The 2/4/8
microbatch bf16 rows matched six of eight. The strict fp32 and controlled bf16
tests pass; changing GEMM batch shapes changes bf16 kernel accumulation and can
flip near-tied logits. The benchmark records this boundary instead of treating
its broad prompt set as a parity test.

The complete raw artifact is
[`m6b-pp2-a6000-2026-08-28.json`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m6b-pp2-a6000-2026-08-28.json).

## Where the code lives

- [`distributed.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/distributed.py) defines consecutive layer
  partitions and accounts asynchronous P2P transfers.
- [`loader.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/loader.py) filters checkpoint tensors to the
  stage-owned state dictionary and restores the tied endpoint weight.
- [`qwen3.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/models/qwen3.py) keeps global layer names while
  running only the local range and mapping global layers to local KV indices.
- [`pipeline.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/pipeline.py) slices request rows, executes the
  alternating send/receive protocol, restores row order, and measures stage
  compute time.
- [`engine.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/engine.py) keeps rank 0's existing scheduler and
  routes every paged forward through the local, TP, or PP execution path.
- [`bench_pp.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/examples/bench_pp.py) runs the matched memory/latency/
  microbatch sweep.

The direct serving example is:

```bash
.venv/bin/torchrun --standalone --nproc-per-node=2 examples/generate.py \
  --pp-size 2 --pp-microbatch-size 1 --backend flashinfer --serve \
  --prompts "2+2=" "Capital of Japan?" "Prime above 100?" "Hello in French?" \
  --max-new-tokens 16
```

Use `--backend gather --no-cuda-graphs` while stepping through the stage cut
and P2P calls. The matched sweep is available through `examples/bench_pp.py`.

## What M6b establishes

M6b is intentionally PP=2 and mutually exclusive with TP. It establishes the
mechanism needed before composition is worth discussing:

- stage-owned weights and per-layer KV capacity;
- one rank-0 control plane over stage-local tensor workers;
- visible bidirectional P2P transfers;
- preserved request identity across asynchronous row microbatches; and
- measured fill/drain bubbles versus microbatch efficiency.

M6c changes a different axis. It will introduce a real MoE path, route tokens
to expert owners with all-to-all, and restore their original order after expert
computation.
