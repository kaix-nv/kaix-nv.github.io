---
layout: post
math: true
title: "Building tinyserve M5c: CUDA graph replay"
date: 2026-08-23 18:13:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Capture fixed-shape decode forwards, replay them through persistent buffers, and fall back safely when dynamic page metadata no longer fits."
---

*Milestone 5, part 2, of [building an LLM inference engine from scratch](/series/tinyserve/).
Previous: [M5a — chunked prefill]({% post_url 2026-08-23-building-tinyserve-m5a %}).*

Code: [`tinyserve` @ `1967b96`](https://github.com/kaix-nv/tinyserve/tree/1967b9604bb9df53f0e03e35231f168f0cbf91d8).

> The M5 letters name roadmap topics, while the implementation landed in
> the order M5a → M5c → M5b. Prefix caching therefore appears after this
> post even though its topic label is M5b.

## The problem: small decode work, repeated launch work

M1 observed a stubborn result: one Qwen3-0.6B decode forward took about
21 ms over a wide range of context lengths. A simple HBM-traffic estimate
put the ideal lower bound near 3 ms on the measured GPU. That estimate was
not a profiler decomposition, but the large, nearly constant gap suggested
that the engine was spending much of each step outside the useful matrix and
attention work.

Why can a small decode be expensive? A decoder layer is written as many
PyTorch operations: normalization, projections, RoPE, attention, residuals,
and MLP operations. Eager execution asks the host to dispatch the resulting
GPU kernels again for every generated token. At batch size one, many kernels
are short enough that Python, framework dispatch, and launch gaps can be a
large part of the wall time.

M5a exposed the same fixed-cost floor from another direction. Splitting a
long prefill into more chunks eventually stopped helping, because every new
forward paid that floor again. The retained measurements did not separate
Python, framework, launch, and GPU execution time, so the question for M5c
is deliberately narrower:

> If we replay the same decode kernel graph instead of redispatching the
> model operation by operation, how much of the observed floor disappears?

## The idea: record a fixed forward, then replay it

A CUDA graph records the GPU work launched by one forward pass, including
kernel order and dependencies. Later steps replay that recorded graph with
one host-side replay call.

Replay does **not** make the model one kernel, and it does not remove all CPU
work. Tinyserve still copies new values into input buffers, refreshes
FlashInfer's page-table metadata with `plan()`, invokes replay, and samples
the output. What disappears is the repeated eager dispatch of the captured
model forward.

The restriction is that capture freezes both tensor shapes and memory
addresses. Tinyserve pays for that rigidity in four small mechanisms:

1. **Batch-size buckets.** Graphs are captured for
   `{1, 2, 4, 8, 16, 32, 64}`. A live batch uses the smallest bucket that
   fits. Batch 3 therefore replays the batch-4 graph.
2. **Dummy rows.** Unused bucket rows receive token 0 at position 0 and use
   the KV pool's scratch block. Their outputs are discarded.
3. **Persistent buffers.** Token IDs, positions, write slots, FlashInfer CSR
   metadata, and logits keep the same addresses for the graph's lifetime.
   Each step changes their contents with `copy_`, not their identity.
4. **Safe eager fallback.** Batch sizes above 64 run eagerly. Prefix sharing
   can also make the flattened page-reference table larger than a captured
   metadata buffer; that case now falls back instead of overflowing the
   graph's fixed storage.

![CUDA graph replay uses fixed buffers and a padded batch-size bucket](/assets/tinyserve/m5c-cuda-graph-replay.svg)

*Figure 1. A live batch of three rows selects the captured batch-4 graph.
The three real rows and one scratch row write into persistent buffers;
FlashInfer refreshes its fixed metadata buffers, then the captured model
forward replays. Only the first three logits are returned.*

Prefill remains eager in this milestone. Prompt shapes vary far more than
decode shapes, and prefill usually contains larger kernels. Graphing prefill
is possible, but it would add another bucketing problem without teaching a
new invariant here. This eager-prefill/graph-decode split is also common in
larger serving engines, but tinyserve implements only the minimal version
shown above.

## The build

[`graph_runner.py`](https://github.com/kaix-nv/tinyserve/blob/1967b9604bb9df53f0e03e35231f168f0cbf91d8/tinyserve/graph_runner.py)
owns capture and replay:

- allocate persistent buffers for each bucket;
- construct FlashInfer in CUDA-graph mode with caller-owned CSR buffers;
- warm up before capture so one-time kernel setup stays outside the graph;
- capture largest bucket first and let non-concurrent captures share a CUDA
  graph memory pool;
- copy the current step's values into the chosen bucket and replay; and
- reject a graph bucket if either its row count or its page-reference buffer
  is too small.

The engine integration remains one decision in `_paged_decode`: use graph
replay when the active batch and metadata fit; otherwise use the existing
eager FlashInfer path. Both paths consume the same slots, positions, block
tables, and context lengths.

One detail looks trivial but is a real invariant: the scratch KV block is
zeroed when the paged cache is created. Dummy rows read that block as an
attention context. Their logits are discarded, but finite scratch data keeps
NaNs from propagating through the captured forward.

Graph construction is lazy: M0–M2 paths never pay for it, and the first
paged FlashInfer call performs capture. The original run observed roughly
three seconds of first-use setup; the exact cost depends on software and
hardware.

## Walking through batch 3

Suppose three requests are ready to decode:

```text
live batch B=3
capture bucket Bc=4

row 0: real token, real position, real slot, real block table
row 1: real token, real position, real slot, real block table
row 2: real token, real position, real slot, real block table
row 3: token 0, position 0, scratch slot, [scratch block]
```

The engine copies these values into the batch-4 buffers. FlashInfer's
`plan()` refreshes `indptr`, flattened block indices, and last-page lengths
without changing their addresses. The graph replays the model forward and
produces four logit rows; tinyserve returns `logits[:3]`.

The dummy row may compute nonsense, which is fine. Correctness requires only
that it cannot read or overwrite a real request's KV and that its result is
never sampled.

## The proof—and its boundary

[`tests/test_graphs.py`](https://github.com/kaix-nv/tinyserve/blob/1967b9604bb9df53f0e03e35231f168f0cbf91d8/tests/test_graphs.py)
checks four contracts:

1. Graph and eager greedy generation produce identical text at live batch
   sizes 1, 3, and 5, covering exact buckets and padded buckets.
2. Graph serving remains text-identical to eager serving while batches
   shrink, block tables change, and a small KV pool forces recomputation
   preemption.
3. Bucket selection counts flattened page references, not merely physical
   blocks, and refuses replay when shared-prefix metadata is too large.
4. A warmed shared prefix is adopted by multiple requests and those requests
   actually decode through graph replay with the same text as a
   caching-disabled eager reference.

“Text-identical” is the precise checked-in claim. These tests do not prove
that every intermediate tensor or logit is bitwise equal, nor do they cover
every FlashInfer, CUDA, dtype, and GPU combination. That stronger statement
would need a retained raw-logit test matrix.

## The payoff: retained milestone measurements

The following tables are historical snapshots from M5c development. They
used Qwen3-0.6B in bf16 with FlashInfer, but the raw logs, exact package
versions, and repeated trials were not retained. They establish the
milestone's direction and approximate magnitude, not a portable performance
promise.

Single-token decode, context 512
([`examples/bench_decode.py`](https://github.com/kaix-nv/tinyserve/blob/1967b9604bb9df53f0e03e35231f168f0cbf91d8/examples/bench_decode.py)):

| B | FlashInfer eager | FlashInfer + graph |
|---:|---:|---:|
| 1 | 22.7 ms / 44 tok/s | **5.2 ms / 194 tok/s** |
| 8 | 23.4 ms / 341 tok/s | **6.1 ms / 1307 tok/s** |
| 32 | 24.1 ms / 1326 tok/s | **8.8 ms / 3620 tok/s** |
| 64 | 24.4 ms / 2618 tok/s | **12.3 ms / 5197 tok/s** |
| 128 | 24.2 ms / 5298 tok/s | 24.9 ms / 5138 tok/s, eager fallback |

At context 2048 and batch 128, the retained eager and graph numbers were
53.5 and 53.7 ms. Larger attention work dominates there, and batch 128 is
outside the captured range anyway.

The batch-1 reduction from 22.7 to 5.2 ms is consistent with eager dispatch
being a major part of the old floor. It does **not** prove that all 17.5 ms
was CUDA launch overhead: graph replay also changes framework execution, and
the measurement does not separately time buffer copies, `plan()`, sampling,
or GPU kernels.

Serving at Poisson arrival rate λ=32, budget 512:

| configuration | goodput | TTFT p50 | ITL p50 / p99 |
|---|---:|---:|---:|
| M5a eager snapshot | 635 tok/s | 98 ms | 27 / 111 ms |
| M5c graph snapshot | **1082 tok/s** | **54 ms** | **7 / 72 ms** |

The earlier M5a freeze microscope was also repeated after graph capture:

| budget | freeze, eager snapshot | freeze, graph snapshot | long TTFT, graph |
|---:|---:|---:|---:|
| effectively unbudgeted | 96 ms | 77 ms | 41 ms |
| 512 | 54 ms | 34 ms | 207 ms |
| 256 | 59 ms | **33 ms** | 350 ms |
| 128 | **75 ms** | **34 ms** | 688 ms |

In this snapshot, the small-budget reversal disappeared after graph replay
reduced the per-forward floor. That result supports the interaction between
chunk size and fixed forward cost. Because the old traces were not retained,
it should not be read as a complete causal decomposition.

## Reproducing the current protocol

Use the project environment; a host-level `pytest` may import the wrong
Python installation.

```bash
.venv/bin/pytest tests/test_graphs.py -q
.venv/bin/pytest tests/ -q

git rev-parse HEAD | tee m5c-decode.txt
.venv/bin/python examples/bench_decode.py \
  --ctx 512 2048 --batch 1 8 32 64 128 \
  --backends flashinfer flashinfer-graph \
  | tee -a m5c-decode.txt
```

Run serving configurations as separate processes with prefix caching disabled
so this comparison isolates graph replay:

```bash
for graphs in off on; do
  git rev-parse HEAD | tee "m5c-serve-${graphs}.txt"
  .venv/bin/python examples/bench_serve.py \
    --rates 32 --n 96 --max-new 128 --long-frac 0.1 \
    --chunk-size 512 --seeds 0 1 2 \
    --cuda-graphs "${graphs}" --prefix-caching off \
    | tee -a "m5c-serve-${graphs}.txt"
done
```

The scripts print model, GPU, dtype, Python, PyTorch, CUDA, FlashInfer, pool
size, and protocol parameters. Preserve those files and report variability
across seeds before making a hardware-level performance claim. These commands
reproduce the current protocol; they cannot recreate the historical tables
bit for bit because the original raw environment was not retained.

## What M5c teaches

CUDA graphs are not “make CUDA faster.” They exchange flexibility for lower
dispatch overhead:

- shapes become buckets;
- addresses become persistent buffers;
- ragged requests need explicit dummy rows and scratch storage;
- dynamic attention metadata must update without changing buffer identity;
- features such as prefix sharing can enlarge metadata even when physical KV
  usage stays small; and
- eager fallback is part of correctness, not an embarrassment.

The most useful lesson is methodological: a flat timing floor suggests a
fixed cost, but an intervention and matched benchmark can only show what
changed—not automatically decompose every microsecond that disappeared.

Next: [M5b — prefix caching]({% post_url 2026-08-23-building-tinyserve-m5b %}).
