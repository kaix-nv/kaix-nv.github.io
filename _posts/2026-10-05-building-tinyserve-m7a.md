---
layout: post
math: true
title: "Building tinyserve M7a: Before optimizing, name the phase"
date: 2026-10-05 08:04:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Separate prefill, decode, host overhead, and graph replay before choosing an optimization."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m7-phase-profiling.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M6d — Context parallelism: split the KV, not the answer]({% include tinyserve-post-url.html slug="building-tinyserve-m6d" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m6d-context-parallelism.md" %}) · Next: [M7b — Pack prompts without padding]({% include tinyserve-post-url.html slug="building-tinyserve-m7b" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7b-ragged-packed-prefill.md" %})

Tinyserve ended M6 with a useful but incomplete result. On Qwen3-8B, its
decode throughput reached about 84–85% of FreeToken and vLLM on the same RTX
A6000. Prefill lagged by more, and changing only the chunk budget could improve
batch-32 throughput while making batch-8 TTFT worse. The
[cross-engine calibration](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/cross-engine-a6000-2026-08-29.md) told us
where the visible gaps were. It did not tell us which line of the engine owned
them.

Replacing attention at this point would be guessing. End-to-end latency mixes
Python scheduling, tensor construction, forward kernels, cache writes,
sampling, and synchronization. M7 therefore begins with attribution rather
than an optimization.

<a href="/assets/tinyserve/m7-phase-profiling.svg"><img src="/assets/tinyserve/m7-phase-profiling.svg"
     alt="One engine step shown as a host lane and CUDA stream, with scheduler, decode, sampling, prompt packing, prefill, and nested KV-write timing regions"></a>

## Why one wall-clock timer lies

CUDA launches are asynchronous. Consider:

```python
t0 = time.perf_counter()
logits = model(input_ids)
elapsed = time.perf_counter() - t0
```

Without a synchronization, `elapsed` mostly measures how long Python took to
enqueue kernels. The kernels may still be running. Adding
`torch.cuda.synchronize()` makes the number honest, but doing that after every
small region changes the schedule we are trying to observe.

The profiler records two clocks:

- `host_ms` is wall time inside the Python region. It includes tensor setup,
  dispatch, and any synchronization forced by converting a GPU token to a
  Python integer.
- `device_ms` comes from CUDA events recorded on the current stream. Events are
  resolved by one synchronization after `serve()` completes.

Neither column replaces the other. A region can have low host time and high
device time because Python returned after launching asynchronous work. Sampling
can show the reverse: argmax is cheap on the GPU, but asking for the integer
waits for the logits that produced it.

## The smallest useful interface

Profiling is opt-in and does not change the return type:

```python
outputs = llm.serve(
    prompts,
    params=params,
    chat=False,
    profile=True,
)
report = llm.last_serve_profile
```

`PhaseProfiler.region(name)` records one host interval and, on CUDA, one event
pair. Repeated regions aggregate into a call count and total time:

```json
{
  "wall_ms": 64.78,
  "phases": {
    "decode_forward": {
      "calls": 7,
      "host_ms": 2.27,
      "device_ms": 36.83
    }
  }
}
```

The report also records request count, chunk size, requested and resolved
decode backend, and the number of actual CUDA-graph replays. Those fields keep
the phase totals attached to the execution mode that produced them.

The serving path records six concepts:

| region | boundary |
|---|---|
| `scheduler` | arrivals, admission, decode reservation, and bounded prefill planning |
| `decode_forward` | feed-tensor construction, page metadata, and one eager forward or graph replay |
| `sampling` | grouped sampling for decode tokens or first tokens |
| `prefill_pack` | left-padded IDs, physical slots, logical positions, and mask construction |
| `prefill_forward` | one whole, packed, or chunked prompt forward |
| `paged_kv_write` | one layer's K/V scatter into physical slots |

The last region is nested inside `prefill_forward` or eager
`decode_forward`. It is a diagnostic subtotal, not another phase to add to the
parent. Adding both would count the same GPU work twice.

## What CUDA graphs hide

M5c captures the complete decode model. Replay exposes one honest
`decode_forward` event interval, but its internal layer launches no longer pass
through Python. That means a profiled graph replay does not emit per-layer
`paged_kv_write` regions for decode.

This is a boundary, not missing data disguised as zero:

- use the normal graphed path to measure the serving configuration users run;
- use `--no-cuda-graphs` when the question specifically requires eager
  per-layer KV-write attribution;
- never compare eager subphase totals with graphed end-to-end latency as if
  they described the same execution path.

M7a keeps profiling single-GPU. Distributed ranks need rank-labelled clocks,
clock-alignment rules, and collective intervals. Mixing that larger problem
into the first attribution slice would make the simple report ambiguous.

## Reading one concrete profile

The runnable example warms the model and graph path first, then profiles the
second request:

```bash
PYTHONPATH=$PWD .venv/bin/python examples/profile_serve.py \
  --model /home/scratch.kaix_coreai/models/Qwen3-0.6B \
  --max-new-tokens 8 \
  --prompts "Question: What is 2 + 2? Answer with only the numeral. Answer:"
```

One RTX A6000 smoke run produced:

| region | calls | host total | device total |
|---|---:|---:|---:|
| prefill forward | 1 | 25.61 ms | 26.00 ms |
| decode forward | 7 | 2.27 ms | 36.83 ms |
| sampling | 8 | 35.81 ms | 0.80 ms |
| scheduler | 18 | 0.59 ms | 0.31 ms |
| paged KV write, nested | 28 | 3.18 ms | 2.36 ms |

The complete call took 64.78 ms. Prefill plus decode device totals account for
62.83 ms. The 35.81 ms sampling host total is not 35 ms of sampling math: each
token conversion is a synchronization point waiting for the preceding forward.
This is exactly why the two clocks stay separate.

For an exact calibration shape, the shared benchmark runner retains one phase
report inside every measured run:

```bash
PYTHONPATH=$PWD .venv/bin/python examples/bench_cross_engine.py \
  --adapter tinyserve \
  --model /home/scratch.kaix_coreai/models/Qwen3-8B \
  --tokenizer /home/scratch.kaix_coreai/models/Qwen3-8B \
  --prompt-tokens 128 --output-tokens 128 --batch-size 1 \
  --phase-profile --output /tmp/m7-decode-profile.json
```

That mode is for attribution. The feature-level throughput comparison omits
`--phase-profile` so event recording cannot improve or regress the headline by
measuring it.

The profile is a single integration smoke run, not a performance claim. The
feature gate still uses the fixed five-repeat Qwen3-8B cross-engine protocol.

## What Qwen3-8B actually spends

Two exact-token profiles used candidate commit `17f2fb9`, two warmups, and
three measured repetitions. The decode-heavy shape was `p128/o128/b1` at the
512-token budget. The prefill-heavy shape was `p2048/o32/b32` at the
8,192-token budget.

For decode-heavy B1:

| region | calls | median device total | share of 3,589.53 ms wall time |
|---|---:|---:|---:|
| decode forward | 127 | 3,518.08 ms | 98.0% |
| prefill forward | 1 | 36.53 ms | 1.0% |
| sampling | 128 | 16.73 ms | 0.5% |
| scheduler | 258 | 3.65 ms | 0.1% |
| paged KV write, nested in prefill | 36 | 0.82 ms | do not add |

Every decode used graph replay. Sampling recorded 3,471.93 ms of host-region
time but only 16.73 ms of device work. The host was waiting for the preceding
decode logits when it converted the sampled token to Python. Moving the
scheduler or rewriting argmax cannot close a decode gap whose device replay
already owns 98% of the call.

For prefill-heavy B32:

| region | calls | median device total | share of 14,232.85 ms wall time |
|---|---:|---:|---:|
| prefill forward | 8 | 12,578.02 ms | 88.4% |
| decode forward | 38 | 1,606.54 ms | 11.3% |
| prefill pack | 8 | 27.07 ms | 0.2% |
| sampling | 46 | 10.80 ms | 0.1% |
| scheduler | 87 | 3.15 ms | less than 0.1% |
| paged KV write, nested | 288 | 53.73 ms | do not add |

The model forward, not token packing or physical cache scatter, is the prefill
bottleneck. The code supplies a concrete mechanism: packed prefill builds an
explicit `[B, 1, P, P]` mask. `_paged_attend()` then expands GQA K/V heads and
uses masked SDPA. Even this equal-length calibration batch pays for the dense
mask and expansion although it contains no real padding.

This attribution does not yet prove which replacement kernel wins. It is
enough to reject scheduler, packing, and KV scatter as the first optimization
targets.

## Feature-level calibration: deliberately no speedup

After profiling was disabled, candidate `17f2fb9` reran the complete
Qwen3-8B suite at both chunk budgets. Each shape retained two warmups and five
measured repetitions. The same GPU started each suite idle at 50–51 degrees C.

Decode rows compare output throughput; higher is better:

| budget | batch | M6d baseline | M7a candidate | change |
|---:|---:|---:|---:|---:|
| 512 | 1 | 35.55 | 35.70 tok/s | +0.4% |
| 512 | 8 | 260.28 | 262.39 tok/s | +0.8% |
| 512 | 32 | 794.44 | 798.22 tok/s | +0.5% |
| 8,192 | 1 | 35.48 | 35.76 tok/s | +0.8% |
| 8,192 | 8 | 263.24 | 265.46 tok/s | +0.8% |
| 8,192 | 32 | 846.07 | 855.42 tok/s | +1.1% |

Prefill rows compare median TTFT; lower is better:

| budget | batch | M6d baseline | M7a candidate | change |
|---:|---:|---:|---:|---:|
| 512 | 1 | 493.28 | 495.24 ms | 0.4% worse |
| 512 | 8 | 2,943.15 | 2,948.23 ms | 0.2% worse |
| 512 | 32 | 10,422.75 | 10,398.98 ms | 0.2% better |
| 8,192 | 1 | 369.54 | 369.45 ms | unchanged |
| 8,192 | 8 | 3,228.97 | 3,233.52 ms | 0.1% worse |
| 8,192 | 32 | 8,143.42 | 8,152.80 ms | 0.1% worse |

These small changes are an unchanged result, not an instrumentation speedup.
The largest movement is 1.1%, while the runs were not opposite-order trials.
Peak memory did not regress: the 512-budget range moved from
35,929–35,977 MiB to 35,927–35,975 MiB, and the 8,192-budget range from
35,929–37,545 MiB to 35,927–37,447 MiB. Sustained rows reached 87 degrees C;
their retained clocks and temperatures keep small differences from being
overinterpreted.

The feature therefore passes its real gate: profiling-disabled behavior is
neutral, while profiling-enabled runs expose a dominant path. The structured
[M7a evidence](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7a-phase-profile-a6000-2026-08-29.json) retains the
baseline deltas, phase medians, memory/thermal envelopes, and SHA-256 hashes
for all raw artifacts.

## Correctness and measurement boundaries

The profiler itself has CPU aggregation and CUDA-event tests. Existing serving
tests continue to cover text and token parity because `profile=False` leaves
the model, scheduler, cache, and return values unchanged. A real 0.6B run
additionally verifies graph warmup, JSON serialization, nested prefill events,
and output completion together.

CUDA events add launch overhead when profiling is enabled. Profiled numbers are
for attribution, not headline throughput. The ordinary cross-engine calibration
runs with profiling disabled and checks whether even the dormant branches moved
throughput, TTFT, memory, or thermals.

## What this milestone can decide

M7a can now answer questions that the end-to-end table could not:

- Is batch-32 prefill dominated by packing, the model forward, or cache writes?
- Does batch-1 decode spend its gap in graph replay or host control?
- Is a slow host region real Python work, or a synchronization waiting for an
  earlier CUDA phase?
- Which eager subphase should receive the first profiler trace or kernel change?

It cannot prove that a replacement kernel is faster. That needs a named
intervention followed by parity and the same feature-level cross-engine gate.

The first evidence-backed intervention is M7b: extract packed prefill behind
an attention-backend seam and add a variable-length fused path that avoids the
dense padded mask and copied GQA heads. Acceptance requires packed/chunked
prefill parity, a smaller `prefill_forward` device total on the exact B32
profile, and improvement in the unprofiled cross-engine suite without trading
away B1/B8 TTFT. Decode kernel attribution follows separately because graph
replay needs a kernel trace rather than more scheduler timing.

## Takeaway

Optimization starts with a causal claim: hold scheduler policy constant,
change one measured path, and observe the expected metric. M7a supplies the
missing first clause by turning one opaque `serve()` duration into explicit
host and device regions.

The Qwen3-8B profiles identify the first bottleneck: packed prefill spends
12.58 of 14.23 seconds inside the forward, while packing and cache writes are
small. M7b can now make the attention seam concrete around that measured path
rather than refactoring the entire engine in anticipation of future research.
