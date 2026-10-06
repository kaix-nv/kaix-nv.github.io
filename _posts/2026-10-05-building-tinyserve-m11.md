---
layout: post
math: true
title: "Building tinyserve M11: Move the KV, not the prompt work"
date: 2026-10-05 08:31:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Move KV between two model owners; compare the host-staged implementation against two ordinary replicas."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m11-prefill-decode.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M10d — Batch the filtering, keep the randomness private]({% include tinyserve-post-url.html slug="building-tinyserve-m10d" fallback="https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/m10d-batched-sampling.md" %}) · Next: [M12 — Different requests, one model forward]({% include tinyserve-post-url.html slug="building-tinyserve-m12" fallback="https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/m12-mixed-prefill-decode.md" %})

**Status: complete for the bounded two-local-GPU dense-Qwen path. Correctness
and lifecycle checks pass; this implementation is slower than two ordinary
replicas on the measured A6000 workloads and remains opt-in.**
This milestone separates prefill and decode onto two local GPUs. It is a small
educational serving path, not a distributed deployment framework.

## 1. The problem that remains after chunking

M10b's network adapter preserves continuous batching. But an HTTP connection
does not get its own GPU: prompt processing and decoding still share one
execution timeline. With eight clients, the measured 2,048-token workload
reached 103.75 output tokens/s while cohort-median first-text latency was
1,792.7 ms. Those finite-cohort numbers motivate an experiment, not a diagnosis
that disaggregation must win.

Chunked prefill bounds prompt work between decode opportunities. In Tinyserve's
single-stream engine, the two phases still run serially. M11 asks a different
question: what if one complete model replica handles prompt work while another
complete replica keeps advancing established decoders?

This is **not pipeline parallelism**. M6b divides layers between devices;
M11 duplicates every layer and divides the request's lifetime. Neither is it
tensor parallelism: no layer's matrix multiplication is split here.

[![Two independent model owners exchange KV, with an acknowledgement before source-page release](/assets/tinyserve/m11-kv-handoff.svg)](/assets/tinyserve/m11-kv-handoff.svg)

## 2. What crosses the boundary?

The decoder needs the attention history computed by prefill, not the original
prompt recomputed on its own GPU. A handoff contains:

- The real prompt K/V for every layer, in logical token order.
- The first sampled token, which has **not** entered KV yet.
- Request identity, prompt length, output budget, and cancellation state.

Physical page numbers are local to each allocator. Copying the source block
table would point the decoder at unrelated memory. Instead, pack only valid
token positions, copy that packed tensor to the destination device, and scatter
it through the destination's independently allocated block table.

The transfer tensor has shape `[layers, 2, prompt_tokens, kv_heads, head_dim]`.
Axis 1 selects K or V. Padding in the final page and the graph scratch page
are not transferred. Its payload size is:

```text
bytes = layers × 2 × prompt_tokens × kv_heads × head_dim × bytes_per_element
```

For Qwen3-4B BF16, 36 layers, 8 KV heads, and head dimension 128 mean 144 KiB
per prompt token. A 2,048-token prompt moves **288 MiB** of KV, before gather,
scatter, allocation, and synchronization costs. That is not a free handoff.
The two A6000s on this host report a PCIe NODE connection, not NVLink.
M11 stages the payload through host memory: the 288 MiB payload crosses a
device-to-host leg and a host-to-device leg. The receipt's `transfer_bytes`
counts the logical payload once, not the sum of those two legs.

## 3. A six-token example

Use a toy block size of four; the real engine uses sixteen. A prompt has six
tokens and requests four generated tokens:

```text
prefill source pages:       [5, 2]       cache length = 6
decode reserved pages:      [9, 1, 7]    room for 6 + 4 - 1 = 9 cached tokens
packed transfer:            K/V for logical positions 0..5 only
after import:               cache length = 6; first generated token is pending
first decode forward:       writes pending token at logical position 6
                            destination page 1, offset 2; predicts output #2
last decode forward:        cache length = 9; output #4 remains uncached
```

The source's logical position 5 lives at page 2, offset 1. At the destination
the same position lives at page 1, offset 1. The tensor values are preserved;
the physical addresses are not.

There are two different lists on the decoder. Immediately after import,
`seq.block_table` is `[9, 1]`, while `job.reserved` holds `[7]`. The third page
is owned but contains no valid history. Only when the next write reaches
logical position 8 does the decoder append page 7 to the visible block table.
Passing all three pages to FlashInfer immediately would incorrectly count
the reserved page as part of attention's history.

The first sampled token is delivered after import. Therefore M11's first-text
latency includes the handoff. An alternative could emit that token on the
prefill worker, shifting the visible transfer delay into the first inter-token
gap; this prototype does not hide the cost that way.

## 4. Ownership and independent progress

One worker owns the prefill model and source pool. Another owns the decode
model and destination pool. Each constructs, calls, and closes its own LLM.
The existing model forwards and attention kernels are reused.

The decoder first reserves enough private pages for the request's complete
prompt plus output budget (minus the uncached final token). If space is not
available, the request waits in FCFS order before prefill starts. This is
deliberately more conservative than M4's incremental reservation: it prevents
decode-side eviction from secretly running a new prefill on the decode GPU.

The prefill worker processes one prompt at a time, in bounded chunks. Other
requests can continue decoding on the other worker. When prefill finishes:

1. Pack valid source K/V and publish a ready packet.
2. The decode owner copies and scatters it into its reserved pages.
3. After the destination copy completes, acknowledge the packet.
4. The prefill owner releases the source pages and starts its next prompt.

Only one packet is in flight from the prefill owner. This bounds staging
memory and makes ownership visible. The copy uses PyTorch's explicit
GPU-to-CPU-to-GPU transfer, not a custom communication kernel, CUDA IPC,
RDMA, or a remote KV service. The decode owner pauses during import; eliminating
that pause is not part of this milestone.

Why stage through the CPU? The first real GPU qualification caught a host
transport failure: CUDA reported peer access, but copying `arange(4096)`
directly from GPU 0 to GPU 1 returned zeros. Copying through the CPU preserved
all values. That isolates the failure from paging and model math; it does not
identify the underlying driver/platform cause. This prototype therefore uses
host staging explicitly, even on hosts where P2P works. A verified P2P fast
path is a future optimization, not a prerequisite for this milestone.

For a one-token budget or first-token EOS, there is no remaining decode work,
so no KV payload needs to cross devices. A zero-token budget needs no model
forward at all.

The event boundary is bounded too: at most 32 requests, with 64 pending
token events per request and a separate terminal event. A slow consumer is
cancelled explicitly rather than silently losing tokens. A completed request
continues to count against admission until its terminal event is drained.

Cancellation is a flag, not a mid-kernel interrupt. Waiting requests can
retire without computation. A cancelled prefill finishes its current chunk;
a ready cancelled packet is discarded and acknowledged. A cancelled decode
releases destination pages after its current forward. Terminal completion
must not claim cleanup while either stage still owns the request's pages.

## 5. Follow the implementation

The implementation lives in [`disaggregated.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/disaggregated.py):

| Boundary | Code | Important invariant |
|---|---|---|
| HTTP or local caller | `SplitLLM.serving_session()` | Both owners load the same checkpoint, config, and dtype. |
| Admission | `_decode_loop()` | Reserve the complete budget before enqueuing prefill; no decode-side preemption. |
| Prompt work | `_prefill_loop()` | One prompt at a time; existing whole-prompt or chunked forwards. |
| Export | `pack_kv()` | Gather exactly `num_cached` positions; page tails are never read. |
| Import | `import_kv()` | Validate layout/dtype, copy, scatter by destination slots, synchronize, then ACK. |
| Continuation | `_paged_decode()` in `engine.py` | Cache the pending token, then predict the next token. |
| Retirement | `_stop_job()` / `_finish()` | Free visible and future pages; wait for source release before terminal completion. |

Both owners are threads, not subprocesses. This keeps the example small and
lets the handoff contain an actual tensor rather than serialized bytes. It
also means they share Python's GIL and the CUDA process context: independent
workers permit overlapping GPU work, but do not guarantee perfect overlap.
Initialization is serial because CUDA graph capture must not overlap another
owner's allocations. Each `GraphRunner` also passes an explicit device-local
capture stream: PyTorch's process-wide default capture stream otherwise
remained on GPU 0 when the second replica tried to capture on GPU 1.
Steady-state forwards remain independent. A fatal owner
failure shuts down both owners and unblocks any outstanding acknowledgement;
this is not a fault-tolerant worker-restart protocol.

Use the existing demo rather than a second generation API:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/python examples/generate.py \
  --model /home/scratch.kaix_coreai/models/Qwen3-0.6B \
  --pd --live --chunk-size 4 --prompts "Why is the sky blue?" "2+2=" \
  --no-chat --max-new-tokens 8 --dtype fp32 --backend gather \
  --no-cuda-graphs --kv-pool-tokens 1024
```

For HTTP, replace `--live --prompts ... --no-chat --max-new-tokens 8` with
`--http --port 8000`; clients provide their own raw prompt and output budget.
`--pd-devices cuda:0 cuda:1` names devices inside `CUDA_VISIBLE_DEVICES`.
Private KV is automatic in this mode. The M11 entry in
[launch.json](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/.vscode/launch.json) uses the same demo.

Useful breakpoints are `_decode_loop()` at reservation, `pack_kv()`,
`import_kv()`, and the call to `_paged_decode()`. Watch `job.state`,
`seq.num_cached`, `seq.block_table`, `job.reserved`, and `job.source_released`.
A debugger may stop all threads together: that is not a measurement of
steady-state overlap.

## 6. Correctness before performance

The local CPU tests cover FP32 greedy continuation with a small, randomly
initialized dense-Qwen model, arbitrary page
maps with poisoned tails, zero/one-token budgets, EOS, idle reuse, admission
and event limits, cancellation, injected worker failures, page reclamation,
and one-thread ownership. A gated prefill test proves that another decoder
can advance before that prefill is allowed to finish. The existing HTTP
adapter also passes completion/usage and disconnect-cleanup tests with two
owners. This is **CPU correctness evidence, not CUDA transport evidence**.

The focused M11 plus existing M10 session/HTTP suite passed **86 CPU tests**.
This includes actual loopback HTTP
execution of both benchmark modes with CPU test models, checking receipt
generation and complete page reclamation, not GPU performance.
The full regression suite with GPU 0 exposed passed **468 tests, with 13
skipped**, including the existing single-GPU graph paths. The dedicated
two-GPU checks run separately. No other GPU user's jobs were interrupted.

[`test_disaggregated_cuda.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tests/test_disaggregated_cuda.py) is opt-in
via `TINY_SERVE_PD_MODEL=/path/to/Qwen3-0.6B` on two exclusive GPUs. It checks
exact imported KV and matched single-request token IDs for FP32 gather,
BF16 FlashInfer, and BF16 decode graphs. All **four two-GPU tests pass** after
the host-staged transport and device-local capture-stream fixes. They
deliberately avoid claiming that arbitrary BF16 batch regrouping is
token-identical.

The real Qwen3-4B HTTP fixture also passes: one staggered client disconnects,
two others finish, and all references in **both** KV pools return to zero.
A three-request overload burst with admission temporarily set to two produces
two completed streams and one HTTP 429 whose JSON body also reports 429.
The exact M11 `launch.json` arguments complete all three demo requests.

Independent threads alone would not prove concurrent GPU execution. A
separate Nsight Systems capture starts **after** model initialization and
graph capture, then records the HTTP lifecycle fixture. CUPTI kernel
timestamps show 237.76 ms with kernels active on both GPUs, including
**34.70 ms of overlapping GEMM execution**. GPU 0 owns only prefill and GPU 1
only decode in this fixture. This establishes actual phase overlap, not
occupancy or a speedup. These instrumented timings are not used in the
performance table below.

## 7. Measure against the right control

[`bench_disaggregated.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/examples/bench_disaggregated.py) implements an
equal-resource comparison. The measured order was **replicas / PD / PD /
replicas**, each in a fresh process. Both modes use
two complete Qwen3-4B BF16 replicas, FlashInfer with decode graphs, private
KV, 16,384 KV tokens per GPU, a chunk budget of 8,192, and aggregate admission
capacity 32. The control routes request `i` to replica `i % 2`; it is a fixed
round-robin baseline, not an optimal load balancer. Both modes use two model
owner threads in one Python process; the control is not a tuned multiprocess
deployment and shares the same GIL constraint as the PD implementation.

The fixtures have eight requests: all short (128 prompt / 128 output tokens),
all long (2,048 / 32), and four short requests followed by four long requests
at 80 ms intervals. Each process runs two warmups and five measured cohorts
per workload: **ten measured cohorts, or 80 completed requests, per workload
and mode** across the two processes.
Receipts retain source hashes, actual dispatch times, successful completion
counts, usage counts, first-text latency, visible text-chunk gaps, throughput,
both GPUs' memory samples, other GPU process IDs, page-reclamation checks,
and execution traces. A run overlapped by a foreign GPU process is rejected.
PD additionally records packed bytes, pack/import time, decode batch sizes,
and overlapping synchronized forward intervals. Wall-clock overlap is not
an SM-utilization measurement. Finite-cohort throughput is not saturation
capacity, and a text chunk is not necessarily one model token.

For each mode, use a new output filename; the harness refuses to overwrite
an existing receipt or start while another GPU process is present. Compare
per-cohort results rather than combining different workloads into one speedup.
Import time includes both host-transfer legs, scatter, and synchronization;
it is not pure link bandwidth. Observed ranges are not confidence intervals,
and GPU clocks were not locked. One earlier attempt was discarded in full
because another process claimed a GPU during its mixed workload; none of its
timings enter the comparison.

The [portable measurement receipt](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m11-prefill-decode-a6000-2026-09-22.json)
retains all measured cohorts, source hashes, test logs, lifecycle checks,
kernel-overlap intervals, and the separate cross-engine calibration. Runs
precede the M11 commit, so per-file hashes identify the measured working-tree
source; the recorded parent revision alone is not the implementation identity.

### Result: correct separation, but not a speedup

[![Equal-budget comparison: two ordinary replicas beat the host-staged prefill/decode pair on throughput and first-text latency](/assets/tinyserve/m11-two-gpu-comparison.svg)](/assets/tinyserve/m11-two-gpu-comparison.svg)

These are medians across ten measured cohorts per cell, on two RTX A6000
48 GB GPUs. First text and chunk-gap columns are medians of the respective
per-cohort metrics, not percentiles pooled across all trials.
As in M10, the harness uses observed order statistics: for an even-sized
cohort, p50 selects the upper middle request rather than interpolating.

| Eight-request workload | Mode | Output tok/s ↑ | First text p50, ms ↓ | Visible chunk-gap p95, ms ↓ |
|---|---|---:|---:|---:|
| Short: 128 / 128 | Two replicas | 425.32 | 149.67 | 19.03 |
| Short: 128 / 128 | Prefill/decode | 357.63 | 389.40 | 18.96 |
| Long: 2,048 / 32 | Two replicas | 168.99 | 886.49 | 19.49 |
| Long: 2,048 / 32 | Prefill/decode | 62.07 | 2,258.63 | 225.21 |
| Mixed arrivals | Two replicas | 232.71 | 258.48 | 18.38 |
| Mixed arrivals | Prefill/decode | 185.44 | 719.49 | 19.91 |

The pair achieves **0.84×, 0.37×, and 0.80×** the control's throughput for
short, long, and mixed cohorts. It also delays first text. There is no
workload-level win to promote as a new default.

The execution receipt helps explain why:

| Workload | Logical KV transferred per cohort | Pack time, ms | Import time, ms |
|---|---:|---:|---:|
| Short | 144 MiB | 2.98 | 50.89 |
| Long | 2,304 MiB | 25.08 | 1,711.11 |
| Mixed | 1,224 MiB | 14.36 | 870.31 |

Times are medians of the **sum across all eight handoffs**, not per-request
latencies. Long prompts carry large histories but generate only 32 tokens:
there is little subsequent decode work over which to amortize the copy.
The decode owner cannot advance any request while it imports another one.

Transfer is not the only difference. This small implementation prefills
one prompt at a time, while the control can pack several prompts. The
prefill/decode pair also builds its decode batch gradually: in the long
fixture, measured decode calls contained only one to three requests, averaging
1.97 rows per call. Therefore the result does not isolate a transport-only
penalty or prove that all disaggregated designs are slower.

An occasional long pause can disappear from pooled p95 chunk gaps. For
example, the median of each long cohort's **worst** observed gap is 656.8 ms
for replicas and 242.1 ms for PD, despite PD's much worse p95 and total
throughput. In this fixture, PD replaces a rarer long pause with recurring
import pauses; a single tail statistic is not a serving-quality verdict.

Sampled peak device memory across the entire three-workload runs was
14,171 / 14,082 MiB for replicas and 13,281 / 12,778 MiB for PD on GPU 0 / 1.
These are NVML samples, not allocation-level peaks. Both modes have the same
KV capacity; different batch shapes and graph/workspace allocations can
still produce different resident memory.

### Keep the one-GPU calibration separate

The M10 HTTP calibration is rerun on the same source after each feature.
Its control is **one GPU**, with Qwen3-4B BF16 weights, fresh requests, two
warmups, and five measured repetitions at 1, 4, and 8 clients.
It is a regression/context check, not the equal-resource control for PD.
Tinyserve uses private KV, 32,768 KV tokens, and an 8,192-token chunk budget
in that run. Reference engines retain their native prefill/cache policies;
the [M10b protocol]({% include tinyserve-post-url.html slug="building-tinyserve-m10b" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m10b-http-streaming.md" %}#m10-closeout-concurrent-http-measurements)
explains those differences.

All 24 conditions completed. Median output tokens/s:

| Engine | Short, 1 client | Short, 4 | Short, 8 | Long, 1 client | Long, 4 | Long, 8 |
|---|---:|---:|---:|---:|---:|---:|
| Tinyserve | 60.11 | 225.11 | 415.26 | 41.80 | 85.66 | 103.82 |
| llama.cpp | 73.96 | 244.39 | 401.30 | 37.19 | 48.85 | 62.08 |
| FreeToken | 72.91 | 285.07 | 543.84 | 49.50 | 108.04 | 132.33 |
| Ollama | 74.73 | 247.14 | 414.24 | 43.89 | 60.47 | 85.63 |

Tinyserve's eight-client results are close to the prior M10 calibration:
412.23 → 415.26 tok/s for short prompts and 103.75 → 103.82 for long ones.
These small differences do **not** establish a speedup; the ordinary serving
path remains unchanged apart from the device-local graph-capture fix.

The receipt records executable hashes and launch arguments for llama.cpp,
FreeToken, and Ollama as well as the current source hashes for Tinyserve.
FreeToken returns 127 and 31 output tokens in these nominal 128- and 32-token
fixtures; throughput uses the actual counts rather than assuming the budget.
NInfer is excluded because its retained checkout targets SM120a, not these
SM86 A6000s. These are local configurations, not universal engine rankings.

## 8. What this milestone establishes

The committed scope is ordinary dense Qwen, greedy sampling, one host, two
full replicas, and independent private KV pools. Prefix sharing, hybrid state,
quantized KV, cross-host transport, mixed TP/PP/EP/CP, stochastic sampling,
and automatic placement are deferred. Existing M10 remains the default.

The correctness, lifecycle, real GPU-overlap, equal-resource comparison,
and cross-engine calibration gates are complete. M11 demonstrates how a
request can move between model owners without recomputing its prompt or
confusing allocator-local page numbers. It also demonstrates why separating
phases is not sufficient to improve performance: prompt batching, transfer
cost, ready decode batch size, and resource balance all matter.

An asynchronous, verified transport or packed prefill would be a separate
experiment with the same two-replica control. Neither is required to finish
this educational milestone, and the measured result does not justify
changing the default serving path.

## References

- [DistServe](https://arxiv.org/abs/2401.09670): separation of prefill and decode
  resources and the trade-off against transferring intermediate state.
- [M3 paged KV]({% include tinyserve-post-url.html slug="building-tinyserve-m3" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m3-paged-kv.md" %}): logical positions and allocator-local pages.
- [M5a chunked prefill]({% include tinyserve-post-url.html slug="building-tinyserve-m5a" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m5-chunked-prefill.md" %}): bounding prompt work on a shared
  GPU, a different mechanism from moving that work to another replica.
