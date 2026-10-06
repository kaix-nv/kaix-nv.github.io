---
layout: post
math: true
title: "Building tinyserve M8g: Write compressed KV without the intermediate tensors"
date: 2026-10-05 08:24:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Fuse checked INT8 KV writes, measure the removed intermediates, and keep the serving default unchanged."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m8g-fused-kv-writes.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M8f — Quantize the history, not just the weights]({% include tinyserve-post-url.html slug="building-tinyserve-m8f" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m8f-kv-cache-quantization.md" %}) · Next: [M9a — Speculative decoding: let the draft guess, let the target decide]({% include tinyserve-post-url.html slug="building-tinyserve-m9a" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m9-speculative-decoding.md" %})

Status: complete as an opt-in, write-only milestone. Correctness, full regression,
paired performance, and monitored cross-engine calibration gates passed.
M8f remains the default INT8 writer; BF16 KV remains the serving default.

[M8f]({% include tinyserve-post-url.html slug="building-tinyserve-m8f" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m8f-kv-cache-quantization.md" %}) made KV storage smaller, but not faster.
Its fixed-capacity INT8 reference was about 30–33% slower than BF16 eager.
In a separately instrumented 2,048-input/32-output, eight-request cohort,
3,528 layer writes took 4,652 ms of host-region time, versus 459 ms for BF16.
These are whole write regions, including launches and synchronization, not
just the arithmetic of rounding a float to an integer.

## Why writing costs more than the equation suggests

For a layer, incoming K and V each have shape `[N, Hkv, D]`: new tokens,
KV heads, and channels. The reference filters out padding, stacks K/V,
converts to FP32, reduces each vector, checks finiteness, computes scales,
rounds codes, computes physical addresses, and scatters codes and scales.
Several of these operations launch separate kernels or allocate temporary
tensors. During decode, `N` can be only one: launch overhead can dominate.

The scale for one vector is still M8f's recipe: maximum absolute channel
value divided by 127, guarded at FP32's finite endpoints. An all-zero vector
uses scale one. Codes round to nearest with ties to even and clamp to
`[-127, 127]`. Each K and V head has its own FP32 scale. This milestone changes
how we execute that recipe, not its quantization error or persistent layout.

[![Reference writes materialize intermediate arrays; the candidate checks input before a fused quantize-and-scatter kernel](/assets/tinyserve/m8g-fused-kv-writes.svg)](/assets/tinyserve/m8g-fused-kv-writes.svg)

## Two kernels, deliberately not one

The native path first validates real vectors on the GPU and copies one
small status byte per input token to the host. A non-finite real K or V must
raise **before any cache code or scale changes**. Scratch rows are ignored,
including NaNs in padding. That synchronization remains intentional.

Only after validation succeeds does a fused kernel process one
`(token, K-or-V, head)` vector per program:

1. Load its `D` channels directly from the input, respecting tensor strides.
2. Reduce the maximum, calculate the guarded scale, and round/clamp codes.
3. Scatter codes and scale directly into the destination page.

Why not write while checking for NaNs? Different GPU programs run independently.
One program could update a good head before another discovers a bad head.
Rejecting the call afterward would not preserve M8f's unchanged-on-error
contract. A validation pass is simpler than transactional writes or rollback.
This is a fused **quantize-and-scatter** kernel, not a claim of one launch for
the entire checked write operation or asynchronous error handling.

## Follow one token into a page

Take page size four, two KV heads, and four channels. Logical position six
with block table `[5, 2]` maps to physical slot `2 * 4 + 2 = 10`.
Suppose that token's K head zero is `[-1, -0.2, 0.3, 0.7]`.
Its scale is approximately `1/127` and codes are `[-127, -25, 38, 89]`.

At the selected layer, the kernel writes:

```text
codes[page=2, K=0, offset=2, head=0, :] = [-127, -25, 38, 89]
scales[page=2, K=0, offset=2, head=0]   = 1/127
```

For page size `P`, KV heads `H`, and channel count `D`, the flat scale offset
is `((page * 2 + kv) * P + offset) * H + head`; the code offset is that
quantity times `D`, plus the channel. The K/V axis comes **before** the token
offset. Treating the pool as `[flat_slot, 2, H, D]` would silently misplace KV.
The caller supplies unique real slots; repeated scratch slots never write.

## Bounded implementation and acceptance

An explicit `kv_write_backend="triton"` selects the candidate;
`"torch"` remains the default/reference. It applies only to M8f's real INT8
pool on CUDA, with FP32/BF16 inputs. Q/DQ storage stays a separate oracle.
CPU, BF16 KV, attention, scheduling, and default serving stay unchanged.
The kernel supports up to 64 KV heads and 256 channels per head, including
non-power-of-two channel counts; masked channels contribute zero to the
maximum. Inputs may be strided, while the slot vector and destination pools
must be contiguous. The writer explicitly selects the destination CUDA
device and its current stream before launching either kernel.

The implementation is small enough to follow end to end:

| Code | Responsibility |
| --- | --- |
| [`generate.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/examples/generate.py) | `--kv-write-backend triton` opts in; the M8g debug configuration uses this path. |
| [`LLM._paged()`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/engine.py) | Pass the writer choice into the private INT8 cache. |
| [`PagedKVCache.write()`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/paged.py) | Dispatch one selected layer's pool and matching scales. |
| [`write_int8_kv()`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/kv_quantization_kernels.py) | Validate metadata, run `_validate_kv`, inspect host status, then launch `_quantize_scatter`. |

Both prompt chunks and decode tokens pass through this writer. A whole fresh
prefill still attends to in-flight floating K/V; chunk continuation and decode
still read through M8f's reconstructed history. Changing the writer does not
change these attention boundaries or turn prefill into integer arithmetic.

Before accepting the candidate we require exact codes/scales, exact logits
under the same M8f execution schedule, poisoned-tail and scratch tests,
unchanged-on-error writes, reuse/cancellation cleanup, and the frozen M8f
quality checks. Timings must include validation and synchronization, not
only the scatter kernel. Paired serving runs compare the old writer with the
candidate in the same process and alternate their order; BF16 eager remains
a separate control. Default-path cross-engine calibration is a regression
check, not evidence of an INT8 speedup.

The reader still reconstructs floating history before attention. Its cost
does not disappear when writes become cheaper. Fused attention reads, CUDA
graphs, FP8/FP4 KV, shared prefixes, distributed execution, and M11 handoff
are outside M8g. We will stop at this measured write-only slice.

## Measurements: keep the boundaries separate

The A6000 check uses unchanged Qwen3-4B BF16 weights, private pages, the
eager reader, and no CUDA graphs. Quality uses M8f's frozen fixture without
retuning: 24 repeated-text/context cases, 768 teacher-forced targets, and
12 sanity questions. Candidate-versus-Q/DQ logits are bit-identical; the
mean NLL change versus BF16 is `-0.000043375`, with no additional sanity
failures. This repeats M8f's narrow quality result, not a broad quality
certification or evidence that quantization improves the model.

### The checked write itself

Each sample contains 100 calls, including temporary status allocation,
validation, device-to-host status copy, host inspection, fused scatter, and
final synchronization. There are ten warmup calls per backend and seven
paired samples with alternating order. K and V have eight heads and 128
BF16 channels per head; values and physical slots are fixed within each pair.

| New tokens in this layer write | Torch reference, µs/call | Triton candidate, µs/call | Median paired speedup |
| --- | ---: | ---: | ---: |
| 1 | 375.91 | 95.48 | 3.92× |
| 8 | 366.17 | 92.34 | 4.00× |
| 128 | 368.34 | 90.64 | 4.14× |
| 512 | 368.98 | 97.73 | 3.92× |

The similar timings across token counts are consistent with removing much
of the small-operation launch overhead. These are amortized host wall times
for a **checked write**, not the GPU kernel's isolated instruction time.
The retained CUDA-event intervals also include host submission gaps; they
must not be described as pure device arithmetic time.

### Serving: an improvement, not a BF16 replacement

Each cohort uses two warmups and seven timed repetitions, a 512-token prefill
budget, and equal 16,384-token KV pools. BF16 eager and the INT8 model are
resident throughout; the causal writer pair reuses the **same INT8 model and
pool**, changing only `kv_write_backend`. Writer order alternates and the
BF16 control moves around that pair. All requests complete their output budget.

| Input / output tokens | Requests | Torch INT8, tok/s | Fused INT8, tok/s | Paired ratio [95% bootstrap interval] | BF16 eager, tok/s |
| --- | ---: | ---: | ---: | --- | ---: |
| 128 / 128 | 1 | 17.99 | 21.40 | 1.185× [1.164, 1.203] | 26.57 |
| 128 / 128 | 8 | 140.43 | 166.49 | 1.184× [1.173, 1.192] | 205.00 |
| 2,048 / 32 | 1 | 14.96 | 17.73 | 1.171× [1.158, 1.206] | 21.81 |
| 2,048 / 32 | 8 | 28.69 | 32.16 | 1.120× [1.117, 1.127] | 42.23 |

M8g improves throughput **12.0–18.5% over the M8f writer** in these cohorts,
but remains about **19–24% slower than BF16 eager**. Ratios are medians of
paired samples, not ratios rounded from the displayed medians. The fixed
codec/layout preserves M8f's 48.44% persistent-KV storage reduction; this is
not a total-process memory or integer-attention claim. No capacity-limited
speedup or equal-byte-budget experiment is repeated in M8g.

The separately instrumented long-prompt/eight-request cohort explains the
remaining gap. Both writers execute 3,528 layer writes and reads. Write-region
CUDA-event time falls from **1,522 to 379 ms**, while read-region time stays
near **2,011 versus 2,048 ms**. Host write-region time falls from **4,703 to
3,639 ms**, not fourfold: the synchronous input check also waits for earlier
queued model work. These nested attribution regions must not be summed with
their parent forward regions or mistaken for isolated kernel instruction time.

The bounded result is therefore useful but incomplete as a general compressed
attention backend. Fusing writes removes measurable execution cost. It does
not remove the floating history reconstruction that M8f's reader still does.

### Correctness and regression boundary

The one-GPU regression suite passes **550 tests, with 13 skipped**, including
**47 M8g tests**. These cover exact codes/scales for strided FP32/BF16 inputs,
zero vectors, ties, tiny/extreme finite values, scratch collisions, arbitrary
page order, poisoned tails, append immutability, slot reuse, no mutation on
non-finite input, and backend guards. Real Qwen3-0.6B tests compare whole and
chunked prefill, ragged prompt batches, and cache-reading decode logits exactly
against the Q/DQ oracle. Native-writer preemption, cancellation, failed-forward
cleanup, and HTTP streaming/reuse are exercised as well.

This is one-GPU validation, not a requalification of distributed or M11
two-GPU paths. The native writer does not enable those combinations.

The exact **M8g** `launch.json` command also completes its three live requests;
the `2+2=` continuation begins with `4`. This is a runnable debug example,
not an additional quality benchmark.

### Default-path calibration is a separate question

The standard Qwen3-8B suite uses BF16 KV, FlashInfer/CUDA graphs, an
8,192-token prefill budget, 131,072 KV slots, two warmups, and five repeats.
M8g's opt-in writer is **disabled**. All requests complete their output budgets.

| Input / output tokens | Requests | Output tok/s |
| --- | ---: | ---: |
| 128 / 128 | 1 | 38.14 |
| 128 / 128 | 8 | 284.34 |
| 128 / 128 | 32 | 915.24 |
| 2,048 / 32 | 1 | 26.63 |
| 2,048 / 32 | 8 | 67.71 |
| 2,048 / 32 | 32 | 79.83 |

These medians are within 0.3% of the retained M10 standard-suite measurements.
That is a historical stability check, not a causal speedup or an exact-token
comparison across engines. The older
[8B cross-engine baseline](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/cross-engine-a6000-2026-08-29.md) also
uses an older software environment; its numbers must not be used to attribute
an M8g effect.

The fresh Qwen3-4B HTTP calibration uses a reserved GPU 0 window, two warmups,
and five repetitions for each of six conditions per engine. All **24 conditions**
complete and pass their correctness smoke check. The process monitor observes
no foreign GPU process in 403 samples over 433.5 seconds, with a nominal
one-second sampling interval. This is observed process isolation, not a hardware
reservation or a guarantee against activity between samples.

| Input / requested output tokens | Concurrent requests | Tinyserve, tok/s | llama.cpp, tok/s | FreeToken, tok/s | Ollama, tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / 128 | 1 | 60.07 | 73.99 | 72.88 | 74.95 |
| 128 / 128 | 4 | 224.77 | 247.63 | 283.31 | 247.22 |
| 128 / 128 | 8 | 418.21 | 407.01 | 543.25 | 411.88 |
| 2,048 / 32 | 1 | 42.13 | 38.03 | 49.73 | 43.85 |
| 2,048 / 32 | 4 | 86.42 | 49.39 | 107.09 | 60.95 |
| 2,048 / 32 | 8 | 104.25 | 64.01 | 132.54 | 86.15 |

These are median end-to-end HTTP output rates, not isolated kernel speeds.
Tinyserve uses its default floating KV and FlashInfer/CUDA-graph path:
**the M8g writer is disabled**. The model is Qwen3-4B with BF16 weights, but
execution paths and cache layouts differ. Tinyserve, llama.cpp, and FreeToken
have 32,768-token pools/context budgets; Ollama has eight slots with a
4,096-token context per request and FP16 KV. Shared-prefix/prompt-cache reuse
is disabled in the benchmark configurations; Ollama reports evaluating all
128 or 2,048 prompt tokens on every measured request.

FreeToken returns 127 or 31 output tokens per request, while the other engines
return the requested 128 or 32. Rates use **actual output counts**. These
differences prevent an exact-workload speedup claim. Tinyserve's medians range
from -0.23% to +1.58% against the retained M8f calibration; that historical
comparison is descriptive, not a paired estimate of an M8g effect. NInfer is
excluded because its local build targets SM120a, not the A6000's SM86.

Tinyserve's accompanying HTTP lifecycle checks also pass: staggered arrivals,
cancellation, overload with matching HTTP/body status 429, and release of all
request KV references. Two earlier calibration attempts encountered foreign
GPU jobs and are **excluded** from the table. Their records remain in the
receipt; only the completed reserved-window rerun is accepted.

The [portable receipt](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m8g-fused-kv-writes-a6000-2026-09-23.json)
retains accepted quality, paired timings, profiles, the standard suite,
cross-engine measurements and process monitoring, source/model fingerprints,
and test logs. M8g's write-only qualification is complete. The native reader
and any default promotion remain separate work: a faster writer has not made
the measured INT8 serving path faster than BF16 eager.
