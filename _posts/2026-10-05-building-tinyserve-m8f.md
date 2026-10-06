---
layout: post
math: true
title: "Building tinyserve M8f: Quantize the history, not just the weights"
date: 2026-10-05 08:23:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Quantize the KV history with per-token, per-head scales, real INT8 pages, and an explicit quality and speed boundary."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m8f-kv-cache-quantization.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M8 — Quantization: fewer bits are a systems contract]({% include tinyserve-post-url.html slug="building-tinyserve-m8" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m8-quantization.md" %}) · Next: [M8g — Write compressed KV without the intermediate tensors]({% include tinyserve-post-url.html slug="building-tinyserve-m8g" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m8g-fused-kv-writes.md" %})

Part of [M8 — quantization]({% include tinyserve-post-url.html slug="building-tinyserve-m8" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m8-quantization.md" %}). Prerequisites:
[M3 paged KV]({% include tinyserve-post-url.html slug="building-tinyserve-m3" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m3-paged-kv.md" %}) and the M8 discussion of scales and Q/DQ.

**Status: implemented and locally qualified; opt-in eager reference.**
M8a–M8e remain complete within their existing scope. M8f extends the same
quantization milestone to KV storage; it does not rename M11 or create M12.
Development continues from the current engine, not an old M8 checkout.
The codec, Q/DQ oracle, genuine INT8 paged storage, and eager attention reader
are implemented. BF16 remains the default. The memory calculations below
describe the representation, not a promised speedup or broad quality guarantee.

## 1. Why quantized weights are only half of the memory story

Weights belong to the model. Every request reuses them. K and V belong to a
request's attention history: each new token adds more vectors, which later
queries read again. Compressing weights does not compress that history.

For one request with $T$ cached tokens, a dense decoder's BF16 KV payload is

$$
M_{\mathrm{KV}} = L \times 2 \times T \times H_{\mathrm{KV}} \times D \times 2
\quad \text{bytes}.
$$

Here $L$ is the layer count, the first 2 selects K and V,
$H_{\mathrm{KV}}$ is the number of **KV heads**, $D$ is the head dimension,
and the final 2 is BF16's bytes per element. With grouped-query attention,
several query heads share a KV head; counting query heads would overestimate
the physical cache. Sum this expression over requests with private histories.
Allocated pages add unused tail slots; shared prefixes change the accounting.

For the Qwen3-4B configuration used in [M11]({% include tinyserve-post-url.html slug="building-tinyserve-m11" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m11-prefill-decode.md" %}),
$L=36$, $H_{\mathrm{KV}}=8$, and $D=128$. A 2,048-token history contains
**288 MiB** of BF16 K/V. M11's eight-long-prompt cohort transferred 2.25 GiB
of logical KV and spent a median 1.71 seconds importing it through host
memory. Those are measured M11 results, not an M8f performance prediction.

KV quantization asks two related questions:

- Can the same GPU hold more histories by storing fewer bytes per token?
- Can attention read less memory without spending more time reconstructing it?

A later transfer path might also move fewer bytes. The first M8f slice is
single-GPU serving; it does not change M11's handoff protocol.

## 2. K errors and V errors enter attention differently

For one query head, write attention as

$$
z_j = \frac{q^\mathsf{T}k_j}{\sqrt D}, \qquad
a = \operatorname{softmax}(z), \qquad
o = \sum_j a_j v_j.
$$

If reconstruction introduces $\delta k_j$ and $\delta v_j$, then

$$
\delta z_j = \frac{q^\mathsf{T}\delta k_j}{\sqrt D}.
$$

Key error changes **which positions receive attention**. Value error changes
**what content those positions contribute**. To first order,

$$
\delta o \approx \sum_j \delta a_j v_j + \sum_j a_j\delta v_j.
$$

This is why one reconstruction-error average is not a complete quality test.
A small key error aligned with a query can change a near-tied attention
decision. A value error at a heavily attended position can matter more than
a larger error at an irrelevant position.

The stored error persists: every later query may read that approximate
vector. We do **not** repeatedly round the same old vector each decode step;
we quantize once when writing it and reuse its code and scale. Nevertheless,
changed attention outputs can alter future activations and generated tokens.
That feedback is different from directly quantizing GDN/KDA's repeatedly
updated recurrent matrix; recurrent-state quantization remains outside M8f.

## 3. Format and scale granularity are separate choices

M8 already explains INT8/INT4, FP8, MXFP8, MXFP4, and NVFP4 encodings.
The same representation contract applies here, but KV arrives online rather
than as a static checkpoint. We must decide when scales become known and
whether appending a token can change the interpretation of old codes.

| Scale grouping | What shares a scale? | Online consequence |
|---|---|---|
| Per-layer, separately for K and V | All heads and positions in a layer's cache | Little metadata; a fixed scale needs calibration and can encounter unseen outliers. |
| Per-head, separately for K and V | Positions and channels of one KV head | Finer calibration, but still not a new scale for every token. |
| Per-token, per-head | The $D$ channels of one token's K or V vector | A new token can be encoded independently; no old-page rewrite. |
| Per-channel over a token group | One channel across several positions | Requires a completed group, fixed calibration, or an explicit unfinished-group policy. |

In this chapter, “per-token, per-head” means reducing over **channels** to
obtain a scale indexed by token and head. “Per-channel over a token group”
means holding the channel fixed and reducing over **tokens**. Naming the
reduction axis prevents an easy implementation mistake.

K and V need not use the same grouping. [KIVI](https://arxiv.org/abs/2402.02750)
reports different favorable groupings for its low-bit scheme: per-channel
keys and per-token values. That is a reason to investigate distributions,
not evidence that one grouping is universally best for every model, bit
width, or kernel. M8f's first INT8 reference uses per-token, per-head scales
for **both**, favoring a simple append-only contract over a claim of optimal
compression or accuracy.

The insertion point matters too. Tinyserve currently normalizes Q/K and
applies RoPE before writing K. M8f quantizes that **post-RoPE K**;
readers must not apply RoPE again. [KVQuant](https://arxiv.org/abs/2401.18079)
instead investigates pre-RoPE key quantization among its techniques. Adopting
that choice would change the read/position contract, not just a cast.

FP8 is a different candidate, not another spelling of INT8. For example,
[vLLM's versioned FP8 KV documentation](https://docs.vllm.ai/en/v0.25.0/features/quantization/quantized_kvcache/)
describes per-tensor and calibrated per-attention-head scaling, with backend
restrictions. Those are not the dynamic per-token scales used here.
Likewise, M8d's NVFP4/MXFP4 codecs do not establish an attention backend that
can consume those cache layouts. Format support, scale layout, and execution
support must be qualified separately.

## 4. The INT8 contract

The implementation uses symmetric signed INT8 codes and **FP32 scales**. Each
layer, token, and KV head gets one K scale and one V scale. For a finite
vector $x$ with $D$ channels:

$$
m = \max_i |x_i|, \qquad
s = \begin{cases} \operatorname{clip}(m/127,s_{\min},s_{\max}) & m > 0 \\ 1 & m = 0. \end{cases}
$$

$$
c_i = \operatorname{clip}\!\left(\operatorname{round}(x_i/s), -127,127\right),
\qquad \widehat{x}_i = s\,c_i.
$$

Compute the reduction and division in FP32; use round-to-nearest with ties
to even. The `-128` code is deliberately unused, so the positive and negative
ranges are symmetric. An all-zero vector stores zero codes and scale 1.
Non-finite real input vectors are errors, not values to silently encode.
The reference reconstructs in FP32 and casts to the attention compute dtype.
The scale floor $s_{\min}=2^{-126}$ is FP32's smallest positive normal value.
The ceiling is $s_{\max}=(F_{\max}/127)(1-2^{-23})$, where $F_{\max}$ is FP32's
largest finite value. These guards avoid a zero/flushed divisor at the tiny
end and overflow from rounding `127 * scale` at the huge end. Ordinary model
activations use the absmax/127 rule. In this readable implementation,
checking for non-finite inputs synchronizes the GPU at each layer's write.
That cost is included in serving measurements, not hidden as setup work.

Why FP32 scales first? Scale rounding would introduce another error source,
and very small FP16 scales can underflow. At $D=128$, the extra four bytes
are only 3.125% of the INT8 payload. FP16 or BF16 scale storage would be a
separate measured choice, not an implicit change to this contract.

### A four-channel example

Take one toy head with $x=[-1.0,-0.2,0.3,0.7]$. Its FP32 scale is
approximately $1/127$. Rounding gives:

| Channel | Original $x_i$ | INT8 code $c_i$ | Reconstructed $\widehat{x}_i$ |
|---|---:|---:|---:|
| 0 | -1.0 | -127 | -1.000000 |
| 1 | -0.2 | -25 | -0.196850 |
| 2 | 0.3 | 38 | 0.299213 |
| 3 | 0.7 | 89 | 0.700787 |

The largest absolute reconstruction error is about 0.00315 before casting
to the compute dtype. For this tiny $D=4$ example, four INT8 bytes plus one
four-byte scale occupy **the same eight bytes** as four BF16 values. Smaller
element types do not guarantee savings once metadata is counted.

## 5. Codes and scales must follow the same page table

The persistent arrays preserve M3's physical indexing:

```text
codes:  [layers, physical_pages + 1, 2, page_size, kv_heads, head_dim]  INT8
scales: [layers, physical_pages + 1, 2, page_size, kv_heads]            FP32
                                   ↑ K/V
```

The extra page is the existing scratch page, outside the request allocator.
It is not extra user capacity. Both arrays must reserve it and define safe
scratch contents. The normal page manager and logical block tables still
count tokens and pages, not individual code or scale allocations.

[![A token's INT8 code vector and FP32 scale use the same physical page and offset](/assets/tinyserve/m8f-kv-layout.svg)](/assets/tinyserve/m8f-kv-layout.svg)

Use a toy page size of four and block table `[5, 2]`. Logical token 5 maps to
physical page 2, offset 1. For layer 3, KV head 0, its key is reconstructed as

```text
codes[3, 2, K, 1, 0, :] * scales[3, 2, K, 1, 0]
```

The value uses the V plane and its **own** scale. Reading the correct code
with a previous owner's scale silently corrupts the history. Allocation,
reuse, cancellation, and tail masking therefore concern both arrays.

If six tokens are cached, appending token 6 writes page 2, offset 2. Only
that new token's code vectors and scales change. The scales of tokens 0–5
remain unchanged, even if token 6 has a much larger magnitude. A per-page
scale that grows on every append would require re-encoding existing codes;
merely updating the scale would reinterpret old data incorrectly.

Unused tails must be excluded or replaced by neutral values **before**
dequantization/attention. A softmax mask cannot repair a NaN introduced by
an uninitialized scale: even zero times NaN is NaN. Tests should poison scale
tails and use sentinel code values to verify that padding cannot contaminate
a real request. Scratch writes must never affect live token metadata.

### What the representation saves

For one K or V head-vector, BF16 stores $2D$ bytes. INT8 with an FP32 scale
stores $D+4$ bytes. At $D=128$:

$$
\frac{M_{\mathrm{INT8+scale}}}{M_{\mathrm{BF16}}}
= \frac{128+4}{256}=0.515625.
$$

For the same Qwen3-4B, 2,048-token history:

| Component | Calculated bytes in MiB |
|---|---:|
| Original BF16 K and V | 288.0 |
| INT8 codes | 144.0 |
| FP32 scales | 4.5 |
| Codes plus scales | 148.5 |

That is **48.44% less logical KV storage**, or about **1.94×** as many tokens
in the same idealized cache-byte budget. It is not a 1.94× larger total GPU:
weights, scratch pages, unused page tails, metadata, workspaces, and transient
dequantized tensors still consume memory. A hypothetical M11 transfer must
carry the scales too; its payload would be 148.5 MiB, not exactly 144 MiB.

## 6. Fake quantization, real storage, and fused reads

[![Three execution paths distinguish a Q/DQ oracle, compressed storage with temporary reconstruction, and a future fused read](/assets/tinyserve/m8f-kv-execution.svg)](/assets/tinyserve/m8f-kv-execution.svg)

**Q/DQ oracle (`int8_qdq`).** Apply the same rounding and reconstruction at cache
writes, then retain reconstructed values in an ordinary floating cache.
This isolates the numerical recipe and supplies a readable reference.
Persistent storage is still floating point: it is not a memory-saving path.

**Real-storage path (`int8`).** Retain only INT8 codes and FP32 scales for the
cache. A PyTorch reference reader gathers valid pages, reconstructs the
requested layer's history into temporary BF16 tensors, and calls attention.
It must not create a persistent floating mirror of the complete KV pool.
Nevertheless, those temporary tensors can reduce peak-memory savings and
make decode slower. Calling this real quantization describes its storage,
not native INT8 attention arithmetic or a latency improvement.

**Possible later fused read.** An attention kernel could read codes/scales
directly from pages, reconstruct tiles in registers, and avoid writing the
full floating history to GPU memory. Quantized storage still does not imply
integer matrix instructions: the reconstructed operands may participate in
floating-point attention. This kernel is a follow-up only after the oracle,
storage path, and measurements are complete.

## 7. Follow the implementation

The format is selected with `LLM(..., kv_quantization="int8",
cuda_graphs=False, prefix_caching=False)`. `none` keeps ordinary floating
storage; `int8_qdq` selects the floating-storage numerical oracle. `auto`
backend selection uses the eager reader for either quantized mode.

| Boundary | Ordinary floating path | M8f behavior |
|---|---|---|
| [`Qwen3Attention.forward`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/models/qwen3.py) | Q/K normalization and RoPE precede paged writes. | Keep Q and model weights unchanged; quantize post-RoPE K and ordinary V. |
| [`PagedKVCache.write`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/paged.py) | Scatter floating K/V to physical slots. | Call [`quantize_kv`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/kv_quantization.py), then scatter matching codes and scales. |
| [`TorchAttentionBackend.attend`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/attention.py) | Gather floating pages for chunks/decode. | Call `cache.gather` to reconstruct the requested layer's history; preserve masks and KV-head mapping. |
| `PagedKVCache.layer_kv` / `FlashInferAttentionBackend` | Pass a floating pool with the model's KV dtype. | Do not pass INT8 codes to this existing contract; reject incompatible backend selection. |
| [`GraphRunner`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/graph_runner.py) | Capture floating-pool decode and scratch-page behavior. | Leave quantized-cache graph capture disabled in the first slice. |
| [`LLM`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tinyserve/engine.py), [`generate.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/examples/generate.py) | Construct the cache and select execution paths. | Add an explicit opt-in cache format with early validation; leave BF16 defaults unchanged. |

There is an important **prefill asymmetry** in today's implementation. Fresh
prefill writes the cache but attends to the in-flight floating K/V directly.
Decode reads the stored cache. Chunked prefill also reads stored history,
including the chunk just appended. Consequently, simply quantizing writes
does not make every prefill attention operation quantized.

M8f preserves those execution boundaries. Both the Q/DQ
oracle and packed-cache path follow the same rules: fresh-prefill
in-flight operands remain floating; chunk/decode cache reads use reconstructed
values. Testing only first-prompt logits could miss an entirely broken KV
reader. Conversely, token equality across different chunk sizes is not a
valid unconditional gate for this lossy design. Hold the chunk schedule fixed
when comparing implementations, and report schedule sensitivity separately.

The same caution applies to optional features. An INT8 cache is a new storage
contract, not automatic support for every caller that directly reads
`cache.pool`. M11's `pack_kv`/`import_kv`, speculative rollback paths, prefix
sharing, and model-parallel backends need their own qualification. Unsupported
combinations fail early rather than silently use the wrong layout.

### Walk one append through the code

Return to the `[5, 2]` block table and four-token pages. Suppose tokens 0–4
are already cached and decode appends token 5:

1. `_paged_decode` computes physical slot `2 * 4 + 1 = 9`, position 5, and
   context length 6. The block table remains `[5, 2]`.
2. Each attention layer projects this token, normalizes Q/K, applies RoPE,
   and calls `cache.write(layer, slots, k, v)`. For one token, K and V each
   have shape `[1, H_kv, D]`.
3. `write` stacks them as `[1, 2, H_kv, D]`. Reducing the last axis produces
   scales of shape `[1, 2, H_kv]`. Both arrays scatter into page 2, offset 1.
   No earlier scale changes. Padding slots are filtered out before encoding;
   the scratch page stays at zero codes and unit scales.
4. `gather` looks up pages `[5, 2]` in **both** arrays, masks offsets beyond
   logical length 6, and reconstructs floating values. The existing reader
   reshapes these pages into token order and performs ordinary attention.
5. At request completion or cancellation, the page manager returns pages 5
   and 2 to its free list. It need not clear them: the next owner's real slots
   overwrite both codes and scales, and unread tails remain masked.

The VS Code configuration **“M8f: INT8 KV pages, scales, and chunked decode”**
uses `generate.py` with a four-token chunk budget. Break at `write` to inspect
codes/scales and at `gather` to inspect reconstruction. Its essential flags
are `--kv-quantization int8 --live --backend gather --no-cuda-graphs
--no-prefix-cache --quantization none`. Change only the format to `int8_qdq`
to inspect the oracle's floating pool; this does not compress its storage.

## 8. Performance questions to answer, not assume

For a fused reader in a KV-read-dominated regime, fewer bytes can help. A
rough upper bound illustrates the limit. Let $f$ be the fraction of baseline
step time attributable to the KV memory traffic that can actually shrink,
and let $r=0.515625$ be the storage ratio. Ignoring added work:

$$
\operatorname{speedup}_{\mathrm{ideal}} = \frac{1}{(1-f)+fr}.
$$

If $f=0.5$, this is only about **1.32×**, not 1.94×. This is an illustrative
model, not a measured bottleneck fraction. It assumes the bytes actually
read by the kernel shrink by $r$ without changing other costs.

The gather-and-reconstruct implementation can violate that assumption: it
reads codes/scales, writes a temporary BF16 history, and attention reads that
history again. Additional launches and write-time quantization cost time too.
Short-context, low-batch decode may be dominated by weights or launch overhead
instead of KV. Compression alone does not establish a latency win.

Run two distinct experiments:

1. **Same token/page capacity and workload:** measure resident cache bytes,
   peak allocated memory, quantize/write time, read/dequantization time,
   TTFT, decode latency, and completed output throughput. First compare
   floating versus INT8 storage through the **same reference backend**;
   compare against the ordinary FlashInfer path separately as a serving
   baseline, without attributing the entire backend difference to quantization.
2. **Same physical cache-byte budget:** allocate more INT8 token slots using
   the code-plus-scale formula and page rounding. Measure admission capacity
   and serving throughput under identical arrivals. Report the actual bytes
   of both pools; this is a capacity experiment, not the fixed-work latency
   experiment above.

Use the predownloaded Qwen3-0.6B for fast correctness work and Qwen3-4B for
the retained serving fixtures. Vary context length and batch size, preserve
warmups and repeated paired measurements, and refresh the existing
cross-engine calibration after implementation. Do not silently quantize the
reference engines' KV or compare different output budgets as identical work.

## 9. Validation must exercise history reads

Three comparisons answer different questions:

- **Codec:** known vectors, zeros, outliers, ties, finite scales, storage
  dtype/shape, and deterministic codes. Check the four-channel example.
- **Implementation:** packed storage versus the same-recipe Q/DQ oracle,
  with identical prompts, histories, batch shapes, and chunk schedules.
  Compare reconstructed vectors, attention outputs, and logits before
  looking at generated text. Cover arbitrary page maps, append boundaries,
  partial pages, ragged batches, cancellation, reuse, and scratch isolation.
- **Quality:** quantized KV versus untouched BF16 KV. This is intentionally
  lossy, so token-exact generation is a diagnostic, not the definition of
  correctness. Near-tied logits can change greedy tokens.

For perplexity or negative log likelihood (NLL), prefill a fixed prefix and
then teacher-force known continuation tokens **through cache-reading decode
steps**. Score each known next token from the preceding logits, append the
known token, and repeat. Both runs see the same text even if their greedy
choices would differ. A single fresh-prefill forward may bypass reconstructed
KV and therefore cannot validate this cache quantizer's quality.

Freeze the corpus snapshot, held-out continuations, context lengths, and
small factual/retrieval prompt set before the first candidate run. Record
average and per-sequence NLL changes, logit drift, top-token agreement, and
additional failures versus BF16. The initial quality budget is at
most **0.02 nats/token** mean NLL increase on that frozen suite, alongside
no new failures on its small sanity set; the NLL budget corresponds to about
a 2.02% perplexity increase. Inspect per-sequence regressions as well: an
average can hide a damaged long-context example. This is a bounded local
acceptance criterion, not a broad model-quality certification.

Quality and performance results may reject a candidate without invalidating
the educational implementation. Keep a failing or slower path explicitly
experimental, report the result, and do not promote it as the default.

## 10. A bounded addition to M8

The first M8f implementation is deliberately limited to dense Qwen,
single-device greedy serving, unchanged BF16 model weights, private paged KV,
dynamic per-token/per-head INT8 codes with FP32 scales, and eager reference
attention for whole prompts, chunks, and decode. It includes the Q/DQ oracle,
real compressed storage, lifecycle checks, a `generate.py` example/debug
configuration, and correctness/quality/memory/performance receipts.
FP32 remains available for reference tests; measured serving uses BF16.

FP8/FP4 KV kernels, fused INT8 paged attention, quantized weights combined
with quantized KV, shared prefixes, CUDA graphs, TP/PP/EP/CP, speculative
decoding, M11 transfer integration, and hybrid recurrent-state compression
are **not** prerequisites for completing this slice. They require separate
design decisions and evidence rather than an expanding M8f checklist.

This keeps the series organized by concept: M8 teaches quantization contracts,
M11 teaches ownership transfer, and a future integration can connect their
already-tested interfaces. Neither old milestone numbers nor completed
commits need to be rewritten.

## 11. What the A6000 measurements say

The retained run uses Qwen3-4B with unchanged BF16 weights on one RTX A6000.
The fixed-capacity comparison keeps 16,384 usable cache-token slots, disables
prefix reuse, and uses a 512-token prefill budget. Each condition has two
warmups and five measured repetitions. BF16/INT8 order alternates; their
model copies and separate cache pools stay resident. GPU 0 has no competing
compute process, but GPU 1 and the host run unrelated work: this is not a
fully isolated machine.

### Correct representation, smaller storage, slower execution

The physical pools include 1,024 usable pages and one scratch page. BF16
allocates **2,306.25 MiB**; INT8 codes plus FP32 scales allocate
**1,189.16 MiB**. The measured tensor allocation agrees with the **48.44%**
cache-storage reduction. It does not reduce model weights or every temporary.

| Prompt / output tokens | Requests | BF16 eager tok/s | INT8 eager tok/s | Paired INT8/BF16 ratio, 95% bootstrap interval | Floating FlashInfer + graphs tok/s |
|---|---:|---:|---:|---|---:|
| 128 / 128 | 1 | 25.26 | 17.06 | 0.673 [0.659, 0.683] | 62.10 |
| 128 / 128 | 8 | 193.29 | 131.78 | 0.680 [0.674, 0.686] | 470.67 |
| 2,048 / 32 | 1 | 20.58 | 14.38 | 0.699 [0.689, 0.746] | 39.39 |
| 2,048 / 32 | 8 | 42.01 | 28.25 | 0.673 [0.672, 0.679] | 72.62 |

Throughput entries are medians. The ratio is the median of paired ratios,
not the ratio of the two displayed medians. Every request completed its
stated output budget. The INT8 reference is **about 30–33% slower** than the
same eager reader with BF16 storage. The final column changes the attention
backend and graph execution too; it is a serving baseline, not an isolated
measurement of quantization overhead.

The one-request 128/128 case illustrates the latency cost: median TTFT rises
from **45.13 to 59.10 ms**, and the median across runs of the p95 token gap
rises from **40.08 to 59.24 ms**. Across these four cases, INT8 did not reduce
the measured peak temporary allocation above the resident-model baseline.
Keeping a compressed pool does not remove the reader's floating intermediates.

A separate instrumented 2,048/32, eight-request cohort contains 3,528 layer
writes and 3,528 layer reads in each eager mode. Total write-region host time
rises from **459 to 4,652 ms**; read-region CUDA-event time rises from **276
to 2,023 ms**. These regions include dispatch, synchronization, gather, and
reconstruction as applicable, not just arithmetic. They are nested in forward
regions, so do not add them to parent times. Instrumented totals are not the
uninstrumented throughput measurements above. A useful optimization would
need to address both the write path and the read path.

### A capacity-limited workload can tell a different story

Now fix the physical **cache-byte** budget instead of its token capacity.
The BF16 pool has 256 usable pages plus scratch (**578.25 MiB**, 4,096 token
slots). The INT8 pool fits 497 usable pages plus scratch (**577.76 MiB**,
7,952 slots) within that budget. We do not spend saved bytes on model weights
or claim this is a fixed-total-GPU-memory experiment.

Submit the same three 2,048-token prompts with 32-token output budgets.
The unchanged scheduler reaches one resident request with BF16 and three
with INT8. Both finish all 96 output tokens with no preemptions. Across five
alternating repetitions after one warmup per mode, median throughput is **21.05 tok/s for
BF16 versus 24.07 tok/s for INT8**. Extra resident requests let decode batch
more work per forward, compensating for the slower reader in this particular
cohort. The three-versus-one admission result is a discrete fit threshold,
not a claim of 3× general capacity. This does not reverse the losing
fixed-capacity timings; it answers a different serving question.

### Quality is a different comparison

The [frozen fixture](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/tests/data/kv_quantization_quality.json) contains eight
original paragraphs repeated to create contexts of 128, 512, and 2,048 tokens.
Each case scores 32 continuation tokens through cache-reading decode:
**24 sequences and 768 scored tokens**. It also contains 12 short factual or
retrieval questions; these use 16-token prefill chunks so the cache reader is
exercised before the answer. The fixture and the +0.02 nats/token mean-NLL
budget were fixed before candidate evaluation.

Packed storage and same-recipe Q/DQ produced **bit-identical logits** on all
scored Qwen3-4B histories. Relative to BF16, mean NLL changed by
**−0.000043 nats/token**, with a worst per-sequence increase of **0.000600**.
Top-1 agreement was 100% on those teacher-forced tokens, and both formats
passed all 12 sanity questions: no additional failures.

This is deliberately a narrow result. Repetition makes the continuations
easy to predict; BF16 mean NLL was only 0.001793. The maximum raw-logit change
was **16.125**, despite unchanged top choices in this fixture. Do not turn a
small average NLL change into a claim that every logit is close, or call this
a representative natural-text perplexity result. The path remains opt-in.

After inspecting that limitation, we added a **post-hoc diagnostic**, not a
replacement acceptance fixture: the same eight original texts, a 48-token
prefix and 32-token continuation, without repetition, using 32-token chunks.
The codec was unchanged. Across its 256 scored tokens, BF16/INT8 mean NLL
was **3.1659/3.1446**, the worst per-sequence increase was **0.009935**, and
top-1 agreement was **96.875%**, not 100%. Maximum raw-logit drift was 2.0;
packed storage still matched Q/DQ exactly. The small NLL decrease does not
establish a quality improvement. The changed top tokens are a useful reminder
that faithful implementation of a lossy recipe is not token-exact BF16 parity.

### Cross-engine calibration remains a separate baseline

The refreshed HTTP calibration uses ordinary **floating KV**, not the new
INT8 mode: Qwen3-4B BF16 weights, two warmups, and five measured cohorts per
condition. Tinyserve uses a 32,768-token cache and 8,192-token prefill budget;
the receipt retains the other engines' server-specific batching settings.
These settings differ from the paired INT8 experiment above. Median completed
output tokens per second are:

| Prompt / requested output tokens | Clients | Tinyserve | llama.cpp | FreeToken | Ollama |
|---|---:|---:|---:|---:|---:|
| 128 / 128 | 1 | 60.03 | 74.01 | 72.91 | 73.66 |
| 128 / 128 | 4 | 225.28 | 246.44 | 283.45 | 248.07 |
| 128 / 128 | 8 | 411.72 | 401.11 | 541.08 | 421.44 |
| 2,048 / 32 | 1 | 41.63 | 37.84 | 49.50 | 42.98 |
| 2,048 / 32 | 4 | 85.39 | 49.44 | 107.20 | 61.37 |
| 2,048 / 32 | 8 | 102.83 | 64.04 | 132.56 | 86.36 |

FreeToken reports **127/31** generated tokens for the **128/32** requested
budgets; the others report 128/32. Rates use those actual counts. Prefix reuse
is disabled/configured off; retained native counters are included, and Ollama
reports all 128/2,048 prompt tokens evaluated in these cohorts. Floating KV
types, engine scheduling, and HTTP emission boundaries still differ. This is
a finite-cohort calibration, not an identical-kernel comparison or a saturated
throughput ranking. NInfer is excluded: its checkout requires SM120a, whereas
the A6000 is SM86.

Tinyserve's six default-path medians stay within about 1% of the preceding
M11 calibration. That is a historical stability check, not a causal speedup
from an opt-in mode that was disabled in this calibration.

### Retained evidence and completion boundary

The [portable receipt](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m8f-kv-quantization-a6000-2026-09-23.json)
retains the source/model/fixture fingerprints, per-sequence quality results,
paired timing samples, physical-byte accounting, phase profiles, actual
cross-engine token counts, server commands, and final validation logs. Full
per-chunk traces remain in the fingerprinted local artifacts named there.

The final one-GPU regression run passed **503 tests**, with **13 skipped**.
M8f's 35 tests include FP32/BF16 Qwen3-0.6B packed-versus-Q/DQ logits, poisoned
tails, arbitrary page maps, chunk boundaries, ragged prefill/decode, scratch
isolation, cancellation, preemption, reuse, failed-write cleanup, HTTP
streaming, and unsupported-mode rejection. The exact M8f `launch.json`
command also completed its three live requests.

M8f is complete as a readable storage-and-serving reference. It is not a
fused kernel, a default change, or a broad model-quality endorsement. A
future optimization can use these same contracts and measurements to test
whether fused writes and tiled reads retain the storage benefit while
removing the measured execution cost.
