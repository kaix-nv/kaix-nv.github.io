---
layout: post
math: true
title: 'Tinyserve, Chapter 14: Serving quantized weights activations and KV'
date: 2026-10-06 00:00:00 -0700
categories:
- tinyserve
- llm-serving
excerpt: When do fewer stored bits improve capacity or serving performance?
book_chapter: 14
source_revision: 14b0c8dbb975350bbd730ac0c702eafce36eb95c
source_document: docs/book/14-quantized-serving.md
code_revision: e20a34815498560f9226a9e057ade31b2b62743f
---

{% include tinyserve-article-style.html %}
{% include tinyserve-book-nav.html %}

*Chapter 14 · Execution Optimization*

A quantized tensor is useful to a server only if its representation survives
loading and reaches a compatible kernel. A four-bit file that expands into
BF16 weights before the first request saves disk space. Keeping those weights
packed on the GPU saves resident memory. Multiplying native four-bit
operands is a further property, requiring suitable hardware and instructions.

We will follow a packed NVFP4 projection from checkpoint bytes to a matrix
product, then follow one generated token into an INT8 KV page. The first
object is static model state shared by requests; the second is mutable state
owned by a request. Both need codes and scales, but their lifetime and
correctness obligations differ.

Tinyserve's measured implementations use an RTX A6000. Its NVFP4 projection
keeps weights packed and reconstructs tiles for BF16 arithmetic. Its INT8
weight path also has a native W8A8 option. INT8 KV saves persistent cache
space, with an eager floating reconstruction reader and an optional fused
writer. These distinctions explain both the capacity savings and the
retained latency losses.

## Three implementations of the same numerical recipe

For a linear projection, flatten leading dimensions into token rows:
$X[M,K]$, weights $W[N,K]$, and output $Y=XW^T$ of shape `[M,N]`.
Quantization changes the approximation to W and possibly X, but the module
must still produce the expected shape and floating output for the next layer.

| Execution | Persistent projection weight | What the dot product consumes |
|---|---|---|
| Full Q/DQ oracle | Reconstructed floating matrix | Floating X and reconstructed W |
| Packed weight-only fallback | Codes and scales | Floating X and a tile of reconstructed W |
| Native low-precision kernel | Codes and scales | Compatible low-precision operands, with wider accumulation |

The Q/DQ oracle is deliberately easy to inspect. It answers whether the
compressed implementation reconstructs the intended values and computes
the intended operation. Comparing that oracle with the original model
answers a different question: how much the quantization recipe changes
the model. Matching the oracle does not establish BF16 quality parity.

Weight-only W8A16 or W4A16 preserves sixteen-bit activations. Tinyserve's
fallback decodes weight tiles inside its matrix kernel, so no complete
floating weight is written to GPU memory. This saves persistent storage
even if the extra conversion instructions make the kernel slower.

For W8A8, each runtime activation row receives an INT8 scale $s_X[m]$,
while the checkpoint supplies one weight scale $s_W[n]$ per output row:

$$
Y_{mn}\approx s_X[m]s_W[n]\sum_k Q_X[m,k]Q_W[n,k].
$$

The inner sum uses INT8 operands and an INT32 accumulator. Activation
quantization is dynamic, including on each CUDA graph replay. For example,
`X=[1,-2,0.5,0]` and `W=[1,0.5,-1,0]` use scales $2/127$ and $1/127$.
Codes `[64,-127,32,0]` and `[127,64,-127,0]` give integer dot $-4064$.
Rescaling gives about $-0.50394$, versus the original dot $-0.5$.
An exact integer sum still contains quantization error from its inputs.

If scales vary along K, they cannot be factored outside the entire dot.
First define the integer partial sum for block b:

$$
P_{mnb}=\sum_{k\in b}Q_X[m,k]Q_W[n,k].
$$

Then scale and combine those partial sums:

$$
Y_{mn}\approx\sum_b s_X[m,b]s_W[n,b]P_{mnb}.
$$

Each block's partial result needs its corresponding scales. This is why
block orientation and hardware scale layouts affect kernel compatibility.
The labels W8A8 and W4A16 describe projection operands, not the dtype of
every residual addition, normalization, cache, or logit in the model.

## Load a packed projection without expanding it

Tinyserve's canonical NVFP4 artifact stores three tensors per eligible
projection. Let $G=\lceil K/16\rceil$:

| Tensor | Physical shape | Dtype | Meaning |
|---|---|---|---|
| `weight_packed` | `[N,8G]` | uint8 | Two E2M1 values per byte, even channel in low nibble |
| `weight_scale` | `[N,G]` | uint8 | One nonnegative E4M3 block scale per sixteen channels |
| `weight_global_scale` | Scalar | FP32 | One positive scale for the matrix |

The manifest records logical shape, block axis, formats, rounding recipe,
and the versioned `row-major-low-nibble-first-v1` layout. Its explicit
format selects the loader. The name of a model directory cannot determine
how to decode its bytes.

The exporter runs an absmax reference codec on the seven bias-free
attention and feed-forward projections in each dense Qwen block. It does
not perform activation-aware calibration, retraining, or weight rotations.
Embeddings, normalization, the vocabulary head, and KV retain their ordinary
representations. A mixed checkpoint therefore needs per-module selection;
one global “four-bit” label cannot describe every tensor.

The loader builds `NVFP4Linear` modules on the meta device, then assigns
safetensors data directly to registered buffers. It checks the exact eligible
module set, shapes, dtypes, and finite scale rules, and rejects artifacts
containing both packed and complete floating projection weights. Inspecting
the loaded module should show no full floating `weight` parameter.

This contract is narrower than “load any NVFP4 checkpoint.” Other producers
may use transposed codes, swizzled scales, different group axes, or different
tensor names. Header inspection can identify declared recipes and observed
shapes; it cannot certify a compatible serving kernel. Tinyserve's Kimi
MXFP4 checkpoint-decoding path illustrates another boundary: it expands
weights to the model dtype at loading, so its packed disk representation
does not imply packed GPU execution.

## Follow one byte through the matrix kernel

Use `K=64`, so each weight row contains four blocks and 32 data bytes.
To reconstruct `W[2,19]`, the kernel performs these address calculations:

```text
code byte offset  = 2*32 + 19//2   = 73
selected nibble   = high nibble   (channel 19 is odd)
scale byte offset = 2*4  + 19//16  = 9
```

If byte 73 is `0xD2`, the high nibble is `0xD`, representing E2M1 value
$-3$. Scale byte `0x38` represents E4M3 value 1. With global scale 0.25,
the reconstructed BF16 weight is $(-3\times1)\times0.25=-0.75$.

[![A packed NVFP4 matrix remains in GPU memory as codes and scales. The W[2,19] example selects a byte, nibble, block scale, and global scale before a tile-local BF16 dot product.](/assets/tinyserve/book14-packed-projection.svg)](/assets/tinyserve/book14-packed-projection.svg)

`_w4a16_kernel` repeats this mapping over tiles of output and input
channels. It loads BF16 activations, decodes E2M1 and E4M3, reconstructs
weights in FP32, casts that tile to BF16, and accumulates the dot in FP32:

```python
# Inside the Triton kernel; q and scale came from packed bytes.
weight = ((q * scale) * global_scale).to(tl.bfloat16)
acc += tl.dot(x, tl.trans(weight))
```

Only the final floating output goes to global memory. Logical K masks
padding beyond the true matrix width. Prefill and decode use the same
packed module; additional token-row tiles can decode the same weight
tiles again. Fewer stored bytes do not remove that repeated instruction work.

The kernel oracle preserves this FP32 multiplication order and BF16 cast
before a dot with FP32 accumulation. Chapter 13's general matrix decoder
uses FP64 intermediates instead. An implementation comparison must match
the intended arithmetic contract, including accumulation policy; otherwise
it mixes codec differences with execution-order differences.

The retained A6000 instruction inspection found BF16 `mma.sync` in PTX and
`HMMA.16816.F32.BF16` in SASS. This establishes BF16 multiplication in the
packed fallback. Native NVFP4 belongs to compatible Blackwell hardware and
requires its own kernel and qualification; it is not supplied by changing
the checkpoint label.

## Weight capacity and latency answer different questions

The [NVFP4 measurement record](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m8e-nvfp4-a6000-2026-09-17.json)
used dense Qwen3-0.6B on one A6000. Its resident model-state tensors were:

| Representation | Model-state bytes | Reduction from BF16 |
|---|---:|---:|
| BF16 | 1,503,264,768 | Baseline |
| INT8 W8A16 or W8A8 | 1,064,239,104 | 29.2% |
| NVFP4 W4A16 | 870,187,792 | 42.1% |

The quantized projections alone shrink about 3.56 times, but unchanged
tensors limit whole-model savings. These are tensor bytes, excluding
CUDA context, KV, graph pools, and allocator reservation. They are not
measurements of memory traffic.

Numerical evidence separates implementation from recipe. All 196
projections passed the stated Q/DQ comparison, while full-model packed
versus Q/DQ top-1 agreement was 99.48% over 192 fixed-text positions.
Against the original BF16 model, agreement was 72.40% and mean KL was
0.2513. Different reduction orders explain why even the first comparison
is not bit-identical; the much larger second difference includes the lossy
quantization itself. This small fixture is not a broad task evaluation.

The paired serving comparison kept all model variants resident, enabled
CUDA graphs, disabled prefix reuse, and rotated execution order over seven
measured repetitions after two warmups. NVFP4 reached 94.5 versus BF16's
232.8 output tokens/s for one 128-input/128-output request. For eight
2,048-input/32-output requests, the rates were 179.7 versus 381.1.
Median paired throughput ratios were 0.406 and 0.472. This packed
implementation reduced storage but lost serving throughput. An isolated
projection timing, a native INT8 result, or a cross-engine baseline cannot
turn that result into a native FP4 acceleration claim.

## A generated token brings a different storage lifetime

Weights are loaded once. KV is produced once per layer and token, then
read by later queries throughout the request. Key errors perturb scores
before softmax; value errors perturb the weighted content afterward.
Both can affect later generated tokens, which in turn change future KV.

Tinyserve quantizes post-normalization, post-RoPE keys and ordinary values.
For each token, layer, and KV head, it stores separate K and V scales over
the D channels. Codes use symmetric INT8; scales use FP32. Zero vectors
use scale 1. Finite-endpoint guards keep scales usable, and nonfinite
real inputs raise before cache mutation.

The persistent layout is

```text
codes  [layers, pages+1, 2, page_size, kv_heads, D]   INT8
scales [layers, pages+1, 2, page_size, kv_heads]      FP32
                        K/V axis
```

The additional page is scratch storage, outside usable request capacity.
With four-token pages and block table `[5,2]`, logical token 5 maps to
physical page 2, offset 1. Its key in layer 3, head 0 uses both
`codes[3,2,K,1,0,:]` and `scales[3,2,K,1,0]`. The value uses a separate
scale in the V plane.

[![Logical token five maps through block table [5,2] to physical page two offset one. Its key and value each carry an INT8 vector and a matching FP32 scale.](/assets/tinyserve/book14-int8-kv-page.svg)](/assets/tinyserve/book14-int8-kv-page.svg)

For a toy key `[-1,-0.2,0.3,0.7]`, scale $1/127$ gives codes
`[-127,-25,38,89]`. Appending another token must not alter that scale.
If a page-wide scale were enlarged without re-encoding old codes, the old
history would suddenly mean different values.

Codes and scales must share allocation, indexing, valid lengths, and
reclamation. Reusing the right code with a previous request's scale silently
corrupts attention. Unused tails must be excluded before reconstruction;
a late attention mask cannot repair NaNs already introduced by multiplying
an uninitialized scale.

For each K or V vector, BF16 costs $2D$ bytes and INT8 plus FP32 scale
costs $D+4$. At $D=128$, the ratio is $132/256=0.515625$, a 48.44%
logical-cache saving. At $D=4$, both cost eight bytes: metadata consumes
the entire apparent saving. Weights, scratch, padding, and temporary
reconstruction still contribute to total memory.

## Write once and reconstruct on later reads

`int8_qdq` applies quantize/dequantize at writes and retains the reconstructed
values in floating storage. `int8` retains codes and scales; its Torch
reader gathers pages and reconstructs the requested layer's floating
history before attention. It has no persistent floating mirror, but it
does allocate floating intermediates for reads.

Fresh prefill attends to in-flight floating K/V while writing the cache.
Chunk continuation and decode read reconstructed stored history. Therefore,
first-prompt logits alone cannot validate the reader. Comparing the two
implementations also requires a matched chunk schedule: changing when
attention begins reading rounded history changes the numerical computation.

The optional fused writer preserves the recipe and layout. It first
validates real input vectors on the GPU and checks a small host status.
Only after success does a second kernel quantize and scatter each
`(token,K-or-V,head)` vector directly to its physical destination.

Two kernels preserve the unchanged-on-error contract. If validation and
mutation occurred together, one GPU program could write valid heads before
another found NaNs. The fusion removes intermediate tensors and launches
within quantize-and-scatter; it does not eliminate checked-write
synchronization or fuse the attention reader.

## Capacity quality and serving measurements

The [KV storage experiment](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m8f-kv-quantization-a6000-2026-09-23.json)
used unchanged Qwen3-4B BF16 weights. At equal cache-token capacity, INT8
eager serving was about 30–33% slower than BF16 eager across four tested
workloads, despite the 48.44% persistent-cache saving. Gathering and
reconstructing history, scale reductions, checks, and launches all cost time.

A separate equal-cache-byte experiment allowed 4,096 BF16 token slots or
7,952 INT8 slots. For three 2,048-token prompts with 32-token outputs,
the scheduler admitted one BF16 request or all three INT8 requests.
Throughput was 21.05 versus 24.07 tokens/s. Extra batching compensated for
the slower reader in that particular capacity-limited cohort. This was
an equal **cache-byte** comparison, not equal total GPU memory, and its
three-versus-one admission threshold is not a general threefold capacity gain.

The [fused-writer comparison](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m8g-fused-kv-writes-a6000-2026-09-23.json)
changed only the writer on the same INT8 model and pool. With equal
16,384-token pools, a 512-token prefill budget, seven alternating measured
repetitions and two warmups, it improved throughput 12.0–18.5% over the
reference INT8 writer. The resulting path remained about 19–24% slower
than BF16 eager. Its roughly fourfold checked-write improvement did not
become a fourfold serving improvement because attention reads and the rest
of the model remained.

Packed storage matched its Q/DQ oracle exactly on the retained quality
fixture. Against BF16, the 768 teacher-forced targets and twelve sanity
questions passed a narrow predeclared contract. Repeated text made many
targets easy; a post-hoc nonrepeated diagnostic had 96.875% top-1 agreement.
Neither result certifies arbitrary natural-text quality or token-exact
BF16 behavior.

The separate experimental direct INT8 reader, M8h, remains unqualified;
the retained [sampling integration record](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10c-request-sampling-cpu-2026-09-25.json)
keeps it on its own branch. It is not the reader described at this runtime
snapshot. A future attention kernel can reconstruct tiles without writing
a complete floating history, but that must earn independent numerical and
serving acceptance.

Quantized KV also does not imply quantized recurrent state. A GDN or KDA
state is updated from its previous approximation, so error can feed into
every subsequent update. It needs a state-specific recipe and evaluation.
Likewise, compressed-cache support does not automatically enable shared
prefixes, speculative rollback, distributed transfer, or CUDA graph capture.
Those callers must understand the same codes, scales, and lifetime rules.

## Follow the storage and execution boundaries

| File | Responsibility |
|---|---|
| [quant_checkpoint.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/quant_checkpoint.py), [loader.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/loader.py) | Export/validate manifests and retain packed projection buffers. |
| [quantization.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/quantization.py) | INT8 weight-only and W8A8 dispatch. |
| [fp4.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/fp4.py) | NVFP4 buffers, execution-matched oracle, and tile reconstruction kernel. |
| [kv_quantization.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/kv_quantization.py), [paged.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/paged.py) | KV codec, physical pages, matched scales, and eager reconstruction. |
| [kv_quantization_kernels.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/kv_quantization_kernels.py) | Validation followed by fused INT8 quantize-and-scatter. |

These paths describe snapshot `e20a348`. Their opt-in settings preserve
the distinction between a working storage format, an accepted numerical
recipe, and a faster server. The next chapter pursues a different way to
amortize target-model work: proposing several tokens before deciding which
ones belong in the committed history.

{% include tinyserve-book-nav.html %}
