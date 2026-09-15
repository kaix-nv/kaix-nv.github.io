---
layout: post
math: true
title: "Building tinyserve M8: Quantization—from bits and scales to serving performance"
date: 2026-09-15 15:55:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "A practical guide to INT8, FP8, MXFP4, and NVFP4: exponent-versus-precision trade-offs, fake and real quantization, packed checkpoints, native arithmetic, and measured serving results."
---

<style>
/* Keep wide format tables and equations scrollable within this article. */
.post-content table { display: block; max-width: 100%; overflow-x: auto; }
.post-content mjx-container[display="true"] { overflow-x: auto; overflow-y: hidden; padding: 0.2em 0; }
.post-content :not(pre) > code { overflow-wrap: anywhere; }
</style>

*Milestone 8 of [building an LLM inference engine from scratch](/series/tinyserve/).
Previous implementation chapter: [M7w — pay once, reuse eight times](https://github.com/kaix-nv/tinyserve/blob/3846ae7c1883acfe483bc1c350db277d9d826ef0/docs/m7w-kda-solve-factors.md).*

Code and evidence: [`tinyserve` @ `3846ae7`](https://github.com/kaix-nv/tinyserve/tree/3846ae7c1883acfe483bc1c350db277d9d826ef0).
This article covers M8a–M8d. Click any figure to open it at full size.

Tinyserve has so far loaded most model weights as BF16. That keeps the
arithmetic easy to inspect, but it leaves a basic serving question unanswered:
what does it actually take to run a model with fewer bits?

The tempting answer is “cast the weights to INT8 or FP4.” That answer skips the
hard parts. A low-precision serving path is a contract between four things:

1. **Representation:** which bit patterns encode the low-precision values?
2. **Quantization scheme:** which values share a scale, and how is that scale
   chosen?
3. **Checkpoint layout:** how are packed values, scales, shapes, and per-layer
   choices serialized?
4. **Execution:** does the kernel merely reconstruct BF16 values, or does the
   GPU multiply the low-precision values natively?

Confusing those layers produces familiar but misleading claims: a model can
carry an `NVFP4` label while expanding every weight to BF16 during loading; a
fake quantize/dequantize (Q/DQ) PyTorch graph can measure the numerical error
while using *more* memory than the original model; and a packed checkpoint can
save memory yet run slower because its kernel spends too much time unpacking.

[![A quantized serving path has separate representation, checkpoint, and
execution contracts](/assets/tinyserve/m8-quantization-contract.svg)](/assets/tinyserve/m8-quantization-contract.svg)

M8 starts with that separation. This chapter develops quantization from one
small matrix, then places INT8, INT4, FP8, MXFP8, MXFP4, and NVIDIA NVFP4 in
the same framework. The implementation and performance results come only
after the format contract is testable; this article does not invent numbers
for kernels that do not exist yet.

**Reading guide.** Start with scales and [fake versus real quantization](#fake-quantization-and-real-quantization-change-different-things),
then the [numerical limits](#numerical-limits-nans-infinities-normals-and-subnormals),
[floating-point bit budget](#how-do-we-choose-exponent-and-fraction-bits),
and [format map](#a-map-of-the-formats-m8-needs-to-explain). The
[performance model](#a-performance-model-for-quantization) explains what fewer
bits can buy. The measured INT8 path follows it; [M8d](#m8d-make-the-fp8fp4-format-contract-executable)
turns the floating-point formats into inspectable bytes.

Here is the implementation boundary at this revision:

| slice | implemented | not implied |
|---|---|---|
| M8a | packed INT8 weights, BF16 tile multiply (W8A16) | native INT8 arithmetic or a speedup |
| M8b | offline INT8 export and direct packed loading | support for arbitrary quantized checkpoints |
| M8c | opt-in native INT8 W8A8 on A6000 | better quality or speed than BF16 |
| M8d | FP8/MXFP8/MXFP4/NVFP4 reference codecs and metadata/header inspection | packed or native FP8/FP4 serving |

## Why serving cares about fewer bits

An $N$-parameter model needs approximately:

$$
\text{weight bytes} = N \times \frac{\text{bits per stored weight}}{8},
$$

before scales, metadata, embeddings kept at higher precision, allocator
rounding, and runtime state. An 8-billion-parameter model is therefore about
16 GB at 16 bits, 8 GB at 8 bits, or 4 GB at 4 bits in the idealized
calculation. Real checkpoints are larger than the last two numbers because
scales and mixed-precision tensors also consume space.

The performance reason is different for prefill and decode:

- **Decode** usually multiplies one or a few token rows by very large weight
  matrices. Re-reading the weights is often the dominant traffic, so packed
  weights can reduce the bytes crossing memory even if arithmetic stays BF16.
- **Prefill** exposes larger matrix multiplications. It can become compute
  bound, so the largest gain may require a kernel that uses the GPU's native
  low-precision matrix instructions rather than unpacking into BF16 first.

Neither phase receives an automatic $2\times$ gain from halving the weight
width. Scale loads, unpacking, activation quantization, non-GEMM layers,
kernel launch cost, and unfavorable matrix shapes all remain. “Half the bits”
is a storage statement, not a latency result.

## Start with a single scale

Suppose a BF16 weight row is:

$$
W = [-1.00, -0.60, -0.20, 0.00, 0.25, 0.50, 0.75, 1.00].
$$

For simple symmetric INT8 quantization, Tinyserve can choose:

$$
s = \frac{\max_i |W_i|}{127} = \frac{1}{127},
$$

then store integer codes:

$$
Q_i = \operatorname{clamp}\left(\operatorname{round}(W_i/s), -127, 127\right).
$$

The eight codes are approximately:

$$
Q = [-127, -76, -25, 0, 32, 64, 95, 127].
$$

The kernel reconstructs an approximation rather than the original values:

$$
\hat W_i = s Q_i.
$$

For example, `-76` reconstructs to about `-0.5984`, not exactly `-0.60`.
That difference is the quantization error. More generally, an affine integer
scheme uses:

$$
Q = \operatorname{clamp}\left(\operatorname{round}(W/s) + z,
q_{\min}, q_{\max}\right),
\qquad
\hat W = s(Q-z),
$$

where $z$ is the **zero point**. Symmetric quantization fixes $z=0$. An
asymmetric scheme lets a shifted integer interval cover a one-sided or skewed
range more tightly, at the price of zero-point handling in storage and
arithmetic. Weight-only LLM kernels commonly favor symmetric schemes because
their zero point is free and trained weight distributions are usually centered
near zero.

The signed INT8 datatype actually contains codes from `-128` through `127`.
Using `max_abs / 127` deliberately leaves `-128` unused so positive and
negative magnitudes share the same scale. Other specifications may use the
full asymmetric integer range; the scale rule is part of the scheme and must
not be inferred from the word `INT8`.
For an unclipped value under round-to-nearest uniform quantization, the
absolute error is at most half a step:

$$
|W_i-\hat W_i| \le \frac{s}{2}.
$$

That compact bound excludes saturation. Once a value is clipped, its error is
the distance to the clipping boundary and can be much larger. It also says
nothing about model quality: the same local weight error can be harmless in
one channel and decisive in another.

## The scale is often more important than the code width

A single tensor can mix very different ranges. Consider:

$$
W =
\begin{bmatrix}
0.02 & 0.04 & 0.06 & 0.08 \\
2.00 & 4.00 & 6.00 & 8.00
\end{bmatrix}.
$$

A per-tensor INT8 scale is set by `8.00`:

$$
s_{tensor}=8/127 \approx 0.06299.
$$

The first row becomes approximately `[0, 1, 1, 1]`; nearly all of its
structure is lost. With one scale per output row, both rows instead become
approximately `[32, 64, 95, 127]` under their own scales. The bit width did not
change. Only the **granularity** changed.

[![Per-tensor, per-channel, and per-block scaling expose different local
ranges](/assets/tinyserve/m8-quantization-granularity.svg)](/assets/tinyserve/m8-quantization-granularity.svg)

Common granularities are:

| granularity | scale count for `W[out, in]` | benefit | cost |
|---|---:|---|---|
| per tensor | 1 | minimal metadata and simple kernel | one outlier controls everything |
| per output channel | `out` | good weight-only accuracy | kernel loads one scale per output row |
| group along `in` | `out × ceil(in/group)` | follows local ranges | more metadata and scale applications |
| 2-D block | one scale per weight tile | can match hardware GEMM tiles | layout and transpose rules become important |
| per token activation | one scale per input row | adapts to each request/token | scale must be computed at runtime |

Smaller groups usually lower representation error because fewer unrelated
values compete for one range. They also increase scale storage and make the
kernel apply more scales inside the reduction. Granularity is therefore both
an accuracy choice and a kernel-layout choice.

Scale overhead can be estimated before writing a kernel. If each group has
$g$ values, each value code uses $b$ bits, and each group scale uses $b_s$
bits, then:

$$
\text{effective bits/value} \approx b + \frac{b_s}{g},
$$

before padding and global metadata. An FP16 scale for every 4-bit group of 128
adds `16/128 = 0.125` bits per value. An 8-bit scale for every 16-value NVFP4
block adds `8/16 = 0.5` bits per value. This is why a “4-bit checkpoint” is
not exactly four bits per parameter.

## Clipping trades rare large error for common small error

The simple `amax` rule maps the largest magnitude to the largest code. One
outlier can then stretch the interval so far that ordinary values round to the
same few bins. A quantizer may instead choose a clipping threshold $c$ below
the true maximum:

$$
s = \frac{c}{q_{\max}}.
$$

Values outside $[-c,c]$ saturate, but values inside receive finer resolution.
Choosing $c$ by mean-squared reconstruction error, percentile, entropy, or a
downstream calibration objective can outperform raw min/max. It also means
that “INT8 per channel” still does not fully specify the quantizer: two tools
can use the same datatype and axes while choosing different scales and
producing different weights.

## Static, dynamic, PTQ, and QAT answer different questions

These terms describe *when* information is collected or adapted:

- **Static quantization** obtains scales before serving, usually from weights
  and a calibration corpus. The inference kernel reuses them.
- **Dynamic quantization** computes some scales from the current runtime
  activation. This follows changing activation ranges but adds a reduction and
  quantization step to the request path.
- **Post-training quantization (PTQ)** quantizes an already-trained model.
  Calibration may be simple min/max or a sophisticated reconstruction search,
  but the base training is not repeated.
- **Quantization-aware training (QAT)** inserts quantization effects into the
  forward path while training or fine-tuning, allowing parameters to adapt to
  the target grid.

## Fake quantization and real quantization change different things

The phrase **fake quantization** sounds dismissive, but it is an important
numerical tool. It asks: *what would this tensor's values be after rounding to
the target format?* It does not, by itself, ask the GPU to store or multiply
those values in that format.

Tinyserve uses **real quantization** more narrowly to mean that packed codes
and their scales are the authoritative runtime representation. The complete
BF16 weight is absent from GPU memory. Native low-precision multiplication is
a further property, not a synonym for real quantization: a packed fallback is
real storage even when it reconstructs one BF16 tile at a time.

[![Fake quantization, packed fallback, and native low-precision execution keep
different tensors resident](/assets/tinyserve/m8-fake-real-quant.svg)](/assets/tinyserve/m8-fake-real-quant.svg)

The distinction has three levels:

| path | resident weight on GPU | value used by multiplication | what it proves |
|---|---|---|---|
| eager fake Q/DQ oracle | complete BF16/FP16 $\hat W$ | BF16/FP16 | numerical effect only |
| real packed fallback | codes $Q$ and scales; no complete $\hat W$ | tiles reconstructed to BF16/FP16 | packed storage and possibly lower traffic |
| real native kernel | codes $Q$, scales, and any quantized activation | matching low-precision MMA operands | packed storage and native compute |

### Fake quantization in an inference oracle

For per-output-row INT8, a minimal PyTorch-style fake quantizer is:

```python
def int8_qdq_per_row(weight):                # BF16 [out, in]
    w = weight.float()
    amax = w.abs().amax(dim=1, keepdim=True)  # [out, 1]
    scale = torch.where(amax == 0, 1.0, amax / 127.0)
    codes_as_float = torch.round(w / scale).clamp(-127, 127)
    dequant = codes_as_float * scale
    return dequant.to(weight.dtype)           # still BF16 [out, in]
```

`round` and `clamp` force every returned value onto the INT8 grid, so
`F.linear(x, int8_qdq_per_row(weight))` exposes the model to INT8 rounding
error. Nevertheless, `codes_as_float` is a floating-point carrier and the
returned weight is BF16. The ordinary BF16 linear kernel runs. If Q/DQ is
executed on every forward, its temporary code and dequantized tensors can use
*more* memory than the original layer. If $\hat W$ is computed once and
cached, the temporaries disappear but the resident weight is still full-size
BF16.

An inference oracle may cast the intermediate codes to `torch.int8` and then
cast them back before `F.linear`. That makes the intermediate codes exact, but
does not change the deployment conclusion: materializing the complete
dequantized matrix before the linear operation is still a Q/DQ reference, not
a packed serving kernel.

Q/DQ nodes do not force one execution strategy. In an eager PyTorch oracle,
dequantization really produces the floating tensor consumed by `F.linear`.
An optimizing compiler may instead recognize an explicit Q/DQ pattern, retain
the packed operand, and lower it to a low-precision kernel. That compiled path
is real quantization. Classify the result from resident tensors and executed
instructions, not from the mere presence of `Quantize` and `Dequantize` nodes
in the graph.

For a `4096 × 4096` matrix:

- BF16 $W$ or $\hat W$ occupies exactly 32 MiB;
- INT8 codes occupy 16 MiB;
- 4,096 FP16 row scales add only 8 KiB;
- fake Q/DQ still needs the 32 MiB dequantized matrix, while a real INT8
  runtime needs roughly 16.008 MiB for persistent weight data.

This comparison excludes allocator rounding and workspace, but it makes the
ownership boundary visible. Counting an INT8 temporary while retaining BF16
$W$ is not a 2× model-memory saving.

### Fake quantization during QAT

Direct rounding has zero derivative almost everywhere, so training would not
learn useful weight updates through it. QAT commonly uses a
**straight-through estimator (STE)**. A minimal expression is:

```python
def qat_int8_per_row(weight):
    dequant = int8_qdq_per_row(weight)
    return weight + (dequant - weight).detach()
```

The forward value equals `dequant`; the detached correction has no gradient,
so the backward path treats the operation approximately like the identity.
Production QAT frameworks add observers, learned or delayed scales, enable
flags, and carefully defined gradient behavior. The important serving fact is
unchanged: QAT teaches floating-point master weights to tolerate a grid. A
separate export step must still create packed codes and checkpoint metadata.

### Real quantization starts at packing

The following snippets illustrate the separation used by M8a in
[`tinyserve/quantization.py`](https://github.com/kaix-nv/tinyserve/blob/3846ae7c1883acfe483bc1c350db277d9d826ef0/tinyserve/quantization.py). The implementation
keeps the codec, the full Q/DQ oracle, and the packed serving path separate so
their tensor lifetimes can be inspected directly. The generic `spec` and
`quantized_linear` below are schematic interfaces, not Tinyserve APIs: the
actual serving module currently implements INT8 only.

An offline INT8 packer changes which tensor is authoritative:

```python
def pack_int8_per_row(weight):                # CPU/offline conversion
    w = weight.float()
    amax = w.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax == 0, 1.0, amax / 127.0)
    codes = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return codes.contiguous(), scale.squeeze(1).to(torch.float32)
```

The checkpoint now stores `codes`, `scale`, the logical matrix shape, and the
scale/layout metadata. A real runtime retains those tensors instead of
immediately rebuilding a full `nn.Linear.weight`:

```python
class QuantizedLinear(nn.Module):
    def __init__(self, codes, scales, spec):
        super().__init__()
        self.register_buffer("weight_packed", codes)
        self.register_buffer("weight_scale", scales)
        self.spec = spec                       # format, axes, block, layout

    def forward(self, x):
        return quantized_linear(x, self.weight_packed,
                                self.weight_scale, self.spec)
```

The module has no full-precision weight parameter. `quantized_linear` can now
choose one of two honest implementations.

A packed W8A16 fallback keeps activations in BF16 and expands only the weight
tile currently being consumed:

```text
for each output tile n0:n1:
    q_tile = load INT8 codes [n1-n0, K]
    s_tile = load row scales [n1-n0, 1]
    w_tile = BF16(FP32(q_tile) * s_tile)
    y[..., n0:n1] = x_BF16 @ transpose(w_tile)
```

The temporary tile is overwritten by the next tile, so no complete
`[out, in]` BF16 weight exists. This path can reduce persistent memory and HBM
weight traffic, but its multiplication remains BF16. Tinyserve's first A6000
W8A16 kernel belongs in this category unless a measured specialized
mixed-input instruction path is used.

A native W8A8 path additionally quantizes each activation row, performs an
INT8-by-INT8 dot product with a wider accumulator, then applies both scales:

```text
qx, sx = quantize_each_runtime_activation_row(x)
acc_i32 = int8_mma(qx, qw)
y = BF16(acc_i32 * sx * sw)
```

FP8, MXFP8, MXFP4, and NVFP4 follow the same systems rule but use their own
element decoders, scale layouts, block partial sums, and hardware
instructions. A native format name in checkpoint metadata is not sufficient;
the running kernel must actually consume those packed operands.

### What changes in Tinyserve

The current [`dequantize_mxfp4()` loader path](https://github.com/kaix-nv/tinyserve/blob/3846ae7c1883acfe483bc1c350db277d9d826ef0/tinyserve/loader.py) reads
two packed E2M1 values per byte, expands E8M0 scales, and returns one complete
model-dtype tensor. `load_model()` then assigns it to an ordinary
[`nn.Linear` projection](https://github.com/kaix-nv/tinyserve/blob/3846ae7c1883acfe483bc1c350db277d9d826ef0/tinyserve/models/kimi.py) before moving the model
to the GPU. This is useful checkpoint-decoding evidence, but the runtime is
BF16: packed-on-disk does not mean packed-in-VRAM.

The intended multi-format serving design changes that boundary in four steps:

1. inspect per-layer quantization metadata before building the final module;
2. replace eligible attention and MLP projections with `QuantizedLinear`;
3. load packed data and scales into registered buffers without calling
   `.to(dtype)` on the codes;
4. dispatch the Q/DQ oracle, packed fallback, or native kernel by format,
   device capability, and supported shape.

M8a–M8c implement this separation for INT8, not a general multi-format
dispatcher. M8d's floating-point codecs and inspector are standalone reference
tools; they do not replace the existing Kimi loader or its BF16 execution.

The oracle remains available for parity tests. The packed path is accepted as
real quantization only when inspection shows no complete BF16 weight buffer.
The native path earns its stronger label only after the compiled instruction
stream and a target-GPU profile match the intended operand format.

## Tensor lifetime changes the quantization problem

“Quantize the model” can refer to tensors with very different lifetimes:

| tensor | when it is created | how often it is reused | characteristic risk |
|---|---|---|---|
| weights | offline, then loaded once | every request and token | model-wide quality drift; dominant resident bytes |
| activations | during each layer call | usually consumed immediately | runtime scale overhead and outliers |
| KV cache | once per layer and token | read by every later token | error persists for the remaining sequence |
| recurrent state | updated every token | fed back into its own next update | error can accumulate through recurrence |
| logits | once per generated position | immediately sampled | small drift can flip a near-tied token |

Weights are the cleanest first serving target because they are static and
their scales can be computed offline. Activation quantization must pay or hide
a runtime range calculation. KV and recurrent state are not merely more
weights: they are request-owned mutable memory with time-dependent error.
That is why M8 explains them but does not quietly fold them into the first
weight-only implementation.

## Weight-only and weight-activation quantization are different kernels

For a linear layer:

$$
Y = XW^\top,
$$

`W8A16` means 8-bit weights and 16-bit activations. A packed fallback can load
a tile of $Q_W$, reconstruct a BF16/FP16 tile using its scales, and feed that
tile to ordinary tensor-core multiplication. This can save weight bandwidth,
but its multiply is not native INT8 merely because the checkpoint is INT8.

With `W8A8`, both operands are quantized. If each activation row has scale
$s_X[m]$ and each weight row has scale $s_W[n]$, the output is approximately:

$$
Y_{mn}
\approx
s_X[m]s_W[n]
\sum_k Q_X[m,k]Q_W[n,k].
$$

The inner sum can use a low-precision dot product and a wider accumulator.
When scales vary along the reduction dimension, they cannot be pulled outside
the complete sum. For block $b$:

$$
Y_{mn}
\approx
\sum_b s_X[m,b]s_W[n,b]
\left(\sum_{k \in b}Q_X[m,k]Q_W[n,k]\right).
$$

That equation explains why block-scaled hardware and scale layout matter: each
partial dot product must be paired with the correct activation and weight
scales before the partial results are combined.

High-precision accumulation is still normal. “FP8 GEMM” generally describes
the input operands, not an FP8 running sum or FP8 model output. Accumulation,
bias, nonlinearities, residual additions, normalization, and final output can
use wider types.

## Format, scheme, method, and checkpoint are not synonyms

Four names often get mixed together:

- **Datatype/format:** INT4, INT8, E4M3, E5M2, E2M1, or E8M0 defines bit-level
  values.
- **Scheme/recipe:** W4A16, W8A8, MXFP8, or NVFP4 adds scale type,
  granularity, dynamic/static behavior, and often accumulator rules.
- **Quantization method:** GPTQ, AWQ, SmoothQuant, and related methods choose
  or transform the quantized parameters. They are not four-bit datatypes.
- **Checkpoint serialization:** GGUF, an AWQ Hugging Face checkpoint, or a
  ModelOpt unified Hugging Face checkpoint defines tensor names, packing,
  layout, metadata, and sharding.

For example, AWQ is an activation-aware *method* for selecting a weight-only
quantization transformation. It can produce integer weights, but “AWQ” alone
does not tell a kernel the nibble order or scale tensor layout. GPTQ uses
approximate second-order information to reduce layer reconstruction error.
SmoothQuant moves activation outlier difficulty into weights through an
equivalent rescaling so W8A8 becomes easier. These methods solve different
accuracy problems; none substitutes for an execution contract.

## How a floating-point bit pattern becomes a value

An integer code and a floating-point code spend their bits differently. An
INT8 quantizer first chooses one external step size $s$; adjacent codes are
always one step apart:

$$
\hat x = s q,
\qquad
q \in \mathbb{Z}.
$$

A binary floating-point code instead divides its own bits into a **sign**,
an **exponent**, and a **fraction** (also called mantissa bits). For an ordinary
normalized finite code:

$$
x = (-1)^S
\times 2^{E-\text{bias}}
\times \left(1 + \frac{F}{2^M}\right),
$$

where $S$ is the sign bit, $E$ is the unsigned exponent field, $F$ is the
unsigned fraction field, and $M$ is the number of fraction bits. The bias lets
an unsigned exponent field describe both negative and positive powers of two.
Codes with an all-zero exponent represent zero and tiny **subnormal** values;
codes at the top of the exponent range may represent large finite values,
infinity, or NaN depending on the exact format specification.

[![Floating-point formats divide bits between sign, exponent, and fraction,
then block-scaled recipes add an external scale](/assets/tinyserve/m8-floating-point-formats.svg)](/assets/tinyserve/m8-floating-point-formats.svg)

The name `E4M3` means four exponent bits and three fraction bits. The sign bit
is normally left implicit in that name, so E4M3 occupies `1 + 4 + 3 = 8`
bits. Likewise, E2M1 occupies four bits. E8M0 is different: it has eight
exponent bits, no sign, and no fraction; it is used as a positive
power-of-two **scale**, not as the signed tensor element.

The exponent controls range; the fraction controls how many values fit inside
each power-of-two interval. Take the E4M3 bits:

```text
0 | 0111 | 100
S |   E  |  F
```

Here $S=0$, $E=7$, the E4M3 bias is 7, and `100` is $F=4$. Therefore:

$$
x = (+1) \times 2^{7-7} \times (1 + 4/8) = 1.5.
$$

Around 1, E4M3 has values separated by $2^{-3}=0.125$. Around 8, the
exponent is three larger, so the separation is $2^{3-3}=1$. Floating point
therefore has approximately constant *relative* resolution: large values get
larger absolute gaps. A uniformly scaled integer keeps the same absolute gap
everywhere inside its clipping range.

E5M2 spends one more bit on the exponent and one fewer on the fraction than
E4M3. That is why it covers much larger magnitudes but distinguishes fewer
nearby values. Forward weights and activations usually benefit from E4M3's
extra precision; E5M2 is more commonly useful for tensors such as training
gradients whose range is harder to bound. An inference checkpoint saying only
`FP8` is still ambiguous: the loader needs the E4M3/E5M2 choice and the scale
contract.

FP4 E2M1 is small enough to enumerate. Ignoring the duplicated sign of zero,
its values are:

$$
0,\ \pm0.5,\ \pm1,\ \pm1.5,\ \pm2,\ \pm3,\ \pm4,\ \pm6.
$$

For example, `0 | 01 | 1` represents $+1.5$, while the subnormal code
`0 | 00 | 1` represents $+0.5$. There is no E2M1 code for `5`, `5.5`, or
`5.75`: they must round to a nearby code after scaling. This very coarse
codebook is why FP4 is nearly always paired with fine-grained block scales.

The most important naming rule is:

> **E2M1 and E4M3 describe element codebooks. MXFP4, MXFP8, and NVFP4
> describe codebooks plus scale formats and grouping rules.**

That separation explains three easily confused schemes:

$$
\begin{aligned}
\text{MXFP8:}\quad
  &\hat x_i = q_i^{E4M3}\,s_{\lfloor i/32 \rfloor}^{E8M0}, \\
\text{MXFP4:}\quad
  &\hat x_i = q_i^{E2M1}\,s_{\lfloor i/32 \rfloor}^{E8M0}, \\
\text{NVFP4:}\quad
  &\hat x_i = q_i^{E2M1}\,s_{\lfloor i/16 \rfloor}^{E4M3}\,s_{global}^{FP32}.
\end{aligned}
$$

MXFP4 and NVFP4 use the same four-bit E2M1 data codebook. They are not the
same format because MXFP4 shares one power-of-two E8M0 scale across 32 values,
whereas NVFP4 shares a fractional E4M3 scale across 16 values and adds one
FP32 global scale. Changing the scale metadata while leaving the packed
nibbles untouched changes every reconstructed value.

## Numerical limits: NaNs, infinities, normals, and subnormals

Bit counts alone do not define a floating-point format. We also need its
exponent bias, which patterns are special, and what happens near zero.
The following reference uses **positive, unscaled magnitudes**. The signed
formats have matching negative values and both `+0` and `-0`.
"Denormal" and "subnormal" mean the same thing here.

### Special values and exponent bias

**NaN** means "not a number": a marker for an invalid or undefined numerical
result, not a very large value. **Infinity** is a separate signed value beyond
every finite magnitude. Whether an overflowing *conversion* saturates to a
finite endpoint, returns infinity, or produces NaN is a conversion policy,
not something the presence of an infinity encoding decides by itself.

In this table, $E$ and $F$ are the unsigned exponent and fraction fields;
the sign bit is omitted from the special-pattern rules.

| format | sign / exponent / fraction bits | exponent bias | NaN encodings | infinity encodings |
|---|---|---:|---|---|
| FP32 | 1 / 8 / 23 | 127 | $E=255,\ F\ne0$ | $E=255,\ F=0$ |
| BF16 | 1 / 8 / 7 | 127 | $E=255,\ F\ne0$ | $E=255,\ F=0$ |
| FP16 | 1 / 5 / 10 | 15 | $E=31,\ F\ne0$ | $E=31,\ F=0$ |
| FP8 E4M3FN | 1 / 4 / 3 | 7 | $E=15,\ F=7$: bytes `0x7f`, `0xff` | none |
| FP8 E5M2 | 1 / 5 / 2 | 15 | $E=31,\ F\ne0$ | $E=31,\ F=0$ |
| FP4 E2M1 | 1 / 2 / 1 | 1 | none | none |
| E8M0 scale | 0 / 8 / 0 | 127 | byte `0xff` | none |

Here **E4M3FN** is the OCP/NVIDIA finite-with-NaNs encoding, exposed by
PyTorch as `float8_e4m3fn`; this chapter shortens its name to E4M3 elsewhere.
Other FP8 variants, such as FNUZ, must not inherit this table by name alone.
See the [OCP MX encoding tables](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
and NVIDIA's [E4M3 definition](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8__e4m3.html).
NVIDIA also documents the concrete [BF16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__BFLOAT16__CONSTANTS.html)
and [FP16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__HALF__CONSTANTS.html)
special-value and endpoint constants.

The bias shifts the exponent's origin: an ordinary E4M3 exponent field `1`
means $1-7=-6$, not $+1$. Increasing only the bias by one would halve the
finite values while leaving their max/min ratio unchanged. Extra exponent
**bits** can widen the range; a different **bias** primarily moves it.

Do not apply the IEEE-style "all exponent bits set means special" rule to
every format. E4M3 uses most of its $E=15$ patterns for finite numbers:
`0 | 1111 | 110` is $2^8(1+6/8)=448$. Only the final fraction pattern is
NaN. E2M1 reserves no patterns for NaN or infinity at all. These choices
recover finite values from a very small code budget, but require explicit
nonfinite-input handling in the converter. M8d rejects NaN/Inf inputs and
saturates finite overflow; that is its chosen reference policy.

### Normal and denormal endpoints

A normal value has an implicit leading **1**. When the exponent field is
zero, a subnormal instead has a leading **0** and holds the exponent at
$1-\text{bias}$:

$$
x_{sub}=(-1)^S\,2^{1-\text{bias}}\frac{F}{2^M}.
$$

For these signed formats, the endpoints follow directly:

$$
\begin{aligned}
x_{min,normal} &= 2^{1-\text{bias}}, \\
x_{min,sub} &= 2^{1-\text{bias}-M}, \\
x_{max,sub} &= (1-2^{-M})\,2^{1-\text{bias}}.
\end{aligned}
$$

The maximum normal must additionally respect the reserved special patterns.
Exact powers-of-two expressions below are authoritative; decimals are rounded.

**Normal values**

| format | max normal / max finite | min normal |
|---|---|---|
| FP32 | $(2-2^{-23})2^{127}\approx3.4028235\times10^{38}$ | $2^{-126}\approx1.1754944\times10^{-38}$ |
| BF16 | $(2-2^{-7})2^{127}\approx3.3895314\times10^{38}$ | $2^{-126}\approx1.1754944\times10^{-38}$ |
| FP16 | $65{,}504$ | $2^{-14}\approx6.1035156\times10^{-5}$ |
| FP8 E4M3FN | $448$ | $2^{-6}=0.015625$ |
| FP8 E5M2 | $57{,}344$ | $2^{-14}\approx6.1035156\times10^{-5}$ |
| FP4 E2M1 | $6$ | $1$ |

**Denormal (subnormal) values**

| format | max denorm | min denorm / min positive |
|---|---|---|
| FP32 | $(1-2^{-23})2^{-126}$ | $2^{-149}\approx1.4012985\times10^{-45}$ |
| BF16 | $(1-2^{-7})2^{-126}$ | $2^{-133}\approx9.1835496\times10^{-41}$ |
| FP16 | $1023\times2^{-24}\approx6.0975552\times10^{-5}$ | $2^{-24}\approx5.9604645\times10^{-8}$ |
| FP8 E4M3FN | $7\times2^{-9}=0.013671875$ | $2^{-9}=0.001953125$ |
| FP8 E5M2 | $3\times2^{-16}\approx4.5776367\times10^{-5}$ | $2^{-16}\approx1.5258789\times10^{-5}$ |
| FP4 E2M1 | $0.5$ | $0.5$ |

For example, E4M3 has seven positive subnormals: $1/512$ through $7/512$.
The next value, $8/512=1/64$, is the smallest normal. The gap at this
boundary stays $1/512$; there is no sudden jump in spacing.

[![E4M3 fills the gap between zero and its first normal with seven evenly
spaced subnormals](/assets/tinyserve/m8-subnormal-boundary.svg)](/assets/tinyserve/m8-subnormal-boundary.svg)

Subnormals provide **gradual underflow**, not full relative precision. Their
absolute step stays fixed as values approach zero, so the relative rounding
error grows. The format can encode these values, but a particular instruction
or compiler mode may flush them to zero. For example, NVIDIA documents this
distinction for FP32 in its [floating-point compiler flags](https://docs.nvidia.com/cuda/floating-point/index.html#compiler-flags).
A stored-format table is not proof that every execution path preserves its
smallest values.

### Dynamic range needs a denominator

"Dynamic range" is ambiguous unless we say which minimum we mean. Here it
is a **dimensionless ratio**, not the signed interval and not a count of
significant bits:

$$
R_{normal}=\frac{x_{max,finite}}{x_{min,normal}},
\qquad
R_{all}=\frac{x_{max,finite}}{x_{min,positive}}.
$$

The second includes subnormals, but never zero, NaN, or infinity. If reporting
range in binary orders of magnitude, use $\log_2 R$ and identify which $R$.

| format | normal-only range $R_{normal}$ | including denormals $R_{all}$ |
|---|---:|---:|
| FP32 | $\approx2.8948021\times10^{76}$ | $\approx2.4283360\times10^{83}$ |
| BF16 | $\approx2.8834944\times10^{76}$ | $\approx3.6908728\times10^{78}$ |
| FP16 | $1{,}073{,}217{,}536$ | $1{,}098{,}974{,}756{,}864$ |
| FP8 E4M3FN | $28{,}672$ | $229{,}376$ |
| FP8 E5M2 | $939{,}524{,}096$ | $3{,}758{,}096{,}384$ |
| FP4 E2M1 | $6$ | $12$ |

These ratios are calculated from the endpoint table, not hardware throughput
claims. For example, E4M3 gives $448/(1/512)=229{,}376$, about 17.8 binary
orders of magnitude, despite storing each value in only eight bits. That does
**not** give it 17.8 bits of precision: it still has only three fraction bits.

**E8M0 is a different case.** It has no sign, zero, fractional significand,
or subnormal encoding. Valid scale bytes `0x00` through `0xfe` mean
$2^{-127}$ through $2^{127}$; `0xff` means NaN. Thus its finite scale ratio is
$2^{254}\approx2.8948022\times10^{76}$. Normal/denormal minima and maxima are
not separate categories for this scale-only encoding. Byte zero denotes the
*smallest scale*, not a zero scale.

### What the limits imply for quantization

**BF16 versus FP16: range is not precision.** BF16's normal exponent range
matches FP32's, but its largest finite value is slightly smaller and its
subnormal tail is much shorter. FP16 has a far narrower range, yet three more
fraction bits than BF16. Near 1, $1+2^{-10}$ is exact in FP16 but rounds to
1 in BF16. Conversely, $2^{16}=65{,}536$ is finite in BF16 but beyond FP16's
largest finite value. Neither datatype dominates both dimensions.

**E4M3 versus E5M2: small values can matter as much as outliers.** At scale
1, E4M3's minimum positive value is $2^{-9}$, so its round-to-nearest zero
boundary is $2^{-10}$. At that exact midpoint, ties-to-even chooses zero.
This gives a small, exact underflow example:

| input, before any external scaling | E4M3 result | E5M2 result |
|---|---|---|
| $2^{-11}=0.00048828125$ | $0$ | exact |
| $2^{-10}=0.0009765625$ | $0$ (tie) | exact |
| $3\times2^{-11}=0.00146484375$ | $2^{-9}=0.001953125$ | exact |

E5M2 preserves these values, but gives up one fraction bit throughout its
normal range. The right question is whether a workload loses more from
clipping/underflow or from coarser rounding of the values it already covers.

**FP4 needs local scales, not just a large global range.** For one block with
a fixed positive effective scale $c$, the E2M1 codebook spans nonzero
magnitudes from $c/2$ to $6c$. Its ratio remains **12**:

$$
\frac{6c}{c/2}=12.
$$

Scaling moves that interval but cannot widen it or add values between its
codes. With an ideal scale $c=1000/6$, a block maximum of `1000` is exact,
but its smallest positive reconstruction is about `83.33`; a weight of `1`
rounds to zero. Moving those weights into different scale groups can preserve
both. A bigger global scale alone cannot.

This is how to read MXFP4 and NVFP4 without inventing new element limits:

| scheme | effective scale $c$ for one block | element limits within that block |
|---|---|---|
| MXFP8 E4M3 / E5M2 | one E8M0 scale per 32 values | corresponding FP8 endpoints above, all multiplied by $c$ |
| MXFP4 | one E8M0 scale per 32 values | min positive $c/2$, max $6c$; ratio 12 |
| NVFP4 | one E4M3 block scale per 16 values times an FP32 global scale | min positive $c/2$, max $6c$; ratio 12 when $c>0$ |

Different blocks can cover very different intervals, so a complete tensor can
span much more than 12:1. E8M0 offers a huge choice of scale exponents but
only powers of two. NVFP4's E4M3 scales offer finer local choices and smaller
groups, while the FP32 global scale moves the tensor-wide range. Neither
changes E2M1's eight nonnegative magnitudes. A rounded NVFP4 block scale of
zero makes that whole block zero; the positive-scale ratio then no longer
applies. Finally, reconstructed products must still fit the destination and
accumulator types: an enormous formal scale range is not an unlimited
execution range.

Special values can also live in the **scale**, not only in the element.
MXFP4's E2M1 codes contain no NaN, but a NaN E8M0 scale marks its whole block
as NaN under the MX contract. M8d's matrix Q/DQ interface instead rejects
nonfinite scales. Inspecting four-bit data alone cannot establish that the
decoded tensor is finite.

The endpoint and rounding examples are checked against actual code patterns
in [`tests/test_quant_formats.py`](https://github.com/kaix-nv/tinyserve/blob/3846ae7c1883acfe483bc1c350db277d9d826ef0/tests/test_quant_formats.py). They test
representation and conversion, not preservation by an unimplemented FP4
serving kernel.

## How do we choose exponent and fraction bits?

If a signed floating-point element has a fixed budget of $B$ bits, then:

$$
1 + E + M = B.
$$

The sign already consumes one bit, so exponent bits $E$ and fraction bits $M$
compete for what remains. Giving a bit to one takes a bit from the other.

| spend more bits on | what improves | what gets worse |
|---|---|---|
| exponent | dynamic range; fewer large values overflow and fewer small values underflow | fewer fraction bits, so ordinary values round more coarsely |
| fraction | resolution between nearby values; lower in-range rounding error | fewer exponent codes, so the smallest and largest magnitudes move closer together |
| special/subnormal encodings | zero-neighborhood behavior, NaN, or infinity semantics | fewer bit patterns remain for ordinary finite values |

For a normal value in the interval $[2^e,2^{e+1})$, $M$ fraction bits give a
spacing of:

$$
\Delta = 2^{e-M}.
$$

Round-to-nearest introduces at most half that spacing before clipping. Near
1, the gap is $2^{-M}$ and the maximum rounding error is $2^{-(M+1)}$.
That makes the fraction-bit trade-off concrete:

| format | gap near 1 | maximum rounding error near 1 | maximum finite magnitude |
|---|---:|---:|---:|
| E4M3 | $0.125$ | $0.0625$ | $448$ |
| E5M2 | $0.25$ | $0.125$ | $57{,}344$ |
| E2M1 | $0.5$ | $0.25$ | $6$ |

These are properties of the unscaled codebooks. An external scale multiplies
both the range and every gap by the same amount.

[![E4M3 spends more bits on precision while E5M2 spends more on range, and
block scaling can reduce the required range](/assets/tinyserve/m8-exponent-mantissa-tradeoff.svg)](/assets/tinyserve/m8-exponent-mantissa-tradeoff.svg)

### Example 1: E4M3 or E5M2?

Assume the external scale is fixed at $s=1$ and we need to represent two
positive values:

$$
X = [1.10,\ 1000].
$$

Around 1, E4M3 contains `1.000`, `1.125`, `1.250`, and so on. E5M2 contains
only `1.00`, `1.25`, `1.50`, and `1.75`. Their rounded results are:

| value | E4M3 result | absolute error | E5M2 result | absolute error |
|---:|---:|---:|---:|---:|
| $1.10$ | $1.125$ | $0.025$ | $1.00$ | $0.10$ |
| $1000$ | $448$ after clipping | $552$ | $1024$ | $24$ |

E4M3 preserves the ordinary value four times more closely, but cannot cover
the large value under this scale. E5M2 sacrifices nearby-value resolution to
keep the outlier finite. Neither split is universally better; the right answer
depends on which error dominates for the tensor and the model.

The scale can move the clipping boundary. Choosing $s=1000/448$ lets E4M3
cover the outlier, but it also multiplies every E4M3 gap by about `2.23`. If
one outlier determines a tensor-wide scale, thousands of ordinary values pay
for its range even though they are not themselves large.

### Example 2: change the scale group before changing the datatype

Consider this toy E2M1 tensor, split visually into two groups of four:

$$
X = [0.6,\ 0.8,\ 1.2,\ 0.0\;|\;100,\ 100,\ 100,\ 100].
$$

With one scale for all eight values, the largest magnitude chooses:

$$
s_{tensor} = 100/6 \approx 16.67.
$$

The first group is divided by `16.67`, producing values between `0` and
`0.072`. The smallest nonzero E2M1 magnitude is `0.5`, so all three nonzero
small values round to zero:

$$
\hat X_{tensor}
= [0,\ 0,\ 0,\ 0\;|\;100,\ 100,\ 100,\ 100].
$$

Now give each four-value group its own scale. The first group uses
$s_0=1.2/6=0.2$, so its normalized values are `[3, 4, 6, 0]`—all exact E2M1
codes. The second group keeps $s_1=100/6$. Reconstruction becomes:

$$
\hat X_{groups}
= [0.6,\ 0.8,\ 1.2,\ 0.0\;|\;100,\ 100,\ 100,\ 100].
$$

This example uses ideal, unconstrained scales and groups of four only to fit
on the page. Its `0.2` and `100/6` scales have not been rounded into E8M0 or
E4M3; real MXFP4/NVFP4 need not reconstruct this example exactly. MXFP4 uses
32-value blocks and NVFP4 commonly uses 16-value blocks. The mechanism is the
same: smaller groups reduce the dynamic range the element codebook must cover.
They can preserve more information without changing E2M1, but require more
scale bytes, more scale loads, and stricter kernel layouts.

### A practical selection procedure

In a serving engine, the choice is not an unconstrained search over imaginary
formats such as E3M4. The checkpoint ABI and target GPU first define which
formats their kernels can consume. Tinyserve should then choose among those
supported candidates:

1. **Choose the tensor role.** Static weights, dynamic activations, KV cache,
   recurrent state, and gradients have different distributions and different
   consequences when an error persists.
2. **Choose a scale boundary.** Start with the coarsest kernel-supported
   granularity, then measure whether outliers force unrelated values into poor
   bins.
3. **Count overflow and underflow separately.** Too many clipped or zeroed
   values indicate insufficient range; try more exponent bits or smaller scale
   groups.
4. **Measure in-range rounding.** If values fit but nearby values collapse,
   prefer more fraction bits, more total bits, or a more expressive scale.
5. **Check the scale format itself.** E8M0 covers an enormous scale range but
   only powers of two. E4M3 scales represent fractional values more closely
   but need a different hardware and metadata path.
6. **Keep accumulation separate.** Choosing E4M3, E5M2, or E2M1 for operands
   does not imply that partial sums should use the same narrow format.
7. **Validate the model and the system.** Codec error selects candidates;
   held-out logits and task quality decide whether they are acceptable, and
   target-GPU measurements decide whether the supported format is useful.

The usual choices now make more sense. Forward weights and activations often
prefer E4M3 because scaling can control their range and three fraction bits
reduce rounding. Training gradients may prefer E5M2 because their range is
harder to bound. FP4 inference accepts the coarse E2M1 codebook only by pairing
it with fine-grained scales such as MXFP4 or NVFP4.

## A map of the formats M8 needs to explain

The following table uses the inference-facing definitions relevant to
Tinyserve. Training recipes can add transpose copies, stochastic rounding,
gradient formats, amax history, and two-dimensional weight scales.

| name | element code | scale organization | nominal payload | central tradeoff |
|---|---|---|---:|---|
| INT8 | signed two's-complement integer | tensor, axis, or group; float scale | 8 bits/value | simple and widely supported, but uniform spacing |
| INT4 | signed four-bit integer | commonly blocks of 64 or 128 | 4 bits/value | compact weight-only storage; aggressive rounding |
| FP8 | usually E4M3 for inference | commonly one floating scale per tensor/axis | 8 bits/value | floating dynamic range inside each code |
| block-scaled FP8 | E4M3 | recipe-specific blocks and floating scales | 8 bits/value + scales | finer local ranges; layout is not standardized by `FP8` alone |
| MXFP8 | E4M3 or E5M2 | one E8M0 scale per 32 values | 8 bits/value + scales | local ranges; scale is power-of-two |
| MXFP4 | E2M1, two values per byte | one E8M0 scale per 32 values | 4 bits/value + scales | very compact; coarse values and scale |
| NVFP4 | E2M1, two values per byte | E4M3 scale per 16 values plus global FP32 scale | 4 bits/value + scales | finer, fractional local scales; Blackwell-native |

[![INT8, FP8, MXFP8, MXFP4, and NVFP4 combine element bits with different
scale structures](/assets/tinyserve/m8-quantization-formats.svg)](/assets/tinyserve/m8-quantization-formats.svg)

### INT8: uniformly spaced codes

INT8 supplies 256 integer bit patterns. After a scale and optional zero point
are chosen, adjacent codes are separated by the same real-valued step $s$.
That uniform grid is easy to implement and debug. Per-output-channel symmetric
W8A16 is therefore a useful first Tinyserve oracle: every weight row has one
scale and one directly inspectable integer row.

INT8 can also be W8A8. In that case activation range collection and INT32
accumulation become part of the kernel contract. An INT8 checkpoint by itself
does not say whether activations are also INT8.

### INT4: integer nibbles, not FP4

INT4 normally stores two signed four-bit codes in one byte. The integer range
has uniform spacing after scaling. FP4 also occupies four bits, but its E2M1
codes have an exponent and therefore nonuniform spacing. Treating them as
interchangeable corrupts every value except accidental overlaps.

NVIDIA TensorRT's current explicit INT4 scheme is weight-only with per-block
scales and supported block sizes of 64 or 128. AWQ and GPTQ commonly target
low-bit integer weight-only deployment, but their calibration algorithms and
checkpoint layouts remain separate concerns.

### FP8 E4M3: a small floating-point value plus an external scale

E4M3 uses one sign bit, four exponent bits, and three mantissa bits. NVIDIA's
inference-facing E4M3 definition has a maximum finite magnitude of 448. An
external scale maps the tensor's useful range into those codes:

$$
Q = \operatorname{cast}_{E4M3}
\left(\operatorname{clip}(X/s,-448,448)\right),
\qquad
\hat X = sQ.
$$

Compared with INT8, FP8 spends some of its bits on an exponent. It represents
a wider range of magnitudes inside one scale, but has fewer mantissa bits for
nearby values. E5M2 offers still more range and less precision and is often
discussed for gradients; M8 inference begins with E4M3.

“FP8” still leaves scale granularity open. Per-tensor FP8, row-wise FP8, and
block-scaled FP8 share an element format but require different metadata and
kernels.

### Block-scaled FP8 is not automatically MXFP8

A deployment recipe can divide FP8 weights or activations into arbitrary
hardware-supported tiles and attach ordinary floating-point scales. One common
LLM pattern uses activation scales over `1 × 128` regions and weight scales
over `128 × 128` regions. The partial dot products are FP8, while scale
application and accumulation follow the block equation above.

MXFP8 is one *specific* block-scaled scheme: its block contains 32 consecutive
values and its scale is E8M0. A checkpoint labeled only `FP8_PB` or
“block FP8” must not be decoded as MXFP8 until its block shape, scale datatype,
axis, and layout confirm that match. The local Kimi conversion report uses the
more specific `FP8_PB_WO` label for its attention weights; M8 must preserve
that distinction in the loader.

### MXFP8: microscaling with power-of-two block scales

MXFP8 divides the last dimension into blocks of 32 FP8 values. The OCP format
allows E4M3 or E5M2; E4M3 is the example here, and M8d provides both reference
codecs. Each block has an E8M0 scale. E8M0 has exponent bits but no mantissa,
so its finite scales are powers of two. Conceptually:

$$
\hat X_i = Q^{E4M3}_i \times s^{E8M0}_{block(i)}.
$$

For 32 values, the payload is 32 data bytes plus one scale byte: 8.25 bits per
value before alignment and other metadata. The extra quarter bit buys a local
range for every 32 values. Because the scale direction is part of the
representation, simply transposing an MX tensor does not generally produce the
same values as quantizing the transpose from the original high-precision
tensor.

### MXFP4: the same microscaling idea with E2M1 values

MXFP4 uses 4-bit E2M1 values and one E8M0 power-of-two scale per 32 values.
E2M1 represents the magnitudes:

$$
0,\ 0.5,\ 1,\ 1.5,\ 2,\ 3,\ 4,\ 6
$$

and their negatives. Two values fit in each byte. A 32-value block therefore
uses 16 payload bytes plus one scale byte, or 4.25 bits per value before
alignment.

Tinyserve already decodes this kind of packed E2M1 plus exponent-only scale in
the small Kimi structural fixture. It currently expands those expert weights
to the model dtype during loading. That proves a value-decoding rule, not
compressed serving: the GPU ultimately holds BF16 weights. M8 must retain the
packed tensor through execution to claim a memory saving.

### NVFP4: smaller blocks and more precise scales

NVFP4 keeps E2M1 data but changes the scale hierarchy:

$$
\hat X_i = Q^{E2M1}_i
\times s^{E4M3}_{block(i)}
\times s^{FP32}_{global}.
$$

The usual inference block contains 16 values. Its E4M3 block scale can encode
fractional values instead of snapping every scale to a power of two. Because
E4M3 does not cover the full global range on its own, one FP32 tensor scale
normalizes all block scales. A 16-value block contains 8 packed payload bytes
and one block-scale byte: 4.5 bits per value before the global scale,
alignment, and mixed-precision tensors.

The smaller block and more expressive scale usually represent local ranges
more closely than MXFP4, but they require twice as many block scales plus a
global scale. NVIDIA's training recipe can use 16-by-16 weight scale blocks
while activations use one-dimensional blocks of 16. The exact axis and layout
must therefore travel with the tensor; the name `NVFP4` alone is insufficient.

For dynamically quantized activations, NVFP4 also quantizes the scale. The
runtime first reduces 16 values to a block `amax`, derives an ideal block
scale, normalizes that scale by the tensor-wide FP32 scale, and casts the
result to E4M3. The E2M1 activation codes are then produced using the
reconstructed scale. This two-stage approximation is why NVIDIA documentation
calls the operation **dynamic double quantization**.

## NVIDIA hardware changes what “supported” means

Tinyserve develops on RTX A6000 GPUs (Ampere, SM86). That machine can validate
all codec equations and physically packed fallback paths, but it cannot prove
native FP8, MXFP8, MXFP4, or NVFP4 tensor-core execution.

| GPU family | INT8 | ordinary FP8 | MXFP8/MXFP4 | NVFP4 |
|---|---|---|---|---|
| Ampere / A6000 | native instructions available | no native path | no native path | no native path |
| Hopper / H100 | native | native FP8 | not Blackwell microscaling hardware | no native NVFP4 |
| Blackwell / B200 | native | native | native microscaling | native NVFP4 |

“Native INT8” in this table means the architecture has an INT8 matrix
instruction for compatible operands. It does not turn W8A16 into an INT8
matrix multiply: a weight-only kernel must still dequantize weights or use a
specialized mixed-input path because its activations remain 16-bit. Likewise,
a GPU may store an FP8 KV cache without having a native FP8 GEMM for the model
weights. The operand types and operation matter, not just the byte format.

M8 therefore defines three explicit execution levels:

1. **Q/DQ oracle:** decode the complete tensor to BF16 and call the ordinary
   linear layer. This checks numerical semantics but saves neither resident
   memory nor bandwidth.
2. **Packed fallback:** keep codes and scales packed in GPU memory, decode only
   the tile being multiplied, and use a wider multiply. This can save storage
   and traffic, but must not be described as native FP8/FP4 compute.
3. **Native kernel:** feed the packed operands and scale layout to matching
   tensor-core instructions. This claim requires compilation/SASS inspection
   and measurement on the target architecture.

The architecture boundary is based on NVIDIA's public TensorRT and Transformer
Engine documentation linked below. M8 will record a native instruction claim
only after the corresponding target-GPU profile confirms it.

## A mixed checkpoint needs per-layer dispatch

The local `Kimi-K3-NVFP4` metadata does not describe a uniform NVFP4 model. Its
`hf_quant_config.json` declares `MIXED_PRECISION`; expert groups are NVFP4 with
group size 16. Its conversion report separately records FP8 per-block
weight-only attention conversion, with some tensors left at high precision.
These are producer declarations: the local large-checkpoint weight files and
index are Git LFS pointers, so they do not establish downloaded tensor contents
or working inference. M8d reports that boundary explicitly.

That leads to a loader rule:

> Resolve quantization from each tensor or layer's metadata, never from the
> model directory name and never from one global CLI label.

A useful runtime object must retain at least:

| field | why the kernel needs it |
|---|---|
| logical shape | packed storage shape is not the matrix shape |
| element format | INT8, INT4, E4M3, or E2M1 changes decoding |
| packed data | the bytes that remain resident on the GPU |
| scale tensor and scale type | scale arithmetic and load width differ |
| scale axis/block shape | maps each partial dot product to its scale |
| optional global scale | required by hierarchical formats such as NVFP4 |
| packing/layout version | defines nibble order, swizzle, transpose, and padding |
| original output dtype | tells the epilogue what the layer returns |

NVIDIA ModelOpt's unified Hugging Face format follows this same principle: the
safetensors contain quantized weights and scales, while
`hf_quant_config.json` records the quantization choices. Supporting the names
without matching their layouts is not checkpoint compatibility.

## Quantization methods address outliers and sensitivity

Naive min/max is the right first equation but not the last accuracy technique.
LLMs contain activation outliers and layers with very different sensitivity.
Four well-known PTQ families illustrate the design space:

- **LLM.int8()** isolates exceptional activation dimensions into a higher
  precision path while most work remains INT8.
- **SmoothQuant** applies a mathematically equivalent per-channel rescaling to
  move quantization difficulty from activations into weights, making W8A8 more
  practical.
- **GPTQ** uses approximate second-order information while quantizing weights
  to reduce layer reconstruction error.
- **AWQ** uses observed activations to identify and protect salient weight
  channels through scaling, without backpropagation-based reconstruction.

The important lesson is not that Tinyserve must implement all four. It is that
one local weight MSE cannot certify model quality. Quantization error is
multiplied, normalized, added through residuals, and fed back across generated
tokens. Calibration data and evaluation data must also be separate; otherwise
the apparent win can be calibration-set overfitting.

## A performance model for quantization

Quantization changes performance through several mechanisms, and they do not
all help the same workload:

1. fewer stored bits can reduce checkpoint and resident-weight memory;
2. fewer bytes can reduce HBM traffic when weights are streamed;
3. native low-precision instructions can increase matrix-multiply throughput;
4. the saved memory can hold more KV cache or a larger batch;
5. quantization, packing, scale loads, conversion, and dispatch add work.

The first four are potential benefits. The fifth is always present somewhere.
Whether the balance is favorable depends on the shape, phase, format, kernel,
GPU, and serving policy.

[![Quantization primarily reduces weight traffic during small-batch decode,
while large prefill needs useful native compute and end-to-end gains remain
bounded by unquantized work](/assets/tinyserve/m8-quantization-performance.svg)](/assets/tinyserve/m8-quantization-performance.svg)

### Start with the roofline of one linear layer

For:

$$
X_{M\times K}W_{N\times K}^{\top}=Y_{M\times N},
$$

the matrix multiplication performs approximately this much arithmetic work,
counting a multiply and an add separately:

$$
F = 2MNK
$$

floating-point operations. A simple traffic model is:

$$
D \approx b_XMK + b_WNK + b_YMN + D_{\text{scales}} + D_{\text{workspace}},
$$

where each $b$ is bytes per element. The arithmetic intensity is:

$$
I = \frac{F}{D}.
$$

If a device sustains compute rate $P$ and memory bandwidth $\beta$, the
roofline lower bound is:

$$
t_{\text{kernel}} \ge
\max\left(\frac{F}{P},\frac{D}{\beta}\right).
$$

Real time is higher because launches, synchronization, quantization,
dequantization, scale application, padding, and imperfect utilization are not
free. The equation is useful because it tells us which optimization can
matter. Reducing $b_W$ attacks the bandwidth term. A native low-precision MMA
raises the relevant compute ceiling. A format label does neither unless the
runtime actually uses the matching storage and kernel.

### Decode example: weight traffic dominates at one token

Take one `4096 × 4096` projection and one decode row:

$$
M=1,\qquad N=K=4096.
$$

The layer performs about `33.6 million` operations but reads a 32 MiB BF16
weight matrix. The input and output are only 8 KiB each, so the approximate
arithmetic intensity is one operation per weight byte. If the matrix is not
already served from cache, this is a streaming, bandwidth-oriented GEMV.

Per-output-row INT8 with FP16 scales stores 16 MiB of codes plus 8 KiB of scales
(Tinyserve M8a instead uses FP32 scales, totaling 16 KiB). At an
*illustrative* sustained bandwidth of `600 GB/s`, reading only the weight
payload has these lower bounds:

$$
t_{\text{BF16}} \ge \frac{32\ \text{MiB}}{600\ \text{GB/s}} \approx 55.9\ \mu s,
$$

$$
t_{\text{INT8}} \ge
\frac{16\ \text{MiB}+8\ \text{KiB}}{600\ \text{GB/s}}
\approx 28.0\ \mu s.
$$

The near-2× ratio is a **weight-read ceiling**, not a kernel prediction. Scale
loads, address calculation, conversion, output traffic, launch time, and
achieved rather than nominal bandwidth reduce it. A tiny model whose weights
hit in L2 may also behave differently from a large checkpoint that streams
from HBM.

The same payload-only calculation exposes scale overhead across formats for
this matrix:

| weight representation | effective bits/value | weight payload | ideal BF16/quantized traffic ratio |
|---|---:|---:|---:|
| BF16 | 16 | 32 MiB | 1.00× |
| INT8, FP16 scale per row | 8.0039 | 16.008 MiB | 2.00× |
| FP8, one tensor scale | approximately 8 | approximately 16 MiB | approximately 2.00× |
| MXFP8, E8M0 scale per 32 | 8.25 | 16.5 MiB | 1.94× |
| INT4, FP16 scale per 128 | 4.125 | 8.25 MiB | 3.88× |
| MXFP4, E8M0 scale per 32 | 4.25 | 8.5 MiB | 3.76× |
| NVFP4, E4M3 scale per 16 | 4.5 | 9 MiB | 3.56× |

The FP4 result is deliberately not “4×.” Fine-grained scales consume bytes.
Padding, alignment, global scales, mixed-precision layers, and workspace make
the complete model ratio smaller still.

### Prefill and batched decode move toward compute

During prefill, $M$ is the number of packed prompt rows presented to the
projection. During batched decode, $M$ is approximately the active batch size.
The same weight tile can now contribute to many output rows.

Ignoring the smaller activation and output terms, BF16 arithmetic intensity is
approximately:

$$
I_{\text{BF16}} \approx
\frac{2MNK}{2NK}=M.
$$

For the same `4096 × 4096` projection at $M=128$, work grows to about
`4.29 billion` operations while the BF16 weight remains 32 MiB. Including the
roughly 1 MiB input and 1 MiB output gives about `120 operations/byte`, rather
than about one. The kernel may now approach a compute roof, depending on the
GPU and achieved implementation efficiency.

That changes which quantization path is useful:

- A packed W8A16 fallback still performs a wider multiplication and adds
  conversion. It may help small-$M$ decode but lose during large prefill.
- Native W8A8 or FP8 can reduce operand traffic **and** use a higher-throughput
  matrix path, so it has a more plausible large-$M$ benefit.
- Dynamic activation quantization computes an `amax`, scale, and packed input
  for every runtime row or block. At very small $M$, that setup may consume the
  latency saved by the GEMM.
- Routed MoE experts often receive small and uneven token groups. Their expert
  weights may again look bandwidth-heavy even when the globally packed prefill
  is large; grouped-kernel layout and routing balance matter.

One kernel should therefore not be assumed optimal for both phases. Tinyserve
should measure a decode-oriented GEMV/small-GEMM path and a prefill-oriented
GEMM path separately.

### Where dequantization happens determines the traffic

Suppose an INT8 weight is dequantized by a separate GPU kernel before calling
BF16 GEMM. Per weight, that path can cause roughly:

```text
read 1 byte INT8
write 2 bytes BF16
read 2 bytes BF16 again in GEMM
--------------------------------
about 5 bytes moved, before scales and cache effects
```

The BF16 baseline needed only the final 2-byte weight read. A quantized
checkpoint can therefore produce *more* runtime traffic when the full
dequantized matrix is materialized.

A useful packed fallback fuses the boundary: load codes and scales, reconstruct
one tile into registers or shared memory, and immediately consume it. It avoids
the global BF16 write and reread. A native kernel goes further by feeding the
low-precision operands and scale layout to the matching matrix instruction.

Packing time belongs to a separate clock. Offline weight conversion affects
artifact-production time but not request latency. Packing BF16 weights during
model loading increases startup time and temporarily requires both the source
and destination, so peak load memory can exceed steady-state memory. Dynamic
activation or cache quantization runs inside serving and must be charged to
the request phase that triggers it. M8 will report conversion time, load time,
and steady-state execution independently.

This gives a practical hierarchy:

| execution path | expected performance meaning |
|---|---|
| eager fake Q/DQ | correctness tool; normally slower and not a speed candidate |
| full-weight dequantize then GEMM | checkpoint compatibility; can move more bytes than BF16 |
| fused packed W8A16 fallback | real memory saving; strongest opportunity at low arithmetic intensity |
| native W8A8 or FP8 | memory and compute opportunity; activation quantization must be amortized |
| native MXFP4/NVFP4 | largest payload reduction on Blackwell; scale layout, alignment, and tile utilization remain part of the kernel |

### Kernel speed is not serving speed

If eligible linear layers occupy fraction $f$ of baseline latency and become
$S_{\text{linear}}$ times faster, Amdahl's law bounds end-to-end speedup:

$$
S_{\text{end-to-end}}
=
\frac{1}{(1-f)+f/S_{\text{linear}}}.
$$

If linear layers are 80% of time and become 2× faster, the complete request is
only:

$$
\frac{1}{0.2+0.8/2}=1.67\times
$$

faster. Attention, KV-cache access, normalization, sampling, scheduling, CPU
work, and communication do not disappear. Quantizing only some projections
also leaves the remaining weight traffic untouched.

Memory capacity can produce a different, discontinuous gain. An idealized
27-billion-parameter model requires about 54 GB for 16-bit weights and 27 GB
for 8-bit weights before scales and mixed-precision tensors. The BF16 model
cannot fit on one nominal 48 GB GPU even before KV cache and workspace, while
an INT8-weight model may fit with room for runtime state. Avoiding tensor or
pipeline parallel communication can matter more than the isolated kernel
ratio. This is a capacity argument; the actual checkpoint must still be
measured.

Even when one request already fits, lower resident weight memory can leave
more space for KV cache, larger batches, CUDA graphs, and concurrent requests.
Throughput or goodput may improve while single-request latency stays flat.
Conversely, a quality regression can require a wider format or more generated
tokens and erase a nominal kernel win. Performance results therefore need both
fixed-work measurements and serving-level measurements.

### What M8 should expect before measuring

These are hypotheses to test, not results:

| candidate | decode expectation | prefill expectation | decisive evidence |
|---|---|---|---|
| eager INT8 Q/DQ | slower | slower | numerical parity only |
| packed INT8 W8A16 on A6000 | likely bandwidth benefit at small batch | uncertain; conversion may dominate | resident bytes, DRAM bytes, phase latency |
| native INT8 W8A8 | setup-sensitive at small batch | more likely to benefit large GEMM | integer MMA instructions and end-to-end timing |
| native FP8 on Hopper/Blackwell | shape- and scale-dependent | compute benefit plausible | FP8 instructions, tensor-pipe utilization, latency |
| native MXFP4/NVFP4 on Blackwell | strong traffic opportunity | strong compute opportunity at suitable tiles | packed bytes, block-scale MMA, alignment, full-model timing |

The table intentionally says “likely,” “uncertain,” and “plausible.” Only the
implemented path on the named GPU can replace those words with measurements.

### M8a result: capacity improved; latency did not

M8a implements the narrow first slice: symmetric INT8 weights, one FP32 scale
per output row, BF16 activations, and a fused Triton kernel for the seven dense
Qwen projection kinds. The kernel loads packed codes and scales and decodes a
tile immediately before a BF16 dot. It never creates a complete BF16 weight,
but it is still a **packed W8A16 fallback**, not a native INT8 matrix multiply.
M8a packs ordinary BF16 safetensors on CPU before device transfer. M8b below
adds the corresponding prepacked checkpoint and direct-loading path.

The local Qwen3-0.6B has 28 layers and therefore 196 converted projections.
Its persistent model state falls from 1,503,264,768 to 1,064,239,104 bytes, a
29.2% reduction. CUDA allocation immediately after load falls 27.8%, from
1,545,222,144 to 1,115,633,664 bytes. The complete-model saving is smaller
than 2x because embeddings, norms, and the large LM head deliberately remain
BF16.

Correctness is measured against what INT8 actually promises. Eight focused
tests cover zero rows, packing, absence of a floating weight, selective loader
replacement, and decode/prefill kernel shapes. The packed kernel is
bit-identical to the complete BF16 Q/DQ oracle for those tested shapes. A
five-prompt last-logit probe has 80% top-1 agreement, 92% mean top-5 overlap,
0.9988 mean cosine similarity, and mean $D_{KL}(p_{BF16}\|p_{INT8})=0.0132$.
Both models answer the `2 + 2` serving smoke with `4`. This is not a held-out
task-quality result, so M8a makes no broader accuracy claim.

The serving benchmark uses local Qwen3-0.6B on one RTX A6000, CUDA graphs,
automatic attention dispatch, an 8,192-token prefill budget, two warmups, and
five measured repetitions. Each row compares the current BF16 and INT8 paths
under the same engine protocol:

| prompt / output / batch | BF16 output tok/s | INT8 output tok/s | INT8 / BF16 | BF16 TTFT | INT8 TTFT |
|---|---:|---:|---:|---:|---:|
| 128 / 128 / 1 | 225.8 | 185.3 | 82.1% | 28.0 ms | 32.7 ms |
| 128 / 128 / 8 | 1,629.9 | 1,390.6 | 85.3% | 28.5 ms | 36.5 ms |
| 128 / 128 / 32 | 4,602.7 | 3,912.3 | 85.0% | 102.4 ms | 185.3 ms |
| 2,048 / 32 / 1 | 159.7 | 124.2 | 77.8% | 57.4 ms | 80.4 ms |
| 2,048 / 32 / 8 | 377.7 | 206.8 | 54.8% | 418.8 ms | 951.6 ms |
| 2,048 / 32 / 32 | 442.4 | 227.1 | 51.3% | 1,065.1 ms | 2,411.2 ms |

That six-shape sweep ran each representation in its own persistent process.
The stricter B1 control kept both models resident, alternated execution order
over seven pairs, and gave 235.16 tok/s for BF16 versus 186.96 tok/s for INT8.
The median paired ratio is 79.44%, with a deterministic percentile-bootstrap
95% interval of 79.36–79.60%. The order-controlled result confirms that the
loss is not explained by which process happened to run first.

The causal conclusion is straightforward: the representation saves memory,
but this first dequantize-plus-BF16 kernel loses every latency and throughput
gate. The larger prefill loss is consistent with conversion overhead and a
simple tiled implementation competing against an optimized BF16 library,
without a native low-precision compute-rate gain. This sweep does not isolate
those costs individually. A separate one-row GEMV experiment
also lost to the retained tiled kernel and was removed. INT8 therefore stays
an explicit `--quantization int8` teaching path; BF16 remains the default.

For calibration only, the frozen August p128/o128 dense-engine study reported
B1 decode throughput of 341 tok/s for FreeToken, 325 for vLLM, 292 for
llama.cpp, and 215 for Ollama. M8a INT8 reaches 185 tok/s under Tinyserve's
more favorable engine-internal timer, so it does not close the external-engine
gap. Those engines use different kernels and, for some lanes, different
quantization/layout contracts; this comparison must not be used to attribute
the difference to INT8 alone. The [structured M8a evidence](https://github.com/kaix-nv/tinyserve/blob/3846ae7c1883acfe483bc1c350db277d9d826ef0/docs/benchmarks/m8a-int8-a6000-2026-09-14.json)
and the [frozen cross-engine protocol](https://github.com/kaix-nv/tinyserve/blob/3846ae7c1883acfe483bc1c350db277d9d826ef0/docs/benchmarks/cross-engine-a6000-2026-08-29.md)
retain the boundaries.

### M8b result: the checkpoint now matches the runtime representation

M8a proved that GPU-resident weights can remain packed, but its input artifact
was still BF16. Every process had to read the larger matrix, allocate the
packed replacement, and temporarily hold both. M8b moves that conversion to
[`examples/export_int8.py`](https://github.com/kaix-nv/tinyserve/blob/3846ae7c1883acfe483bc1c350db277d9d826ef0/examples/export_int8.py), outside the serving
startup path.

[![M8a repacks BF16 weights during every startup, while M8b stores canonical
packed tensors and builds their buffers on the meta device before direct
loading](/assets/tinyserve/m8b-packed-checkpoint.svg)](/assets/tinyserve/m8b-packed-checkpoint.svg)

For a concrete Qwen3-0.6B projection, the exporter replaces:

```text
model.layers.0.self_attn.q_proj.weight         BF16 [2048, 1024]
```

with:

```text
model.layers.0.self_attn.q_proj.weight_packed  INT8  [2048, 1024]
model.layers.0.self_attn.q_proj.weight_scale   FP32  [2048]
```

`tinyserve_quantization.json` records schema version 1, the `int8` format,
`per_output_row` granularity, symmetric quantization, scale dtype, every
logical matrix shape, and the canonical tensor keys. This is deliberately a
small Tinyserve teaching format, not a claim of compatibility with ModelOpt,
GGUF, or another engine's INT8 layout.

The direct loader follows five visible steps:

1. read and validate the manifest before loading tensor payloads;
2. build the ordinary dense-Qwen module tree on the `meta` device;
3. replace exactly the manifest's eligible projections with empty packed
   buffers and validate every logical shape;
4. make `weight_packed` and `weight_scale` the target state-dict keys, then
   read them without casting away INT8 or FP32;
5. transfer the completed packed model to the GPU.

Automatic dispatch is intentionally conservative:

| checkpoint | `--quantization auto` | explicit override |
|---|---|---|
| ordinary BF16 | ordinary BF16 loading | `int8` performs M8a runtime packing |
| M8b manifest present | direct packed loading | `int8` also loads packed; `none` rejects |

The exporter refuses to overwrite an existing output path, preserves
non-quantized tensors and tokenizer/configuration files, and updates a
sharded safetensors index when the source is sharded. The local output remains
ignored under `.tinyserve-models/`; it is a generated artifact, not source.

On the real local Qwen3-0.6B checkpoint, the sum of top-level artifact files
falls from 1,519,197,900 to 1,080,241,642 bytes, a 28.9% reduction. The packed
file contains 196 quantized modules, 392 code/scale tensors, 115 unquantized
tensors, and no eligible BF16 projection weight.

Fresh-process load measurements alternate runtime-packed and prepacked order
over three repetitions:

| path | median load | median peak host RSS | CUDA allocation after load |
|---|---:|---:|---:|
| BF16 artifact → runtime pack | 1.002 s | 2.537 GiB | 1,115,633,664 bytes |
| prepacked artifact → direct load | 0.641 s | 1.619 GiB | 1,115,633,664 bytes |

Direct loading is 1.56x faster in this warm-cache experiment and reduces peak
host RSS by 36.2%. GPU allocation is intentionally identical: both paths end
with the same packed model. All 507 real-checkpoint state tensors compare
exactly, and a fixed-prompt forward produces bit-identical logits with maximum
difference zero. A five-repeat p128/o128/B1 cross-engine-protocol calibration
reaches 184.93 tok/s, versus M8a runtime packing's 185.32 tok/s under the same
settings: a 0.2% difference around the unchanged execution path. M8b changes
storage and startup—not request execution or M8a's losing latency result.

The [structured M8b evidence](https://github.com/kaix-nv/tinyserve/blob/3846ae7c1883acfe483bc1c350db277d9d826ef0/docs/benchmarks/m8b-packed-checkpoint-a6000-2026-09-14.json)
retains hashes, raw samples, measurement boundaries, and the local artifact
path.

## M8c: native INT8 is a different computation

M8a stores INT8 weights but multiplies reconstructed BF16 values. M8b makes
those codes directly loadable. M8c now reuses the **same checkpoint** and
changes the dot product: quantize the current activations to INT8, multiply
INT8 codes, accumulate in INT32, and rescale the result to BF16 or FP16.
“W8A8” describes the operands of these projection dots—not every tensor or
operation in the transformer.

[![M8c activation quantization, native integer dot, and rescaling, with a
four-element numerical example](/assets/tinyserve/m8c-native-int8.svg)](/assets/tinyserve/m8c-native-int8.svg)

### One scale follows a token; the other follows an output channel

Flatten a projection input into $X\in\mathbb{R}^{M\times K}$: $M$ token rows,
$K$ input channels. Its checkpoint contains weight codes
$Q_W\in\mathbb{Z}^{N\times K}$ and $N$ weight scales $s_W$, one per output
channel. For each token row $m$, the first kernel calculates:

$$
s_X[m] = \frac{\max_k |X[m,k]|}{127},\qquad
Q_X[m,k] = \operatorname{clamp}
\left(\operatorname{round}(X[m,k]/s_X[m]), -127, 127\right).
$$

An all-zero row uses scale one and zero codes. Rounding is nearest-even.
These are **dynamic per-token activation scales**: they are recomputed on
every forward, including CUDA-graph replay. No activation calibration
dataset or activation scales in the checkpoint are required.

The second kernel computes:

$$
A[m,n] = \sum_{k=0}^{K-1} Q_X[m,k]Q_W[n,k],\qquad
Y[m,n] = \operatorname{BF16}\left(
\operatorname{FP32}(A[m,n])\,s_X[m],s_W[n]\right).
$$

$A$ is an INT32 accumulator held inside the kernel, not a materialized
global-memory matrix. Its two scales broadcast along different axes:
`sx[:, None]` follows token rows; `sw[None, :]` follows output channels.
Only the final floating result is stored. Unlike a fake-Q/DQ implementation,
this path never constructs a floating-point weight matrix for the dot.

For a concrete example, let:

```text
X  = [  1,   -2,  0.5, 0]       sx = 2/127
W  = [  1,  0.5,   -1, 0]       sw = 1/127
Qx = [ 64, -127,   32, 0]
Qw = [127,   64, -127, 0]

INT32 dot = 64*127 - 127*64 - 32*127 = -4064
rescaled  = -4064 * (2/127) * (1/127) ≈ -0.50394
BF16 out  = -0.50390625                 original floating dot = -0.5
```

The integer dot can be exact while its answer differs from the original
floating dot. The approximation happened when choosing the codes, not
because the integer accumulator is inaccurate. This is why kernel parity
and model quality are different tests.

### Follow one projection through the implementation

`QuantizedLinear.forward()` flattens the leading dimensions and selects
`w8a8_linear()` in [`int8_kernels.py`](https://github.com/kaix-nv/tinyserve/blob/3846ae7c1883acfe483bc1c350db277d9d826ef0/tinyserve/int8_kernels.py):

1. `_quantize_rows_kernel` reduces each row to its maximum absolute value,
   writes INT8 activation codes, and writes an FP32 scale. These temporary
   buffers require $MK+4M$ bytes, in addition to the floating input/output.
2. `_w8a8_kernel` loads activation and weight code tiles, using zeros outside
   partial tiles. `tl.dot(..., out_dtype=tl.int32)` performs the integer dot.
3. The same kernel converts the accumulator to FP32, applies both scales,
   and stores the requested BF16/FP16 output.

For $M\le16$, the output tile is $16\times64$; larger batches use
$32\times64$. Both accumulate through the input channels in steps of 64.
This intentionally small implementation supports $1\le K\le32768$.
Besides bounding the row-reduction work, that limit bounds the largest
possible sum by $32768\times127^2=528515072$, safely below the INT32 limit.
Inputs are expected to be finite.

Each projection quantizes its own input. Q/K/V do **not** share a quantized
activation buffer yet, even when they consume the same normalized hidden
state. M8c therefore adds one quantizer launch per projection: 196 extra
kernel launches per Qwen3-0.6B forward. Graph replay reduces host submission
overhead but does not erase those kernels or their memory traffic.

The slow `w8a8_reference()` computes the bounded integer sum exactly using
FP64, then reproduces the FP32 rescaling order. It is a testing oracle, never
a serving fallback. Tests cover BF16/FP16, nearest-even ties, zero rows,
partial tiles, the largest supported input width, noncontiguous inputs,
unchanged packed checkpoint tensors, and graph replay with changed values.

On the tested RTX A6000, generated PTX contains
`mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32`; its compiled
SASS contains `IMMA.16832.S8.S8.SAT`. That instruction evidence, together
with the persistent INT8 buffers and exact oracle tests, supports the native
INT8 claim. It does not establish Tensor Core utilization or a bandwidth
improvement; those require separate profiling.

### Selecting and debugging the path

Weight representation and compute selection are separate choices:

| Input checkpoint | Weight choice | Compute choice | Result |
|---|---|---|---|
| ordinary BF16 | `--quantization none` | default | BF16 baseline |
| ordinary BF16 | `--quantization int8` | default | pack at load, W8A16 |
| M8b packed INT8 | `--quantization auto` | default | direct load, W8A16 |
| M8b packed INT8 | `--quantization auto` | `--int8-compute w8a8` | direct load, native W8A8 |

The Python equivalent is `LLM(packed_path, int8_compute="w8a8")`. An ordinary
BF16 checkpoint also needs `quantization="int8"`; otherwise the loader
rejects the inconsistent request. W8A8 is restricted to dense Qwen on one
CUDA GPU, BF16/FP16 inputs, Triton, and compute capability 8.0 or newer. It
has been validated on SM86 here, not on every newer architecture. There is
no silent CPU or floating-point fallback.

The **M8c: native INT8 W8A8 (dynamic activations)** launch configuration
opens `examples/generate.py` using the local M8b artifact with graph replay
disabled for stepping. Useful host-side breakpoints are
`QuantizedLinear.forward()` and `w8a8_linear()`. Python debugging can inspect
the dispatched shapes and buffers; it does not single-step GPU instructions.

### Quality is not implied by native execution

On six fixed teaching texts, covering 192 logit positions, M8c's token-weighted
mean KL divergence from BF16 is 0.0642, versus 0.0252 for W8A16. Top-1
agreement is 87.0% versus 92.7%, and top-5 overlap is 87.9% versus 94.3%.
Dynamic scaling avoids a calibration pass, but an outlier in a token row
still sets the quantization step for every channel in that row.

These are diagnostics, **not** a held-out task-quality qualification. The
arithmetic smoke test also passes, but neither it nor a small logit corpus
justifies a general accuracy claim. M8c does not implement SmoothQuant,
outlier routing, or mixed-precision exceptions to repair that drift.

### Serving performance: better than the fallback, not better than BF16

Two paired experiments keep BF16, W8A16, and W8A8 models resident together on
physical GPU 1 (RTX A6000). Each uses two warmups, six measured repetitions,
and rotating execution order. Prefix reuse is off, the prefill budget is
8192 tokens, each model has 32768 KV slots, and decode graphs are on. These
paired runs also enable the phase-profile hooks for all three variants.

| Paired workload | BF16 output tok/s | W8A16 output tok/s | W8A8 output tok/s | W8A8 / W8A16 | W8A8 / BF16 |
|---|---:|---:|---:|---:|---:|
| 128 prompt / 128 output / B1 | 233.37 | 185.03 | 214.78 | 1.162x [1.155, 1.171] | 0.921x [0.914, 0.941] |
| 2048 prompt / 32 output / B8 | 380.46 | 206.81 | 352.41 | 1.704x [1.700, 1.706] | 0.926x [0.924, 0.927] |

Throughputs are medians. Ratios are medians of **paired** throughput ratios,
with percentile-bootstrap 95% intervals; they need not equal the ratio of
the displayed medians. Every request emitted its requested output count in
these runs. The diagnostic smoke prompt returned `4` for all variants.

The improvement over W8A16 is real in these local measurements, especially
for the prefill-heavy workload. It is not a win over the BF16 baseline. Even
on the decode-heavy row, W8A8's median TTFT is 37.44 ms versus 32.39 ms for
W8A16 and 26.31 ms for BF16: an improvement in whole-request throughput does
not imply an improvement in the first-token wait.

The standalone `bench_w8a8.py` diagnostic separates two other boundaries:

- **Projection microbenchmarks** capture 100 calls inside a CUDA graph, so
  Python replay submission does not dominate a microsecond kernel. W8A8
  includes the quantizer and integer dot; a separate quantizer-only column
  exposes its cost. Repeated small weights may stay in cache, so these are
  not measurements of HBM bandwidth or complete-model latency.
- **Fixed-work model phases** use identical token IDs, prefill lengths, batch
  sizes, and 32 teacher-forced decode steps for all variants. They measure an
  eager contiguous-KV forward with synchronized host and CUDA-stream timing.
  They do not include the serving scheduler or decode graphs, and therefore
  must not be substituted for the serving numbers above.

Selected projection measurements make that distinction concrete (median
microseconds, lower is better; W8A8 already includes the quantizer):

| Projection shape $(M,N,K)$ | BF16 | W8A16 | W8A8 total | Quantizer alone |
|---|---:|---:|---:|---:|
| Q projection, $(1,2048,1024)$ | 8.16 | 6.66 | 5.52 | 1.58 |
| Down projection, $(1,1024,3072)$ | 12.22 | 18.13 | 14.51 | 4.61 |
| Q projection, $(2048,2048,1024)$ | 90.98 | 205.17 | 103.91 | 10.83 |
| Down projection, $(2048,1024,3072)$ | 138.51 | 382.80 | 184.20 | 32.39 |

The quantizer-only timing is a separately timed diagnostic, not an extra
term to add to the W8A8 total. At full-model scale, fixed-work prefill with
2048 tokens and B8 takes 393.36 ms in BF16, 1097.09 ms in W8A16, and 492.94 ms
in W8A8. The native path is 2.22x faster than the fallback in that phase,
but still slower than BF16. The same experiment's 32 **eager** decode steps
take 683.79, 868.73, and 1022.49 ms respectively. Eager submission includes
the extra quantizer launches and host gaps; its ordering differs from
graph-enabled serving. Neither timing boundary replaces the other.

The six-shape **cross-engine-protocol refresh, Tinyserve side**, uses five
repetitions, 131072 KV slots, and phase profiling off. It reruns all three
Tinyserve paths with the common prompt builder:

| Prompt / output / batch | BF16 tok/s | W8A16 tok/s | W8A8 tok/s |
|---|---:|---:|---:|
| 128 / 128 / 1 | 241.78 | 188.71 | 220.02 |
| 128 / 128 / 8 | 1691.84 | 1420.38 | 1659.82 |
| 128 / 128 / 32 | 4759.44 | 3962.24 | 4911.84 |
| 2048 / 32 / 1 | 164.30 | 125.63 | 156.07 |
| 2048 / 32 / 8 | 384.57 | 207.00 | 351.77 |
| 2048 / 32 / 32 | 445.01 | 227.36 | 404.34 |

W8A8 improves on W8A16 in all six rows. One row, short-prompt B32, is about
3.2% above BF16; the other five remain below it. These complete suites ran
sequentially, not in paired order, and have no thermal telemetry. Treat the
small B32 advantage as a shape-specific observation to confirm, not grounds
for default promotion. Use the paired table for the two tested causal
comparisons; do not mix its different profiling/KV settings with this table.

The external-engine figures in the
[August calibration](https://github.com/kaix-nv/tinyserve/blob/3846ae7c1883acfe483bc1c350db277d9d826ef0/docs/benchmarks/cross-engine-a6000-2026-08-29.md) are still
historical references, **not fresh M8c runs** of llama.cpp, Ollama, FreeToken,
or vLLM. They also differ in format and timing boundary. M8c's evidence does
not establish an apples-to-apples INT8 ranking against those engines.

The implementation keeps one quantizer plus one dot launch per projection.
For small matrices, the extra reduction, code writes, scale reads, and launch
cost can outweigh a cheaper dot. For large matrices, avoiding weight-tile
dequantization helps, but a simple educational tile kernel still competes
with an optimized BF16 library path. Native arithmetic removes one possible
bottleneck; it does not remove the whole execution pipeline.

**Decision:** retain M8c as an opt-in educational native path. BF16 remains
the overall default, and W8A16 remains the default computation for INT8
artifacts. No task-quality approval or across-the-board latency win has
been established. The existing driver/NVML mismatch also prevents recording
the clock and thermal envelope; the paired intervals describe variation
within these runs, not that unmeasured source of uncertainty.

The [structured M8c evidence](https://github.com/kaix-nv/tinyserve/blob/3846ae7c1883acfe483bc1c350db277d9d826ef0/docs/benchmarks/m8c-native-int8-a6000-2026-09-15.json)
retains checkpoint/source hashes, instruction excerpts, the literal quality
corpus, microbench samples, fixed-phase measurements, paired samples, and
all six calibration conditions. The full regression run passes 159 tests
with 9 skipped; correctness and performance were checked on physical GPU 1.

## M8d: make the FP8/FP4 format contract executable

The format map is useful only if we can follow its bits back to numbers.
[`quant_formats.py`](https://github.com/kaix-nv/tinyserve/blob/3846ae7c1883acfe483bc1c350db277d9d826ef0/tinyserve/quant_formats.py) adds a small reference
implementation with a deliberately ordinary layout: row-major bytes, blocks
along the input dimension $K$, and the first FP4 value in the low nibble.
It is not a ModelOpt checkpoint loader or a tensor-core layout adapter.

The `QuantizedMatrix` object carries `format`, `logical_shape`, `data`,
`scales`, `original_dtype`, optional `global_scale`, and a layout version.
The format table fixes the element decoder, block width, and scale type:

| reference format | element | one scale covers | scale storage |
|---|---|---|---|
| `fp8_e4m3` / `fp8_e5m2` | E4M3 / E5M2 | one output row | FP32 |
| `mxfp8_e4m3` / `mxfp8_e5m2` | E4M3 / E5M2 | 32 adjacent columns | E8M0 |
| `mxfp4` | E2M1 | 32 adjacent columns | E8M0 |
| `nvfp4` | E2M1 | 16 adjacent columns | E4M3 plus one FP32 matrix scale |

`quantize_matrix()` pads incomplete blocks with zero and records the original
shape. `dequantize_matrix()` reconstructs the full floating-point matrix and
trims that padding. This is **Q/DQ reference code**, not a packed serving
kernel. Its `nbytes` counts stored data, scales, global scale, and padding;
it does not include Python objects or allocator overhead.

### Follow an NVFP4 matrix down to its first byte

Use two rows of 16 values, each with only two nonzero entries:

```python
from tinyserve.quant_formats import quantize_matrix, dequantize_matrix

w = torch.zeros(2, 16)
w[0, :2] = torch.tensor([6.0, 3.0])
w[1, :2] = torch.tensor([2688.0, 1344.0])
packed = quantize_matrix(w, "nvfp4")
restored = dequantize_matrix(packed)
```

The matrix maximum is `2688 = 6 × 448`. Our absmax recipe chooses a global
scale of `2688 / (6 × 448) = 1`. Row 0's block scale is `6 / (6 × 1) = 1`;
row 1's is `2688 / (6 × 1) = 448`. Those are exact E4M3 values, encoded by
bytes `0x38` and `0x7e` respectively.

After dividing by the reconstructed scales, both rows begin with `[6, 3]`.
Their E2M1 codes are `0b0111` and `0b0101`. Low-nibble-first packing puts
them into the same byte: `(0x5 << 4) | 0x7 = 0x57`.

[![Two NVFP4 rows share the same packed codes but reconstruct different
weights through their block scales](/assets/tinyserve/m8d-format-bytes.svg)](/assets/tinyserve/m8d-format-bytes.svg)

Decoding row 1 reverses those steps: the low nibble of `0x57` means `6`,
the scale byte `0x7e` means `448`, and the global scale means `1`, giving
`6 × 448 × 1 = 2688`. The high nibble similarly gives `1344`.
The representation uses 16 data bytes, two block-scale bytes, and four
global-scale bytes: **22 bytes**, versus 64 bytes for BF16. This example is
exact because its values were chosen from the codebooks; typical weights
incur rounding error in both their element values and scales.

### A format does not uniquely choose its quantizer

M8d makes the conversion policy explicit:

- Element encoding accepts finite inputs, rounds to nearest with ties to
  even, and saturates overflow. E2M1's midpoint `2.5` rounds to `2`, while
  `3.5` rounds to `4`. The FP8 decoders also recognize NaN and, for E5M2,
  infinity bit patterns; the encoders intentionally reject nonfinite inputs.
- MX uses the floor-exponent scale recipe in section 6.3 of the
  [OCP MX specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf):
  $s=2^{\lfloor\log_2 a_{max}\rfloor-\lfloor\log_2 q_{max}\rfloor}$,
  limited to the representable scale range. An MXFP4 block with maximum `7`
  chooses scale `1` and clips that value to `6`; maximum `8` chooses scale `2`
  and is exact. A ceiling-based recipe would make different trade-offs.
- E8M0 byte `0` means $2^{-127}$, not zero. Byte `127` means `1`, and `255`
  means NaN. We use scale `1` for an all-zero MX block.
- NVFP4 uses the one-dimensional absmax scheme described in
  [Transformer Engine's NVFP4 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html).
  Values are quantized against the **rounded, stored** block and global
  scales. An extremely small block scale can round to zero, making that
  block all zero. Training transforms and stochastic rounding are not used.

Scale calculations use FP64 intermediates before rounding stored FP32 scales;
FP32 scales are bounded below by their smallest positive subnormal. This
choice makes the reference inspectable, but it is not a claim of byte-for-byte
agreement with every producer's arithmetic. Nor is finite-input saturating
conversion a complete implementation of all OCP conversion modes. A hardware
kernel may additionally require padded, transposed, or swizzled scale storage
that this canonical layout deliberately does not claim to provide.

### Inspect declarations without pretending they are execution

[`quant_inspect.py`](https://github.com/kaix-nv/tinyserve/blob/3846ae7c1883acfe483bc1c350db277d9d826ef0/tinyserve/quant_inspect.py) reads ModelOpt JSON and
safetensors headers without importing model code or materializing weights.
It keeps three questions separate:

1. What does the producer **declare** for a module? Exclusion rules take
   priority; otherwise the most-specific matching declaration is reported.
   Equally specific conflicting rules are ambiguous, not silently selected.
2. What shapes and dtypes are **observed** in available tensor headers?
   Missing shards, invalid files, and Git LFS pointers remain visible.
3. Can Tinyserve execute that layout? **Not established by this tool.** A
   `headers-complete` inventory is not kernel compatibility or model quality.

The local checks make that distinction concrete:

| local artifact | metadata/header result | conclusion |
|---|---|---|
| `Kimi-K3-NVFP4` | mixed precision; 552 NVFP4 and 2,790 `FP8_PB_WO` declaration entries; all 96 weight shards and the index are LFS pointers | metadata available, tensor inventory unavailable |
| tiny `kimi-k3` fixture | one real shard; 6,608 tensor headers: 376 BF16, 88 FP32, 6,144 U8; no ModelOpt quantization JSON | header inventory complete, quantization recipe not inferred from U8 alone |

Declaration entries can include group names and aliases; those numbers are
**not counts of unique executed layers**. In particular, `FP8_PB_WO` stays
block FP8 weight-only, not automatically MXFP8. No weights are downloaded and
the conversion report remains producer-provided evidence, not an independent
verification of tensor values.

[`examples/inspect_quantization.py`](https://github.com/kaix-nv/tinyserve/blob/3846ae7c1883acfe483bc1c350db277d9d826ef0/examples/inspect_quantization.py)
provides both a small `--demo` matrix and `--model` header inspection. The
`M8d: FP8/FP4 codecs and checkpoint metadata` entry in
[launch.json](https://github.com/kaix-nv/tinyserve/blob/3846ae7c1883acfe483bc1c350db277d9d826ef0/.vscode/launch.json) runs both on the CPU. Useful breakpoints
are scale selection in `quantize_matrix()`, `pack_nibbles()`, and
`resolve_rule()`; no serving process or native FP4 GPU is required.

The focused tests cover every FP8 byte against PyTorch's E4M3FN/E5M2 decoding,
all finite-code round trips and rounding midpoints, all E2M1 codes, E8M0 edge
values, odd-column packing, block boundaries, scale underflow, and header-only
inspection. With the numerical-limit checks added, the full regression suite
passes **205 tests, with 9 skipped**; the format/inspection subset passes 46.
The [M8d evidence receipt](https://github.com/kaix-nv/tinyserve/blob/3846ae7c1883acfe483bc1c350db277d9d826ef0/docs/benchmarks/m8d-format-contract-2026-09-15.json)
records the local inspection boundary. No FP8/FP4 serving performance has been
measured: M8d changes neither the serving loader nor its kernels.

## How M8 measures correctness

Bit-identical logits are not a sensible quantization gate: rounding is the
feature. The validation ladder instead asks progressively more meaningful
questions:

1. **Codec parity:** hand-written examples recover the exact values implied by
   every code and scale, including zero, saturation, nibble order, padding, and
   block boundaries.
2. **Linear parity:** the packed kernel matches the oracle for its specified
   codes, scales, accumulation, and rounding order. M8a checks a BF16
   dequantize-then-matmul oracle; M8c checks an integer-sum oracle. These
   different computations need not produce identical outputs.
3. **Layer drift:** compare hidden-state absolute error, relative error, and
   cosine similarity at representative layers.
4. **Logit drift:** measure KL divergence, top-k overlap, and top-1 agreement
   against BF16 on a fixed prompt corpus.
5. **Sequence behavior:** compare generated decisions and divergence position;
   token equality is reported but is not the only metric.
6. **Task quality:** run a held-out language or task evaluation before calling
   a recipe acceptable beyond the educational fixture.

Every result must identify the exact checkpoint, format, scale granularity,
excluded layers, calibration corpus, evaluation corpus, and seed. “The INT8
model is accurate” is not falsifiable without those controls.

## How M8 measures systems behavior

Memory and speed are separate gates:

- serialized checkpoint bytes;
- resident packed-weight bytes, including scales and padding;
- peak GPU allocation during load, prefill, and decode;
- isolated GEMV/GEMM latency for the actual Qwen shapes;
- prefill latency at short and long prompts;
- decode latency and throughput at several batch sizes;
- HBM bytes, tensor-pipe activity, and executed instruction types under NCU;
- end-to-end serving throughput with the same scheduler and cache settings.

For phase timing, prefill uses fixed packed-token counts and decode uses fixed
active batches and a fixed number of teacher-forced or forced-length steps.
That prevents a changed sampled token or early EOS from changing the amount of
work. Report TTFT, time per output token, throughput, and goodput separately;
none is a substitute for the others.

The baseline and candidate will live in the same process where memory allows,
alternate measurement order, warm both paths, and report repeated paired
samples with uncertainty. Record GPU clocks, power/thermal state, software
versions, model shape, batch, prompt length, and generated-token count. NCU
must distinguish DRAM from L2 traffic and confirm the intended instruction
family; reduced allocation alone is not evidence of lower HBM traffic. A
packed path can land as an opt-in educational feature when it saves memory but
loses latency. It becomes a default only if the relevant end-to-end gate also
passes.

Cross-engine calibration will use the closest available quantized format in
llama.cpp, Ollama, FreeToken, and ninfer. A GGUF `Q8_0` model is not numerically
identical to a per-channel Tinyserve INT8 model, so those measurements are
system comparisons, not causal kernel A/B tests. The Tinyserve BF16-versus-
quantized pair remains the causal boundary.

## The bounded M8 roadmap

M8 is one quantization milestone, not an invitation to optimize every datatype
on every GPU. M8a completes the INT8 codec, Q/DQ oracle, packed linear, and
A6000 execution measurement. M8b completes the offline artifact, versioned
metadata, and direct loader. M8c adds dynamic activation quantization and an
opt-in native INT8 dot while retaining the same packed artifact. M8d adds a
standalone FP8/MXFP8/MXFP4/NVFP4 representation, reference codecs, Q/DQ, and
read-only ModelOpt metadata/header inspection. Later slices remain bounded:

- adapt one explicitly identified checkpoint layout to packed modules;
- extend reference coverage when that layout requires it, such as generic
  two-dimensional block FP8 or a specific INT4 recipe;
- packed CUDA fallbacks that never materialize the complete BF16 weight;
- architecture-gated native NVIDIA paths only when the required GPU is
  available for correctness, SASS, memory, and latency validation.

Further INT8 kernel tuning is deferred; its measured-losing paths remain
opt-in rather than delaying the rest of the format work.

Embeddings, normalization, and initially the LM head remain BF16. KV-cache,
GDN/KDA recurrent-state, and attention-probability quantization are separate
stateful problems and stay out of M8. INT4/AWQ/GPTQ performance kernels and QAT
also remain future work even though this chapter explains where they fit.

That boundary gives the project breadth at the format layer and depth at one
complete serving path. Most importantly, every future optimization still has
the readable BF16 model and explicit Q/DQ equation as an oracle.

## References

- Open Compute Project,
  [Microscaling Formats (MX) Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
  — element encodings, E8M0 scales, and the recommended scale-selection rule.
- NVIDIA TensorRT,
  [Quantization Schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
  — current INT8, INT4, FP8, MXFP8, and NVFP4 value/scale definitions.
- NVIDIA TensorRT,
  [Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html)
  — explicit Q/DQ, packed four-bit weights, dynamic quantization, and double
  quantization.
- PyTorch,
  [`FakeQuantize`](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)
  — the framework's floating-point simulation of quantize-then-dequantize.
- NVIDIA Transformer Engine,
  [FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html),
  [MXFP8](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html),
  and [NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html).
- NVIDIA,
  [Floating-Point 8: An Introduction to Efficient, Lower-Precision AI Training](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
  — per-tensor FP8, generic block scaling, and MXFP8 distinctions.
- NVIDIA,
  [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).
- NVIDIA TensorRT-LLM,
  [Quantization](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html)
  — current format, model, and hardware support matrix.
- NVIDIA Nsight Compute,
  [Roofline Charts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts)
  — arithmetic intensity, peak compute, and memory-bandwidth interpretation.
- NVIDIA TensorRT,
  [Performance Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
  — measure-first benchmarking and controlled latency/throughput analysis.
- NVIDIA ModelOpt,
  [Unified Hugging Face Checkpoint](https://nvidia.github.io/Model-Optimizer/deployment/3_unified_hf.html).
- Dettmers et al.,
  [LLM.int8()](https://papers.neurips.cc/paper_files/paper/2022/hash/c3ba4962c05c49636d4c6206a97e9c8a-Abstract-Conference.html).
- Xiao et al.,
  [SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html).
- Frantar et al.,
  [GPTQ](https://arxiv.org/abs/2210.17323).
- Lin et al.,
  [AWQ](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html).
