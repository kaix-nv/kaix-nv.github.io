---
layout: post
math: true
title: 'Tinyserve, Chapter 13: Numerical formats and quantization fundamentals'
date: 2026-10-06 00:00:00 -0700
categories:
- tinyserve
- llm-serving
excerpt: What do low-precision codes and scales represent and what information is lost?
book_chapter: 13
source_revision: 14b0c8dbb975350bbd730ac0c702eafce36eb95c
source_document: docs/book/13-quantization-fundamentals.md
code_revision: e20a34815498560f9226a9e057ade31b2b62743f
---

{% include tinyserve-article-style.html %}
{% include tinyserve-book-nav.html %}

*Chapter 13 · Execution Optimization*

A weight occupies sixteen bits in a BF16 checkpoint. Could it occupy eight,
or four, without making the model unusable? Answering that question requires
more than choosing a smaller datatype. We need to know which numbers its
bits represent, how a tensor's values are mapped onto those numbers, and
what information must accompany the stored codes.

We will follow one row through quantization, derive the limits of the common
floating formats, and then build block-scaled representations from them.
The central distinction is between a **number format** and a **quantization
scheme**. E2M1 specifies sixteen four-bit patterns. NVFP4 adds a particular
organization of scales around those patterns. Neither name, by itself,
specifies how a serving kernel executes the resulting matrix multiplication.
That execution question belongs to Chapter 14.

## One row through scale rounding and reconstruction

Take this small weight row:

```text
W = [-1.00, -0.60, -0.20, 0,
      0.25,  0.50,  0.75, 1.00]
```

A symmetric INT8 quantizer chooses a positive scale $s$, stores integer
codes $q_i$, and later reconstructs floating approximations $\hat w_i$:

$$
\begin{aligned}
s&=\frac{\max_i|w_i|}{127}=\frac1{127},\\
r_i&=\operatorname{round}(w_i/s),\\
q_i&=\operatorname{clip}(r_i,-127,127),\\
\hat w_i&=sq_i.
\end{aligned}
$$

The codes are `[-127,-76,-25,0,32,64,95,127]`. The second code reconstructs
to $-76/127\approx-0.598425$, rather than $-0.60$. The original value is
gone unless a separate floating copy was retained. Dequantization means
interpreting the stored approximation; it does not reverse the lost rounding.

[![An eight-value row becomes integer codes plus a scale, then reconstructs approximate values. The scale is stored separately from the codes.](/assets/tinyserve/book13-quantization-pipeline.svg)](/assets/tinyserve/book13-quantization-pipeline.svg)

Signed two's-complement INT8 has codes from $-128$ through $127$. This
scheme deliberately leaves $-128$ unused to keep equal positive and negative
magnitudes. Signed INT4 similarly has codes $-8$ through $7$; a symmetric
scheme may use $-7$ through $7$. Both have uniform spacing after scaling.
Neither has an exponent bias, infinity, NaN, or a special subnormal region.
Their smallest positive reconstructed magnitude is $s$, and the positive
endpoint is $127s$ or $7s$ for these symmetric schemes.

An affine integer scheme also stores a zero point $z$:

$$
\begin{aligned}
r&=\operatorname{round}(x/s)+z,\\
q&=\operatorname{clip}(r,q_{min},q_{max}),\\
\hat x&=s(q-z).
\end{aligned}
$$

The zero point shifts the integer interval over the real values. It can
help represent a one-sided distribution, but the loader and kernel must
agree about its axis, dtype, and subtraction. Saying only “INT8” specifies
none of those choices.

For nearest rounding of an unclipped value, the absolute error is at most
$s/2$. A clipped value has no such bound: mapping an input of 10 into a
range ending at 1 produces error 9. Even the bounded local error is not a
bound on output-token changes. A small perturbation can change a near-tied
logit after many layers.

## Scales decide which values compete for resolution

Consider two rows with different magnitudes:

$$
W=\begin{bmatrix}
0.02&0.04&0.06&0.08\\
2&4&6&8
\end{bmatrix}.
$$

One tensor-wide scale, $8/127\approx0.06299$, encodes the first row as
approximately `[0,1,1,1]`. Its distinct values mostly collapse. One scale
per output row gives both rows the codes `[32,64,95,127]`, interpreted with
their respective scales. The code width stayed eight bits; the scale
boundary changed the error dramatically.

For a weight matrix `W[N,K]`, a per-tensor scheme stores one scale,
per-output-channel scaling stores `N`, and groups of `G` input channels
store `N × ceil(K/G)`. Activations can instead use one scale per token row.
The axis matters: scaling a row of weights is different from scaling an
input feature shared across all output rows.

Smaller groups let unrelated ranges choose separate scales, but consume
more bytes and require more scale applications. With $b$ data bits and
$b_s$ scale bits for each group of $G$ values, the idealized storage is

$$
b_{effective}=b+\frac{b_s}{G}\quad\text{bits per value}.
$$

Padding, tensor-wide scales, and high-precision exceptions add further cost.
A four-bit code with an eight-bit scale per sixteen values uses 4.5 bits
per value before those additions, not four.

Absmax scaling preserves the largest magnitude but lets one outlier control
the resolution of its entire group. Choosing a clipping threshold $c$ below
the true maximum, with $s=c/q_{max}$, trades larger errors on clipped values
for smaller steps on the rest. Calibration selects such parameters using
representative data and an objective: reconstruction error, activation
statistics, or downstream loss. A calibration set must remain separate
from the held-out data used to judge the finished recipe.

Static scales are established before serving. Dynamic scales are computed
from the current activation or cache write, adding runtime reductions and
conversions. Post-training quantization changes an already trained model;
quantization-aware training exposes training to a rounding grid. These are
independent descriptions: post-training quantization can use dynamic
activation scales, and a trained recipe still needs a deployment exporter.

## How floating point spends its bits

Floating-point codes replace the uniform integer grid with a sign,
exponent, and fraction. For an ordinary positive-exponent finite encoding,

$$
x=(-1)^S2^{E-bias}\left(1+\frac{F}{2^M}\right).
$$

$S$ is the sign bit, $E$ the unsigned exponent field, $F$ the unsigned
fraction field, and $M$ the number of fraction bits. “Mantissa bits” in
names such as E4M3 means these fraction bits; the leading 1 of a normal
value is implicit. The bias makes an unsigned field describe negative as
well as positive powers of two.

For E4M3, `0 | 0111 | 100` gives $S=0$, $E=7$, $F=4$, and bias 7:
$2^{7-7}(1+4/8)=1.5$. It occupies eight bits because the sign is additional
to the four exponent and three fraction bits.

[![E4M3 and E5M2 spend the same eight bits differently. A fixed-scale example contrasts precision around one with range at one thousand.](/assets/tinyserve/book13-format-tradeoff.svg)](/assets/tinyserve/book13-format-tradeoff.svg)

Within $[2^e,2^{e+1})$, adjacent normal values are spaced by
$\Delta=2^{e-M}$. E4M3's gap is 0.125 near 1 and 1 near 8. Floating point
therefore offers roughly constant relative resolution across normal
exponents. An externally scaled integer offers constant absolute resolution.

BF16 and FP16 illustrate why bit width is insufficient. Both occupy sixteen
bits. BF16 uses eight exponent bits and seven fraction bits, while FP16 uses
five and ten. Near 1, $1+2^{-10}$ is exact in FP16 but rounds to 1 in BF16.
Conversely, 65,536 is finite in BF16 but exceeds FP16's largest finite
value, 65,504. More range and more precision are different benefits.

## Special values and exponent bias

NaN means “not a number,” a marker for invalid or undefined numerical
results. Infinity is a distinct value beyond every finite magnitude. A
format's ability to encode infinity does not determine whether a converter
uses it: overflow can saturate to a finite endpoint under an explicit
conversion policy.

The following rules describe the exact encodings used here. In the special
columns, $E$ and $F$ are integer fields, independent of sign.

| Format | Sign / exponent / fraction bits | Bias | NaN patterns | Infinity patterns |
|---|---|---:|---|---|
| FP32 | 1 / 8 / 23 | 127 | E=255, F nonzero | E=255, F=0 |
| BF16 | 1 / 8 / 7 | 127 | E=255, F nonzero | E=255, F=0 |
| FP16 | 1 / 5 / 10 | 15 | E=31, F nonzero | E=31, F=0 |
| FP8 E4M3FN | 1 / 4 / 3 | 7 | E=15, F=7 | None |
| FP8 E5M2 | 1 / 5 / 2 | 15 | E=31, F nonzero | E=31, F=0 |
| FP4 E2M1 | 1 / 2 / 1 | 1 | None | None |
| E8M0 scale | 0 / 8 / 0 | 127 | Byte 255 | None |

E4M3FN is the finite-with-NaNs encoding exposed by PyTorch as
`float8_e4m3fn`. Its NaNs are bytes `0x7f` and `0xff`. Other names such as
FNUZ have different rules and must not inherit this table. NVIDIA documents
the concrete [E4M3](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8__e4m3.html)
and [E5M2](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8__e5m2.html)
types; the [OCP MX specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
defines the corresponding MX element and scale encodings.

Do not apply “all exponent bits set means special” universally. E4M3
uses most of its highest-exponent patterns for finite values. Its largest
positive code, `0 | 1111 | 110`, represents
$2^{15-7}(1+6/8)=448$. E2M1 reserves no patterns for infinity or NaN.
These small formats spend scarce codes on finite values at the expense of
special-value expressiveness.

Changing a bias while preserving the same finite exponent patterns shifts
their magnitudes; it does not add more ratios between them. Increasing
the number of exponent bits can expand the available range. Those two
changes should not be conflated.

## Normal and subnormal endpoints

When the exponent field is zero, the leading significand becomes 0 and
the exponent remains $1-bias$:

$$
x_{sub}=(-1)^S2^{1-bias}\frac{F}{2^M}.
$$

These **subnormal**, or denormal, values fill the interval between zero and
the smallest normal. For the signed formats above,

$$
\begin{aligned}
x_{min,normal}&=2^{1-bias},\\
x_{min,sub}&=2^{1-bias-M},\\
x_{max,sub}&=(1-2^{-M})2^{1-bias}.
\end{aligned}
$$

The maximum finite normal must additionally respect reserved patterns.
All endpoints below are **positive, unscaled magnitudes**; signed formats
also represent their negatives and positive and negative zero. Exact
expressions take precedence over rounded decimals.

| Format | Maximum normal and finite value | Minimum normal |
|---|---|---|
| FP32 | $(2-2^{-23})2^{127}\approx3.4028235\times10^{38}$ | $2^{-126}\approx1.1754944\times10^{-38}$ |
| BF16 | $(2-2^{-7})2^{127}\approx3.3895314\times10^{38}$ | $2^{-126}$ |
| FP16 | $65,504$ | $2^{-14}\approx6.1035156\times10^{-5}$ |
| FP8 E4M3FN | $448$ | $2^{-6}=0.015625$ |
| FP8 E5M2 | $57,344$ | $2^{-14}$ |
| FP4 E2M1 | $6$ | $1$ |

| Format | Maximum denormal | Minimum denormal and positive value |
|---|---|---|
| FP32 | $(1-2^{-23})2^{-126}$ | $2^{-149}\approx1.4012985\times10^{-45}$ |
| BF16 | $(1-2^{-7})2^{-126}$ | $2^{-133}\approx9.1835496\times10^{-41}$ |
| FP16 | $1023\times2^{-24}\approx6.0975552\times10^{-5}$ | $2^{-24}\approx5.9604645\times10^{-8}$ |
| FP8 E4M3FN | $7\times2^{-9}=0.013671875$ | $2^{-9}=0.001953125$ |
| FP8 E5M2 | $3\times2^{-16}\approx4.5776367\times10^{-5}$ | $2^{-16}\approx1.5258789\times10^{-5}$ |
| FP4 E2M1 | $0.5$ | $0.5$ |

The wider-format constants are also exposed by NVIDIA's
[FP16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__HALF__CONSTANTS.html)
and [BF16](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__BFLOAT16__CONSTANTS.html)
APIs. Tinyserve's format tests check actual bit patterns, including
subnormals, rather than relying only on `finfo.smallest_normal`.

For E4M3, positive subnormals are $1/512,2/512,\ldots,7/512$; the first
normal is $8/512=1/64$. The spacing remains $1/512$ across that boundary.
This is gradual underflow. Relative precision deteriorates toward zero
because the absolute step stays fixed. Under nearest-even rounding, the
midpoint $1/1024$ rounds to zero, while $3/2048$ rounds to $1/512$.

An encoding table describes what storage can represent. Arithmetic modes
may flush subnormals to zero, and conversion policies can differ. Those
properties require checking the actual instructions and compiler settings.

## Dynamic range needs an explicit denominator

We use “dynamic range” as a dimensionless positive-magnitude ratio:

$$
\begin{aligned}
R_{normal}&=\frac{x_{max,finite}}{x_{min,normal}},\\
R_{all}&=\frac{x_{max,finite}}{x_{min,positive}}.
\end{aligned}
$$

The second includes subnormals; neither includes zero, infinity, or NaN.
It is not the signed interval and not the number of significant bits.

| Format | Normal-only ratio | Ratio including denormals |
|---|---:|---:|
| FP32 | $\approx2.8948021\times10^{76}$ | $\approx2.4283360\times10^{83}$ |
| BF16 | $\approx2.8834944\times10^{76}$ | $\approx3.6908728\times10^{78}$ |
| FP16 | $1,073,217,536$ | $1,098,974,756,864$ |
| FP8 E4M3FN | $28,672$ | $229,376$ |
| FP8 E5M2 | $939,524,096$ | $3,758,096,384$ |
| FP4 E2M1 | $6$ | $12$ |

These ratios follow by dividing the endpoints. E4M3 spans
$448/(1/512)=229,376$, or about 17.8 binary orders of magnitude, while
still having only three fraction bits. A wide range does not imply fine
resolution inside that range.

E8M0 is a scale encoding with no sign, zero, or subnormal region. Bytes
`0x00` through `0xfe` represent $2^{-127}$ through $2^{127}$, and `0xff`
represents NaN. Byte zero therefore means the smallest **positive scale**,
not scale zero. Its finite scale ratio is $2^{254}$; separate normal and
denormal endpoints do not apply.

## Spend one bit on range or precision

For a signed element of fixed width $B$, $1+E+M=B$. Moving one bit from
fraction to exponent increases available exponent patterns while reducing
the number of values inside each normal power-of-two interval.

Keep the external scale at 1 and encode `[1.10,1000]` with nearest
rounding and finite saturation:

| Input | E4M3 reconstruction | Absolute error | E5M2 reconstruction | Absolute error |
|---:|---:|---:|---:|---:|
| 1.10 | 1.125 | 0.025 | 1.0 | 0.10 |
| 1000 | 448 | 552 | 1024 | 24 |

E4M3's extra fraction bit helps the small ordinary value; E5M2's extra
exponent bit helps the large value. At the other end, $2^{-11}$ is exact
in E5M2 but rounds to zero in E4M3 at scale 1. Range failures include
underflow as well as clipping large values.

Changing the external scale can move both endpoints. It multiplies every
gap by the same factor, however, and cannot add code points. With E2M1
and an ideal effective scale $c>0$, the nonzero magnitudes run from $c/2$
to $6c$. Their ratio remains 12. A group containing both 1 and 1000
cannot represent both accurately by choosing a larger scale alone.

Grouping may help more than changing element format. For
`[0.6,0.8,1.2,0 | 100,100,100,100]`, one ideal E2M1 scale of $100/6$
rounds the first three values to zero. Separate four-value groups with
scales $0.2$ and $100/6$ reconstruct the whole example exactly: the first
group's normalized values are `[3,4,6,0]`. These are illustrative ideal
scales and group sizes, not actual MXFP4 or NVFP4 layouts. Their scale
formats introduce an additional rounding step.

## Build MX and NVFP4 from element codes and scales

The complete positive E2M1 magnitude table is

$$
0,\;0.5,\;1,\;1.5,\;2,\;3,\;4,\;6.
$$

The sign bit supplies the negative half, including negative zero. Its
sixteen patterns contain no representation of 5 or 5.5. E2M1 midpoint
2.5 rounds to 2 and midpoint 3.5 to 4 under ties-to-even. An INT4 nibble
uses the same number of bits but a different codebook; interpreting one
as the other changes the values.

| Scheme | Element format | Shared scale | Block width | Ideal bits per value |
|---|---|---|---:|---:|
| Ordinary FP8 | E4M3 or E5M2 | Recipe-defined floating scale | Tensor, row, or block | 8 plus scales |
| MXFP8 | E4M3 or E5M2 | E8M0 | 32 | 8.25 |
| MXFP4 | E2M1 | E8M0 | 32 | 4.25 |
| NVFP4 used here | E2M1 | E4M3 plus one FP32 matrix scale | 16 | 4.5 plus global scale |

A generic block-FP8 checkpoint is not automatically MXFP8. Its scale dtype,
block shape, axis, and physical layout must match. Similarly, the OCP MX
definition specifies element grouping but does not prescribe one universal
in-memory byte arrangement for every implementation.

MX reconstruction is $\hat x_i=q_i s_{block(i)}$. Its power-of-two scale
moves the element grid in binary steps. Tinyserve's reference uses

$$
s=2^{\lfloor\log_2 a_{max}\rfloor-\lfloor\log_2 q_{max}\rfloor},
$$

bounded to E8M0's range. Thus an MXFP4 block maximum of 7 chooses scale
1 and saturates to 6; a maximum of 8 chooses scale 2 and reconstructs
exactly. A different scale-selection policy can produce different bytes
while using the same element encoding. An all-zero block uses scale 1.

For NVFP4, reconstruction is

$$
\hat x_i=q_i^{E2M1}s_{block(i)}^{E4M3}g^{FP32}.
$$

The sixteen-value block gives finer locality than thirty-two values, and
E4M3 offers fractional scales rather than powers of two only. The global
scale moves the collection of block scales into its representable range.
NVIDIA's [NVFP4 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)
also describes training variants using two-dimensional 16-by-16 weight
groups. Tinyserve's reference uses one-dimensional groups along K; a
training variant is not an interchangeable checkpoint layout.

The reference chooses $g=a_{global}/(6\times448)$ and rounds the ideal
local scale $(a_{block}/6)/g$ into E4M3. It then quantizes elements using
the **stored, reconstructed scales**, not an unrounded ideal. This matters:
rounding scales and rounding elements are two separate approximations.
An extremely small local scale can become zero, zeroing its whole block.
A NaN scale can contaminate an entire block even if every element code is
finite. Tinyserve's matrix decoder rejects nonfinite scales.

For an inspectable exact case, create two rows of sixteen values. Row 0
begins `[6,3]`; row 1 begins `[2688,1344]`; the rest are zero. The global
scale is 1, and the block scales are 1 and 448, encoded as `0x38` and
`0x7e`. Both rows normalize to `[6,3]`, with E2M1 codes `0x7,0x5`.
Low-nibble-first packing produces byte `0x57` for both rows. Their bytes
match but their decoded weights differ because their scales differ.
Sixteen data bytes, two scale bytes, and a four-byte global scale total
22 bytes, versus 64 bytes for BF16.

## A numerical oracle is different from compressed execution

This small fake quantizer exposes INT8 rounding to a floating computation:

```python
def int8_qdq_per_row(weight):
    w = weight.float()
    amax = w.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax == 0, 1.0, amax / 127.0)
    q = torch.round(w / scale).clamp(-127, 127)
    return (q * scale).to(weight.dtype)
```

Calling `F.linear(x, int8_qdq_per_row(weight))` still multiplies a complete
floating weight matrix. It measures the numerical effect of the grid and
can allocate extra temporaries. Even casting `q` briefly to INT8 does not
make the eventual floating matrix compressed.

Real packed storage makes codes and scales authoritative and removes the
complete floating weight. A kernel may decode only its current tile to
BF16, or consume native low-precision operands. Those are distinct execution
levels. For QAT, an expression such as
`weight + (dequant - weight).detach()` uses the quantized forward value
with an approximate identity gradient; it still needs a later packed export.

The hardware boundary matters too. Tinyserve's measured A6000, an Ampere
SM86 GPU, supports native INT8 matrix operations but lacks native FP8/FP4
matrix arithmetic. H100 supports ordinary FP8; Blackwell adds microscaling
and NVFP4 capabilities. Supported instructions, operand layouts, and shapes
must all match. NVIDIA's [precision overview](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.13/user-guide/examples/fp8_primer.html)
describes these generation boundaries. Running a reference codec on an
A6000 does not qualify a native Blackwell kernel.

## Follow the numerical contract in code

| File | Responsibility |
|---|---|
| [quant_formats.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/quant_formats.py) | Element decoding, nearest-even encoding, nibble packing, scale recipes, and full-matrix reconstruction. |
| [quantization.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/quantization.py) | INT8 per-row packer and Q/DQ oracle, alongside serving implementations developed next. |
| [quant_inspect.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/quant_inspect.py) | Compare declared recipes with checkpoint headers without treating metadata as executable support. |
| [test_quant_formats.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/test_quant_formats.py) | All FP8 byte patterns, FP4 codes, rounding ties, endpoints, scale limits, and layout cases. |

These contracts describe runtime snapshot `e20a348`. The examples above
contain the essential encodings without requiring access to the companion
source. Codec checks establish what the bytes mean. The next chapter
follows those bytes through loading, matrix kernels, and a request-owned
KV cache, where correct representation is only the first requirement.

{% include tinyserve-book-nav.html %}
