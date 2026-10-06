---
layout: post
title: "Building tinyperf, chapter 10: Precision and sparsity as passes"
date: 2026-09-29 12:10:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/10-precision-and-sparsity/
excerpt: "Running a model in FP8 or FP4 (8- and 4-bit floating point), with 4-bit weights, or with half its weights pruned is advertised as 2×, 4×, 4× and 2× faster. What does each buy, operation by operation? And how does a performance model express \"this deployment\" without a second model of the network?"
redirect_from:
  - /tinyperf/perf-modeling/2026/08/31/building-tinyperf-m35.html
  - /tinyperf/perf-modeling/2026/09/08/building-tinyperf-m49.html
---

*[Building tinyperf](/series/tinyperf/) · Part II: One model on one GPU · Code: [`tinyperf/passes.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/passes.py), `apply_recipe`, `apply_weight_only`, `apply_sparsity`, and [`tinyperf/gemm_model.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/gemm_model.py), `estimate_gemm` · Every table and plot in this chapter comes from `python3 book/scripts/ch10_precision.py`.*

Running a model in FP8 or FP4 (8- and 4-bit floating point), with 4-bit
weights, or with half its weights pruned is advertised as 2×, 4×, 4× and
2× faster. What does each buy, operation by operation? And how does a
performance model express "this deployment" without a second model of
the network?

The short answer: each option changes at most three numbers of an
operation: the bytes of its weights, the bytes of its other operands and
its math rate. What that buys depends on which the operation was
spending. Decode spends bytes, so shrinking the weights helps it.
Prefill spends math, so only a faster rate helps, and 4-bit weights
multiplied in 16 bits don't. No step gains the headline factor, because
some operations shrink less or not at all: the output head, the norms,
the launches (each kernel's fixed cost to start), and the KV cache,
which the shipped FP4 recipe (a choice of format for each operation)
keeps at fp8 and so shrinks only 2×. Passes (functions that rewrite a
graph; chapter 5) express a deployment as tags on the graph the builder
already emitted.

By the end of this chapter you will know:

- what each option changes in one GEMM, and why 4-bit weights with
  16-bit math move the decode crossover (the batch at which a weight
  GEMM turns math-bound) down 3.8 times;
- how three passes tag a built graph, and how the GEMM model reads them;
- what FP8 and FP4 buy class by class, and why "everything in FP4"
  claims 3.7× where the deployable recipe gets 2.0×;
- what 4-bit experts (the small feed-forward networks of a
  mixture-of-experts, or MoE, layer; chapter 9) bought gpt-oss-20b, and
  how close the model gets;
- what 2:4 sparsity (two of every four weights zero) would buy,
  unmeasured like FP8 and FP4;
- how each option changes the memory left for the KV cache.

## What a format changes in one GEMM

Chapter 2 gave a data type two properties: its bytes per element and the
tensor-core rate it maps to. A weight GEMM, `C[M,N] = A[M,K] · B[K,N]`
with the activations in A and the weights in B, has three numbers a
format can change: A's bytes, B's bytes and the math rate.

- **Compute precision, FP8 or FP4.** A and B are stored in the narrow
  type and multiplied at its rate: twice bf16's for fp8, four times for
  fp4, which of this chapter's GPUs only the B200 has. All three change.
- **Weight-only quantization.** B is stored in 4 bits and unpacked to 16
  inside the kernel, so the math keeps the 16-bit rate. Here the format
  is MXFP4 (chapter 9): 32 four-bit values share an 8-bit scale, 4.25
  bits or 0.53125 bytes per weight. Only B shrinks, 3.76 times.
- **2:4 sparsity.** Two of every four weights along K are zero. The
  tensor cores skip them, at twice the dense rate, and B keeps the
  survivors plus a 2-bit index each: `0.5 + 0.125 / bytes` of dense,
  0.5625 at 16 bits. B shrinks and the rate doubles.

Table 10.1 prices one 4096 × 4096 weight GEMM in five formats on an
H100, at a decode-like 8 rows and a prefill-like 8,192, by the roofline
(the larger of the math time and the memory time; chapter 2) and by
chapter 3's tile model ("model"), which splits the output into tiles,
each computed by one thread block, or CTA (cooperative thread array).

```
Table 10.1  One 4096 x 4096 weight GEMM on an H100 in five formats: datasheet rates
                            bytes per value  tensor  weights crossover    8 rows: speed-up   8192 rows: speed-up
  format                          A       B  TFLOPS       MB      rows   roofline    model     roofline    model
  bf16                            2    2.00     989     33.6       345       1.00     1.00         1.00     1.00
  fp8                             1    1.00    1979     16.8       345       2.00     1.63         2.00     1.98
  4-bit weights, bf16 math        2    0.53     989      8.9        92       3.72     1.73         1.00     1.03
  2:4 sparse, bf16                2    1.12    1979     18.9       467       1.77     1.50         2.00     1.39
  2:4 sparse, fp8                 1    0.62    3958     10.5       519       3.19     2.12         4.00     2.61
  bf16 us: 8 rows roofline 10.0, model 13.1; 8192 rows roofline 277.8, model 301.9
  4-bit weights at 8 rows, model: math 4.5 us (64-row tiles, 64 CTAs), DRAM 2.8 us, launch 3.0 us
  RTX A6000 crossover rows: bf16 224, 4-bit weights 60, 2:4 bf16 283
  fp4 on the H100: ValueError: H100_SXM has no tensor throughput for fp4 — supported: ['bf16', 'fp16', 'fp8', 'int8', 'tf32']
```

The crossover is the fewest rows at which the roofline calls the GEMM
math-bound. fp8 halves the bytes and doubles the rate, which cancel: 345
rows, as for bf16 (chapter 2). 4-bit weights shrink only the bytes, so
the crossover moves down by nearly the byte ratio: 92 rows, and 60
instead of 224 on the RTX A6000. 2:4 doubles the rate but less than
halves the bytes, so it moves up.

![Log-log roofline time of a 4096 by 4096 weight GEMM on an H100 in four
formats, from 1 to 4096 rows.](/assets/tinyperf-book/ch10-formats.svg)

*Figure 10.1. The GEMM by the roofline. Each curve is flat while the
weights' bytes set the time and rises once the math does. fp8 halves
both parts; 4-bit weights lower only the flat part, turn up at 92 rows
and join bf16 at 345; 2:4 lowers the flat part to 0.56 and joins fp8
above 467 rows.*

At 8 rows each format gains what it takes off the weight bytes; at 8,192
rows, what it adds to the rate. The tile model mostly gains less: at 8
rows the launch doesn't shrink, and the model prices the 4-bit GEMM's
math above its bytes: its best tile pads 8 rows to 64, and each of 64
CTAs runs the whole K loop, 4.5 µs against 2.8 µs of streaming. At 8,192
rows 2:4 reaches 1.39×, not 2× (see "2:4 sparsity"). And the H100 has no
fp4 rate; asking for one is an error.

## Passes: tags on a built graph

A model builder (chapter 6) writes the network's math in the model's
dtype. A deployment decides, op by op, what to store and multiply
narrower. tinyperf keeps the jobs apart: the builder emits the graph
once, then a *pass* (chapter 5) walks it and tags each op it applies
to:

- `apply_recipe(graph, {prefix: dtype})` sets an op's compute dtype;
- `apply_weight_only(graph, {prefix: bytes per weight})` sets the width
  its weights are stored at;
- `apply_sparsity(graph, prefixes)` marks its weights 2:4 sparse.

`apply_recipe`, docstring trimmed:

```python
def apply_recipe(graph: Graph, recipe: dict) -> int:
    n = 0
    for op in graph:
        if op.op_type not in ("gemm", "fmha", "linattn"):
            continue
        for prefix, dt in recipe.items():
            if op.name.startswith(prefix):
                op.attrs["dtype"] = dt
                n += 1
                break
    return n
```

The first prefix that matches an op's name wins. Only the math families
(an op's family is its kind; chapter 5) take a dtype: GEMMs, fused
attention (`fmha`, all of attention in one kernel; chapter 7) and linear
attention (chapter 8); chapter 5's memory-bound `rw` operators keep the
model's. The weight-only pass runs the same loop over GEMMs, with a
refusal:

```python
            if op.name.startswith(prefix):
                if not op.name.startswith(WEIGHT_GEMM_PREFIXES):
                    raise ValueError(f"{op.name} has no weight operand to quantize")
                op.attrs["weight_nbytes"] = float(nbytes)
...
MXFP4_BYTES = 0.53125                        # 4-bit weights + e8m0 scale per 32
WEIGHT_ONLY_MXFP4_EXPERTS = {"moe_gate_up": MXFP4_BYTES, "moe_down": MXFP4_BYTES}
```

`WEIGHT_GEMM_PREFIXES` lists the GEMMs with a weight matrix; attention's
`Q·Kᵀ` and `P·V` multiply activations and are not on it.
`apply_sparsity` is the same loop, writing `op.attrs["sparse"] = True`.
A recipe is a dictionary. Here is the FP4 serving recipe that ships,
`RECIPE_FP4_SERVING`, trimmed to the entries a dense or MoE transformer
uses:

```python
    return {
        "qkv_proj": DType.FP4, "attn_out": DType.FP4,
        "ffn_": DType.FP4, "moe_gate_up": DType.FP4, "moe_down": DType.FP4,
        # routers stay high precision in every shipped checkpoint
        # (gpt-oss lists mlp.router in modules_to_not_convert)
        "moe_router": DType.FP16,
        ...
        "attn_fmha": DType.FP8, "attn_qk": DType.FP8, "attn_pv": DType.FP8,
        ...
        "lm_head": DType.FP16,
    }
```

Weight GEMMs in fp4, attention (and so its cache) in fp8, the head and
the router (the GEMM that picks each token's experts; chapter 9) in 16
bits. This chapter's FP8 recipe is the same dictionary with fp8 for fp4;
tinyperf ships it as `RECIPE_FP8_SERVING`. Table 10.2 shows what four
deployments write on Qwen3-8B; each row but the head's stands for 36
layers.

```
Table 10.2  The tags each pass puts on a Qwen3-8B decode graph (one layer and the head)
  op             family             FP8 recipe            FP4 recipe         4-bit weights               2:4 all
  qkv_proj       gemm                dtype=fp8             dtype=fp4   weight_nbytes=0.531                sparse
  attn_fmha      fmha                dtype=fp8             dtype=fp8                     -                     -
  attn_out       gemm                dtype=fp8             dtype=fp4   weight_nbytes=0.531                sparse
  ffn_gate_up    gemm                dtype=fp8             dtype=fp4   weight_nbytes=0.531                sparse
  ffn_down       gemm                dtype=fp8             dtype=fp4   weight_nbytes=0.531                sparse
  lm_head        gemm               dtype=fp16            dtype=fp16                     -                     -
  the 7 rw ops (ln_attn, rope, residual_attn, ln_ffn, swiglu, residual_ffn, ln_final): no tags
  ops tagged                                 6                     6                     4                     4
```

### How the model reads the tags

The scheduler (chapter 5) prices each op in its `dtype` tag, if it has
one, and hands `sparse` and `weight_nbytes` to chapter 3's
`estimate_gemm`. These are the lines chapter 3 trimmed from it:

```python
    in_b, out_b = dtype.nbytes, out_dtype.nbytes
    ...                                              # no rate for this dtype: ValueError
    macs_per_sm_clk = device.tensor_macs_per_sm_clk[dtype.key]
    b_scale = 1.0
    macs_per_sm_clk *= math_scale
    if weight_nbytes is not None:
        b_scale = weight_nbytes / in_b
    if sparse:
        macs_per_sm_clk *= device.sparse_math_multiplier
        b_scale *= 0.5 + 0.125 / in_b                # nonzeros + 2-bit indices
    ...
            percta_bytes = ctas * (bm + bn * b_scale) * k_split * in_b + out_bytes
    ...
            window = (rows * bm + cols * bn * b_scale) * in_b
    ...
            ideal_bytes = batch * (m * k + k * n * b_scale) * in_b + out_bytes
    ...
            dram_s = device.dram_time_s(dram_bytes) / dram_scale
```

The dtype picks the tensor-core rate and both operands' bytes. `b_scale`
rescales B wherever chapter 3 counts its bytes: from the L2 cache to
the SMs (streaming multiprocessors), the L2 reuse window, the minimum
DRAM traffic. `sparse_math_multiplier` is a device field, 2.0 by default
(chapter 2). C keeps the graph's dtype. `math_scale` and `dram_scale`
are constants of the calibrated tier (below), the tier that prices with
constants fitted to one GPU and software stack (chapter 4).

Fused attention takes one dtype for everything (`estimate_fmha`):

```python
    math_s = flops / (device.peak_tflops(dtype) * 1e12 * eff)
    dram_bytes = dtype.nbytes * batch * (m * k + kv * (k + out_dim) + m * out_dim)
```

An fp8 tag on attention means fp8 math and an fp8 KV cache: the
`kv · (k + out_dim)` term is the cache.

**Why tags, not a rebuilt graph?** Rebuilding is the naive estimate:
build the model with `dtype=FP4` and every tensor is fp4, the norms, the
head and the cache included. A tag answers per op, and it composes: the
same passes work on dense, MoE, hybrid and latent-attention graphs
(chapters 8–9) at any parallel layout (chapters 11–12), the serving
simulator (chapter 14) applies them to every step, and the capacity
model (chapter 6's count of the batch that fits in memory) reads the
same dictionaries. No builder knows they exist.

## Precision recipes: FP8 on an H100, FP4 on a B200

Table 10.3 prices a Qwen3-8B decode step and prefill, class by class,
under the FP8 recipe on an H100 and the FP4 recipe on a B200, at
datasheet rates (chapter 4's projected tier).

```
Table 10.3  What a precision recipe buys, class by class: Qwen3-8B, datasheet rates
  each cell: the class's share of the bf16 step, and its speed-up under the recipe
                             H100, FP8             H100, FP8             B200, FP4             B200, FP4
                       decode 8 x 4096      prefill 8 x 2048       decode 8 x 4096      prefill 8 x 2048
  weight GEMMs             64%   1.83x           83%   1.96x           59%   2.50x           87%   3.78x
  attention                22%   1.87x            5%   1.99x           19%   1.74x            4%   1.97x
  LM head                   5%   1.00x            0%   1.00x            5%   1.00x            0%   1.00x
  other ops                 9%   1.00x           12%   1.00x           18%   1.00x            9%   1.00x
  step, ms            7.19 ->   4.39    298.29 -> 169.35      3.72 ->   2.11    173.49 ->  58.92
  step speed-up                  1.64x                 1.76x                 1.76x                 2.94x
```

![Horizontal bars of each class's speed-up under the FP8 recipe on an
H100 and the FP4 recipe on a B200, in decode and prefill.](/assets/tinyperf-book/ch10-op-classes.svg)

*Figure 10.2. Table 10.3's speed-ups. The head and the memory-bound
operators gain nothing, which holds whole steps to 1.64–2.94×.*

**Weight GEMMs** gain most in prefill, where their math or L2 traffic
sets the time and the format shrinks both: 1.96× for fp8 and 3.78× for
fp4, near the rate ratios. In decode they gain the byte ratio less what
doesn't shrink:

```
Worked example  Why the B200's weight GEMMs gain 2.5x, not 4x, in an FP4 decode step (batch 8)
  weights: 6.95 B parameters = 13.9 GB in bf16, 1.74 ms at 8000 GB/s; 3.47 GB in fp4, 0.43 ms
  launches: 144 GEMMs x 3 us = 0.43 ms in either format
  sum: bf16 2.17 ms, fp4 0.87 ms, 2.50x; the model: 2.17 and 0.87 ms, 2.50x
```

Launches are half the fp4 GEMM time, and not as an artifact of the
tier: a kernel replayed from a CUDA graph (launches recorded once and
replayed as one) on a B200 costs 3.5 µs (chapter 4, Table 4.6).

**Attention**, tagged fp8, gains up to 2×, held to 1.74–1.87× in decode
by its launch. **The head**, kept at 16 bits, and **the `rw` operators**,
untagged, gain nothing; on the B200's decode step they are 23% of the
bf16 time. So a step gains 1.64–1.76× in decode and 1.76–2.94× in
prefill. "4× from
FP4" is a prefill number for GEMMs alone.

### The KV cache decides the long-context answer

Decode attention's share of the step grows with the context (chapter
6). Table 10.4 holds the batch at 64 and prices three versions of each
recipe: with attention and its cache left in bf16, as shipped (fp8
cache), and naive (every tensor in the format).

```
Table 10.4  As the KV cache grows: Qwen3-8B decode, 64 sequences, speed-up over bf16, datasheet rates
                                                    B200, FP4                       H100, FP8
   context B200 bf16 ms attention 16-bit cache  recipe   naive    16-bit cache  recipe   naive
      1024         4.41       30%        1.42x   1.77x   2.20x           1.31x   1.67x   1.76x
      2048         5.62       45%        1.31x   1.82x   2.44x           1.22x   1.74x   1.81x
      4096         8.03       62%        1.20x   1.87x   2.76x           1.14x   1.82x   1.87x
      8192        12.86       76%        1.11x   1.92x   3.13x           1.08x   1.89x   1.92x
     16384        22.53       86%        1.06x   1.95x   3.45x           1.04x   1.93x   1.96x
     32768        41.85       93%        1.03x   1.97x   3.68x           1.02x   1.96x   1.98x
  16-bit cache: the recipe with attention and its cache left in bf16; naive: every tensor in the format
  64 sequences fit on the B200 (chapter 6's rule) up to 8192 tokens in bf16, 32768 with the FP4 recipe
  64 sequences fit on the H100 (chapter 6's rule) up to 4096 tokens in bf16, 8192 with the FP8 recipe; rows past 8192 tokens price a batch that fits in neither format
```

![Speed-up over bf16 against context for Qwen3-8B decode of 64
sequences: FP4 naive, FP4 recipe, FP4 GEMMs with a 16-bit cache, and the
H100's FP8 recipe.](/assets/tinyperf-book/ch10-context.svg)

*Figure 10.3. Table 10.4's B200 speed-ups, and the H100's FP8 recipe:
each tends to the speed-up of its cache's format.*

At 32,768 tokens the B200's step is 93% attention, and its speed-up is
the cache's: 1.03× with a 16-bit cache, 1.97× with fp8, 3.68× with fp4.
The naive 3.68× is datasheet arithmetic for "FP4"; the shipped recipe
gets 1.97×, and the gap widens with the context, where the speed-up was
wanted most. On the H100 the naive estimate stays within 6% of the
recipe, whose cache is fp8 too: the naive
estimate misleads when the cache can't follow the weights.

The last lines add a caveat: from 16,384 tokens, 64 sequences fit a
B200 only with the recipe, and the H100 in neither format. Part of a
recipe's value is capacity.

## Weight-only quantization: 4-bit weights, 16-bit math

A GPU without a 4-bit rate, such as the RTX A6000, can still store
4-bit weights. A weight-only kernel such as Marlin (chapter 9) unpacks
and scales them to 16 bits for the tensor cores. The model prices that:
B at `weight_nbytes`, the math at the activation dtype, scaled in the
calibrated tier (chapter 4) by a fitted `weight_only_math_efficiency`,
0.66 on the RTX A6000. At a few rows the tile model may also split K
(split-K, chapter 3): several thread blocks share each tile's K loop,
and a second kernel adds their partial sums.

```
Table 10.5  A 4096 x 4096 weight GEMM with 4-bit weights and bf16 math, RTX A6000 at its fitted rates
   rows  bf16 us  bound  4-bit, x0.66 math  bound  speed-up  4-bit, x1.0 math  speed-up
      1     52.2   dram               20.3   dram     2.58x              20.2     2.59x
      8     52.3   dram               21.7   dram     2.41x              20.9     2.50x
     32     52.8   dram               27.9   math     1.90x              23.7     2.23x
     64     53.6   dram               40.9   math     1.31x              28.2     1.90x
    128     55.1   dram               78.2   math     0.70x              52.8     1.04x
    256    102.1   math              152.9   math     0.67x             102.1     1.00x
   8192   2419.4   math             3663.9   math     0.66x            2419.4     1.00x
  fitted rates: bf16 116.1 TFLOP/s, DRAM 691 GB/s, launch 3.5 us
  4-bit roofline crossover: datasheet 60 rows, fitted x0.66 math 32; split-K at 1 and 8 rows: 4, 4
```

At a few rows the 4-bit GEMM is 2.4–2.6× faster: the weight stream is
3.76 times shorter, but the launches aren't; at these rows the model
splits K, which adds a second launch. By 32 rows its math binds, before
the datasheet roofline's 60, because the fitted rate times 0.66 is
lower: at that rate the roofline turns at 32. From 128 rows it is slower
than bf16, 0.66× at large sizes, the constant itself. Without the
constant it would break even. For prefill, 4-bit weights with 16-bit
math gain nothing at best.

The 0.66 was fitted end to end (to whole prefill times, not to a kernel
timed alone) on one model and one GPU: gpt-oss-20b with 4-bit experts,
at batch 8 and 32, checked on batch 1. It absorbed more than unpacking;
chapter 22 tells what it hid. No 4-bit dense GEMM has been timed here,
and Table 10.5's two columns are two guesses, not a bracket: the one
4-bit kernel measured, Marlin on gpt-oss's experts, runs at 0.34 of the
dense rate at 128 tokens per launch (below), under both.

### MXFP4 experts in gpt-oss-20b

gpt-oss-20b ships with its experts in MXFP4 and the rest (attention,
router, embedding, head) in bf16
(`data/validation/vllm_gpt_oss_20b_rtx_a6000.json`); tinyperf's
dictionary is `WEIGHT_ONLY_MXFP4_EXPERTS`. A decode step gains in
proportion to the experts' share of what it reads:

```
Worked example  What a gpt-oss-20b decode step reads, bf16 and MXFP4 experts, batch 1
  experts: 19.1 B of 20.9 B parameters; one step at batch 1 reads 4 of 32 per layer, 2.39 B
  bytes read: bf16 7.21 GB; MXFP4 experts 1.27 GB + the rest in bf16 2.44 GB = 3.71 GB (1.95x fewer)
```

At batch 1 the bf16 remainder, mostly attention and the head, is two
thirds of the bytes left, so the step reads 1.95× fewer, not 3.76×. A
bigger batch touches more experts (chapter 9) and gains more.

Both formats' expert kernels were timed inside real decode steps, each
launch paired with its step's routing (chapter 9's Table 9.3 has bf16
alone):

```
Table 10.6  The expert kernel at decode, bf16 against MXFP4 (Marlin), timed inside real decode steps: gpt-oss-20b, RTX A6000
         experts touched     gate-up us per expert        down us per expert  MXFP4 rate / fitted
  batch    bf16    MXFP4     bf16   MXFP4 speed-up     bf16   MXFP4 speed-up         gate-up/down
      1     4.0      4.0     48.0    15.7    3.05x     24.4     9.1    2.68x            0.81/0.70
      8    12.7     12.7     47.2    14.6    3.24x     23.8     7.8    3.06x            0.87/0.82
     32    18.6     18.3     49.9    16.1    3.10x     24.4     8.5    2.87x            0.79/0.75
  bytes per weight 2 / 0.53125 = 3.76 times fewer; the model streams bf16 experts at 1.00 and Marlin at 0.80 of the fitted 691 GB/s: 3.01x
```

Per touched expert Marlin is 2.7–3.2× faster against 3.76× fewer
bytes, because it streams below the full rate (chapter 9); the model's
3.01× falls inside the measured range.

Prefill surprises. For a dense GEMM the model says 4-bit weights lose
(Table 10.5), yet gpt-oss-20b's MXFP4 prompts ran 1.31–1.56× faster than
its bf16 ones (Table 10.9), which is consistent with the kernels, not
the format:

```
Worked example  Expert math in prefill on this engine, RTX A6000 (fitted dense bf16 rate 116.1 TFLOP/s)
  Marlin, MXFP4 experts, measured by tokens per launch: 128 39, 256 52, 1024 73, 4096 87, 16384 90 TFLOP/s
  bf16 experts, as the model prices them: 0.44 x 116.1 = 51.1 TFLOP/s at best (moe_math_efficiency, fitted end to end)
  the generic weight-only path: 0.66 x 116.1 = 76.6 TFLOP/s (weight_only_math_efficiency, fitted end to end)
```

Marlin, timed alone under the engine's own routing, runs faster than
the model's fitted price for the bf16 experts from 256 tokens per
launch. A speed-up is a ratio of two kernels, not two formats, and the
model comes close because it carries a rate for each. The branch of
`_exec_gemm` (`scheduler.py`) that picks the rates is too tangled to
quote; as pseudocode, leaving out a second measured table, for steps
that mix decodes with a prompt chunk (chapter 16):

```
# pseudocode: the weight-only branch of _exec_gemm, calibrated tier
math_scale = weight_only_math_efficiency           # 0.66, fitted
if the op is an expert GEMM:
    dram_scale = weight_only_dram_efficiency       # 0.80, Table 10.6
    if tokens per launch >= 128 and a measured Marlin rate table exists:
        math time = FLOPs / measured rate(tokens per launch)
        time = max(math time, weight stream at 0.80, L2 time) + launch
```

On gpt-oss-20b, 0.66 prices no prefill at all. It remains for decode
launches and unmeasured weight-only GEMMs.

## 2:4 sparsity

Two sparsity recipes ship: the feed-forward GEMMs (`SPARSITY_2_4_FFN`),
and every weight GEMM but the router and head (`SPARSITY_2_4_ALL`).
Table 10.7 prices both on Qwen3-8B on an H100, alone and with FP8.

```
Table 10.7  2:4 sparsity on Qwen3-8B, one H100, datasheet rates: step time and speed-up over dense bf16
                          decode 8 x 4096  decode 64 x 4096   prefill 8 x 2048   gate-up time vs dense
  deployment                    ms      x        ms      x        ms      x      decode     prefill
  bf16, dense                 7.19  1.00x     17.49  1.00x    298.29  1.00x   1.00 dram     1.00 l2
  2:4 FFN                     5.77  1.25x     16.07  1.09x    242.94  1.23x   0.58 dram     0.71 l2
  2:4 all                     5.38  1.34x     15.67  1.12x    227.57  1.31x   0.58 dram     0.71 l2
  FP8 recipe                  4.39  1.64x      9.63  1.82x    169.35  1.76x   0.52 dram     0.51 l2
  FP8 recipe + 2:4 all        3.61  1.99x      8.85  1.98x    139.05  2.15x   0.35 dram     0.39 l2
  attention under 2:4 all, time vs dense: 1.00, 1.00, 1.00
  apply_sparsity(g, ('attn_qk',)): ValueError: attn_qk has no weight operand to sparsify
```

- **Decode is the bytes story.** The gate-up GEMM takes 0.58 of its
  dense time: the 0.5625 byte factor plus a launch. The step gains
  1.25–1.34× at 8 sequences, 1.09–1.12× at 64, where the cache, which
  sparsity doesn't touch, is most of the step.
- **Prefill is the rate story, cut short.** The rate doubles, but the
  gate-up GEMM falls only to 0.71. In the tile model it is bound by L2
  traffic, the panels every CTA pulls into shared memory (chapter 3).
  2:4 shrinks B's panels, not A's, so that traffic falls to 0.71 while
  the math halves. The L2 bandwidth behind it is an estimate on no
  datasheet (chapter 2).
- **Attention gains nothing.** It has no weights; tagging it is an
  error.

Tags compose: with FP8 as well, the step gains about 2×.

Sparse kernel efficiency is assumed equal to dense: chapter 3's kernel
model with the rate doubled and B shrunk. Real sparse kernels have their
own tiles and read index metadata; none has been timed here.

## Memory

Shrinking weights frees memory; a narrower cache frees more. For
gpt-oss-20b, Table 10.8 also gives vLLM's KV pool, the memory the engine
sets aside for the cache (chapter 17), with `max_num_seqs`, its cap on
running requests, at 64:

```
Table 10.8  What each pass does to memory
  Qwen3-8B: the largest decode batch that fits (chapter 6's rule: weights + KV + working memory <= 90%)
  GPU   deployment             weights GB  KV bytes/token   8192  32768
  H100  bf16                         16.4         147,456     46     11
  H100  2:4 all                      10.3         147,456     51     12
  H100  FP8 recipe                    9.4          73,728    103     25
  B200  bf16                         16.4         147,456    120     30
  B200  FP4 recipe                    6.0          73,728    258     64
  gpt-oss-20b on one RTX A6000: the weights, and vLLM's KV pool (chapter 17's rule; max_num_seqs 64)
  deployment                   weights GB  checkpoint GB  pool tokens  engine logged
  bf16 everywhere                   41.82          41.83       67,312              -
  MXFP4 experts (weight-only)       13.74          13.76      638,432        607,344
```

For a dense model the cache is the capacity story (chapter 6): 2:4 frees
6 GB on the H100 and admits 5 more sequences of 8,192 tokens; the FP8
recipe frees 7 GB, halves the cache, and admits more than twice as many.

For gpt-oss-20b the weights decide: 4-bit experts free 28 GB of a 48 GB
card. The model's weight bytes match both checkpoints' files within
0.2%, the one direct check of this chapter's byte accounting. vLLM's
pool (chapter 17) grows from 67,312 to 638,432 tokens, 5% above the
607,344 the engine logged
(`data/validation/predictions_gpt_oss_20b_rtx_a6000_online_b_before_measurement.txt`).

> **Field note: one width, two places.** For a while the step prices
> streamed gpt-oss-20b's experts at 4.25 bits while capacity held them
> at bf16. The simulated server's 67,312-token pool capped its batch
> near 45 requests, one of three reasons it queued at 4 requests per
> second where the engine never did. A deployment is one decision; read
> it from one place.

## How close is it?

First, what is unmeasured. Every GEMM tinyperf has timed on an H100 or
a B200 was fp16 (`data/calibration/measurements_*.json`), and no
engine run in `data/validation/` used an fp8 cache, an FP8 or FP4
recipe, or a sparse checkpoint. Tables 10.1, 10.3, 10.4 and 10.7 are
projections at datasheet rates. What can be checked is the byte
accounting, and Table 10.8's weights match a real 4-bit checkpoint.

The one measured deployment is gpt-oss-20b served by vLLM on one RTX
A6000, as shipped and with its experts converted to bf16: batches of 1,
8 and 32, prompts of 512–8,192 random tokens, TTFT (time to first
token) and TPOT (time per output token) as in chapter 6. Both runs let
vLLM choose its attention kernel, Triton's for gpt-oss on this GPU. The
model follows the repository's tests: MXFP4 with that kernel's measured
price and Marlin's measured rates; bf16 with FlashAttention-2's price,
the one its fused-MoE constant (the 0.44 above) was fitted with. Each
ratio is the model's time over the measured one, so above 1 the model
reads high; a set of ratios is summed up by its range or its typical
error, `exp(mean |ln ratio|) − 1`, the geometric mean distance from 1
(chapter 3).

```
Table 10.9  gpt-oss-20b with MXFP4 experts on one RTX A6000 (vLLM): model/measured, and the speed-up over bf16
                       MXFP4, model/measured    speed-up over bf16, TTFT            TPOT
  batch prompt  TTFT ms  ratio  TPOT ms  ratio    measured   model        measured   model
      1    512     59.4  0.922     7.05  1.050       1.56x   1.75x           1.64x   1.62x
      1   2048    183.7  0.981     7.21  1.034       1.34x   1.36x           1.62x   1.61x
      1   8192    850.0  1.022     7.72  0.995       1.32x   1.19x           1.58x   1.60x
      8    512    344.0  0.924    12.68  1.016       1.42x   1.50x           2.42x   2.16x
      8   2048   1374.7  0.969    13.77  0.968       1.38x   1.45x           2.19x   2.12x
      8   8192   6925.8  0.996    14.90  1.017       1.31x   1.19x           1.99x   1.98x
     32    512   1299.9  0.948    18.07  1.025       1.42x   1.54x           2.38x   2.30x
     32   2048   5494.6  0.967    20.22  1.006       1.38x   1.45x           2.26x   2.18x
  MXFP4 TTFT: 0.922-1.022, typical error 4.1%; TPOT: 0.968-1.050, typical error 2.3%
  speed-up, model/measured: TTFT 0.90-1.12, typical error 7.2%; TPOT 0.89-1.01, typical error 3.0%
  bf16, model/measured: TTFT 0.907-1.033 (8192-token prompts 0.921, 0.907); TPOT 0.907-1.038
  bf16 priced with Triton attention instead: TTFT 0.992-1.084; predicted speed-up at 8192-token prompts 1.40x
  measured speed-up: TTFT 1.31-1.56x, TPOT 1.58-2.42x
  weight_only_math_efficiency at 1.0 instead of 0.66: TTFT moves at most 0.0%, TPOT at most 0.5%
```

**Decode.** MXFP4 TPOT lands at 0.968–1.050, typical error 2.3%, and the
predicted speed-up over bf16 at 0.89–1.01 of the measured. Both are held
out, with nothing in the model fitted to them: the routing came from a
separate measurement (chapter 9), Marlin's 0.80 from Table 10.6. The
speed-up grows from about 1.6× at batch 1 to 2.0–2.4× at 8 and 32, with
the experts' share of the bytes.

**Prefill.** MXFP4 TTFT lands at 0.922–1.022, typical error 4.1%, from
Marlin's rate and the attention kernel's, each measured alone. The
predicted speed-up is furthest off at the ends: 1.19× against 1.31–1.32×
on the 8,192-token prompts, where the bf16 model reads 8–9% low, and
1.75× against 1.56× at 1 × 512. bf16 TTFT is not held out: its constant
was fitted to it. Priced with Triton's attention instead, bf16 TTFT
reads 0.99–1.08 and the 8,192-token speed-up 1.40×; the 8–9% is
consistent with part of the attention price living inside the fitted
0.44.

And 0.66? At 1.0 instead, no TTFT moves and no TPOT moves more than
0.5%: the constant fitted to this grid no longer shapes it. A caveat
covers all of it: the grid has been a test since it was measured, and
every later model change had to keep it in bounds.

## Where it breaks

- **Nothing narrow or sparse has been timed.** Every FP8, FP4 and 2:4
  number assumes a 16-bit kernel's fraction of peak.
- **One dtype per op.** An attention tag sets both the cache's bytes and
  the math rate. An fp8 cache read by 16-bit math gets the bytes and not
  the rate, which no tag can say.
- **Scales.** `DType.FP4` is 0.5 bytes; an MXFP4 weight with its scale is
  0.53125. The weight-only pass counts the scale, a recipe doesn't: its
  fp4 weights are about 6% light against MXFP4.
- **Names.** An unlisted name is silently left alone. gpt-oss's
  sliding-window layers (which attend only to recent tokens; chapter 8)
  fuse to `attn_swa_fmha`, which no prefix of `RECIPE_FP4_SERVING`
  matches, so half its attention is priced at 16 bits while capacity
  stores that cache at fp8.
- **Few rows.** The model's split-K costs a second launch, so a 4-bit
  GEMM's math floor binds at 8 rows on an H100 (Table 10.1); a kernel
  splitting K in one launch, like chapter 3's stream-K, wouldn't pay it.
- **One constant, fitted end to end.** 0.66 was fitted to one model's
  prefill times on one GPU; elsewhere it is a guess.
- **Accuracy.** The model prices whatever a recipe asks for; whether
  the model's answers survive is the checkpoint's problem.

## What you built

- Formats as changes to A's bytes, B's bytes and the math rate, and
  three passes that tag a built graph with them by name.
- The decode crossover: unchanged by FP8, down 3.8× with 4-bit weights,
  up with 2:4.
- Recipes priced class by class, and the naive all-FP4 3.7× against the
  deployable 2.0× at long context.
- Evidence: MXFP4 experts on gpt-oss-20b, TPOT within 5% and its
  speed-up over bf16 within 11%.

## Exercises

1. Price a cache-only recipe, `{"attn_fmha": DType.FP8}`, on the H100
   across Table 10.4's contexts. From what context does it beat the
   "16-bit cache" column?
2. Give fused attention a separate cache dtype, so an fp8 cache can be
   read by bf16 math. How much of attention's prefill gain in Table 10.3
   survives?
3. Add `"attn_swa_fmha"` to a copy of `RECIPE_FP4_SERVING` and price a
   gpt-oss-20b decode step on a B200 with and without it. How far off
   was the step?
4. Time a 4-bit weight-only GEMM (4096 × 4096 weight) from 1 to 1,024
   rows on a GPU you have. Is `weight_only_math_efficiency` a constant
   or a curve?
5. Time a 2:4 sparse GEMM against its dense twin, fit a sparse
   efficiency as chapter 4 fits dense ones, and redo Table 10.7's prefill
   column.

---

*[← Chapter 9: Mixture of experts]({% post_url 2026-09-29-tinyperf-09-mixture-of-experts %}) · [Contents](/series/tinyperf/) · [Chapter 11: Collectives and tensor parallelism →]({% post_url 2026-09-29-tinyperf-11-collectives-and-tensor-parallelism %})*
