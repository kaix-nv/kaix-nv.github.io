---
layout: post
title: "Building tinyperf, chapter 2: A GPU as a handful of rates"
date: 2026-09-29 12:02:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/02-a-gpu-as-a-handful-of-rates/
excerpt: "A performance model can't simulate tens of billions of transistors. It stands for the GPU with a few numbers, and every price it gives (a predicted time) is built from them. Which numbers, and what do they tell you before any detailed model exists?"
redirect_from:
  - /tinyperf/perf-modeling/2026/08/18/building-tinyperf-m1.html
---

*[Building tinyperf](/series/tinyperf/) · Part I: Pricing one kernel · Code: [`tinyperf/device.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/device.py), `Device`, and [`tinyperf/datatypes.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/datatypes.py) · Every table and the plot in this chapter come from `python3 book/scripts/ch02_device.py`.*

A performance model can't simulate tens of billions of transistors. It
stands for the GPU with a few numbers, and every price it gives (a
predicted time) is built from them. Which numbers, and what do they
tell you before any detailed model exists?

The short answer: about a dozen rates, nearly all from a public
datasheet. How many multiply-accumulates the GPU's tensor cores (its
matrix-multiply units) do per clock, how fast its memory moves bytes,
how much fast on-chip memory it has, and what it costs to start a
kernel, one function run on the GPU. Two of them, peak math and
memory bandwidth, already give a price for any operation, the
*roofline*, and their ratio says which of the two limits you. For LLM
inference the answer is stark: decoding a single sequence can use less
than 1% of a GPU's math.

By the end of this chapter you will know:

- which fields describe a device, and what each one prices;
- why the rates are stored per SM (streaming multiprocessor), one of
  the GPU's many processors, and per clock, and what clock a
  datasheet's peak implies;
- the roofline, arithmetic intensity (FLOPs per byte moved) and the
  ridge point (the intensity at which math and memory take equally
  long), and where prefill and decode sit against them;
- how to ask what-if questions of a device, and which operations each
  change helps;
- what datasheet rates are worth on real GPUs: 0.72–0.77 of the peak
  math in the best GEMM of cuBLAS, NVIDIA's matrix-multiply library,
  and, on an RTX A6000 streaming for seconds, 0.90 of the memory
  bandwidth.

## What a GPU is, to a performance model

A GPU is a set of **streaming multiprocessors (SMs)**: 132 on an H100
SXM, 84 on an RTX A6000. Each SM is a processor with its own schedulers,
registers and a small fast memory. A *kernel*, one function launched on
the GPU, divides its work among the SMs.

The work that dominates an LLM is matrix multiplication, and it runs on
each SM's **tensor cores**: units that multiply small blocks of matrices
in a single instruction. Their rate is counted in **multiply-accumulates
(MACs)** per clock. One MAC is a multiply and an add, so it counts as two
floating-point operations (FLOPs). An H100 SM has four tensor cores
(NVIDIA's H100 whitepaper) that together do 2,048 fp16 MACs per clock,
512 each.

Data lives at three levels:

- **DRAM**, the GPU's main memory: stacks of HBM (high-bandwidth
  memory) on datacenter parts, GDDR graphics-memory chips on workstation
  cards. It holds the weights and the KV cache (the keys and values of
  past tokens that attention rereads; chapter 6), and it is the slowest
  level.
- The **L2 cache**, 6 to 126 MB on the GPUs in this chapter, shared by
  all SMs. Everything read from DRAM passes through it.
- **Shared memory**, 100 to 228 KB per SM, managed by the kernel
  itself. A GEMM kernel stages the tiles it is working on here (blocks
  of its matrices; chapter 3).

![An H100 SXM drawn as rates: 132 SMs, each with four tensor cores and
shared memory, above an L2 cache and HBM DRAM, with NVLink to the
node's other GPUs and InfiniBand to other nodes; each number carries a
tag naming what it prices.](/assets/tinyperf-book/ch02-gpu.svg)

*Figure 2.1. An H100 SXM as tinyperf describes it. The model doesn't
know how the SMs are wired. It knows how many there are, how fast each
multiplies, how big and fast each memory level is, how fast the links
to other GPUs are, and what a kernel launch costs. Each dark tag names
the group of Table 2.1, below, whose price the number feeds. The bottom
lines multiply these into the two numbers the rest of the chapter
uses.*

## The device file

tinyperf keeps each GPU in a JSON file under `data/devices/`. The rule
for what goes in it is strict: a field earns its place only if some
price uses it. Here is the H100 as loaded, grouped by what each field
prices; two fields the file doesn't set take the class's defaults:

```
Table 2.1  The H100 SXM as loaded: its file's fields and the class defaults, by what each prices
  prices           field                     value
  math time        sm_count                  132
                   boost_clock_ghz           1.83
                   tensor_macs_per_sm_clk    fp16 2048, bf16 2048, tf32 1024, fp8 4096, int8 4096
                   sustained_clock_fraction  1.0   (class default)
                   sparse_math_multiplier    2.0   (class default)
  memory time      dram_bw_gbps              3352
                   l2_size_mb                50
                   l2_bw_gbps                11000
                   smem_kb_per_sm            228
  launch cost      kernel_launch_us          3.0
  collectives      nvlink_bw_gbps            450
                   nvlink_hop_latency_us     1.0
                   gpus_per_node             8
                   ib_bw_gbps                50.0
                   ib_latency_us             5.0
  memory capacity  hbm_gb                    80
```

Each group, field by field:

- **Math time.** `sm_count` SMs run at `boost_clock_ghz`, and
  `tensor_macs_per_sm_clk` says how many multiply-accumulates one SM's
  tensor cores do per clock, one entry per data type; their product is
  the peak (next section). `sustained_clock_fraction` scales the clock
  for a GPU that can't hold it (below). `sparse_math_multiplier` is the
  speed-up the tensor cores give a weight matrix with 2:4 structured
  sparsity, two zeros in every four values (chapter 10).
- **Memory time.** `dram_bw_gbps` is the rate every byte to or from
  DRAM is charged. `l2_size_mb` decides how much data kernels can share
  in the L2 without going back to DRAM, and `l2_bw_gbps` is the rate
  from the L2 into the SMs. `smem_kb_per_sm` is a capacity, not a rate:
  it decides which tiles a GEMM kernel can use (chapter 3). The section
  after next takes these one by one.
- **Launch cost.** `kernel_launch_us` is a fixed cost every kernel pays
  to start and finish, 3 µs by default. A near-empty GEMM called from
  PyTorch takes 18.6–34.3 µs (Table 2.7), far above that default;
  chapter 4 fits the cost of a PyTorch call.
- **Collectives**, the exchanges a group of GPUs sharing a model make
  together (chapter 11). Within a node, GPUs talk over NVLink, NVIDIA's
  direct link between them: `nvlink_bw_gbps` is each GPU's bandwidth in
  one direction, and `nvlink_hop_latency_us` the fixed cost each time
  data passes from one GPU to the next. `gpus_per_node` says how many
  GPUs share that link, eight on an H100 server. A group larger than a
  node also crosses InfiniBand, the network between nodes, at
  `ib_bw_gbps` per GPU and `ib_latency_us` per hop.
- **Memory capacity.** `hbm_gb`, the size of the DRAM, is not a time,
  but it decides whether the weights and the KV cache fit (chapter 6).

A few more fields serve special cases: the host link for offloading,
keeping weights or cache in the CPU's memory (chapter 17); the NVLink
domain, the GPUs one NVLink fabric connects when that is more than a
node (72 in a GB200 NVL72 rack); and power and price for energy and
cost (Appendix A).

Notice what's missing. There is no rate for arithmetic off the tensor
cores. Softmax, normalization and activation functions run on the SMs'
ordinary arithmetic units, and tinyperf prices them by the bytes they
move alone (chapter 5). There is no fp32 entry either: tensor cores take
fp32 data through the tf32 format (TensorFloat-32: fp32's 8-bit
exponent with a 10-bit mantissa), which has its own entry.

## Rates per SM per clock

The datasheet gives a peak in TFLOPS. The file doesn't store it. It
stores the factors, and derives the peak:

```
peak_flops = sm_count × macs_per_sm_per_clock[dtype] × 2 × clock
```

The reason is that the factors change independently, and each change is
a one-field edit:

- **Clocks change.** A GPU at its power limit runs below its rated
  clock. Scaling the clock scales the math and nothing else.
- **SM counts change.** Products of the same chip enable different
  numbers of SMs: the H100 PCIe has 114, the SXM part 132, with the same
  SM.
- **Data types are entries.** On an H100, fp8 (8-bit floating point)
  is one more entry, twice fp16 per SM.

Table 2.2 derives the peaks and sets them against NVIDIA's figures: the
A100, H100 and RTX A6000 datasheets, and the HGX B200, DGX B200 and
GB200 NVL72 specification pages. The B200 and GB200 figures (NVIDIA's
Blackwell generation, after the H100's Hopper) are for whole systems,
so the script divides them by 8 and by 72 GPUs. Most figures
are quoted with 2:4 sparsity, twice the dense rate, and the script
halves them.

```
Table 2.2  Peak dense tensor TFLOPS: derived from the device files, against the datasheets
  GPU          dtype  SMs  MACs/SM/clk  clock GHz  derived  datasheet  derived/datasheet  datasheet implies GHz
  RTX A6000    fp16    84          512      1.800    154.8      154.8              1.000                  1.800
  A100 SXM     fp16   108         1024      1.410    311.9      312.0              1.000                  1.411
  H100 SXM     fp16   132         2048      1.830    989.4      989.5              1.000                  1.830
  H100 SXM     fp8    132         4096      1.830   1978.9     1979.0              1.000                  1.830
  B200 SXM     fp16   148         3814      1.965   2218.4     2250.0              0.986                  1.993
  B200 SXM     fp8    148         7628      1.965   4436.7     4500.0              0.986                  1.993
  B200 SXM     fp4    148        15256      1.965   8873.5     9000.0              0.986                  1.993
  GB200 NVL72  fp16   148         3814      1.965   2218.4     2500.0              0.887                  2.214
  GB200 NVL72  fp8    148         7628      1.965   4436.7     5000.0              0.887                  2.214
  GB200 NVL72  fp4    148        15256      1.965   8873.5    10000.0              0.887                  2.214
```

The first three GPUs match within 0.1 TFLOPS. That is by construction:
their per-SM rates and clocks come from the same public sources, so this
checks the transcription, not the model.

The last column is the clock each datasheet peak implies at the file's
per-SM rate. For the H100 it is 1.83 GHz, which is what the file stores
in `boost_clock_ghz`. Is that the GPU's boost clock? NVIDIA's datasheet
doesn't give the SXM part's clock, but the PCIe part shows the pattern:

```
Worked example  The clock a datasheet peak implies
  H100 PCIe: 756.5 TFLOPS / (114 SMs x 2048 MACs x 2) = 1.620 GHz; its product brief's boost clock is 1.755 GHz (0.92 of it)
  B200 at 4096 MACs/SM/clk (twice the H100's): 2250.0 TFLOPS / (148 SMs x 4096 MACs x 2) = 1.856 GHz; the file stores 3814 MACs at 1.965 GHz
```

The PCIe card's tensor peak is quoted at a clock 8% below the boost
clock in its own product brief. So the H100 file follows a convention:
`boost_clock_ghz` holds the clock that reproduces the datasheet peak,
whatever the GPU's fastest clock is. For the A100 and the RTX A6000 the
implied clocks, 1.41 and 1.80 GHz, are the boost clocks (the A100's in
NVIDIA's architecture whitepaper; the A6000's datasheet says its peaks
are at the boost clock).

**The B200** breaks the convention, and its file is 1.4% low: an error.
Its per-SM rate, 3,814, is not a hardware number; the earlier
generations' rates are powers of two (512, 1,024, 2,048). NVIDIA gives
neither the per-SM rate nor the clock. If a B200 SM does 4,096 fp16 MACs
per clock, twice an H100's, the datasheet implies 1.86 GHz (the second
line above), not the file's 1.965.

**The GB200** is 11% low. Its file copies the B200's per-SM rate and
clock, but NVIDIA's rack figures work out to 2.5 dense PFLOPS per GPU,
against 2.25 for the B200. Copying per-SM rates to a sibling product is
what the layout is for; here the sibling's peak differs, presumably
through its clock, and the table catches it at once.

Nor is the datasheet's clock the one a GPU sustains: a card at its power
limit lowers its clock until its power fits. `sustained_clock_fraction`
multiplies the clock in the peak. No device file sets it. It is for
projections on hardware nobody has measured, such as a fleet known to
run power-limited. "How close is it?" below shows what a real card's
clock does.

## Memory: DRAM, L2 and shared memory

```
Table 2.3  The memory levels in the device files
  GPU          DRAM GB  DRAM GB/s  datasheet  L2 MB  L2 GB/s, estimate  L2 GB/s, fitted  smem KB/SM
  RTX A6000         48        768        768      6               2000             4000         100
  A100 SXM          80       2039       2039     40               4500                -         164
  H100 SXM          80       3352       3350     50              11000             8250         228
  B200 SXM         180       8000       8000    126              18000            27000         228
```

**DRAM bandwidth** is on every datasheet, and the files carry it. It is
the rate the roofline charges for every byte.

**L2 capacity** decides how much data kernels can share without going
back to DRAM. Chapter 3 uses it to price how much a GEMM's tiles reuse
each other's inputs.

**L2 bandwidth**, the rate from the L2 into the SMs, is on no datasheet.
The files' numbers are estimates. Chapter 4 fits the model's constants
to measured GEMMs, and the fit moves this one by factors of 0.75 to 2
(the fitted column; the A100 has no measurements). Chapter 3 finds that
L2 bandwidth rarely limits a GEMM; when it does, this guess sets the
price.

**Shared memory** is recorded as a capacity per SM. Its bandwidth isn't
in the file: the model assumes a kernel whose tiles fit keeps its tensor
cores fed.

## The roofline

With a peak rate and a bandwidth we can price any operation. Count its
FLOPs and the bytes it must move to and from DRAM at the minimum (every
input read once, every output written once). Then:

```
time = max(flops / peak_flops, bytes / memory_bandwidth)
```

The `max` says the GPU overlaps moving data with computing on it, so the
slower of the two sets the time. It is also a floor: a kernel whose
inputs start in DRAM can beat neither term.

Divide an operation's FLOPs by its bytes and you get its **arithmetic
intensity**, in FLOPs per byte: how much work it does with each byte it
moves. Rewritten in those terms, the attainable rate of math is

```
attainable_flops = min(peak_flops, intensity × memory_bandwidth)
```

On log-log axes that is a roof: a line rising with the bandwidth, which
flattens when it reaches the peak. The corner is the **ridge point**,
`peak_flops / memory_bandwidth`. An operation with an intensity below it
is *memory-bound*: its bytes set its time, and the tensor cores idle
part of it. Above it, the operation is *math-bound*.

```
Table 2.4  Ridge points: FLOPs per byte of DRAM traffic at which math time equals memory time
  GPU         fp16/bf16    fp8    fp4
  RTX A6000         202      -      -
  A100 SXM          153      -      -
  H100 SXM          295    590      -
  B200 SXM          277    555   1109
  A100 SXM -> H100 SXM: fp16 math x3.17, DRAM bandwidth x1.64
  H100 SXM -> B200 SXM: fp16 math x2.24, DRAM bandwidth x2.39
```

These are large numbers. An H100 at fp16 needs 295 FLOPs for every byte
it fetches from DRAM, or its tensor cores wait; adding two fp16 vectors
does one FLOP per 6 bytes.

The fp16 ridge nearly doubled from the A100 to the H100, whose math grew
3.17 times and bandwidth 1.64 times. It dipped slightly on the B200,
whose bandwidth grew faster than its math (2.39 times against 2.24).
What does move right every generation is the ridge of the narrowest
floating-point format each one offers, because a new format doubles the
math rate without adding bandwidth: 153 for the A100's fp16, 590 for the
H100's fp8, 1,109 for the B200's fp4 (4-bit floating point).

A ridge moving right means more work falls to its left, where the math
is mostly idle and what you have bought is bandwidth. It also means a
kernel's main job is to use each byte many times, which is what chapter
3's tiles and L2 reuse are about.

## The intensity of LLM work

Where do an LLM's operations sit against these ridges? Take the
multiply of B rows of activations by a weight matrix of K × N, in fp16.
It does `2 · B · K · N` FLOPs. At the minimum it reads the weights,
`2 · K · N` bytes, and the activations, `2 · B · K`, and writes the
output, `2 · B · N`. Divide:

```
intensity = 2·B·K·N / (2·(K·N + B·K + B·N)) = 1 / (1/B + 1/K + 1/N)
```

While B is small next to K and N, the intensity is close to B: a weight
GEMM does about as many FLOPs per byte as it has rows. As B grows it
levels off, and it never exceeds `K·N / (K + N)`, 2,048 for a
4096 × 4096 weight.

- **Decode** runs one row per sequence: 1 FLOP per byte for a single
  sequence, about B for a batch of B.
- **Prefill** runs one row per prompt token: about T for a prompt of T
  tokens up to a few hundred, but only a third of T (1,365) at 4,096
  tokens on this weight.
- **Decode attention**, each new token attending over its sequence's
  cached keys and values, is different. Each sequence reads its own KV
  cache, so batching adds bytes as fast as FLOPs. With grouped-query
  attention (GQA: several query heads share one key and value head;
  chapter 6), each 2-byte KV value feeds one MAC, 2 FLOPs, per query
  head in its group, so at fp16 the intensity equals the group size:
  32/8 = 4 for Qwen3-8B, at any batch.

> **Field note: FLOPs per weight, or per byte?** This project's first
> device model put single-sequence decode at "about 2 FLOPs per weight
> byte". That is FLOPs per *weight*: one multiply-accumulate, two FLOPs.
> At fp16 a weight is two bytes, so it is 1 FLOP per byte. Either way decode
> sits two orders of magnitude below every ridge, but the slip would
> halve the batch at which decode becomes math-bound.

Table 2.5 prices one layer of Qwen3-8B this way: its 4096 × 4096
attention output projection at several row counts (decode sequences or
prefill tokens), and its decode attention for 64 sequences at a context
of 4,096 tokens. The last columns are the most of the peak math the
roofline allows, `min(1, intensity / ridge)`.

```
Table 2.5  The arithmetic intensity of LLM work, fp16, and the most of peak math the roofline allows
  operation (one Qwen3-8B layer)  FLOP/byte  RTX A6000   A100 SXM   H100 SXM   B200 SXM
  decode, 1 sequence                    1.0       0.5%       0.7%       0.3%       0.4%
  decode, 8 sequences                   8.0       4.0%       5.2%       2.7%       2.9%
  decode, 64 sequences                 62.1      30.8%      40.6%      21.0%      22.4%
  decode, 256 sequences               227.6     100.0%     100.0%      77.1%      82.1%
  decode attention, 64 x 4096           4.0       2.0%       2.6%       1.4%       1.4%
  prefill, 512 tokens                 409.6     100.0%     100.0%     100.0%     100.0%
  prefill, 4096 tokens               1365.3     100.0%     100.0%     100.0%     100.0%
```

Decoding one sequence can use at most 0.3% of an H100's math. Even 64
sequences use at most about a fifth of it. At 256 sequences the GEMM is
math-bound on the two Ampere GPUs (the A6000 and A100, NVIDIA's
generation before the H100), whose ridges are lowest, and still just
memory-bound on the H100 and B200. Decode attention stays at 1–3% no
matter the batch. Prefill is math-bound everywhere from 512 tokens.

![Log-log roofline of four GPUs at fp16, with five operations of a
Qwen3-8B layer marked on every roof.](/assets/tinyperf-book/ch02-roofline.svg)

*Figure 2.2. The roofline of four GPUs at fp16. Each roof rises with the
GPU's DRAM bandwidth and flattens at its peak, the ridge point at the
corner. The dots mark one Qwen3-8B layer's operations on each roof.
Decode of one sequence and decode attention sit far down every slope,
where the GPUs differ only in bandwidth; 64 sequences are still on it.
At 256 sequences only the two Ampere GPUs have reached their flat roofs.
A 4096-token prefill sits at every GPU's peak.*

Where exactly does decode cross over? Solving for the batch at which the
GEMM's intensity reaches the ridge:

```
Worked example  The decode batch at which a 4096 x 4096 weight GEMM turns math-bound (roofline)
  RTX A6000   fp16 224   (fp16 ridge 202; crossover / ridge 1.111)
  A100 SXM    fp16 166   (fp16 ridge 153; crossover / ridge 1.085)
  H100 SXM    fp16 345, fp8 345   (fp16 ridge 295; crossover / ridge 1.169)
  B200 SXM    fp16 321, fp8 321, fp4 321   (fp16 ridge 277; crossover / ridge 1.158)
```

Two things stand out. The crossover is 8.5–17% above the ridge itself,
because the activations' bytes grow with the batch. And it is the same
batch at every precision. A narrower format doubles the math rate, which
doubles the ridge, but it also halves the bytes per weight, which
doubles the operation's intensity. The two cancel: the crossover batch
is, roughly, the tensor cores' MACs per second divided by the weights
per second the memory delivers. (Weight-only quantization, 4-bit weights
multiplied in 16 bits, shrinks the bytes without raising the math rate,
and moves the crossover down; chapter 10.)

So, by the roofline, a decode step's weight GEMMs stay memory-bound
until the batch reaches a few hundred sequences, and its attention never
turns math-bound. That is why serving engines batch decodes. Chapter 3
prices the GEMMs more carefully, and chapter 6 builds a whole decode
step.

## The code

Here is the `Device` class, trimmed to what this chapter uses. The
`...` stands for the fields of later chapters (collectives, the NVLink
domain, the host link, sparsity, power and price); the `nvl_domain_gpus`
and `available` helpers and `load`'s docstring are also cut.

```python
@dataclass(frozen=True)
class Device:
    name: str
    sm_count: int
    boost_clock_ghz: float
    # dense tensor-core MACs (multiply-accumulates) per SM per clock, by dtype key
    tensor_macs_per_sm_clk: dict = field(default_factory=dict)
    dram_bw_gbps: float = 0.0          # GB/s
    l2_size_mb: float = 0.0
    l2_bw_gbps: float = 0.0            # GB/s (approximate; calibrate if you can)
    smem_kb_per_sm: float = 0.0        # usable shared memory per SM
    kernel_launch_us: float = 3.0      # launch + prolog/epilog overhead per kernel
    nvlink_bw_gbps: float = 0.0        # per-direction bandwidth per GPU
    ...
    sustained_clock_fraction: float = 1.0
    hbm_gb: float = 0.0                # device memory capacity

    def peak_tflops(self, dtype: DType) -> float:
        """Dense peak at the sustained clock, TFLOPS. 1 MAC = 2 FLOPs."""
        macs = self.tensor_macs_per_sm_clk.get(dtype.key)
        if macs is None:
            raise ValueError(f"{self.name} has no tensor throughput for {dtype.key}")
        return self.sm_count * macs * 2 * self.boost_clock_ghz * self.sustained_clock_fraction / 1e3

    def math_time_s(self, flops: float, dtype: DType) -> float:
        return flops / (self.peak_tflops(dtype) * 1e12)

    def dram_time_s(self, nbytes: float) -> float:
        return nbytes / (self.dram_bw_gbps * 1e9)

    def l2_time_s(self, nbytes: float) -> float:
        if self.l2_bw_gbps <= 0:
            return 0.0
        return nbytes / (self.l2_bw_gbps * 1e9)

    def ridge_flops_per_byte(self, dtype: DType) -> float:
        """Arithmetic intensity at which math time equals DRAM time."""
        return self.peak_tflops(dtype) * 1e12 / (self.dram_bw_gbps * 1e9)

    def roofline_time_s(self, flops: float, nbytes: float, dtype: DType) -> float:
        """Speed-of-light estimate: whichever of math or memory dominates."""
        return max(self.math_time_s(flops, dtype), self.dram_time_s(nbytes))

    @classmethod
    def load(cls, name: str, **overrides) -> "Device":
        path = DEVICE_DIR / f"{name}.json"
        spec = json.loads(path.read_text())
        dev = cls(**spec)
        if overrides:
            dev = replace(dev, **overrides)
        return dev
```

Every derived number is a method, so nothing derived can go stale when
a field changes. A dtype without an entry raises: asking an A100 for its
fp8 peak is an error, not zero. The `DType` enum in `datatypes.py`
carries just two properties per type, its bytes per element (fp16 2,
fp8 1, fp4 0.5) and the key that picks the tensor-core rate. Those are
the only two ways a data type reaches a price.

## What-if devices

The class is frozen, so a what-if is a new device: `load` takes keyword
overrides and returns a copy with those fields replaced.

```python
fat = Device.load("h100_sxm", dram_bw_gbps=2 * 3352)          # twice the bandwidth
slow = Device.load("h100_sxm", sustained_clock_fraction=0.85)  # a power-limited clock
```

Table 2.6 prices Table 2.5's operations on both, by the roofline. Its
speed-ups are the as-shipped time divided by the what-if's, so 0.85x is
a slowdown:

```
Table 2.6  What-if H100s on the roofline: which operations each edit helps
  operation (one Qwen3-8B layer)  FLOP/byte  as shipped us   bound  speed-up, 2x bandwidth  speed-up, clock x0.85
  decode, 1 sequence                    1.0           10.0  memory                   2.00x                  1.00x
  decode, 8 sequences                   8.0           10.0  memory                   2.00x                  1.00x
  decode, 64 sequences                 62.1           10.3  memory                   2.00x                  1.00x
  decode, 256 sequences               227.6           11.3  memory                   1.30x                  1.00x
  decode attention, 64 x 4096           4.0          320.6  memory                   2.00x                  1.00x
  prefill, 512 tokens                 409.6           17.4    math                   1.00x                  0.85x
  prefill, 4096 tokens               1365.3          138.9    math                   1.00x                  0.85x
```

Doubling the bandwidth halves the price of every operation that stays
memory-bound and leaves prefill untouched; a slower clock does the
reverse. Decode of 64 sequences costs 3% more than decode of one,
because the weights, not the rows, set its time. At 256 sequences, twice
the bandwidth halves the ridge to 148, the GEMM crosses into math-bound,
and it gains 1.30 times instead of 2.

This is the first answer a performance model gives about hardware:
which resource an operation spends. A decode-heavy workload buys
bandwidth, a prefill-heavy one buys math. The same override answers
"what if 8 SMs were disabled?" (`sm_count=124`).

## How close is it?

What are the datasheet rates worth on a real GPU? tinyperf's
calibration set times 27 fp16 GEMMs through cuBLAS on each of
three GPUs, one `torch.matmul` call at a time, the median of 30 timings
with a warm cache (`data/calibration/measurements_*.json`). Chapter 4
fits the model to them. Here we only ask what the best of them achieves;
nothing is fitted in Table 2.7.

```
Table 2.7  What the datasheet rates are worth: fp16 GEMMs through cuBLAS, one call at a time
  GPU        best TFLOPS  of peak         best shape  one-row GEMM us   GB/s  of datasheet  at datasheet us  128^3 GEMM us
  RTX A6000        116.7     0.75    2048x32768x4096            213.0    630          0.82            174.8           34.3
  H100 SXM         759.8     0.77     8192x8192x8192             59.7   2248          0.67             40.1           19.8
  B200 SXM        1606.6     0.72    2048x32768x4096             41.5   3235          0.40             16.8           18.6
  one-row GEMM: 1 x 8192 x 8192, which streams a 134 MB weight
```

**Math.** The best GEMM on each GPU reaches 0.72–0.77 of the derived
peak, so a roofline at datasheet rates would price these GEMMs at about
three quarters of their measured time. Chapter 3's tile model, which
prices how a kernel cuts a GEMM into blocks, explains little of that
for large GEMMs (Table 3.2 puts an 8192³ GEMM on an A100 at 98.9% of
peak). The rest is not explained here; on the A6000 it is consistent
with a power-limited clock (below; not profiled).

**Memory.** A one-row GEMM against an 8192 × 8192 weight is a decode
step's shape: it streams 134 MB and does almost no math. The A6000 moves
it at 0.82 of its datasheet bandwidth, the H100 at 0.67, the B200 at
0.40. The B200 number is not its memory speed. At the datasheet rate the
whole transfer takes 16.8 µs, and a 128³ GEMM, which moves almost
nothing, takes 18.6 µs. Every call pays a fixed cost to be launched from
PyTorch and to start and finish, and on the fastest GPU that cost is as
large as the work. Chapter 4 separates the two by fitting a rate and a
fixed cost together.

Timed over seconds of back-to-back calls, the picture changes.
`tools/measure_power.py` repeats each load for 12 to 23 seconds on an
RTX A6000 while it samples the card's power
(`data/power/rtx_a6000_measured.json`):

```
Sustained  RTX A6000: loads repeated back to back for 12-23 s, against one call
  8192^3 fp16 GEMM                              106.1 TFLOPS  0.69 of peak       drawing 299 W of its 300 W limit
  8192^3 fp16 GEMM, one call (Table 2.7's set)  106.8 TFLOPS  0.69 of peak
  1 GB copy (1 GB read, 1 GB written)           682.9 GB/s    0.89 of datasheet
  one-row GEMM, 8192 x 8192 weight              694.9 GB/s    0.90 of datasheet  drawing 299 W
```

Over many calls the one-row GEMM streams at 0.90 of the datasheet rate,
the same as a plain copy, and the same fraction chapter 4 fits for this
card's DRAM (`dram_efficiency` in `data/calibration/rtx_a6000.json`).
The 8192³ GEMM runs at 0.69 of peak, the same as one call of it, while
drawing 299 W of the card's 300 W limit: consistent with a clock held
down by power, but not proof, since the tool's clock readings from this
run are not usable.

Another run does show the clock. While the same card served Qwen3-8B
through vLLM, an open-source serving engine, a monitor sampled
nvidia-smi once a second
(`data/validation/gpu_telemetry_long_varied_rtx_a6000.json`):

```
Clocks  RTX A6000 serving Qwen3-8B (vLLM), nvidia-smi once a second, 751 samples at >= 90% utilization
  SM clock MHz: p10 1470, median 1680, p90 1815 (range 1425-1905); the device file's clock 1800
  power: median 294 W of 300 W; clock held down by the power cap in 428 samples, by thermal slowdown in 331
```

The clock wandered between 1,425 and 1,905 MHz, with a median 7% below
the 1,800 MHz the datasheet's peaks assume. nvidia-smi reported the
power cap holding it down in more than half the samples, and thermal
slowdown in nearly half. The rated clock is neither a floor nor a
ceiling: the card ran above it at times, and below it most of the time.

These are the numbers a calibrated model has to carry. Chapter 4 fits
them as constants per GPU: a fraction of peak math, a fraction of
bandwidth, a fixed cost per kernel.

## Where it breaks

- **Datasheet numbers are not achievable rates, and some are guesses.**
  cuBLAS's best GEMM reaches 0.72–0.77 of peak math, and the shortfall
  depends on the shape (chapter 3) and the clock. Where a datasheet gives
  only a peak, the clock and per-SM rate behind it are guesses, and a
  file can be wrong outright (the B200 1.4% low, the GB200 11%). L2
  bandwidth is on no datasheet; fitting moves the estimates by factors of
  0.75 to 2.
- **Clocks move.** The real clock depends on power, temperature and
  the work. The model has one clock per device, and
  `sustained_clock_fraction` is one number for all work.
- **One number per memory level hides how memory is used.** The
  bandwidth assumes long, sequential streams in which every byte fetched
  is used. A scattered read moves whole 32-byte sectors to use part of
  one; small transfers never reach the full rate. The roofline's bytes
  are also the minimum, each input read once, where a real kernel may
  read its inputs several times (chapter 3).

## What you built

- A GPU described by about a dozen rates, stored per SM per clock, with
  the peaks derived, so that clocks, SM counts and data types are
  one-field edits.
- The roofline,
  `time = max(flops / peak_flops, bytes / memory_bandwidth)`, with
  arithmetic intensity and the ridge point.
- The intensity of LLM work: a weight GEMM does about as many FLOPs per
  byte as it has rows, so decode does about B and prefill about T, while
  decode attention stays at the GQA group size at any batch.
- What-if devices by override, and the first thing they tell you: which
  resource an operation spends.
- Evidence: cuBLAS reaches 0.72–0.77 of peak math; an A6000 sustains
  0.90 of its bandwidth.

## Exercises

1. Add an H100 PCIe device file: 114 SMs, 2,000 GB/s, and the SXM part's
   per-SM rates. Which clock do you store, 1.62 GHz or 1.755 GHz, and
   why? What are its fp16 ridge point and its decode crossover batch?
2. Fix the GB200 file with one field so that its fp16 peak matches the
   datasheet's 2.5 PFLOPS. Which rows of Tables 2.5 and 2.6 change on a
   GB200, and which don't?
3. Price Table 2.6's operations on an H100 with 8 SMs disabled. Which
   operations notice?
4. Compute the intensity of a whole decode step of Qwen3-8B at 64
   sequences and a context of 4,096: all 36 layers' weight GEMMs and
   their KV cache reads together. Is the step memory- or math-bound on a
   B200? At what context do the KV cache's bytes outweigh the weights'?
   (Chapter 6 builds this step.)
5. On a GPU you have, repeat an 8192³ fp16 `torch.matmul` for twenty
   seconds while `nvidia-smi` reports clock and power. What fraction of
   peak do you sustain, and at what clock?

---

*[← Chapter 1: What a performance model is for]({% post_url 2026-09-29-tinyperf-01-what-a-performance-model-is-for %}) · [Contents](/series/tinyperf/) · [Chapter 3: Pricing a GEMM →]({% post_url 2026-09-29-tinyperf-03-pricing-a-gemm %})*
