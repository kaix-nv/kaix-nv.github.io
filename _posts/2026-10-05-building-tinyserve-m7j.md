---
layout: post
math: true
title: "Building tinyserve M7j: Fuse Kimi's learned residual choice"
date: 2026-10-05 08:08:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Fuse Kimi attention-residual selection with an online softmax and retain the readable reference."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m7j-fused-kimi-residual.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M7i — Pack Kimi experts, then batch the selected work]({% include tinyserve-post-url.html slug="building-tinyserve-m7i" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7i-packed-kimi-experts.md" %}) · Next: [M7k — Make GDN prefill chunkwise]({% include tinyserve-post-url.html slug="building-tinyserve-m7k" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7k-chunkwise-gdn.md" %})

M7i removed the routed-MoE launch storm. That changed the profile again.
Kimi's attention-residual mixer now consumes `15.9%` of batch-1 decode model
time and `13.7%` at batch 4. One model forward invokes it 34 times: once after
almost every attention sublayer, once after every MLP path, and once before
the final norm.

The arithmetic is tiny. Each invocation chooses a learned mixture of at most
four 256-value residual vectors. The readable PyTorch expression nevertheless
launches a chain for concatenation, FP32 normalization, scoring, softmax,
matrix multiplication, and casting. M7j keeps that expression as the oracle
and implements the same decision as one small Triton kernel written directly
for Tinyserve. It does not add FLA or another attention library.

[![The PyTorch oracle materializes candidate and normalization tensors before
softmax and matrix multiplication. The M7j kernel streams each candidate into
an online softmax accumulator and writes the same output in one
launch.](/assets/tinyserve/m7j-fused-kimi-residual.svg)](/assets/tinyserve/m7j-fused-kimi-residual.svg)

## First understand what AttnRes chooses

This is not the ordinary Transformer residual `x + f(x)`. Kimi carries two
kinds of values through the layer stack:

- `prefix_sum [N,H]` accumulates the current block's sublayer outputs;
- `block_residual [N,R,H]` retains a stream at layers 0, 8, and 16.

Here `N` is the number of real token rows, `H=8` for the tiny checkpoint,
and `R` grows from one to three. Before selected sublayers, AttnRes places the
current prefix after the saved streams, scores all `R+1` candidates, and
returns a weighted mixture.

For candidate vector $v_c$, RMS-normalized vector $\hat v_c$, learned norm
scale $w$, and one-row projection $p$, the score is

$$
\hat v_c = \frac{v_c}{\sqrt{\operatorname{mean}(v_c^2)+\epsilon}},
\qquad
s_c = \sum_h \hat v_{c,h} w_h p_h.
$$

The selected residual is

$$
y = \sum_c \operatorname{softmax}(s)_c v_c.
$$

Both the score and softmax depend on every candidate. We therefore cannot
replace this operation with addition or choose only the highest-scoring
vector.

## A concrete three-candidate example

Use two saved vectors plus the current prefix, ignore the tiny epsilon, and
let the combined score weight $w \odot p=[1,0]$:

```text
saved 0: [ 0, 1]  -> RMS-normalized [ 0.000, 1.414] -> score  0.000
saved 1: [-1, 0]  -> RMS-normalized [-1.414, 0.000] -> score -1.414
prefix:  [ 2, 0]  -> RMS-normalized [ 1.414, 0.000] -> score  1.414
```

Softmax turns those scores into approximately `[0.187, 0.045, 0.768]`.
The output is the mixture of the original, unnormalized vectors:

```text
0.187 * [0,1] + 0.045 * [-1,0] + 0.768 * [2,0]
    = [1.491, 0.187]
```

That last detail matters. RMS normalization decides the weights, but the
weighted sum uses the original residual values.

## Why the direct expression costs several launches

The PyTorch oracle mirrors the equation closely:

```python
values = torch.cat([block_residual, prefix_sum.unsqueeze(1)], dim=1)
values_float = values.float()
normalized = values_float * torch.rsqrt(
    values_float.pow(2).mean(-1, keepdim=True) + norm.eps
)
score_weight = norm.weight.float() * projection.weight.squeeze(0).float()
probabilities = (normalized * score_weight).sum(-1).softmax(-1).unsqueeze(1)
output = torch.matmul(probabilities, values_float).squeeze(1).to(values.dtype)
```

It materializes `[N,R+1,H]` values and normalized values, then reads the
original candidates again for the matrix multiplication. The tensors are
small, so GPU execution and temporary-allocation overhead are large relative
to the useful arithmetic. Repeating the chain 34 times per forward turns that
local inefficiency into a measurable decode cost.

## Stream the candidates through an online softmax

The native kernel assigns one Triton program to each token row. It keeps one
`H`-value FP32 accumulator in registers and visits the saved residuals in
their original order, followed by the prefix. For each candidate it computes
the RMS and score, then updates a numerically stable online softmax.

If `m`, `d`, and `a` are the running maximum, denominator, and weighted-vector
accumulator, a new score $s_c$ and vector $v_c$ update them as follows:

$$
m' = \max(m,s_c), \qquad
\alpha = e^{m-m'}, \qquad
\beta = e^{s_c-m'},
$$

$$
d' = \alpha d + \beta, \qquad
a' = \alpha a + \beta v_c.
$$

After the final candidate, the program writes `a / d` in the input dtype.
There is no concatenated candidate tensor, no normalized tensor, and no
separate softmax or matrix-multiply launch. FP32 RMS, scoring, exponentials,
and accumulation preserve the oracle's numerical boundary.

This is the same idea used to make attention softmax streamable, applied to a
much smaller learned residual choice. The kernel is deliberately narrow: Kimi
has only `R=1..3`, so a compile-time loop can unroll every candidate.

## Keep the optimization boundary explicit

`apply_attention_residual()` exposes three backends:

- `torch` always runs the readable oracle;
- `triton` requires the native contract and fails clearly otherwise;
- `auto`, the serving default, selects Triton only when the contract fits.

The contract requires CUDA inference, contiguous BF16 or FP16 tensors, equal
dtypes, and one to three saved residuals. CPU, FP32, autograd, unexpected
shapes, or a future Kimi variant with more streams stays on PyTorch. An
optimization for one checkpoint must not silently become a different model
equation for another.

The backend switch also makes the comparison causal. One loaded model can run
the same prompts first with `torch`, then with `triton`, without changing the
checkpoint, cache, scheduler, expert path, or sampling policy.

## Numerical and lifecycle checks

The focused tests cover all native shapes used by the tiny checkpoint:

1. batch rows 1 and 4, residual counts 1 through 3, and hidden widths from
   the 8-value fixture through a synthetic 7,168-value production-like row;
2. native BF16 output against the FP32-accumulating PyTorch expression;
3. automatic CPU fallback and an explicit error for unsupported forced use;
4. four-request continuous serving with identical generated token ids and
   text across `torch` and `triton` backends.

The complete suite then exercises KDA, MLA, MoE, hybrid cache gather/commit,
chunked prefill, scheduling, and the earlier dense/distributed paths. The
measured tree passes `101` tests in `382.71 s`.

## Measured effect

Both September 3, 2026 runs used the same loaded BF16 tiny Kimi-K3 model on one
RTX A6000. Each condition had two warmups and five measured repetitions. The
backend order alternated within every repetition; the independent second run
started with the opposite backend.

The isolated benchmark performs 200 mixer invocations per timed sample. Across
batch sizes 1, 2, and 4 and all three residual counts, speedup ranges from
`4.64×` to `5.17×` across the final-source runs.

The end-to-end table reports median generated-token throughput:

| run | batch | PyTorch oracle | native kernel | speedup |
|---|---:|---:|---:|---:|
| PyTorch-first | 1 | 24.19 tok/s | 26.43 tok/s | 1.092× |
| PyTorch-first | 2 | 45.92 tok/s | 50.20 tok/s | 1.093× |
| PyTorch-first | 4 | 92.09 tok/s | 98.60 tok/s | 1.071× |
| Triton-first | 1 | 24.48 tok/s | 26.84 tok/s | 1.097× |
| Triton-first | 2 | 46.98 tok/s | 51.96 tok/s | 1.106× |
| Triton-first | 4 | 91.05 tok/s | 94.02 tok/s | 1.033× |

Every paired repetition generated identical token ids. Peak allocated memory
was also identical between backends at each batch size: `436.6 MiB`,
`457.2 MiB`, and `506.7 MiB`.

The five-repeat B4 rows disagree about the predeclared 5% gate: the first run
measures `+7.1%`, while its opposite-order replica measures only `+3.3%`.
Hiding that row would make the result look cleaner but less trustworthy. Two
deeper B4-only controls therefore use three warmups and 11 alternating
repetitions:

| B4 control | PyTorch oracle | native kernel | speedup |
|---|---:|---:|---:|
| PyTorch-first | 83.94 tok/s | 99.64 tok/s | 1.187× |
| Triton-first | 82.36 tok/s | 89.90 tok/s | 1.092× |

The absolute rates drift across short runs, and low sampled clocks often fall
between requests rather than inside useful GPU work. That makes cross-run
absolute throughput a poor causal signal here. Both deeper, within-process
alternating controls clear 5%, while the full sweeps show no B1 regression
(`+9.2%` and `+9.7%`). M7j therefore clears the promotion gate, with the
variance retained as an explicit measurement limitation.

## Run and debug it

The **M7j: fused Kimi residual mixer** launch configuration forces the native
backend over four requests. Set breakpoints in this order:

1. `KimiDecoderLayer.forward()` — inspect when the saved stream grows at
   layers 0, 8, and 16;
2. `apply_attention_residual()` — inspect the backend contract and dispatch;
3. `fused_attention_residual()` — inspect the flattened row count, hidden
   width, residual count, and Triton launch grid;
4. `attention_residual_torch()` — force `--kimi-residual-backend torch` to
   step through the executable oracle.

```bash
.venv/bin/python examples/generate.py \
  --model /home/scratch.kaix_coreai/models/kimi-k3 \
  --serve --hybrid-slots 4 --chunk-size 512 \
  --prompts "2+2=" "Kimi" "A longer prompt" \
    "The capital of France is" \
  --no-chat --max-new-tokens 4 --backend gather \
  --kimi-residual-backend triton --verbose
```

Python's debugger can stop at the dispatch and launch wrapper but cannot
single-step a GPU program. Read `_attention_residual_kernel()` beside the
small numerical example above, then use `torch` backend breakpoints to inspect
the same intermediate values.

The paired benchmark records every repetition, token id, peak allocation,
checkpoint/source hashes, and GPU samples:

```bash
.venv/bin/python examples/bench_kimi_residual.py \
  --device cuda:0 --first-backend torch \
  --output .tinyserve-bench/m7j/kimi-residual-ab.json
```

`examples/profile_kimi.py` records the module-level CUDA-event attribution.
Use `--residual-backend torch` to inspect the M7i oracle or `auto` for the
current path; profiler-instrumented wall time is diagnostic rather than a
throughput result.

The structured [M7j evidence](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7j-fused-kimi-residual-a6000-2026-09-03.json)
retains the two-run medians, validation result, implementation fingerprint,
and hashes of both raw artifacts.

## Takeaway

After a large bottleneck moves, re-profile before guessing again. M7i made
expert execution fast enough that 34 tiny learned residual decisions became
visible. Their math did not need approximation or a third-party attention
stack; it needed a launch shape matching the actual problem.

M7j keeps the direct PyTorch equation as an oracle and fuses only its stable,
small inference contract. The remaining profile is now dominated by broader
KDA work. That category contains projections, causal convolutions,
normalization, and recurrence, so the next step is another attribution pass—not
an assumption that the recurrence kernel must be next.
