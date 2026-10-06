---
layout: post
math: true
title: "Building tinyserve M7o: Fuse KDA's pair construction"
date: 2026-10-05 08:13:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Fuse construction of the two causal KDA pair matrices while leaving the rest of the algorithm readable."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m7o-fused-kda-pairs.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M7n — Tune the KDA chunk before fusing the kernel]({% include tinyserve-post-url.html slug="building-tinyserve-m7n" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7n-kda-chunk64.md" %}) · Next: [M7p — Fuse KDA's correction and state tail]({% include tinyserve-post-url.html slug="building-tinyserve-m7p" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7p-fused-kda-tail.md" %})

M7n made a 2,048-token prompt faster by changing KDA's inner chunk from 32 to
64 tokens. The post-change profile still assigned 78.9% of the model interval
to KDA, and 96.0% of that interval to the chunkwise rule. The expensive-looking
triangular solve accounted for only 6.2% of isolated CUDA operator time. The
larger problem was the repeated pair-matrix construction around it.

M7o fuses that measured region and nothing else. A small Triton kernel written
inside Tinyserve replaces the repeated subtract, mask, exponential, multiply,
reduction, and accumulation chain. It writes the two causal pair matrices
directly. The PyTorch triangular solve and the four state/output matrix
multiplications remain visible and executable.

[![The M7n PyTorch oracle visits four 32-channel tiles and materializes a
token-pair-channel decay tensor for each tile. M7o streams all 128 channels in
one Triton launch per KDA chunk and directly emits the lower and query-pair
matrices. The right-hand side, triangular solve, output calculation, and
recurrent-state update remain unchanged.](/assets/tinyserve/m7o-fused-kda-pairs.svg)](/assets/tinyserve/m7o-fused-kda-pairs.svg)

## Begin with the two matrices

Consider one batch row, one head, and one KDA inner chunk of width `C`. For
token `t`, source token `i`, and key channel `d`, let `G[t,d]` be the
cumulative log decay inside this chunk. Here `q` already includes the rule's
`1/sqrt(K)` query scaling. KDA needs two causal coefficients:

$$
A_{t,i} = \sum_d k_{t,d} k_{i,d}
           e^{G_{t,d}-G_{i,d}},
\qquad
P_{t,i} = \sum_d q_{t,d} k_{i,d}
           e^{G_{t,d}-G_{i,d}},
\qquad i \le t.
$$

`A` describes how earlier key updates interact when KDA solves the chunk.
`P` describes how each query reads those solved updates. If `b[t]` is the
sigmoid of Kimi's learned update gate, the lower-triangular system matrix is

$$
L_{t,i} =
\begin{cases}
1, & i=t,\\
b_t A_{t,i}, & i<t,\\
0, & i>t.
\end{cases}
$$

The kernel returns `L` and `P`, each shaped `[B,H,C,C]`. It does not replace
the solve that consumes `L`, and it does not change what `P` means.

## A concrete three-token row

Use `C=3`, two key channels, and inspect target token `t=2`. Suppose

```text
k0 = [1, 0]          G0 = [-0.1, -0.2]
k1 = [0, 1]          G1 = [-0.3, -0.5]
k2 = [0.707, 0.707]  G2 = [-0.6, -0.9]
q2(raw) = [0.6, 0.8] -> q2 = [0.424, 0.566] after 1/sqrt(2)
b2 = 0.8
```

For source `i=0`, the channel-wise decay from token 0 to token 2 is
`exp(G2-G0) = [0.607, 0.497]`. The decayed source key is therefore
`[0.607, 0]`, giving

```text
A[2,0] = k2 · [0.607, 0] = 0.429
P[2,0] = q2 · [0.607, 0] = 0.257
L[2,0] = b2 * A[2,0]     = 0.343
```

For source `i=1`, the same steps produce approximately
`A[2,1]=0.474`, `P[2,1]=0.379`, and `L[2,1]=0.379`. The diagonal is fixed to
one, so the kernel writes the final rows

```text
L[2,:] = [0.343, 0.379, 1.000]
P[2,:] = [0.257, 0.379, 0.700]
```

Rows above the causal diagonal are zero. The real checkpoint repeats this
calculation over 128 channels, two heads, every token row, and every inner
chunk.

## What the readable path was doing

The M7n implementation avoids a prompt-sized `[T,T,K]` allocation by reducing
32 key channels at a time. For `K=128`, one chunk makes four trips through this
loop:

```python
relative_log_decay = G[:, :, :, None, tile] - G[:, :, None, :, tile]
relative_decay = relative_log_decay.masked_fill(~causal, -inf).exp()
decayed_source_key = key[:, :, None, :, tile] * relative_decay
A += (key[:, :, :, None, tile] * decayed_source_key).sum(-1)
P += (query[:, :, :, None, tile] * decayed_source_key).sum(-1)
```

This is a useful equation oracle: every intermediate has a name and can be
inspected in a debugger. It is a poor GPU launch shape. Each tile materializes
`[B,H,C,C,32]`, launches elementwise work and two reductions, accumulates into
`A` and `P`, and then discards the large temporary. The next tile repeats the
same chain.

## Stream channels and write the answer once

`_kda_pair_kernel` launches a two-dimensional grid. One program owns one
target token `t` and up to 16 source tokens `i`; the grid covers every batch,
head, target, and causal source block. The program loads all 128 channels,
forms `G[t]-G[i]`, applies the causal mask before the exponential, and reduces
both dot products in FP32. It writes `L[t,i]` and `P[t,i]` directly.

The causal mask must be applied before `exp`. Above the diagonal,
`G[t]-G[i]` can be positive and large even though that pair is illegal.
Computing the exponential first could create an infinity that later masking
does not reliably contain.

The implementation also accepts arbitrary tensor strides. Kimi's causal
convolution naturally produces a token-major view whose channel dimension is
not contiguous. Copying that view solely to satisfy a kernel would add work at
the exact boundary M7o is trying to remove.

## Keep the rest of the equation readable

For input state `S_in`, the unchanged code performs these steps after the
kernel:

```python
rhs = b * (v - (k * exp(G)) @ S_in)
correction = torch.linalg.solve_triangular(L, rhs, upper=False)
output = (q * exp(G)) @ S_in + P @ correction
S_out = exp(G_last) * S_in + (k * final_decay).T @ correction
```

That is one triangular solve and four matrix multiplications. Combining them
could remove more launches, but M7o does not assume that the next-looking code
is the next bottleneck. Its post-change profile must first show where time
moved.

## Make fallback and forced debugging explicit

The public option `--kda-kernel-backend` has three values:

- `auto`, the serving default, uses the native kernel only when its contract
  fits;
- `torch` always runs the readable M7n pair builder; and
- `triton` requires the native contract and raises a clear error otherwise.

The deliberately narrow native contract is CUDA inference with FP32 working
tensors, `1 ≤ C ≤ 64`, and `K=128`. These are the actual inner shapes of the
tiny Kimi-K3 checkpoint. CPU, autograd, a wider chunk, or a future Kimi shape
falls back under `auto`. Decode remains on the one-token recurrence and never
calls this prefill kernel.

Padding is still an identity transition. Before cumulative decay or pair
formation, masked query, key, value, decay, and beta rows are replaced with
zero. Poisoning those input rows with `NaN` is a useful test: if output or
state becomes non-finite, the mask was applied too late.

## Numerical boundary

The focused native tests exercise complete and partial chunks at lengths 17,
64, and 65 using the channel-strided layout produced by causal convolution.
A two-row length-67 case gives the second row only 33 real tokens, poisons its
remaining inputs, and checks both finite output and unchanged padding state.
CPU tests cover automatic fallback and strict forced-Triton failure.

At the complete BF16 model boundary, the PyTorch and Triton pair builders run
the same fixed token inputs with separate fresh Kimi caches:

| prompt | max logit difference | max recurrent-state difference | ordered top 5 | argmax |
|---:|---:|---:|:---:|:---:|
| 32 | 0 | 2.33e-10 | equal | equal |
| 128 | 0 | 2.33e-10 | equal | equal |
| 512 | 0 | 1.16e-10 | equal | equal |
| 2,048 | 0 | 1.16e-10 | equal | equal |

The fixed input produces bit-identical BF16 logits, although the state is not
bit-identical. Other inputs may expose reduction-order differences, so tests
still use bounded tolerances and decision equality rather than assuming exact
arithmetic. The public `generate.py` path also completes a 75-token model
prefill—74 raw prompt tokens plus the start token—followed by recurrent decode.

## Paired promotion gate

The performance decision uses one loaded BF16 tiny Kimi-K3 model. For each
prompt length, two warmups precede five retained repetitions. PyTorch runs
first on even repetitions and Triton runs first on odd repetitions. Each
measurement includes a complete single-prompt generation call and one output
token; the isolated rule is measured separately.

The gate requires at least 10% lower full-model latency at 512 and 2,048 prompt
tokens, no more than 5% regression at 32 or 128, less than 32 MiB additional
peak allocation at 512, and equal first-token decisions. Results are retained
only from an idle GPU with stable telemetry.

The retained September 7, 2026 run used physical GPU 1, exposed as logical
`cuda:0` with `CUDA_VISIBLE_DEVICES=1`. No compute process was present before
the run, and every telemetry sample observed the same 1,800 MHz SM clock.

| prompt | PyTorch pairs | Triton pairs | lower latency | peak-memory change | first token |
|---:|---:|---:|---:|---:|:---:|
| 32 | 48.20 ms | 40.23 ms | 16.53% | 0 MiB | equal |
| 128 | 130.66 ms | 114.74 ms | 12.18% | −19.39 MiB | equal |
| 512 | 194.26 ms | 150.23 ms | 22.66% | −9.33 MiB | equal |
| 2,048 | 471.61 ms | 279.78 ms | 40.68% | 0 MiB | equal |

All conditions pass. The native path also improves short prompts instead of
merely staying under the regression limit. Lower memory at 128 and 512 comes
from removing pair-channel temporaries; both paths converge on the same peak
at 32 and 2,048 because another live tensor determines the allocator peak.

The isolated KDA rule shows that this is the intended local effect:

| tokens | PyTorch pairs | Triton pairs | speedup |
|---:|---:|---:|---:|
| 32 | 1.094 ms | 0.528 ms | 2.07× |
| 128 | 1.928 ms | 0.907 ms | 2.13× |
| 512 | 7.403 ms | 3.304 ms | 2.24× |
| 2,048 | 29.446 ms | 13.138 ms | 2.24× |

## Profile after fusion

A matched diagnostic runs the same profiler separately with the PyTorch and
Triton pair builders. At 2,048 tokens:

| diagnostic | PyTorch pairs | Triton pairs | reduction |
|---|---:|---:|---:|
| model CUDA-event interval | 659.64 ms | 406.28 ms | 38.41% |
| KDA interval | 519.87 ms | 265.80 ms | 48.87% |
| chunkwise-rule interval | 498.12 ms | 244.44 ms | 50.93% |
| isolated self CPU operator time | 35.36 ms | 14.83 ms | 58.06% |
| isolated self CUDA operator time | 18.52 ms | 4.50 ms | 75.71% |
| KDA share of model interval | 78.81% | 65.42% | −13.39 points |

The native pair kernel itself runs 32 times—once per C64 chunk—and accounts
for only 0.383 ms of isolated self CUDA time. The remaining 128 batched matrix
multiplications take 1.964 ms and 32 triangular solves take 1.124 ms. Together
they now own 68.7% of isolated CUDA time. This is the evidence for studying a
state/output tail kernel next; it is not evidence to enlarge M7o after the
current boundary already passes.

## Same-checkpoint calibration

The causal claim above comes only from the interleaved PyTorch/Triton run. A
separate-process calibration shows the remaining distance to the optimized
Transformers + FLA implementation using the same checkpoint, BF16, prompt
lengths, and one output token:

| prompt | Tinyserve M7o | Transformers + FLA | Tinyserve / reference |
|---:|---:|---:|---:|
| 32 | 44.44 ms | 119.82 ms | 0.37× |
| 128 | 117.01 ms | 122.76 ms | 0.95× |
| 512 | 166.11 ms | 124.02 ms | 1.34× |
| 2,048 | 284.25 ms | 142.73 ms | 1.99× |

M7n's corresponding long-prompt gap was 3.50×. The cross-run comparison is
context rather than a causal A/B, but it shows that native pair construction
closes a substantial part of the known prefill gap. Tinyserve still does not
depend on FLA. Ollama, llama.cpp, FreeToken, and ninfer remain excluded because
this tiny structural Kimi-K3 checkpoint is not a model boundary they share.

## Run and debug it

The **M7o: fused KDA pair construction** launch configuration uses
`examples/generate.py`, forces the native backend, and supplies a prompt that
crosses the 64-token boundary. Set breakpoints in this order:

1. `kda_rule()` — prefill selects chunkwise KDA; one-token decode selects the
   recurrence;
2. `chunkwise_kda()` — inspect the first width-64 chunk and partial final
   chunk;
3. `can_fuse_kda_pairs()` — inspect the narrow dispatch contract;
4. `fused_kda_pairs()` — inspect strides, output matrices, and launch grid;
5. the line after `solve_triangular()` — see that the existing solve and state
   path consume the native matrices unchanged.

Python can stop before and after a Triton launch but cannot single-step GPU
instructions. Force `--kda-kernel-backend torch` to inspect every pair
intermediate from the concrete example.

```bash
.venv/bin/python examples/generate.py \
  --model /home/scratch.kaix_coreai/models/kimi-k3 \
  --prompt "Explain fused KDA pair construction with a concrete trace." \
  --no-chat --max-new-tokens 2 \
  --kda-prefill-backend chunkwise --kda-chunk-size 64 \
  --kda-kernel-backend triton --verbose
```

The paired benchmark records raw repetitions, generated token ids, peak
allocation, GPU telemetry, source hashes, and the checkpoint fingerprint:

```bash
.venv/bin/python examples/bench_kda_kernel.py --device cuda:0 \
  --prompt-tokens 32 128 512 2048 \
  --micro-lengths 32 128 512 2048 --chunk-size 64 \
  --micro-iterations 5 --warmups 2 --repeats 5 \
  --output .tinyserve-bench/m7o/kda-pairs-ab.json
```

The post-change diagnostic profile is:

```bash
.venv/bin/python examples/profile_kimi_prefill.py \
  --device cuda:0 --prompt-tokens 32 128 512 2048 \
  --kda-chunk-size 64 --kda-kernel-backend triton \
  --operator-tokens 2048 --warmups 2 --repeats 5 \
  --output .tinyserve-bench/m7o/kimi-prefill-profile.json
```

The compact [M7o evidence](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7o-fused-kda-pairs-a6000-2026-09-07.json)
retains the gate, numerical boundary, matched profile, calibration, hashes,
and limitations. The one-visible-GPU suite passes 115 tests and skips nine
unchanged two-GPU TP, PP, EP, and CP cases.

## Takeaway

Fusion is most useful when its boundary follows a profile. M7o removes the
four channel-tile launch chains and their pair-channel temporaries, while
keeping the KDA equation, state lifecycle, comparison oracle, and unsupported
fallbacks explicit. The result is a 40.7% long-prompt latency reduction and a
smaller 1.99× same-checkpoint reference gap. Re-profiling then identifies a
specific next experiment: the solve and four state/output matrix
multiplications, not another speculative change to pair formation.
