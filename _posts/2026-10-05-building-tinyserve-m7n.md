---
layout: post
math: true
title: "Building tinyserve M7n: Tune the KDA chunk before fusing the kernel"
date: 2026-10-05 08:12:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Increase the KDA chunk to 64 only after numerical, whole-model, and paired performance checks."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m7n-kda-chunk64.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M7m — Count the launches before writing the next KDA kernel]({% include tinyserve-post-url.html slug="building-tinyserve-m7m" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7m-kimi-prefill-profile.md" %}) · Next: [M7o — Fuse KDA's pair construction]({% include tinyserve-post-url.html slug="building-tinyserve-m7o" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7o-fused-kda-pairs.md" %})

M7m found an unexpectedly useful one-line experiment. Kimi Delta Attention
(KDA) spent most of long-prompt prefill inside a Python loop over 32-token
chunks, while the triangular solve itself consumed only 3.1% of the isolated
CUDA operator time. A 64-token chunk looked faster, but a promising sweep is
not yet a safe default.

M7n promotes that geometry only after checking the complete boundary: the
equation, poisoned padding, recurrent state, model logits, token decisions,
short-prompt latency, long-prompt latency, and memory. It then profiles the
promoted path so the next milestone follows evidence rather than the newest
interesting-looking operator.

[![A 2,048-token scheduler prefill remains one model forward. Inside each KDA
layer, changing the inner chunk from 32 to 64 tokens halves Python loop trips
from 64 to 32 and doubles the bounded pair matrix width. State crosses only
chunk boundaries. The paired measurement shows the larger chunk is neutral at
32 tokens and increasingly faster at 128, 512, and 2,048 tokens.](/assets/tinyserve/m7n-kda-chunk64.svg)](/assets/tinyserve/m7n-kda-chunk64.svg)

## Two chunks with different jobs

Tinyserve now has two unrelated quantities called a chunk:

- the **scheduler prefill budget** decides how many queued prompt tokens enter
  one model forward, so decoders get another scheduling turn; and
- the **KDA inner chunk** divides the tokens already inside that forward so the
  linear-attention equation uses bounded intermediate matrices.

For the retained 2,048-token measurement, the scheduler deliberately admits
the whole prompt in one forward. KDA then partitions that tensor internally:

```text
scheduler action: [ token 0 ................................ token 2047 ]

C=32: S0 -> [32 tokens] -> S1 -> [32 tokens] -> ... -> S64
C=64: S0 -> [64 tokens] --------> S1 -> [64 tokens] -> ... -> S32
```

`S` is the fixed recurrent KDA state. Every token inside a bracket is solved
together; only the final state crosses into the next bracket. A 130-token
input with `C=64` becomes `[64, 64, 2]`. The final two-token chunk uses the
same equation—there is no semantic padding to 64.

Changing the KDA chunk therefore cannot improve scheduling fairness. It only
changes how one KDA layer evaluates an already selected batch.

## Why bigger can be faster, and why it cannot grow forever

Let the prompt length be `T` and the inner chunk width be `C`. There are
approximately `T/C` Python loop iterations. Each iteration constructs causal
pair matrices with `C × C` token positions, so their total element count grows
roughly as:

$$
\frac{T}{C} C^2 = TC.
$$

Increasing `C` creates a tradeoff:

- fewer loop iterations, launches, masks, and state hand-offs; but
- larger pair matrices and more within-chunk arithmetic and temporary storage.

At `T=2048`, moving from 32 to 64 halves the loop from 64 iterations to 32,
but doubles the total pair entries from 65,536 to 131,072 per channel tile.
M7m's `C=128` control was already slower than `C=64`: saved dispatch no longer
paid for the additional quadratic work. Sixty-four is a measured crossover,
not a universal constant inferred from the equation.

## The code change is intentionally boring

The KDA algebra is unchanged. M7n changes the default from 32 to 64 at every
public boundary:

```python
def chunkwise_kda(..., chunk_size: int = 64): ...
def kda_rule(..., chunk_size: int = 64): ...

class LLM:
    def __init__(..., kda_chunk_size: int = 64): ...
```

The model setter, attention-layer field, `generate.py`, and Kimi profiling CLI
use the same value. A test inspects these boundaries so a future edit cannot
silently leave the engine and direct model API with different defaults.

Decode does **not** process a 64-token block. `kda_rule()` retains the recurrent
update for one-token decode and short inputs; the default matters only after
the existing prefill crossover selects the chunkwise path. The old C32 path
also remains selectable as the comparison oracle.

## Check the numerical boundary before timing it

The small equation test compares chunkwise output and final state with the
token recurrence across chunk sizes `1, 8, 16, 32, 64`. Its 17-token input
exercises partial final chunks. A separate poisoned-padding test verifies that
masked rows neither emit useful output nor mutate state.

The complete tiny Kimi-K3 model then runs the same fixed BF16 input with C32
and C64. These are maximum absolute differences over the vocabulary logits
and all twelve KDA recurrent states:

| prompt | max logit difference | max state difference | ordered top 5 | argmax |
|---:|---:|---:|:---:|:---:|
| 32 | 0 | 0 | equal | equal |
| 128 | 0 | 2.91e-9 | equal | equal |
| 512 | 0 | 2.74e-9 | equal | equal |
| 2,048 | 9.77e-4 | 3.36e-5 | equal | equal |

Different chunk boundaries change floating-point association, so bit identity
is not required at 2,048 tokens. The leading decisions remain identical. A
CPU FP32 integration test also keeps a 33-token full-model C32/C64 comparison
in the normal suite.

## Paired promotion gate

Both candidates run on one loaded model. Each exact-length prompt has two
warmups and five retained repetitions; order is C32→C64 on even repetitions
and reversed on odd repetitions. Times include Tinyserve's complete
single-prompt generation call for one output token. The table uses medians:

| prompt | C32 | C64 | C64 lower latency | peak-memory change | first token |
|---:|---:|---:|---:|---:|:---:|
| 32 | 52.70 ms | 52.90 ms | −0.37% | 0 MiB | equal |
| 128 | 161.51 ms | 135.95 ms | 15.82% | +15.92 MiB | equal |
| 512 | 328.41 ms | 239.23 ms | 27.16% | +9.33 MiB | equal |
| 2,048 | 976.35 ms | 557.35 ms | 42.92% | 0 MiB | equal |

This passes the predeclared gate: at least 10% lower latency at 512 and 2,048
tokens, no regression beyond 5% at 32 or 128, less than 32 MiB additional peak
allocation at 512, and equal paired token decisions. The 32-token difference
is noise-sized; C64 becomes useful as the number of avoided loop trips grows.

## Profile again after promotion

The same diagnostic profile used in M7m shows what the geometry change removed
at 2,048 tokens. These are separate profiled runs, not the paired promotion
measurement:

| diagnostic | C32 (M7m) | C64 (M7n) | change |
|---|---:|---:|---:|
| KDA loop trips per layer | 64 | 32 | −50.0% |
| model CUDA-event interval | 1,148.3 ms | 816.7 ms | −28.9% |
| KDA interval | 999.2 ms | 644.2 ms | −35.5% |
| chunkwise-rule interval | 976.0 ms | 618.5 ms | −36.6% |
| isolated self CPU operator time | 70.7 ms | 43.6 ms | −38.3% |
| isolated self CUDA operator time | 16.8 ms | 18.6 ms | +10.7% |

The larger matrices do slightly more useful CUDA work while substantially
reducing host-side operator work and stream gaps. Call counts expose the same
tradeoff: batched matrix multiplies fall from 256 to 128, reductions from 512
to 256, and triangular solves from 64 to 32. The larger triangular solves now
total 1.15 ms, still only 6.2% of self CUDA operator time.

KDA still owns 78.9% of the long-prompt model interval, and the chunkwise rule
owns 96.0% of KDA. The next target therefore remains the fragmented multiply,
reduction, batched-matmul, mask, and state/output chain—not a standalone
triangular solver.

## Same-checkpoint calibration

The paired C32/C64 run establishes causality. A separate-process calibration
only shows the remaining distance to the optimized Transformers + FLA path:

| prompt | Tinyserve C64 | Transformers + FLA | Tinyserve / reference |
|---:|---:|---:|---:|
| 32 | 52.90 ms | 142.01 ms | 0.37× |
| 128 | 135.95 ms | 146.38 ms | 0.93× |
| 512 | 239.23 ms | 145.65 ms | 1.64× |
| 2,048 | 557.35 ms | 159.41 ms | 3.50× |

The reference uses the same checkpoint, BF16 dtype, prompt lengths, and one
generated token, but it is not an interleaved A/B and uses an external FLA
kernel. Ollama, llama.cpp, FreeToken, and ninfer are not included because this
tiny structural Kimi-K3 checkpoint is not a supported common model boundary
for those engines. Tinyserve itself still contains no FLA dependency.

## Debug the promoted path

The **M7n: KDA 64-token inner chunks** launch configuration runs
`examples/generate.py` with a prompt long enough to cross a 64-token boundary.
Use these breakpoints:

1. `kda_rule()` — confirm prefill selects `chunkwise` while decode selects
   `recurrent`;
2. `chunkwise_kda()` — inspect `start`, `end`, and the partial final chunk;
3. the channel-tile loop — distinguish the 32-channel tile from the 64-token
   chunk; and
4. the final state assignment — see the only value carried between chunks.

For the retained paired measurement:

```bash
.venv/bin/python examples/bench_kda.py --device cuda:0 \
  --prompt-tokens 32 128 512 2048 --chunk-sizes 32 64 \
  --micro-lengths 32 128 --micro-iterations 10 \
  --kda-kernel-backend torch \
  --warmups 2 --repeats 5 \
  --output .tinyserve-bench/m7n/kda-chunk64-ab.json
```

The post-promotion profile is:

```bash
.venv/bin/python examples/profile_kimi_prefill.py \
  --device cuda:0 --prompt-tokens 32 128 512 2048 \
  --kda-chunk-size 64 --kda-kernel-backend torch \
  --operator-tokens 2048 \
  --warmups 2 --repeats 5 \
  --output .tinyserve-bench/m7n/kimi-prefill-profile-c64.json
```

The compact [M7n evidence](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7n-kda-chunk64-a6000-2026-09-04.json)
retains the gate, numerical boundary, profile, hashes, and limitations.

## Takeaway

Chunk size is an engineering balance, not a decorative hyperparameter. C64
doubles within-chunk pair work but halves Python iterations; on this A6000 and
tiny Kimi shape, the avoided dispatch wins decisively for long prompts without
changing decisions. Re-profiling is equally important: it shows that geometry
harvested the cheap win, while the remaining 3.50× long-prompt reference gap
still calls for a native fused KDA prefill kernel.
