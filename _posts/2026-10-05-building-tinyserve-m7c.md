---
layout: post
math: true
title: "Building tinyserve M7c: Fuse work inside the decode graph"
date: 2026-10-05 08:06:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Fuse RMSNorm work inside an existing decode graph, then measure the graph and the complete serving path separately."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m7c-fused-rmsnorm.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M7b — Pack prompts without padding]({% include tinyserve-post-url.html slug="building-tinyserve-m7b" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7b-ragged-packed-prefill.md" %}) · Next: [M7d — Separate attention mechanism from cache policy]({% include tinyserve-post-url.html slug="building-tinyserve-m7d" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7d-pluggable-attention-cache.md" %})

M5c removed Python and framework launch overhead from decode by capturing one
CUDA graph per batch-size bucket. That was a large win, but a graph does not
turn many GPU kernels into one. It records the kernels and their dependencies,
then replays that same device graph every token.

M7a measured the consequence on Qwen3-8B: a batch-1 decode request spent 98%
of its wall time inside graph replay. M7c therefore opens that replay, counts
its kernels, and changes one repeated operation. The result is deliberately
small: keep the RMSNorm equation, replace its nine recorded GPU operations
with one fused kernel, and use it only in captured decode.

<a href="/assets/tinyserve/m7c-fused-rmsnorm.svg"><img src="/assets/tinyserve/m7c-fused-rmsnorm.svg"
     alt="One Qwen3-8B decode token shown with 145 RMSNorm calls. The PyTorch reference expands each norm into nine graph nodes, while CUDA graph capture records one FlashInfer fused RMSNorm node. Normalization nodes fall from 1305 to 145 and all graph kernels fall from 2353 to 1193; eager execution remains on the reference path."></a>

## CUDA graph replay is not kernel fusion

These optimizations remove different costs:

- **CUDA graph capture** records launches once so the CPU can submit the whole
  dependency graph with one replay call.
- **Kernel fusion** makes one device program perform work that several device
  programs previously performed.

Suppose a model operation expands into three kernels, `A -> B -> C`. Graph
capture changes three host launches into one `graph.replay()`, but the GPU
still schedules and executes `A`, then `B`, then `C`. Fusion changes the device
graph itself to one kernel, `ABC`.

M5c answered “how do we stop launching the same graph from Python?” M7c asks
the next question: “what did we put inside that graph?”

## Trace before changing code

The trace used Qwen3-8B BF16 on one RTX A6000, batch 1, context 128, and five
warmed CUDA-graph decode steps. PyTorch profiler adds synchronization to the
host path, so its wall time is not used as a latency result. CUDA kernel event
counts and durations inside the recorded replay remain the diagnostic signal;
ordinary unprofiled runs provide the latency result.

The baseline replay contained 2,353 kernel events per token. Projection
kernels consumed 84.6% of its device time:

| projection group | calls per layer | device time per token |
|---|---:|---:|
| Q and output projections | 2 | 10.75 ms |
| gate, up, and down projections | 3 | 9.57 ms |
| K and V projections | 2 | 1.08 ms |
| final vocabulary projection | once per token | 1.75 ms |

FlashInfer paged attention consumed only 0.70 ms per token. Replacing the
attention kernel first would optimize about 3% of device time and cannot close
the measured decode gap.

Projection fusion looked more promising, but a temporary A/B rejected it for
this milestone. Concatenating Q/K/V and gate/up weights reduced launches while
reading the same BF16 weight bytes. QKV fusion improved batch-1 latency by
less than 1%, gate/up by about 1%, and both together by 1.6%. That gain did not
justify changing checkpoint loading and every distributed projection layout.

The trace exposed a better educational target. Qwen3-8B runs four RMSNorms in
each of 36 layers—input, post-attention, Q, and K—plus one final norm:

```text
36 * 4 + 1 = 145 RMSNorm calls per decode token
```

Each call is small, but the readable PyTorch expression becomes nine graph
nodes. Those norms account for 1,305 of the replay's 2,353 kernels. The
individual kernels are not the largest kernels in the trace; their repeated
dependency chains are the opportunity.

## The equation does not change

For a row `x` and learned scale `w`, RMSNorm is

$$
\operatorname{RMSNorm}(x)
= \frac{x}{\sqrt{\operatorname{mean}(x^2) + \epsilon}} \odot w.
$$

The original implementation makes its numerical intent obvious:

```python
dtype = x.dtype
x = x.float()
x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
return (x * self.weight.float()).to(dtype)
```

That is the correctness oracle. On CUDA BF16/FP16, FlashInfer can perform the
reduction, normalization, scale, and output in one kernel. RMSNorm flattens
every leading dimension into rows because the
operation always reduces only the final dimension:

```python
if use_fused_kernel and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
    flat = x.reshape(-1, x.shape[-1])
    return flashinfer_rmsnorm(flat, weight, eps).view_as(x)
```

This covers both hidden states such as `[B, 1, 4096]` and per-head Q/K tensors
such as `[B, 1, H, 128]` without giving the kernel any model-specific shape.

## Fuse only what the graph captures

The first prototype enabled the fused kernel for every BF16 forward. Greedy
tokens still matched, but a rare hidden-state rounding difference accumulated
to a 0.17 vocabulary-logit delta. It pushed two of 455,808 elements in M7b's
ragged-versus-padded attention comparison beyond that test's existing
tolerance. Widening the tolerance would hide a boundary instead of explaining
it.

M7c therefore makes CUDA graph capture the optimization boundary:

```python
with _fused_rmsnorm_capture(model):
    # warm up each decode bucket
    # capture the model forward
```

The context manager enables fused RMSNorm on every dense or MoE model norm,
warms up and captures the decode graphs, then restores all flags. CUDA graph
replay keeps the recorded fused kernels. Ordinary prefill, eager decode, CPU,
and FP32 execute the original PyTorch expression.

That split is useful beyond this one kernel. An optimized graph may use a
numerically compatible implementation without silently replacing the oracle
used to check other layouts.

## What had to remain true

The checks follow the boundary introduced above:

1. Standalone hidden-state and per-head BF16 shapes compare fused output with
   the FP32 PyTorch formula.
2. FP32 rejects the fused dispatch and stays on the reference expression.
3. CUDA-graph and eager generation produce identical text at batch sizes 1,
   3, and 5, covering exact and padded capture buckets.
4. Prefix sharing, shrinking batches, and preemption still run through graph
   replay with the same output as eager execution.
5. After capture, every model norm has its fused flag restored to false.
6. The complete suite retains HF logits/text parity, KV-cache parity, packed
   and chunked prefill, paged KV, and independent TP/PP/EP/CP coverage.

The final suite result is recorded in the structured M7c evidence linked
below.

## Did the graph itself shrink?

Yes. The same five-step trace was repeated after the implementation:

| signal | M7b graph | M7c graph | change |
|---|---:|---:|---:|
| RMSNorm calls per token | 145 | 145 | same equation |
| RMSNorm graph nodes per token | 1,305 | 145 | −88.9% |
| all kernel events per token | 2,353 | 1,193 | −49.3% |
| summed kernel duration per token | 27.38 ms | 25.52 ms | −6.8% |

An unprofiled 40-step CUDA-event check measured 27.51 ms/token for the retained
M7b graph and 25.81 ms/token for M7c, a 6.2% improvement. Allocated memory was
unchanged at 17,392.5 MiB in both runs.

The event count is the causal evidence: the implementation predicted eight
fewer nodes for each of 145 norms, or 1,160 removed nodes. The trace observed
exactly `2,353 - 1,193 = 1,160` fewer kernels per token.

## Cross-engine calibration

Signed code revision `9082719` ran the fixed Qwen3-8B standard suite with
FlashInfer decode, CUDA graphs, prefix caching off, a 512-token prefill budget,
two warmups, and five measured repetitions. The final amend adds documentation
and module commentary; the measured executable path is unchanged. M7b's
retained same-host artifacts are the baseline. Higher throughput and lower
inter-token latency are better.

| shape | batch | M7b throughput | M7c throughput | change | M7b ITL P50 | M7c ITL P50 | improvement |
|---|---:|---:|---:|---:|---:|---:|---:|
| p128/o128 | 1 | 35.822 | 38.163 tok/s | +6.5% | 27.82 | 26.12 ms | 6.1% |
| p128/o128 | 8 | 262.455 | 281.249 tok/s | +7.2% | 29.68 | 27.66 ms | 6.8% |
| p128/o128 | 32 | 803.027 | 858.763 tok/s | +6.9% | 35.29 | 32.82 ms | 7.0% |
| p2048/o32 | 1 | 23.180 | 24.247 tok/s | +4.6% | 28.40 | 26.55 ms | 6.5% |
| p2048/o32 | 8 | 44.227 | 45.285 tok/s | +2.4% | 95.26 | 92.89 ms | 2.5% |
| p2048/o32 | 32 | 49.281 | 50.019 tok/s | +1.5% | 151.49 | 149.42 ms | 1.4% |

The workload mix explains the pattern. The first three rows generate 128
tokens, so decode dominates and exposes the full 6–7% change. The final rows
prefill 2,048 tokens but generate only 32. Their unchanged eager prefill
dilutes a decode-only optimization as batch size grows. TTFT moved only
0.3–1.8%, also consistent with a change that starts after the first token.

Memory did not regress: M7b's 512-budget artifacts peaked between 35,927 and
36,001 MiB; M7c peaked between 35,873 and 35,933 MiB. Both suites reached 87
degrees C. The decode-heavy B1 runs used the same sampled 1,905 MHz SM clock;
M7c B8 ran at a lower clock than M7b, while B32 differed by only 15 MHz.

External engines were not rerun because their revisions, model fingerprint,
environment, and hardware did not change. Against the frozen references, M7c
now reaches 91.7% of both FreeToken and vLLM throughput at B1, 89.7%/90.7% at
B8, and 85.1%/85.0% at B32. These remain directional comparisons: the external
engines use streaming HTTP while tinyserve's timing boundary is internal.

## Run it

The ordinary graph benchmark exercises the fused captured path:

```bash
PYTHONPATH=$PWD .venv/bin/python examples/bench_decode.py \
  --model /home/scratch.kaix_coreai/models/Qwen3-0.6B \
  --ctx 512 --batch 1 8 --pool-tokens 8192 \
  --backends flashinfer flashinfer-graph
```

To step through the implementation, use the **M7c: fused decode RMSNorm**
launch configuration. Break first in `_fused_rmsnorm_capture()` to see the
temporary semantic switch, then in `RMSNorm.forward()` while the graph buckets
warm up. The debugger cannot single-step a replayed CUDA graph; after capture,
`GraphRunner.run()` submits the already-recorded device work.

The structured [M7c evidence](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7c-fused-rmsnorm-a6000-2026-08-29.json)
retains trace counts, fixed-suite medians and deltas, memory and thermal
envelopes, source revision, and SHA-256 hashes of every raw artifact.

## Takeaway

A CUDA graph is a recorded program, not a fused kernel. It eliminates repeated
host submission while preserving every device node placed inside it. Once
M7a identified replay as the batch-1 bottleneck, tracing that program made the
next step mechanical: count repeated nodes, fuse one operation, and confirm
that the exact predicted nodes disappear.

M7c changes no scheduler policy, KV layout, attention algorithm, or checkpoint
format. It changes 145 copies of one equation from nine graph nodes to one,
keeps the readable implementation as an executable oracle, and closes another
measured part of the decode gap.
