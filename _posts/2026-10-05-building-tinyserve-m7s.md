---
layout: post
math: true
title: "Building tinyserve M7s: Replace the Python expert loop with indexed grouped GEMMs"
date: 2026-10-05 08:17:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Build an indexed expert schedule on the GPU and execute packed expert work without a Python loop."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m7s-native-indexed-moe.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M7r — Measure the MoE launch storm before writing a grouped kernel]({% include tinyserve-post-url.html slug="building-tinyserve-m7r" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7r-kimi-moe-profile.md" %}) · Next: [M7t — Profile the state-carrying KDA tail]({% include tinyserve-post-url.html slug="building-tinyserve-m7t" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7t-kda-tail-profile.md" %})

M7r found an awkward tradeoff in Kimi's routed experts. For at most 32 token
rows, gathering selected weights makes three large batched contractions fast,
but the temporary already reaches `64 MiB`. Longer prompts use little temporary
memory, but execute one active expert at a time from Python. A single block then
submits more than 1,200 CUDA kernels.

M7s keeps both useful properties. It sorts token-expert assignments on the GPU,
so rows for one expert become a matrix, then evaluates all groups with two
fixed Triton launches. It neither gathers `[N,K,I,D]` weights nor loops over
expert IDs in Python.

[![Four tokens are flattened into eight assignments, sorted by expert for two
grouped GEMM kernels, scattered back to route order, then weighted and reduced
per token.](/assets/tinyserve/m7s-indexed-moe.svg)](/assets/tinyserve/m7s-indexed-moe.svg)

## Start with one concrete route table

Use four tokens and two selected experts per token:

```text
assignment  token  slot  expert
    0         0     0      2
    1         0     1      5
    2         1     0      2
    3         1     1      7
    4         2     0      5
    5         2     1      7
    6         3     0      2
    7         3     1      5
```

Flattening preserves where every result must eventually return. Sorting those
assignment IDs by expert produces:

```text
expert 2 → [0, 2, 6]
expert 5 → [1, 4, 7]
expert 7 → [3, 5]
```

Now expert 2 can read token rows 0, 1, and 3 as one matrix and multiply them by
one set of expert-2 weights. After the down projection, assignment ID 6 tells
the kernel to scatter that result back to token 3, slot 0. Route order is not
lost; it is carried by the sorted assignment IDs.

The real checkpoint uses:

```text
N = token rows                 K = 16 routes per token
E = 64 experts                 A = N × K assignments
D = 256 routed channels        I = 256 intermediate channels
w1, w3 = [E,I,D]               w2 = [E,D,I]
```

For `N=128`, there are `A=2,048` rows to sort. Each assignment ID `a` maps back
to its token with integer division, `token = a // K`; its original route slot
is already encoded by the flattened output position.

## Build the schedule without returning to Python

The dispatcher constructs four small GPU tensors:

```python
flat_experts = indices.reshape(-1)
order = torch.argsort(flat_experts)
counts = torch.bincount(flat_experts, minlength=E)
expert_offsets = pad(cumsum(counts), (1, 0))

tiles = ceil_div(counts, BM)
tile_offsets = pad(cumsum(tiles), (1, 0))
```

`expert_offsets[e:e+2]` bounds expert `e`'s rows in `order`.
`tile_offsets` performs the same job for fixed `BM=16` row tiles. A Triton
program compares its task ID with the 64 tile offsets to find the owning
expert. Zero-count experts have duplicate offsets and receive no work.

The actual number of tiles remains on the GPU. The host launches a safe upper
bound:

$$
\sum_e \left\lceil \frac{\mathrm{count}_e}{BM} \right\rceil
\leq
\left\lceil \frac{A}{BM} \right\rceil + E.
$$

Programs beyond the GPU-computed total mask themselves out. This small amount
of excess launch geometry avoids `.item()`, a device-to-host synchronization,
and another Python scheduling dependency.

## Use two grouped GEMM kernels

The first kernel owns an expert row tile and 32 output channels. It gathers the
corresponding token rows, loads `w1[e]` and `w3[e]`, and computes:

$$
G = X_e W_{1,e}^{T}, \qquad U = X_e W_{3,e}^{T}.
$$

It applies Kimi's SiTU activation in FP32 after the same BF16 projection
rounding boundary as the PyTorch path, then stores one sorted activation matrix
`[A,I]`.

The second kernel multiplies those rows by `w2[e]`:

$$
Y_e = \operatorname{SiTU}(G,U) W_{2,e}^{T}.
$$

It scatters each row directly to its original flattened assignment position,
forming `[N,K,D]`. The existing PyTorch code still applies router weights and
sums over `K`. Keeping that last step unchanged isolates the optimization to
expert dispatch and makes the grouped path a useful oracle.

One program per assignment would look simpler, but it would turn every row into
a matrix-vector product and reread a `256 × 256` expert matrix for very little
work. Sorting creates matrix-matrix tiles that can reuse weights and feed
`tl.dot` with tensor-core-friendly shapes.

## Keep the backend contract narrow

`kimi_moe_backend` has three modes:

| mode | behavior |
|---|---|
| `torch` | Preserve M7r: selected weights through 32 rows, then the grouped Python oracle. |
| `auto` | Preserve the selected short path; use Triton only for the exact supported long shape; otherwise fall back to grouped PyTorch. |
| `triton` | Keep the selected short path; require the native contract when the memory guard selects the long path. |

The native contract is deliberately the tiny Kimi-K3 fixture: CUDA inference,
BF16 contiguous tensors, `E=64`, `K=16`, and `D=I=256`, with more than 32 token
rows. This is an educational kernel tied to measured shapes, not a generic MoE
runtime.

## Check the full model, not only the expert kernel

One loaded BF16 model ran the same exact prompt under `torch` and `auto`.
Two warmups preceded five retained repetitions, and order alternated within
each prompt length. The table reports median full-forward latency:

| prompt | M7r `torch` | M7s `auto` | reduction | selected path |
|---:|---:|---:|---:|---|
| 32 | 38.23 ms | 38.11 ms | 0.3% | selected in both |
| 128 | 102.09 ms | 38.87 ms | 61.9% | grouped → Triton |
| 512 | 107.54 ms | 41.89 ms | 61.0% | grouped → Triton |
| 2,048 | 158.62 ms | 95.68 ms | 39.7% | grouped → Triton |

The measured peak allocation is unchanged in every pair. In the isolated block,
temporary allocation remains `2.6`, `10.4`, and `41.4 MiB` at 128, 512, and
2,048 rows. It scales with the `[A,I]` activation and `[A,D]` result, not with
selected weight matrices.

Maximum full-model logit difference is `0.001953125`, below the `0.004` bound,
and all four ordered top-three lists match. The 32, 512, and 2,048 token
argmaxes also match exactly.

The 128-token fixture exposes an important numerical caveat. The grouped
reference gives tokens 81998 and 97685 the exact same BF16 maximum
`0.1552734375`; `argmax` chooses 81998 even though `topk` lists 97685 first.
Both the earlier selected-weight path and the native path choose 97685 after a
one-ULP-sized perturbation. Transformers+FLA also chooses 97685 for this fixed
prompt. M7s therefore records a tie-break exception instead of claiming exact
generated-token parity with an unstable grouped result.

The broader same-checkpoint calibration preserves M7q's outputs exactly:
Tinyserve and Transformers+FLA agree on 5/5, 2/5, 5/5, and 5/5 one-token
requests at 32, 128, 512, and 2,048 prompt tokens. The partial 128-token match
already existed in M7q and comes from this random checkpoint's near-tied logits.

| prompt | Tinyserve | Transformers+FLA | Tinyserve/reference |
|---:|---:|---:|---:|
| 32 | 45.89 ms | 129.73 ms | 0.354× |
| 128 | 41.44 ms | 127.26 ms | 0.326× |
| 512 | 50.06 ms | 133.10 ms | 0.376× |
| 2,048 | 104.97 ms | 141.32 ms | 0.743× |

Ollama, llama.cpp, FreeToken, and ninfer remain outside this calibration because
the tiny structural Kimi-K3 checkpoint is not a common supported boundary.
Transformers+FLA is a calibration reference, not an implementation dependency;
the M7s kernels are written from scratch in Tinyserve.

## Verify that the launch storm is gone

The same isolated-block profiler used by M7r now sees:

| rows | path | CUDA kernel events | native up/down launches | CUDA kernel time |
|---:|---|---:|---:|---:|
| 32 | selected | 61 | 0 / 0 | 0.756 ms |
| 128 | Triton | 66 | 1 / 1 | 0.220 ms |
| 512 | Triton | 83 | 1 / 1 | 0.393 ms |
| 2,048 | Triton | 83 | 1 / 1 | 1.046 ms |

Before M7s, the three long rows submitted 1,215, 1,265, and 1,352 kernels.
Sorting still uses several PyTorch CUDA operations, so “two kernels” describes
expert computation rather than the entire dispatcher. The decisive change is
that expert count no longer multiplies three projections and all their support
operators.

In the post-change full-model profile, routed MoE falls to about 35%, 34%, and
19% of the model interval at 128, 512, and 2,048 tokens. At 2,048 tokens KDA is
again dominant at 58%. That profile selects the next investigation; it does not
yet justify another implementation.

## Debug the implementation

The **M7s: native indexed Kimi MoE** launch configuration forces the native
backend on a prompt longer than 32 rows. Follow these boundaries:

1. `KimiSparseMoeBlock.forward()` — inspect `selected_bytes`,
   `native_supported`, and `last_expert_path`;
2. `fused_indexed_moe()` — inspect `order`, `expert_offsets`, and
   `tile_offsets` without calling `.item()`;
3. `_indexed_moe_up_kernel` — map `task` to `expert`, `sorted_row`,
   `assignment`, and `token` using the four-token example; and
4. `_indexed_moe_down_kernel` — watch the result scatter back by assignment ID
   before the unchanged route-weight reduction.

The compact [M7s evidence
manifest](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7s-native-indexed-moe-a6000-2026-09-10.json) retains the
protocol, paired medians, parity exception, post-profile counts, hashes, and
external calibration boundary.

## Takeaway

MoE sparsity says which weights are needed; it does not automatically schedule
those weights efficiently. M7s turns routing metadata into a GPU work schedule:
sort assignments, describe expert row ranges with offsets, run grouped GEMM
tiles, and scatter by the original assignment IDs.

The important result is causal and bounded. Removing the Python expert loop
cuts long-prompt full-model latency by 40–62% on this fixture without increasing
measured peak memory. The specialized contract, retained PyTorch oracle, and
documented BF16 tie keep that result honest.
