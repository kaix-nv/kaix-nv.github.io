---
layout: post
math: true
title: "Building tinyserve M7d: Separate attention mechanism from cache policy"
date: 2026-10-05 08:07:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Separate attention execution from cache retention policy without confusing interfaces with qualified combinations."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m7d-pluggable-attention-cache.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M7c — Fuse work inside the decode graph]({% include tinyserve-post-url.html slug="building-tinyserve-m7c" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7c-fused-rmsnorm.md" %}) · Next: [M7e — Linear attention is a different kind of memory]({% include tinyserve-post-url.html slug="building-tinyserve-m7e" fallback="https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/m7e-native-linear-attention.md" %})

By M7c, tinyserve had three ways to read paged KV—PyTorch gather,
FlashInfer, and context-parallel distributed softmax—and two ways to retire a
block—return it immediately or preserve it as a warm prefix. They worked, but
they were not independent choices in the code.

The Qwen attention layer inspected FlashInfer wrappers and context-parallel
state. `PagedKVCache` contained both physical allocation and the warm-prefix
LRU. The scheduler even reached through the cache facade to inspect a block's
refcount. Adding a new attention algorithm or cache lifecycle would therefore
start as edits across unrelated modules.

M7d changes no attention equation, block layout, or scheduling decision. It
extracts the implementations that already exist behind two small interfaces:

- `AttentionBackend` decides **how a query reads visible KV**.
- `CachePolicy` decides **what happens to blocks during their lifecycle**.

<a href="/assets/tinyserve/m7d-backend-policy.svg"><img src="/assets/tinyserve/m7d-backend-policy.svg"
     alt="A concrete decode token at logical position 21 maps through block table [7,2] to physical slot 37. The scheduler reaches the block manager through either a no-cache or prefix-cache policy. The Qwen layer always writes KV before delegating the read to a Torch, FlashInfer, or context-parallel attention backend. The two choices are independent."></a>

This is dependency inversion in its smallest useful form. It is not a plugin
registry, dynamic loader, or configuration framework. The interfaces are
useful immediately because each has multiple real implementations and parity
tests today.

## The coupling before M7d

One paged attention method previously owned all of these decisions:

```python
cache.write(layer_idx, slots, k, v)

if cache.is_context:
    return context_parallel_attention(...)
if ctx.prefill and ctx.fi_prefill_wrapper is not None:
    return ctx.fi_prefill_wrapper.run(...)
if ctx.prefill:
    return torch_prefill(...)
if ctx.chunk_start is not None:
    return torch_shifted_causal_chunk(...)
if ctx.fi_wrapper is not None:
    return ctx.fi_wrapper.run(...)
return torch_gather_decode(...)
```

The branches are individually reasonable. The problem is ownership: a model
layer knows whether the engine installed FlashInfer, which wrapper was
planned, and whether KV is distributed across context ranks. A fourth backend
would add another engine resource, another forward-context field, and another
model branch.

The cache had the same shape of problem. `PagedKVCache` both implemented the
physical pool and decided that a registered zero-reference block should enter
a warm LRU. Turning prefix caching off replaced that behavior with immediate
freeing, but the two policies were expressed as repeated `if self.prefix`
branches rather than implementations of one lifecycle contract.

## One concrete decode token

Use block size 16 and a sequence whose block table is `[7, 2]`. Logical tokens
0–15 live in physical block 7. Tokens 16–31 live in physical block 2.

Suppose 21 tokens already have KV and the newest sampled token is ready for
its decode forward. Its logical position is 21, so the engine computes:

```text
logical block  = 21 // 16 = 1
physical block = block_table[1] = 2
offset         = 21 % 16 = 5
physical slot  = 2 * 16 + 5 = 37
```

The engine builds one `PagedForwardContext` containing slot 37, the padded
block-table tensor, context length 22, and the selected attention backend.
Every Qwen layer then performs the same two-stage operation:

```python
cache.write(layer_idx, ctx.slots, k, v)
return ctx.attention.attend(layer, q, k, v, ctx, attn_mask)
```

The write is invariant. New rotated K and V must reach slot 37 regardless of
how the current query reads its context. Only the second line varies.

This ordering matters. A fresh prefill attends directly to its in-flight K/V,
but a chunk and a decode step read the new token back alongside older pages.
Putting the write in a backend would let implementations accidentally disagree
about cache state. Keeping it in the model makes the shared contract visible.

## AttentionBackend: prepare once, attend in every layer

The interface has two operations:

```python
class AttentionBackend:
    def prepare(self, ctx):
        """Prepare shape-dependent metadata once before the model forward."""

    def attend(self, layer, q, k, v, ctx, attn_mask=None):
        raise NotImplementedError
```

`prepare()` belongs at the model-forward boundary, not inside each transformer
layer. FlashInfer converts block tables and context lengths into CSR metadata
once, then all layers reuse that plan. The pipeline runner prepares each sliced
row microbatch through the same method. CUDA-graph capture owns a backend with
fixed metadata buffers and replans it before replay.

The implementations retain the previous behavior:

| backend | fresh prefill | chunked prefill | decode |
|---|---|---|---|
| `TorchAttentionBackend` | in-flight causal SDPA, with the padded mask when needed | gather pages and apply the shifted causal mask | gather pages and mask the last-block tail |
| `FlashInferAttentionBackend` | ragged packed prefill from `indptr` boundaries | not selected; chunks retain the Torch path | planned paged kernel walks the block table without gathering |
| `ContextParallelAttentionBackend` | rank-local KV plus global MAX/SUM softmax | same | same |

The table is intentionally not a universal capability matrix. The engine
selects a backend only for modes it implements. For example, selecting
FlashInfer decode does not force a partial chunk through an unsupported
FlashInfer path.

After the extraction, `models/qwen3.py` contains no FlashInfer wrapper checks
and no context-parallel attention implementation. It knows only the invariant
projection, RoPE, KV write, and one backend call.

## CachePolicy: physical memory is not a retention decision

The physical mechanism remains in two classes:

- `BlockManager` owns the free list, refcounts, and peak allocation.
- `PagedKVCache` owns the tensor pool, slot mapping, layer reads/writes, and a
  facade used by the engine and scheduler.

`CachePolicy` receives lifecycle events through that facade:

```python
num_reclaimable
reclaim_one(manager)
keep_on_free(block)
on_zero_ref(block)
match_prefix(tokens)
adopt(sequence, matches, manager)
register_full_blocks(sequence)
count_live_matches(matches, manager)
```

M7d supplies two policies:

- `NoCachePolicy` returns every zero-reference block to the free list. Prefix
  matching, adoption, and registration are no-ops.
- `PrefixCachePolicy` owns the chained-hash index, warm LRU, resurrection,
  hit/eviction counters, and first-writer registration from M5b.

The allocator does not know which policy is installed:

```python
if not manager.free_list:
    policy.reclaim_one(manager)
return manager.alloc()
```

Likewise, freeing asks whether the policy wants custody of a zero-reference
block. The no-cache policy answers no; the prefix policy answers yes only for
a registered block and moves it into its warm LRU after the last owner leaves.

### The scheduler leak

Prefix matching creates one subtle capacity case. A matched warm block must be
counted because adoption pins reclaimable storage. A matched block already
held by another live sequence is a free ride: adoption only increments its
refcount and consumes no new physical block.

The scheduler previously identified those blocks by reading
`cache.manager.ref_count` directly. M7d replaces that one leak with:

```python
free_rides = cache.count_live_matches(matches)
```

The capacity equation and admission result do not change. The policy now owns
the meaning of a reusable live match, so a later policy can change its
lifecycle without another scheduler edit.

## What is independent—and what is not yet promised

The two seams are orthogonal:

- Switching Torch gather to FlashInfer changes how Q reads the same physical
  blocks; it does not change allocation, sharing, or eviction.
- Switching `NoCachePolicy` to `PrefixCachePolicy` changes block retention and
  reuse; it does not change the attention computation over the visible blocks.

M7d deliberately does **not** implement sliding-window or heavy-hitter
eviction. Those policies change which cached tokens are visible, not merely
whether an unused full block stays warm. The first selective policy should
teach us the smallest additional attention-view contract required; inventing
that contract before an implementation would hide assumptions about chunks,
partial blocks, and per-query selection.

The milestone also adds no runtime plugin discovery, entry points, dynamic
imports, or mixed-backend batching. A Python object passed through the forward
context is enough to establish the architecture.

## What had to remain true

The contract tests make both sides observable:

1. An injected recording backend sees that the model wrote finite K/V into the
   expected physical slots before calling `attend()`.
2. The same scheduler admission and release trace runs with explicit
   `NoCachePolicy` and `PrefixCachePolicy` objects; only the latter retains a
   matchable full block.
3. Existing prefix lifecycle tests still cover register, fork, warm,
   resurrect, clear, and LRU eviction.
4. Existing gather, ragged FlashInfer, eager/graph decode, and context-parallel
   tests still exercise their former numerical paths.
5. The complete suite passes 73 tests, including independent TP2, PP2, EP2,
   and CP2 workers.

## Cross-engine calibration

M7d is an ownership refactor, so its performance gate is neutrality. The fixed
Qwen3-8B BF16 suite used FlashInfer decode, CUDA graphs, prefix caching off, a
512-token prefill budget, two warmups, and five measured repetitions. The M7c
artifacts are the same-host baseline. Higher throughput and lower inter-token
latency are better.

| shape | batch | M7c throughput | M7d throughput | change | M7c ITL P50 | M7d ITL P50 | change |
|---|---:|---:|---:|---:|---:|---:|---:|
| p128/o128 | 1 | 38.163 | 38.171 tok/s | +0.0% | 26.12 | 26.10 ms | −0.1% |
| p128/o128 | 8 | 281.249 | 282.016 tok/s | +0.3% | 27.66 | 27.60 ms | −0.2% |
| p128/o128 | 32 | 858.763 | 852.253 tok/s | −0.8% | 32.82 | 33.11 ms | +0.9% |
| p2048/o32 | 1 | 24.247 | 24.212 tok/s | −0.1% | 26.55 | 26.68 ms | +0.5% |
| p2048/o32 | 8 | 45.285 | 45.523 tok/s | +0.5% | 92.89 | 92.42 ms | −0.5% |
| p2048/o32 | 32 | 50.019 | 50.447 tok/s | +0.9% | 149.42 | 147.96 ms | −1.0% |

Every throughput and ITL change is within 1%. Median TTFT was 0.2–1.4% lower
across the six rows, also noise-sized. There is no measured speedup to claim;
the result says the new Python ownership boundary did not move the existing
device paths.

Raw `nvidia-smi` peaks on physical GPU 1 were 36,053–36,113 MiB, including its
195 MiB idle display allocation. Subtracting that fixed baseline gives
35,858–35,918 MiB versus M7c's 35,873–35,933 MiB: unchanged. The candidate
suite reached 88 degrees C versus 87 for M7c, with sampled active clocks from
1,680 to 1,905 MHz.

One first pass of the final B32 condition overlapped a short unrelated GPU
allocation. Its contaminated repetition and memory sample were rejected; a
clean five-repetition rerun supplies the row above and the retained artifact
hash. Preserving that failed sample is more honest than selecting its plausible
median without checking telemetry.

External engines were not rerun because their revisions, checkpoint
fingerprints, software environment, and hardware did not change. Against the
frozen streaming-HTTP references, M7d reaches 91.7% of FreeToken/vLLM
throughput at B1, 89.9%/90.9% at B8, and 84.4%/84.3% at B32. As before,
tinyserve's timer is engine-internal, so these ratios are directional rather
than a leaderboard.

The structured [M7d evidence](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7d-pluggable-policy-a6000-2026-08-30.json)
retains exact medians and deltas, the corrected memory/thermal envelope,
source revision, test command, and SHA-256 hashes of all accepted raw artifacts.

## Run it

The **M7d: attention and cache policy** launch configuration uses the normal
`generate.py` entry point with two prompts sharing a full prefix. Useful
breakpoints are:

1. `Scheduler.schedule()` at `count_live_matches()`—policy-dependent capacity
   is queried without exposing refcounts.
2. `PagedKVCache.alloc_block()` and `free_block()`—the physical facade delegates
   reclamation and retention.
3. `Qwen3Attention._paged_attend()`—the common KV write occurs here.
4. `TorchAttentionBackend.attend()`—the selected backend reads the page table.

The same path runs directly with:

```bash
PYTHONPATH=$PWD .venv/bin/python examples/generate.py \
  --model /home/scratch.kaix_coreai/models/Qwen3-0.6B \
  --serve --backend gather --no-cuda-graphs --chunk-size 16 \
  --prompts \
  "A shared prefix long enough to fill one cache block. Name one color." \
  "A shared prefix long enough to fill one cache block. Name one city." \
  --max-new-tokens 4 --verbose
```

## Takeaway

An interface is valuable when it separates decisions that already vary. M7d
does not predict every future attention algorithm or cache policy. It extracts
three existing attention mechanisms and two existing cache lifecycles, then
pins their shared boundaries with parity tests.

For one decode token, the invariant story is now short: the engine maps a
logical position to a physical slot, the model writes K/V there, and the
selected backend reads the visible context. Separately, the selected policy
decides whether blocks are free, warm, shared, or reclaimable. That is enough
structure for the next experiment without reopening the whole serving stack.
M7e follows a different but related pressure: a native linear-attention model
introduces recurrent state that is neither an attention backend nor a KV-block
lifecycle. See [M7e]({% include tinyserve-post-url.html slug="building-tinyserve-m7e" fallback="https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/m7e-native-linear-attention.md" %}).
