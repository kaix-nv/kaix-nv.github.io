---
layout: post
math: true
title: "Building tinyserve M3: Paged KV cache"
date: 2026-08-17 13:01:26 -0700
categories: [tinyserve, llm-serving]
excerpt: "Replace contiguous KV reservations with block tables, demand allocation, and a paged-native attention kernel."
---

*Milestone 3 of [building an LLM inference engine from scratch](/series/tinyserve/).
Previous: [M2 — static batching]({% post_url 2026-08-17-building-tinyserve-m2 %}).*

Code: [`tinyserve` @ `179b4e0`](https://github.com/kaix-nv/tinyserve/tree/179b4e020f9dd5a45d990681724d61b337a4c3da).

## The problem

M2 exposes three costs around the KV cache, but they have different
causes:

1. **Reservation waste.** Every sequence must preallocate
   `prompt_len + max_new_tokens` slots up front. Ask for 128 new tokens,
   stop at EOS after 9 generated tokens, and the rest of the reservation
   was held — unusable by anyone — the whole time. Capacity is set by the
   *worst case*, not actual usage.
2. **A fixed rectangular batch.** The batch is one `[B, max_seq, ...]`
   tensor, so a finished row can't leave — it decodes garbage until the
   whole batch drains (two-thirds of M2's end-to-end output was zombie
   tokens), and ragged prompts burn pad slots.
3. **Attention traffic.** The padded path uses `repeat_interleave` to
   expand each layer's GQA keys and values before masked attention. That
   creates extra memory traffic and large temporary tensors.

Paging directly fixes reservation waste. It also lets the engine remove
a finished sequence without relocating another sequence's cache. Paging
alone does **not** make attention over those blocks fast; that requires a
kernel that reads through the block table without gathering or expanding
the cache. M3 deliberately combines both pieces: a paged allocator and a
paged-native FlashInfer decode kernel.

## The idea

Cut the physical pool into fixed-size **blocks** (16 tokens here, one
allocation at startup). Give each sequence a **block table** — a list
mapping its logical positions to physical blocks. Token at logical
position `p` lives at physical slot

```
block_table[p // block_size] * block_size + p % block_size
```

For a small example, use four-token blocks and this block table:

```text
block_size = 4
block_table = [7, 2]

logical position 5
  block-table index = 5 // 4 = 1
  physical block    = block_table[1] = 2
  offset in block   = 5 % 4 = 1
  physical slot     = 2 * 4 + 1 = 9
```

![Paged KV mapping from eight logical token positions through block table
entries 7 and 2 into non-contiguous physical KV blocks](/assets/tinyserve/m3-paged-kv-layout.svg)

*Figure 1. The model sees one contiguous logical sequence. The block table
places its first four tokens in physical block 7 and its next four in
physical block 2; attention follows the table rather than relying on
physical adjacency. The same block table is used at every model layer,
where each physical slot holds that layer's K and V for one token.*

That is a page-table walk, not coincidentally. Blocks are allocated **on
demand**. With M3's 16-token blocks, a sequence whose *total cached
length* is 9 tokens needs one block; a sequence with 17 tokens needs two.
Only the final block can be partially empty, so internal fragmentation is
at most 15 token slots per live sequence. Blocks are freed when a
sequence finishes.

Nothing about the pool is rectangular: sequences of wildly different
lengths coexist, each fragmenting the *logical* space freely while
the physical pool remains reusable at block granularity.

Block table entries carry a **refcount**. For M3 every block has exactly
one owner and the refcount is dead weight — it's here because prefix
caching (M5) shares prompt blocks between requests, and retrofitting
ownership tracking is much harder than carrying it from day one.

The catch: attention now has to *read* K/V through the block table. SDPA
can't; you either gather blocks back into contiguous tensors (a copy per
step — correct, slow) or use a kernel that walks the table in place.
This is the milestone where the engine grows its first real kernel
dependency: **FlashInfer**'s `BatchDecodeWithPagedKVCacheWrapper`, with
our own gather implementation kept as the always-available correctness
reference. Prefill needs neither: a prompt's K/V is computed in-flight,
so we attend on it directly and only *write* it into blocks.

Paging makes another engine policy cheap: **shrink the batch**. M3 removes
a row when it emits EOS, frees its blocks, and runs later steps with a
smaller B. The allocator makes that removal cheap; the decision to remove
the row belongs to `generate_paged`. Admitting *new* work into the freed
capacity mid-flight is continuous batching — M4 adds that scheduler.

M3 also chooses to prefill prompts one at a time with zero padding, so
each writes its K/V compactly into its own blocks. Serialized prefill is
an implementation choice for this milestone, not a requirement of
paging.

## The build

New: [`paged.py`](https://github.com/kaix-nv/tinyserve/blob/179b4e020f9dd5a45d990681724d61b337a4c3da/tinyserve/paged.py) — `BlockManager`
(alloc/free/refcount over a LIFO free list, ~40 lines), `PagedKVCache`
(the pool: `[layers, blocks, 2, block_size, kv_heads, head_dim]`, plus
`write()`), `Sequence` (tokens + block table; grows into M4's scheduler
unit), and `PagedForwardContext` — the little struct of slots/block
tables/context lengths the engine hands the model each forward. That
context is deliberately the seam where other attention backends will
plug in later (M6). Modified: attention gets `_paged_attend` (write to
slots, then in-flight / FlashInfer / gather); the engine gets
`generate_paged`.

Bugs, in order of instructiveness:

- **The slot-scatter corruption.** `write()` originally flat-viewed the
  pool as `(-1, 2, H, D)` and indexed rows by slot id. But the pool's
  k/v axis sits *between* block and offset — `[blocks, 2, block_size, ...]`
  — so that flat view interleaves k and v pages, and every slot with
  offset ≠ 0 landed somewhere wrong. The insidious part: **all tests that
  only exercised prefill passed**, because prefill attends on in-flight
  K/V and merely writes the (corrupted) cache. The first decode step read
  it back and produced logits off by 19.4. Lesson: in a paged system,
  writes and reads go through different code paths — a test isn't testing
  the write until it *reads through the page table*.
- **The kernel-selection trap.** PyTorch 2.5+ has `enable_gqa=True`,
  which lets SDPA's fused kernels index KV heads in place — exactly what
  M2's postmortem asked for. In the original A6000 microbenchmark,
  `enable_gqa` measured 0.45 ms versus 2.1 ms for explicit expansion.
  Adding a dense `attn_mask` selected the math path and measured 6.7 ms.
  Those numbers are a historical snapshot; SDPA dispatch depends on the
  PyTorch, CUDA, GPU, shapes, dtype, and arguments. Policy is therefore
  explicit in attention: no mask → `enable_gqa`; mask → expand KV. The
  general lesson is to inspect or pin SDPA's backend and benchmark the
  exact call shape. See PyTorch's
  [`scaled_dot_product_attention` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

## The proof

[`tests/test_paged.py`](https://github.com/kaix-nv/tinyserve/blob/179b4e020f9dd5a45d990681724d61b337a4c3da/tests/test_paged.py):

1. `BlockManager` unit tests: exhaustion raises, freed blocks are reused
   LIFO, refcounted double-owner blocks survive one free, double-free
   asserts.
2. **fp32 output parity**: `generate_paged` (gather) and
   `generate_batch` (M2 padded) return identical decoded strings on three
   ragged prompts. This is exact at the returned-string level; the test
   does not compare logits or token IDs directly.
3. **Leak check**: after generation every block is back in the free list.
4. **Kernel smoke check**: FlashInfer vs gather on the same paged layout,
   bf16 — the first mismatch must occur after at least 80% of the shorter
   decoded string. This is deliberately weaker than token parity because
   different kernels may flip a token when the top logits are nearly
   tied; it is not proof of a fixed absolute prefix length.

All pass (13 total).

## The payoff

Decode step cost, three backends, same engine
([`examples/bench_decode.py`](https://github.com/kaix-nv/tinyserve/blob/179b4e020f9dd5a45d990681724d61b337a4c3da/examples/bench_decode.py)). The table is
the retained M3 snapshot from one RTX A6000. The raw output and complete
software environment were not retained, so a new run should be expected
to differ:

**ctx 512** (ms/step | tok/s):

| B | padded (M2) | paged-gather | paged-FlashInfer |
|---:|---:|---:|---:|
| 1 | 24.8 \| 40 | 28.7 \| 35 | 23.8 \| 42 |
| 32 | 24.8 \| 1291 | 37.1 \| 863 | 26.1 \| 1226 |
| 128 | 73.2 \| 1750 | 122.7 \| 1043 | **26.2 \| 4880** |

**ctx 2048:**

| B | padded (M2) | paged-gather | paged-FlashInfer |
|---:|---:|---:|---:|
| 1 | 24.2 \| 41 | 26.7 \| 37 | 24.1 \| 41 |
| 32 | 70.6 \| 453 | 117.6 \| 272 | **27.3 \| 1171** |
| 128 | OOM | 436.6 \| 293 | **56.3 \| 2273** |

Three stories in there:

- **FlashInfer stays nearly flat through most of the table.** Under a
  one-read KV traffic model, 29.8 GB divided by 56.3 ms is **529 GB/s of
  effective KV bandwidth**, or 69% of the A6000's advertised peak. This
  is a useful roofline estimate, not a profiler measurement: it does not
  prove that every byte was read exactly once or that the kernel was
  strictly bandwidth-bound. FlashInfer's
  [`BatchDecodeWithPagedKVCacheWrapper` documentation](https://docs.flashinfer.ai/api/attention.html)
  defines the paged interface, not its exact HBM transaction count.
- **The gather path shows the cost of leaving the paged layout.** Same
  block tables and pool, but gathering into contiguous tensors, expanding
  GQA heads, and running masked SDPA together cost 3–8× at scale. This is
  an end-to-end backend comparison; it does not isolate the copy alone.
- **Padded OOMs where paged doesn't.** At B=128/ctx 2048 the contiguous
  benchmark OOMed while the paged FlashInfer path completed at 2273
  tok/s. No peak-allocation trace was retained, so the result should not
  be attributed solely to the cache or one expanded-K/V temporary.

A second historical snapshot recorded memory and useful throughput on 32
ragged prompts with `max_new_tokens=128`, most ending early at EOS. The
original prompt list, measurement harness, and raw output are not present
in this commit:

|  | tokens held at peak | KV memory |
|---|---:|---:|
| contiguous reservation (M2) | 4800 | 0.54 GB |
| paged peak, demand + free-on-finish | 1152 | 0.13 GB |

The recorded peak is **4.2× less KV memory for the same workload** — the
combination of demand allocation and freeing finished rows. An honest
wash to report: end-to-end
*useful* throughput on that workload is 362 tok/s vs M2's 406. Freeing
finished rows helps, but M3 prefills prompts one at a time (32 small
forwards instead of one padded batch) and pays per-step Python for slot
computation and kernel planning. At 0.6B scale, overhead giveth back.
The prefill serialization is M5's chunked-prefill problem; the Python
tax is M5's CUDA-graphs problem. Paging's *throughput* value at this
scale is capacity (the OOM row), not speed — its speed value arrives
with the kernel at large B×ctx.

## Reproducing

```bash
uv pip install -e '.[kernels]'      # flashinfer; engine falls back to gather without it
pytest tests/ -q                    # 13 tests
python examples/bench_decode.py --ctx 512 2048 --batch 1 32 128
```

These commands reproduce the correctness suite and rerun the decode
sweep; they do not reproduce the historical memory/useful-throughput
snapshot exactly. Record the model revision, GPU, PyTorch, CUDA,
FlashInfer, command output, and prompt list for any new comparison.

Only the protocol survives for the historical 0.45/2.1/6.7 ms SDPA
matrix: B=32, Hq=16, Hkv=8, D=128, L=2048, one decode query, 50 warmed
runs, with `torch.nn.attention.sdpa_kernel` used to pin backends. No
runnable script or raw output is retained in this commit.
