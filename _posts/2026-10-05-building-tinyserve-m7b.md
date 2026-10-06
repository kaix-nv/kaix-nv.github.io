---
layout: post
math: true
title: "Building tinyserve M7b: Pack prompts without padding"
date: 2026-10-05 08:05:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Pack real prompt tokens without padding while preserving positions, request boundaries, and paged KV ownership."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m7b-ragged-packed-prefill.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M7a — Before optimizing, name the phase]({% include tinyserve-post-url.html slug="building-tinyserve-m7a" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7-phase-profiling.md" %}) · Next: [M7c — Fuse work inside the decode graph]({% include tinyserve-post-url.html slug="building-tinyserve-m7c" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m7c-fused-rmsnorm.md" %})

M5a taught tinyserve to pack several fresh prompts into one forward. That
removed repeated per-forward overhead, but it reused M2's rectangular batch:
every prompt was left-padded to the longest row, attention received a dense
mask, and grouped-query attention copied each KV head across its query-head
group.

That was a good first implementation because its correctness was easy to
inherit from static batching. M7a finally measured its cost. In the exact
Qwen3-8B `p2048/o32/b32` profile, eight packed-prefill forwards consumed
12.58 of 14.23 seconds. Packing the input and scattering K/V consumed only
27 ms and 54 ms respectively. The bottleneck was not *deciding* which prompts
to pack. It was how attention represented the packed prompts.

M7b keeps the scheduler policy and changes that representation: concatenate
real tokens, describe prompt boundaries with an `indptr` array, and let a
fused variable-length attention kernel enforce causality inside each prompt.

<a href="/assets/tinyserve/m7b-ragged-packed-prefill.svg"><img src="/assets/tinyserve/m7b-ragged-packed-prefill.svg"
     alt="A concrete three-token and five-token prompt shown first as a padded rectangular batch, then as one flat ragged token row whose indptr boundaries produce two independent causal attention triangles"></a>

## Packing and padding are different ideas

These terms are easy to collapse into one vague idea, so separate them:

- **Packing** says several requests share one model forward. It reduces the
  fixed cost paid per forward.
- **Padding** is one way to make the packed requests fit a rectangular tensor.
  It is convenient, but it computes positions that do not belong to a prompt.
- **Ragged attention** keeps packing while replacing padding with explicit
  sequence boundaries understood by the kernel.

M7b does not undo the packing lesson from M5a. It finishes it.

## A concrete batch

Suppose the scheduler selects two complete prompts:

```text
A = [a0, a1, a2]              length 3
B = [b0, b1, b2, b3, b4]      length 5
```

The old temporary layout was:

```text
input_ids = [[PAD, PAD, a0, a1, a2],
             [ b0,  b1, b2, b3, b4]]
positions = [[  0,   0,  0,  1,  2],
             [  0,   1,  2,  3,  4]]
```

An explicit boolean mask had shape `[2, 1, 5, 5]`. It hid A's padding keys
and enforced a causal triangle in both rows. Padding queries still needed a
valid diagonal, and their K/V writes were redirected to the cache's scratch
block.

The new layout is:

```text
input_ids = [[a0, a1, a2, b0, b1, b2, b3, b4]]
positions = [[ 0,  1,  2,  0,  1,  2,  3,  4]]
indptr    = [0, 3, 8]
```

`indptr[i]:indptr[i+1]` is request `i`. Here `[0:3]` belongs to A and
`[3:8]` belongs to B. FlashInfer uses those boundaries to apply two causal
triangles: A cannot see B, B cannot see A, and neither prompt has pad tokens.

The positions must also restart at zero. RoPE encodes a token's logical
position inside its own sequence; flat buffer index 3 is B's position 0, not
position 3.

## The physical KV cache does not become ragged

The flat row is only the input layout for this forward. Each sequence already
owns paged-KV blocks allocated by the scheduler. Tinyserve concatenates the
corresponding physical slot IDs in exactly the same token order:

```text
slots = [A.slot(0), A.slot(1), A.slot(2),
         B.slot(0), B.slot(1), B.slot(2), B.slot(3), B.slot(4)]
```

Every attention layer computes Q, K, and V, applies RoPE, then scatters K/V
through this flat `slots` array. Attention itself reads the just-computed K/V
directly because these are fresh complete prompts. The pool write is still
required: the next decode step needs those values through each sequence's page
table.

This separation matters:

```text
temporary model layout: flat real tokens + indptr
persistent cache layout: per-sequence logical positions -> physical blocks
```

Changing the first does not change block allocation, prefix hashes, cache
ownership, or later paged decode.

## One plan, every layer

`_paged_prefill_batch()` remains the scheduler-facing helper. It chooses the
ragged path only when all of these are true:

1. the batch contains more than one fresh complete prompt;
2. FlashInfer is the selected backend; and
3. the engine is single-GPU.

The new `_paged_prefill_batch_ragged()` performs four pieces of setup inside
the existing `prefill_pack` region:

```python
indptr = [0, 3, 8]
input_ids = [[a0, a1, a2, b0, b1, b2, b3, b4]]
positions = [[0, 1, 2, 0, 1, 2, 3, 4]]
final_indices = [2, 7]

prefill_wrapper.plan(
    qo_indptr=indptr,
    kv_indptr=indptr,
    num_qo_heads=...,
    num_kv_heads=...,
    head_dim_qk=...,
    causal=True,
)
```

Query and KV lengths are equal because every token in each fresh prompt is
both a query and an in-flight key/value. `plan()` builds auxiliary metadata
once. Every transformer layer then reuses it:

```python
o = prefill_wrapper.run(
    q.reshape(total_tokens, num_q_heads, head_dim),
    k.reshape(total_tokens, num_kv_heads, head_dim),
    v.reshape(total_tokens, num_kv_heads, head_dim),
)
```

FlashInfer handles grouped-query attention inside the kernel, so K/V heads are
not copied with `repeat_interleave`. Decode and prefill are sequential, which
also lets their wrappers share one 128 MiB workspace.

## Returning one logit row per prompt

Flattening exposes one subtle bug opportunity. `last_token_only=True` used to
select the last tensor column. On the flat row that would select only `b4` and
silently drop A.

The forward context therefore carries `prefill_output_indices = [2, 7]`.
After the final norm, the model gathers those hidden states before the large
vocabulary projection:

```text
flat hidden [8, hidden] --index [2, 7]--> [2, hidden]
                                      --lm_head--> [2, vocab]
```

Selecting before `lm_head` is important. Projecting all eight prompt tokens
and discarding six would waste both compute and a large logits allocation.

## The reference path stays executable

The old padded implementation was not deleted. `backend="gather"`, FP32
tests, and distributed serving continue to use it. That gives M7b an unusually
strong oracle: the optimized and reference layouts live beside each other and
can receive the same model, token lists, page allocator, and output check.

The first M7b slice deliberately does not fuse:

- a single whole prompt, which already uses maskless causal SDPA;
- a partial chunk, which must attend to an earlier cached prefix;
- a prefix-cache hit that begins in the middle of a sequence; or
- TP, PP, EP, and CP execution.

Those paths retain their existing behavior. This is an educational milestone,
so a narrow dispatch condition is preferable to claiming that one kernel
already covers every prefill mode.

## Correctness gate

The existing FP32 test still proves that padded packing returns the same final
logits as prefill performed one prompt at a time. M7b adds a BF16 kernel test
with unequal prompt lengths 19, 47, and 33:

1. run the padded reference into one set of physical blocks;
2. release and clear those blocks;
3. run the ragged FlashInfer path into a fresh set; and
4. compare all three final vocabulary-logit rows; then
5. feed one fixed next token per sequence through reference paged decode and
   compare again, forcing the check to read K/V back from physical pages.

The test uses numerical logit tolerance rather than exact argmax equality.
Different BF16 attention kernels can reorder two nearly tied top logits even
when the layouts agree numerically; token identity is not a stable assertion
at such a tie. The second comparison proves that flat slot order did not merely
produce correct in-flight attention—it also wrote each prompt into the right
physical blocks. The complete chunked-prefill file verifies whole-versus-
chunked cache parity, packed-versus-individual parity, interleaved decode, and
end-to-end output equality. The final repository run passed all 67 tests,
including the independent TP, PP, EP, and CP lanes, in 358.16 seconds.

## Did the named phase move?

Yes. The focused profile repeats M7a's exact Qwen3-8B shape: 2,048 prompt
tokens, 32 output tokens, batch 32, an 8,192-token prefill budget, two warmups,
and three measured runs on one RTX A6000.

| signal | M7a padded | M7b ragged | change |
|---|---:|---:|---:|
| prefill-forward device total | 12,578.02 ms | 11,014.55 ms | −12.4% |
| cache-write device subtotal | 53.73 ms | 54.44 ms | unchanged |
| decode-forward device total | 1,606.54 ms | 1,603.89 ms | unchanged |
| complete call wall time | 14,232.85 ms | 12,660.63 ms | −11.0% |
| output throughput | 71.25 tok/s | 80.06 tok/s | +12.4% |
| TTFT P50 | 8,011.12 ms | 7,033.05 ms | −12.2% |

The expected phase moved while the two controls stayed flat. That is stronger
than an unexplained end-to-end win: removing padding, the dense mask, and GQA
expansion predicts lower prefill-forward time, not lower decode or cache-write
time, and that is what the profile shows.

## Cross-engine calibration

Signed measurement revision `2fafb45` then ran the complete unprofiled
Qwen3-8B suite at both fixed chunk budgets. The final amend changes tests and
documentation only; the measured `tinyserve/` subtree is identical. Every
condition used two warmups and five measured repetitions. The table below
shows the prefill-heavy `p2048/o32` rows; throughput is higher-is-better and
TTFT is lower-is-better.

| budget | batch | M7a throughput | M7b throughput | change | M7a TTFT | M7b TTFT | improvement |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 23.174 | 23.180 tok/s | +0.0% | 495.24 | 495.49 ms | −0.1% |
| 512 | 8 | 44.271 | 44.227 tok/s | −0.1% | 2,948.23 | 2,950.58 ms | −0.1% |
| 512 | 32 | 49.184 | 49.281 tok/s | +0.2% | 10,398.98 | 10,382.91 ms | +0.2% |
| 8,192 | 1 | 25.557 | 25.515 tok/s | −0.2% | 369.45 | 370.16 ms | −0.2% |
| 8,192 | 8 | 60.164 | 66.532 tok/s | **+10.6%** | 3,233.52 | 2,823.13 ms | **+12.7%** |
| 8,192 | 32 | 70.253 | 79.272 tok/s | **+12.8%** | 8,152.80 | 7,117.42 ms | **+12.7%** |

The controls say as much as the wins. B1 never enters the multi-prompt fused
path and stays flat. At budget 512, a 2,048-token prompt is chunked rather than
packed and all three batch rows stay within 0.2%. Only B8 and B32 at budget
8,192 repeatedly execute ragged whole-prompt prefill, and only those rows move
by 10–13%.

The decode-heavy `p128/o128` throughput rows moved between −0.02% and +0.60%.
Their prompt is small relative to 128 decode tokens, so M7b only trims TTFT:
1.0–1.9% in the packed B8/B32 rows and effectively zero at B1. This is not a
decode optimization, and the calibration does not present it as one.

Peak memory did not regress. The 512-budget maximum changed from 35,975 to
36,001 MiB; the 8,192-budget maximum changed from 37,447 to 37,443 MiB. Both
sustained suites reached 87 degrees C. The candidate B32 treatment even reached
a lower 1,500 MHz sampled clock than M7a's 1,545 MHz minimum, yet retained the
double-digit win. The experiment was not opposite-order, so sub-percent
movements remain noise; the profiled mechanism and the 10–13% treatment effect
are the promotion evidence.

External engines were not rebuilt or rerun because their revisions, model
fingerprints, and host environment did not change. Against the frozen matched
references, M7b narrows but does not erase the TTFT gap. At B8 it moves from
3,234 to 2,823 ms, versus vLLM's 2,366 ms and FreeToken's 1,472 ms. At B32 it
moves from 8,153 to 7,117 ms, versus 5,848 and 5,203 ms respectively. Those
references include streaming HTTP while tinyserve's timer is engine-internal,
so the remaining comparison is directional, not a claim of API-level parity.

The 512-token control is important. A 2,048-token prompt cannot enter whole-
prompt packing under that budget, so M7b should not improve those prefill rows.
The 8,192-token budget packs four such prompts per forward and directly
exercises the new kernel. Reporting both prevents a scheduler-policy change
from masquerading as a kernel improvement.

## Run it

The ordinary demo now exposes the prefill budget. Three short prompts fit in
one packed forward and select the ragged FlashInfer path:

```bash
PYTHONPATH=$PWD .venv/bin/python examples/generate.py \
  --model /home/scratch.kaix_coreai/models/Qwen3-0.6B \
  --serve --backend flashinfer --no-cuda-graphs --chunk-size 512 \
  --prompts "2+2=" "Why is the sky blue?" "Name three colors." \
  --max-new-tokens 4 --verbose
```

To step through the implementation, use the **M7b: ragged packed prefill**
launch configuration and break in `_paged_prefill_batch_ragged()`, then in
`Qwen3Attention._paged_attend()`.

The focused measurement uses the shared calibration runner with
`--prompt-tokens 2048 --output-tokens 32 --batch-size 32 --chunk-size 8192
--phase-profile`. Headline calibration omits `--phase-profile` and runs the
complete fixed suite.

The structured [M7b evidence](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m7b-ragged-prefill-a6000-2026-08-29.json)
retains phase medians, all standard-suite deltas, memory/thermal envelopes,
commands, source revision, and SHA-256 hashes of the raw artifacts.

## Takeaway

Packing answers “how many requests share this forward?” Padding answers “how
do unequal rows fit a rectangle?” They are separate decisions. A serving
engine can keep the first and eliminate the second.

M7b turns prompt boundaries into kernel metadata instead of fake tokens. The
result is small enough to read end to end: flatten IDs, positions, and physical
slots; plan one ragged layout; reuse it in every layer; select one final hidden
state per prompt. On the phase M7a identified, that removed 12.4% of device
time without moving decode or cache-write work.
