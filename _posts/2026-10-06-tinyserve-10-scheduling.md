---
layout: post
math: true
title: 'Tinyserve, Chapter 10: Scheduling prefill and decode'
date: 2026-10-06 00:00:00 -0700
categories:
- tinyserve
- llm-serving
excerpt: Who runs next and how much prompt work can run before decoders progress?
book_chapter: 10
source_revision: 14b0c8dbb975350bbd730ac0c702eafce36eb95c
source_document: docs/book/10-scheduling.md
code_revision: e20a34815498560f9226a9e057ade31b2b62743f
---

{% include tinyserve-article-style.html %}
{% include tinyserve-book-nav.html %}

*Chapter 10 · The Serving Runtime*

Two requests are streaming tokens. A third arrives with a long prompt. The
server must process that prompt before it can answer, but a whole-prompt
forward delays the next token for both existing requests. Adding the new
request to a queue does not solve this interference. We need to decide when
it enters the active set, how much work it gets per iteration, and how that
work is executed.

This chapter follows four requests through Tinyserve's implementation. It
assumes you know what prefill, decode, and a paged KV cache are: prefill
processes an uncached history, decode extends that history one token at a
time, and a page table maps logical token positions to physical KV storage.
We will use small token counts so that every iteration can be inspected.

## Three separate scheduling decisions

These mechanisms cooperate, but they solve different problems:

| Decision | Mechanism | What it changes |
|---|---|---|
| Which requests participate now? | Continuous batching | Batch membership can change at iteration boundaries. |
| How much prompt work runs this iteration? | Chunked prefill | A shared prompt-token budget bounds each iteration's prefill work. |
| How are the selected tokens represented? | Packing and mixed execution | Multiple requests can share model operations without sharing attention histories. |

A static batch waits for its original members to finish before taking a new
batch. Continuous batching retires finished requests and admits waiting ones
while other requests remain active. Here, “continuous” means repeated
decisions between iterations, not that a request can interrupt a running GPU
kernel. Tinyserve's ordinary session executes a decode forward, then one or
more prefill forwards. It does not preempt a model call halfway through.

Without chunking, the gap between two decode opportunities can include an
entire long prompt. With a budget of four tokens, a ten-token prompt takes
three iterations: four tokens, four tokens, then two. Existing decoders get
an opportunity to run between those chunks.

[![Whole-prompt prefill delays the next decode; chunking spreads the same ten prompt tokens across three iterations.](/assets/tinyserve/book10-prefill-interference.svg)](/assets/tinyserve/book10-prefill-interference.svg)

*The widths are schematic, not timings. This first picture keeps A and B
active throughout; the trace below adds departures and another arrival.
All ten prompt tokens still need processing. Chunking bounds token work,
not milliseconds: attention cost also depends on the existing history and
execution backend. Token markers indicate internal sampling, not delivery
to a network client.*

Chunking alone can still leave many small model calls. Packing addresses
that second problem. We will first trace the scheduling decisions, then
change the execution layout while keeping the selected work fixed.

## A four-request example

Use an ordinary dense model, greedy sampling, unquantized weights and KV,
and enough cache pages to avoid preemption. Disable prefix reuse and EOS
termination so that output lengths are predictable. Each page holds four
tokens; the shared prompt budget is four tokens per iteration.

We begin after A and B have each produced their first output token:

| Request | Prompt length | Total output limit | Arrival in the trace |
|---|---:|---:|---|
| A | 5 | 5 | Already emitted A1 |
| B | 3 | 3 | Already emitted B1 |
| C | 10 | 2 | Before iteration 1 |
| D | 3 | 2 | Before iteration 3 |

`A1` means A's first generated token, not its numerical token ID. Prompt
ranges are half-open: `C[4:8]` means positions 4, 5, 6, and 7 of C's prompt.

A's KV initially contains its five prompt tokens. A1 has been sampled and
appended to A's token history but has **not yet been processed by the model**.
Thus `num_cached = 5` and `len(tokens) = 6`. A's next decode call consumes
A1 at position 5 and samples A2. B starts with the same one-token gap:
three cached prompt tokens and B1 waiting to be consumed.

The ordinary session's iteration is, in simplified form:

1. Handle cancellation and requests requiring no output.
2. Admit waiting requests whose cache reservation fits.
3. Reserve space for decoders, preempting if necessary.
4. Decode the surviving `RUNNING` requests once; emit and retire as needed.
5. Spend the shared prompt budget on `PREFILLING` requests in arrival order.

Admission happens once before decode. If a decoder finishes and frees pages
in step 4, a waiting request can use that newly available capacity at the
next iteration's admission, not through a second admission pass this time.

## Memory admission is not the chunk budget

C's ten-token prompt needs three pages of four slots, even though its first
chunk processes only four tokens. In this implementation, admission reserves
space for the **full current token history**, not just the next chunk and
not the entire future output allowance. Those pages initially contain no
valid KV for C. `num_cached` determines how much has actually been computed.

With prefix reuse enabled, live shared pages can reduce the new reservation;
warm cached pages still consume capacity when adopted. The scheduler also
leaves a one-page admission watermark, which decode may later consume.
The example disables reuse so that admission
and incremental computation are easy to distinguish.

The waiting queue uses **FCFS: first come, first served**. If its first request
cannot be admitted, the scheduler does not skip it to admit a smaller one
behind it. Once resident, prefilling requests also consume the prompt budget
in arrival order. This simple policy is understandable, but it can cause
head-of-line blocking. It is not a fairness or latency guarantee.

For an iteration with $N_{\text{decode}}$ surviving decoders and prefill chunks of lengths
$c_1,\ldots,c_n$, the selected query-token count is

$$
Q = N_{\text{decode}} + \sum_{i=1}^{n} c_i.
$$

Writing the prompt budget as $B_{\text{prompt}}$ (the `chunk_size` setting),

$$
\sum_{i=1}^{n} c_i \leq B_{\text{prompt}}.
$$

The budget is shared across prefilling requests. It excludes decode tokens.
Two decoders plus four prompt tokens therefore make six query tokens, not
four. It is also separate from the cache capacity reservation above.

## Five iterations, including arrivals and departures

Here is the complete trace. “Decode” lists the requests that were ready to
decode at the start of that iteration; finishing prefill does not earn a
second, decode step in the same iteration.

| Iteration | Decode | Prompt work, in FCFS order | Outputs sampled this iteration |
|---:|---|---|---|
| 1 | A, B | C[0:4] | A2, B2 |
| 2 | A, B | C[4:8] | A3, B3; B finishes |
| 3 | A | C[8:10], D[0:2] | A4, C1 |
| 4 | A, C | D[2:3] | A5, C2, D1; A and C finish |
| 5 | D | None | D2; D finishes |

[![Every iteration of the four-request example, with the selected work and ordinary versus mixed model-call counts.](/assets/tinyserve/book10-iteration-trace.svg)](/assets/tinyserve/book10-iteration-trace.svg)

In iteration 1, C becomes resident and caches its first four prompt tokens.
It cannot sample an output yet: its final prompt token has not been processed.
In iteration 2, C reaches eight cached tokens while B reaches its output limit.

Iteration 3 shows what “shared budget” means. C needs only two more tokens,
so D receives the two tokens left over. C emits C1 after its final prompt
position is processed; D does not emit anything. If each request received
four tokens independently, this would no longer be the same budget policy.

Iteration 4 decodes A and C and processes D's last prompt token. A and C
retire, while D emits D1 and becomes a decoder for iteration 5. No artificial
padding request keeps the batch at its original size.

The test [test_book_scheduling.py](https://github.com/kaix-nv/tinyserve/blob/14b0c8dbb975350bbd730ac0c702eafce36eb95c/tests/test_book_scheduling.py) executes
this example against a small real CPU model in both layouts. It checks
events, retirement, cached cursors, model-call shapes, packed metadata, and
cache reclamation. That is executable documentation of this trace, not a
GPU numerical or performance qualification.

## What a prompt chunk actually changes

The core operation in `_paged_chunk_prefill` is an append of a contiguous
range of one request's history:

```text
start = seq.num_cached
end   = min(len(seq.tokens), start + this_chunk_budget)
input = seq.tokens[start:end]
positions = start, start + 1, ..., end - 1
```

The model writes KV for those positions and attends to the already cached
history plus the allowed part of the new chunk. After the forward,
`num_cached` advances to `end`. A partial chunk does not append an output
token. The last chunk uses the logits at the final prompt position to sample
the first output, then transitions the request to `RUNNING` unless that
output already finishes it.

For C's second chunk, the query positions are 4 through 7, and the visible
KV range contains positions 0 through 7. Query position 4 may see keys 0
through 4, but not keys 5 through 7. If `i` is a chunk-local query index and
`j` is a key's logical position, the causal rule is

$$
j \leq \texttt{start} + i.
$$

A triangle starting at zero over the new chunk alone would be wrong: it
would discard the earlier prefix. Positions for RoPE must likewise continue
from `start`; every chunk is not a fresh sequence starting at position zero.

[![Request lifecycle from waiting through incremental prefill and decode to terminal cleanup, including preemption and the uncached sampled token.](/assets/tinyserve/book10-request-lifecycle.svg)](/assets/tinyserve/book10-request-lifecycle.svg)

*For a nonterminal decoder after sampling, `len(tokens) = num_cached + 1`.
The final sampled token need never enter KV: a finished request can release
its cache immediately. Cleanup resets its cached cursor; that reset is not
evidence that the forward failed to compute its history.*

## Packing does not choose the schedule

The budget decides **which token ranges** to process. Packing decides how
to arrange the selected ranges for execution. Tinyserve has several related,
but distinct, paths:

| Path | Inputs it can combine | Representation |
|---|---|---|
| Ordinary whole-prompt batching, `_paged_prefill_batch` | Consecutive fresh, complete prompts that fit the remaining budget | The reference path uses left padding; the single-device FlashInfer path can use a ragged flat token row. |
| Ordinary chunking, `_paged_chunk_prefill` | One partial or already-started request | One contiguous slice; separate model forward. |
| Opt-in mixed execution, `append_forward` | Decode tokens and selected prompt chunks | One flat token row with per-request boundaries and histories. |

Consequently, `_paged_prefill_batch` is not another name for chunked prefill.
It batches whole prompts; the session's budget loop decides when it is
eligible. Nor does “batching” automatically mean that padding has disappeared.

In our ordinary iteration 3, A's decode is one forward. C's final two prompt
tokens are another. D's first two tokens are a third. The selected ranges
are useful and bounded, but there are still three trips through the model.

Mixed execution represents every contribution as an **append span**:
`(sequence, start, end)`. A decoder contributes one token; a prefiller
contributes as many as its turn and the remaining prompt budget allow.
The mixed session first records surviving decoders, then selects prefills:

```python
budget = self.chunk_size
for req in self.scheduler.running:
    if req.state != PREFILLING or budget == 0:
        continue
    start = req.seq.num_cached
    end = min(len(req.seq.tokens), start + budget)
    spans.append(AppendSpan(req.seq, start, end))
    requests.append(req)
    budget -= end - start
```

This excerpt selects work; it does not run attention or update the cursor.
`append_forward` constructs the inputs, executes the model, and commits
the cached cursors afterward. A decode-only iteration uses the existing
decode path instead of constructing this mixed layout.

## Inside iteration 3's packed forward

At the start of iteration 3, A has seven cached tokens and A3 waiting at
position 7. C has eight cached prompt tokens. D has none. Their spans are
therefore `A[7:8]`, `C[8:10]`, and `D[0:2]`.

[![Iteration 3 packs five input rows while retaining independent positions, physical KV slots, attention histories, and output selections.](/assets/tinyserve/book10-packed-step.svg)](/assets/tinyserve/book10-packed-step.svg)

The important arrays are small enough to write out:

```text
packed row:            0    1    2    3    4
request:               A    C    C    D    D
logical position:      7    8    9    0    1
query boundaries:     [0, 1, 3, 5]
rows needing logits:  [0, 2]
```

`[0, 1, 3, 5]` is a prefix sum of query lengths `[1, 2, 2]`. It says A owns
packed rows `[0:1)`, C owns `[1:3)`, and D owns `[3:5)`. The positions are
not `arange(5)`: packed row number is not logical position.

KV destinations are independent again. Suppose the page tables are A:
`[9, 2]`, C: `[4, 12, 6]`, and D: `[7]`. These are illustrative page IDs,
not a promise about allocator order. Write a request's block table as $T$.
For page size four,

$$
\text{slot}(p) = 4\,T[\lfloor p/4\rfloor] + (p \bmod 4).
$$

The five writes go to slots `[11, 24, 25, 28, 29]`. A flat input row has not
made the KV histories contiguous or shared. C's position-8 query can read
C's keys 0 through 8; position 9 can read 0 through 9. Neither may read A
or D. Request boundaries, page tables, valid lengths, and causal masking
jointly enforce that separation.

Embeddings, projections, and feed-forward operations can work over all five
rows. Attention still needs the three independent histories. The reference
gather backend loops over spans and runs attention separately for each;
the FlashInfer append wrapper can consume the batched paged metadata.
**One model forward is not a claim of one attention kernel.**

Finally, which rows need the vocabulary projection? A's last row produces
A4. C's last row produces C1. D is still partial, so it produces no output.
The hidden tensor is `[1, 5, H]`, where `H` is hidden width. Selecting packed
rows `[0, 2]` gives `[1, 2, H]`; the LM head projects this to `[1, 2, V]`,
where `V` is vocabulary size. `append_forward` removes the singleton batch
axis to return `[2, V]`. In the implementation, `output_rows = [0, 1]`
identifies the emitting **spans**, A and C; `prefill_output_indices = [0, 2]`
identifies the corresponding **packed hidden rows**. Those index spaces
must not be confused.

The two layouts select the same work in this example, but their internal
sampling timestamps differ. The ordinary path samples and records A4 after
the decode forward, before running C's chunk. The mixed path records A4
and C1 after the shared forward. In both paths, synchronous `step()` returns
the accumulated events only after the whole iteration and synchronization;
the online worker delivers them afterward. An internal token timestamp is
therefore not network-delivery latency. Packing does not guarantee that
every decoder gets its token earlier.

## Memory pressure and model-specific state

Our trace has spare pages. Under pressure, a decoder crossing a page
boundary needs an additional page. If reservation cannot succeed, the
scheduler preempts the most recently admitted resident request, whether
`RUNNING` or `PREFILLING`. It releases
that request's page references, clears its block table and cached cursor,
and places it at the front of the waiting queue.

Preemption retains the token history and request-local sampling state. On
readmission, prefill recomputes the uncached history, including generated
tokens already emitted; it does not sample those tokens a second time.
Reusable prefix pages may remain owned by the prefix cache. Releasing a
request's references is not the same as erasing every physical page.
Recomputation costs time, and this last-in-first-out victim policy does not
provide a bound on a request's wait.

The storage policy is also model-specific. Hybrid linear-attention models
carry recurrent state as well as ordinary-attention or MLA history.
Tinyserve's hybrid scheduler reserves a private state slot per resident
request and bounds the resident count. That is not an interchangeable pool
of ordinary KV pages; do not assume the dense model's preemption behavior
or the mixed path applies automatically. Hybrid prefill also distinguishes
the scheduler's token budget from a linear-attention kernel's internal
chunk size. They belong to different layers of the implementation.

## When fewer forwards help—and when they do not

Chunking trades long uninterrupted prompt work for more frequent decode
opportunities. Smaller chunks can improve responsiveness but increase
per-chunk overhead and reduce useful matrix sizes. Larger chunks do the
opposite. The budget should therefore be evaluated against both time to
first token (TTFT) and inter-token latency (ITL), not throughput alone.

Packing may reduce model-call overhead and give token-wise operations more
rows over which to reuse weights. It does not remove the underlying token
work or the need to read each request's attention history. Account for host
work, launch overhead, token-wise operations, and attention across all model
forwards. If $t_f$ is the time attributed to model forward $f$, a simple
accounting model is

$$
t_{\text{iteration}} \approx t_{\text{host}} + \sum_f t_f.
$$

This is an accounting aid, not an additive prediction of overlapped GPU
execution. Changing the number and shape of forwards can change every term.
Packing overhead and backend choice also matter.

A bounded measurement illustrates the opportunity. On 2026-10-05, a
Qwen3-0.6B BF16 experiment on one RTX A6000 compared separate and mixed
**eager gather** execution. Both used a 512-token prompt budget, cold private
KV, and a 32,768-token cache pool. After two warmup pairs, six measured pairs
alternated which path ran first; output tokens and counts matched in every
measured pair. The ratio below is the median paired separate-time / mixed-time:
greater than one favors mixed.

| Prompt / output tokens per request | Requests | Total model forwards, separate → mixed | Iterations mixing decode and prefill | Time ratio, 95% bootstrap interval |
|---|---:|---:|---:|---|
| 128 / 128 | 1 | 128 → 128 | 0 | 1.003 [0.971, 1.029] |
| 128 / 128 | 8 | 130 → 129 | 1 | 1.029 [0.985, 1.059] |
| 2048 / 32 | 8 | 91 → 63 | 28 | 1.155 [1.121, 1.192] |

The one-request case has nothing to mix. The short-prompt concurrent case
mixes only once and does not establish a clear gain. The long-prompt case
has repeated overlap and benefits in this experiment. GPU clock locks had
been reset, but clocks were not fixed. These small-sample paired results
are not a universal speedup claim. The sanitized
[measurement record](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m12-post-reset-performance-a6000-2026-10-05.json)
and [qualification report](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/m12-mixed-prefill-decode.md) retain the scope.

There is an important counterexample: in the separate Qwen3-8B standard
suite, mixed eager execution did not consistently beat separate eager
execution, and the graph-enabled default led all six tested conditions.
That sequential suite is not the counterbalanced paired experiment above, but
it is enough to reject promoting mixed execution as a faster default.
Changing the attention backend can also change the work performed, so a
whole-path gain must not be attributed solely to packing.

Numerical qualification is separate. Keeping histories private preserves
the intended attention operation; it does not guarantee bit-identical BF16
results after changing matrix shapes and floating-point execution order.
The original BF16 compatibility gate against the old layout **still fails**.
The opt-in path passed an explicitly adopted contract using independent
attention checks, shape-matched packing checks, and bounded quality tests.
That does not retroactively turn the old gate into a pass. FreeToken's
reference run also failed the matched-output-length check and remains an
excluded, failed comparison—not evidence for a ranking.

The qualified mixed slice remains opt-in: eager, single-device, greedy,
ordinary dense Qwen with unquantized weights and KV. This chapter does not
extend those claims to CUDA graphs, stochastic sampling, hybrid models,
quantized execution, or distributed serving.

## Reading the implementation

The mechanism is spread across a few small ownership boundaries:

| Location | Responsibility |
|---|---|
| [scheduler.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/scheduler.py) | Request states, FCFS admission, page reservation, and preemption. |
| [session.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/session.py) | Ordinary iteration, shared prompt budget, emission, and retirement. |
| [engine.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/engine.py) | `_paged_prefill_batch`, `_paged_chunk_prefill`, and `_paged_decode`. |
| [mixed.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/mixed.py) | Append spans, packed inputs, per-request attention metadata, and mixed iteration. |
| [attention.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/attention.py) | Backend attention behavior, including the shifted causal mask. |
| [models/qwen3.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/models/qwen3.py) | Position handling, KV writes, and selecting hidden rows before the LM head. |

These examples describe code at `e20a348`; the new chapter adds explanation,
figures, and the CPU trace test, not a new serving policy. The linked source
repository currently requires access; the chapter's worked example does not.

The central separation is useful beyond this implementation: admission
answers whether a request can reside in memory, scheduling selects its next
token range, and execution maps that range onto model operations. Continuous
batching changes membership, chunking bounds prompt work, and packing changes
the representation of that work. Keeping those decisions separate makes
both the code and its performance results easier to reason about.

{% include tinyserve-book-nav.html %}
