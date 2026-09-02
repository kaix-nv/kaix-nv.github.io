---
layout: post
math: true
title: "Building tinyserve M7h: Chunk and pack hybrid prefill"
date: 2026-09-02 11:35:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Bound hybrid prompt work with a real-token budget, pack consecutive FCFS chunks into one forward, and keep padding out of recurrent, convolution, KV, and MLA state."
---

*Milestone 7h of [building an LLM inference engine from scratch](/series/tinyserve/).
Previous: [M7g — hybrid state becomes a scheduler resource]({% post_url 2026-09-01-building-tinyserve-m7g %}).
Next: [M7i — pack Kimi experts, then batch the selected work]({% post_url 2026-09-02-building-tinyserve-m7i %}).*

Code: [`tinyserve` @ `ed96c7c`](https://github.com/kaix-nv/tinyserve/tree/ed96c7c49805012028a19ede50c1685c7bdb6bae).

M7g solved ownership: every Qwen3.5 or Kimi request retained its own recurrent
state and growing history while decode batched live requests. It left prompt
execution deliberately simple. After decode, the engine ran one entire FCFS
prompt.

That leaves two different problems:

1. one long prompt can still delay every decoder for the duration of its full
   recurrence scan;
2. running many short prompts separately repeats model-forward overhead.

**Chunking bounds the first problem. Packing solves the second.** M7h applies
both without weakening M7g's state-ownership contract.

![Chunked prefill interleaves a long prompt with decode, while packing spends
the remaining FCFS token budget](/assets/tinyserve/m7h-hybrid-prefill.svg)

## One budget, counted in real prompt tokens

Each hybrid engine step still starts with decode. It then walks the
`PREFILLING` requests in FCFS order and assigns at most `chunk_size` real
tokens:

```python
budget = chunk_size
for request in prefilling_fcfs:
    start = request.seq.num_cached
    end = min(request.prompt_len, start + budget)
    actions.append((request, start, end))
    budget -= end - start
    if budget == 0:
        break
```

The selected chunks share one model forward. Packing changes the number of
forward calls, not accounting: a 1-token row plus a 3-token row consumes four
tokens of budget even though the rectangular input contains six positions.

The scheduler never skips a partially filled FCFS head to find a more
convenient shape. If the head consumes the whole budget, later prompts wait
for the next step.

## Follow three prompts with `chunk_size = 4`

The debugger example uses Kimi's tokenizer:

```text
A = "Hi"                                                   1 token
B = "This is a deliberately longer prompt for chunked prefill"  10 tokens
C = "ok"                                                   1 token
```

All three own state slots. The first prefill step packs A with the first three
tokens of B:

```text
prefill forward 0: A[0:1) + B[0:3)       real-token cost = 4
```

A is now `RUNNING`; B remains `PREFILLING`. Decode therefore gets another
opportunity before B advances:

```text
decode A
prefill forward 1: B[3:7)                real-token cost = 4
decode A
prefill forward 2: B[7:10) + C[0:1)      real-token cost = 4
```

This trace demonstrates both mechanisms. B is split across engine steps, and
otherwise-unused room in the first and final steps executes neighboring FCFS
work.

## Why ordinary padding would corrupt hybrid state

The first packed forward needs a rectangular `[B=2, T=3]` tensor:

```text
A: Hi₀  PAD  PAD       valid length = 1
B: B₀   B₁   B₂        valid length = 3
```

An attention-only model can mask A's padding. A hybrid model has three more
ways for those pads to leak into future tokens:

- a recurrent update can decay and rewrite A's matrix twice;
- a width-4 convolution can shift both pads into A's retained tail;
- full-attention or MLA storage can append two fake history entries.

M7h carries `token_lens=[1,3]` and derives a boolean `token_mask` rather than
assuming every tensor position is real.

### Recurrent state: padded transitions are no-ops

Gated DeltaNet and KDA still compute the rectangular tensor, but commit a
candidate state only for real rows at time `t`:

```python
candidate = transition(state, token_t)
state = torch.where(token_mask[:, t, None, None, None], candidate, state)
```

The padded row may produce a throwaway hidden value. Its persistent matrix is
bit-for-bit the state after its last real token.

### Convolution state: retain the tail at each row's real endpoint

Right padding is useful because real outputs appear first and therefore do not
depend on later pads. The retained tail cannot simply use the rectangular
tensor's final three columns, however. For row `r`, M7h copies the three
columns ending at:

```text
previous_tail_width + token_lens[r]
```

A's tail stops after `Hi₀`; B's stops after `B₂`.

### Growing history: append and expose real positions only

`HybridBatchCache` and `KimiBatchCache` copy only `new[:token_lens[r]]` into
each row's full-attention or MLA history. Their attention mask is now
three-dimensional:

```text
visible[batch row, query in this chunk, key in full history]
```

A's padded queries receive one finite dummy key for numerical portability,
but their results are discarded. Every real query sees its old prefix plus
the causal part of its own chunk.

### Logits: select the final real token, not column `T-1`

For a shorter row, `hidden[:, -1]` is padding. The batch cache therefore
supplies `output_indices = token_lens - 1`. The model projects exactly one
hidden row per request, taken from its final real chunk token.

After the forward, `commit()` scatters updated fixed state and real history
back to the same private caches. A request becomes `RUNNING` and samples its
first token only when `end == prompt_len`.

## The numerical parity boundary

Packed BF16 matmuls need not reproduce a batch-1 reduction bit-for-bit. On the
random Kimi fixture, the retained four-prompt calibration has maximum prefill
logit error `0.00293`, with the same initial argmax. One row later reaches an
exact BF16 tie between its leading candidates; batch geometry changes the tie
break and its generated string diverges after four matching tokens.

That is not evidence of state leakage: the error is bounded, argmax and the
top-three candidate set are stable, both run orders reproduce the same
result, and the long-prompt workload is text-identical across chunk sizes.
Qwen3.5 and the three-prompt Kimi acceptance workloads remain text-identical
to independent generation. The correct invariant is numerical parity plus
deterministic state ownership, not universal string identity for a random
near-tied model.

## Validation

The M7h acceptance tests exercise both hybrid architectures:

| gate | result |
|---|---|
| real-token budget | every retained packed action satisfies `sum(end-start) <= chunk_size` |
| actual packing | at least one forward contains chunks from two FCFS requests |
| actual chunking | the long request appears in more than one prefill forward |
| decode interleaving | a decode of A occurs between every retained pair of B chunks |
| padding safety | unequal `[1,3]` rows preserve Qwen3.5 and Kimi generated text versus solo execution |
| BF16 numerical parity | four packed Kimi prompts stay within `0.004` logits with identical argmax and top-three sets |
| state-slot lifecycle | the M7g release/reuse tests remain green under packed admission |
| regression partitions | 25 focused M7e–M7h/scheduler tests and 91 unique repository tests pass; one late 14 GiB FP32 prefix-cache allocation requires a fresh process after earlier pools |

## The latency tradeoff

The September 1, 2026 long-prompt experiment used the tiny Kimi fixture on one
RTX A6000 in BF16. Request A decoded 16 tokens. A 386-token request B arrived
at `t=0.15 s`. Each condition had one warmup and three measured repetitions.

| prefill budget | median worst ITL of A | median TTFT of B | total elapsed |
|---:|---:|---:|---:|
| `32` tokens | 0.316 s | 4.072 s | 6.201 s |
| `4096` tokens (whole prompt) | 0.763 s | 0.884 s | 3.149 s |

Chunking reduces A's worst inter-token gap by **2.41×**. It also makes B wait
4.61× longer for its first token and nearly doubles total completion time.
`chunk_size` is therefore an ITL-versus-TTFT/throughput control, not a free
speedup.

## Same-checkpoint batching calibration

The fixed four-prompt calibration repeats M7g's two opposite-order sweeps,
with one warmup and three measured repetitions per order. The table reports
the pooled median of six measurements:

| execution path | batch 1 | batch 2 | batch 4 |
|---|---:|---:|---:|
| Tinyserve sequential `generate()` | 7.40 tok/s | 7.27 tok/s | 7.32 tok/s |
| Tinyserve M7h `serve()` | 7.13 tok/s | 14.00 tok/s | 26.94 tok/s |
| Transformers + FLA static batch | 9.21 tok/s | 17.61 tok/s | 33.54 tok/s |

Packing is slightly negative at batch 1, where it adds metadata without work
to amortize. It makes scheduled Tinyserve 1.93× faster than sequential
execution at batch 2 and 3.68× faster at batch 4. Relative to M7g's scheduled
path, M7h improves batch-2 throughput by 1.24× and batch-4 throughput by
1.68×. Transformers remains 1.25–1.29× faster; its fused FLA transitions and
static cohort remain the optimization target, not a scheduler-equivalent
comparison.

## Run and debug it

```bash
.venv/bin/python examples/generate.py \
  --model /home/scratch.kaix_coreai/models/kimi-k3 \
  --serve --hybrid-slots 3 --chunk-size 4 \
  --prompts "Hi" \
    "This is a deliberately longer prompt for chunked prefill" "ok" \
  --no-chat --max-new-tokens 4 --backend gather --verbose
```

The **M7h: chunked packed hybrid prefill** launch configuration runs this
example. Set breakpoints in this order:

1. `_serve_hybrid()` while building `actions` — watch the FCFS budget become
   `[A:1, B:3]`;
2. `KimiBatchCache.__init__()` — inspect `token_lens`, `token_mask`, and
   `output_indices`;
3. `kda_recurrence()` — watch `torch.where` retain A's state on its pads;
4. `KimiDeltaAttention._causal_conv()` — see A and B choose different tail
   endpoints;
5. `KimiBatchCache.update_mla()` — inspect the per-query causal visibility;
6. `batch_cache.commit()` — see cursors advance by one and three real tokens.

The latency experiment is:

```bash
.venv/bin/python examples/bench_hybrid_prefill.py \
  --device cuda:1 \
  --output .tinyserve-bench/m7h/prefill-tradeoff.json
```

## What M7h establishes

M7h makes hybrid prefill an engine-step resource. Decode gets the first turn;
prompt work has a real-token bound; leftover capacity packs the next FCFS
request; and padding cannot mutate private recurrent, convolution, KV, or MLA
state.

It still copies request state into transient batches, runs recurrence in
Python, and invokes routed experts in small groups. The next profile showed
that state gather/commit was only about `0.39%` of wall time while routed MoE
consumed roughly three quarters of model time. M7i therefore optimizes expert
execution first while keeping M7h's token-budget, mask, cursor, and ownership
invariants unchanged.
