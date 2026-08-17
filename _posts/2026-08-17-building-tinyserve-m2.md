---
layout: post
math: true
title: "Building tinyserve M2: Static batching"
date: 2026-08-17 11:06:15 -0700
categories: [tinyserve, llm-serving]
excerpt: "Batch ragged prompts with left-padding, preserve per-sequence positions, and expose the throughput gains and wasted work of a fixed batch."
---

*Milestone 2 of [building an LLM inference engine from scratch](/series/tinyserve/).
Previous: [M1 — the KV cache]({% post_url 2026-07-14-building-tinyserve-m1 %}).*

Code: [`tinyserve` @ `1715e66`](https://github.com/kaix-nv/tinyserve/tree/1715e661a7edfc9588b9d1ba8af319081d7e27c7).

## What M2 adds

M1 generates one sequence at a time. During each decode step, the model
reads its weights and launches thousands of kernels just to produce one
token:

`M1: one decode step → one output token`

Most of the GPU is idle. M2 puts several sequences into the same model
forward:

`M2: one decode step → B output tokens`

The model still launches the same operations and uses the same weights,
but now those operations work on `B` rows. This can greatly increase total
throughput.

M2 uses **static batching**: the batch is created once and keeps the same
rows until every request finishes. It is intentionally simple. Its two
bookkeeping problems are:

1. prompts have different lengths;
2. requests finish at different times.

We will follow one batch from beginning to end to see how both are
handled.

## A batch from start to finish

Suppose tokenization produces two prompts:

```text
short prompt: [a, b, c]        length = 3
long prompt:  [u, v, w, x, y]  length = 5
```

The batch size is `B = 2`. The longest prompt has length `P = 5`, so
the model needs a rectangular input tensor with shape `[B, P] = [2, 5]`.

### Step 1: left-pad the prompts

Tinyserve adds padding to the **left** of the shorter prompt:

```text
input_ids
short: [pad, pad, a, b, c]
long:  [u,   v,   w, x, y]
```

The code constructs two companion tensors:

```text
real
short: [False, False, True, True, True]
long:  [True,  True,  True, True, True]

positions
short: [0, 0, 0, 1, 2]
long:  [0, 1, 2, 3, 4]
```

Each tensor answers a different question:

- `input_ids`: which token ID is stored in each physical column?
- `real`: does this column contain a real prompt token?
  `True` means real token; `False` means padding.
- `positions`: what is the token's logical position inside its own
  sequence?

Token `a` is physically stored in column 2, but it is still position 0
of the short sequence. Pad positions are set to 0 because their keys will
be hidden by the attention mask.

The implementation uses the EOS token ID in the pad slots. That value is
only a placeholder: the mask, not the token ID, determines whether a
column is padding.

Why left-padding? The final real token of every prompt lands in column
`P - 1`:

```text
short: [pad, pad, a, b, c]
long:  [u,   v,   w, x, y]
                           ^ final prompt token for both rows
```

The first generated token is predicted from these final prompt tokens, so
the engine can select `logits[:, -1]` for every row. Right-padding would
also work, but prefill would need to gather a different final column for
each prompt.

### Step 2: prefill the whole batch

Prefill sends all `[B, P]` tokens through the model in one forward pass.
The attention mask enforces two rules:

1. a query can only see keys at or before itself (causal attention);
2. a real query cannot see a padding key.

For the short prompt, the meaningful attention relationships are:

```text
query a → a
query b → a, b
query c → a, b, c
```

No real token sees either `pad` column.

In code, the prefill mask has shape `[B, 1, P, P]`. A `True`
entry means “this query may attend to this key”; a `False` entry
means “block this key.”

Pad slots still run through every transformer layer because the tensor is
rectangular. Their outputs are ignored.

> **Why add the diagonal?** It lets each pad query see itself instead of
> having no visible keys. The tested PyTorch build already returns a
> finite zero for a fully masked query, but the diagonal avoids depending
> on backend-specific behavior.

Prefill writes K and V for all `P` physical columns into the cache,
including the pad columns. The pad K/V entries cannot affect generation
because every later attention mask keeps them hidden.

Finally, `last_token_only=True` applies the expensive vocabulary
projection only to the final column. The result has shape `[B, vocab]`:
one next-token distribution per prompt.

### Step 3: sample the first generated token

Sampling turns the two distributions into two token IDs:

```text
next_ids = [g_short_0, g_long_0]
```

These are the first output tokens. They are recorded in each request's
output, but they are not in the KV cache yet. To predict the next tokens,
the engine feeds this `[B, 1]` column back into the model.

### Step 4: decode one aligned column at a time

Both sampled tokens are written to physical cache column `P`:

```text
short cache: [pad, pad, a, b, c, g_short_0]
long cache:  [u,   v,   w, x, y, g_long_0]
                                  ^ physical column P
```

The key mask grows by one visible column:

```text
short keys: [False, False, True, True, True, True]
long keys:  [True,  True,  True, True, True, True]
```

The physical cache column is shared, but the RoPE position is different
for each row:

| row | physical cache column | logical RoPE position |
|---|---:|---:|
| short | `P = 5` | `length = 3` |
| long | `P = 5` | `length = 5` |

At decode step $s$, row $i$ uses

$$
\text{position}_{i,s} = \text{prompt_length}_i + s.
$$

For the next step, both rows write physical column 6, while their logical
positions become 4 and 6. Tinyserve therefore tracks two coordinate
systems:

- **physical cache columns**, shared by the rectangular batch;
- **logical token positions**, different for each sequence.

This makes batched generation follow the same position sequence as
generating each prompt alone.

> **RoPE note:** Adding the same constant offset to every token in a row
> cancels from query-key dot products in exact arithmetic. True logical
> positions are still preferable because they match the unpadded path
> numerically and do not waste the model's finite position range on pads.

The decode attention mask has shape `[B, 1, 1, current_cache_length]`.
It keeps the original pad columns hidden and exposes every real prompt
and generated token.

The loop repeats:

```text
sample one token per row
        ↓
record unfinished rows
        ↓
feed the sampled [B, 1] column into the model
        ↓
append its K/V and produce the next logits
```

### Step 5: keep finished rows in the batch

The engine maintains one Boolean per row:

```text
done = [False, False]
```

When the short request samples EOS:

```text
done = [True, False]
```

Tinyserve stops recording tokens for that row. However, the row remains
in every model forward until the long request also finishes. Its later
tokens and logits are discarded.

This cannot corrupt another request—attention never crosses the batch
dimension—but it wastes computation. That wasted row is the defining
limitation of **static** batching. Continuous batching will replace
finished rows with new requests in M4.

## The algorithm in one view

The complete control flow in [`generate_batch()`](https://github.com/kaix-nv/tinyserve/blob/1715e661a7edfc9588b9d1ba8af319081d7e27c7/tinyserve/engine.py)
is:

```python
input_ids, real, positions, lengths = left_pad(prompts)

prefill_mask = (causal & real_keys) | diagonal
logits = model(
    input_ids,
    positions,
    cache,
    prefill_mask,
    last_token_only=True,
)

done = [False] * batch_size
key_real = real

for step in range(max_new_tokens):
    next_ids = sample(logits)             # [B]
    # Record next_ids only for unfinished rows.
    # Then mark rows that sampled EOS as done.

    if every_row_is_done or step == max_new_tokens - 1:
        break

    key_real = append_visible_column(key_real)
    positions = lengths + step            # [B, 1]
    decode_mask = key_real[:, None, None, :]
    logits = model(next_ids[:, None], positions, cache, decode_mask)
```

The implementation is spread across three small surfaces:

- [`engine.py`](https://github.com/kaix-nv/tinyserve/blob/1715e661a7edfc9588b9d1ba8af319081d7e27c7/tinyserve/engine.py) builds the padded tensors and
  drives prefill and decode.
- [`models/qwen3.py`](https://github.com/kaix-nv/tinyserve/blob/1715e661a7edfc9588b9d1ba8af319081d7e27c7/tinyserve/models/qwen3.py) accepts per-row
  positions and explicit attention masks.
- [`sampling.py`](https://github.com/kaix-nv/tinyserve/blob/1715e661a7edfc9588b9d1ba8af319081d7e27c7/tinyserve/sampling.py) samples `[B, vocab]`
  logits into `[B]` token IDs.

## Two practical lessons

### Project only the logits that will be sampled

A first version applied the language-model head to every prefill
position. Its output shape was:

$$
B \times P \times \text{vocab_size}.
$$

With `B=256`, `P=512`, a vocabulary of 151,936, and bf16 elements, that
tensor needs about 40 GB. Only the last prompt position is used for
sampling, so `last_token_only=True` reduces the output to
`[B, 1, vocab]`. The transformer still processes every padded position;
only the unnecessary vocabulary projections are skipped.

### Benchmark inference under `inference_mode`

An early benchmark accidentally retained autograd graphs across decode
steps and ran out of memory. The engine already uses
`@torch.inference_mode()`; standalone measurement scripts must do the
same. Otherwise the benchmark measures a training graph that the serving
path never creates.

## How correctness is checked

Batching should change throughput, not generated tokens. The tests in
[`tests/test_batching.py`](https://github.com/kaix-nv/tinyserve/blob/1715e661a7edfc9588b9d1ba8af319081d7e27c7/tests/test_batching.py) check three levels:

1. Ragged random prompts decoded in one fp32 batch produce exactly the
   same greedy token IDs as decoding each prompt alone.
2. `generate_batch()` on real text prompts returns the same strings as
   individual `generate()` calls.
3. A short request batched with a longer request returns only its own
   output; tokens computed after its EOS are not recorded.

These tests exercise padding, masks, per-row RoPE positions, cache
updates, sampling, and finished-row bookkeeping together. The full
M2-era test suite contains eight passing tests.

## What batching buys

The following measurements are a historical M2 snapshot from one RTX
A6000. Here, `tok/s = B / step_time`: it counts one decoded token for
every row in the microbenchmark.

At a context length of 512:

| batch size | ms/step | raw tok/s | speedup over B=1 |
|---:|---:|---:|---:|
| 1 | 22.1 | 45 | 1× |
| 8 | 22.4 | 356 | 8× |
| 32 | 24.1 | 1327 | 29× |
| 64 | 41.1 | 1556 | 35× |
| 256 | 141.0 | 1815 | 40× |

Up to batch size 32, step latency stays close to 22 ms while each step
produces 32 times as many tokens. This is the fixed decode cost being
shared across the batch.

Longer contexts move the saturation point earlier:

| batch size | ms/step at ctx 2048 | raw tok/s | KV cache size |
|---:|---:|---:|---:|
| 1 | 22.1 | 45 | 0.2 GB |
| 8 | 23.4 | 342 | 1.9 GB |
| 16 | 40.2 | 398 | 3.7 GB |
| 32 | 71.1 | 450 | 7.5 GB |
| 128 | 257.1 | 498 | 29.8 GB |

The reason is KV-cache traffic. Model weights are shared across the
batch, but every sequence has its own keys and values. The amount read
by attention grows roughly with:

$$
\text{batch size} \times \text{context length}.
$$

At batch size 128 and context length 2048, the cache holds 29.8 GB.
Reading that many bytes at the A6000's peak 768 GB/s would take roughly
39 ms, while the measured step took 257 ms. M2's GQA
`repeat_interleave` likely adds substantial memory traffic, but no
profiler trace was retained, so the measurement does not prove an exact
bottleneck breakdown.

Raw throughput is also not the same as useful throughput. In one
end-to-end run with 32 ragged prompts and `max_new_tokens=128`, the
engine produced 406 returned tokens/s versus roughly 1330 raw tokens/s.
The gap includes padded prefill work, computation for rows that finished
early, and other end-to-end overhead. The original benchmark did not
retain a per-cause breakdown.

## What M2 teaches

Static batching demonstrates the main throughput idea: one model forward
can advance many requests at once. It also exposes two wastes created by
the rectangular batch:

- padding consumes prefill compute and KV-cache space;
- finished rows keep consuming decode compute.

M3 introduces paged KV storage so sequences need not occupy equal
contiguous reservations. M4 introduces continuous batching so a new
request can take a finished row's place.

## Reproducing

Run the correctness tests and a small real batch:

```bash
pytest tests/ -q
```

```python
from tinyserve import LLM

llm = LLM("Qwen/Qwen3-0.6B")
outputs = llm.generate_batch(
    ["Why is the sky blue?", "2+2="],
    max_new_tokens=64,
    verbose=True,
)
for output in outputs:
    print(output)
```

The original scripts, raw output, prompt list, and environment capture
for the performance tables were not retained in this commit. Treat those
numbers as historical evidence, not as a benchmark that can currently be
reproduced exactly.
