---
layout: post
title: "Building tinyperf, chapter 16: The mixed step"
date: 2026-09-29 12:16:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/16-the-mixed-step/
excerpt: "A serving engine with chunked prefill doesn't stop its running sequences to read a new prompt (chapter 14). It cuts the prompt into chunks and runs each chunk in the same step as the sequences that are decoding. Chapter 15 priced a step of decodes alone. This chapter asks what a step that carries both costs, a mixed step, and what the engine does between steps that no kernel shows."
redirect_from:
  - /tinyperf/perf-modeling/2026/09/08/building-tinyperf-m54.html
---

*[Building tinyperf](/series/tinyperf/) · Part IV: Serving · Code: [`tinyperf/serving.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/serving.py), `mixed_step_us`, `chunk_us` and `simulate` · Every table and the plot in this chapter come from `python3 book/scripts/ch16_mixed_step.py`.*

A serving engine with *chunked prefill* doesn't stop its running
sequences to read a new prompt (chapter 14). It cuts the prompt into
chunks and runs each chunk in the same step as the sequences that are
decoding. Chapter 15 priced a step of decodes alone. This chapter asks
what a step that carries both costs, a *mixed step*, and what the engine
does between steps that no kernel shows.

The short answer: a mixed step is one forward pass over the combined
rows. The weights stream once for decodes and chunk together; every GEMM
runs at the combined row count, with cuBLAS's row curve and tile edges;
the LM head runs only on the rows that sample; each side runs its own
attention; and on FlashAttention-2 the decode rows read their cache once
per query head. Neither the larger of the two parts nor their sum is
right. Between steps the GPU doesn't wait, but requests do: the engine
plans each step while the one before it runs, so a request that arrives
during step k usually misses step k+1 (the model assumes it always does
and admits it at k+2), and each request spends about 25 ms outside the
steps.

By the end of this chapter you will know:

- how to compose a mixed step from its parts, and why the combined rows,
  not the two parts, set its price;
- how the token budget shared by decodes and chunks shapes each step;
- what happens between steps: async scheduling, a per-request overhead,
  and a host that waits for its GPU;
- how to time one step inside a running server, and why the client's
  clock misreads it;
- how close the model gets: 36 steps within 0.97–1.07 of the engine's
  clock, and an attention backend that halves a step.

## Two prices that don't work

Every measurement in this chapter is Qwen3-8B (32 query heads over 8 KV
heads) on one RTX A6000, served by vLLM with its default attention
backend on this GPU, FlashAttention-2 (FA2 in the tables). Take a step
on that server: 63 sequences decoding from 2,048-token prompts, beside a
128-token chunk of a new prompt. A step of the decodes alone costs the
model 56.4 ms, and a step of the chunk alone 29.4 ms. What does the step
that carries both cost?

Two answers suggest themselves. If the chunk's math hides under the
decodes' weight traffic, the step costs the larger part: the *maximum*.
If the two share nothing, it costs their *sum*. Table 16.1 prices five
steps both ways, against the step as the engine itself timed it (the
engine's clock, described below). In the tables, *prompt* is the
decoding sequences' prompt length; their context is that plus the tokens
they have generated.

```
Table 16.1  A mixed step against its two parts: Qwen3-8B, FlashAttention-2, RTX A6000, engine's clock, ms
  prompt  decodes  chunk  decodes alone  chunk alone     max     sum  one forward  measured
     256        8    128           25.2         29.4    29.4    54.6         41.5      38.8
     256        8   1024           25.2        151.8   151.8   177.0        166.3     166.3
    1024       32    512           32.9         77.0    77.0   109.8        115.3     118.0
    2048       63    128           56.4         29.4    56.4    85.8        147.9     146.2
    2048       63   1024           56.4        151.8   151.8   208.1        277.1     266.9
  all 36 cells, price/measured: max 0.33-0.91; sum 0.59-1.41; one forward 0.97-1.07
```

Neither holds. The maximum is always low, 0.33–0.91 of the measured
step. The sum reads high on small steps (54.6 ms against 38.8) and low
where the decodes are many and long (85.8 against 146.2). The parts
interact, and the "one forward" column, which this chapter builds,
prices the interaction: 0.97–1.07 over all 36 cells of the grid.

## One forward over the combined rows

vLLM doesn't run a mixed step as two steps. It lays the step's rows side
by side, one per decoding sequence and one per chunk token, and runs the
model's forward once. Follow chapter 6's layer through that batch:

- **Per-token work at the combined rows.** Every GEMM, norm, rotary
  embedding and residual runs once, over 63 + 128 = 191 rows. The
  weights stream once for everyone, which is what the maximum got right.
  But a GEMM's time at 191 rows is chapter 3's measured row curve at 191
  rows, not the decodes' price plus the chunk's.
- **Each side's attention.** The chunk's queries attend to its prefix
  (the part of its prompt already in the cache) and to each other,
  `prefix + (n + 1)/2` keys on average for n new tokens (chapter 6).
  Each decode row attends to its own cache: chapter 7's decode kernel at
  the batch's mean context.
- **The re-read.** Beside a chunk, FA2's decode rows lose their GQA
  packing (chapter 7): each row's cache is read once per query head,
  less what the L2 catches. FlashInfer and vLLM's Triton kernel keep the
  packing and pay nothing.
- **The LM head at the rows that sample.** vLLM samples one row per
  decoding sequence and one per chunk (a partial prompt's token is
  discarded). A chunk's other rows never reach the head.
- **The sampler** on those rows (chapter 15). On this stack a step lasts
  its forward plus its sampler to within 0.2 ms; the engine's clock
  finds no gap between steps (the provenance note in
  `data/calibration/rtx_a6000.json`).

So:

```
mixed step = per-token work (decodes + chunk tokens)
           + attention (chunk, prefix) + decode attention (decodes, mean context)
           + re-read (decodes, context, chunk)                  FA2 only
           + LM head and sampler (decodes + chunks)
```

![Six mixed steps as stacked bars of per-token work, LM head, the
chunk's attention, the decodes' attention and the re-read, with the
measured step marked on each.](/assets/tinyperf-book/ch16-composition.svg)

*Figure 16.1. What a mixed step is made of. With 8 decodes a step is
per-token work, and the chunk's size sets it. With 63 decodes from
2,048-token prompts, FA2's re-read is the largest part beside a short
chunk; FlashInfer, which keeps its packing, doesn't pay it.*

```
Table 16.2  The parts of a mixed step as the model composes it: Qwen3-8B, RTX A6000, ms
  prompt decodes chunk     backend  rows  per-token  head+sampler  chunk attn  decode attn  re-read   model  measured
     256       8   128         FA2   136       36.9           1.9         0.2          1.2      1.3    41.5      38.8
     256       8  1024         FA2  1032      157.6           1.9         4.2          1.2      1.5   166.3     166.3
    2048      63   128         FA2   191       38.1           2.1         0.2         29.6     77.9   147.9     146.2
    2048      63   128  FlashInfer   191       38.1           2.1         0.2         27.9      0.0    68.3      69.1
    2048      63  1024         FA2  1087      158.0           2.1         4.2         29.6     83.2   277.1     266.9
    2048      63  1024  FlashInfer  1087      158.0           2.1         4.2         27.9      0.0   192.2     187.6
```

With 8 decodes the decodes barely register, and a 1,024-token chunk
makes the step about four times as long as a 128-token one. With 63
decodes at a context of about 2,085 tokens beside a 128-token chunk,
per-token work is 38.1 ms and the re-read 77.9 ms: over half the step,
and more than the decodes' own attention. Under FlashInfer the re-read
is gone, and the step takes 69.1 ms on the engine's clock, 68.3 in the
model.

Two parts follow the step's row count, and neither smoothly:

```
Worked example  Two things the combined row count decides: Qwen3-8B, RTX A6000, CUDA graph
  LM head: 1 row 1.83 ms, 64 rows 1.89 ms, 1024 rows 11.17 ms, 2048 rows 22.31 ms
  per-token work of all 36 layers: 1024 rows 145.7 ms, 1025 rows 156.3 ms (+7.3%), 1032 rows (8 decodes + a 1024-token chunk) 157.6 ms
```

The LM head is the model's widest GEMM, 151,936 outputs. On one row it
streams its weights in 1.83 ms; on 2,048 rows it is bound by math, 22.31
ms. Priced on every chunk token, it would add about 20 ms to a full
step. And the per-token work crosses cuBLAS's tile edge at 1,024 rows
(chapter 3): one row more costs 7.3%, and a 1,024-token chunk beside 8
decodes is 1,032 rows, on the far side. Chapter 4's field note "three
zeros" tells how a fitted constant once stood where this edge and the
re-read now stand.

## The code

`mixed_step_us` composes the step (docstring and comments trimmed):

```python
    def mixed_step_us(self, decode_batch: int, kv_len: int, chunk_kv: int, chunk_new: int,
                      chunk_seqs: int = 1, shared_prefix: int = 0) -> float:
        d = self.decode_us(decode_batch, kv_len, shared_prefix) if decode_batch else 0.0
        if not chunk_new:
            return d
        if self.pp > 1 or self.comm_overlap or self.offload_weights or self.offload_kv or shared_prefix:
            c = self.chunk_us(chunk_seqs, chunk_kv, chunk_new)
            if not d:
                return c
            return max(d, c, d + c - self.weight_stream_us()) + self.mixed_reread_us(decode_batch, kv_len, chunk_new)
        tok = self._parts(decode_batch + chunk_seqs * chunk_new, 1, 1, decodes=decode_batch)["tok"]
        head = self._parts(decode_batch + chunk_seqs, 1, 1)["head"]
        att_c = self._parts(chunk_seqs, chunk_kv, chunk_new)["attn"]
        att_d = self.decode_attention_us(decode_batch, kv_len) if decode_batch else 0.0
        return tok + head + att_d + att_c + self.mixed_reread_us(decode_batch, kv_len, chunk_new)
```

`tok` prices the graph of a decode step with
`decode_batch + chunk_seqs * chunk_new` rows and keeps everything but
attention and the LM head: the per-token work at the combined rows, to
which the calibrated scheduler applies the row curve (chapters 3 and 4).
`head` prices the same graph at the rows that sample and keeps only the
LM head. `att_c` builds the chunks as sequences that each advance
`chunk_new` tokens past a prefix of `chunk_kv`, and keeps their
attention. `att_d` is chapter 7's decode attention at the mean context,
for the real decode rows: vLLM runs a step that carries a chunk outside
its full CUDA graph, so chapter 15's padded rows don't apply.
`mixed_reread_us` is chapter 7's, and returns 0 on a backend that packs.
The sampler is added by the simulator, per step.

The split into parts is `_parts` (docstring trimmed):

```python
    def _parts(self, batch: int, seq: int, chunk: int, decodes: int | None = None) -> dict:
        seq = max(KV_BUCKET, math.ceil(seq / KV_BUCKET) * KV_BUCKET) if seq > 1 else 1
        key = ("parts", batch, seq, chunk) + ((decodes,) if decodes is not None else ())
        if key not in self._memo:
            out = {"tok": 0.0, "attn": 0.0, "head": 0.0}
            stage = {"moe_decode_rows": decodes} if decodes is not None else {}
            for r in self._graph_report("decode", batch, seq, chunk, **stage).rows:
                k = "attn" if r.op_type == "fmha" else ("head" if r.name in ("lm_head", "logits_allgather") else "tok")
                out[k] += r.total_us
            self._memo[key] = out
        return self._memo[key]
```

`_graph_report` builds the decode graph and returns chapter 5's report,
one row per operation; `_parts` sorts the rows into attention, the LM
head and the rest. A prefix is rounded up to a multiple of 256 tokens,
the step-price cache's bucket (chapter 14). `decodes` tells an MoE's
expert kernels how many of the rows are decodes (chapter 9).

The branch at the top of `mixed_step_us` is older. With pipeline stages
(chapter 12), overlapped collectives (chapter 11), offloading or a
shared prefix (chapter 17), the step is still priced as the sum less one
weight pass, never less than the larger part, plus the re-read. Its
chunk comes from `chunk_us`, which prices every sequence advancing n
tokens, with the LM head on every token.

## Chunk-only steps and the token budget

Chapter 14's scheduler gives each step a token budget, vLLM's
`max_num_batched_tokens`: 2,048 on this GPU. Decodes are scheduled first
and each spends one token; the waiting prompts share what is left, in
arrival order. In `simulate`:

```python
                budget = chunk_tokens - len(decoders)
                chunked: list = []
                for w in waiting:
                    if budget <= 0:
                        break
                    c = min(budget, w.prompt - w.prefilled)
                    if c <= 0:
                        continue
                    chunked.append((w, c))
                    budget -= c
```

So a full step carries 2,048 rows whatever its mix, and the mix decides
what else it pays:

```
Table 16.3  One full token budget, shared: 2,048 tokens a step, decodes at context 1024 beside a new prompt, RTX A6000
  decodes  prompt tokens   FA2 ms  decode attn + re-read  FlashInfer ms  FA2 us per prompt token
        0           2048    292.1                    0.0          292.1                    142.6
        8           2040    297.6                    5.6          294.1                    145.9
       32           2016    315.0                   23.3          298.9                    156.2
       63           1985    345.8                   54.5          305.2                    174.2
  a step of the chunk alone through the older chunk graph: 316.6 ms (20.5 ms more LM head, 4.1 ms more attention from a 256-token minimum prefix)
```

Every row does the same per-token work over 2,048 rows. The decodes add
their attention and, on FA2, their re-read: at 63 decodes a full step
costs 345.8 ms against 292.1 for the chunk alone, and each prompt token
174.2 µs instead of 142.6. Under FlashInfer the same step costs 305.2
ms. On FA2, a decode batch slows the prompt it carries mostly through
its re-read.

A step with no decodes, such as a burst of new prompts on a quiet
server, is the same forward with nothing beside the chunks: 292.1 ms for
2,048 tokens, where the chunk graph charges 316.6 (4.1 ms of it from a
256-token minimum prefix). The engine's step logs, where both mechanisms
were found (in-sample; recorded when measured):

```
Recorded  The engine's step logs from two online sweeps of Qwen3-8B on vLLM, RTX A6000, as recorded
  steps of prompt chunks alone: 93; model/measured median 1.078 with the LM head on every token, 0.999 as one forward
  steps carrying prompt tokens: 2596; decodes + prompt tokens at most 2048; 841 steps at exactly 2048, 814 of them with decodes
```

Several prompts can share a step. `simulate` prices them as equal chunks
at their mean length and mean prefix:

```python
                pre_kv = max(1, sum(w.prefilled for w, _ in chunked) // len(chunked))
                step_us = lat.mixed_step_us(len(decoders), kv_now, pre_kv,
                                            max(1, new_tokens // len(chunked)), len(chunked))
```

The per-token work and the head come out right to a few rows (the mean
is rounded down); the chunks' attention is an approximation.

## Between steps

A request's latency also depends on when it gets into a step, and on
what it costs outside the steps.

**Async scheduling.** vLLM's engine thread plans step k+1 while the GPU
runs step k, so the GPU never waits for the host; it then processes step
k's tokens and plans k+2 while k+1 runs. A request that reaches the
engine during step k usually misses the plan for k+1. The model assumes
the plan for k+1 is made as step k begins, so every such request misses
it and first runs in step k+2 (Figure 16.2). An idle engine plans at
once. In `simulate`, a step's plan sees only the arrivals from before
the previous step began:

```python
        cutoff = prev_step_start if (async_scheduling and not host_sync_forward and (running or waiting)) else t
        while i < len(pending) and pending[i].arrival_us <= cutoff:
            queue.append(pending[i])
            i += 1
```

Table 16.4 follows one arrival through the model's steps, with the
budget at work:

```
Table 16.4  A 4,000-token prompt arrives at 3,000.0 ms beside 40 decoding sequences: the model's steps
   step  starts ms  lasts ms  decodes  prompt tokens  already prefilled
    k-1     2967.5      29.1       40              -                  -
      k     2996.7      29.1       40              -                  -
    k+1     3025.8      29.2       40              -                  -
    k+2     3055.0     298.0       40           2008                  0
    k+3     3353.0     329.5       40           1992               2008
  TTFT 707.5 ms = per-request overhead 25.0 + waiting 55.0 (the rest of step k, then k+1) + 298.0 + 329.5
  with each step planned just before it runs (no async scheduling): TTFT 678.3 ms
```

![A timeline of five steps on the GPU, the engine thread's plans one
step ahead, and a prompt that arrives during step k and first runs in
step k+2.](/assets/tinyperf-book/ch16-async-timeline.svg)

*Figure 16.2. The model admits an arrival at step k+2. It makes the plan
for each step as the step before it begins (the ticks), so a prompt that
arrives during step k waits for the rest of k and all of k+1. The model
adds the per-request overhead to the request's time rather than delaying
its arrival; since it is the same for every request, the queue is the
same either way.*

The prompt arrives 3.3 ms into step k, waits 55.0 ms, and is read in two
chunks, 2,048 − 40 = 2,008 tokens and then the rest. The second chunk
costs more because its queries attend to the first chunk's 2,008 tokens.
Planning each step just before it runs would save one decode step, 29.2
ms of TTFT (time to first token, chapter 14).

**The per-request overhead.** The model adds `online_overhead_us` to
every request's TTFT and end-to-end time, and to no step. It stands for
everything outside the steps: the HTTP request, handing it to the
engine, and streaming the answer back (not separated). It was fitted on
an idle server. `vllm bench serve` sent 24 requests, one every five
seconds on average, so none queued, and the median TTFT less the step
that carried the prompt is the overhead:

```
Table 16.5  The per-request overhead, from an idle server: Qwen3-8B on vLLM, one request every 5 s on average, ms
  prompt  requests  TTFT median  model step  difference
     128        24         47.7        29.8        17.9
    1024        24        176.3       152.2        24.1
  the calibration's online_overhead_us: 25.0 ms per request
```

The calibration's 25 ms was set from these probes when the model priced
their steps 2–6 ms below today's prices, so the differences were then 20
and 30 ms (the calibration's provenance note); against today's steps it
is 1–7 ms high. It is one constant per request, measured with nothing
queued.

**How long an arrival really waits.** Figure 16.2's rule says an arrival
waits for the rest of the step in progress plus one whole step: 1.0 to
2.0 decode steps, 1.5 on average. The injected prompts of the next
section measure it:

```
Table 16.6  How long an arrival waits beside running decodes, in decode steps
  an injected prompt's TTFT, less its mixed step and the 25 ms per-request overhead, over the engine's decode step
                                                mean      range
  measured, 60 prompts injected in 20 cells     0.88  0.29-1.50
  joins step k+1 (planned just before it runs)  0.50  0.00-1.00
  joins step k+2 (async scheduling, the model)  1.50  1.00-2.00
  with Table 16.5's differences (17.9-24.1 ms) as the overhead: mean 0.92-1.14
  prompts that waited under one step: 39 of 60 at 25 ms, 20 at 17.9 ms
  two online sweeps at 1-3 req/s, 1,024-token prompts, TTFT median model/measured: k+2 1.03-1.08, k+1 0.74-0.94
```

The measured wait, 0.88 steps on average (0.92–1.14 with Table 16.5's
differences as the overhead), lies between the two rules, and 20 to 39
of the 60 prompts, depending on the overhead, waited under one step,
which the k+2 rule never allows. The injected prompts sit nearer k+1;
the load sweeps favour k+2 (one is in-sample: the rule was adopted with
it in view), where the model's median TTFT reads 1.03–1.08, against
0.74–0.94 with k+1. Both are consistent with the engine making its plan
for k+1 partway into step k, once it has processed step k−1's tokens, so
that the earlier arrivals in a step still catch the next plan (not
profiled). The model charges the whole step, and at low load reads TTFT
a few percent high for it.

**A host that waits for its GPU.** Async scheduling assumes the engine
thread returns as soon as it has launched a forward. If the forward
blocks until the GPU finishes, both halves of the rule change. The plan
for k+1 is made after step k ends, so an arrival during k joins k+1. And
with async scheduling the thread launches step k+1 before it processes
step k's tokens; if that launch blocks until k+1 finishes, k's tokens
leave only when k+1 ends. vLLM's P2P NCCL KV connector on a
disaggregated prefill server (one GPU prefills and ships the KV cache to
another that decodes; chapter 20) does this: it synchronizes on every
layer. `simulate(host_sync_forward=True)` models it:

```
Worked example  A forward that blocks the host: two 1,024-token prompts, the second 50 ms after the first
  forward runs ahead of the host : first answer at 176.8 ms, second at 328.5 ms
  host waits for each forward    : first answer at 328.5 ms, second at 328.5 ms
```

The first answer waits a whole step for the second prompt's prefill. On
that server, turning async scheduling off releases every answer at its
own step's end; chapter 20 measures both.

## Measuring one step inside a server

A mixed step can only be measured inside an engine, beside real decodes.
`tools/measure_mixed_step.py` does it on a live vLLM server. B
background requests, with prompts of 256, 1,024 or 2,048 random token
ids, decode 500 tokens each, greedily. At 3, 5 and 7 s the tool sends a
prompt of C tokens with `max_tokens=1`; its prefill rides in a step
beside the B decodes. The grid is B = 8, 32 or 63 and C = 128 to 1,024:
36 cells.

The step is read on the engine's own clock. A small patch to vLLM's
model runner (`tools/vllm_patches/step_timing`) records CUDA events on
the model's stream at the start of every forward and around the sampler,
reads them back without a sync, and logs one line per step: its period
(the GPU time from its forward's start to the next one's, idle time
included), its decodes and their contexts, and each chunk's tokens and
prefix. The tool keeps the steps with exactly B decodes and one C-token
chunk.

The obvious alternative is the client's clock: the step that carries the
new prompt delays every background request's next token, so the largest
gap after the injection should be the mixed step. On the same cells:

```
Table 16.7  The client's largest inter-token gap against the engine's clock: FlashAttention-2 grid, ms
  prompt  decodes  chunk  client gap  engine clock  client/engine
     256        8    256        51.5          60.1           0.86
     256       63    256        68.8          79.6           0.86
    1024        8    128        35.7          43.0           0.83
    2048        8    512        88.7         102.6           0.86
     256       63   1024       182.9         180.4           1.01
  20 cells with three clean injections: client/engine 0.83-1.01, median 0.91; pure decode steps 1.00-1.02
  16 cells whose first injection met background prompts still prefilling (its TTFT 1.9-19.2 s): its gap 303-311 ms, client/engine up to 2.35
```

The client reads low: 0.83–1.01 of the engine's clock, 0.91 at the
median, although its pure decode steps agree within 2%. A client sees a
token when the engine has processed its step's output, which under async
scheduling overlaps the next step. The shortfall is consistent with part
of a long step's delay landing in a neighbouring gap rather than in the
largest one (not profiled). And a client can't see what a step carried.
In 16 cells, with 32 or 63 long background prompts, the first injection
arrived while those prompts were still prefilling. It waited 1.9–19.2 s,
and the largest gap after it was a background prefill step of 303–311
ms. The engine's log shows each step's composition and drops such steps;
the client's median reads up to 2.35 there.

## How close is it?

Table 16.8 is the grid on the engine's clock, each cell priced by the
current model at the decode context the engine recorded. Nothing in the
model was fitted to these steps: the re-read, the row curve and the
decode attention come from their kernels timed alone (chapters 3 and 7).
The model was committed before the grid ran, so the grid is held out;
since then it has been a test that every change to the model must pass.

```
Table 16.8  The mixed step on the engine's clock: Qwen3-8B, FlashAttention-2, RTX A6000 (held out)
  measured ms and model/measured, by the injected chunk; context = the decode rows' mean, as recorded
  prompt  decodes  context           128           256           512          1024
     256        8      447    38.8  1.07    60.1  1.05    93.2  1.01   166.3  1.00
     256       32      394    43.1  1.02    69.9  1.00   102.5  0.98   174.2  0.99
     256       63      342    57.0  1.00    79.6  1.00   110.3  1.00   180.4  1.01
    1024        8     1172    43.0  1.04    64.4  1.03    96.4  1.01   166.6  1.02
    1024       32     1065    55.6  1.04    83.5  1.01   118.0  0.98   186.0  1.01
    1024       63     1044    92.2  1.00   116.2  0.99   146.3  1.01   212.6  1.03
    2048        8     2144    47.2  1.05    68.5  1.04   102.6  0.99   174.0  1.00
    2048       32     2068    76.2  1.05   106.6  1.01   142.5  0.97   211.4  1.00
    2048       63     2085   146.2  1.01   169.8  1.01   199.9  1.02   266.9  1.04
  the model, all 36 cells:                   0.97-1.07, typical error 1.9%
  without the re-read, all 36 cells:         0.48-1.04, typical error 20.4%
  without cuBLAS's row curve, all 36 cells:  0.78-0.99, typical error 12.8%
  without either, all 36 cells:              0.42-0.92, typical error 41.0%
  the older branch, all 36 cells:            0.83-1.05, typical error 8.2%
  without the re-read, 8 decodes from 256-token prompts, 128- and 256-token chunks: 1.04, 1.03
```

Every cell is within 0.97–1.07, a typical error of 1.9%. Each mechanism
earns its place: without the re-read the typical error is 20.4%, without
the row curve 12.8%, without both 41.0%. The older branch of
`mixed_step_us`, the sum less one weight pass plus the re-read, reads
8.2%. The highest ratios are on short steps beside 8 decodes, 1.03–1.07
at 128- and 256-token chunks, consistent with chapter 7's finding that
the re-read's straight line runs high at few rows; but with 8 decodes
from 256-token prompts the model reads 1.03–1.04 high even without the
re-read (not explained).

The context is an input, and the model needs it right. The predictions
committed before the run assumed the decode rows would be 250 tokens
into their replies. At the injections they were fewer (the recorded
contexts in Table 16.8), and those predictions read higher:

```
Recorded  Predictions committed before the grid was measured, model/measured
  at a planned context (the background prompt + 250 tokens): 1.00-1.14, typical error 5.7%
```

## The attention backend changes the step

Under FlashInfer (chapter 7) the model drops the re-read and swaps in
FlashInfer's decode constants. FA2's grid was run a second time beside
FlashInfer's; Table 16.9 divides by that run.

```
Table 16.9  FlashInfer's mixed steps over FlashAttention-2's second run, 8 and 32 decodes: engine's clock, measured ratio
  prompt  decodes    128    256    512   1024
     256        8   1.02   1.01   0.98   1.01
     256       32   0.91   0.97   0.94   0.94
    1024        8   0.98   1.00   0.97   0.97
    1024       32   0.78   0.84   0.85   0.90
    2048        8   0.93   0.97   0.92   0.94
    2048       32   0.66   0.72   0.74   0.82
  model/measured, all 36 cells: FlashInfer 0.96-1.02, FlashAttention-2's second run 0.97-1.06
  FlashAttention-2's grid, measured in two separate runs: second/first 0.99-1.03, median 1.002
```

The ratio has the re-read's shape. At 8 decodes the two backends cost
the same within 8%; at 32, FlashInfer's advantage grows with the context
and shrinks as the chunk grows. It is largest at 63 decodes, the rows of
chapter 7's Table 7.8. The model prices both backends' steps within
0.96–1.07 (held out: the FlashInfer run was predicted before it ran),
and the measurement repeats: FA2's grid, run twice, agrees within
0.99–1.03. A step that halves (Table 16.2) changes the queue behind it;
chapter 18 follows the backend to the knee.

## Where it breaks

- **One model, one GPU, one engine.** Every step here is Qwen3-8B on one
  RTX A6000 under vLLM 0.15.1. An MoE's mixed step also changes which
  experts it touches (chapter 9).
- **Several chunks at their mean.** A step's chunks are priced as equal
  chunks at their mean prefix: right for the per-token work, not for
  attention, which grows faster than linearly with a chunk's length.
  Prefixes round up to 256 tokens.
- **The re-read's form.** Chapter 7's straight line reads high at few
  rows and multiplies the decode kernel's whole time, floor included.
- **The older branch** (Table 16.8) still prices steps with pipeline
  stages, overlap, offloading or a shared prefix, and `chunk_us` a
  cached-prefix prefill, with the LM head on every new token: about 9 ms
  too much at 1,024.
- **The async rule** charges a whole step that arrivals often don't wait
  (Table 16.6).
- **The per-request overhead** is one constant, measured with nothing
  queued; under load the frontend shares the host with the engine thread
  (not measured).
- **The client's clock** under-reads long steps; measure on the engine.

## What you built

- A mixed step as one forward over the combined rows: per-token work at
  the combined row count, with cuBLAS's row curve and tile edges; the LM
  head and sampler at the rows that sample; each side's attention; and
  FA2's re-read.
- Chunk-only steps as the same forward, and a token budget the decodes
  spend first.
- What happens between steps: plans made a step ahead, so an arrival
  during step k usually misses k+1 (the model assumes it always does); a
  per-request overhead of 25 ms, fitted on an idle server; a blocking
  forward that holds answers for a step.
- A measurement of one step inside a running server on the engine's own
  clock, and the reasons a client's clock misreads it.
- Evidence: 36 held-out steps within 0.97–1.07 (typical error 1.9%);
  FlashInfer's steps under half of FA2's beside many long decodes (Table
  16.2).

## Exercises

1. Price a step that carries two prompts of different lengths exactly,
   each chunk's attention on its own, and compare it with `simulate`'s
   mean-chunk price. When does the difference pass 2%?
2. Add a plan lag L to `simulate`: arrivals in the first L µs of a step
   still join step k+1. Which L matches Table 16.6's 0.88 steps, and
   does it keep the sweeps' TTFT within 5%?
3. With FA2 at 63 decodes and a 2,048-token context, how large a chunk
   would bring the re-read below 10% of the step, and does it fit the
   2,048-token budget? What does that say about choosing a budget for
   each backend?
4. Re-run `tools/measure_mixed_step.py --step-timing` holding each
   injection until every background request has streamed its first
   token. Do the 16 unusable cells come back, and does the client's gap
   still read low?
5. Give `chunk_us` the mixed step's rule for the LM head, and price a
   prefill of 1,024 new tokens after a cached 4,096-token prefix both
   ways. How much does the prefill change?

---

*[← Chapter 15: What a decode step is made of]({% post_url 2026-09-29-tinyperf-15-what-a-decode-step-is-made-of %}) · [Contents](/series/tinyperf/) · [Chapter 17: Memory under load →]({% post_url 2026-09-29-tinyperf-17-memory-under-load %})*
