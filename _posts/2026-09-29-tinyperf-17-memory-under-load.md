---
layout: post
title: "Building tinyperf, chapter 17: Memory under load"
date: 2026-09-29 12:17:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/17-memory-under-load/
excerpt: "Chapter 6 counted the requests that fit on a GPU with a rule of thumb: 90% of the memory, less the weights, divided by one request's cache. A serving engine measures instead, at start-up, and allocates one fixed pool of KV-cache blocks. How big is that pool, really? And when the requests in flight need more cache than it holds, what happens to their latency?"
redirect_from:
  - /tinyperf/perf-modeling/2026/09/08/building-tinyperf-m51.html
  - /tinyperf/perf-modeling/2026/09/08/building-tinyperf-m53.html
  - /tinyperf/perf-modeling/2026/09/25/building-tinyperf-m72.html
  - /tinyperf/perf-modeling/2026/09/25/building-tinyperf-m74.html
---

*[Building tinyperf](/series/tinyperf/) · Part IV: Serving · Code: [`tinyperf/capacity.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/capacity.py), `vllm_kv_pool`, and [`tinyperf/serving.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/serving.py), `simulate` · Every table, and Figures 17.1 and 17.3, come from `python3 book/scripts/ch17_memory.py`.*

Chapter 6 counted the requests that fit on a GPU with a rule of thumb:
90% of the memory, less the weights, divided by one request's cache. A
serving engine measures instead, at start-up, and allocates one fixed
pool of KV-cache blocks. How big is that pool, really? And when the
requests in flight need more cache than it holds, what happens to their
latency?

The short answer: the pool is the memory the engine asks for, less the
weights, less the peak of a profiling pass it runs before serving, less
a little memory outside PyTorch. Each term can be priced from the
model's shape and the engine's settings, and the result lands within 1%
of vLLM's pool on eight held-out dense configurations. When the pool
runs out, vLLM evicts the newest running request and later recomputes
it from scratch. Priced that way, the model reproduces the engine's
preemption counts within 9%, its peak batch within 3% and its latency
percentiles within 4%. A scheduler that reserves each request's whole
length instead runs 22–49% fewer requests at a time wherever the engine
preempts, and gets the median wait wrong by up to 16.5 times.

By the end of this chapter you will know:

- how vLLM sizes its KV pool at start-up, term by term, and why a
  mixture-of-experts model needs one more term;
- how paged KV preempts, and what recompute costs;
- why reserving a request's full length under-batches and misprices the
  queue;
- how a cached prefix turns a prefill into a chunk step, and how a batch
  reads a shared prefix once;
- what offloading to the host costs on every step.

## Where the memory goes

vLLM 0.15.1 sizes its cache in five steps when it starts:

1. It asks CUDA how much memory the device has and takes
   `gpu_memory_utilization` of it, 0.9 by default: the memory it
   *requests*.
2. It loads the weights.
3. It runs a *profiling pass* and records PyTorch's peak.
4. It measures the memory held outside PyTorch.
5. It gives what is left to the KV cache, in blocks of 16 tokens.

![Two stacked bars of 47.40 GiB each: weights, profiling peak, memory
outside PyTorch, the KV pool, and the part not requested; the MoE bar
adds a workspace.](/assets/tinyperf-book/ch17-pool-stack.svg)

*Figure 17.1. Where the memory goes when vLLM starts, for a dense model
on one GPU and an MoE model on each of two. The dashed line is what the
engine requests; the KV pool is what is left of it.*

As a formula, with `max_num_batched_tokens` the token budget (the most
tokens one step may carry, chapter 14) and `max_num_seqs` the most
sequences that may run at once:

```
requested = ceil(gpu_memory_utilization × memory CUDA reports)
pool      = requested − weights − profiling peak − memory outside PyTorch
peak      = max(forward over the token budget, sampler over its rows) + an MoE's workspace
tokens    = the whole 16-token blocks that fit in the pool
```

With `VLLM_LOGGING_LEVEL=DEBUG` vLLM logs every term;
`tools/measure_kv_pool.py` starts a server and reads them. Table 17.1
sets the model beside the log.

```
Table 17.1  vLLM's KV pool term by term: Qwen3-8B on one RTX A6000, vLLM's defaults (GiB)
  gpu_memory_utilization 0.9, max_num_batched_tokens 2048, max_num_seqs 256
  term                                                          model vLLM logged
  memory CUDA reports                                           47.40       47.40
  requested: 0.9 x that                                         42.66       42.66
  weights                                                       15.26       15.27
  profiling peak                                                 1.40        1.40
    sampler: 256 rows x 151,936 logits x 38.6 bytes              1.40           -
    forward: 2,048 tokens x 3 x 16,384 values x 2 bytes          0.19           -
  held outside PyTorch                                           0.04        0.04
  KV pool                                                       25.97       25.95
  KV pool in tokens                                           189,072     188,944   model/logged 1.0007
  KV bytes per token 147,456; one 16-token block 2.25 MiB; the sampler's peak holds 10,182 tokens of cache
  CUDA reports 0.9875 of the nameplate 48 GiB; 48 x 10^9 bytes is 44.70 GiB
  chapter 6's rule, 90% of 48 x 10^9 bytes less the weights: 181,879 tokens (0.963)
```

Term by term:

- **The memory CUDA reports.** The RTX A6000 is sold as a 48 GB card;
  CUDA reports 47.40 GiB, about 48 GiB less the driver's share. Chapter
  6's rule read 48 × 10⁹ bytes, 44.70 GiB. tinyperf reads the nameplate
  as GiB, scaled by the fraction measured on this card.
- **The weights,** per GPU. Under tensor parallelism (tp=2: each GPU
  holds half of every weight matrix; chapter 11) the embedding and LM
  head are split by vocabulary too, 1/tp per GPU; a tied head is stored
  once; a weight-only format such as MXFP4 (chapter 10) counts at its
  packed width.
- **The profiling peak,** the larger of two passes. The sampler computes
  fp32 logits over the whole vocabulary for every row and sorts each row
  for top-p sampling: 38.6 bytes per logit, fitted to Qwen3-8B at 128 to
  512 rows. The forward is charged the MLP's live activations, 3 × (FFN
  width + hidden) values per token, the width being the GPU's share of
  the FFN (top-k experts' for an MoE): a form, not a fitted constant,
  checked on Table 17.2's forward-bound rows (in-sample).
- **Memory outside PyTorch:** 0.04 GiB on one GPU, 0.03 at tp=2, fitted.

The bytes per logit, the memory outside PyTorch and the fraction CUDA
reports are the fitted constants; the forward's form was chosen with the
same grid in view. The function, with type hints, docstring and the
returned breakdown trimmed:

```python
CUDA_VISIBLE_FRACTION = 47.399841 / 48
VLLM_SAMPLER_BYTES_PER_LOGIT = 38.6
VLLM_NON_TORCH_GIB = {1: 0.04, 2: 0.03}   # tp=2: Qwen3-8B, two configurations
VLLM_FUSED_MOE_CHUNK_SIZE = 16384


def cuda_memory_bytes(device):
    return device.hbm_gb * CUDA_VISIBLE_FRACTION * 2 ** 30


def vllm_kv_pool(p, device, tp=1, gpu_memory_utilization=0.9, max_num_batched_tokens=2048,
                 max_num_seqs=256, recipe=None, block_tokens=16, weight_only=None):
    requested = math.ceil(cuda_memory_bytes(device) * gpu_memory_utilization)
    weights = _weights_local_bytes(p, tp, 1, recipe, weight_only=weight_only)
    rows = min(max_num_seqs, max_num_batched_tokens)
    sampler = rows * p.vocab * VLLM_SAMPLER_BYTES_PER_LOGIT
    width = (p.top_k * p.ffn_hidden if p.n_experts else p.ffn_hidden) / tp
    forward = max_num_batched_tokens * 3 * (width + p.hidden) * p.dtype.nbytes
    workspace = 0.0
    if p.n_experts:
        m, n = min(max_num_batched_tokens, VLLM_FUSED_MOE_CHUNK_SIZE), 2 * p.ffn_hidden / tp
        workspace = (m * p.top_k * (max(n / 2, p.hidden) + max(n, p.hidden))
                     + (max_num_batched_tokens * p.hidden if max_num_batched_tokens > m else 0)) * p.dtype.nbytes
    activation = max(sampler, forward) + workspace
    non_torch = VLLM_NON_TORCH_GIB.get(tp, VLLM_NON_TORCH_GIB[1]) * 2 ** 30
    kv_bytes = requested - weights - activation - non_torch
    per_token = _kv_local_bytes_per_token(p, tp, recipe)
    tokens = max(0, int(kv_bytes // (per_token * block_tokens)) * block_tokens)
    return dict(tokens=tokens, ...)
```

vLLM's fused-MoE layer works in chunks of up to 16,384 tokens, so its
scratch buffers stay one chunk long; past one chunk, its output takes a
`tokens × hidden` buffer of its own. Which pass sets the peak depends on
the two settings:

```
Table 17.2  The profiling peak, the larger of two passes: Qwen3-8B, one RTX A6000, util 0.9 (GiB, in-sample)
   seqs  tokens  sampler  forward  model peak  logged peak  pool model/logged
     32    4096     0.17     0.38        0.38         0.38             1.0006
     64    2048     0.35     0.19        0.35         0.48             1.0052
    128    2048     0.70     0.19        0.70         0.71             1.0009
    256    2048     1.40     0.19        1.40         1.40             1.0007
    512    2048     2.80     0.19        2.80         2.79             0.9999
     64    8192     0.35     0.75        0.75         0.76             1.0006
     64   16384     0.35     1.50        1.50         1.51             1.0006
  all 20 configurations (utilization 0.5-0.9, tp 1 and 2):
    12 where a pass the model prices sets the peak: 0.9999-1.0025
    6 at 64 sequences, where the peak reads 0.47-0.50 against a sampler of 0.35: 1.0043-1.0082
    2 started only once, with a cold compile cache: 1.0269-1.0299
  cold compile cache (a token budget not started before), 9 starts later repeated: 6 read 0.13-0.72 GiB above the repeat (pool 0.973-0.995 of it), 3 the same
  the forward overtakes a 64-row sampler above 3,818 tokens
```

The sampler grows with the sequences, to 2.8 GiB at 512. The forward
grows with the token budget and lands within 0.01 GiB of the log. Two
things are unpriced. At 64 sequences the peak sits at 0.47–0.50 GiB
whatever the budget, above the sampler's 0.35, for reasons not yet
found; it costs up to 1% of the pool. And a start whose compile cache
is cold, here one at a token budget not started before, can read a
higher peak: 6 of the 9 such starts that were later repeated read
0.13–0.72 GiB above the repeat, and the two configurations started only
once read 1.027–1.030.

## How close is it?

Ten configurations the model had not seen were started on this machine
after their predictions were committed: four other models (Qwen3-0.6B,
4B, 14B and the MoE Qwen3-30B-A3B) and a new tp=2 configuration of
Qwen3-8B. The criterion, stated with them: the pool within 2%. After the
first two MoE cells missed, three more MoE configurations followed (the
table's last three rows). Each configuration was started at least twice,
and the comparison uses the last start.

```
Table 17.3  The pool on held-out configurations: vLLM 0.15.1 on RTX A6000s, predictions committed before any start
  model           util tokens seqs tp     model    logged  ratio no workspace  peak GiB model/logged
  Qwen3-0.6B       0.9   2048  256  1   375,520   375,552 0.9999            -           1.40 / 1.39
  Qwen3-0.6B       0.5   8192   64  1   207,840   206,576 1.0061            -           0.35 / 0.48
  Qwen3-4B         0.9   2048  128  1   250,688   250,192 1.0020            -           0.70 / 0.71
  Qwen3-4B         0.8  16384  256  1   211,088   210,160 1.0044            -           1.40 / 1.47
  Qwen3-14B        0.9   2048   64  1    96,736    95,808 1.0097            -           0.35 / 0.48
  Qwen3-14B       0.95   4096  512  1    96,240    96,096 1.0015            -           2.80 / 2.81
  Qwen3-14B        0.9   2048  256  2   360,144   358,272 1.0052            -           1.40 / 1.41
  Qwen3-30B-A3B    0.9   2048   64  2   299,712   296,432 1.0111       1.0203           0.47 / 0.60  (in-sample)
  Qwen3-30B-A3B   0.85   8192  256  2   216,848   215,840 1.0047       1.0552           1.90 / 1.92  (in-sample)
  Qwen3-8B         0.8   8192  128  2   430,528   429,424 1.0026            -           0.70 / 0.76
  Qwen3-30B-A3B    0.9  16384  128  2   272,960   271,072 1.0070       1.0876           1.70 / 1.76
  Qwen3-30B-A3B    0.9   4096  512  2   243,536   243,312 1.0009       1.0233           3.05 / 3.03
  Qwen3-30B-A3B    0.8  32768  256  2   151,408   148,368 1.0205       1.1861           2.52 / 2.64
  dense, 8 held out: 0.9999-1.0097; chapter 6's rule 0.87-1.77
  MoE, the first 2: as committed, without the workspace term, 1.0203, 1.0552; with it (in-sample) 1.0111, 1.0047
  MoE, the next 3 (committed with the term, held out): 1.0070, 1.0009, 1.0205; without it 1.0876, 1.0233, 1.1861
  weights, all 13: model/logged 0.991-1.000
  first starts within 0.2% of the last: 12 of 13; the other, Qwen3-14B 0.9/2048/64: 89,232 against 95,808 tokens (0.931)
  gpt-oss-20b 0.9/2048/64, MXFP4 experts at 4.25 bits (in-sample): model 638,432, engine 607,344 (1.051); weights model 12.80 GiB (13.74 GB), engine 13.7 GiB
```

All eight dense configurations land within 1%, from 0.6 to 14 billion
parameters, utilization 0.5 to 0.95, tp 1 and 2; the worst is the
64-sequence floor. Chapter 6's rule, which knows neither the utilization
nor the profiling pass, reads 0.87–1.77.

**A workspace that is never freed.** The MoE's first two cells missed,
1.020 and 1.055, with logged peaks of 0.60 and 1.92 GiB where the model
then had 0.35 and 1.40. The cause is in vLLM's source: its fused-MoE
layer takes two scratch buffers, `tokens × top_k × max(n/2, hidden)` and
`tokens × top_k × max(n, hidden)` values (n the GPU's share of the
gate-and-up width), from a workspace manager that keeps one buffer,
grows it to the largest request and never frees it. Still held when the
sampler runs, it adds to the peak: for Qwen3-30B-A3B at tp=2, 0.125 GiB
at a 2,048-token budget and 0.5 at 8,192 (Figure 17.1).

The term has no fitted constant, but its form was found by studying
those two misses, so they are in-sample for it (1.011 and 1.005). It was
then committed before three new configurations were started, which are
held out: 1.007, 1.001 and 1.0205, the last just past the criterion.
Its peak is 0.12 GiB above the model's, the size of the forward's hidden
states at its 32,768-token budget (32,768 × 2,048 × 2 bytes), perhaps
still held when the sampler runs (not profiled).

A weight-only format counts at its packed width: for gpt-oss-20b's MXFP4
experts the model's pool is 638,432 tokens against the engine's 607,344
(1.051, in-sample; chapter 10), both recorded in
`data/validation/predictions_gpt_oss_20b_rtx_a6000_online_b_before_measurement.txt`.
The model's 12.80 GiB (13.74 GB) matches the checkpoint (chapter 10);
the engine reports 13.7 GiB, 0.9 GiB more (not profiled), most of the
gap.

## Paged KV and preemption

Of chapter 14's two admission policies, *reservation* holds each
request's prompt plus maximum output from admission (room reserved but
not yet written is *phantom*), while *paged* admission holds only the
16-token blocks a request's tokens fill, so the pool runs full and, when
a running request needs a block and none is free, someone must give
theirs up (Figure 17.2).

![Top row: a pool of 24 blocks shared by requests A, B and C; the pool
fills, the newest request C is preempted and its blocks freed, and later
C resumes. Bottom row: a reserving scheduler holds 12 blocks each for A
and B, 7 of them empty, while C waits.](/assets/tinyperf-book/ch17-paged-kv.svg)

*Figure 17.2. Paged KV and a preemption. A request's blocks need not be
adjacent. The reserving scheduler below never preempts, but holds empty
blocks for output not yet written.*

vLLM 0.15.1's scheduler decides who. Each step:

- Running requests go first, in admission order, each allocating blocks
  for the tokens it writes this step.
- When one cannot get its blocks, the newest running request is
  *preempted*: all its blocks are freed and it goes to the front of the
  queue, repeatedly until the blocks are found.
- A preempted request resumes by *recomputation*: its prompt and every
  token it had generated are prefilled again, like a new prompt.
- A step that preempted anyone admits no one new.

`simulate(kv_paging="paged", chunk_tokens=...)` schedules the same way.
Its step scheduler, a closure inside `simulate`, with the docstring, the
allocation after the preemption loop and the admission loop's body
trimmed:

```python
    def paged_schedule():
        budget = chunk_tokens
        decs, chs, n_pre = [], [], 0
        k = 0
        while k < len(order) and budget > 0:
            r = order[k]
            target = r.prompt + r.recompute
            new = min(budget, target - r.prefilled) if r.prefilled < target else 1
            slots = r.prefilled + new if r.prefilled < target else r.prompt + r.generated
            need = pages(slots) - blocks.get(id(r), 0)
            while need > free_pages[0]:
                v = order.pop()                                 # the newest running request
                free_pages[0] += blocks.pop(id(v), 0)
                (waiting if v in waiting else running).remove(v)
                v.recompute, v.prefilled = v.generated, 0
                v.preemptions += 1
                n_pre += 1
                queue.insert(0, v)
                if v is r:
                    break
            ...
        if not n_pre:
            while queue and budget > 0 and len(order) < max_batch:
                ...                        # admit the queue's head while its blocks are free
        preempted_total[0] += n_pre
        return decs, chs, n_pre
```

`queue` holds requests not yet admitted, preempted ones at its front;
`order` the admitted ones, oldest first; `waiting` those still
prefilling and `running` those decoding; `pages(n)` is ⌈n/16⌉. A
prefilling request takes a chunk of the token budget, a decoding one a
single token, and `need` is the new blocks they require. While those are
not free, `order.pop()` preempts the newest request, possibly the one
asking; its `recompute` becomes its generated count, so readmission
re-prefills all of it. Only a step without preemptions admits from the
queue. On three requests in a pool too small for them:

```
Worked example  Three 1,000-token prompts, 600 output tokens each, in a 231-block pool (3,696 tokens)
  step 233 (6.36 s): the pool is full; the newest request is preempted after 231 output tokens and its blocks are freed
  step 600 (15.63 s): the oldest has written its last token; the preempted request recomputes 1,231 tokens (prompt + output) in one chunk
  first token (s): 0.17, 0.48, 0.48; finished (s): 15.65, 15.85, 24.95
  with room for all three (6,400 tokens): finished (s): 15.87, 15.92, 15.92
```

The preempted request loses 9 seconds, waiting for a request to finish
and then re-prefilling a cache it already had. Its time to first token
(TTFT, chapter 14) is unchanged; the loss is in its end-to-end time.

## How close is it? The pool under pressure

Three sweeps pushed vLLM past its pool under `vllm bench serve`, with
prefix caching off: Poisson arrivals, lengths from a recorded trace,
every output run to full length. The simulator replays the same trace
(`bench_requests`, chapter 14) through the engine preset with each
sweep's memory share, `VLLM.with_(gpu_memory_utilization=util)`, and
vLLM's defaults otherwise: 256 sequences, the 2,048-token budget, paged
KV, CUDA-graph sizes up to 256 (chapters 14 and 15):

```
Table 17.4  Three sweeps past the pool: vLLM 0.15.1 serving Qwen3-8B on one RTX A6000, prefix caching off
  max_num_seqs 256, 2048-token budget; fit = the pool over the mean prompt + output
  sweep  util     pool requests    prompts   outputs prompt + output   fit
  A       0.5   50,896      240  1024-1024   512-512           1,536  33.1
  H1      0.9  188,944      160  1038-3069  517-1533           3,145  60.1
  H2      0.5   50,896      240     54-973   79-1452           1,283  39.7
```

Sweep A helped build the scheduler: in-sample. H1 and H2 are held out:
new traces, predicted before they ran, with criteria stated then,
including preemptions within 15% and the peak batch within 5%. vLLM's
step log records each preemption, so the schedule itself is compared.

```
Table 17.5  Qwen3-8B at KV-cache saturation on one RTX A6000: vLLM's scheduler, model/measured
  sweep A is in-sample; H1 and H2 were predicted before they ran (held out); the pool is the engine's logged one
  sweep  rate TTFT p50    p95   TPOT  tok/s   preemptions   peak decodes  recomputed
        req/s                                model/engine   model/engine     (model)
  A       1.5    1.006  1.034  1.002  1.001      100 / 103        43 / 43        42%
  A       2.5    0.992  1.008  1.008  1.000       97 / 95         45 / 45        42%
  A       3.5    1.000  0.999  1.007  0.997      117 / 118        46 / 46        51%
  H1     0.75    0.996  1.002  1.006  0.997        0 / 0          70 / 68         0%
  H1     1.25    0.997  1.002  0.996  1.001      154 / 169        80 / 80        82%
  H1        2    1.014  1.026  1.018  0.988      172 / 168        87 / 86        90%
  H2        2    0.961  0.977  0.985  1.008      236 / 237        61 / 61       102%
  H2        3    0.979  0.981  0.988  1.011      238 / 235        64 / 63       108%
  H2        5    0.968  0.982  0.987  1.012      254 / 248        73 / 74       115%
  recomputed: the prompt and output tokens re-prefilled after preemptions, as a share of the prompts
  H1 and H2, engine steps: model/engine 0.999-1.003
  H1 and H2 on the derived pools (189,072 and 51,008 tokens): TTFT p50 0.97-1.01, TPOT 0.99-1.02, preemptions 0.94-1.09 of the engine's
  H1 at 1.25 req/s, the paged model's schedule: 42 of 160 requests preempted, one of them 24 times; first token for the first 80 arrivals 0.3-2.1 s, the last 40 63-126 s
  H1 measured: output tokens/s 579, 507, 519; TTFT p50 0.7, 2.0, 3.4 s; p95 1.6, 118.5, 153.4 s
```

On the six held-out cells, TTFT's median and 95th percentile land within
0.96–1.03, time per output token (TPOT) within 0.98–1.02 and throughput
within 1.2%. So does the schedule: preemptions within 9% (154 against
169 is the worst), the peak batch within 3%, steps within 0.3%. On this
chapter's derived pools, median TTFT and TPOT stay within 3%.

Preemption has a price. On H1, once the pool binds between 0.75 and
1.25 requests per second, the engine's output *falls* from 579 to 507
tokens per second; in the model's schedule the re-prefilled tokens come
to 82–90% of the prompts. On H2, with short prompts and long outputs,
they exceed the prompts.

It also shapes the tail. At 1.25 requests per second the median request
waits 2.0 seconds for its first token, the 95th percentile two minutes.
In the model's schedule (the measurement keeps only percentiles) the
first 80 arrivals, admitted beyond what the pool could hold at full
length, wait at most 2.1 seconds; then preempted requests hold the front
of the queue, and the last 40 wait 63–126 seconds. Figure 17.3 follows
the trace across rates.

![TTFT p50 and p95 against arrival rate on the H1 trace: the preempting
model through the measured points, the reserving model off them from
0.75 requests per second.](/assets/tinyperf-book/ch17-tails.svg)

*Figure 17.3. Time to first token as load passes the pool, H1 trace.
Solid: the model with vLLM's preempting scheduler; dashed: a scheduler
that reserves prompt + output; dots: measured. Below 0.75 requests per
second the pool never binds and the two agree.*

## What reservation costs

Table 17.6 prices the same cells with a scheduler that reserves prompt
plus output at admission. Each request's cap here is its actual output,
the most a reserving scheduler could know, so its only phantom room is
output still to come.

```
Table 17.6  The same cells under a reserving scheduler (prompt + output held from admission), model/measured
  sweep  rate TTFT p50    p95   TPOT  tok/s  peak decodes
        req/s                                 resv/engine
  A       1.5     1.15   1.00   0.85   1.01      33 / 43
  A       2.5     1.12   1.01   0.84   0.97      33 / 45
  A       3.5     1.19   1.01   0.81   1.02      33 / 46
  H1     0.75     1.01   4.90   0.98   0.98      59 / 68
  H1     1.25    16.51   0.65   0.69   1.17      62 / 80
  H1        2    13.33   0.77   0.68   1.17      63 / 86
  H2        2     1.89   1.57   0.73   0.91      37 / 61
  H2        3     1.50   1.34   0.71   0.91      37 / 63
  H2        5     1.37   1.26   0.68   0.90      38 / 74
  where the engine preempted, the reserving peak is 0.51-0.78 of the engine's
```

Reservation caps the batch at what the pool holds at full length, as
Table 17.4's fit column predicted: 33 on sweep A where the engine ran
43–46, 59–63 on H1 against 68–86, 37–38 on H2 against 61–74. Its steps
are too short, and TPOT reads 0.68–0.85 wherever the pool binds. The
queue is wrong in both directions: on H1 at 1.25 requests per second the
median wait is 16.5 times the engine's and the 95th percentile 0.65 of
it; at 0.75, where the engine never preempts, the 95th percentile reads
4.9, queued behind phantom room.

The throughput column is a prediction only: vLLM has no reserving mode.
On H1 at 1.25 and 2 requests per second the model says reservation would
deliver 17% more tokens per second, never recomputing (0.98 at 0.75); on
H2, 9–10% fewer, with far smaller batches. Over-admission is a bet on
the traffic, and the model prices both sides.

## Prefix caching

Chat traffic repeats itself: system prompts, conversation histories. A
*prefix cache* keeps a shared prefix's blocks in the pool; a request
that finds its prefix there prefills only the rest.

**The prefill becomes a chunk step.** A hit of P cached tokens leaves S
new ones, which attend to all P + S: the step chunked prefill already
prices (chapter 14), new tokens against an existing cache.
`StepLatencyModel.prefill_us` prices a hit as exactly that. `per_seq` is
one prompt's length, `shared` says every sequence has the same prefix,
and `prefill_overhead_us` is a per-prefill constant only the B200's
calibration carries:

```python
        if cached_tokens > 0:
            cached = min(cached_tokens, per_seq - 1)
            overhead = (self.calibration.prefill_overhead_us or 0.0) \
                if self.calibration else 0.0
            return self.chunk_us(max(1, n_seqs), cached, per_seq - cached,
                                 shared_prefix=cached if shared else 0) + overhead
```

**A batch can read the prefix once.** A decode step reads every
sequence's whole cache (chapter 6), so eight sequences sharing a prefix
read the same blocks eight times. *Cascade attention* reads the shared
blocks once for all the batch's queries and merges the result with each
sequence's own suffix. In a decode or chunk step of more than one
sequence, `build_llm_graph(shared_prefix=)` gives each sequence
attention over its own suffix and adds one over the shared part, folded
over the batch:

```python
        if shared:
            sq = g.BatchedMatMul("attn_qk_shared", q, batch=nkv_l, m=group * s * b_local,
                                 n=shared, k=hd, count=L_ctx, out_dtype=DType.FP32)
            ...                           # its softmax and PV, the same shape
```

Its batch is the KV heads alone, with every sequence's queries as rows,
so the shared keys are read once per KV head.

The evidence: Qwen3-8B on vLLM with prefix caching on, 4,096-token
prompts whose first P tokens are one shared system prompt, at batch 1
and 8; a warm-up request filled the cache, and the uncached control ran
with caching off. TTFT and TPOT are measured as in chapter 6, each cell
the median of three runs.

```
Table 17.7  Prefix caching on silicon: Qwen3-8B, vLLM on one RTX A6000, 4,096-token prompts, P tokens shared
  batch     P suffix  TTFT ms model/meas  vs P=0  TPOT ms no sharing  cascade
      1     0   4096    579.0      1.052    1.00    24.64      1.011    1.011
      1  2048   2048    319.6      1.081    0.55    24.71      1.008    1.008
      1  3584    512     97.2      0.977    0.17    24.70      1.009    1.009
      1  3968    128     47.2      0.710    0.08    25.14      0.991    0.991
      8     0   4096   4595.2      1.053    1.00    32.06      0.992    0.992
      8  2048   2048   2473.9      1.104    0.54    30.30      1.050    0.948
      8  3584    512    661.6      1.072    0.14    29.04      1.095    0.907
      8  3968    128    192.1      0.991    0.04    28.94      1.099    0.889
  TTFT with suffixes of 512 tokens or more: 0.98-1.10
```

The TTFT predictions were committed before the measurement: held out.
The grid has since been a test that later changes had to keep within
0.94–1.12 for suffixes of 512 tokens or more; those cells land within
0.98–1.10. Caching seven eighths of a prompt takes 83% off its TTFT at
batch 1, not 87.5%: the suffix still attends to the whole prompt. The
128-token suffix at batch 1 reads 0.71 of the 47.2 ms measured,
consistent with a fixed cost per prefill that this GPU's calibration has
no constant for (not profiled).

TPOT held the surprise. The committed predictions said sharing would
not change decode, and at batch 1 it did not. At batch 8 decode got
faster, from 32.1 to 28.9 ms with 3,968 of 4,096 tokens shared. Cascade
attention was added after this measurement, so its column is
in-sample: without sharing the model reads 5–10% high, with full
cascade 5–11% low: consistent with the engine reading the shared
prefix once for part of its work (not profiled). Table 17.7 reports
both bounds, and a test holds the measurement between them.

## Offloading: fitting is not running

Host memory is the next tier: keep part of the weights or the cache in
CPU memory and bring it over PCIe. Each device file carries its host
link's public rate per direction, `host_bw_gbps`: 32 GB/s for the A100
and RTX A6000, 64 GB/s for the H100 and B200. Since each step reads
every weight and the cache of every sequence it decodes, offloading
moves bytes out of the capacity model and into every step.
`StepLatencyModel` prices it as

```
link time = (offloaded weight bytes + offloaded bytes of the cache this step reads) / host_bw_gbps + one launch
step      = max(compute, link time)     or compute + link time, without overlap
```

```
Table 17.8  Offloading over the host link: Llama-3-70B bf16 on H100 SXMs, PCIe at 64 GB/s per direction (not measured)
  weights on the host, one H100, batch 1, context 4096
   fraction  GPU GB  host GB  fits  TPOT ms  host link ms
        0.0     141        0    no        -             -
        0.5      71       71   yes     1102          1102
        0.8      28      113   yes     1764          1764
  a second H100 instead (tp=2): TPOT 30 ms
  KV cache on the host, eight H100s (tp=8), context 131072, batch 8
   fraction max batch  TPOT ms  host link ms
       0.00        10     28.8           0.0
       0.25        13    167.8         167.8
       0.50        20    335.5         335.5
```

This is bytes over a datasheet rate. Llama-3-70B's 141 GB do not fit an
80 GB H100. With half the weights on the host they fit, and every token
waits 1.1 seconds for 71 GB to cross the link, 37 times the 30 ms of a
second GPU. The cache is the more deceptive target: at 131,072 tokens of
context, a quarter of it on the host fits 13 sequences instead of 10 and
makes each step 5.8 times longer. No fraction of an *active* cache is
cheap over PCIe. What an engine can offload cheaply is cache no running
step reads, such as a preempted request's blocks: a scheduling policy,
which exercise 3 builds.

## Where it breaks

- **One engine version on one GPU.** The fitted numbers come from vLLM
  0.15.1 on RTX A6000s.
- **Unpriced terms:** the 64-sequence floor, hidden states at very large
  budgets, and a cold compile cache.
- **Scope.** `simulate` derives the pool for one pipeline stage without
  attention data parallelism, context parallelism, offloading or 2:4
  sparsity, and uses chapter 6's rule elsewhere.
- **Sliding windows.** The cache per token counts gpt-oss-20b's windowed
  layers as growing with every token, as the engine's pool figure does;
  whether blocks leaving the window are freed is not modeled.
- **Preemption:** newest first with recompute only; no swap to the host,
  and no paging with prefix caching (`simulate` refuses it).
- **Prefix caching:** all or nothing on the declared prefix; tiny
  prefills miss low; cascade attention is a bracket.
- **Offloading:** unmeasured; bandwidth only, no contention on a shared
  host bridge.

## What you built

- vLLM's KV pool from first principles, including an MoE's workspace
  that is never freed: eight held-out dense configurations within 1%,
  and the workspace term at 1.0009–1.0205 on three held-out ones.
- Paged KV as vLLM schedules it, newest-first preemption and recompute
  included: six held-out saturation cells with TTFT and TPOT within
  0.96–1.03, preemptions within 9%, the peak batch within 3%. Reserving
  instead shrinks batches 22–49% and misprices the median wait up to
  16.5 times.
- Prefix hits as chunk steps (held-out TTFT 0.98–1.10), cascade
  attention as a bracket, offloading as bytes over the host link.

## Exercises

1. Start vLLM on a GPU you have, read its start-up log with
   `tools/measure_kv_pool.py`, and compare with `vllm_kv_pool`. Does
   CUDA report the same 0.9875 of the nameplate as on the A6000?
2. Add the forward's hidden states (tokens × hidden × 2 bytes) to the
   peak. Does it close the 32,768-token miss, and what does it do to the
   512-sequence cells?
3. Add a swap tier to `paged_schedule`: a preempted request's blocks go
   to the host at `host_bw_gbps` and come back instead of being
   recomputed. On H1 at 1.25 requests per second, which costs less?
4. Compose prefix caching with paged KV, so a preempted request's full
   blocks stay cached and part of its recompute becomes a hit. How much
   of Table 17.5's recompute share would that save on H2?

---

*[← Chapter 16: The mixed step]({% post_url 2026-09-29-tinyperf-16-the-mixed-step %}) · [Contents](/series/tinyperf/) · [Chapter 18: The knee →]({% post_url 2026-09-29-tinyperf-18-the-knee %})*
