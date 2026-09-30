---
layout: post
title: "Building tinyperf, chapter 9: Mixture of experts"
date: 2026-09-29 12:09:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/09-mixture-of-experts/
excerpt: "Many of today's largest language models replace each feed-forward layer with a set of smaller feed-forward networks, the experts, and a small router that sends each token to a few of them. A token pays for the math of its few experts, while the model holds the knowledge of all of them. For a performance model this raises one question with a surprisingly deep answer: what does an MoE layer cost to run?"
redirect_from:
  - /tinyperf/perf-modeling/2026/08/21/building-tinyperf-m11.html
  - /tinyperf/perf-modeling/2026/09/02/building-tinyperf-m43.html
  - /tinyperf/perf-modeling/2026/09/22/building-tinyperf-m65.html
  - /tinyperf/perf-modeling/2026/09/26/building-tinyperf-m76.html
  - /tinyperf/perf-modeling/2026/09/26/building-tinyperf-m77.html
  - /tinyperf/perf-modeling/2026/09/27/building-tinyperf-m78.html
---

*[Building tinyperf](/series/tinyperf/) · Part II: One model on one GPU · Code: [`tinyperf/nets/routing.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/nets/routing.py) and the MoE block of `build_llm_graph` in [`tinyperf/nets/transformer.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/nets/transformer.py) · Every table and the plot in this chapter come from `python3 book/scripts/ch09_moe.py`.*

Many of today's largest language models replace each feed-forward layer
with a set of smaller feed-forward networks, the *experts*, and a small
*router* that sends each token to a few of them. A token pays for the
math of its few experts, while the model holds the knowledge of all of
them. For a performance model this raises one question with a
surprisingly deep answer: what does an MoE layer cost to run?

The short answer has two halves. When a step carries thousands of
tokens, an MoE layer costs what its active parameters' math costs, like
a dense layer of that size. When a step carries a few dozen tokens, as
every decode step does, it costs the time to read the weights of the
experts the step touches. How many experts that is can't be derived from
the model's configuration. It depends on the router, on the text, and on
how the serving engine put the step together, and it has to be measured.

By the end of this chapter you will know:

- how an MoE layer becomes one grouped GEMM, and why its cost has two
  regimes;
- how to count the experts a step touches, and why the textbook formula
  counts up to twice as many as a real router touches;
- the four ways a serving engine's steps change that count: batches out
  of step, reply position, CUDA-graph padding and prompt chunks;
- how close the model gets on two MoE models, and what it still can't
  predict.

## An MoE layer

![Six tokens pass through a router that picks two of eight experts for
each; the picks land on five experts, whose weights the step
reads.](/assets/tinyperf-book/ch09-moe-layer.svg)

*Figure 9.1. An MoE layer. The router is a small GEMM followed by a
top-k: it scores every expert for every token and keeps the k best. Each
token's hidden state then goes through its k experts, each an ordinary
feed-forward network (a gate-and-up GEMM, an activation, a down GEMM),
and the outputs are summed with the router's weights. Here 12 picks land
on 5 of the 8 experts. The other 3 experts' weights are never read in
this step.*

Three models appear in this chapter:

```
Table 9.1  Three MoE models: parameters in billions
  model            layers hidden experts top-k expert width  total active
  mixtral-8x7b         32   4096       8     2        14336   46.7   12.9
  gpt-oss-20b          24   2880      32     4         2880   20.9    4.2
  qwen3-30b-a3b        48   2048     128     8          768   30.5    3.4
```

An expert holds `3 · hidden · width` parameters, its gate, up and down
matrices: 3 · 2880 · 2880 = 24.9 million for gpt-oss-20b, or 49.8 MB in
bf16. The counts in the table include both embedding tables; gpt-oss's
own model card leaves the input embedding out of its active count, which
gives its 3.6 billion.

Total parameters decide how much memory the model needs. Active
parameters decide how much math each token does. For serving, a third
number decides how long a decode step takes: how many experts the step
touches.

## Two regimes

A step that carries T tokens makes `T · k` (token, expert) pairs. The
pairs land on some number of distinct experts, and each touched expert
runs its rows through its own weights. tinyperf builds this as a single
*grouped GEMM*: chapter 3's GEMM with a batch dimension, where the batch
is the experts touched and each batch entry has that expert's rows.

```
batch  = experts touched
m      = rows per expert = T · k / experts touched
flops  = 2 · T · k · (expert parameters)
bytes  = experts touched · (expert parameters) · bytes per weight
```

The FLOPs follow the tokens; the weight bytes follow the experts
touched. Table 9.2 prices one layer of gpt-oss-20b's experts this way,
with chapter 3's model at the RTX A6000's fitted rates. It assumes
uniform routing, which the rest of this chapter will correct.

```
Table 9.2  One gpt-oss-20b layer's experts as a grouped GEMM, bf16, uniform routing, RTX A6000
  tokens  experts  rows/expert  weights MB   GFLOP       us  us/token  bound
       1        4            1         199     0.2    296.2    296.17  dram/dram
       8       21            2        1045     1.6   1525.7    190.71  dram/dram
      32       32            4        1593     6.4   2322.7     72.59  dram/dram
      64       32            8        1593    12.7   2325.9     36.34  dram/dram
     512       32           64        1593   101.9   2396.3      4.68  dram/dram
    2048       32          256        1593   407.7   3618.3      1.77  math/math
    8192       32         1024        1593  1630.7  14382.4      1.76  math/math
```

Read it from the top. One token touches its 4 experts and reads their
199 MB of weights to do 0.2 GFLOP of work: pure weight streaming. Eight
tokens touch 21 experts, five times the bytes, and the time grows with
the experts, not the tokens. From 32 tokens every expert is touched, and
the time stays flat: 512 tokens cost 3% more than 32. Between about
1,300 and 1,900 tokens the math catches up with the weights (chapter 3's
tile steps make the crossover ragged), and from 2,048 tokens the cost
per token settles at 1.8 µs, the price of the active parameters' math.
Between one token and 2,048, the cost per token falls by a factor of
about 170.

Prefill lives at the bottom of the table and decode at the top. In
decode, which is where a server spends most of its steps, the question
that decides the step's cost is how many experts it touches.

Before counting them, check the claim that an expert's cost at decode
is just its weights. Table 9.3 times the expert kernel of a serving
engine, vLLM, which every measurement in this chapter uses. The kernel
was timed inside real decode steps, launch by launch, and each launch
was paired with the routing that layer chose in that step.

```
Table 9.3  The engine's expert kernel, timed inside real decode steps: gpt-oss-20b bf16, RTX A6000
  batch  experts touched  gate-up us  us per expert   GB/s  of fitted rate
      1             4.00       191.9           48.0    692            1.00
      8            12.71       600.5           47.2    702            1.02
     24            14.98       733.5           49.0    675            0.98
     32            18.58       926.8           49.9    662            0.96
     33            19.13      1289.1           67.4    492            0.71
     48            20.82      1402.8           67.4    492            0.71
     64            22.75      1541.9           67.8    489            0.71
  one expert's gate-up weights: 33.2 MB = 48.0 us at the fitted 691 GB/s
  MXFP4 weights (the Marlin kernel): 0.70-0.87 of the fitted rate
```

Up to 32 sequences the kernel costs 48 µs per touched expert, which is
exactly the time to read one expert's gate-and-up weights at this GPU's
measured bandwidth. The kernel's time is the touched experts' weights and
nothing else. (The experts touched here are fewer than uniform routing
would predict; the next two sections are about why.)

At 33 sequences something else happens: the rate falls to 0.71 and
stays there. That is the kernel, not the routing. This GPU has no tuned
configuration for the kernel, so vLLM uses its default: row blocks of 16
while a launch carries no more tokens than there are experts, and row
blocks of 64 above that, which pads each touched expert's rows to 64.
Why the larger blocks stream more slowly is not something chapter 3's
tile model reproduces, so the model carries the measured result: 0.70 of
the rate for launches past the switch.

The same model also ships with 4-bit weights, in the MXFP4 format (4.25
bits per weight including its shared scales). vLLM runs those through a
different kernel, Marlin, which unpacks the weights to bf16 inside the
GEMM. It streams them at 0.70–0.87 of the rate; the model uses 0.80.

## Counting the experts a step touches

Suppose the router is fair: each token picks k distinct experts out of
e, every expert equally likely. A given expert is among one token's
picks with probability `k/e`. Across T tokens choosing independently, it
escapes all of them with probability `(1 − k/e)^T`. So the expected
number of experts touched is

```
touched = e · (1 − (1 − k/e)^T)
```

For one token this gives exactly k, as it must. For many tokens it
approaches e.

```
Table 9.4  Experts touched per layer by a decode step of B sequences
     B    gpt-oss-20b (32, top-4)    Qwen3-30B-A3B (128, top-8)
             uniform    balanced        uniform      balanced
     1           4.0           4            8.0             8
     2           7.5           8           15.5            16
     4          13.2          16           29.1            32
     8          21.0          32           51.6            64
    16          28.2          32           82.4           128
    32          31.6          32          111.8           128
    64          32.0          32          125.9           128
   128          32.0          32          128.0           128
```

The "balanced" columns are the rule you get if you imagine a router
that spreads its picks perfectly: `min(e, T · k)`. It is exact at one
token and wrong everywhere between. At 8 sequences on gpt-oss-20b it
says every expert is touched, where fair random picks reach 21.

> **Field note: the balanced rule.** The model's first MoE version
> used `min(e, T · k)`. On gpt-oss-20b's first run on real hardware,
> decode at batch 8 priced 50% high, where batch 1 was within 9% and
> batch 32 was 13–17% high. The balanced rule and a fair router agree at
> batch 1 (both say 4) and nearly agree at batch 32 (32 against 31.6);
> they differ most at batch 8 (32 against 21), and so did the error.
> The rest of batch 32's error was the real router's concentration,
> the subject of the next section.

The formula extends to a router with favourites. Give expert i a
popularity `p_i`, and let its probability of being among one token's k
picks be `q_i = min(1, c · p_i)`, with c chosen so the `q_i` sum to k.
Then `touched = Σ 1 − (1 − q_i)^T`. It still gives exactly k for one
token. tinyperf keeps this form: a model may set a skew for a Zipf
popularity law, and one that sets none, like any model nobody has
measured, gets uniform routing. Measured models read a table instead,
for reasons the next section makes plain. Here is the code:

```python
def law_touched(n_experts, top_k, assignments, skew=0.0):
    k = min(top_k, n_experts)
    t = assignments / k
    if skew <= 0:
        return n_experts * (1 - (1 - k / n_experts) ** t)
    return sum(1 - (1 - q) ** t for q in _inclusion(n_experts, k, skew))


def distinct_experts(p, n_experts, assignments, skew=None):
    law_skew = p.routing_skew if skew is None else skew
    table = p.routing_distinct if (n_experts == p.n_experts and skew is None) else None
    if not table:
        return law_touched(n_experts, p.top_k, assignments, law_skew)
    t = assignments / min(p.top_k, n_experts)
    pts = sorted((float(k), float(v)) for k, v in table.items())
    if t <= pts[0][0]:
        return pts[0][1] * t / pts[0][0]
    for (t0, d0), (t1, d1) in zip(pts, pts[1:]):
        if t <= t1:
            x, x0, x1 = math.log2(t), math.log2(t0), math.log2(t1)
            return d0 + (d1 - d0) * (x - x0) / (x1 - x0)
    return max(pts[-1][1], law_touched(n_experts, p.top_k, assignments, law_skew))
```

`_inclusion` finds c by bisection. A measured table is read in tokens
per step, interpolated in log2 between entries, with the law beyond its
last entry.

## Real routers touch fewer experts, and which ones depends on the text

Counting a real router's choices takes an instrument: hooks on the
router modules inside the serving engine, recording each token's k
experts in every layer of every step. Table 9.5 shows what they read.

```
Table 9.5  Measured against uniform: experts touched per layer by a decode step of B sequences
  Qwen3-30B-A3B (of 128), random-token prompts, sampled
  B                 2      4      8     16     32     64    128
  measured       12.3   22.7   33.0   43.0   58.5   77.0   93.6
  uniform        15.5   29.1   51.6   82.4  111.8  125.9  128.0
  gpt-oss-20b (of 32), by workload, greedy
  B                 2      4      8     16     32     64    128
  chat            6.4    9.3   13.2   16.5   19.3   21.6   23.8
  random          6.1    9.4   13.1   17.4   21.3   24.4   26.5
  prose           7.1   10.9   14.6   20.7   22.6   26.5   28.6
  code            7.1   11.3   16.7   21.8   25.9   29.0   30.3
  mixed           7.2   12.3   18.3   24.0   28.0   30.2   31.1
  uniform         7.5   13.2   21.0   28.2   31.6   32.0   32.0
  Qwen3-30B-A3B, B=8, six prompt sets: 26.2 to 41.7 experts
```

![Distinct experts touched by a gpt-oss-20b decode step against batch
size, for five workloads and for uniform routing.](/assets/tinyperf-book/ch09-touched-experts.svg)

*Figure 9.2. gpt-oss-20b's router against uniform routing. Every
workload touches fewer experts than uniform routing at every batch above
one, and the workload decides by how much.*

The gpt-oss-20b workloads: *chat* is news articles with a request to
summarize them, the model's reply decoded; *prose* is WikiText-103;
*code* is Python sources; *random* is random token ids; *mixed* puts
prose, code and chat in one batch.

Three things stand out.

- **Real routers concentrate.** At 32 sequences Qwen3-30B-A3B's router
  touches 58.5 experts per layer. Uniform routing says 112. A decode
  step reads half the expert weights the textbook formula charges it
  for.
- **The text decides.** On gpt-oss-20b at 32 sequences, chat replies
  touch 19.3 experts and a batch mixing prose, code and chat touches
  28.0. Tokens about similar things go to similar experts. A batch that
  mixes subjects touches the union of their experts.
- **Batches vary.** Six batches of 8 random-token prompts on Qwen3 touch
  anywhere from 26 to 42 experts. A table is a mean. A single batch can
  be far from it.

A router's choices also change as a reply goes on:

```
Table 9.6  Routing drifts over a reply: gpt-oss-20b, 512-token prompts, experts touched per decode step
  workload, B    steps 1-8   9-32  33-128  all 128
  chat, 16             5.0   12.7    18.4     16.5
  random, 8            8.2   10.3    11.2     10.9
  mixed, 8            16.2   17.6    18.7     18.3
  random, 8, by prompt length: 512 tokens 10.9, 2048 tokens 14.5, 8192 tokens 13.4; Table 9.5 averages them: 13.1
```

Sixteen chat replies touch 5 experts over their first 8 tokens and 18
over tokens 33 to 128. Every reply opens the same way, then goes its
own way. So a routing table has to be measured the way the step times
it describes are measured: over the same decode steps. The prompt's
length matters too, which is why Table 9.5's 13.1 for random tokens at
batch 8 is the average of 10.9, 14.5 and 13.4 over three lengths.

> **Field note: count over the steps you time.** One routing
> measurement counted the experts at each prompt's last position, the
> token just before decoding starts. For 8 random-token prompts on
> gpt-oss-20b it read 9.7. Averaged over the 128 decode steps a
> time-per-token measurement covers, the same prompts touch 13.1. The
> 9.7 looked like a direct measurement, so for a while the kernel was
> blamed for the difference: fitted constants said it streamed weights
> at 0.70 of the memory rate, then slower still "in company". Timed
> launch by launch against the right routing (Table 9.3), it ran at the
> full rate all along. Chapter 22 tells the whole story.

## What the step contains

A serving engine doesn't decode fixed batches that start together.
Three of its habits matter here; chapters 14–16 cover each in full.

- Requests arrive and finish at any time, so the sequences in a step are
  at different points in their replies. Measurements read off a server
  handling a stream of requests are called *online* below.
- A decode step runs as a *CUDA graph*: a recorded sequence of kernel
  launches that the GPU replays as one, which removes most of the launch
  cost (chapter 4). vLLM records a graph for each of a set of batch
  sizes, such as 1, 2, 4, 8, 16, 24 and 32, and runs a step of B decodes
  in the smallest graph with at least B rows.
- With *chunked prefill*, a new prompt is split into chunks that ride
  along in decode steps, so one step can carry decodes and part of a
  prompt.

Each of these changes how many experts a step touches.

**Out of step.** Sequences at different points in their replies have
less alike tokens, and share fewer experts:

```
Table 9.7  What an online step adds: experts touched per layer, read off the engine as it serves
  decodes                             8     16     32     64
  gpt-oss-20b, a batch in step     13.1   17.4   21.3   24.4
  gpt-oss-20b, online              14.5   20.0   24.3   29.1
  Qwen3-30B-A3B, a batch in step   33.0   43.0   58.5   77.0
  Qwen3-30B-A3B, online            35.4   47.4   62.5   87.3
```

An online step touches 11–19% more experts on gpt-oss-20b and 7–13%
more on Qwen3. These counts come from the engine itself as it serves,
read inside its CUDA graphs. vLLM can record each row's routed experts
into a buffer that is part of the captured graph.

**Reply position.** Table 9.6 showed routing spreading over a reply.
In a server, a step's position in the replies varies with the
workload: short replies keep every step early. Grouping online steps by
their decodes' median position in their replies:

```
Table 9.8  Early in their replies, sequences crowd: gpt-oss-20b online, experts touched
  decodes   median reply position under 64   256 or more
        4                              9.1          12.4
       12                             14.8          22.6
       16                             16.7          24.9
```

At 12 decodes, a step early in its replies touches 15 experts and a step
late in them touches 23. The model reads a factor off the median
position and applies it to the decodes beyond the first; one token
always touches its k. The factor is 1 near position 128, where the online
table was measured, 0.47 at position 10 and 1.52 at about 400. So far
only gpt-oss-20b has one.

```python
def effective_decodes(n, reply_pos, by_position):
    return round((1 + (n - 1) * _log2_interp(by_position, max(reply_pos, 1))) * 4) / 4
```

The builder then scales the step's count by the table's ratio between
the effective and the real decodes.

**Padding.** A decode step with 9 sequences runs the CUDA graph captured
for 16. The other 7 rows are not empty: they hold whatever tokens were
left in the engine's input buffer, usually the tail of the last prompt.
The router routes them like any other token.

![A decode step of 9 sequences in a 16-row CUDA graph: the 7 stale rows
route to experts of their own, so the step touches more experts than a
full step of 16.](/assets/tinyperf-book/ch09-padded-step.svg)

*Figure 9.3. A padded step reads more expert weights than a full one.*

```
Table 9.9  Padded rows route too: Qwen3-30B-A3B, CUDA-graph decode steps
  padded rows                           1    2    3    4    5    6    7
  experts they touch on their own       8   14   19   22   27   30   33
  decodes in the step                     8      9     12     15     16
  graph size                              8     16     16     16     16
  experts, the real rows               35.4   37.6   42.1   49.5   47.4
  experts, as the step runs            35.4   56.4   53.0   53.1   47.4
  engine step time, ms                 21.3   30.4   31.4   29.7   29.5
```

On their own, p stale rows touch about as many experts as p sequences
would. But they are prompt text and share few experts with the decodes:
the 7 stale rows add 18.8 experts to the step of 9, where 7 more decodes
would add 9.8. So a step of 9 decodes touches 56.4 experts, more than a
full step of 16 (47.4). The step times come from a separate run of the
same workload, since the routing trace slows every step: the steps of
9, 12 and 15 decodes take 29.7–31.4 ms, the full 16 takes 29.5. A model
that prices only the real rows says the step of 9 is the cheaper one.
The model instead reads a table measured by real decodes, padded rows
included.

**Prompt chunks.** A step that carries a chunk of a new prompt has many
tokens, but they come from one text:

```
Table 9.10  A prompt chunk routes to a concentrated set: steps carrying a prefill chunk
  Qwen3-30B-A3B, tokens in the step      110   199   362   643   896  1259  1887
  experts touched (of 128)                89    96   101   102   100   102   111
  gpt-oss-20b, tokens in the step        350   681   891  1327  2048
  experts touched (of 32)               28.0  28.1  28.6  26.9  28.5
  gpt-oss-20b, tokens in the step         513-768   769-1024  1025-1536
    real rows, beside <= 8 decodes           24.2       25.0       25.2
    real rows, beside > 8 decodes            27.2       27.5       27.8
  gpt-oss-20b, one prompt of 256-8192 tokens, 648 layer launches: the busiest expert takes a median 99% of the tokens (lower quartile 97%)
```

Uniform routing says every one of these steps touches every expert.
They touch about 100 of Qwen3's 128 and 27–29 of gpt-oss's 32. The last
line shows how concentrated a single prompt is: in almost every layer,
one of gpt-oss-20b's experts is among the four picks of nearly every
token. The mix matters too: at the same number of tokens, a chunk beside
8 decodes or fewer touches 24–25 of gpt-oss's experts, and one beside
more decodes 27–28.

So the model counts experts from tables keyed by what the step
contains: decodes, tokens for chunk-carrying steps, a padded table by
real decodes, and a reply-position factor. In the builder, without
expert parallelism (chapter 12 covers that branch), with comments and a
bookkeeping line trimmed:

```python
            e_local = p.n_experts
            assign = tokens * p.top_k
            active_local = touched_count(p, e_local, assign, routing_skew)
            padded = moe_real_tokens is not None and moe_real_tokens < tokens and p.routing_padded
            if padded:
                active_local = max(1, min(e_local, int(padded_touched(p.routing_padded, moe_real_tokens) + 0.5)))
            if moe_route_decodes is not None and p.routing_distinct and e_local == p.n_experts:
                real_rows = moe_real_tokens if padded else tokens
                scale = (distinct_experts(p, p.n_experts, moe_route_decodes * p.top_k, routing_skew)
                         / distinct_experts(p, p.n_experts, real_rows * p.top_k, routing_skew))
                active_local = max(1, min(e_local, int(active_local * scale + 0.5)))
            m_e = max(1, math.ceil(assign / active_local * moe_imbalance))
```

`touched_count` is `distinct_experts` rounded to whole experts.
`moe_real_tokens` is the real decodes in a padded step, where
`padded_touched` reads the padded table. `moe_route_decodes` is the
effective decodes from the reply position. `active_local` becomes the
grouped GEMM's batch, and `m_e` its rows per expert. `moe_imbalance`
scales the rows for a hot expert. It matters when the expert math binds,
in prefill and under expert parallelism (chapter 12), and not in
weight-bound decode.

## Rows per expert, and the kernel

The count decides the bytes. The rows per expert, `T · k / touched`,
decide how the kernel runs, and the model carries two facts about that,
both measured on the engine.

The first is Table 9.3's block-size switch.

The second is about prefill. A lone prompt sends nearly all its tokens
to one expert (Table 9.10), so its rows per expert are far from even.
The MXFP4 kernel picks its row-block size from the mean rows per expert
across all 32 experts, a spread that no expert actually has, and runs
up to 21% slower for it. Chapter 4's calibrated tier, the model's
constants fitted to one GPU and software stack, stores that kernel's
measured rate by tokens per launch, timed on the engine's own routing.
The untuned bf16 kernel's prefill is a fitted constant, 0.44 of dense
efficiency, which belongs to one engine version on one GPU. Chapter 22
is about what constants like it can hide.

## How close is it?

**Decode steps, fixed batches.** Qwen3-30B-A3B served across two RTX
A6000s with tensor parallelism (tp=2: each GPU holds half of every
weight matrix; chapter 11), each batch decoding side by side and timed
step by step inside the engine. Nothing in the model was fitted to these
timings: the routing table came from a separate, routing-only
measurement, and the kernel constants from gpt-oss-20b.

```
Table 9.11  Qwen3-30B-A3B decode steps on the engine's clock, tp=2 on two RTX A6000s
  batch  measured ms  model/measured  with uniform routing
      1         9.27           1.026                 1.026
      8        19.04           1.073                 1.407
     16        28.27           0.971                 1.431
     32        38.86           0.972                 1.435
     48        50.05           0.935                 1.288
     64        55.20           0.978                 1.275
```

With the measured router the steps land at 0.94–1.07. With uniform
routing they read 28–44% high from batch 8 up, because the model streams
experts the router never picked.

**Real text.** gpt-oss-20b on one RTX A6000, fed prompts from real text,
with its time per output token (TPOT) measured at batches of 8 to 64:

```
Table 9.12  gpt-oss-20b on real text, one RTX A6000: decode step (TPOT) model/measured, batch 8-64
  run                   workload table  random-token table       uniform
  bf16, mixed text           1.00-1.04           0.77-0.86     1.08-1.16
  bf16, chat replies         0.96-1.01           0.97-1.09     1.38-1.51
  MXFP4, mixed text          1.04-1.07           0.85-0.93     1.10-1.18
```

Only the workload's own table lands near 1. The random-token table
reads 23% low on mixed text, which touches more experts, and up to 9%
high on chat, which touches fewer. Uniform routing reads high
everywhere, by up to 51% on chat. The right table matters in both
directions.

**Online, with padding.** On a Qwen3-30B-A3B server under load, each of
12,470 decode steps was priced at its recorded batch and context and
compared with the engine's own time for it. These numbers were recorded
when measured, not recomputed by the current model:

```
Recorded  Qwen3-30B-A3B online decode steps, model/engine median, as recorded at measurement time
  with the padded table: unpadded 1.013, padded 0.991 (4659 and 7811 steps)
  without it:            unpadded 1.007, padded 0.908
```

## Where it breaks

- **Synthetic prompts.** Every table read off a server under load came
  from benchmark prompts of random tokens. Real text was measured only
  in fixed batches (Table 9.12). A real server's mix of subjects has to
  be measured before its tables can be trusted.
- **New models and workloads.** Without a measurement the model falls
  back to uniform routing. That is an upper bound on weight traffic:
  safe for capacity planning, pessimistic for decode latency by as much
  as Table 9.12's last column.
- **Layers.** The tables average all layers. Some layers may route more
  concentrated than others.
- **Composition.** The tables are keyed by tokens, decodes, padding and
  reply position, not by everything that matters. A prompt's tail
  beside a few decodes still touches fewer experts than its key says.
  On gpt-oss-20b with short replies, some early mid-size batches and
  padded steps still price low.
- **Kernels.** The streaming rates and the prefill constant belong to
  one engine version on one GPU with untuned kernel configurations.
- **Expert parallelism.** When experts are spread across GPUs, the count
  per GPU, the busiest GPU's rows and how tokens travel all matter.
  Chapter 12 covers them.

## What you built

- An MoE layer as one grouped GEMM: FLOPs from the tokens and top-k,
  bytes from the experts touched, and chapter 3's model to price it.
- Two regimes: weight streaming, proportional to the experts touched, up
  to about a thousand tokens per step on this GPU; math, proportional to
  the active parameters, beyond.
- The count of experts touched: exact for a fair router,
  `e · (1 − (1 − k/e)^T)`, and measured tables for real ones, keyed by
  workload and by what the step contains.
- Evidence: decode steps within 0.94–1.07 on a held-out model where
  uniform routing reads 1.28–1.44, and real-text decode within
  0.96–1.07 when the workload's table is used.

## Exercises

1. Measure a routing table for your own workload with
   `tools/measure_moe_routing.py`, using prompts from your logs. Where
   does it fall in Table 9.5?
2. Some models add *shared* experts that every token uses, beside the
   routed ones. Extend `law_touched` and the grouped GEMM for them. At
   what batch do they stop mattering for decode?
3. If the engine zeroed its padded rows instead of leaving stale tokens,
   every padded row would be the same token and would touch k experts at
   most. Using Table 9.9, how much faster would a step of 9 decodes be?
4. Price a step's experts as the union of two independent sets: its
   decodes' (from the decode table) and its chunk's (from the chunk
   table). Does that explain why a prompt's tail beside few decodes
   touches fewer experts than a chunk beside many?

---

*[← Chapter 8: Attention variants]({% post_url 2026-09-29-tinyperf-08-attention-variants %}) · [Contents](/series/tinyperf/) · [Chapter 10: Precision and sparsity as passes →]({% post_url 2026-09-29-tinyperf-10-precision-and-sparsity %})*
