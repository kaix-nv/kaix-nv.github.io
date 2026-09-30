---
layout: post
title: "Building tinyperf, chapter 19: Speculative decoding"
date: 2026-09-29 12:19:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/19-speculative-decoding/
excerpt: "A decode step reads every weight of the model to give each sequence one token (chapter 6). For Qwen3-8B on an RTX A6000, tinyperf's calibrated model prices that step at 24.2 ms for one sequence (chapter 14 measured a 24.4 ms TPOT), and only a faster memory makes that one stream faster. Speculative decoding changes what a step produces. A cheap draft model guesses the next few tokens, the large target model checks them all in one step and keeps the ones it agrees with, and one expensive step yields several tokens. When is that true, how far ahead should the draft guess, and what does a performance model need to know that it can't derive?"
redirect_from:
  - /tinyperf/perf-modeling/2026/08/28/building-tinyperf-m17.html
  - /tinyperf/perf-modeling/2026/08/29/building-tinyperf-m27.html
  - /tinyperf/perf-modeling/2026/08/30/building-tinyperf-m28.html
---

*[Building tinyperf](/series/tinyperf/) · Part IV: Serving · Code: [`tinyperf/spec_decode.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/spec_decode.py), `expected_tokens` and `SpecStepLatencyModel`, and [`tinyperf/mtp.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/mtp.py), `MTPStepLatencyModel` · Every table and figure in this chapter comes from `python3 book/scripts/ch19_speculation.py`.*

A decode step reads every weight of the model to give each sequence one
token (chapter 6). For Qwen3-8B on an RTX A6000, tinyperf's calibrated
model prices that step at 24.2 ms for one sequence (chapter 14 measured
a 24.4 ms TPOT), and only a faster memory makes that one stream
faster. Speculative decoding changes what a step produces. A cheap
*draft* model guesses the next few tokens, the large *target* model
checks them all in one step and keeps the ones it agrees with, and one
expensive step yields several tokens. When is that true, how far ahead
should the draft guess, and what does a performance model need to know
that it can't derive?

The short answer: a cycle of k drafted tokens yields
`(1 − α^(k+1)) / (1 − α)` tokens on average, where α is the chance that
each guess is accepted. It costs k draft steps plus one target step
over k+1 positions, and while that step is memory-bound it costs about
one decode step. With Qwen3-0.6B drafting four tokens for Qwen3-8B at
α = 0.8, the model projects 3.36 tokens in 38 ms, 2.14 times plain
decoding. Three things take the gain away: large batches, which push
the verify step past the ridge point; a draft whose own KV cache grows
with the context; and acceptance that falls with position, which makes
deep drafts a waste. A model's own prediction head prices at about 5% of a
target step. No speculation number in this chapter is measured, and the
acceptance is an input the model can't derive.

By the end of this chapter you will know:

- how a draft-and-verify cycle works, and why its output is exactly the
  target's;
- the expected tokens per cycle, and the speedup and break-even
  acceptance that follow from the draft's cost;
- why verifying is nearly free while the step is memory-bound, and the
  regime map where it stops;
- why a small draft costs more than its parameter count suggests;
- what a multi-token-prediction (MTP) head costs as a draft;
- why acceptance that decays with position makes the best depth
  shallow;
- what evidence there is, and what is missing.

## Draft, then verify

Speculative decoding runs in *cycles* (Figure 19.1). The draft runs k
ordinary decode steps and proposes k tokens. The target then runs one
forward pass over them. As a prefill does for a prompt, the pass gives
the target's next-token distribution at every position: after the
context, after the first draft, and so on to after the last, k+1 in
all. This is chapter 14's *chunk step*, m new query rows per sequence
against the existing KV cache, with m = k+1.

![A timeline: four target decode steps emit four tokens by 96.9 ms;
two draft-and-verify cycles emit three and five tokens by
75.9 ms.](/assets/tinyperf-book/ch19-timeline.svg)

*Figure 19.1. Plain decoding against two cycles of speculative
decoding, to scale. In the first cycle the target accepts two drafts
and rejects the third, replacing it with a token of its own; the fourth
draft, built on the rejected third, and the target's prediction after
it are thrown away. In the second it accepts all four and adds a
fifth. Every cycle emits the accepted drafts plus one.*

The target decides left to right by *rejection sampling*. With p its
distribution at a position and q the draft's, the drafted token x is
accepted with probability min(1, p(x)/q(x)). At the first rejection the
target samples a replacement from what is left, `max(0, p − q)`
normalized, and the cycle ends. If all k drafts pass, it samples one
more token from the last position. Leviathan, Kalman and Matias, and
Chen and colleagues (both 2023) proved that the emitted tokens are
distributed exactly as the target's own samples: speculation changes
the speed, never the output. Under greedy decoding the rule reduces to
"accept if the draft picked the target's top token".

The chance that a position is accepted is `Σ min(p(x), q(x))`, one
minus the *total-variation distance* between the two distributions.
Its average is the *acceptance rate* α.

## How many tokens a cycle yields

Assume every drafted position is accepted with the same probability α,
independently of the others (*i.i.d.*: independent and identically
distributed). Drafted position i is kept only if positions 1 to i all
are, with probability α^i, and the cycle always emits one token of the
target's own. So the expected tokens per cycle are

```
E = 1 + α + α² + ... + α^k = (1 − α^(k+1)) / (1 − α)
```

At α = 0 a cycle yields one token, like a decode step; at α = 1, k+1.
As k grows E approaches `1 / (1 − α)`: at α = 0.8 even endless drafting
yields five tokens a cycle, because survival to position i shrinks
geometrically.

```
Table 19.1  Expected tokens per cycle, E = (1 - alpha^(k+1)) / (1 - alpha): i.i.d. acceptance, k drafted tokens
  alpha   k = 1   k = 2   k = 4   k = 8  k -> inf
   0.60    1.60    1.96    2.31    2.47      2.50
   0.80    1.80    2.44    3.36    4.33      5.00
   0.90    1.90    2.71    4.10    6.13     10.00
```

`expected_tokens` sums the series term by term, so that one loop also
handles the decaying acceptance of a later section (docstring and a
comment trimmed):

```python
def expected_tokens(alpha: float, k: int, decay: float = 1.0) -> float:
    assert 0.0 <= alpha <= 1.0 and k >= 1 and 0.0 < decay <= 1.0
    total, reach = 1.0, 1.0          # the verified token always lands
    for i in range(1, k + 1):
        reach *= alpha * decay ** (i - 1)
        total += reach
    return total
```

## What a cycle costs

A cycle runs k draft steps and one verify step:

```
time per token = (k · t_draft + t_verify) / E
speedup        = E / (k · c + v),    c = t_draft / t_decode,  v = t_verify / t_decode
```

The speedup is plain decoding's time per token over speculation's.
Speculation pays when `E > k·c + v`: the cheaper the draft, the less
often it has to be right. `SpecStepLatencyModel` wraps two of chapter
14's step models, one per model, and prices exactly this (docstrings,
type annotations and the constructor's bookkeeping trimmed):

```python
class SpecStepLatencyModel:
    def __init__(self, target, draft, device, alpha, k=4, tp=1, ep=1,
                 methodology=Methodology.PROJ, decay=1.0, stack="graph"):
        self.target = StepLatencyModel(target, device, tp=tp, ep=ep,
                                       methodology=methodology, stack=stack)
        # the draft runs replicated (tp=1): it is small, and sharding a
        # tiny model would drown it in collective latency
        self.draft = StepLatencyModel(draft, device, methodology=methodology, stack=stack)
        ...

    def decode_us(self, batch, kv_len, shared_prefix=0, real=None):
        cycle_us = (self.k * self.draft.decode_us(batch, kv_len, shared_prefix)
                    + self.target.chunk_us(batch, kv_len, self.k + 1, shared_prefix))
        return cycle_us / expected_tokens(self.alpha, self.k, self.decay)
```

`decode_us` returns the time per emitted token. The class has the step
model's interface, so chapter 14's `simulate` accepts it, and every
sequence then advances one token per "step" at that price: a
*mean-field* model, in which every sequence gets the average yield.
(`prefill_us`, not shown, prefills both models.)

Here is one cycle on the book's usual set-up, Qwen3-8B on the RTX A6000
at its fitted rates with kernels in a CUDA graph. The draft is
Qwen3-0.6B, the smallest model of the family: a draft must share the
target's vocabulary, and these two share a tokenizer.

```
Worked example  One cycle: Qwen3-8B with a Qwen3-0.6B draft, RTX A6000, batch 1, context 1024, k = 4
  target decode step:            24.21 ms, 1 token
  draft steps, 4 x 3.41 ms:      13.64 ms
  verify step, 5 positions:      24.32 ms (1.005 decode steps)
  cycle:                         37.96 ms
  alpha 0.8: E = 3.36 tokens, 11.29 ms per token, 2.14x plain decoding
  SpecStepLatencyModel.decode_us gives the same within 1e-9 us
  break-even: E = cycle / decode step = 1.57 tokens, alpha = 0.37
```

Checking five positions costs what producing one does. The four drafts
cost 0.56 of a target step, so a cycle must yield 1.57 tokens to break
even, which takes α = 0.37.

```
Table 19.2  Speedup over plain decoding at batch 1, by acceptance and depth: Qwen3-8B + Qwen3-0.6B, RTX A6000, context 1024
             alpha   k = 1   k = 2   k = 3   k = 4   k = 8  best k, 1-16
               0.0    0.88    0.78    0.70    0.64    0.47       1: 0.88
               0.3    1.15    1.09    0.99    0.91    0.67       1: 1.15
               0.5    1.32    1.37    1.32    1.24    0.94       2: 1.37
               0.6    1.41    1.53    1.53    1.47    1.16       2: 1.53
               0.7    1.50    1.71    1.78    1.77    1.50       3: 1.78
               0.8    1.59    1.91    2.07    2.14    2.03       5: 2.16
               0.9    1.68    2.12    2.41    2.61    2.87       8: 2.87
  break-even alpha    0.13    0.23    0.30    0.37    0.53
```

Every extra draft costs a step whether or not its token survives, so
the break-even acceptance rises with depth, from 0.13 at k = 1 to 0.53
at k = 8, and the best depth rises with acceptance.

## A small draft is not a small fraction

Qwen3-0.6B has 7.3% of Qwen3-8B's parameters, yet its step prices at
14% of the target's:

```
Table 19.3  A draft step's price over the target's decode step (c), RTX A6000 in a CUDA graph
  target / draft               params d/t   KV d/t  batch    ctx 256   ctx 1024   ctx 4096  ctx 16384
  qwen3-8b / qwen3-0.6b              7.3%    77.8%      1      0.14       0.14       0.16       0.22
                                                       32      0.18       0.28       0.48*      0.66*
  llama2-7b / tinyllama-1.1b        16.3%     4.3%      1      0.19       0.19       0.17       0.14
                                                       32      0.16       0.11       0.07*      0.05*
  d/t: the draft's over the target's; * both models' weights and caches exceed 90% of 48 GB
  qwen3-8b / qwen3-0.6b: KV bytes per token 147,456 and 114,688
  llama2-7b / tinyllama-1.1b: KV bytes per token 524,288 and 22,528
  qwen3-0.6b step, batch 1, context 1024: layer GEMMs 1.88 ms, LM head 0.47, attention 0.47, small ops 0.60; 311 kernels x 3.5 us = 1.09 ms of launches in all
```

Three costs don't shrink with the parameter count. A draft launches
as many kernels per layer as the target, 311 in all, and their
launches are a third of its 3.41 ms. Its LM head spans the same
vocabulary. And it reads a KV cache of its own at every draft step.
Qwen3-0.6B keeps the same eight KV heads of 128 as Qwen3-8B, over 28
layers instead of 36, so its cache per token is 78% of the target's.
From batch 1 to 32 at 1,024 tokens c doubles, 0.14 to 0.28, and it
keeps rising with the context into cells that need more than 48 GB.
TinyLlama for Llama-2-7B goes the other way: Llama-2-7B gives every
query head its own KV head (chapter 6), the draft's cache is 4.3% of
the target's, and its relative cost falls as the context grows. So c is
not a constant: it depends on the batch, the context and the two
models' attention designs.

## When verifying is free

A verify step of b sequences reads the same weights and caches as a
decode step, but runs every GEMM on b·(k+1) rows instead of b. A weight
GEMM on M rows does about M FLOPs per byte (chapter 6), so verification
stays memory-bound, and costs about one decode step, while b·(k+1) is
below the A6000's fitted ridge point of 168 (in practice the 128-row
tile edge). Past it, GEMM time grows with the rows. `chunk_us` prices
the step, with the LM head on all k+1 rows and the context rounded up
to 256.

```
Table 19.4  What verifying costs: a chunk step of k+1 positions over a decode step, Qwen3-8B, RTX A6000, context 1024
  batch  decode ms   k = 1   k = 2   k = 3   k = 4   k = 7
      1      24.21    0.99    1.00    1.00    1.00    1.01
      8      26.23    0.99    1.01    1.02    1.09    1.09
     16      28.18    1.01    1.09    1.07    1.10    1.17
     32      32.49    1.04    1.10    1.12    1.31    1.74
     64      41.56    1.05    1.31    1.53    1.88    2.27
    128      58.85    1.32    1.60    1.84    2.16    3.16
             k + 1       2       3       4       5       8
  ridge point, fitted: 168 FLOP/byte
  batch 8, k = 4, decode -> verify, ms: GEMMs 21.32 -> 23.71, attention 2.21 -> 1.91, LM head 1.86 -> 1.87, small ops 0.85 -> 1.17
  batch 64, k = 4, decode -> verify, ms: GEMMs 23.25 -> 55.98, attention 15.00 -> 14.42, LM head 1.89 -> 3.50, small ops 1.42 -> 4.05
```

Up to 128 rows a verify step costs at most 17% more than a decode step.
The 9% at 40 and 48 rows (batch 8 at k = 4, batch 16 at k = 2) is
cuBLAS's jump above 32 rows (chapter 15). At batch 64, four drafts put
the GEMMs on 320 rows and more than double them, 23.25 to 55.98 ms, and
the step costs 1.88 decode steps. A whole
step stays below k+1 decode steps because attention reads the same
cache for any number of queries; only the GEMMs past the ridge follow
the rows. (The attention figures in the footer show a flaw: Where it
breaks.)

At α = 0.8 the rising cost eats the gain:

```
Table 19.5  Speedup over plain decoding at alpha 0.8: Qwen3-8B + Qwen3-0.6B, RTX A6000, context 1024
  batch      c   k = 1   k = 2   k = 3   k = 4   k = 7  best k, 1-16
      1   0.14    1.59    1.91    2.07    2.14    2.09       5: 2.16
      8   0.18    1.54    1.78    1.90    1.86    1.78       3: 1.90
     16   0.22    1.47    1.60    1.71    1.70    1.54       3: 1.71
     32   0.28    1.37    1.48    1.51    1.38    1.13       3: 1.51
     64   0.36    1.28    1.20    1.13    1.01    0.87       1: 1.28
    128   0.46    1.01    0.97    0.92    0.84    0.65       1: 1.01  *
  c: a draft step over a decode step
  * the target's and the draft's caches don't fit in 90% of 48 GB (the simulator charges only the target's)
  sequences of 1024 tokens that fit: target alone, 177 by max_batch and 192 in simulate's pool (vllm_kv_pool, 196,704 tokens); with the draft's weights and cache by the 90% rule, about 95
```

Both terms of the denominator grow with the batch: v through the rows,
c through the draft's cache. The gain falls from 2.16 at one sequence
to 1.28 at 64, and the best depth from 5 to 1. The right number of
drafts belongs to the operating point, not to the pair of models.
Figure 19.2 adds the context, for both pairs.

![Two heat maps of speedup by batch (1 to 128) and context (256 to
16,384): the Qwen3 pair falls below 1 at large batch and long context;
the Llama pair rises with context.](/assets/tinyperf-book/ch19-regimes.svg)

*Figure 19.2. The regime map at k = 4 and α = 0.8; dashed, faded
cells need more than 48 GB with both caches. Left, Qwen3-8B with a
Qwen3-0.6B draft: 2.17 at one sequence and 256 tokens, falling with
batch and context. The only loss that fits is batch 128 at 256 tokens
(0.85); the losses at 4,096 tokens and more need more memory. Right,
Llama-2-7B with a TinyLlama-1.1B draft: in cells that fit, the gain
grows mildly with the batch, 1.92 to 2.07 at 1,024 tokens and 1.99 to
2.25 at 4,096.*

At long context both steps are mostly cache reads, and a cycle spreads
the target's read over E tokens. That helps only if the draft's own
reads are small.

> **Field note: a conclusion that belonged to one pair.** When this
> project first priced the verify step as a chunk step, it drew a
> general lesson from the pair it used, Llama-2-7B and TinyLlama: the
> folklore that speculation fades at large batches holds only at short
> context, and at long context the gain grows with the batch, so
> speculation is a throughput tool. The right panel of Figure 19.2
> shows why it looked so: Llama-2-7B's cache per token is 23 times the
> draft's. But the gain grows only mildly in the cells that fit on this
> GPU, and the cells where the Llama pair gains most don't fit either.
> The left panel, a grouped-query target with a draft of its own
> family, gains least where the right one gains most. A regime map
> describes one pair of models on one GPU.

## A draft that ships with the model: MTP heads

Some models are trained with a *multi-token prediction* (MTP) head: one
extra decoder layer that takes the trunk's final hidden state and the
embedding of the token just sampled, and predicts the token after it
through the trunk's own LM head. DeepSeek-V3 and Qwen3.8-27B (the
hybrid of chapter 8) ship one. As a draft it needs no second model or
tokenizer, and its cache is one layer; chapter 6's `kv_bytes_per_token`
counts that layer, so capacity charges it. tinyperf prices the head as
a model of its own, the target's dimensions with `mtp_layers` layers of
full attention (docstrings trimmed):

```python
def mtp_draft_params(p: TransformerParams) -> TransformerParams:
    assert p.mtp_layers, f"{p.name} ships no MTP head"
    return dataclasses.replace(
        p, name=f"{p.name}-mtp", n_layers=p.mtp_layers,
        full_attn_interval=1, lin_key_heads=0, lin_value_heads=0,
    )

class MTPStepLatencyModel:
    ...
    def _cycle_us(self, batch: int, kv_len: int) -> float:
        return (self.k * self.draft.decode_us(batch, kv_len)
                + self.target.chunk_us(batch, kv_len, self.k + 1))
```

The cycle is a separate draft's, except that the head runs at the
target's tensor parallelism. `break_even_alpha` (not shown) bisects for
the α at which `cycle / E` equals a decode step.

```
Worked example  The MTP head of Qwen3.8-27B as a draft: what one step reads, B200
  LM head, shared with the trunk: 248,320 x 5,120 x 2 bytes = 2.54 GB
  the head's one decoder layer: 0.74 GB of weights
  not in the model: the input projection joining hidden state and embedding, 2h x h = 0.10 GB
  one MTP step at batch 1, context 4096: 0.572 ms, of which the LM head 0.407 (71%)
```

The LM head dominates: a vocabulary of 248,320 makes the output
projection three and a half times the layer. The head is one layer in
64, but its step prices at 5% of the target's.

```
Table 19.6  The MTP head as a draft: Qwen3.8-27B on a B200, calibrated tier, k = 1
  batch  context  target ms  MTP step ms  % of target  LM head share  verify/decode  break-even alpha
      1     4096      10.92        0.572         5.2%            71%          1.001             0.053
      8     4096      11.58        0.598         5.2%            69%          1.003             0.055
     32     4096      13.84        0.659         4.8%            62%          1.013             0.060
     64     4096      16.87        0.755         4.5%            56%          1.020             0.065
      1    32768      11.22        0.590         5.3%            69%          1.001             0.053
     32    32768      23.24        1.246         5.4%            33%          1.008             0.061
  held out: the target's decode step against vLLM on a B200 without speculation (chapter 8), 5 cells: model/measured 0.878-0.958
  published, DeepSeek-V3 technical report, section 5.4.3: second-token acceptance 0.85-0.90 with one MTP module (E = 1.85-1.90 at k = 1), 1.8 times TPS
```

At c ≈ 0.05, against 0.14 for the separate draft (Table 19.3), the
head breaks even near 5% acceptance. The target's step it rests on is
held out against vLLM, 4–12% low (chapter 8).

An MTP head is trained to look one token ahead. An engine can draft
deeper by running it again on its own output, and `MTPStepLatencyModel`
prices k drafts as k runs of the head, LM head included. Whether the
second run guesses as well as the first is the next question.

## How far ahead to draft

A draft runs forward from the last verified token. Its second guess is
conditioned on its first, its third on two guesses, and errors
compound, so acceptance falls with position, which the i.i.d. formula
ignores. The simplest model of it is geometric: position i is
accepted with probability `α · decay^(i−1)`, given that position i−1
was, the `decay` argument of `expected_tokens`.

```
Table 19.7  Expected tokens per cycle when acceptance decays with position: alpha 0.8 at the first, alpha x decay^(i-1) at the i-th
  depth k      i.i.d.   decay 0.9   decay 0.8   decay 0.6
        1        1.80        1.80        1.80        1.80
        2        2.44        2.38        2.31        2.18
        3        2.95        2.75        2.57        2.29
        4        3.36        2.97        2.68        2.31
        8        4.33        3.17        2.73        2.32
       16        4.89        3.17        2.73        2.32
    limit        5.00        3.17        2.73        2.32
```

With decay the yield saturates: at decay 0.8 every draft past the
fourth adds 0.05 tokens in all. The cost doesn't saturate, since every
draft is another step, so the optimum is shallow:

```
Table 19.8  Time per token by depth, the MTP head run k times: Qwen3.8-27B, B200, batch 1, context 4096, alpha 0.8 at the first position, ms
  depth k      i.i.d.   decay 0.9   decay 0.8   decay 0.6
        1        6.39        6.39        6.39        6.39
        2        4.95        5.08        5.22        5.53
        3        4.29        4.60        4.92        5.51
        4        3.94        4.46        4.93        5.72
        6        3.64        4.59        5.28        6.22
        8        3.60        4.92        5.71        6.72
       12        3.79        5.65        6.57        7.73
       16        4.15        6.39        7.43        8.75
   best k           8           4           3           3
  speedup       3.04x       2.45x       2.22x       1.98x
  plain decoding: 10.92 ms per token
  the i.i.d. choice, k = 8, if acceptance decays: 2.22x at decay 0.9, 1.91x at decay 0.8, 1.62x at decay 0.6
  Qwen3-8B + Qwen3-0.6B, RTX A6000, batch 1, context 1024: best k 5 (i.i.d., 2.16x), 3 (decay 0.9, 1.93x), 3 (decay 0.8, 1.81x), 2 (decay 0.6, 1.71x)
```

![Speedup against the number of drafted tokens, 1 to 16: the i.i.d.
curve peaks at 3.04 at k = 8; decaying acceptance peaks at k = 3 or 4
and falls after.](/assets/tinyperf-book/ch19-depth.svg)

*Figure 19.3. Speedup by depth for Qwen3.8-27B's MTP head run k times,
B200, one sequence, 4,096 tokens, first-position acceptance 0.8. Only
the i.i.d. curve keeps rising past k = 4.*

The i.i.d. formula does more than overstate the gain, 3.04 times where
decay 0.8 gives 2.22. It also picks the wrong depth, eight where three
are best. If acceptance decays by 0.8, those eight give 1.91 times
against 2.22 at k = 3. The separate draft shifts the same way, from 5
to 2 or 3. A wrong recommendation is worse than a wide error bar.

`decay` defaults to 1, so the model's default is the assumption this
section argues against. Picking a decay with no measurement would
replace a visible assumption with a hidden one.

## Acceptance is an input

Nothing in the model derives α. It measures how often two models agree,
so it belongs to the draft, the target, the text and the sampling
settings, not to the model sizes. There are two sources.

- **Publications.** Papers report acceptance for their draft methods,
  often by task. DeepSeek-V3's technical report (§5.4.3) says its
  second-token acceptance "ranges between 85% and 90% across various
  generation topics", for 1.8 times the tokens per second (Table 19.6):
  another model and system, so not a check, but consistent with
  E = 1.85–1.90 from a head costing a few percent.
- **Your workload.** Engines report it: vLLM, for one, logs the mean
  acceptance length and a cumulative acceptance rate by position. A few
  runs on real prompts give α and its decay (Exercise 2). Its headline
  "Avg Draft acceptance rate" is not α but accepted over drafted
  tokens, (E − 1)/k:

```
Worked example  An engine's acceptance counters when alpha is 0.8 at every position, k = 4
  mean acceptance length (tokens per verify step): 3.36 = E
  acceptance by position, cumulative: 0.80, 0.64, 0.51, 0.41
  accepted over drafted tokens, (E - 1) / k: 0.59; taken as alpha it gives E = 2.27, not 3.36
```

## How close is it?

The repository holds no measurement of speculative decoding
(`data/validation` has none), so every speculation number in this
chapter is a projection. What can be checked is the algebra, and the
part of the verify step that has been timed.

**The yield.** Table 19.9 simulates speculative sampling on a toy
vocabulary and compares the mean tokens per cycle with
`expected_tokens`. The decaying rows use a draft that at position i
proposes the target's distribution with weight `α·decay^(i−1)` and
otherwise a token the target never emits.

```
Table 19.9  The formula against speculative sampling simulated: a six-token vocabulary, 100,000 cycles per row
  draft                           k  formula E  simulated
  the same q at every position    1      1.800      1.800
                                  2      2.440      2.442
                                  4      3.362      3.363
                                  8      4.329      4.335
  decaying, 0.80 x 0.8^(i-1)      1      1.800      1.800
                                  2      2.312      2.314
                                  4      2.682      2.682
                                  8      2.728      2.728
  target p = (0.4, 0.3, 0.15, 0.1, 0.05), and 0 for the sixth token; draft q = (0.2, 0.35, 0.25, 0.1, 0.1), and 0; alpha = sum of min(p, q) = 0.80
  emitted tokens' frequencies against p, every row: within 0.002
```

The formula agrees within 0.006 tokens, and the emitted tokens follow
the target's distribution within 0.002. This checks the arithmetic and
its assumptions, not any real draft.

**The verify step's GEMMs.** Chapters 3 and 15 timed Qwen3-8B's four
layer GEMMs and its LM head alone, as vLLM calls them, at 39 row counts
from 1 to 1,280. Every cell of Table 19.10 is a measured row count, so its
ratio is the verify step's GEMM cost over the decode step's, measured.

```
Table 19.10  A verify step's weight GEMMs, measured: Qwen3-8B's 144 layer GEMMs and LM head timed alone at b and b(k+1) rows, cuBLAS in a CUDA graph, RTX A6000
  batch  b rows ms    k = 1: ms  ratio  model    k = 3: ms  ratio  model    k = 7: ms  ratio  model
      1      22.83        22.60   0.99   1.00        22.64   0.99   1.01        22.71   0.99   1.02
      8      22.71        23.14   1.02   1.00        23.72   1.04   1.02        25.08   1.10   1.08
     16      23.14        23.72   1.03   1.02        25.08   1.08   1.08        27.19   1.18   1.17
     32      23.72        25.08   1.06   1.06        27.19   1.15   1.15        45.92   1.94   1.94
     64      25.08        27.19   1.08   1.08        45.92   1.83   1.83        73.31   2.92   2.93
    128      27.19        45.92   1.69   1.69        73.31   2.70   2.71       145.30   5.34   5.36
  ratio: measured b(k+1) rows over b rows; model: the calibrated tier's same ratio
  past the ridge, time follows rows: 512 -> 1024 rows, x1.98 measured
  the verify GEMMs, model/measured: 1.00-1.02 with the row curve (fitted to these timings above 32 rows); held out, the tile model alone 0.83-1.02
```

The calibrated model's verify GEMMs read 1.00–1.02 of the
measurement, but above 32 rows that is *fitted*: its row curve came
from these timings. The tile model alone, *held out*, reads up to 17%
low.

Unchecked: any acceptance rate, the draft's step (priced with
constants fitted to Qwen3-8B's kernels), attention with k+1 queries,
the engine's own work per cycle, and the mean-field serving model.
Exercise 5 is the measurement this chapter lacks.

## Where it breaks

- **Acceptance is assumed**, i.i.d. unless you pass a decay, and the
  decay is a one-number guess at a curve that must be measured.
- **Mean field.** Every sequence advances E tokens per cycle. Real ones
  accept different numbers, so TPOT's spread and tail are not modeled.
- **The draft's catch-up.** Each cycle's first draft step must take in
  the up to k+1 tokens the last verify emitted; it is priced as one.
- **Verify attention is priced low.** Above one query per sequence,
  attention gets chapter 7's generic fused-attention price, without the
  decode kernel's fitted floor and read rate, so it costs less than a
  decode step's though it reads more (1.91 against 2.21 ms at batch 8,
  Table 19.4). One draft at batch 1 prices at 0.99 of a decode step.
- **Off-bucket contexts.** `chunk_us` rounds the context up to 256
  tokens; `decode_us` interpolates.
- **Memory.** The draft's weights and cache are never charged against
  capacity (Table 19.5's footer). An MTP head's cache is.
- **In the simulator** `SpecStepLatencyModel` has no `sampler_us` and
  no calibration, so the sampler and the 25 ms per-request overhead
  (chapter 14) vanish under speculation. With no `mixed_step_us`,
  chunked prefill raises an `AttributeError`: only prefill-first
  scheduling runs.
- **Padding.** Its `decode_us` ignores `real`, so attention is charged
  for the padded CUDA-graph batch (chapter 15).
- **The MTP head's input projection** (2h × h, 0.10 GB here) is left
  out.
- **Other drafts**, such as trees, EAGLE- and Medusa-style heads and
  n-gram lookup, are not modeled.

## What you built

- The cycle: k draft steps and one chunk step, exact by rejection
  sampling.
- The yield `E = (1 − α^(k+1)) / (1 − α)`, and with decay.
- The speedup `E / (k·c + v)` and the break-even acceptance.
- The regime map: verification free below the ridge; c rising with the
  draft's cache.
- MTP heads: about 5% of a target step, mostly the LM head.
- The best depth under decay: about 3, not the i.i.d. 8.
- Evidence: simulation of the algebra and measured verify GEMMs; no
  measured speculation.

Appendix A turns these speedups into cost per million tokens.

## Exercises

1. **The best depth.** With v = 1 and i.i.d. acceptance, find the k
   that maximizes `E(α, k) / (1 + k·c)`. Check it against Table 19.2's
   best-k column (c = 0.14; at α = 0.6, k = 2 and 3 tie at 1.53).
2. **Fit acceptance from a log.** If an engine reports the fraction of
   drafts accepted at each position i, that is the product of the first
   i positions' acceptance. Recover α and a decay, and redo Table 19.8
   for your workload.
3. **Charge the draft's memory.** Add the draft's weights and cache to
   `simulate`'s KV budget and rerun a chapter 14 load sweep with
   speculation. Does the smaller pool or the step price set the knee?
4. **Give speculation a sampler.** Add `sampler_us`, a `calibration`
   and `mixed_step_us` to `SpecStepLatencyModel`. What should a mixed
   step with speculation cost, and the sampler on k+1 rows per
   sequence?
5. **Measure it.** Serve Qwen3-8B with a Qwen3-0.6B draft in vLLM at
   k = 2 and 4, batches 1 and 32, greedy decoding. Give the logged
   acceptance to the model and compare its TPOT with the measured.
   Which assumption of this chapter fails first?

---

*[← Chapter 18: The knee]({% post_url 2026-09-29-tinyperf-18-the-knee %}) · [Contents](/series/tinyperf/) · [Chapter 20: Disaggregated prefill and decode →]({% post_url 2026-09-29-tinyperf-20-disaggregated-serving %})*
