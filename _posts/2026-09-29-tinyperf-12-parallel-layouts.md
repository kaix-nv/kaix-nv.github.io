---
layout: post
title: "Building tinyperf, chapter 12: Parallel layouts"
date: 2026-09-29 12:12:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/12-parallel-layouts/
excerpt: "Chapter 11 spread a model over the GPUs of a node by slicing every weight matrix. That has two limits: it all-reduces twice per layer, and it stops paying as the group grows, in decode when the ring's latency outgrows each GPU's shrinking work, in prefill at the node boundary, where NVLink gives way to a much slower network. There are four other ways to spread a model, and large deployments combine them: cut the stack of layers into stages, run copies of the attention for different requests, cut each sequence into pieces, and spread the experts of an MoE model over GPUs. For each, this chapter asks: what does it split, what does it copy, and what does it cost?"
redirect_from:
  - /tinyperf/perf-modeling/2026/09/02/building-tinyperf-m44.html
  - /tinyperf/perf-modeling/2026/09/08/building-tinyperf-m46.html
  - /tinyperf/perf-modeling/2026/09/08/building-tinyperf-m47.html
  - /tinyperf/perf-modeling/2026/09/19/building-tinyperf-m62.html
---

*[Building tinyperf](/series/tinyperf/) · Part III: Many GPUs · Code: [`tinyperf/nets/transformer.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/nets/transformer.py), `build_llm_graph`; [`tinyperf/serving.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/serving.py), `StepLatencyModel`; [`tinyperf/capacity.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/capacity.py) and [`tinyperf/comm_model.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/comm_model.py) · Every table in this chapter comes from `python3 book/scripts/ch12_layouts.py`.*

Chapter 11 spread a model over the GPUs of a node by slicing every
weight matrix. That has two limits: it all-reduces twice per layer, and
it stops paying as the group grows, in decode when the ring's latency
outgrows each GPU's shrinking work, in prefill at the node boundary,
where NVLink gives way to a much slower network. There are four other
ways to spread a model, and large deployments combine them: cut the
stack of layers into stages, run copies of the attention for different
requests, cut each sequence into pieces, and spread the experts of an
MoE model over GPUs. For each, this chapter asks: what does it split,
what does it copy, and what does it cost?

The short answer: each is a lever on one resource, in one phase.

- A **pipeline stage** splits the layers. It is a memory lever: weights
  and cache per GPU fall with the number of stages. It doesn't speed up
  a weight-bound decode step or one prompt, though a prompt's chunks
  can overlap.
- **Attention replicas** split the requests and copy the attention. They
  cut collective traffic, but a lone prompt pays for every replica.
- **Context parallelism** splits each sequence and copies all the
  weights. It divides a long prefill by its width; in decode it divides
  only the cache traffic.
- **Expert parallelism** splits the experts. Each MoE layer pays to move
  tokens there and back, in one of two patterns, and the step waits for
  the busiest GPU.

By the end of this chapter you will know:

- how tinyperf lays a model over `tp × cp × dp × pp` GPUs, and where
  the experts may sit;
- what a pipeline step costs, and why a prompt's chunks pipeline where
  the prompt can't;
- what context parallelism buys in prefill and in decode;
- two ways to move tokens to experts, why an idle replica is not idle,
  and where routing imbalance bites;
- how close the model gets on two RTX A6000s, and which layouts have
  never been measured.

## One rank layout

tinyperf describes a deployment by four degrees. The number of GPUs,
or *ranks*, in it is the *world*, their product:

```
world = tp × cp × dp × pp
```

- `tp`: chapter 11's tensor-parallel group, each GPU holding `1/tp` of
  every weight matrix;
- `dp`: attention replicas, tensor-parallel groups each serving its own
  share of the requests;
- `cp`: context-parallel groups, each a tensor-parallel group holding
  `1/cp` of every sequence;
- `pp`: pipeline stages, each holding a block of consecutive layers.

Expert parallelism, `ep`, is not a fifth factor. It says where the
experts sit, and the model allows two answers. With `ep = 1` every GPU
of a tensor-parallel group holds `1/tp` of every expert, like a dense
feed-forward layer. With `ep = tp × dp` each expert sits whole on one
GPU, and a layer's experts spread over all the attention ranks of a
stage. Anything in between is refused. A layout given as `tp=4, ep=8`
can only mean two replicas, so the builder infers `dp = ep / tp` when
`ep > tp`.

![Eight GPUs as four pairs. Each pair is a tensor-parallel group running
the attention for its own sixteen requests; each GPU also holds sixteen
whole experts, and a dispatch band spans all eight.](/assets/tinyperf-book/ch12-rank-layout.svg)

*Figure 12.1. Eight GPUs as `tp=2 × dp=4`, with `ep=8`, for a model with
128 experts. Each pair is an attention replica serving a quarter of the
requests; each GPU holds 16 whole experts, and in every MoE layer tokens
cross the whole band to their experts and back.*

In every layout `batch` is the **global** batch: every sequence in
flight, across all replicas and micro-batches. An attention replica runs
`ceil(batch / dp)` of them. The expert layer sees them all. The capacity
model counts the same way: `max_batch` returns `dp` times what one
replica holds.

## Pipeline parallelism

A pipeline of `pp` stages gives each stage `L/pp` consecutive layers, as
evenly as integers allow, with the embedding on the first stage and the
LM head on the last. A token's activations, `rows × hidden` values, are
handed from each stage to the next: `pp − 1` point-to-point transfers,
over NVLink if the whole layout fits in one node, otherwise all over the
fabric.

A stage holds only its layers' weights and cache, so a stage is first a
memory lever. What it does to time depends on the phase. By default the
model assumes the engine keeps `pp` micro-batches (slices of the batch)
in flight, one per stage, so every GPU stays busy. A decode step over
`B` sequences is then one traverse of a `B/pp` micro-batch through all
the stages: while one micro-batch crosses all stages, each of the others
does too, one stage apart, so every sequence gains a token per traverse.
Count what one GPU does per step: it streams its stage's weights once
per micro-batch, `pp` times, as many bytes as without the pipeline. Its
cache reads and its math cover only its own layers, `1/pp` of them.

```
Table 12.1  Pipeline stages: Llama-3-70B on H100s, tp=8 in each stage, datasheet rates
  layout    GPUs  hand-off    decode step vs pp=1, context 4096  one 8k prompt  weights  max batch
                                  b=1      b=8     b=64    b=256             ms   GB/GPU    at 4096
  tp8 pp1      8  -              1.00     1.00     1.00     1.00          299.4     17.6        323
  tp8 pp2     16  fabric         1.00     0.98     0.86     0.67          302.1      8.8        750
  tp8 pp4     32  fabric         1.00     0.97     0.79     0.52          307.5      4.5       1597
  tp8 pp8     64  fabric         1.01     0.96     0.76     0.45          318.3      2.4       3273
  tokens/s per GPU at batch 8: pp1 88, pp2 45, pp4 23, pp8 11
```

Every hand-off here crosses the fabric. At batch 1 a step pays only the
hand-offs, under 1%. At batch 8 the step is the weights, and the
pipeline buys 2–4%: each doubling of the GPUs halves the tokens per
second per GPU. At 64 and 256 sequences the cache and the math take
over, and they shrink with the stage: 256 sequences step in 0.45 of the
time on eight stages. The weights per GPU halve with every doubling, and
the batch that fits grows from 323 to 3,273.

The one-prompt column goes the other way. Stage 2 needs stage 1's
output, so one prompt crosses the stages in turn while the others idle,
the pipeline's *bubble*, and every boundary adds a hand-off of the whole
prompt's activations: 6% more on eight stages.

The *chunks* of a prompt are different. With chunked prefill (chapter
14), an engine feeds a prompt in pieces no larger than its token budget
(the most tokens it runs in one step; 8192 here). Chunk 2 can enter
stage 1 while chunk 1 is on stage 2: its attention needs chunk 1's keys
and values only for its own stage's layers, which are on the same GPU.

![Two timelines on two stages. One unchunked prompt occupies stage 1 and
then stage 2, four time units in all. The same prompt in four chunks
finishes in two and a half.](/assets/tinyperf-book/ch12-pipeline-chunks.svg)

*Figure 12.2. A prompt through two stages. Unchunked, it takes one
traverse, as on one GPU, and each stage idles half the time. In four
chunks, stage 1 starts chunk 2 while stage 2 runs chunk 1, and the
prompt finishes in five chunk stage-times instead of eight.*

A wave of `c` chunks drains through `pp` stages in `c + pp − 1`
stage-times, where one stage-time is a chunk's traverse over `pp`. The
time to first token (TTFT) is then

```
TTFT = traverse(prompt) × (c + pp − 1) / (pp × c)
```

As chunks multiply, this approaches `1/pp` of the traverse. An engine
may split the batch into groups in flight; the wave then has
`groups × c` chunks, each `1/(pp·c)` of one group's traverse. One
measured cell, worked through:

```
Worked example  A prefill wave through two stages: 8 prompts of 8192 tokens, Qwen3-8B, two RTX A6000s
  65536 tokens in 8192-token chunks: 8 chunks, in 2 groups of 4 sequences
  one GPU: 8 chunks x 2 half-models = 16 stage-times; two stages: 8 + 2 - 1 = 9 stage-times
  model: one GPU 10719.5 ms; pp=2 6112.0 ms (0.570 of one GPU; 9/16 = 0.5625)
  measured: one GPU 10420.5 ms; pp=2 6690.9 ms (0.642 of one GPU)
  the engine's pipeline step cost, fitted: pp_step_overhead_us = 2200
```

Here each chunk is a whole prompt: the budget cuts the batch into steps,
and steps pipeline whether they hold pieces of one prompt or several
prompts. No measured cell splits a single prompt.

The worked example's last line is the engine's own cost. vLLM runs each
stage in a process of its own, and the stages coordinate every step. On
two RTX A6000s that costs about 2.2 ms a step. We don't know its parts:
it is a fitted constant, and it belongs to that engine version on that
pair of GPUs. In a prefill wave the model adds it once per group
traverse, scaled with the traverse, not once per chunk step.

## Data-parallel attention

When experts span more GPUs than a tensor-parallel group, the other
GPUs are replicas of the attention: each serves its own requests with
its own copy of the attention weights, and the replicas meet only
inside the expert layers. vLLM runs this layout with
`--data-parallel-size` and `--enable-expert-parallel`. For a dense model
the replicas never meet, and `dp` is simply several servers. Table 12.2
prices five layouts; TPOT, the time per output token, is the decode
step.

```
Table 12.2  Attention replicas: gpt-oss-120b bf16 on eight B200s, all-to-all dispatch, datasheet rates
  decode: global batch 64 at context 4096; prefill: 8 prompts of 8192 tokens, or one
  layout       experts/GPU seqs/replica  TPOT ms  comm TTFT 8 x 8k  1 x 8k  GB/GPU max batch
  tp8             all, 1/8           64     6.43   20%       145.1    20.9    30.5      6680
  tp8 ep8         16 whole           64     6.30   22%       125.1    18.3    30.5      6680
  tp4 dp2 ep8     16 whole           32     6.00   18%        99.9    26.8    31.0      6724
  tp2 dp4 ep8     16 whole           16     5.91   16%        87.3    44.8    32.0      6704
  tp1 dp8 ep8     16 whole            8     5.91   12%        80.9    80.9    34.1      6608
  one 8k prompt, ms of attention and the rest / MoE compute / collectives: tp8 5.6 / 7.5 / 7.8; tp1 dp8 ep8 21.5 / 45.5 / 13.9
```

Every row is the same eight GPUs holding the same model: per-GPU memory
moves by 12% and the batch that fits by 2%, the growth being each
replica's own attention weights and embedding tables. What moves is
where the collectives are. Tensor parallelism all-reduces twice per
layer across eight GPUs; eight replicas pay only the expert dispatch.
The decode step improves 8%, and the collectives' share falls from 20%
to 12%. Eight prompts improve 44%, each replica running one with its
attention unsliced.

The surprise is the lone prompt: on eight replicas it takes almost four
times as long as on one tensor-parallel group, as long as eight prompts
do. Its attention and projections run on one GPU (21.5 ms against 5.6),
but most of the growth is in the MoE layers (45.5 ms against 7.5): the
seven idle replicas are not idle, as expert parallelism will explain.

## Context parallelism

This axis splits the sequence itself, differently in each phase.

**Prefill: ring attention.** Each of the `cp` groups holds `s/cp` tokens
of every sequence, and every projection, norm and feed-forward layer
runs on that shard: GEMM rows fall by `cp`. Attention needs every
earlier key, so the groups pass their blocks of keys and values around a
ring while each computes its own queries against them. Implementations
split the sequence in a zigzag so each group gets an equal share of the
causal work; the model keeps the average causal key length, divides the
query rows by `cp`, and prices the circulation as one all-gather of the
layer's KV, not overlapped with compute.

**Decode: shard the cache.** A decode step has one query per sequence
and a long cache. The query is copied to every group, each attends to
its `1/cp` of the cache, and a small all-reduce combines the partial
outputs with their softmax normalizers. Every GEMM runs in full in every
group: the weights are copied, so weight traffic doesn't shrink. Only
cache traffic does.

```
Table 12.3  Context parallelism: Llama-3-70B on H100s, tp=8 in each group, datasheet rates
                      TTFT, s  TPOT, batch 8, ms          GB/GPU  max batch
  layout    GPUs   32k   128k       4k      128k  weights KV/seq    at 128k
  tp8 cp1      8  1.38   8.77    11.33     23.74     17.6   5.37         10
  tp8 cp2     16  0.69   4.40    11.53     17.74     17.6   2.68         20
  tp8 cp4     32  0.36   2.23    11.75     14.86     17.6   1.34         40
  tp8 cp8     64  0.20   1.19    12.34     13.89     17.6   0.67         80
```

Prefill scales nearly with the groups: a 128k-token prompt takes 8.8 s
on one node and 1.2 s on eight, because both its terms, the GEMM rows
and the quadratic attention, shard. Decode at 4k tokens gets 9% *slower*
on eight groups: every GPU still streams its full share of the weights,
and the combine is pure overhead. At 128k the cache is the traffic, and
eight groups take 41% off the step. The weights per GPU never move; the
cache per sequence divides, and the batch that fits grows eightfold.

## Expert parallelism

With `ep` ranks, each holds `E/ep` whole experts, and every MoE layer
moves tokens to their experts' ranks and back, in one of two patterns.

**All-to-all.** Each rank sends each routed (token, expert) row to the
rank that holds the expert, and a second all-to-all brings the outputs
back. A token travels `top_k` times. Dedicated expert-parallel kernels,
such as DeepEP's, work this way, and it is the model's default.

**All-gather and reduce-scatter.** Without such kernels, vLLM
all-gathers every token's hidden state and router scores across the
replicas, and each rank routes the whole global batch and runs its local
experts on the rows that chose them. A reduce-scatter then sums the
partial outputs and returns each replica its own rows. A token crosses
once to each other replica, whatever its `top_k`. The builder calls this
`ep_dispatch="allgather"`.

With `tp = 1`, per token of its own, each GPU sends

```
all-to-all:                  2 · top_k · hidden · (ep − 1)/ep
all-gather + reduce-scatter: (dp − 1) · (2 · hidden + experts)
```

values, so the all-to-all moves about `top_k/dp` times as much.

```
Table 12.4  Two ways to move tokens: gpt-oss-120b (hidden 2880, top-4) on B200s, tp=1, ep=dp
  per MoE layer, a prefill of 1024 tokens on each replica; KB each GPU sends per token of its own
  dp = ep  all-to-all KB      us  all-gather + reduce-scatter KB      us  ratio
        2          23.04    34.2                           11.78    21.4   1.96
        4          34.56    51.3                           35.33    52.2   0.98
        8          40.32    65.9                           82.43   113.8   0.49
  all-to-all: 2 x top_k x hidden x (ep-1)/ep; all-gather + reduce-scatter: (dp-1) x (2 x hidden + experts)
  equal at dp = top_k x 2 x hidden / (2 x hidden + experts) = 3.91
```

The two cross near `dp = top_k`, at 3.9 here. On two replicas the gather
moves half the bytes; on eight, twice.

**Lockstep.** The replicas must step together, because every MoE layer's
collective needs all of them. With CUDA graphs on, vLLM pads every
replica to the largest replica's token count, and a replica with no
requests runs a dummy batch of that size. The router routes dummy tokens
like real ones. Unlike chapter 9's padded rows, which hold stale real
tokens, a dummy batch is one token repeated (chapter 9's exercise 3). So
the expert layer sees `dp × ceil(batch/dp)` sequences. At global batch 1
on two replicas, each GPU's expert rows are those of one GPU running the
whole request. That is Table 12.2's lone prompt: at `dp=8`, one prompt
costs what eight do.

> **Field note: the dummy token.** Priced as a random token, the dummy
> left the batch-1 decode cells of the two-A6000 run of Table 12.8 at
> 0.82–0.85. Hooks on both replicas' routers showed the real token and
> the dummy touching 7.25 distinct experts, where two random tokens
> touch 6.1: the repeated token picks the same `top_k` experts, which
> barely overlap the real token's. Adding `top_k` experts moved the
> cells to 0.95–0.98 (0.97–1.00 today).

**Experts per rank.** How many experts a step touches is a property of
the router (chapter 9), not of the layout. The builder reads the global
count from chapter 9's table by tokens (its padded and reply-position
refinements apply only without expert parallelism) and spreads it evenly
over the ranks; on gpt-oss-20b the two halves of a two-way split were
measured touching nearly equal counts.

```
Table 12.5  Experts each rank reads per layer in a decode step: gpt-oss-20b, dp=2 ep=2, two RTX A6000s
  global  one GPU  two ranks: tokens  experts in all  per rank  measured TPOT,
   batch  experts         per launch  (real + dummy)                2 GPUs / 1
       1        4                  2         4.0 + 4         4       1.09-1.13
       8       13                  8            13.1         7       0.66-0.77
      32       21                 32            21.3        11       0.85-0.87
  each rank holds 16 experts; above 16 tokens per launch the engine's kernel runs 64-row blocks at 0.70 of the rate (chapter 9)
```

At batch 1 each rank reads as many experts as one GPU does, and pays the
collectives on top: two GPUs are 9–13% slower than one. At batch 8 each
reads about half, and two GPUs step in 0.66–0.77 of the time. At batch
32 each still reads about half, 11 of 21, but its kernel launch now
carries 32 gathered tokens for 16 local experts. That crosses chapter
9's block-size switch, which one GPU reaches only past 32 sequences, and
the gain shrinks to 13–15%.

**The hot rank.** A step waits for its most loaded rank. tinyperf takes
that load as a declared input, `moe_imbalance`: the busiest rank's
routed rows over the mean (with `ep = 1`, the hottest expert's), 1.0 for
perfect balance. It scales the rows per expert of the grouped GEMM that
the step is priced on.

```
Table 12.6  The hot rank: gpt-oss-120b on eight B200s, tp=4 dp=2 ep=8, datasheet rates
  imbalance        prefill 8 x 2048        decode, batch 64      decode, batch 4096
                     ms rows/expert          ms rows/expert          ms rows/expert
        1.0       25.56         512        6.00           3       20.75         128
        1.5       30.94         768        6.01           4       21.90         192
        2.0       36.32        1024        6.01           5       22.05         256
```

A prefill's expert GEMMs are bound by math, so extra rows cost time: 42%
at twice the mean load. A decode step of 64 streams the same expert
weights whether they serve 3 rows or 5, and doesn't move. Even at 4,096
sequences the hot rank's 256 rows per expert cost 6%. Imbalance taxes
prefill and very large decode batches; a weight-bound decode step
doesn't feel it.

## The code

The layout enters `build_llm_graph` at the top. These are the lines
that set it, with the lines between them trimmed:

```python
    if dp is None:
        dp = ep // tp if ep > tp else 1
    assert dp >= 1 and (ep == 1 or ep == tp * dp), \
        f"experts must be TP-sharded (ep=1) or span all attention ranks (ep=tp*dp={tp * dp}), got ep={ep}"
    ...
    b_local = -(-batch // dp)                    # sequences on this rank
    ...
    s_full = seq_len if phase == "prefill" else chunk
    if phase == "prefill":
        kv = math.ceil((seq_len + 1) / 2)
        s = -(-s_full // cp)                     # this rank's token shard
    else:
        kv = seq_len + (math.ceil((chunk + 1) / 2) if chunk > 1 else 0)
        kv = -(-kv // cp)                        # this rank's KV shard
        s = s_full                               # queries replicated over cp
    tokens_global = batch * s                    # across all dp ranks
    tokens = b_local * s                         # rows on this rank
```

`tokens` is the row count of this rank's GEMMs and `kv` the cache length
its attention reads: in prefill `cp` divides the rows, in decode the
cache. A helper, `_cp_comm`, adds the collective after attention: in
prefill an all-gather of the layer's keys and values, in decode an
all-reduce of each head's partial output and softmax statistics.

Chapter 9 quoted the MoE block's branch for `ep = 1`. Here is the branch
for `ep > 1`, comment lines trimmed:

```python
        if ep > 1:
            e_local = p.n_experts // ep
            tokens_padded = dp * tokens             # every replica at the max
            assign = tokens_padded * p.top_k         # token-expert pairs across ranks
            rows_local = max(1, assign // ep)
            tokens_real = tokens_global
            if tokens_padded > tokens_real:
                spread = min(p.n_experts, distinct_experts(p, p.n_experts, tokens_real * p.top_k, routing_skew) + p.top_k)
            else:
                spread = distinct_experts(p, p.n_experts, assign, routing_skew)
            active_local = max(1, min(e_local, int(spread / ep + 0.5)))
            m_e = max(1, math.ceil(assign / ep / active_local * moe_imbalance))
            e_width = p.ffn_hidden
```

`tokens_padded` is lockstep: every replica at the largest one's count.
`spread` is the global count of experts touched, plus `top_k` for a
dummy batch, and `active_local` is its share per rank: the grouped
GEMM's batch. `m_e` is the hot rank's rows per expert. Then the tokens
move, in the chosen pattern (comment lines trimmed, the grouped GEMMs
elided):

```python
        routed = Tensor("moe_routed", (rows_local, h), dt)
        if ep > 1 and ep_dispatch == "allgather":
            if dp > 1:
                g.AllGather("moe_dispatch", Tensor("moe_dp_in", (tokens, h + p.n_experts), dt),
                            group_size=dp, count=L_moe)
        elif ep > 1:
            routed = g.AllToAll("moe_dispatch", routed, group_size=ep, count=L_moe)
        ...                                          # the grouped GEMMs of chapter 9
        if ep > 1 and ep_dispatch == "allgather":
            if dp > 1:
                g.ReduceScatter("moe_combine", Tensor("moe_dp_out", (tokens_padded, h), dt),
                                group_size=dp, count=L_moe)
            if tp > 1:
                down = g.AllReduce("moe_allreduce", Tensor("moe_out", (tokens, h), dt),
                                   group_size=tp, count=L_moe)
        elif ep > 1:
            down = g.AllToAll("moe_combine", down, group_size=ep, count=L_moe)
```

The all-gather carries `h + n_experts` values per token: the hidden
state and the router's scores. Chapter 11 prices all four collectives.

Memory follows the same rules. `_weights_local_bytes` divides a GPU's
weights by `tp` and its experts as the layout places them, then scales
both by the heaviest stage's share of the layers; `cp` leaves them
whole, since the weights are copied. It divides the cache instead, in
`max_batch`, which also multiplies the result by `dp`.

A pipeline is not a graph at all. `StepLatencyModel` builds one graph
per stage, each with its share of the layers, and adds them up with the
hand-offs and the engine's step cost:

```python
        n_mb = min(self.n_mb, batch)
        mb = -(-batch // n_mb)
        total = 0.0
        for i, n in enumerate(self._stage_layers):
            total += self._graph_us(phase, mb, seq, chunk, shared,
                                    include_embed=(i == 0),
                                    include_head=(i == self.pp - 1),
                                    n_layers_override=n)
        rows = mb * (seq if phase == "prefill" else chunk)
        total += (self.pp - 1) * p2p_us(self.device,
                                        rows * self.p.hidden * self.p.dtype.nbytes,
                                        self._cross_node)
        if self.calibration and self.calibration.pp_step_overhead_us:
            total += self.calibration.pp_step_overhead_us
        return total
```

`n_mb` is the micro-batches in flight, `pp` by default. A prefill wave
applies the chunk formula to that traverse:

```python
        groups = min(self.n_mb, n_seqs)
        g = -(-n_seqs // groups)
        n_chunk = (-(-(g * b) // self.chunk_tokens)) if self.chunk_tokens else 1
        group_us = self._price("prefill", n_seqs, b)
        chunks = groups * n_chunk
        return group_us * (chunks + self.pp - 1) / (self.pp * n_chunk) + overhead
```

`_price` returns one micro-batch's traverse, here one group's.

## How close is it?

Two layouts have met silicon, both on chapter 11's two RTX A6000s,
joined by a PCIe host bridge, under vLLM. Their all-gathers and
reduce-scatters are priced from the pair's measured NCCL curves. Each
run repeats chapter 6's grid: TTFT is the time for a batch of prompts to
produce one token each, TPOT the mean decode step over the next 128.

**Pipeline, pp=2, Qwen3-8B.** The engine forms micro-batches by its own
rule. With `max_num_seqs`, its limit on sequences per step, above the
batch, it puts groups in flight only when the batch's prompts span more
than one budget, so the model runs `min(pp, ceil(tokens / 8192))` groups
per cell.

```
Table 12.7  Pipeline parallelism on silicon: Qwen3-8B, pp=2 on two RTX A6000s under vLLM
  batch prompt groups   TTFT ms model/meas vs 1 GPU  TPOT ms model/meas vs 1 GPU  pipeline step cost
      1    512      1      75.5      1.062     0.95    25.87      1.018     1.08  fitted
      1   2048      1     289.7      1.030     0.99    26.20      1.018     1.08  fitted
      1   8192      1    1293.8      1.054     1.01    28.53      0.983     1.11  fitted
      8    512      1     557.5      1.009     0.99    27.67      1.000     1.10  fitted
      8   2048      2    1869.2      0.937     0.82    27.88      1.018     1.00  held out
      8   8192      2    6690.9      0.913     0.64    35.37      0.957     0.90  held out
     32    512      2    1717.0      0.977     0.78    30.05      0.960     1.00  held out
     32   2048      2    5602.8      0.933     0.62    35.35      0.970     0.84  held out
  TTFT 0.91-1.06; TPOT, the 4 cells the step cost was fitted on 0.98-1.02, the other 4 0.96-1.02
```

Decode in one group is 8–11% slower than on one GPU: the step is the
weights, the pipeline buys nothing, and the engine's step cost is added.
Batches that span several budgets run in 0.62–0.82 of one GPU's time.
The step cost was fitted to the TPOT of the four one-group cells. The
other four are held out for that constant, though the group rule was
read from them. The TTFT column is in-sample: the chunk wave was found
on these cells.

**Expert parallelism, dp=2 ep=2, gpt-oss-20b.** Two attention replicas,
16 experts on each GPU, vLLM's all-gather dispatch. A cell's time is the
slower replica's; the two agreed within 1% in every cell. Routing uses
chapter 9's table for random-token prompts.

```
Table 12.8  Expert parallelism on silicon: gpt-oss-20b bf16, dp=2 ep=2 on two RTX A6000s under vLLM
                TTFT                                  TPOT                      
  batch prompt       ms model/meas all-to-all vs 1 GPU      ms model/meas vs 1 GPU
      1    512    121.7      0.921      1.114     1.31   13.06      0.972     1.13
      1   2048    406.8      1.068      1.300     1.66   12.73      1.002     1.09
      1   8192   1871.0      0.955      1.157     1.67   13.21      0.983     1.09
      8    512    415.1      1.037      1.264     0.85   20.23      0.911     0.66
      8   2048   1794.8      0.959      1.169     0.94   19.95      0.935     0.66
      8   8192   7893.6      0.903      1.095     0.87   22.78      0.859     0.77
     32    512   1767.7      0.964      1.178     0.96   37.19      0.954     0.87
     32   2048   7201.1      0.954      1.164     0.95   39.05      0.931     0.85
  TTFT 0.90-1.07 (all-to-all pricing 1.10-1.30); TPOT batch 1 0.97-1.00, batch 8 0.86-0.94, batch 32 0.93-0.95
```

Priced as an all-to-all, the same prefills read 10–30% high: the
dispatch pattern matters. Decode at batch 8 still reads 6–14% low, and
we have not found why.

No constant was fitted to these cells, but the committed prediction's
misses led to several mechanisms now in the model: the link's slower
all-gather and reduce-scatter, the expert kernel running its activation
over every gathered pair rather than its own, the dummy token and the
per-rank expert count. So read the current columns as in-sample. The
held-out test is the prediction committed before each run, recorded then
and not recomputed:

```
Recorded  Predictions committed before each run, model/measured
  pp=2 (Qwen3-8B): TTFT 1.01-1.66, TPOT 0.90-1.15
  dp=2 ep=2 (gpt-oss-20b): TTFT 0.81-0.95, TPOT 0.78-1.00
  dp=2 ep=2 against one GPU, faster or slower: the committed prediction matches silicon in 16 of 16 TTFT and TPOT cells
```

The pipeline's committed prefill read up to 1.66: it didn't know that
the engine's steps pipeline. The expert-parallel one read low but called
every direction: slower than one GPU at batch 1, faster above.

## Where it breaks

- **Two GPUs, one link, one engine.** Every measurement here is width
  two over PCIe with vLLM: nothing over NVLink, beyond two GPUs or
  across nodes.
- **Unmeasured layouts.** Context parallelism, attention replicas
  without experts and the all-to-all dispatch have never been measured.
  Tables 12.1–12.4 and 12.6 are projections from chapter 11's
  collectives and the single-GPU model.
- **Lockstep at scale.** The dummy batch was measured on two replicas.
  Table 12.2's lone prompt at eight replicas assumes the engine pads the
  same way.
- **Pipeline schedules.** Decode is priced in the steady state: no
  ramp-up or drain, no uneven stages, no interleaved schedules. The
  2.2 ms step cost belongs to one engine version on one GPU pair.
- **Chunks of one prompt.** Each measured chunk is a whole prompt. No
  measured cell splits a single prompt, as Figure 12.2 does.
- **Placement.** A collective is priced by its group size, and a
  hand-off by whether the whole layout fits in one node, not by where
  the ranks actually sit.
- **Ring attention** is priced without overlap, an upper bound, and
  with the zigzag split's perfect balance.
- **Imbalance** is an input. The model can't predict a router's hot
  rank, or uneven requests across replicas.

## What you built

- One rank layout, `world = tp × cp × dp × pp`, with two places for
  the experts and a global `batch`.
- Pipeline stages: a decode step as one traverse of a micro-batch, a
  prefill wave as `(c + pp − 1)/(pp · c)` of a traverse, a fitted step
  cost.
- Attention replicas that move collectives from attention to experts.
- Context parallelism: sharded rows in prefill, a sharded cache in
  decode.
- Expert parallelism: two dispatch patterns crossing near `dp = top_k`,
  lockstep padding, a global expert count spread over ranks, a hot rank
  that taxes prefill.
- Evidence: a pipeline on two GPUs at 0.91–1.06 and expert parallelism
  at 0.86–1.07 (in-sample; the pipeline's held-out TPOT 0.96–1.02),
  where the committed predictions read 0.78–1.66.

## Exercises

1. Add the pipeline's ramp-up to a request's first decode step: `pp − 1`
   extra stage-times. How much does it matter at batch 8 against batch
   256 in Table 12.1?
2. Add a `dp_imbalance` knob, like `moe_imbalance`, for replicas that
   receive uneven requests. At what skew does `tp1 dp8` lose Table
   12.2's decode lead over `tp8`?
3. Using Table 12.4's formulas, find where the two dispatch patterns
   cross for a model with top-8 routing, hidden 7,168 and 256 experts.
   Check your answer with `build_llm_graph`.
4. Rerun Table 12.7's cells with a token budget of 2,048
   (`chunk_tokens`), so that every multi-prompt cell runs two groups.
   Which cells should decode faster than one GPU, and by how much?

---

*[← Chapter 11: Collectives and tensor parallelism]({% post_url 2026-09-29-tinyperf-11-collectives-and-tensor-parallelism %}) · [Contents](/series/tinyperf/) · [Chapter 13: Training →]({% post_url 2026-09-29-tinyperf-13-training %})*
