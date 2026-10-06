---
layout: post
math: true
title: "Building tinyserve M6c: Expert parallelism: route the token, not the whole model"
date: 2026-10-05 08:02:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Follow four tokens through expert routing, all-to-all dispatch, and distributed MoE execution."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m6c-expert-parallelism.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M6b — Pipeline parallelism: split the layer stack]({% include tinyserve-post-url.html slug="building-tinyserve-m6b" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m6b-pipeline-parallelism.md" %}) · Next: [M6d — Context parallelism: split the KV, not the answer]({% include tinyserve-post-url.html slug="building-tinyserve-m6d" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m6d-context-parallelism.md" %})

Tensor parallelism splits every dense layer. Pipeline parallelism splits the
layer stack. Neither describes a mixture-of-experts model, where a router
chooses a small subset of many alternative MLPs for each token.

[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) has 128 experts
and activates eight per token. Its BF16
checkpoint contains 56.9 GiB of weights, including 54.0 GiB in experts. One
48 GiB A6000 cannot hold that checkpoint. The useful question for M6c is
therefore not "can two GPUs run the same dense MLP?" It is:

> Can each GPU retain only its own experts while every token still reaches
> exactly the experts selected by the pretrained router?

That is expert parallelism.

## Dense MLP versus sparse MoE

A dense Qwen3 layer sends every token through one SwiGLU MLP:

$$
\operatorname{MLP}(x)=W_{down}\left(\operatorname{SiLU}(W_{gate}x)
\odot W_{up}x\right).
$$

A sparse MoE layer first computes router probabilities over $E$ experts:

$$
p(x)=\operatorname{softmax}(W_r x), \qquad
\mathcal{K}(x)=\operatorname{topk}(p(x), K).
$$

Qwen3 renormalizes the selected weights, evaluates only those experts, and
adds their weighted outputs:

$$
y(x)=\sum_{e\in\mathcal{K}(x)}
\frac{p_e(x)}{\sum_{j\in\mathcal{K}(x)}p_j(x)}\operatorname{MLP}_e(x).
$$

The checkpoint fixes the router and expert weights. Inference does not use a
training-time load-balancing loss, capacity factor, or token dropping. If the
router sends many tokens to one rank, that rank simply has more work.

## A concrete four-token example

The figure uses four experts and top-2 routing so every movement fits on one
page. The real model performs the same steps with 128 experts and top-8.

<a href="/assets/tinyserve/m6c-expert-parallelism.svg"><img src="/assets/tinyserve/m6c-expert-parallelism.svg"
     alt="Four stages show source-token ownership, packing by destination, expert execution across two GPUs, reverse dispatch, weighted combination, and restoration of original token order"></a>

Suppose rank 0 owns experts E0–E1 and source tokens t0–t1; rank 1 owns E2–E3
and source tokens t2–t3. Token t0 selects E0 and E2. Its hidden row must be
copied once to each expert owner. After both expert outputs return, rank 0
restores t0's two routes and computes the weighted sum. The same happens for
the other tokens.

Three different orders matter:

1. **Token order** identifies t0, t1, t2, and t3 in the model input.
2. **Packed route order** groups repeated hidden rows by destination and
   expert so communication is contiguous and each active expert gets a batch.
3. **Return order** follows the packed send buffer. The source must invert the
   packing permutation before combining the top-k results.

Confusing these orders does not necessarily crash. It can attach a valid
expert output to the wrong token, producing plausible-looking garbage.

## Why tinyserve assigns source-token slices

The surrounding Qwen3 attention path remains replicated under EP. Both ranks
therefore enter an MoE block with the same `[B,T,H]` hidden tensor and maintain
the same paged KV state. Dispatching every token from both ranks would double
the expert work.

Instead, `token_range()` gives each rank a balanced contiguous slice of the
flattened $N=B\times T$ tokens. For EP=2 and five tokens, rank 0 owns tokens
`[0,3)` and rank 1 owns `[3,5)`. Each rank dispatches only its source slice.
After expert combination, an equal-padded token all-gather reconstructs all
$N$ outputs in original order on both ranks. The next replicated attention
layer can then proceed unchanged.

This final gather is part of tinyserve's chosen decomposition, not an
intrinsic third phase of every EP system. A system that keeps distinct token
partitions through the dense layers can avoid it, but then attention, KV
ownership, scheduling, and logits gathering must all understand distributed
request rows. M6c keeps those concerns fixed so the expert data movement stays
isolated and readable.

## The implementation, one layer at a time

Let `N` be the number of flattened tokens, `K=8`, and `H=2048`.
`Qwen3SparseMoeBlock.forward()` does the following:

1. The replicated router produces selected expert ids and weights `[N,K]`.
2. Rank `r` takes its `[N_r,H]` source slice and repeats every row `K` times,
   producing `[N_r*K,H]` routes.
3. A stable sort by global expert id creates the packed activation buffer.
4. A small `[EP, experts_per_rank]` count tensor is gathered. Its row sums are
   the variable input splits; its destination column supplies receive splits.
5. The first `all_to_all_single` sends packed hidden rows to expert owners.
6. Received segments from every source are regrouped by local expert. An
   expert with zero routes performs no MLP call.
7. A reverse `all_to_all_single` returns outputs using the transposed split
   relationship.
8. The source inverts its stable sort, reshapes to `[N_r,K,H]`, applies the
   retained router weights, and sums across `K`.
9. The token all-gather restores `[N,H]` on every rank.

`ExpertPartition` assigns experts 0–63 to rank 0 and 64–127 to rank 1. The
module list contains `Identity` placeholders for non-owned global expert ids,
so checkpoint names remain unchanged while non-owned tensors disappear from
the rank's state dictionary. The loader opens each safetensors shard and reads
only target keys; it never materializes the other rank's 27 GiB of experts.
Routers, attention, embeddings, normalization, and the LM head are replicated.

Rank 0 still owns tokenization, arrivals, scheduling, sampling, output, and
metrics. Workers mirror its explicit M4/M5 plans, so packing, chunked prefill,
prefix adoption, and recompute preemption do not acquire a second control
plane. Variable route splits keep CUDA graphs disabled for this milestone.

One subtle cost remains in packed prefill: left-padding rows are finite but
masked, and the dense model still carries them through the MLP. M6c therefore
routes padding tokens too. The retained route counts include those routes.
Removing them requires a varlen token-packed model interface, not an EP-only
special case.

## Correctness evidence

The tests build the argument in layers:

- CPU tests prove disjoint expert state dictionaries, contiguous expert
  ownership, non-divisible token ranges, and target-only checkpoint loading.
- A two-GPU fp32 block test forces balanced routing, a destination with no
  received tokens, and an odd token count. Every result matches an unsharded
  MoE reference after dispatch and permutation inversion.
- A deterministic tiny `qwen3_moe` checkpoint proves paged generation and
  `serve()` token parity through packed/chunked prefill, prefix reuse, and
  forced preemption. A separate BF16 gate preserves tokens with eager
  FlashInfer decode and confirms that EP disables CUDA graphs.
- The real Qwen3-30B-A3B layer-0 MoE output matches a separately loaded
  unsharded 128-expert BF16 reference with maximum absolute difference
  `0.015625`. For the chat-formatted prompt “What is 2 + 2? Reply with only
  the number,” both the official Transformers model and the end-to-end EP
  model greedily select token `19`, which decodes to `4`.

The chat template matters here. Qwen3-30B-A3B is instruction-tuned; a raw
string such as `2+2=` asks it to continue text rather than answer a user.
That raw continuation is not a semantic correctness test. The independent
Transformers comparison and tinyserve use the same tokenizer, checkpoint,
BF16 dtype, disabled thinking mode, and greedy first-token decision.

The BF16 difference comes from a different accumulation order: the reference
adds expert-grouped outputs, while EP restores top-k order and reduces that
axis. The selected experts and weights are unchanged.

## What the two A6000s measured

The retained run used two RTX A6000 GPUs, PyTorch 2.13.0+cu130,
Transformers 5.15.1, BF16 weights, eager PyTorch paged attention, no prefix
caching, a 2,048-token KV pool, four greedy output tokens, and three repeats
per batch size. These short controlled prompts measure this implementation;
they are raw completions chosen to measure the serving path, not answer
quality, and they are not a production Qwen throughput claim.

First, capacity:

| quantity | observed or derived |
|---|---:|
| full BF16 checkpoint | 56.871 GiB |
| expert weights | 54.000 GiB |
| replicated weights | 2.871 GiB |
| parameters per EP rank | 29.871 GiB |
| peak reserved per rank, real-model check | 32.902 GiB |

EP=1 has no latency row because it cannot hold this BF16 checkpoint on one
47.4 GiB device. Treating an OOM attempt as a benchmark would add no useful
information.

The serving sweep reports medians across the three repeats:

| batch | goodput (tok/s) | TTFT p50 (ms) | ITL p50 (ms) | unused experts |
|---:|---:|---:|---:|---:|
| 1 | 5.05 | 234 | 168 | 75.3% |
| 2 | 7.65 | 358 | 213 | 63.4% |
| 4 | 12.35 | 510 | 266 | 45.7% |
| 8 | 18.95 | 597 | 362 | 32.1% |

Larger batches give each active expert more rows per call and improve
goodput, but they also increase time to the first token and the interval
between tokens. The falling unused-expert fraction is not automatically
"better balance": the mean per-layer maximum-to-mean load ratio was still
8.90 at batch 8. With only a few routed tokens and 128 choices, sparse,
uneven work is the expected decode regime.

Four output tokens require four forwards. Each rank therefore observed 384
activation all-to-alls, 192 small route-count exchanges, and 192 token
gathers: per layer, per forward, there are two all-to-alls, one count gather,
and one token gather. A separately labelled four-request single-forward
profile synchronized every collective. Its slower 493 ms wall time included
at most 42.3 ms of accumulated all-to-all event time on either rank. That
timing mode is attribution evidence, not the normal fast path.

The complete measurements and token ids are retained in
[`benchmarks/m6c-ep2-a6000-2026-08-28.json`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m6c-ep2-a6000-2026-08-28.json).

## What M6c deliberately does not optimize

- The expert loop uses ordinary PyTorch MLP calls. A fused grouped-GEMM kernel
  would be the next performance step, but would hide the routing mechanism in
  this milestone.
- Dense weights and KV are replicated, so EP adds expert-weight capacity, not
  long-context capacity. M6d changes KV ownership.
- EP is degree two and independent of TP and PP. No combined topology is
  claimed from a two-GPU machine.
- CUDA graphs remain off because destination splits depend on router output.
- The benchmark uses the readable gather attention backend and short outputs;
  it does not compare against a different engine or quantized checkpoint.

## Takeaway

Expert parallelism is a routing problem before it is a matrix-multiplication
problem. The hard invariant is not merely that each rank owns different
weights. Every repeated route must return to the right source token, receive
the right router weight, and reconstruct the original residual order.

M6c now makes a 56.9 GiB MoE checkpoint fit across two 48 GiB GPUs while
preserving the M0–M5 serving lifecycle. M6d will split a different object:
the sequence's KV state, using exact distributed-softmax statistics rather
than expert routing.
