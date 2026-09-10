---
layout: post
math: true
title: "Building tinyperf M59: Compressed attention, cross-checked: three mechanisms and a bug that batching hid"
date: 2026-09-10 11:09:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Running the DeepSeek-V4 build through an independent model, cell by cell: the indexer is skipped when there is nothing to select, it writes its logits, hyper-connections launch six kernels not thirty-eight — and a batch-8 cell exposed a latent-attention bug the model had carried since milestone 41."
---

*Milestone 59 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `nets/transformer.py`; errata on milestones 41 and 58.*

Milestone 58 built DeepSeek-V4's compressed attention from the public
report and configs and said, plainly, that none of it had met anything
but its own arithmetic. No V4 checkpoint fits this box, so the next best
check was cross-model: run the same net through an independent production
analytical model on the same GPU, cell by cell, and go through the
disagreements one component at a time. As with every earlier round of
this, the numbers themselves stay internal; what is public is what the
comparison changed in tinyperf, and why.

Four things came out of it. Three are mechanisms the model did not have.
One is a bug it had since milestone 41.

## The indexer is skipped when there is nothing to select

A compressed-sparse layer's indexer scores every compressed key so the
attention can keep the top-k. When the compressed cache holds fewer
entries than k, the top-k is all of them and the selection is the
identity — and an engine that knows this skips the indexer entirely: no
query projection, no scores, no top-k. The reference does. tinyperf now
does, on both the compressed-sparse path and milestone 41's plain
sparse-attention path: V4-Flash's 4:1 layers have no indexer below 2048
tokens of context (512 compressed keys against a top-512), GLM-5.3 has
none below 2048 tokens (top-2048). It only ever removes work, and it
removes it exactly where short contexts had been paying for a selection
that chose everything.

## The indexer writes its logits

Milestone 58 priced the indexer as a fused streaming op: score, select,
never materialize. That was half right. DeepSeek's public kernel
structure for the lightning indexer computes head-reduced logits — one
fp32 number per (query, key), the per-head scores folded in the epilogue
— and a separate top-k pass reads them back. tinyperf now writes and
reads that matrix. At 8k tokens it is nothing; at a million tokens it is
a terabyte per layer each way and 15% of the prefill. The per-head score
tensor, 64× larger, still never lands: an implementation that
materialized it (the reference does, then runs its ReLU, its head
weights and its head-reduce as separate full-size passes) pays several
times the fused kernel's cost, and that is the largest single
disagreement between the two models at long context — an op-granularity
assumption on their side, stated as such.

## What hyper-connections launch

Manifold hyper-connections widen the residual stream 4× and, per
sub-block, mix it with a tiny projection, run a Sinkhorn normalization on
a 4×4 matrix, gate a 4→1 read before the block and a 1→4 write after it.
Written out as PyTorch ops that is about nineteen kernels a side, thirty-
eight per layer, most of them elementwise passes over a stream four times
the hidden size — and the reference prices exactly that chain. A fused
implementation needs three launches per sub-block: the mix projection, a
read-side kernel and a write-side kernel. tinyperf had two and now has
three; six per layer against thirty-eight. The gap this leaves against
the reference is deliberate and now in the envelope as an assumption:
tinyperf prices the engine DeepSeek's own overhead figures imply, not an
unfused one. On a 43-layer decode step that assumption is worth about
2 ms out of 11 on a B200.

## The bug

At batch 1 every attention component agreed with the reference to within
10%. At batch 8, tinyperf's decode attention came out four times the
reference's, and it grew with batch and context while the reference's
stayed flat. That shape — fine at batch 1, wrong in proportion to
batch × context — pointed at a per-sequence quantity being multiplied by
something that should not multiply it.

Multi-latent and compressed attention keep one latent K/V per sequence
that every query head reads. Milestone 41's absorbed-decode path batched
the attention matmul over *query heads*, one row each, and so charged the
shared latent once per head: sixty-four times its size. The GQA path
never had this problem because it batches over KV heads with the group's
query heads stacked as rows, which is exactly the right structure here
too — group equals all heads. The fix is that one change of batching, on
both paths. GLM-5.3's decode attention at batch 8 shrinks 2.4×, Kimi-K3's
2.6×, V4-Flash's 3×, and all three are now a low single-digit share of
their decode steps. Milestones 41 and 58 carry errata; their qualitative
claims stand, their attention figures were high.

Batch 1 hid this for two milestones because at batch 1 with a short
context the duplicated bytes were a few tens of microseconds per layer,
under the launch floor. Cross-model checks earn their keep on the cells
where the two models scale differently, not on the ones where they agree.

## What the check did not change

The expert GEMMs, the output and query projections, the compressor's
projection, the norms and the head all agreed within the dense envelope
this series has carried since milestone 22 — the skinny-GEMM bandwidth
efficiency and per-kernel floors that separate the projection tier from
the calibrated one. Prefill at short context sits where dense prefill
sits. Nothing in the compressed mechanisms themselves — the pooling, the
window, the top-k cap, the 128:1 dense path — needed a constant.

## One more lesson, about reading someone else's model

The reference's fast inference path builds three instances of the graph:
the prefill, the first decode step, and a last instance that folds the
remaining decode steps into one row, its times and byte counts an exact
multiple. The earlier comparisons had used the non-fast path, where every
step is its own instance, and the parser assumed that. The first table
out of this check said the reference's decode step was nine times
tinyperf's. It was six decode steps. The byte columns caught it — every
GEMM's traffic was exactly six times the first step's — before any model
was touched. Check the shapes before the ratios.

## Exercises

1. The unfused hyper-connection chain costs ~38 launches per layer; the
   fused one 6. Measure a real mHC implementation's kernel count with a
   profiler and place it on that line.
2. Context parallelism for the compressed path would shard the indexer's
   queries, the one thing tensor parallelism cannot split. Add it and see
   where the million-token prefill lands on 8, 16 and 32 GPUs.
3. Re-run the SOTA matrix's GLM and Kimi rows against the earlier
   comparison data with the latent-attention fix: which cells move, and
   do they move toward the reference?
