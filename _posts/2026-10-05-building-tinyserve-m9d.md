---
layout: post
math: true
title: "Building tinyserve M9d: When the same history produces different logits"
date: 2026-10-05 08:28:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Use matched token histories to separate BF16 numerical effects from speculative control flow."
source_revision: 8e1849882b75725e9e357cbc136c9f7602c1fb79
source_document: docs/m9d-bf16-numerical-closeout.md
---

{% include tinyserve-article-style.html %}

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Repository snapshot: [`tinyserve` @ `8e18498`](https://github.com/kaix-nv/tinyserve/tree/8e1849882b75725e9e357cbc136c9f7602c1fb79). Measurements below retain their original dates and acceptance limits. Click a figure to open it at full size.

Previous: [M9c — Verify draft tokens without gathering KV]({% include tinyserve-post-url.html slug="building-tinyserve-m9c" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m9c-direct-paged-attention.md" %}) · Next: [M10a — From a blocking engine call to live request streams]({% include tinyserve-post-url.html slug="building-tinyserve-m10a" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m10-online-serving.md" %})

M9c's page-addressing and causal-mask tests passed, but its full-model BF16
logit gate failed. The gather-verification control also failed. That leaves
an important question: is the difference introduced by the new attention
reader, by processing several query rows together, or by both?

M9d is a bounded numerical investigation, not another serving feature. It
keeps the serving code and the qualification thresholds unchanged. The
result should explain the remaining limitation, not silently redefine what
counts as passing.

## Compare predictions before the histories branch

Once two greedy runs choose different tokens, every later comparison mixes
numerical error with genuinely different input text. Teacher forcing avoids
that problem: first generate an ordinary reference continuation, then give
each candidate exactly those input tokens, regardless of its predictions.

For reference outputs `[y0,y1,y2,y3,y4]`, aligned logit row `i` predicts
`yi` from `prompt + outputs[:i]`:

| Output row | Inputs processed to produce it |
| --- | --- |
| 0 | Prompt prefill; keep its final prediction |
| 1 | Append `y0` |
| 2 | Append `y1`, after `y0` |
| 3 | Append `y2`, after `y0,y1` |
| 4 | Append `y3`, after `y0,y1,y2` |

Five output predictions need only four continuation inputs. Feeding `y4`
would produce a prediction outside this five-output fixture. M9c's earlier
probe omitted the prefill row and examined only eight continuation inputs;
M9d aligns every output position in an up-to-32-token continuation, including
positions where the earlier free-running fixture diverged.

[![Matched-history replay changes query grouping or the attention reader,
but never lets a candidate prediction change the next input.](/assets/tinyserve/m9d-matched-history.svg)](/assets/tinyserve/m9d-matched-history.svg)

Grouping inputs changes execution, not the intended causal history. For
example, `[y0,y1,y2]` can be processed together to predict `[y1,y2,y3]`, with
the causal mask preventing each row from reading later inputs. A final
short chunk is allowed. The query counts here are **1, 3, and 5**, analogous
to one-token decode and verification with two or four proposals.

These are fixed teacher-forced groups, not a replay of the actual draft's
proposals, rejected tokens, or rollback schedule. This isolates numerical
behavior on matched text; it does not establish lossless speculative
generation or task accuracy.

## Change one axis at a time

The diagnostic runs the same history through this grid:

| KV/attention path | One query | Three queries | Five queries |
| --- | --- | --- | --- |
| Contiguous Torch | decode | causal append | causal append |
| Paged Torch gather | decode | causal append | causal append |
| Direct FlashInfer | paged decode | paged append | paged append |

Read a row to investigate query-shape effects. Read a column to investigate
cache/attention-path effects at the same query count. Changing query count
can also select a different attention implementation, so a row comparison
does not by itself attribute all error to GEMM. The first-layer trace helps
separate these mechanisms.

BF16 covers all nine modes; FP32 covers the six Torch modes because the
direct FlashInfer adapter does not support FP32. Each dtype generates its
own ordinary reference. Agreement within FP32 is not an assertion that
BF16 and FP32 must generate identical text.

## What the diagnostic records

[`diagnose_speculative.py`](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/examples/diagnose_speculative.py) reuses the
M9c logit comparator. Its gate is still: finite values, raw-logit NRMSE at
most 0.01, and mean KL divergence at most 0.01. Every comparison names its
reference explicitly. No threshold is tuned after seeing the result.

The additional fields explain the result rather than replacing the gate:

- Per-output top-1 choices and the first differing position.
- The top-two logit gap, including exact BF16 ties.
- Each model's preference between the two differing choices.
- Mean-centered NRMSE, which removes a per-row constant logit offset.

Softmax is unchanged by a constant offset. That makes centering useful for
diagnosis, but not a license to declare a failed raw-logit gate passed. A
unit test explicitly adds a constant offset and checks that the original
gate still fails despite unchanged probabilities and greedy choices.

The predeclared detailed trace is the code-writing prompt, index 3. Hooks
capture its first-layer input normalization, Q/K/V projections, attention
output, MLP output, each decoder-layer output, and final normalization.
All captures are aligned to the same output rows and copied to CPU. This
instrumentation synchronizes execution; **none of its durations are
performance measurements**. Hooks and private pages are released even if
a forward fails.

## Results and decision

The September 20 A6000 run used Qwen3-4B, PyTorch 2.13.0+cu130, and
FlashInfer 0.6.14. All six prompts generated 32 reference tokens, so every
mode was checked at all 192 aligned output positions per dtype.

| Evidence | FP32 | BF16 |
| --- | ---: | ---: |
| Pairwise prompt comparisons | 42 | 90 |
| Passed the unchanged logit gate | 42/42 | 0/90 |
| Largest aggregate logit NRMSE | 0.000377% | 3.123% |
| Largest mean KL | 0.0000000166 | 0.001847 |

FP32 retained every top-1 choice across all comparisons. BF16's logit gates
all failed NRMSE, not finiteness or KL. These are aggregate errors over
32 predictions and the whole vocabulary; per-position errors are retained
in the raw results. The pair counts overlap references and are not 90
independent prompts.

Against contiguous one-query decoding, the BF16 teacher-forced agreement
counts were:

| Candidate | Prompts with all 32 choices matching |
| --- | ---: |
| Contiguous, three or five queries | 4/6 each |
| Gather, one query | 5/6 |
| Gather, three or five queries | 4/6 each |
| Direct, one query | 6/6 |
| Direct, three queries | 5/6 |
| Direct, five queries | 4/6 |

The newly covered disagreements begin at **output index 14** on the code
prompt and **index 13** on the KV-caching prompt. Indices start at zero, so
these are the fifteenth and fourteenth output tokens. Those positions were
outside M9c's eight-input probe. Agreement under teacher forcing does not
mean an actual draft/verify run follows the same grouping or generates the
same continuation.

### The trace separates two sources

On the code prompt, first-layer normalized inputs were bitwise identical
across query sizes. Yet the first Q/K/V projections already differed when
one-token forwards became five-token forwards. These differences precede
the cached attention read. Changing only the attention reader with five
queries kept those projections identical; differences then appeared inside
attention.

Concretely, this model's normalized input has shape `[1,T,2560]`, and its K
projection weight has shape `[1024,2560]` (eight KV heads of width 128).
Each row still asks for the same dot products in `X @ Wk.T`, whether `T=1`
or `T=5`. The observed BF16 outputs nevertheless differ with the row count.
This locates a shape-sensitive numerical effect before attention; it does
not identify the underlying GEMM algorithm, which this experiment did not
profile.

| Captured stage | Gather T=5 versus gather T=1 NRMSE | Direct T=5 versus gather T=5 NRMSE |
| --- | ---: | ---: |
| First-layer input normalization | 0% | 0% |
| First-layer K projection | 0.221% | 0% |
| First-layer attention output | 0.318% | 0.097% |
| Decoder layer 17 output | 1.216% | 1.178% |
| Final normalization | 1.296% | 1.251% |

These are activation diagnostics, not the vocabulary-logit qualification
gate. Layer indices start at zero. Errors do not have to increase at every
layer, and an absolute activation difference is not meaningful without its
scale. Both comparisons first change a continuation row at output index 1;
their shared prefill prediction remains unchanged.

The inference is limited but useful: shape-dependent projection rounding
and attention-backend differences both contribute before the final head.
This is not evidence that only the last vocabulary projection is responsible,
or that the new direct reader is the sole source of drift. The earlier
causal-mask, page-poisoning, rollback, and FP32 tests still matter: a small
logit discrepancy alone would not prove cache addressing correct.

### Small errors can change a greedy decision

On the code prompt, contiguous ordinary decoding tied token `def` (ID 750)
with `Wait` (ID 14190). Argmax chose the lower ID, `def`. Five-query direct
execution instead preferred `Wait` by 0.125. On the KV prompt, an ordinary
preference gap of 0.125 became a tie, selecting a different lower-ID token.

| Prompt / output index | Token | Contiguous T=1 logit | Direct T=5 logit |
| --- | --- | ---: | ---: |
| Code / 14 | `def` (750) | 22.375 | 22.250 |
| Code / 14 | `Wait` (14190) | 22.375 | 22.375 |
| KV / 13 | ` Without` (17147) | 20.125 | 20.125 |
| KV / 13 | ` For` (1752) | 20.000 | 20.125 |

At these magnitudes, adjacent BF16 values are 0.125 apart. Moving a score
by one representable step can therefore create or break the tie.

A zero-margin decision has no rounding tolerance: an arbitrarily small
change can select another token. That is why low average KL, a small RMS
error, and identical greedy output are different contracts. This is also
why matching six short ordinary runs cannot establish universal BF16 parity.

Mean-centering did not eliminate the measured drift. It remains a diagnostic
only, and all original failed gate results are preserved. No replacement
tolerance or accuracy claim is introduced.

### Closeout

This investigation did **not** demonstrate an allocation, causal-mask,
acceptance, or rollback bug to fix. That is a scoped finding, not a proof of
correctness on arbitrary models or prompts. Serving code is unchanged:
FP32 remains the reference, BF16 speculation remains experimental, and the
ordinary serving default is unchanged.

There is consequently no new performance result to claim. The
[M9c paired timings and fresh cross-engine calibration]({% include tinyserve-post-url.html slug="building-tinyserve-m9c" fallback="https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/m9c-direct-paged-attention.md" %}#diagnostic-performance)
remain the latest measurements. No serving-code fix was made that would
require rerunning those comparisons. M9 closes here as a single-request
greedy teaching implementation; sampling, scheduler integration, shared
prefixes, and graph optimization remain deferred.

The [M9d evidence receipt](https://github.com/kaix-nv/tinyserve/blob/8e1849882b75725e9e357cbc136c9f7602c1fb79/docs/benchmarks/m9d-bf16-closeout-a6000-2026-09-20.json)
retains the named comparisons, failed gates, mismatch details, layer trace,
source hashes, and validation commands. Instrumented and uninstrumented
logits are checked for exact equality before accepting a trace.

Final validation reported **374 tests passed, 9 skipped**, including 13 new
diagnostic tests. Skips are not counted as verified behavior. The figure was
rendered and checked for text overflow; no serving implementation was
modified to obtain these results.

## Inspect it locally

The `M9d: explain BF16 speculative drift` debugger configuration runs the
code-writing fixture. Break in `replay()`, `compare()`, or
`compare_stages()`. In a per-position record, `output_index` refers to a
prediction, not to the input token processed at that step.

The diagnostic creates no draft model: it studies the target's numerical
behavior on fixed histories. M9a's acceptance loop, M9b's private-page
rollback, and M9c's direct reader remain the serving implementations.
