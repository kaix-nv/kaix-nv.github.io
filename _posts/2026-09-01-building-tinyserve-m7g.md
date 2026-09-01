---
layout: post
math: true
title: "Building tinyserve M7g: Hybrid state becomes a scheduler resource"
date: 2026-09-01 13:35:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Give every hybrid request private recurrent and growing history, then gather ragged live state for batched decode without merging ownership."
---

*Milestone 7g of [building an LLM inference engine from scratch](/series/tinyserve/).
Previous: [M7f — Kimi is KDA + MLA + MoE]({% post_url 2026-09-01-building-tinyserve-m7f %}).*

Code: [`tinyserve` @ `1099214`](https://github.com/kaix-nv/tinyserve/tree/1099214bd1e59e41eb511d463db331f8fe1937b9).

M7e and M7f could generate one sequence correctly. They could not serve a
stream. Each `generate()` call created one cache, mutated it until completion,
then discarded it. There was no answer to three basic serving questions:

1. Which request owns this recurrent matrix?
2. Where does its growing full-attention or MLA history live while another
   request runs?
3. When a request finishes, what capacity becomes available to the FCFS queue?

M7g makes those answers explicit. The scheduler owns a bounded number of
resident hybrid-state slots. Each admitted request gets one private cache.
Execution may batch several caches, but ownership never becomes batch-wide.

![Three requests share two hybrid-state slots; live state gathers for one
decode step and scatters back to its owners](/assets/tinyserve/m7g-hybrid-state-lifecycle.svg)

## A request owns more than token history

Ordinary Qwen3 scheduling mostly moves block tables around. The model writes
one K/V pair per token, and the request's block table says where that history
lives. A hybrid request owns several state families at once:

| model | fixed per-request state | growing per-token state |
|---|---|---|
| Qwen3.5 | Gated DeltaNet convolution tails and recurrent matrices | K/V in six full-attention layers |
| Kimi-K3 | three convolution tails and a channel-wise KDA matrix per KDA layer | compressed `(latent, designated key)` history in five MLA layers |

The fixed tensors cannot be reconstructed by looking at the newest token. They
summarize the entire prefix. Accidentally giving request B request A's matrix
does not cause a shape error; it silently makes B continue A's memory.

M7g therefore adds two fields to `Request`:

```python
state_slot: int | None
model_state: HybridCache | KimiCache | None
```

`HybridScheduler.schedule()` removes one id from a fixed free-slot queue and
constructs that request's complete cache. `finish()` drops the cache reference
and returns the id. A waiting request cannot enter until a slot is available.

This is reservation rather than reactive allocation: the growing arrays are
sized for `prompt_length + max_new_tokens` at admission. Decode cannot later
discover that its next token needs an unavailable block, so M7g does not need
recompute preemption yet.

## Follow three requests through two slots

Use the debugger example with:

```text
A = "2+2="
B = "Kimi"
C = "A longer prompt"
resident slots = 2
```

All three arrive, but only A and B enter:

```text
slot 0 -> A cache     slot 1 -> B cache     waiting -> C
```

The engine prefills A. On the next iteration it decodes A before prefilling B.
Once both are `RUNNING`, one decode forward contains two rows. A has the longer
history, but B does not inherit A's padding or cursor.

After A emits its third token, the scheduler records:

```text
release request A, slot 0
admit   request C, slot 0
```

C receives a newly zeroed recurrent state and an empty history even though the
numeric slot id is reused. B remains in slot 1 without moving. The retained
test trace is:

```text
admit A/0, admit B/1, release A/0,
admit C/0, release B/1, release C/0
```

This is the lifecycle invariant: **a slot may be reused across time; state may
not be reused across requests.**

## Batching execution without merging ownership

The individual caches have different cursors. Suppose A has six cached tokens
and B has three. A one-token decode needs shapes that a normal rectangular
cache cannot express:

```text
A fixed state: [layers, 1, ...]    A history: [layers, 1, 6, ...]
B fixed state: [layers, 1, ...]    B history: [layers, 1, 3, ...]
```

M7g uses a deliberately readable gather/forward/scatter transition:

1. concatenate fixed convolution and recurrent tensors along batch;
2. copy growing histories into a zero-padded buffer sized for the longest row
   plus the new token;
3. build a visibility mask from the per-row cursors;
4. run one model forward for the newest token of every live request;
5. scatter the updated fixed state and newly appended history entry back to
   each request cache.

For the example, the visibility rows after appending are:

```text
A: 1 1 1 1 1 1 1
B: 1 1 1 1 0 0 0
```

Qwen3.5 applies that mask to its full-attention layers. Kimi applies it to MLA
scores before aggregating compressed latents. KDA and Gated DeltaNet do not
need a history mask: their recurrent matrices already have one batch row per
request.

The copies are not a performance endpoint. They are the smallest mechanism
that proves ragged execution does not change ownership. A persistent slot
kernel can later replace gather/scatter while preserving this interface.

## The padding bug that looked impossible

The first Kimi batch passed the attention mask and still produced token id 0
for the shorter row. Its logits were NaN.

The padded latent buffer had been allocated with `empty()`. Softmax correctly
assigned zero probability to padded positions, but the following matrix
multiplication still encountered `0 × NaN`. IEEE floating point keeps that as
NaN, so masking the score was too late.

M7g zero-initializes every padded K/V or latent history buffer. The mask
controls visibility; finite zero padding controls numerical safety. A ragged
implementation needs both.

## Engine-step order

The hybrid loop preserves the most important M4 policy: decode before prefill.

```text
admit arrivals while free slots exist
gather every RUNNING request -> one decode forward -> scatter
sample and release any finished request
prefill one whole FCFS PREFILLING request
sample its first token; mark it RUNNING
```

Only one new prompt runs per step, so a burst cannot trigger one prefill
forward per admitted request before decoding resumes. A single long prompt can
still delay decoders because M7g does not split the hybrid recurrence yet.
That is the concrete motivation for M7h.

## Validation

The acceptance ladder checks ownership separately from model algebra:

| gate | result |
|---|---|
| one-slot scheduler unit test | second FCFS request waits, then reuses released slot 0 with a new state object |
| ragged Kimi decode | unequal histories remain finite and match independent generation |
| staggered Kimi serving | three requests / two slots are text-identical to three solo runs |
| staggered Qwen3.5 serving | three requests / two slots are text-identical to three solo runs |
| M7e/M7f plus scheduler regression | `23 passed` together, including the three new M7g tests |
| milestone-wide regression | all `89` collected tests pass when the memory-heavy chunked and prefix groups run in fresh processes |

The comparison is against `generate()`, not just another scheduler layout.
That keeps the single-request M7e/M7f paths as semantic oracles.

The process split matters on the 48 GiB validation GPU: one monolithic run
passes 77 non-prefix tests, then two FP32 chunked-prefill cases request a
14 GiB paged pool after earlier fixtures have retained memory. The complete
chunked file passes `6/6` in a fresh process, and prefix caching passes `10/10`
in another. This is allocation isolation, not a relaxed correctness gate.

## Same-checkpoint calibration

The September 1, 2026 snapshot used the tiny random Kimi-K3 checkpoint, one
RTX A6000, BF16, four fixed raw-text prompts, eight generated tokens per
request, and two opposite-order sweeps. Each sweep used one warmup and three
measured repeats; the table reports the median of all six measurements.
Loading, MXFP4 decompression, and first-shape compilation were outside the
timed rows.

| execution path | batch 1 | batch 2 | batch 4 |
|---|---:|---:|---:|
| Tinyserve sequential `generate()` | 7.29 tok/s | 7.49 tok/s | 7.46 tok/s |
| Tinyserve M7g `serve()` | 7.30 tok/s | 11.32 tok/s | 16.03 tok/s |
| Transformers + FLA static batch | 8.64 tok/s | 16.63 tok/s | 31.74 tok/s |

M7g is neutral at batch 1, 1.51× faster than sequential Tinyserve at batch 2,
and 2.15× faster at batch 4. That is the expected mechanism win: several
request rows now share each model forward.

The result is not an order artifact. Comparing medians within each sweep gives
1.56× at batch 2 in both orders and 2.13–2.16× at batch 4. Temperatures stayed
between 53–61 °C. The 250 ms monitor caught Tinyserve at both idle and boosted
clocks (210–1,800 MHz) because its short kernels are separated by Python work;
the fused FLA path stayed at 1,800 MHz. The opposite-order sweep is therefore
part of the protocol rather than an optional cleanup.

Transformers remains 1.18×, 1.47×, and 1.98× faster respectively. Its row is
a same-checkpoint batching calibration, not a continuous-scheduler comparison:
it receives one fixed padded batch and uses optimized FLA transitions. M7g
copies ragged state every token and loops through routed experts in Python.

The random fixture has near-tied logits, so different BF16 associations can
change generated strings across engines. Cross-engine correctness remains the
fixed-prompt M7f logit/top-five gate. Within Tinyserve, solo and scheduled
strings were identical for every retained workload. Each JSON artifact records
both the base Git revision and a SHA-256 fingerprint over the exact M7g runtime
sources, so these review-build measurements do not masquerade as M7f results.

## Run and debug it

```bash
.venv/bin/python examples/generate.py \
  --model /home/scratch.kaix_coreai/models/kimi-k3 \
  --serve --hybrid-slots 2 \
  --prompts "2+2=" "Kimi" "A longer prompt" \
  --no-chat --max-new-tokens 3 --backend gather --verbose
```

The **M7g: scheduler-owned hybrid state** launch configuration runs this
example. Set breakpoints in this order:

1. `HybridScheduler.schedule()` — watch C remain waiting after slots 0 and 1
   are assigned;
2. `KimiBatchCache.__init__()` — compare `start_lens` and the zero-padded
   latent histories;
3. `KimiBatchCache.update_mla()` — inspect the per-row `visible` mask;
4. `_serve_hybrid()` after `batch_cache.commit()` — see updated state return to
   the same request objects;
5. `HybridScheduler.finish()` — watch slot 0 move back to `free_slots`, then
   get assigned to C.

The retained calibration uses:

```bash
.venv/bin/python examples/bench_hybrid_serve.py \
  --adapter tinyserve-serve --device cuda:1 \
  --output .tinyserve-bench/m7g/kimi-serve.json
```

## What M7g establishes

M7g turns Qwen3.5 and Kimi state into an explicit serving resource. Arrivals,
FCFS admission, bounded residency, ragged batched decode, completion, and slot
reuse no longer depend on a single `generate()` call.

It does not yet implement hybrid prefix sharing, preemption, chunked or packed
prefill, CUDA graphs, persistent state slots, distributed execution, or a
fused recurrent scan. M7h should first restore chunked/packed prefill so one
long hybrid prompt cannot monopolize the interval between decoder steps.
