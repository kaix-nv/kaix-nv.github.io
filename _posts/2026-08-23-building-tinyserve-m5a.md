---
layout: post
math: true
title: "Building tinyserve M5a: Chunked prefill"
date: 2026-08-23 16:48:40 -0700
categories: [tinyserve, llm-serving]
excerpt: "Bound prefill interference with a per-step token budget, pack short prompts, and align chunk attention with a shifted causal mask."
---

*Milestone 5, part 1, of [building an LLM inference engine from scratch](/series/tinyserve/).
Previous: [M4 — continuous batching]({% post_url 2026-08-18-building-tinyserve-m4 %}).*

Code: [`tinyserve` @ `d55b086`](https://github.com/kaix-nv/tinyserve/tree/d55b086ba2828cc5b2e416410661914e3d5e54d7).

## The problem: one prompt can interrupt every decoder

M4 reconsidered batch membership after every engine iteration, but it
still treated each admitted prompt as indivisible work. If a long prompt
arrived while other requests were generating, M4 prefetched the whole
prompt before the next decode. Every streaming request waited behind it.

The retained M4 measurement captured the symptom: at λ=32 requests/s,
median inter-token latency (ITL) was 25 ms while the reported p99 was
408 ms. One number cannot identify the cause of every spike, but the
execution order makes one source of interference certain: decode could
not resume until every newly admitted prompt forward finished.

There are two ways prompt work becomes a long interruption:

1. One long prompt creates one long prefill forward.
2. A burst of short prompts creates many sequential forwards. Even when
   each prompt is small, every model invocation has a non-zero fixed
   cost from Python, framework dispatch, and kernel launches.

Chunking bounds the first problem. Packing is needed for the second.
M5a implements both while preserving a small, explicit scheduler.

## The scheduling invariant

One M5a engine iteration always does work in this order:

1. Admit as many FCFS waiting requests as the KV pool can hold.
2. Decode every request already in `RUNNING`, preempting if decode needs
   more KV blocks than are free.
3. Spend at most `chunk_size` prompt tokens across the FCFS
   `PREFILLING` requests. The default budget is **512 tokens**.

The third step has two cases:

- Pack consecutive whole prompts that fit the remaining budget into one
  ragged batched forward.
- If the FCFS head does not fit whole, run only its next chunk. Later
  requests cannot leapfrog it.

![One engine step decoding first, packing two whole prompts, and spending
the remaining budget on a long prompt chunk](/assets/tinyserve/m5-prefill-budget-step.svg)

*Figure 1. With a 12-token example budget, decode advances A and B first.
The engine then packs whole prompts C and D, consuming seven prompt
tokens in one forward, and spends the remaining five tokens on the head
of long prompt E. C and D emit their first tokens and become `RUNNING`;
E remains `PREFILLING` for the next iteration. The production default is
512 rather than 12—the small number only makes the accounting visible.*

This budget bounds **prompt tokens per iteration**, not wall time. The
next decode still waits for the prompt forwards from the previous
iteration, and their duration depends on batch shapes and backend
overhead. A smaller budget generally shortens the amount of prompt work
between decodes, but it also takes more iterations to finish a long
prompt. `chunk_size` is therefore an ITL-versus-TTFT control, not a
latency guarantee.

## Packing whole prompts

Packing reuses M2's left-padded prefill layout, but only inside one model
forward. Suppose prompt lengths are 4 and 7:

```text
input row 0: [pad pad pad a0 a1 a2 a3]
input row 1: [b0  b1  b2  b3 b4 b5 b6]
position 0:  [  0   0   0  0  1  2  3]
position 1:  [  0   1   2  3  4  5  6]
```

Real tokens receive per-sequence RoPE positions. A causal-and-real-key
mask prevents them from attending to padding. Pad queries are allowed to
see themselves so SDPA never receives an all-masked row and produces a
NaN.

The paged cache needs a physical write slot for every rectangular input
position, including pads. Real positions walk their sequence's block
table. Every pad position points at one scratch slot outside the
allocator-managed KV blocks. Those writes may collide, but no attention
path or block table ever reads the scratch slot, so its final value is
irrelevant.

After the forward, each sequence's real K/V is compact in its own paged
blocks; the rectangular padding disappears with the temporary input.
This is different from M2, where padded rows remain rectangular through
decode.

## Chunk attention needs a shifted causal mask

Let a request already have `S` cached tokens and let its next chunk have
`C` tokens. Chunk query row `i` represents logical position `S + i`, so
it may attend to key position `j` exactly when

$$
M_{i,j} = [j \le S+i],
\qquad 0 \le i < C.
$$

The engine first writes the current chunk's K/V to its allocated slots,
then gathers the request's blocks. The explicit mask exposes all earlier
cached positions plus the causal part of the current chunk, while hiding
future chunk tokens and unwritten capacity in the last block.

![Ordinary local causal masking compared with the shifted causal mask
needed by a chunk after a cached prefix](/assets/tinyserve/m5-shifted-causal-mask.svg)

*Figure 2. The cached prefix has positions 0–3 and the new chunk has
queries 4–6. Ordinary `is_causal` applies the local rule `j <= i`: q4
sees only k0, q5 sees k0–k1, and q6 sees k0–k2. It therefore hides the
recent prefix and the chunk's own valid keys. The shifted rule
`j <= S+i` exposes k0–k4, k0–k5, and k0–k6 respectively. Allocated but
unwritten k7 remains hidden.*

This is why simply calling SDPA with `is_causal=True` is wrong for a
multi-token query appended to a longer K/V sequence. The causal triangle
must be aligned to logical sequence positions, not to the chunk's local
row numbers.

## `PREFILLING`, admission, and preemption

M5a inserts a state between admission and generation:

```text
WAITING --admit--> PREFILLING --final chunk + first token--> RUNNING
   ^                    |                                      |
   |                    +----------- preempt ------------------+
   +---------------- free KV, reset num_cached to zero --------+
```

At admission, the scheduler allocates enough blocks for the request's
entire **current token history**—the prompt for a new request, or prompt
plus generated tokens after a previous preemption. It does not reserve
the request's future output tokens.

Allocating at admission makes capacity accounting immediate and atomic:
the next request sees the reduced free-block count. It also removes M4's
gap between deciding that several prompts fit and allocating their
blocks later.

This is a reservation, not a promise that the blocks can never move.
`reserve_for_decode()` may explicitly select a `PREFILLING` request as a
LIFO preemption victim. Preemption frees every block, clears
`num_cached`, and returns the request to the front of `WAITING`. On
readmission it recomputes from logical position zero; it never resumes
from stale partial K/V.

## The build—and the two designs that failed

The retained development notes record three versions:

**Attempt 1: one chunk from one request per iteration.** The shifted mask
was correct, but admission throughput was capped at one request per
iteration. In the recorded λ=32 run, arrival rate exceeded service rate,
the queue grew, and median TTFT reached roughly one second.

**Attempt 2: spend the budget across requests, one forward per request.**
The queue became stable, but smaller budgets did not improve the ITL tail
as expected. The timing pattern was consistent with repeated per-forward
overhead: a burst of five short prompts still required five model
invocations. No profiler artifact was retained, so the old notes do not
separate Python, framework, launch, and GPU execution time.

**Final design: pack whole prompts; chunk only the oversized head.** The
engine adds:

- `_paged_prefill_batch()` for a ragged batch of complete prompts;
- the `chunk_start` branch in paged attention for the shifted mask;
- `PREFILLING` plus a per-iteration token-budget loop in `serve()`.

An iteration performs at most one packed whole-prompt forward and, if
the next head is too large, one chunk forward. Decode is still a separate
model invocation. Production engines commonly token-pack decode and
prefill work into a unified variable-length forward; M5a deliberately
stops before that complexity so the scheduling invariant stays visible.

## The proof

The complete suite now contains **25 passing tests**. The M5a-specific
coverage in [`tests/test_chunked.py`](https://github.com/kaix-nv/tinyserve/blob/d55b086ba2828cc5b2e416410661914e3d5e54d7/tests/test_chunked.py) and
[`tests/test_scheduler.py`](https://github.com/kaix-nv/tinyserve/blob/d55b086ba2828cc5b2e416410661914e3d5e54d7/tests/test_scheduler.py) checks:

1. **Chunk parity:** 61 random tokens, chunk size 7, block size 16. The
   awkward sizes force chunks across block boundaries. Final fp32 logits
   match whole prefill within `1e-3`, with identical argmax.
2. **Packed-prefill parity:** three ragged rows of 19, 47, and 33 tokens
   exercise left padding and scratch writes. Every row's final logits
   match individual paged prefill within `1e-3`, with identical argmax.
3. **End-to-end chunk parity:** `serve(chunk_size=8)` produces the same
   greedy strings as unchunked `generate_paged()`.
4. **Decode/prefill interleaving:** a long prompt arriving mid-generation
   leaves both its own output and the existing request's output equal to
   their solo runs.
5. **Partial-prefill preemption:** a microscopic pool forces a half-filled
   `PREFILLING` request to lose its blocks. It returns to `WAITING` with
   `num_cached == 0` and is later admitted with fresh blocks.
6. **Invalid budget:** `chunk_size <= 0` raises instead of leaving a
   prefilling request in a no-progress loop.

All numerical parity checks use greedy decoding, fp32, and the PyTorch
gather backend. They establish the code paths used by this milestone;
they do not establish stochastic-sampling equivalence under regrouping,
FlashInfer chunk-prefill parity, or performance portability.

## A retained measurement snapshot

The following numbers come from the original M5a development run, not a
new statistically controlled benchmark. The notes identify the workload
as Qwen3-0.6B on one RTX A6000, bf16 + FlashInfer, 96 Poisson arrivals at
λ=32 requests/s, seed 0, `max_new=128`, and 10% long prompts. The raw
output, exact model revision, complete software environment, and profiler
trace were not retained.

| configuration | reported goodput | reported TTFT p50 | reported ITL p50/p99 |
|---|---:|---:|---:|
| M4, sequential per-request prefill | 660 tok/s | 52 ms | 25 / **408 ms** |
| M5a, packed and effectively unbudgeted | 639 tok/s | 86 ms | 26 / 155 ms |
| M5a, budget **512** | 635 tok/s | 98 ms | 27 / **111 ms** |
| M5a, budget 256 | 547 tok/s | 650 ms | 26 / 114 ms |
| M5a, budget 128 | 407 tok/s | 1840 ms | 54 / 113 ms |

Within this single-run snapshot, packing is associated with the largest
ITL-tail change: 408 to 155 ms before imposing a practical budget. A
512-token budget reports a further reduction to 111 ms with similar
goodput. Smaller budgets sharply increase TTFT and reduce goodput.

The queueing explanation is more precise than “small chunks are slow.”
If mean prompt length is $\bar P$, the offered prompt work is roughly
$\lambda\bar P$ tokens/s. The engine's effective prefill capacity is
approximately the budget divided by average iteration time, adjusted for
packing and prompt shapes. When offered work exceeds that capacity, the
`PREFILLING` queue grows. Because iteration time itself changes with the
budget, the safe threshold must be measured rather than inferred from
`chunk_size` alone.

The retained isolated experiment placed eight short-prompt decoders at
t=0 and one roughly 2.6k-token prompt at t=0.3 s with `max_new=192`:

| budget | reported worst decoder gap | reported long-prompt TTFT |
|---:|---:|---:|
| effectively unbudgeted | 96 ms | 103 ms |
| 512 | **54 ms** | 326 ms |
| 256 | 59 ms | 598 ms |
| 128 | 75 ms | 1080 ms |

This snapshot shows the intended tradeoff: budget 512 roughly halves the
largest observed decoder gap while tripling the long request's TTFT. The
non-monotonic 256/128 tail is consistent with paying fixed forward costs
across more iterations, but the missing trace means it is not proof that
kernel-launch overhead alone caused the reversal.

The 70B/16k-prompt comparison in the original notes was an extrapolation,
not a measurement. Larger models and prompts make prefill interference
more consequential, but the size of that effect depends on hardware,
kernels, parallelism, and workload.

## Reproducing the protocol

The benchmark now has explicit modes for both tables. It prints model,
GPU, dtype, Python, PyTorch, CUDA, FlashInfer, pool size, budgets, and
seeds. Save that output together with the exact Git commit:

```bash
pytest tests/ -q

git rev-parse HEAD | tee m5a-budget-sweep.txt
python examples/bench_serve.py \
  --rates 32 --n 96 --max-new 128 --long-frac 0.1 \
  --chunk-sizes 100000 512 256 128 --seeds 0 1 2 \
  | tee -a m5a-budget-sweep.txt

git rev-parse HEAD | tee m5a-freeze-microscope.txt
python examples/bench_serve.py \
  --freeze-microscope --freeze-arrival 0.3 \
  --freeze-decoders 8 --freeze-long-repeats 80 --freeze-max-new 192 \
  --chunk-sizes 100000 512 256 128 \
  | tee -a m5a-freeze-microscope.txt
```

For this checked-in workload, `100000` is **effectively unbudgeted**: it
is larger than all prompt work that can be admitted in one iteration. It
is not a special infinite-budget code path.

These commands reproduce the current protocol, not the historical table
bit-for-bit. The original raw output and exact prompt text for the freeze
experiment were not retained; the new `--freeze-microscope` mode fixes
that ambiguity for future runs and prints the actual tokenized long-prompt
length. Use multiple seeds, preserve every output file, and report
variability before making hardware or performance claims.

## What M5a teaches

Chunked prefill is not just “split a tensor.” It is a joint contract
between scheduling, masking, and memory accounting:

- decode gets a chance to run before each bounded unit of prompt work;
- packing amortizes per-forward cost across short prompts;
- the shifted causal mask preserves logical attention across chunk
  boundaries;
- upfront block reservation makes admission accounting honest, while
  preemption remains explicit and recoverable;
- a smaller budget helps only while prefill service capacity stays above
  offered prompt work.

That is the minimal complete lesson. Kernel fusion, CUDA graphs, and
production scheduling policy can improve the constants later without
changing these invariants.
