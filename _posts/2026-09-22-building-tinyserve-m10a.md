---
layout: post
math: true
title: "Building tinyserve M10A: From a blocking engine call to live request streams"
date: 2026-09-22 00:00:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "Live submission, token events, cancellation, and single-owner KV lifetime: the core of an online serving engine."
---

<style>
.post-content table { display: block; max-width: 100%; overflow-x: auto; }
.post-content :not(pre) > code { overflow-wrap: anywhere; }
</style>

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Code and evidence: [`tinyserve` @ `e533be6`](https://github.com/kaix-nv/tinyserve/tree/e533be67d15772292b1acbce44ce67dcadac4222). Click any figure to open it at full size.

Previous: [M9d — Numerical closeout](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/docs/m9d-bf16-numerical-closeout.md).

**Status: M10a implemented and validated within the scope below.** The session API, token
events, and cancellation below are implemented for single-device, greedy,
ordinary dense Qwen. The HTTP/text adapter is covered separately in
[M10b — HTTP streaming]({% post_url 2026-09-22-building-tinyserve-m10b %}).
This chapter began as the design proposal; the examples now describe the
initial implementation, not a production server.

Before M10, tinyserve could already keep several requests in flight, admit new work at
scheduled arrival times, and reuse a finished request's KV capacity. But its
caller had to supply the entire request list before `serve()` started. It got
completed results back only when that call returned.

An interactive application needs a different boundary. Alice starts a
generation; Bob submits a prompt while Alice is still reading; Alice clicks
Stop; Carol arrives and should be able to use the released capacity. The
engine cannot know that whole conversation at call entry.

M10 opens that boundary without changing the transformer, attention kernels,
or scheduling policy. The first step is an engine that can accept commands
and return token events between execution steps. A network interface comes
after that mechanism is understandable and tested.

## 1. What “online serving” adds

Several related ideas solve different problems:

| Concept | Question it answers | Tinyserve today |
| --- | --- | --- |
| Autoregressive generation | What is the next token for this history? | Implemented |
| Continuous batching | Which requests participate in the next iteration? | M4, implemented |
| Chunked prefill | How much prompt work runs before the next decode opportunity? | M5a, implemented |
| Packed prefill | Can several prompts share a model forward? | M5a and later packed paths, implemented |
| Live request intake | Can an unknown future request join an already-running engine? | M10a |
| Incremental output | Can the caller observe tokens before the whole workload finishes? | M10a, at step boundaries |
| Request cancellation | Can the caller stop one request and release its ownership? | M10a, at step boundaries |
| HTTP streaming | How do remote clients exchange commands and output? | Separate M10b adapter |

M10 does **not** introduce continuous batching for the first time. Nor does
streaming make generation non-autoregressive: the model still needs an
earlier token before it can predict a later one. Streaming changes when the
caller can observe results.

“Asynchronous” also does not mean two Python handlers should run the model
at once. Clients can be concurrent while one engine owner serializes
scheduler decisions and launches batched GPU work. Batching, concurrent
clients, and simultaneous kernel execution are different things.

## 2. The boundary before M10

The original ordinary loop, retained as
[`LLM._serve_legacy()`](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/tinyserve/engine.py), works as follows:

1. Templates and tokenizes the supplied prompt list.
2. Builds every `Request`, its sampling parameters, and its arrival time.
3. Keeps future arrivals in a local `pending` deque.
4. Runs the scheduler/model loop until pending and active work are empty.
5. Returns a list of completed `RequestOutput` objects.

`arrivals=[0.0, 0.1, ...]` delays when known requests become eligible. It is
useful for load experiments, but it is not an external submission channel.
The model computes tokens incrementally inside the loop; the public call
does not expose those tokens incrementally. Its local `emit()` helper
appends tokens, records timing, and constructs a completed output.

This is a **closed workload**: the complete membership is known upfront.
An online engine must support an **open workload**: it can become idle,
receive another request later, and continue without rebuilding its model,
KV pool, or scheduling session.

[![M10 separates the caller, a single-owner serving session, and the existing
GPU execution helpers. M10b adds the HTTP adapter.](/assets/tinyserve/m10-serving-boundaries.svg)](/assets/tinyserve/m10-serving-boundaries.svg)

The important change is lifetime. Model weights and the physical KV pool
already outlive an individual request. The scheduler and request registry
must also outlive an individual caller's submission.

## 3. The approach: expose one engine step

M10a adds a small, synchronous [`ServingSession`](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/tinyserve/session.py)
over one `LLM`. It retains the existing forward helpers and scheduler.

| Operation | Responsibility | Runs a model forward? |
| --- | --- | --- |
| `submit(prompt, params, chat=...) -> request_id` | Validate, tokenize, assign a session-unique ID, and enqueue the request | No |
| `step() -> list[Event]` | Resolve pending cancellations, schedule one iteration, execute it, and return its events | Yes, if work is runnable |
| `cancel(request_id) -> bool` | Mark a live request for cancellation at the next step boundary | No |
| `has_work() -> bool` | Report pending cancellation or unfinished request work | No |
| `close() -> list[Event]` | At a safe boundary, cancel remaining requests, release their references, and close the session | No new forward |

One owner calls these methods serially. `step()` is synchronous: it may
wait for model execution and sampled IDs. It does not wait for a future
request to arrive. An idle, open session returns an empty event list; the
caller decides when to wait or submit more work. Idle is not closed.

Only one active session may own an LLM's scheduler-facing cache at a time.
Calling another generation API or opening a second session on that same LLM
while the first is active must not create competing owners. This is an
explicit restriction, not a promise of thread safety.

For example, given an existing `llm`, sampling `params`, and an event consumer
called `handle`:

```python
session = llm.serving_session(chunk_size=4, backend="auto")
try:
    alice = session.submit("Alice's prompt", params, chat=False)
    handle(session.step())

    bob = session.submit("Bob's later prompt", params, chat=False)
    handle(session.step())

    session.cancel(alice)
    while session.has_work():
        handle(session.step())
finally:
    handle(session.close())
```

The external caller can run application logic between steps. Later, a
network worker can do the same with queued commands. M10a needs neither an
HTTP framework nor a second scheduler.

### What stays inside a step

The step preserves the existing ordinary serving order:

1. Apply cancellation requests before selecting any new GPU work.
2. Admit waiting requests in the existing first-come, first-served (FCFS) order.
3. Reserve decode capacity, retaining the current preemption policy.
4. Batch-decode the selected `RUNNING` requests and record their tokens.
5. Spend at most `chunk_size` prompt tokens across `PREFILLING` requests,
   retaining whole-prompt packing and partial-chunk execution.
6. Return ordered token and terminal events; keep unfinished requests for
   the next call.

This is one scheduler iteration, **not one CUDA kernel**. It may contain a
decode forward followed by several prefill forwards. It does not drain a
long prompt or a whole request before returning control to the caller.

The existing states keep their meaning: `WAITING` has not been admitted,
`PREFILLING` is filling its cached history, `RUNNING` can decode a next token,
and `FINISHED` is terminal. A preempted request returns to `WAITING` so its
history can be reconstructed on readmission.

The existing `_paged_decode()`, `_paged_prefill_batch()`, and
`_paged_chunk_prefill()` remain the GPU execution helpers. Their Torch,
FlashInfer, and eligible CUDA-graph dispatch should not be reimplemented
inside a network handler.

## 4. A concrete four-step example

Use a four-token prefill budget and three requests:

- **A:** five prompt tokens; up to four output tokens.
- **B:** three prompt tokens; up to two output tokens; submitted after step 1.
- **C:** two prompt tokens; up to two output tokens; submitted after step 2.

A is cancelled after step 2. Assume no EOS, no preemption, and enough KV
capacity. For a visible memory example, use eight **illustrative four-token
pages**, with prefix caching disabled. The existing default page size is
16; the smaller pages here make the arithmetic readable.

`a0`, `b0`, and so on name hypothetical output tokens, not measured model
predictions. The deterministic lifecycle test in
[`tests/test_session.py`](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/tests/test_session.py) checks this ordering and
page accounting independently of a checkpoint's particular output tokens.

| Step | Commands before the step | Decode phase | Prefill phase, budget 4 | Events returned |
| --- | --- | --- | --- | --- |
| 1 | Submit A | None | A: first 4 of 5 prompt tokens | None |
| 2 | Submit B | None | A: remaining 1; then B: all 3 | A:`a0`, B:`b0` |
| 3 | Cancel A; submit C | B: consume `b0`, emit `b1`, finish | C: all 2 | A:cancelled; B:`b1`, B:length; C:`c0` |
| 4 | None | C: consume `c0`, emit `c1`, finish | None | C:`c1`, C:length |

[![Four engine steps show live arrivals, a shared prefill budget,
token events, and cancellation without stopping the surviving requests.](/assets/tinyserve/m10-live-request-steps.svg)](/assets/tinyserve/m10-live-request-steps.svg)

Step 2 uses **two forwards**: a one-token cached append for A, then a fresh
three-token prefill for B. They share a four-token budget; they are not a
single fused prefill/decode batch. Packing several fresh whole prompts is
still a different case, handled by the existing packing helper.

After step 2, A has five cached prompt tokens and one emitted but **uncached**
token, `a0`. B similarly has three cached prompt tokens and pending `b0`.
The next decode would consume each pending token and predict its successor.
Cancelling A prevents that future decode; it does not retract `a0`.

The page ownership is equally concrete:

| Boundary | Request-owned pages, with illustrative physical IDs |
| --- | --- |
| After step 1 | A owns `[0,1]`: admission reserved both pages, even though only four prompt tokens were computed |
| After step 2 | A owns `[0,1]`; B owns `[2]` |
| Beginning of step 3 | Cancellation releases A's references; C can allocate a now-free page such as `[0]` |
| After step 3 | B has finished and released `[2]`; only C owns `[0]` |
| After step 4 | No request owns a page; all eight pages are free under this no-prefix-cache assumption |

Physical IDs are chosen for illustration, not a guarantee about the
allocator's free-list order. Submission itself allocates no KV pages;
admission and decode growth retain their existing allocation responsibilities.

## 5. Cancellation is an ownership operation

Removing a request from a list is insufficient. A request can own a block
table, live references to shared prefix blocks, sampling state, and pending
output metadata. Cancelling it means that no later step may select it and
its request-owned resources must be released exactly once.

| State when cancellation is applied | Required action |
| --- | --- |
| `WAITING`, including a preempted request | Remove it from the queue; do not admit or recompute it |
| `PREFILLING` | Remove it from the resident set and release its reserved page references, including not-yet-written space |
| `RUNNING` | Remove it from future decode batches and release its page references |
| Already terminal or unknown ID | No-op; never release another reference or emit a second terminal event |

A cancelled request becomes terminal with reason
`cancelled`; no separate resumable cancellation state is needed.
`cancel()` returns true only when it newly marks a live request. Repeating
it, cancelling an unknown ID, or cancelling a completed request returns
false. Request IDs are not reused within a session.

### Why wait for a step boundary?

A host-side cancellation does not safely interrupt a CUDA kernel that is
reading that request's KV pages. Reassigning those pages while the old
forward still uses them would mix requests' data.

M10a therefore accepts commands only between synchronous `step()` calls.
`step()` synchronizes pending CUDA work before returning. This matters for
partial prefill: a chunk that does not finish the prompt has no sampled
token to copy to the CPU, so it cannot rely on token materialization to
implicitly wait for the GPU.
In the M10b HTTP adapter, a cancellation arriving during a forward sets a
per-request flag. The owner handles it after the current step completes,
before planning the next one. If the request already finished in that step,
the later cancellation is a no-op. Already emitted or buffered tokens cannot
be recalled.

Cancellation responsiveness consequently depends on the remaining duration
of the current step. A token-count prefill budget limits prompt work, but is
not a hard millisecond deadline. Cold compilation, graph capture, prompt
length, and kernel costs still matter.

### Released references are not necessarily empty VRAM

The example disabled prefix caching. With prefix caching enabled,
[`free_sequence()`](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/tinyserve/paged.py) drops the request's references;
it does not destroy every physical page:

- A prefix block still referenced by B must remain valid when A is cancelled.
- A registered full block with no live request references can remain warm
  and reclaimable under the existing cache policy.
- Private, unregistered pages can return to the free list.

The physical pool tensor stays allocated for reuse. Tests must check
reference counts and `cache.num_available`, not demand that every block
be unregistered or that GPU memory usage fall to zero. Cancellation releases
an owner; cache policy decides what reusable content remains.

## 6. Token events are not text fragments

The engine boundary returns token IDs, not network messages:

```text
TokenEvent(request_id, token_index, token_id, emitted_at)
FinishedEvent(request_id, reason, finished_at, stats)
```

`token_index` is zero-based within that request's generated sequence. Token
events retain sampled IDs, including EOS. A text adapter hides EOS; the
legacy `RequestOutput` adapter preserves its current EOS-stripping behavior.
Keeping token identity explicit makes parity and output ordering testable.

The terminal `stats` carry existing request counters such as preemptions
and reused prompt tokens. Together with the token events and arrival time,
they let the compatibility driver reconstruct today's completed output
and timing fields. After creating the terminal event, the session can remove
the request from its active registry; it must not retain every completed
request's token history forever. IDs remain unique through a counter, not
an ever-growing archive of finished requests.

A sampled EOS produces its token event followed by a terminal `eos` event.
A budget-ending token produces its token event followed by `length`; EOS
takes precedence if both conditions occur together. Cancellation produces
only the terminal event at the next boundary. A zero-output budget
finishes without running prefill. An accepted request gets one terminal event
under normal execution, and no token event after it.

This is a local lifecycle guarantee, not durable exactly-once delivery over
a network. A failed process cannot promise that a client received an event.

Detokenization is a separate concern. Token pieces are not necessarily
complete words or independently printable characters, so blindly concatenating
`decode([token])` results is not a sufficient streaming-text contract. M10b
must maintain per-request decoding state, emit only stable text, and flush
any remaining text at termination. Its checks must include byte fragments,
whitespace, special tokens, and agreement with final whole-sequence decoding.

### Engine emission time and client-visible time differ

M10a returns an event list at the **end** of a step. A token sampled during
decode can therefore wait behind that step's subsequent prefill work before
the caller receives it. This first design streams at step boundaries; it
does not claim to overlap network transmission with GPU execution.

Record the token's engine emission timestamp separately from when the
caller receives the list. Later HTTP measurements add transport and client
timing. Report where tokenization, queueing, and output buffering fall in
each timing boundary; moving them outside a timer is not a speedup.

There is one measurement correction relative to the retained loop: decode
events are timestamped **after** sampled IDs are materialized on the CPU.
The earlier loop could stamp decode before its `.tolist()` wait. Therefore
paired performance comparisons use complete call wall time and actual token
counts, not differences in those internal timestamp conventions.

## 7. Keep the scheduling machinery; move its lifetime

The code change is an extraction, not a new policy:

| Existing piece | Role in M10a |
| --- | --- |
| `LLM` and `_paged()` | Continue owning model weights, physical KV, backends, and graph resources |
| `Request`, `Sequence`, `Scheduler` | Retain per-request state and allocation policy; add targeted cancellation handling |
| Local `serve()` request registry and scheduler | Become session-owned state that persists across calls to `step()` |
| Local `emit()` logic | `ServingSession._emit()` is the new path's sole token-append/finish function; it produces ordered events |
| Paged forward helpers | Execute the selected work unchanged |
| Existing `serve(prompts, arrivals=...)` | `serve_offline()` drives the new core for the supported path, preserving completed outputs and arrival semantics |

The compatibility driver knows all requests upfront, submits them at the
specified arrival times, repeatedly steps, and collects results. Waiting
for a future scheduled arrival belongs in that driver, not inside the live
session's `step()`. A real client can instead submit a request that did not
exist when the session was created.

M10a starts with ordinary dense Qwen, one GPU, and greedy decoding. It must
retain eligible graph replay, prefix reuse, chunking, and packing in that
path; exposing control must not silently turn off those optimizations.
Unsupported online combinations fail explicitly. Existing hybrid,
distributed, and other generation entry points remain on their current
paths rather than being partially migrated. In particular, migrating the
ordinary greedy `serve()` path must not remove its existing stochastic
mode; calls outside the first supported slice retain their old path.
Stochastic regrouping is not covered by a greedy parity claim.

The retained loop is named `LLM._serve_legacy()`. It remains executable for
those other paths and supplies the paired performance control; it is not a
second public online API. Greedy sampling uses batched `argmax`, the same
operation selected by `sample(..., temperature=0)`.

`close()` is idempotent: its first call releases any remaining request
references and returns their cancellation terminals; subsequent calls
return no events. It closes the session, not the caller-owned LLM. New
submissions or steps on a closed session are errors. A `try/finally` or
context manager is required around the session so an exception does not
leave live references behind. Unexpected model failures propagate; automatic
recovery from a damaged CUDA context is outside this teaching milestone.
If the caller needs cancellation events for unfinished work, it should
consume the explicit `close()` return value; context-manager exit performs
cleanup but does not deliver that list to an event handler.

After a failed forward, cleanup also clears warm prefix entries. Decode can
publish a full block before its forward writes that block; reusing it after
a failed write would trust unverified contents. This conservative error path
may discard otherwise valid warm entries. Normal close and cancellation do
not clear them, and neither path deallocates the physical pool tensor.

## 8. The boundary to M10b

M10a establishes the session contract; [M10b]({% post_url 2026-09-22-building-tinyserve-m10b %}) adds
HTTP around it. The adapter's main job is translation:

```text
request handler -> command inbox -> single engine owner -> token events
                                              |
                                      per-request outbox
                                              |
                                text decoder -> response stream
```

The handler must not mutate the scheduler or call a model forward directly.
A client disconnect becomes a cancel command. The engine worker drains
commands between steps and routes events by request ID. When idle, it waits
for commands rather than busy-spinning on empty `step()` results.

Both the command inbox and per-request output buffers must be bounded. A
slow client must not block the shared engine owner or grow memory without
limit. M10b must terminate an overflowing stream and cancel its request,
not silently drop tokens or pause every other client. Precise queue limits,
HTTP error mapping, text-stream framing, and the supported schema belong
to the separate M10b chapter—not M10a.

This keeps networking small while still teaching the important resource
boundary. Authentication, persistence, distributed routing, multi-model
hosting, and production deployment are not part of M10a or its completion
criteria.

## 9. Follow the implementation

Start with the `M10a: live requests, token events, and cancellation` debugger
configuration in [launch.json](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/.vscode/launch.json). It runs the existing
[`examples/generate.py`](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/examples/generate.py) entry point:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/generate.py \
  --model /home/scratch.kaix_coreai/models/Qwen3-0.6B \
  --live --chunk-size 4 --cancel-first-after 2 \
  --prompts "Why is the sky blue?" "2+2=" "Capital of Japan?" \
  --no-chat --max-new-tokens 8 --dtype fp32 --backend gather \
  --no-cuda-graphs --kv-pool-tokens 1024
```

The demo submits one prompt before each successive step. Unlike `serve()`,
the engine receives only that request at submission time. It prints raw
token/terminal events as each step returns, then decodes each complete
history at the end. It deliberately does not claim to be a streaming-text
decoder. Real tokenizer lengths differ from the five/three/two-token fixture.

Useful breakpoints, in order:

1. `ServingSession.submit()` / `_submit_tokens()`: copy parameters, create a
   unique ID, and queue the request without allocating KV.
2. `ServingSession.step()` and `Scheduler.schedule()`: inspect the waiting
   deque, resident states, block tables, and cache cursors.
3. `_paged_chunk_prefill()` or `_paged_prefill_batch()`: distinguish a
   partial history append from packing several fresh whole prompts.
4. `ServingSession._emit()`: see the pending uncached token, its index, and
   the EOS/length terminal decision.
5. `Scheduler.cancel()` / `free_sequence()`: inspect reference counts and
   `num_available` before another request reuses capacity.

`session.stats` counts steps, decode forwards, packed-prefill forwards and
requests, chunk forwards, history tokens processed, and cancellations.
The history-token count includes recomputation after preemption.
`serve(profile=True)` retains the existing phase report and graph-replay count.

### A reference-path bug caught during extraction

Repeated FP32 tests exposed an older issue in Torch gather decode, in both
the retained loop and the new session. Gathering full pages also gathers
unwritten tail slots and padded page-table entries. A boolean attention mask
does not make a value-side `0 * NaN` finite, so uninitialized bytes could
produce non-finite logits and an incorrect greedy token.

The gather backend now zeros invalid lanes in the **gathered copies** before
attention. It does not overwrite physical KV or shared prefixes. A test
fills the unused pool with NaNs and checks finite logits plus identical
tokens against a zero-filled control for both loops. FlashInfer decode and
the already length-sliced chunk path are unchanged.

## 10. Validation and measurement boundaries

The acceptance checks cover:

1. **Live membership:** B and C are created after earlier steps; no hidden
   upfront prompt list is needed. An idle session accepts later work.
2. **Output contract:** streamed token IDs reconstruct the completed greedy
   output; per-request indices are ordered, and terminal events occur once.
   Check EOS, output budget, and zero-output completion.
3. **Cancellation:** waiting, prefilling, running, and preempted requests stop
   without changing peers' ownership; repeated/late cancellation is harmless.
4. **KV ownership:** cancel a partially filled page and a shared-prefix owner;
   prove surviving readers remain valid. Check close/error cleanup and
   repeated sessions without invalidating legitimate warm cache entries.
5. **Execution path:** counters establish that packing, chunking, prefix
   hits, preemption, and eligible graph replay still occur where expected.
6. **Numerics:** use FP32 controlled greedy parity first, then record BF16
   agreement and drift separately. Different batch shapes can change BF16
   near-ties, as M9 demonstrated; cancellation must not be advertised as
   universally token-invariant without that qualification.
7. **Measurements:** compare the extracted ordinary path with the retained
   implementation under matched arrivals and cache conditions. Report
   throughput, engine/client TTFT and ITL boundaries, step overhead, and
   cancellation-to-release latency. Rerun the established cross-engine
   calibration after implementation, retaining actual token counts and
   internal-versus-HTTP timing differences.

No M10 speedup is assumed. The primary gain is a missing capability: one
long-lived engine can accept, advance, stream, and stop individual requests
while reusing the serving mechanisms built in earlier milestones. Performance
measurements must show what that new control boundary costs.

### Initial A6000 results

The full single-device run passed **397 tests**, with **9 skipped**. It
includes the new lifecycle/ownership tests, real tiny-transformer FP32
checks, the NaN-poisoned reference test, and existing serving tests. This is
not a new two-GPU qualification. The exact M10a debugger configuration also
completed: A was cancelled while B and C continued to their output budgets.

Checkpoint comparisons retain raw token IDs, not just decoded strings:

| Checkpoint / dtype | Same-workload old loop vs session | Live survivors vs independent requests |
| --- | ---: | ---: |
| Qwen3-0.6B / FP32 | 6/6 match | 2/2 match |
| Qwen3-0.6B / BF16 | 6/6 match | 2/2 match |
| Qwen3-8B / BF16 | 6/6 match | 1/2 match |

The 8B mismatch compares different batch shapes, not identical executions.
It reinforces the BF16 limitation already seen in M9: a live membership
change is not a promise of bitwise-identical greedy text. The fixed-workload
extraction checks and every measured old/new pair did match. These are
bounded fixtures, not a broad model-quality evaluation.

[`examples/bench_session.py`](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/examples/bench_session.py) keeps one 8B BF16
model loaded, disables prefix reuse, warms both paths twice, and alternates
their order for seven measured pairs per condition. The prefill budget is
512, capacity is 32,768 token slots, and eligible FlashInfer CUDA graphs
remain enabled. The ratio is old-loop wall time divided by session wall time;
above 1 favors the session.

| Prompt / output budget / batch | Median ratio | Bootstrap 95% interval |
| --- | ---: | ---: |
| 128 / 128 / 1 | 1.0013 | 1.0006–1.0025 |
| 128 / 128 / 8 | 1.0002 | 0.9995–1.0007 |
| 2048 / 32 / 1 | 0.9996 | 0.9963–1.0016 |
| 2048 / 32 / 8 | 1.0001 | 0.9987–1.0016 |

All **28 measured pairs** have identical token sequences. The practical
result is neutral overhead, not a new inference-speed claim: the largest
median difference is about 0.13%. These intervals describe this paired run,
not variation across machines or a broad workload distribution.

The live fixture also makes buffering visible. In the final 0.6B runs,
cancellation-to-step-return was about 48 ms in FP32 and 50 ms in BF16. That
includes the same step's remaining work and is an **upper bound** on release
latency; it is not the time spent freeing pages. BF16 live execution recorded
16 graph replays. A token computed before a prefill forward can wait tens of
milliseconds for step-end delivery, so raw engine emission time must not be
presented as client-visible TTFT or ITL.

### Standard Qwen3-8B calibration

The separate six-condition standard suite uses an 8,192-token prefill budget,
131,072 KV slots, two warmups, and five repetitions. All requests emit the
requested token budget in these samples. These are absolute measurements,
not paired speedups against an older milestone.

| Prompt / output budget | Batch | Output tok/s | Engine-emission TTFT P50, ms |
| --- | ---: | ---: | ---: |
| 128 / 128 | 1 | 38.07 | 36.19 |
| 128 / 128 | 8 | 284.58 | 183.64 |
| 128 / 128 | 32 | 916.76 | 694.61 |
| 2048 / 32 | 1 | 26.71 | 370.10 |
| 2048 / 32 | 8 | 67.79 | 2819.22 |
| 2048 / 32 | 32 | 79.87 | 7102.95 |

The long-prompt, 32-request row still pays for serially scheduled prefill
work; exposing a live API does not remove that computation. The retained
[historical cross-engine baseline](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/docs/benchmarks/cross-engine-a6000-2026-08-29.md)
uses an older software environment, so its deltas are not attributed to M10a.

### Fresh cross-engine reference: Qwen3-4B, one request

A separate refresh uses the existing BF16 4B artifacts, 8,192-token capacity
and prefill budget, two warmups, and five repetitions. All four engines pass
the `2 + 2 = 4` smoke test. Throughput below uses the harness's observed
output counts rather than assuming every engine reaches the requested budget.

| Engine | Timing boundary | 128 / 128 output tok/s | 2048 / 32 output tok/s | Count per request, short / long |
| --- | --- | ---: | ---: | ---: |
| tinyserve | in-process engine | 62.06 | 44.24 | 128 / 32 |
| llama.cpp | streaming HTTP client | 73.92 | 37.91 | 128 / 32 |
| FreeToken | streaming HTTP client | 72.98 | 50.03 | 127 / 31 |
| Ollama | streaming HTTP client, native cache policy | 74.86 | 44.22 | 128 / 32 |

FreeToken's count difference and Ollama's native prompt-cache behavior are
not hidden inside a supposedly identical-work comparison. Nor should
engine emission TTFT be ranked directly against HTTP arrival TTFT. NInfer
does not run here: its checkout requires SM120a, whereas the A6000 is SM86.
These are serving calibration samples, not an exact-token agreement test
between implementations.

Tinyserve's prior M9c snapshots were 62.41 and 44.27 output tok/s for these
two shapes. The fresh values are within 0.6%; this unpaired refresh does not
establish a speed change. Together with the paired control, the conclusion
is that M10a adds live control while leaving the existing performance gap
essentially unchanged.

The [M10a evidence receipt](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/docs/benchmarks/m10a-online-serving-a6000-2026-09-20.json)
retains commands, source/model fingerprints, test status, token comparisons,
paired samples and intervals, standard-suite samples, external-engine
revisions, timing boundaries, telemetry ranges, and raw-artifact hashes.
The paired FlashInfer run preceded the independent Torch gather sanitization;
its decode and chunk branches were unchanged. Final correctness, the full
suite, and the standard/cross-engine refresh use the final implementation.
