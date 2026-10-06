---
layout: post
math: true
title: 'Tinyserve, Chapter 11: Request lifecycle and streaming'
date: 2026-10-06 00:00:00 -0700
categories:
- tinyserve
- llm-serving
excerpt: How does a request retain its state and randomness from arrival to retirement?
book_chapter: 11
source_revision: 14b0c8dbb975350bbd730ac0c702eafce36eb95c
source_document: docs/book/11-live-requests.md
code_revision: e20a34815498560f9226a9e057ade31b2b62743f
---

{% include tinyserve-article-style.html %}
{% include tinyserve-book-nav.html %}

*Chapter 11 · The Serving Runtime*

A request arrives while another is generating. A client presses Stop. A
third client reads more slowly than tokens are produced. These are changes
in ownership and delivery, not changes to the transformer equation. A live
serving runtime must keep each request's history, randomness, output, and
cleanup aligned while the batch changes around it.

Tinyserve separates a synchronous serving session from the network adapter.
One owner advances the model and scheduler. HTTP tasks exchange commands
and text through bounded queues. This arrangement lets clients progress
independently without giving each socket permission to call the model.
We will follow three requests through four engine steps, then follow their
events from sampling to the client.

## The session outlives a request

A blocking `serve(prompts)` call knows its complete workload at entry and
returns completed results. An open session instead persists while requests
arrive, retire, and leave the engine temporarily idle. Model weights, KV
storage, and backend resources stay loaded across those idle periods.

The session exposes five operations:

| Operation | Meaning |
|---|---|
| `submit()` | Validate, tokenize, assign a request ID, and enqueue work. |
| `step()` | Select and execute one scheduler iteration; return its events. |
| `cancel()` | Mark a live request for retirement at the next safe boundary. |
| `has_work()` | Report whether unfinished request work remains. |
| `close()` | Release remaining ownership and close the session. |

Submission does not run a model forward or allocate prompt pages. Admission
does that later. An idle `step()` returns an empty list; it does not wait
for a new client. A closed session rejects further work, while an idle open
session can accept it. Only one active session may own an LLM's cache-facing
scheduler state.

The calls are serial. A synchronous `step()` can include a batched decode,
one or more prefill calls, sampling, and device synchronization. It is not
one CUDA kernel and is not an interruptible unit. The HTTP adapter makes
intake asynchronous around this boundary, rather than making the session
itself safe for concurrent callers.

## Three requests across four steps

Use a four-token shared prompt budget, enough cache capacity, no prefix
reuse, and no EOS before the stated output limits. A has five prompt tokens
and an output limit of four. B has three prompt tokens and a limit of two.
C has two prompt tokens and a limit of two. Labels A1, B1, and C1 stand
for sampled IDs, not literal text.

| Step | Commands before execution | Decode work | Prompt work | Returned events |
|---:|---|---|---|---|
| 1 | Submit A | None | A positions 0–3 | None |
| 2 | Submit B | None | A position 4; B positions 0–2 | A1, B1 |
| 3 | Cancel A; submit C | Consume B1, sample B2 | C positions 0–1 | A cancelled; B2; B length; C1 |
| 4 | None | Consume C1, sample C2 | None | C2; C length |

[![A, B, and C enter over four steps. A is cancelled before step three. B and C emit terminal events after their last token, and each step returns events only after all its forwards finish.](/assets/tinyserve/book11-live-timeline.svg)](/assets/tinyserve/book11-live-timeline.svg)

At the end of step 1, A has four computed prompt tokens but already owns
enough pages for its full five-token prompt. In step 2, its final prompt
position yields A1. B's whole prompt fits the remaining three-token budget
and yields B1. These are two forwards sharing a budget, not one combined
model call in the ordinary session path.

Both first outputs are uncached when returned. A's cancellation before
step 3 prevents A1 from being consumed in another forward. It does not
retract A1. B consumes B1, reaches its output limit after sampling B2, and
releases its pages. C finishes prefill and waits until step 4 to decode.

With four-token pages, A holds two pages and B one after step 2. Cancellation
releases A's references before step 3 admission, so C can reuse that space.
After step 3, only C remains. After step 4, all request references are gone.
If prefix reuse were enabled, full registered pages could remain warm;
release of request ownership would not imply freeing the underlying pool
tensor or removing every reusable prefix.

## Events carry identity and order

The engine produces two kinds of event:

```text
TokenEvent(request_id, token_index, token_id, emitted_at)
FinishedEvent(request_id, reason, finished_at, stats)
```

`token_index` is zero-based within one request's generated history. The
request ID persists even when the request changes its temporary batch row.
The event is therefore routable without relying on whichever row produced
the logits this iteration.

An EOS sample produces its token event followed by an `eos` terminal. An
output-budget-ending sample produces its token event followed by `length`.
EOS takes precedence if both apply. Cancellation produces a terminal without
a new token. A zero-output request terminates without prefill. Under normal
execution, one accepted request produces one terminal and no later tokens.

The terminal carries counters such as preemptions and reused prompt tokens.
After constructing it, the session removes the request from its active
registry. A monotonically increasing ID supplies uniqueness; the runtime
does not preserve every finished history merely to avoid reusing an ID.

This is an engine lifecycle contract. It does not establish durable delivery.
A network connection or process can disappear between event creation and
receipt. The server can retain local order while being unable to tell whether
a disconnected client consumed its last bytes.

## Randomness belongs to that same request

Suppose A uses seed 17 and B uses seed 29. With one global random generator,
B's extra sampling call would consume a draw that A might otherwise use.
Cancelling B could then change A even if A's logits remained identical.
A process-wide seed reproduces an execution order, not a request isolated
from other clients.

The session copies sampling parameters on submission and creates a device
`torch.Generator` for each stochastic request with a positive output budget.
It stores that generator under the request ID, not the batch row. An explicit
seed initializes it once; an omitted seed comes from operating-system
randomness. Greedy requests do not need a generator or consume random draws.

The sampling chain is concrete. At positive temperature, scores are centered
and filtered in FP32, divided by temperature, restricted by top-k, restricted
by top-p, then sampled from the survivors. With initial probabilities
`[0.55, 0.25, 0.15, 0.05]`, temperature one and top-k three leave normalized
probabilities approximately `[0.579, 0.263, 0.158]`. Top-p 0.8 retains the
first two because together they cross 0.8. Their final probabilities are
`[0.6875, 0.3125]`.

The threshold-crossing token must survive. Tinyserve shifts the drop mask
one position, so filtering never removes the very token needed to reach the
requested mass. Its strict greater-than comparison also defines behavior
at exact threshold equality. Top-k uses a value threshold, so tied logits
at the kth value can leave more than k survivors. These details belong to
the sampling contract, not just to an implementation's aesthetics.

An intermediate prefill chunk consumes no randomness. Completing prefill
and each later output sample advance only the owning request's stream.
Preemption discards computed KV but retains generated IDs and the generator.
Readmission recomputes existing history without sampling those old outputs
again. Cancellation, EOS, output exhaustion, and session closure retire the
generator with the request.

Identical logits, parameters, seed, device, and sampling implementation can
therefore isolate A from unrelated draws. Changing the backend, dtype,
batching, or chunking can change A's logits before sampling. A seed cannot
repair that upstream numerical difference. The runtime uses its selected
scalar per-request sampler for stochastic rows; the experimental batched
sampler is not implied by having several requests in one forward.

## Cancellation waits for safe ownership transfer

Cancelling a waiting request removes it before admission. Cancelling a
prefilling request also releases its reserved but not-yet-written pages.
Cancelling a decoder removes it from future decode membership and releases
its current references. Repeating cancellation, or cancelling an unknown
or terminal ID, does not release anything twice.

The safe boundary comes after pending device work. A partial prefill chunk
may produce no sampled ID, so there may be no token copy that implicitly
waits for CUDA completion. The session explicitly synchronizes before
returning control. Otherwise the next caller could cancel a request and
reuse its pages while an earlier kernel still accessed them.

An HTTP cancellation arriving during a forward sets a flag. The owner
handles it after that step returns, before selecting another step. If
the request already finished in the current step, there is no live owner
left to cancel. This bounds cancellation responsiveness by the remaining
step duration, which depends on workload and execution cost; a prompt-token
budget is not a millisecond deadline.

`close()` is idempotent. It synchronizes, retires remaining requests, and
detaches the session from the LLM. Failed-forward cleanup also removes
potentially untrustworthy warm prefix entries. A worker that fails becomes
unavailable; cleanup tests do not imply recovery from a damaged CUDA context.

## From sampled ID to visible text

The network adapter has one worker thread that constructs, uses, and closes
the LLM and session. HTTP tasks own validation, socket lifetime, and Server-
Sent Event framing. They submit commands to the worker and consume their
own output channels. They never enter the model directly.

[![A single model owner turns session token events into per-request text queues. Independent HTTP writers consume the queues; cancellation returns as a flag. Engine emission, step return, and client receipt are separate timing boundaries.](/assets/tinyserve/book11-stream-boundaries.svg)](/assets/tinyserve/book11-stream-boundaries.svg)

There are at least three relevant times. A token is sampled and timestamped
inside `step()`. The entire step then finishes and returns its event list.
Finally, the worker decodes and queues text, and the HTTP task transmits it.
A decode token can wait behind subsequent prefill work inside the same step.
Its internal emission timestamp is not client-visible time to first text.

Token IDs also do not map one-to-one to text chunks. With a ByteLevel tokenizer,
a token can contain the first two bytes of the three-byte UTF-8 euro sign,
`E2 82`, while the next supplies `AC`. Decoding each independently would
produce replacement characters. Tinyserve keeps one incremental UTF-8
decoder per request, emits complete characters, retains incomplete bytes,
skips special IDs, and flushes at normal termination.

The reference text is whole-sequence decoding with special tokens skipped
and whitespace cleanup disabled. Unsupported tokenizer decoder types are
rejected at startup. Usage counts sampled token IDs, including EOS, rather
than counting text fragments or re-tokenizing the result.

## A slow reader must not stop the engine

The default adapter permits 32 outstanding requests and 64 queued items per
output channel, plus a separate terminal slot. Outstanding includes pending
commands, active generation, and completed-but-undelivered responses. A
request releases its admission slot only after both the engine and HTTP
owner have relinquished it. Otherwise rapid disconnects during one long
forward could bypass the bound.

The worker never blocks waiting for a client to empty its queue. If the
queue fills, it clears pending items, marks that stream failed, requests
cancellation, and preserves an error terminal. It does not silently drop
text and later send a normal `length` completion. The separate terminal
slot ensures that data items cannot crowd out termination metadata.

HTTP cancellation has a per-channel flag independent of the submission
inbox, so a full inbox cannot prevent it. Network write deadlines bound a
stalled writer, while the event loop remains free to serve another socket.
An error discovered before headers can use an HTTP status such as 400,
429, or 503. After streaming starts, the adapter emits an error in the
stream and terminates without a success marker. A disconnected peer may
receive neither.

## What the measurements establish

Request-local sampling adds functionality and host/device work. A retained
Qwen3-0.6B A6000 experiment compared the selected session against its greedy
control with two warmups and five paired repetitions in each of four
prompt/output and batch conditions. Median control/candidate time ratios
ranged from 0.996 to 1.005; all passed the predeclared five-percent overhead
gate with matching greedy IDs. This supports the lifecycle addition without
establishing a serving speedup.

In separate fixed-logit measurements, batch 1/8/32 greedy sampling took
0.028/0.038/0.068 ms, while temperature 0.8, top-k 20, top-p 0.9 took
0.538/3.927/16.469 ms. These include host dispatch and token readback and
describe the sampler alone. Per-request filtering over the vocabulary
accumulates work as the batch grows; those numbers cannot be substituted
for whole-model or HTTP throughput. The
[retained sampling receipt](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m10c-request-sampling-a6000-2026-09-25.json)
records the experiment's scope.

The supported live session is ordinary dense Qwen on one device. Distributed,
hybrid, speculative, and disaggregated execution have separate contracts;
their existence elsewhere in the repository does not make every online
sampling combination supported. Tests isolate fixed-logit RNG behavior,
session lifetime, socket delivery, and controlled model replay at their
respective boundaries.

## Follow the implementation

| Source at runtime snapshot `e20a348` | Responsibility |
|---|---|
| [session.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/session.py) | Single owner, events, request generators, sampling, and terminal cleanup. |
| [online.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/online.py) | Worker inbox, bounded output channels, cancellation flags, and byte decoding. |
| [server.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/server.py) | Completion request validation, SSE framing, socket writes, and disconnects. |
| [sampling.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/sampling.py) | Greedy and per-request stochastic selection. |

A request owns its logical history and random stream across scheduling
changes. The session owns safe execution boundaries. The transport owns
delivery to a potentially slow or disconnected reader. Keeping those
lifetimes explicit lets one request stop without corrupting its peers or
turning their shared model into a collection of competing callers.

{% include tinyserve-book-nav.html %}
