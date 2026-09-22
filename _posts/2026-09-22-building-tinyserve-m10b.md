---
layout: post
math: true
title: "Building tinyserve M10B: From token events to an HTTP text stream"
date: 2026-09-22 00:01:00 -0700
categories: [tinyserve, llm-serving]
excerpt: "One model owner, bounded output channels, correct text streaming, and measured HTTP concurrency across four engines."
---

<style>
.post-content table { display: block; max-width: 100%; overflow-x: auto; }
.post-content :not(pre) > code { overflow-wrap: anywhere; }
</style>

*Part of [building an LLM inference engine from scratch](/series/tinyserve/).*

Code and evidence: [`tinyserve` @ `e533be6`](https://github.com/kaix-nv/tinyserve/tree/e533be67d15772292b1acbce44ce67dcadac4222). Click any figure to open it at full size.

Previous: [M10a — Live serving sessions]({% post_url 2026-09-22-building-tinyserve-m10a %}).

**Status: implemented and validated within the scope below.** M10b adds local
HTTP completion streaming for single-device, greedy, ordinary dense Qwen.
It leaves the model, attention kernels, and scheduling policy unchanged.

## 1. Why a session is not yet a server

M10a can accept a prompt between calls to `step()`. But a browser cannot call
that Python method. A network adapter must translate incoming requests into
session commands and translate outgoing token IDs into text. It also has to
notice when the reader goes away.

There are two independent clocks: the engine finishes a synchronous step,
and the network accepts bytes for a particular client. Making them the same
clock would let one slow reader stall every sequence in the batch.

We therefore keep one engine owner and give each client its own bounded
outbox. HTTP concurrency does not mean concurrent calls into the model.

Writing `async def` around `session.step()` would not make that synchronous
GPU step nonblocking for the HTTP event loop. The engine therefore runs on
its own thread. HTTP tasks can wait for sockets or outbox notifications while
the worker advances the batch. This changes ownership and delivery, not the
autoregressive dependency between successive model tokens.

[![HTTP clients share one engine owner, with independent bounded outboxes](/assets/tinyserve/m10b-http-lifecycle.svg)](/assets/tinyserve/m10b-http-lifecycle.svg)

## 2. A concrete request lifecycle

Alice posts a prompt with an eight-token budget. The HTTP task checks the
small JSON schema and places a command in the inbox. The engine worker
tokenizes the prompt and calls the existing session submission seam. Only
this worker touches the model, scheduler, and KV cache.

While Alice is decoding, Bob posts a second prompt. At the next step boundary
the worker admits Bob, then runs the existing decode/chunked-prefill policy.
The returned events carry request IDs, so Alice's token goes to Alice's
outbox and Bob's token goes to Bob's outbox. The HTTP tasks can write those
outputs independently.

If Alice disconnects during a GPU forward, her handler marks cancellation.
The worker observes that flag at the next boundary and calls `cancel()`;
the next `step()` releases Alice's page references. Bob continues. There is
no mid-kernel interruption. An idle worker sleeps on a wake-up event until
a submission, cancellation, or shutdown arrives.

M10a's delivery boundary still applies: decode events become visible only
after the whole step, including any following prefill work, has returned.
M10b does not introduce compute/network overlap inside a step.

Intake itself costs time: the worker tokenizes newly queued prompts before
its next step. Draining at most 32 commands bounds the count, not a strict
latency in milliseconds. Chunked prefill bounds GPU prompt work; it does not
make CPU tokenization or network delivery free.

## 3. Token IDs are not text fragments

A byte-level BPE token can end halfway through a UTF-8 character. For example,
the euro sign is the three bytes `E2 82 AC`. Suppose one token provides
`E2 82` and the next provides `AC`. Decoding each token separately produces
replacement characters, not `€`.

The supported Qwen tokenizer uses a `ByteLevel` decoder. For each request we:

1. Map the token's vocabulary characters back to their original bytes.
2. Skip special token IDs, just as whole-output decoding does.
3. Feed bytes to a persistent incremental UTF-8 decoder.
4. Emit only complete characters; retain unfinished bytes for the next token.
5. Flush at normal completion, using replacement characters if the generation
   genuinely ends with incomplete UTF-8.

The reference is `tokenizer.decode(ids, skip_special_tokens=True,
clean_up_tokenization_spaces=False)`. Whitespace is preserved. One token may
produce no text, or several characters; one text chunk is not one token.
Unsupported decoder types must fail at startup, not silently corrupt text.

## 4. A deliberately small wire contract

`GET /health` reports worker readiness. `POST /v1/completions` accepts one raw
string prompt, the configured model name, `stream: true`, `temperature: 0`,
and a nonnegative integer `max_tokens`. There is no chat template. A seed is
accepted for the existing calibration client but has no effect on greedy
decoding. `stream_options.include_usage` requests final token counts.

Responses use Server-Sent Event framing: `data: <JSON>`, a blank line, and
eventually `data: [DONE]`. Text is in `choices[0].text`; terminal reasons are
`stop` for EOS and `length` for the output budget. Usage counts sampled token
IDs, including EOS, rather than re-tokenizing text. This is a **completion
stream subset**, not a claim of full OpenAI API compatibility. Unsupported
fields and modes are rejected explicitly.

Before response headers, validation failures use HTTP 400; full admission
uses 429; an unavailable worker uses 503. After headers, failures use an SSE
error and terminate the stream without a success marker. Disconnected clients
cannot receive a terminal event.

## 5. Bounded resources and slow readers

The default capacity is 32 outstanding requests, covering pending commands,
active generations, and completed-but-undelivered responses. Each outbox
holds at most 64 events, plus a separate terminal slot. Request bodies are
limited to 64 KiB. The model context bounds token histories.

A disconnected request keeps its admission slot until the worker has also
released its engine ownership. Otherwise, a stream of connect/disconnect
events during a long forward could bypass the limit. For a positive output
budget, an individual prompt plus budget must fit the pool minus the scheduler's reserved page;
this conservative check rejects it with 400 before it can fail the shared
scheduler. It does not reserve that entire budget upfront for every request.

The worker never waits for an outbox to empty. If it fills, that stream is
explicitly failed and its generation cancelled. Pending text can be discarded
only with that error, never followed by a normal completion. Cancellation has
its own per-request flag, so a full submission inbox cannot block it. A send
timeout also cancels a client whose network write stops making progress.

The terminal slot is separate from the text queue. Normal completion drains
queued text before its terminal event. Overflow replaces pending text with an
explicit error; it cannot accidentally report `length` after losing bytes.
If the peer is no longer reading, delivery of even the error is not guaranteed.
The default body-read and individual network-write deadlines are 30 seconds.
These do not bound a model forward or allow interrupting a CUDA kernel.

These are educational ownership rules, not production deployment guarantees.
Bind to loopback. Authentication, TLS, persistence, multi-model routing,
distributed and hybrid models, stochastic sampling, and speculative decoding
are outside this slice.

## 6. Validation and measurements

The initial M10b single-device suite passed **428 tests**, with **9 skipped**,
including 31 new adapter tests. After the concurrent closeout and error-body
fix below, the full suite passed **443 tests**, with **9 skipped**. Only GPU 0
was exposed; this is not a new multi-GPU qualification.
The exact M10b debugger configuration also served a real completion and shut
down cleanly. Diagram rendering, local links, Ruff, compilation, and JSON
checks passed. Injected forward failures test ownership cleanup, not recovery
from a damaged CUDA context; a failed worker becomes unavailable.

- Streamed text equals whole-output decoding, including split Unicode,
  whitespace, special tokens, invalid bytes, and a partial final character.
- Concurrent clients retain their own ordered outputs; all session calls
  execute on one worker thread.
- Disconnect, overflow, shutdown, and forward failures release ownership.
- Full queues reject predictably without preventing cancellation.
- Real-model output agrees with the in-process path under matched execution
  conditions; floating-point batch-shape differences remain a separate issue.
- Same-engine HTTP overhead and cross-engine calibration retain actual output
  counts, client timing boundaries, model revisions, and cache policies.

The goal is a working network boundary, not a promised throughput improvement.

### Real-model correctness and the BF16 boundary

On Qwen3-0.6B, the three sequential HTTP examples matched independent
`serve()` outputs in both FP32 and BF16. The three simultaneous clients
matched all three independent controls in FP32, but only two in BF16.
Concurrent arrival changes execution grouping; as in M9d/M10a, this is not a
promise that BF16 greedy output remains identical across different batch
shapes. The adapter does not change the model's numerical contract.

Both dtypes passed 102 tokenizer cases, including arbitrary vocabulary-ID
sequences, and a real TCP disconnect released all request page references.
The BF16 check exercised 63 CUDA graph replays. These are bounded correctness
fixtures, not a broad model-quality evaluation.

### What the HTTP boundary costs

The paired Qwen3-4B BF16 experiment keeps one model and one engine worker
resident. The direct control submits to that worker's mailboxes and consumes
its decoded text without HTTP. The candidate uses real loopback HTTP/SSE and
the existing `requests` client. Both disable prefix reuse, use 8,192 KV token
slots and an 8,192-token prefill budget, and retain eligible CUDA graphs.

Each condition warms both paths twice, then measures seven pairs with
alternating order. All 14 measured pairs have the same output text and generated
token count. The ratio below is direct latency divided by HTTP latency;
below one means HTTP costs time. The interval bootstraps the paired ratios.

| Prompt / output budget / batch | Median direct / HTTP ratio | Bootstrap 95% interval | Median paired HTTP latency overhead |
| --- | ---: | ---: | ---: |
| 128 / 128 / 1 | 0.9832 | 0.9639–0.9864 | 1.71% |
| 2048 / 32 / 1 | 0.9843 | 0.9794–0.9925 | 1.59% |

This isolates transport/framing/client costs on top of the **new worker**, not
all M10b overhead relative to the original `serve()` loop. It is not a
cross-engine comparison, and it does not establish behavior under high
concurrency, long queues, or remote networks. No kernel speedup is claimed.

For a network client, the benchmark's TTFT means time to the **first nonempty
text fragment**, not the model's first sampled token. It includes request
intake, tokenization, queueing, step-end buffering, UTF-8 assembly, and HTTP
delivery. Some tokens produce no visible text. Subsequent chunk gaps are not
necessarily individual token latencies either.

### Fresh cross-engine HTTP calibration

The same A6000 and Qwen3-4B BF16 artifacts were used for a fresh single-request
comparison, with two warmups and five measured repetitions per condition.
Unlike M10a's in-process Tinyserve row, **all four rows now use HTTP client
timing**. Each engine passed the small `2 + 2 = 4` smoke check.

| Engine | 128-token prompt / 128-token budget, output tok/s | 2,048-token prompt / 32-token budget, output tok/s | Actual sampled output counts |
| --- | ---: | ---: | --- |
| Tinyserve HTTP | 60.90 | 42.07 | 128 / 32 |
| llama.cpp | 74.34 | 37.43 | 128 / 32 |
| FreeToken | 73.01 | 50.14 | 127 / 31 |
| Ollama | 74.92 | 43.80 | 128 / 32 |

Throughput is observed output count divided by whole client makespan, not
steady-state decode speed. FreeToken's reported counts differ from the
requested budgets, so its rows are not identical output workloads. Ollama
retains its native prompt-cache behavior; the Tinyserve measurements disable
prefix reuse, and llama.cpp uses `--no-cache-prompt`. These policy differences
remain even after matching the transport boundary. NInfer was excluded:
the current checkout requires SM120a while this A6000 is SM86.

Tinyserve is still slower on the short-prompt workload. M10b's result is the
working network capability and its measured cost, not a performance win.
Do not divide these HTTP rates by the older M10a in-process rates and call
that a controlled regression: their boundaries differ. The paired worker
experiment above is the controlled estimate of HTTP's additional latency.

The [portable validation and calibration receipt](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/docs/benchmarks/m10b-http-streaming-a6000-2026-09-21.json)
retains source hashes, commands, checkpoint fingerprints, reference revisions,
raw paired samples, HTTP outputs, actual token counts, and GPU telemetry.
The tested optional transport was Uvicorn 0.53.0. Results are dated September
21, 2026; they describe this A6000 and these bounded fixtures.

### M10 closeout: concurrent HTTP measurements

The September 22 closeout asks a different question: **does the HTTP path
actually combine overlapping requests, and what does that buy the client?**
It uses Qwen3-4B BF16 on the same A6000, with all four engines measured from
the loopback HTTP client.

Eight clients are not necessarily an eight-row model batch. Client arrival
times, prompt lengths, prefill work, and completion times change which
sequences reach each decode forward. The closeout therefore retains both
client concurrency and the actual Tinyserve forward-size histogram, using
benchmark-only wrappers rather than changing the model or scheduler.

The bounded protocol uses Qwen3-4B BF16, the existing 128/128 and 2,048/32
prompt/output workloads, and client cohorts of 1, 4, and 8. Each cohort starts
together; the next begins only after it drains. Two warmup cohorts precede
five measured cohorts. This is a finite-cohort scaling experiment, **not**
a sustained arrival-rate or service-level guarantee.

Throughput is the reported generated-token count divided by the whole cohort
makespan. First-text and request latency are measured per client; visible
stream-chunk gaps remain distinct from token intervals. The receipt retains
individual requests and sampled GPU memory. Percentiles from these small
cohorts are descriptive, not reliable production tail estimates. Both
server-reported and re-tokenized output counts are retained. A stream ending
without its terminal marker fails the measurement.

Server settings stay fixed as client concurrency changes: up to eight
execution slots where configurable, approximately
32,768 aggregate KV token slots, and at least 4,096 context tokens per slot.
Tinyserve's 32-request admission limit is not a fixed execution-batch size.
Tinyserve uses an 8,192-token prefill budget, disables prefix reuse, and keeps
eligible decode CUDA graphs. FreeToken uses eight running requests, a naive
cache, and an 8,192-token prefill limit. llama.cpp uses eight slots and
`--no-cache-prompt`, retaining its native prefill and RAM-cache behavior.
Ollama's startup log confirms eight slots, 4,096 context tokens per slot, and
1,024-token batch/microbatch settings; its native cache policy remains a
difference. These are recorded configurations, not identical schedulers.
The earlier one-slot, 8,192-token-capacity measurements are a different
configuration, not the control for a claimed speedup here.

[![HTTP throughput and first-text latency at one, four, and eight concurrent clients](/assets/tinyserve/m10b-http-concurrency.svg)](/assets/tinyserve/m10b-http-concurrency.svg)

Lines show the median of five cohort metrics; whiskers show the observed
minimum and maximum, **not** confidence intervals. The first-text metric is
each cohort's client p50, summarized across the five cohorts. The harness
uses observed order statistics: for an even-sized cohort, p50 selects the
upper middle observation rather than averaging the two middle values.

| Engine | Prompt / output budget | 1 client, tok/s | 4 clients, tok/s | 8 clients, tok/s |
| --- | --- | ---: | ---: | ---: |
| Tinyserve HTTP | 128 / 128 | 59.89 | 225.44 | 412.23 |
| llama.cpp | 128 / 128 | 73.76 | 244.94 | 394.57 |
| FreeToken | 128 / 128 | 72.91 | 284.81 | 540.61 |
| Ollama | 128 / 128 | 74.35 | 247.57 | 409.10 |
| Tinyserve HTTP | 2048 / 32 | 41.31 | 86.18 | 103.75 |
| llama.cpp | 2048 / 32 | 38.05 | 49.20 | 63.87 |
| FreeToken | 2048 / 32 | 49.63 | 108.68 | 132.67 |
| Ollama | 2048 / 32 | 44.29 | 61.03 | 85.87 |

All four pass the small arithmetic smoke check. FreeToken reports 127 and
31 generated tokens per request; the other engines report 128 and 32. The
table uses those reported counts, not requested budgets. Along with native
cache and prefill differences, this prevents treating the table as an exact
same-work kernel comparison. NInfer remains excluded by the SM120a/SM86
hardware mismatch.

Three observations explain the result:

1. **The network path preserves real batching.** In Tinyserve's eight-client
   short workload, 631 of 639 measured decode forwards have eight real rows
   (98.7%). In the long workload, 149 of 161 do (92.5%). The remaining forwards
   reflect arrivals and completions, not a failure to batch. These count
   request rows, not padded CUDA-graph capacity.
2. **Aggregate throughput is not individual responsiveness.** Tinyserve's
   short-workload throughput grows about 6.9× from one to eight clients,
   while median first-text latency rises from 46.7 to 169.0 ms. For the long
   workload, throughput grows only 2.5× and first text rises from 242.6 to
   1,792.7 ms. More requests share decode forwards, but their prompt work
   still has to run before they can join those forwards.
3. **This does not close every performance gap.** FreeToken remains faster
   in every measured condition. Tinyserve can exceed llama.cpp or Ollama in
   some concurrent conditions under these policies, but that is not evidence
   that the HTTP adapter accelerated the model. Arrival grouping and prefill
   policy affect the result; no kernel or scheduler optimization was made.

The 8,192-token budget also matters to interpretation. Eight 2,048-token
prompts contain 16,384 prompt tokens, so they cannot all fit in one prefill
step. The traces show whole prompts packed into forwards of at most four
rows, rather than individual prompts being split into smaller chunks. This
run checks concurrent serving with the existing policy; it does **not**
measure the benefit of a smaller chunk budget.

For Tinyserve, completed-request latency and visible chunk gaps show the
cost experienced after admission:

| Prompt / output budget | Clients | Median request latency, ms | Visible chunk gap p50 / p95, ms |
| --- | ---: | ---: | ---: |
| 128 / 128 | 1 | 2,136 | 16.35 / 16.95 |
| 128 / 128 | 4 | 2,268 | 16.98 / 17.66 |
| 128 / 128 | 8 | 2,459 | 18.01 / 18.90 |
| 2048 / 32 | 1 | 760 | 16.77 / 17.45 |
| 2048 / 32 | 4 | 1,469 | 18.23 / 19.10 |
| 2048 / 32 | 8 | 2,439 | 20.88 / 22.17 |

Request latency uses the median of cohort p50s; chunk-gap percentiles pool
the observed gaps from all measured requests in that condition. The largest
sampled GPU-0 residency across the six conditions is 16,395 MiB for Tinyserve,
12,709 MiB for llama.cpp, 14,133 MiB for FreeToken, and 12,801 MiB for Ollama.
These include resident weights, cache, graphs, and workspaces; they are NVML
samples, not exact allocation peaks or per-request memory costs.

### Staggered arrivals, cancellation, and overload

Three additional clients were dispatched at approximately 0, 81, and 160 ms.
The first disconnected at 241 ms; the two survivors each received 32 tokens
and completed normally. The worker recorded exactly one cancellation and
released every request-owned KV reference after draining. Decode forwards
contained one or two real rows as membership changed.

A separate burst temporarily lowered the idle worker's admission limit to
two. Three requests produced two complete HTTP 200 streams and one HTTP 429
rejection; all page references were released afterward. This is an admission
boundary check, **not** a throughput measurement. It does not establish
behavior under sustained overload or slow internet connections.

The initial burst exposed a response-format bug: HTTP 429 was correct, but
its JSON body inherited an internal default of `status: 503`. The fix passes
the actual status to the error-body helper. Regression tests cover 400, 404,
405, 408, 413, 429, and 503; all six previously inconsistent paths fail against
the old implementation and pass against the corrected one. Tinyserve's six
measurements and lifecycle checks were rerun after that fix; the chart and
tables use only the corrected-server results.

[`bench_online_load.py`](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/examples/bench_online_load.py) owns the six Tinyserve
conditions and these lifecycle checks. From the repository root, with GPU 0
reserved and idle, run:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONPATH="$PWD" \
  .venv/bin/python examples/bench_online_load.py \
  --model /path/to/Qwen3-4B --output .tinyserve-bench/m10-closeout/tinyserve
```

Use an empty output directory. Each condition retains raw requests and a
measured-cohort forward histogram; `receipt.json` records source hashes and
whether all lifecycle checks completed. The benchmark wrappers add small
host-side instrumentation work, so these are instrumented HTTP measurements,
not an overhead-free kernel benchmark. CPU real-socket tests cover the harness
and lifecycle assertions; the real-model GPU run checks those boundaries with
actual model execution. Model kernels and scheduler policy are unchanged.

The [closeout receipt](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/docs/benchmarks/m10b-http-concurrency-a6000-2026-09-22.json)
retains all 24 conditions, raw client timings and outputs, forward histograms,
memory samples, lifecycle results, exact commands, source hashes, and test
results. The earlier run that exposed the error-body bug is identified as
superseded, not mixed into the final performance summary.

## 7. Follow the implementation

The implementation has three small ownership boundaries:

| Component | Owns | Does not own |
| --- | --- | --- |
| [`CompletionApp`](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/tinyserve/server.py) | JSON validation, SSE framing, socket lifetime | Scheduler or model calls |
| [`EngineWorker`](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/tinyserve/online.py) | LLM construction/cleanup, submission inbox, one `ServingSession` | Network writes |
| `OutputChannel` and `ByteLevelTextDecoder` | Bounded per-client output and incomplete UTF-8 bytes | KV scheduling policy |

`EngineWorker._run()` captures arrival time, tokenizes once, and uses the
same `_submit_tokens()` seam as the offline driver. Each iteration drains
the bounded inbox, applies cancellation flags, calls `step()`, and routes
events. Decoding text runs on that same worker, before output enters the
channel. The HTTP event loop only consumes already decoded fragments.

In `CompletionApp`, the stream-writing task and disconnect-watching task run
concurrently. Either ending cancels its sibling; the handler's `finally`
marks the request for engine cancellation. A stalled network write suspends
only that HTTP task. Uvicorn handles HTTP parsing and sockets; the tiny ASGI
application contains the adapter policy. Run **one** Uvicorn worker: multiple
workers would each construct their own model and GPU pool.

Install the optional transport into the project environment with
`uv pip install --python .venv/bin/python -e ".[server]"` (or use
`python -m pip install -e ".[server]"` inside a pip-enabled environment), then select
**M10b: HTTP streaming and disconnect cancellation** in
[`launch.json`](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/.vscode/launch.json). It starts the existing
[`generate.py`](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/examples/generate.py) entry point with Qwen3-0.6B, FP32,
gather attention, a four-token prefill budget, and loopback port 8000.

From another terminal, send Alice's request:

```bash
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-0.6B","prompt":"The capital of France is","max_tokens":32,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
```

Send Bob's request from a second terminal while Alice is running. Pressing
Ctrl-C in Alice's **client** terminal disconnects that client, not the server.
Useful breakpoints are `EngineWorker._run()`, `ServingSession.step()`,
`ByteLevelTextDecoder.push()`, and `CompletionApp._stream()`. Inspect the
request ID, `channel.cancelled`, `active`, and `session.cache.manager.ref_count`.
Do not call session methods manually from the debugger while the worker runs.

[`bench_online.py`](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/examples/bench_online.py) checks real sockets and text
parity, then offers alternating direct-mailbox/HTTP measurements with one
resident model and prefix reuse disabled. The existing
[`bench_cross_engine.py`](https://github.com/kaix-nv/tinyserve/blob/e533be67d15772292b1acbce44ce67dcadac4222/examples/bench_cross_engine.py) can address this
endpoint with `--adapter openai`: here that label selects its completion-SSE
client, not a declaration of full API compatibility.

## References

- [ASGI HTTP messages](https://asgi.readthedocs.io/en/latest/specs/www.html):
  request bodies, response writes, and disconnect notification.
- [Hugging Face tokenizer decoders](https://huggingface.co/docs/tokenizers/api/decoders):
  the byte-level decoding boundary.
