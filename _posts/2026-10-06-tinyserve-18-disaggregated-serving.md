---
layout: post
math: true
title: 'Tinyserve, Chapter 18: Disaggregated prefill and decode'
date: 2026-10-06 00:00:00 -0700
categories:
- tinyserve
- llm-serving
excerpt: When is transferring KV between model owners worth the added cost?
book_chapter: 18
source_revision: 14b0c8dbb975350bbd730ac0c702eafce36eb95c
source_document: docs/book/18-disaggregated-serving.md
code_revision: e20a34815498560f9226a9e057ade31b2b62743f
---

{% include tinyserve-article-style.html %}
{% include tinyserve-book-nav.html %}

*Chapter 18 · Distributed Serving*

Prefill and decode share a model but stress it differently. A prompt supplies
many known input tokens at once; a decoder supplies one new token per active
request and repeatedly reads its history. If both phases share one execution
timeline, prompt work can delay established streams. Chunking bounds how
much prompt work runs between decode opportunities. Disaggregation changes
the owner: one model replica processes prompts and another continues them.

That separation requires transferring the computed history. The decoder
must receive valid KV, allocate its own destinations, inherit the pending
first output token, and agree when the source can release memory. Tinyserve
implements this handoff between two local GPUs. Its retained measurement
is slower than two ordinary replicas on the tested workloads. Understanding
both the protocol and that result is essential to understanding the design.

## Split a request lifetime rather than a model

Tensor parallelism splits layer operations and pipeline parallelism splits
layers. Here both GPUs hold every model layer. GPU P runs prompt prefill;
GPU D runs decode for requests whose prompts are ready. P and D can work
on different requests at the same time.

The unit transferred is a request's state at a phase boundary. It includes
its identity, prompt length, output allowance, cancellation state, the first
sampled token, and prompt KV for every attention layer. The first sampled
token is still uncached, exactly as in the single-device loop. D consumes
that token to produce the second output.

The public factory uses the same checkpoint, configuration, and dtype for
both owners. Matching shape alone would be insufficient: KV computed with
different weights or positional conventions is not a valid continuation
state. This implementation restricts the path to ordinary dense Qwen,
unquantized weights and KV, private pages, and greedy sampling.

Each GPU has an independent allocator. A source page ID is meaningful only
within P's pool. Sending P's block table and interpreting it as D's would
retrieve unrelated memory. The handoff preserves tensor values and logical
positions while explicitly remapping physical storage.

## Follow a six-token request

Use a page size of four. The prompt contains six tokens and requests four
outputs, Y1 through Y4. Assume no EOS. P's prompt table is `[5, 2]`.
D reserves pages `[9, 1, 7]`, enough for the maximum cached history:

$$
\mathrm{cached\ limit}=P+N-1=6+4-1=9.
$$

Only nine positions require KV because Y4 is the terminal sampled token
and need never be consumed. The destination reservation is three pages,
`ceil(9 / 4)`. Admission also leaves its configured page watermark.

Before any prefill is enqueued, the decode owner acquires that complete
reservation. D initially exposes `[9, 1]` as the prompt table and keeps
`[7]` in a separate future-page list. Owned future capacity is not valid
history and must not be presented as an additional filled page to attention.

[![A six-token prompt moves from source pages five and two to destination pages nine and one. Page seven is reserved for future decode, while the first output token remains uncached. The acknowledgement precedes source release.](/assets/tinyserve/book18-kv-handoff.svg)](/assets/tinyserve/book18-kv-handoff.svg)

P processes the six prompt tokens and samples Y1. It packs exactly logical
positions 0–5 into a contiguous payload. Source position 5 was in page 2,
offset 1; D scatters the same values into page 1, offset 1. Unused offsets
2 and 3 in the final source page never become payload bytes.

After import, D has `num_cached = 6`, visible table `[9, 1]`, and future
reservation `[7]`. Y1 enters the generated token history and becomes
available as an event. The first decode consumes Y1 at logical position 6,
page 1 offset 2, and samples Y2. The next consumes Y2 at position 7,
page 1 offset 3, and samples Y3.

Before consuming Y3 at position 8, D moves page 7 from the future list into
the visible table. That forward caches position 8 and samples Y4. The
request finishes with nine cached positions and a final uncached output.
Cleanup releases both visible pages and any unused future reservation.

This trace exposes three separate quantities: the number of reserved token
slots, the valid cached cursor, and the generated-token count. Advancing
one does not automatically advance the others.

## Reserve first so decode never has to prefill

The ordinary scheduler grows capacity incrementally and can preempt a
request, discarding KV and reconstructing history later. That policy would
undermine a dedicated decode owner: an evicted request could force D to
run prefill, or require a second handoff protocol.

Tinyserve instead reserves the complete prompt-plus-output budget before
P starts. If the FCFS head cannot fit on D, it waits. This is conservative:
early EOS may leave capacity reserved but unused, and smaller requests
behind the head do not bypass it. The benefit is a simple invariant that
every accepted prefill has a destination able to finish its remaining
generation without decode-side eviction.

The source also needs room for its prompt. The public submission limit
uses the available per-request capacities of both owners and the model
context limit. Reservation establishes space, not an entitlement to ignore
later cancellation or an error. Page ownership remains tied to the job
through every state transition.

P works on one prompt at a time, using the existing whole-prompt or bounded
chunk helpers. D batches ready decoders. Chunking still matters for source
cancellation responsiveness, but its budget no longer alternates directly
with D's decode work on the same GPU.

## Pack import acknowledge and release

The transfer tensor has shape

```text
[layers, 2, prompt tokens, KV heads, head width]
```

The second axis selects K or V. `pack_kv()` gathers valid positions through
the source table and makes a contiguous payload. `import_kv()` checks its
shape and dtype, copies values to D, scatters through D's table, and waits
for completion before the source is acknowledged.

The sequence is ordered:

1. P completes prefill and publishes a ready packet containing payload and Y1.
2. D imports the payload into its owned prompt pages.
3. D acknowledges the packet after the destination work is complete.
4. P releases source pages and payload ownership, then reports source release.

Only one payload is in flight from P. P waits for the acknowledgement
before starting the next prompt, bounding staging memory and making the
ownership boundary explicit. D may start continuation once import is
complete, but terminal cleanup does not claim that the request is fully
retired until the source has also reported release.

The actual transport stages through host memory: source GPU to CPU, then
CPU to destination GPU. It does not use a verified direct peer-copy fast
path, CUDA IPC, RDMA, or a remote cache service. During initial qualification,
a direct two-GPU value-copy check failed on the host despite reported peer
access, while host staging preserved the values. That observation justifies
the selected path on that host without diagnosing its driver-level cause.

Import is synchronous on the decode owner. While D copies and scatters a
new request, its existing requests do not advance. Disaggregation removes
prompt compute from D but inserts import pauses there. Independent workers
do not remove that dependency.

Y1 is emitted after import, so first-text latency includes the transfer.
Emitting it earlier from P would move much of the visible delay into the
first inter-token gap. The current design keeps one output owner and a
clear handoff boundary; its latency measurements include that choice.

## Quantify the handoff before expecting a gain

For L layers, prompt length P, Hkv heads, width D, and s bytes per element,
the logical payload is

$$
\mathrm{bytes}_{\mathrm{handoff}}=2LPH_{kv}Ds.
$$

Qwen3-4B BF16 has 36 layers, eight KV heads, and head width 128, giving
147,456 bytes, or 144 KiB, per prompt token. A 2048-token prompt transfers
288 MiB of logical KV. Host staging moves that payload across both a
device-to-host and a host-to-device leg. A counter reporting 288 MiB once
does not describe the total link bytes across both legs.

A basic latency decomposition for the first output includes queueing,
prefill, pack, transfer/import, and event delivery. It is an accounting
model, not a claim that all stages overlap or that their costs are fixed.
Gather, scatter, synchronization, and allocation can matter alongside raw
bandwidth. A long prompt followed by a short answer offers little decode
work over which to amortize the handoff.

Separating phases can help when saved interference and independent progress
outweigh these added costs and when work is balanced between the owners.
If P is the bottleneck, D receives requests too slowly to build a useful
batch. If D is the bottleneck, destination reservations limit admission and
leave P idle. An equal one-prefill/one-decode split is a workload choice,
not an automatically balanced service.

For a one-output budget or first-token EOS, there is no continuation and
the payload is omitted. A zero-output budget runs no model forward. These
cases follow from the pending-token invariant rather than requiring a
transfer of KV that nobody will read.

## Cancellation and failure have two owners to retire

A waiting job can terminate without computation. A cancelled prefiller
finishes its current chunk before releasing pages. A ready packet for a
cancelled job is discarded and acknowledged so P cannot remain blocked
forever. A decoding job releases D's visible pages and future reservations
after its current forward reaches a safe boundary.

The job carries both a terminal reason and a `source_released` condition.
A reason such as cancellation is not enough to emit final completion while
P still holds source memory. D reports the terminal only after the cleanup
conditions are satisfied. This makes a terminal event an ownership boundary
inside the process, though it still cannot guarantee that a network client
received the event.

The session bounds outstanding requests and pending token events. Slow
consumers cause explicit cancellation rather than successful truncation.
Completed jobs continue counting against admission until their terminal
events are drained. This prevents output retention from silently becoming
an unbounded queue outside the KV budget.

Each owner is a thread that constructs, calls, and closes its own LLM.
Initialization is serialized because graph capture and allocations interact
within one process; captured streams are device-local. Steady-state GPU
work can overlap, while Python's shared interpreter still contributes host
constraints. A fatal owner failure stops both, drains acknowledgement
dependencies, and cleans up. There is no transparent restart or distributed
fault-tolerance protocol.

## Compare with two ordinary replicas

The relevant resource control has two full model replicas too. Each ordinary
replica handles prefill and decode for its assigned requests; it incurs no
cross-device KV handoff. Comparing disaggregation only with one GPU would
confound phase separation with adding hardware and model capacity.

[![Two ordinary replicas each run both phases. The disaggregated pair dedicates one GPU to prefill and one to decode with host-staged imports. A retained throughput chart shows replicas ahead for short, long, and mixed workloads.](/assets/tinyserve/book18-replica-comparison.svg)](/assets/tinyserve/book18-replica-comparison.svg)

The retained experiment used two RTX A6000s, Qwen3-4B BF16, FlashInfer with
decode graphs, private KV, 16,384 token slots per GPU, an 8192-token chunk
budget, and aggregate admission capacity 32. Both modes used two owner
threads in one process. The control routed request i to replica `i % 2`,
a fixed round-robin policy rather than an optimal balancer.

Fresh processes ran in replicas/PD/PD/replicas order. Each process used
two warmup and five measured cohorts per workload, yielding ten measured
eight-request cohorts per mode and workload. Short meant 128 prompt and
128 output tokens; long meant 2048 and 32; mixed used four short followed
by four long requests at 80 ms intervals.

| Workload | Two replicas tokens/s | Prefill/decode tokens/s | First text p50, replicas → PD |
|---|---:|---:|---:|
| Short | 425.32 | 357.63 | 149.67 → 389.40 ms |
| Long | 168.99 | 62.07 | 886.49 → 2258.63 ms |
| Mixed | 232.71 | 185.44 | 258.48 → 719.49 ms |

These are medians across cohorts; the first-text values summarize each
cohort's p50, not one pooled request distribution. The disaggregated path
achieved 0.84, 0.37, and 0.80 times replica throughput and delayed first
text in all three workloads. It remains an opt-in mechanism rather than
a faster default. The
[portable measurement record](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/docs/benchmarks/m11-prefill-decode-a6000-2026-09-22.json)
retains the workload and implementation identities.

The long cohort moved 2304 MiB of logical KV across eight handoffs. Its
median sum of import times was 1711.11 ms, including both transfer legs,
scatter, and synchronization. P also processed prompts individually, while
the control could pack prompts. D's long-workload batches averaged only
1.97 rows. The result therefore does not isolate transport as the sole
cause or establish that every disaggregated design is slower.

A separate instrumented GPU trace observed 34.70 ms of overlapping GEMM
execution across owners. That establishes real overlap, not a throughput
win or an occupancy claim. The headline runs had unlocked GPU clocks and
finite cohorts, so their throughput is not a saturation-capacity estimate.
Socket-visible text-chunk gaps are likewise not exactly one-token ITL.

## Follow the implementation

| Source at runtime snapshot `e20a348` | Responsibility |
|---|---|
| [disaggregated.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/disaggregated.py) | Two owners, reservations, pack/import, acknowledgement, events, and cleanup. |
| [engine.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tinyserve/engine.py) | Existing prefill and decode forwards used by each replica. |
| [test_disaggregated.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/test_disaggregated.py) | CPU page remapping, poisoned tails, budgets, cancellation, and failure ownership. |
| [test_disaggregated_cuda.py](https://github.com/kaix-nv/tinyserve/blob/e20a34815498560f9226a9e057ade31b2b62743f/tests/test_disaggregated_cuda.py) | Scoped exact KV transport and matched single-request CUDA continuation checks. |

The supported path does not combine model-parallel modes, prefix reuse,
quantized KV, hybrid recurrent state, or stochastic handoff. Those would
change what state crosses the boundary and how its owners coordinate.
Within the implemented scope, the central requirement is complete: a
destination reserves space, imports valid history, acknowledges completion,
and lets the source retire before final ownership is declared released.
The measured cost shows why that correctness mechanism must be evaluated
against an equally provisioned serving alternative.

{% include tinyserve-book-nav.html %}
