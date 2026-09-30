---
layout: post
title: "Building tinyperf, chapter 14: A serving simulator"
date: 2026-09-29 12:14:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/14-a-serving-simulator/
excerpt: "Every price so far has been one step. A server never runs one step. Requests arrive when their users send them, join a batch that is already running and leave when their replies end, and a request's latency is its wait for a place plus the steps it shared with everyone else. Chapter 1's Table 1.4 showed the effect: on one RTX A6000 serving Qwen3-8B, the p95 time to first token was 0.77 s at 3 requests per second and 2.7 s at 4. How do you turn priced steps into TTFT and TPOT under load?"
redirect_from:
  - /tinyperf/perf-modeling/2026/08/21/building-tinyperf-m13.html
  - /tinyperf/perf-modeling/2026/08/28/building-tinyperf-m18.html
  - /tinyperf/perf-modeling/2026/08/28/building-tinyperf-m19.html
  - /tinyperf/perf-modeling/2026/08/28/building-tinyperf-m20.html
---

*[Building tinyperf](/series/tinyperf/) · Part IV: Serving · Code: [`tinyperf/serving.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/serving.py), `simulate` and `StepLatencyModel`, and [`tools/bench_trace.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tools/bench_trace.py) · Every table and the plot in this chapter come from `python3 book/scripts/ch14_serving.py`.*

Every price so far has been one step. A server never runs one step.
Requests arrive when their users send them, join a batch that is
already running and leave when their replies end, and a request's
latency is its wait for a place plus the steps it shared with everyone
else. Chapter 1's Table 1.4 showed the effect: on one RTX A6000 serving
Qwen3-8B, the p95 time to first token was 0.77 s at 3 requests per
second and 2.7 s at 4. How do you turn priced steps into TTFT and TPOT
under load?

The short answer: play the server forward. An event loop admits
requests, composes each step as the engine's scheduler does, asks the
step model for its price, advances a clock by it and records when each
request gets its tokens. The step model caches its prices, so thousands
of steps cost hundreds of graphs. Replayed on the arrivals a benchmark
actually sent, it predicted a vLLM server's TPOT with a typical error
of 2% and its TTFT of 5–7%, before the runs. It is furthest off where a
run tips into saturation, requests arriving faster than the GPU can
finish them.

By the end of this chapter you will know:

- what TTFT, TPOT, inter-token latency, throughput, goodput and an SLO
  measure, and why they need a simulation;
- the simulator's event loop, from arrival to report;
- how a cache of step prices makes thousands of steps cheap, and why
  its context buckets are safe;
- its scheduling policies: admission, chunked prefill, the chunk step;
- how to replay a benchmark's own arrivals, and why it matters;
- how close it comes to a vLLM server under load.

## Why a queue needs a simulation

**Requests arrive at random.** The standard model is *Poisson
arrivals*: at an average rate of λ per second, requests arrive
independently, so the gaps between them are exponential with mean 1/λ.
Bursts and lulls come with the average. Benchmark tools send exactly
this.

**The batch changes every step.** Engines use *continuous batching*:
the batch is re-formed at every step, so a request joins as soon as it
is admitted and leaves as soon as it finishes, instead of waiting for a
whole batch to complete. Each step's price depends on who is in it
(chapter 6), and who is in it depends on the arrivals and on earlier
steps. More load makes larger batches, larger batches make longer steps,
and longer steps keep requests in the system longer. Textbook queueing
formulas assume a request's service time is independent of the others;
here it isn't.

**Percentiles matter.** Latency targets are written on the tail: the
median request can be answered in a fraction of a second while one in
twenty waits seconds. A simulation produces every request's latency,
the whole distribution.

The metrics, defined here for the rest of the book (Figure 14.1):

- **TTFT**, time to first token: from the moment the client sends a
  request to the moment it receives the first output token, including
  waiting, the prefill and the server's own path (HTTP, tokenization).
- **Inter-token latency (ITL)**: the gap between two consecutive output
  tokens of one request.
- **TPOT**, time per output token: for a request with n output tokens,
  (last token time − first token time) / (n − 1), the mean of its
  inter-token latencies. Benchmarks report its mean, median and p95 over
  requests. On a fixed batch (chapters 1 and 6) it is one decode step.
- **Throughput**: output tokens per second over the run, from its start
  to the last token received; or requests completed per second.
- **SLO**, service-level objective: a latency target, such as TTFT at
  most 1 s and TPOT at most 100 ms for each request.
- **Goodput**: throughput counting only the requests that meet the SLO.

![A timeline: engine steps of different widths above; below, one
request arrives, waits, has its prompt processed in two chunks, then
gains a token at the end of every step.](/assets/tinyperf-book/ch14-request-life.svg)

*Figure 14.1. One request's life. Its prompt rides in two steps beside
other requests' decodes, and the step that finishes the prompt emits
the first token. After that each inter-token gap is one step, longer
when the step carries another prompt's chunk.*

## The event loop

tinyperf's `simulate` is a *discrete-event simulation*: nothing happens
between events, so the clock jumps from one to the next. The events are
the engine's step boundaries, plus the next arrival when the engine is
idle (Figure 14.2).

![Six boxes in a loop: arrivals, admission, compose the step, price the
step, advance the clock, update the requests; pricing calls a step-price
cache; with no request left, a report.](/assets/tinyperf-book/ch14-event-loop.svg)

*Figure 14.2. The event loop. Each turn is one engine step. Only boxes 4
and 5 price anything; admission knows the GPU only through its KV
pool.*

The real function is about 450 lines, with paged memory, prefix caching
and disaggregation (prefill and decode on separate servers; chapter 20)
woven through nested helper functions. Here is the loop in the mode this
chapter validates, as **pseudocode**:

```python
# Pseudocode of simulate(): chunked prefill, admission by reservation.
t = 0
while any request is still to arrive, queued, prefilling or running:
    # 1. arrivals: a step is planned while the previous one runs (chapter 16)
    queue += requests that arrived before the previous step began (by now, if idle)
    if nothing is queued, prefilling or running:
        t = next arrival; continue
    # 2. admission, first come first served
    while queue and len(running) + len(prefilling) < max_batch \
            and reserved + queue[0].prompt + queue[0].max_gen <= kv_budget:
        reserved += queue[0].prompt + queue[0].max_gen
        prefilling.append(queue.pop(0))
    # 3. compose: one token per running request; prompts share the rest
    decoders, budget, chunks = list(running), chunk_tokens - len(running), []
    for w in prefilling while budget > 0:
        c = min(budget, w.prompt - w.prefilled); chunks.append((w, c)); budget -= c
    # 4. price the step
    if chunks:
        price = lat.mixed_step_us(len(decoders), mean context, mean chunk start,
                                  mean chunk size, len(chunks))
    else:
        price = lat.decode_us(padded_batch(len(decoders)), mean context, real=len(decoders))
    # 5. advance the clock
    t += price + lat.sampler_us(len(decoders) + len(chunks))
    # 6. update the requests
    for w, c in chunks:
        w.prefilled += c
        if w.prefilled == w.prompt:          # the last chunk samples the first token
            w.ttft_us = t - w.arrival_us; w.generated = 1; move w to running
    for r in decoders: r.generated += 1
    for r in running with r.generated == r.gen:
        r.finish_us = t; reserved -= r.prompt + r.max_gen; remove r
add the per-request overhead to every TTFT and finish time (chapter 16)
```

Box 4's details belong to later chapters: CUDA-graph padding
(`padded_batch`: the engine replays the graph captured for the next size
up), the mean context and the sampler to chapter 15, the price of a step
carrying prompt chunks to chapter 16. With
`chunk_tokens=None` the loop runs the older *prefill-first* policy: a
step that admits requests is a dedicated prefill of their prompts,
`lat.prefill_us(tokens, n_seqs=len(admit))`, and the running decodes
wait for it.

Three pieces of the real code carry the loop. The scheduler plans each
request with `max_gen`, the cap the client asked for, while `gen` is how
long the reply turns out to be (the fields for preemption, prefix
caching and disaggregation, and a default for `max_gen`, trimmed):

```python
@dataclass
class Request:
    arrival_us: float
    prompt: int
    gen: int                           # actual output length (unknown to scheduler)
    max_gen: int = 0                   # API cap the scheduler must plan for
    ttft_us: float | None = None       # arrival -> first token (end of prefill)
    finish_us: float | None = None
    generated: int = 0
    prefilled: int = 0                 # prompt tokens processed (chunked mode)
    ...

    @property
    def tpot_us(self) -> float:
        return (self.finish_us - (self.arrival_us + self.ttft_us)) / max(1, self.gen - 1)
```

The step price, in the chunked branch (a comment and a branch for MoE
routing by reply position, chapter 9, trimmed):

```python
            new_tokens = sum(c for _, c in chunked)
            kv_now = sum(r.prompt + r.generated for r in decoders) / len(decoders) if decoders else 0
            n_dec = padded_batch(len(decoders), cudagraph_sizes) if decoders else 0
            if chunked:
                pre_kv = max(1, sum(w.prefilled for w, _ in chunked) // len(chunked))
                step_us = lat.mixed_step_us(len(decoders), kv_now, pre_kv,
                                            max(1, new_tokens // len(chunked)), len(chunked))
            ...
            else:
                step_us = lat.decode_us(n_dec, kv_now, real=len(decoders)) if decoders else 0.0
```

And the report, which reads percentiles off the requests (prefix-cache
counters and the other accessors, such as `tpot_ms`, trimmed):

```python
@dataclass
class ServingReport:
    requests: list
    makespan_us: float
    kv_budget_tokens: int
    steps: int
    ...

    def _q(self, values, q):
        if len(values) == 1:
            return values[0]
        qs = statistics.quantiles(values, n=100)
        return qs[min(98, max(0, int(q * 100) - 1))]

    def ttft_ms(self, q=0.5):
        return self._q([r.ttft_us for r in self.requests], q) / 1e3
```

On an idle server the loop reduces to a sum (vLLM, 24 requests at 0.2
per second, almost none overlapping):

```
Worked example  One request on an idle server: Qwen3-8B, RTX A6000, top-p sampling, ms
  1024-token prompt: TTFT = 25.0 overhead + 151.7 step + 0.5 sampler = 177.2; measured median 176.3
   128-token prompt: TTFT = 25.0 overhead + 29.4 step + 0.5 sampler = 54.8; measured median 47.7
  1024-token prompt: TPOT = 24.2 decode step at the mean context + 0.5 sampler = 24.7; measured median 24.4
```

The 25 ms per-request overhead was fitted to these two probes with an
earlier step model; the 128-token probe now reads 7 ms high. It is the
server's path outside the GPU steps (chapter 16). Everything else under
load comes from the loop.

## Thousands of steps, hundreds of graphs

Pricing a step means building its graph and running `execute` (chapter
1's `_graph_report`), a few milliseconds of CPU. A seven-rate sweep
takes 15,059 steps (Table 14.1, the runs of Table 1.4) with 7,898
*distinct shapes*: distinct sets of arguments to the step model, what an
exact model would have to price. The batch's mean context grows by about
a token per step, so shapes rarely repeat. `StepLatencyModel` keeps
every price in a dictionary keyed by the step's shape, and prices
contexts only at multiples of 256 tokens, interpolating between
neighbouring buckets (type annotations, the docstring and the lines
computing the MoE arguments trimmed):

```python
KV_BUCKET = 256      # decode kv lengths are bucketed for the latency memo

    def _price(self, phase, batch, seq, shared=0, moe_real=None, route=None):
        key = (phase, batch, seq, shared) + ...
        if key not in self._memo:
            ...
            self._memo[key] = self._traverse_us(phase, batch, seq, shared=shared, **graph)
        return self._memo[key]

    def decode_us(self, batch, kv_len, shared_prefix=0, real=None, reply_pos=None):
        lo = int(kv_len // KV_BUCKET) * KV_BUCKET
        f = (kv_len - lo) / KV_BUCKET
        ...
        p_lo = self._price("decode", batch, max(lo, 1), shared_prefix, moe_real, route)
        us = p_lo if f == 0 else p_lo + f * (self._price("decode", batch, lo + KV_BUCKET, shared_prefix, moe_real, route) - p_lo)
        ...                                # attention at the real batch, not the padded one (chapter 15)
        return us
```

`_traverse_us` builds and prices the graph, one per pipeline stage when
there are several. A mixed step is cached in three parts: the work
proportional to tokens at its exact row count (a GEMM's cost is not
monotone in its rows, chapter 3), the chunk's attention with its
context rounded up to a bucket, and the decodes' attention,
interpolated.

```
Table 14.1  The step-price cache over one load sweep: Qwen3-8B, RTX A6000, 240 requests of 1024 tokens in, 128 out
  req/s  engine steps  with a prompt  distinct shapes  graph builds  new builds, shared cache
      1          7777            239             2724           106                       106
      2          3202            238             2716           188                        91
      3          1543            226             1504           359                       203
      4           731            205              650           399                       141
      5           647            202              548           393                        43
      6           602            194              512           371                        31
      8           557            168              513           334                        24
  all seven rates: 15059 steps, 7898 distinct step shapes, 639 graph builds with one shared cache
```

The cache builds 639 graphs for them. Pass one `StepLatencyModel` to
every `simulate` of a sweep: the last rates build only 24–43 new
graphs.

Why is interpolation safe? A decode step's price is the weight GEMMs,
which don't depend on the context, plus attention: bytes proportional
to the context over a rate, plus a fixed cost per call (chapter 7).
That is a straight line in the context, and interpolation on a straight
line is exact. Where the price bends, it isn't:

```
Table 14.2  What bucketing costs: decode steps interpolated between 256-token buckets, against exact prices
  model             GPU        tier       shapes        worst  at batch, context  price bends at
  qwen3-8b          RTX A6000  calibrated    320       +0.03%              32, 1  none
  qwen3-8b          B200       calibrated    320       +0.02%              32, 1  none
  gpt-oss-20b       RTX A6000  calibrated    320       -0.14%            32, 128  128-token window
  glm-5.3-flash     H100 SXM   projected     320       -1.61%            1, 2049  indexer, past 2048
  deepseek-v4-flash H100 SXM   projected     320       -5.23%            1, 2052  indexer, past 2048
  mixed steps, qwen3-8b, RTX A6000, chunks resuming at 257-3000 tokens, context rounded up: +0.10% to +1.09% on 5 shapes
```

The shapes are batches 1 and 32 at five points in every bucket up to
8,192 tokens. Qwen3-8B's 0.03% is not a bend: the first bucket's lower
price is taken at context 1 but weighted as if at 0. gpt-oss-20b's
sliding-window layers stop growing at 128 tokens (chapter 8), inside
the first bucket. The sparse-attention models are the real hazard:
their indexer, the kernels that choose which tokens to read (chapter
8), switches on just past 2,048 tokens, a jump that interpolation
spreads from 2,048 to 2,304, up to 5.2% low. For a dense model the
buckets cost nothing measurable; don't interpolate across a jump: price
both sides of it exactly.

## Scheduling policies

**Admission: reserve or page.** A request's KV cache grows by a token
per step, to a length the scheduler doesn't know. With
`kv_paging="reservation"` a request is admitted only if its prompt plus
`max_gen` fits beside what the running requests have reserved. With
`kv_paging="paged"` it is admitted against what requests actually hold,
in pages that grow as they generate, as in vLLM; when pages run out, the
newest request is preempted (chapter 17). The difference matters when
clients ask for far more tokens than they use:

```
Table 14.3  Admission when outputs are uncertain: 200 requests of 1024 tokens in, 256-512 out, capped at 8192
  req/s  admission   TTFT p50 ms  TTFT p95 ms  TPOT ms  tok/s  peak batch  preemptions
      1  reservation         257         3284     36.3    388          21            0
      1  paged               251          420     36.9    388          30            0
      2  reservation       24213        50254     39.8    491          21            0
      2  paged               454         2454     64.0    695          64            0
  KV pool: 196,704 tokens; a reservation holds 1024 + 8192 tokens, so 21 requests fit; the batch cap is 64
```

Reservation holds 9,216 tokens for replies that use at most 1,536, so
the batch never passes 21, and at 2 requests per second the queue grows
for as long as the load lasts. Paged admission fills the batch to its
cap without a preemption and serves 42% more tokens per second. The
benchmarks below ask for exactly the tokens they use, so there the
batch cap binds first (64 × 1,152 = 73,728 tokens against a pool of
196,704) and reservation admits what paging would; chapter 17's Table
17.6 shows where reservation costs.

**Chunked prefill.** A long prompt run on its own stalls every running
decode for its whole length. *Chunked prefill* bounds each step by a
*token budget* (vLLM's `max_num_batched_tokens`): the decodes are
scheduled first, one token each, and waiting prompts fill what is left,
in chunks. The step that finishes a prompt samples its first token.
Chapter 16 prices the mixed step. Table 14.4 turns the knob on long,
varied prompts:

```
Table 14.4  The token budget: 160 requests of 2048 in, 128 out, both +-90%, at 1.5 req/s, Qwen3-8B, RTX A6000
  scheduler             TTFT p50 ms  TTFT p95 ms  TPOT ms  TPOT p95 ms  longest step ms  steps with prompts
  prefill-first                 571         2377     69.1        132.4             1785                 133
  budget 256                   4980         8645     62.1         74.8               79                1403
  budget 512                    875         3885     67.3        118.5              131                 713
  budget 1024                   740         3091     73.5        150.3              220                 378
  budget 2048 (vLLM's)          614         2192     68.9        138.0              348                 221
  budget 8192                   597         2108     67.4        135.0             1191                 131
  output throughput 182-186 tok/s in every row; longest step: the step's price, sampler excluded
```

The budget trades the first token against the worst gap between tokens.
Prefill-first has the lowest median TTFT, but one of its steps runs
1,785 ms, a pause every running request sees. A budget of 256 caps every
step at 79 ms, but long prompts then need many steps to get in, and the
median TTFT reaches 5 s. vLLM's budget on this GPU, 2,048, keeps TTFT
near prefill-first's with a longest step of 348 ms. Throughput barely
moves: at this load every policy but the smallest budget keeps up.

**The chunk step.** A step giving each of `batch` sequences m new
tokens against an existing context is chapter 6's decode phase with
`chunk=m`: m query rows per sequence, attending to the context plus the
causal half of the chunk. `StepLatencyModel.chunk_us` prices it. With
m = 1 it is a decode step; it also prices a prompt chunk's attention in
a mixed step, the uncached part of a prompt whose prefix is cached
(chapter 17), and the check of drafted tokens in speculative decoding
(a draft proposes tokens, the model checks them in one step; chapter
19).

## Replaying a benchmark

For a what-if with no benchmark behind it, `poisson_requests(n, rate,
prompt, gen, seed)` draws exponential gaps. To predict a measured run,
the simulator needs the arrivals the benchmark actually sent. vLLM's
`vllm bench serve` is deterministic in its seed, so
`tools/bench_trace.py` rebuilds its trace with the same calls (numpy):

```python
def bench_trace(num_prompts, input_len, output_len, range_ratio=0.0, seed=0, burstiness=1.0, num_special=0):
    rng = np.random.default_rng(seed)
    real = input_len - num_special
    prompts = rng.integers(math.floor(real * (1 - range_ratio)), math.ceil(real * (1 + range_ratio)) + 1, size=num_prompts)
    ...                                # the output lengths, drawn the same way
    np.random.seed(seed)
    gaps = np.random.gamma(shape=burstiness, scale=1.0 / burstiness, size=num_prompts)
    unit = np.cumsum(gaps)
    unit = unit * num_prompts / unit[-1]
    return {...}
```

A gamma distribution of shape 1 is the exponential, so at the default
burstiness the arrivals are Poisson. The benchmark rescales them so the
last request goes out at exactly n/rate; the schedule at any rate is the
unit-rate schedule divided by the rate, which is all `bench_requests`
does (docstring trimmed):

```python
def bench_requests(trace: dict, rate_per_s: float) -> list:
    return [Request(arrival_us=a / rate_per_s * 1e6, prompt=pl, gen=g, max_gen=g)
            for a, pl, g in zip(trace["unit_arrivals_s"], trace["prompts"], trace["outputs"])]
```

The rebuilt traces match what the engine's logs recorded for two
sweeps, request for request:

```
Recorded  The rebuilt traces against the requests two measured sweeps actually sent
  1024 in, 128 out, 7 rates             1680 requests: 0 prompt and 0 output lengths differ; send times off by 1.77 ms on average, 9.8 ms at most
  2048 in, 128 out, both +-90%, 5 rates  800 requests: 0 prompt and 0 output lengths differ; send times off by 2.46 ms on average, 13.8 ms at most
```

Near saturation a run's queue depends on its particular bursts:

```
Table 14.5  One process, many traces: TTFT p50 in ms, Qwen3-8B, RTX A6000, 240 requests of 1024 in, 128 out
  req/s  trace 1: measured   model  trace 2: measured   model  model, 8 other Poisson traces
    3.0                327     353                302     311             304 to 388
    4.0                660     679               1015    1042             456 to 1625
```

At 4 requests per second two runs of the same workload, differing only
in the benchmark's seed, measured median TTFTs 1.54 times apart, and on
their own traces the model reproduces both; eight other samples of the
same process span 456 to 1,625 ms (chapter 18).

> **Field note: three predictions.** Each cell of Table 14.8's first
> three sweeps had three predictions written before its run: the mean
> over eight other Poisson samples, which read 0.71–1.26 on the median
> TTFT; the older step model on each run's own trace, 0.84–1.26; and
> the model with three fixes found in earlier runs' step logs, among
> them attention on a padded step's real rows only (chapter 15),
> 0.97–1.10 (Table 14.8). Comparing with the wrong sample of arrivals
> and a wrong step price were separate errors, each with its own fix.

## The shape of a load curve

Table 14.6 runs the simulator on one benchmark trace at a range of
rates, with an SLO of TTFT at most 1 s and TPOT at most 100 ms:

```
Table 14.6  The simulator against load: Qwen3-8B, RTX A6000, 240 requests of 1024 tokens in, 128 out
  req/s  decodes/step  TPOT ms  TTFT p50 ms  TTFT p95 ms  tok/s  done/s  good requests  goodput tok/s
   0.50           2.1     27.5          227          342     64    0.50        240/240             64
   1.00           3.9     31.1          233          370    126    0.99        240/240            126
   2.00           9.5     39.9          247          499    248    1.94        240/240            248
   3.00          19.8     59.6          355          787    364    2.85        239/240            363
   3.25          23.8     68.2          391          990    392    3.06        225/240            368
   3.50          29.2     81.1          437         1076    418    3.26        177/240            308
   3.75          35.9     94.5          587         1423    442    3.45         98/240            180
   4.00          41.7    110.6         1378         3126    451    3.52         22/240             41
   5.00          47.1    115.6         6162        12896    463    3.62          7/240             13
   6.00          50.6    115.8         8533        19263    473    3.70          0/240              0
   8.00          54.7    113.8        11238        26748    490    3.82          0/240              0
  SLO: a request is good if its TTFT <= 1 s and its TPOT <= 100 ms; goodput: the good requests' tokens over the run, counted by this script
```

![Two panels against offered load, 0.5 to 8 requests per second: TTFT
median and p95 on a log scale, flat to about 3.5 then rising steeply;
mean TPOT rising to about 110 ms at 4, then level. The model's lines
run through the measured points.](/assets/tinyperf-book/ch14-load-curves.svg)

*Figure 14.3. TTFT and TPOT against load: the simulator (dashed) and
vLLM (solid lines through the measured points) on the same trace.*

The mean batch grows with the load, from 2 decodes per step to 55,
and TPOT follows it from 27.5 ms to about 115: every added sequence
adds its cache read to every step (chapter 6). While the GPU keeps up,
TTFT stays near one prefill plus a short wait. Past about 3.5 per
second it can't: completions level off at 3.5–3.8 per second, the
excess waits in the queue, and TTFT grows by seconds while TPOT stops
rising. Goodput shows where to stop: it peaks at 3.25 requests per
second with 225 of 240 requests inside the SLO; at 4 only 22 are, while
throughput is still rising. (`ServingReport` doesn't compute goodput;
the script counts the good requests' tokens.) Flat, then a *knee*, the
load where TTFT turns sharply upward: chapter 18 is about where it
falls and why it is so sharp.

## How close is it?

The evidence is Qwen3-8B in bf16 on one RTX A6000, served by vLLM
0.15.1 (at most 64 running requests, prefix caching off, a budget of
2,048 tokens) and loaded by `vllm bench serve` with random-token
prompts, forced output lengths and Poisson arrivals. The server sampled
with top-p, its default for this model, and the model prices that. The
simulator runs `simulate(..., max_batch=64, chunk_tokens=2048)` on each
run's own trace, the set-up of `test_serving_dynamics_envelope` in
`tests/test_core.py`.

**In-sample.** The first sweep is the one the serving mechanisms were
developed against; chapter 1's Table 1.4 showed its p95 TTFT. The other
metrics:

```
Table 14.7  In-sample: Qwen3-8B under vLLM on one RTX A6000, 240 requests of 1024 tokens in, 128 out
  req/s  TTFT p50 ms  model/meas  TPOT ms  model/meas  tok/s  model/meas
      1        221.7       1.050    30.56       1.019    126       1.000
      2        237.9       1.039    39.52       1.010    248       1.000
      3        328.2       1.083    59.45       1.003    364       1.001
      4       1063.3       1.296   106.36       1.040    456       0.990
      5       5967.5       1.033   112.86       1.024    468       0.989
      6       8341.7       1.023   113.19       1.023    478       0.990
      8      11074.7       1.015   111.42       1.021    493       0.993
  typical error: TTFT p50 7.3%, TPOT 2.0%, tok/s 0.6%; the p95 TTFT is Table 1.4's
```

TPOT is within 4% at every rate and throughput within 1.1%. The median
TTFT is 2–8% high except at the knee, 4 requests per second, where it
reads 30% high; Table 1.4 shows how far a 2% step error moves TTFT
there (chapter 18).

> **Field note: the first prediction under load.** The first
> comparison with this sweep put the knee one rate late, at 5 requests
> per second, read the median TTFT 11–30% low below it and throughput
> 10–11% high past it
> (`data/validation/comparison_qwen3_8b_rtx_a6000_serving.txt`). Every
> step price had been checked on fixed batches. What was missing
> happened between them: a mixed step had been priced as the larger of
> its parts, and the server's path outside the steps had no cost.
> Chapter 16 tells both stories.

**Held out.** Five later sweeps had their predictions committed before
they ran. Three probe the scheduler near saturation: the same workload
on two new seeds, and long prompts of varied length. The current model
still gives the predictions as written. For these three the engine also
logged its steps, so the table compares the simulator's schedule with
the engine's: the number of steps, and the mean decodes in each.
`simulate(..., trace=steps)` records each step's start, end, decodes and
chunks.

```
Table 14.8  Held out: sweeps whose predictions were written before the runs, model/measured
  sweep: tokens in/out     req/s  measured TTFT p50 ms  TTFT p50  TTFT p95   TPOT  tok/s  engine steps  decodes/step
  2048/128, +-90%, seed 1    1.0                   558     0.977     1.031  1.006  1.000         0.989         1.011
                             1.5                   624     0.969     1.007  0.976  1.001         1.012         0.989
                             2.0                   997     0.974     0.983  0.966  1.002         1.040         0.962
                             2.5                  2633     1.095     1.166  1.032  0.983         0.998         1.002
  1024/128, seed 1           3.0                   327     1.078     1.035  1.006  1.000         0.993         1.007
                             3.5                   424     1.012     1.003  1.007  1.000         0.994         1.006
                             4.0                   660     1.028     1.120  1.033  0.994         0.985         1.015
                             4.5                  2484     1.082     1.071  1.019  0.993         0.992         1.008
                             5.0                  4176     1.008     1.042  1.022  0.992         0.997         1.003
  1024/128, seed 2           3.0                   302     1.033     0.987  1.012  1.000         0.990         1.010
                             3.5                   430     1.012     1.018  1.008  1.000         0.993         1.007
                             4.0                  1015     1.027     1.067  1.019  0.993         0.989         1.011
                             4.5                  1975     1.091     1.071  1.020  0.992         0.993         1.007
                             5.0                  3549     1.026     1.025  1.019  0.994         1.000         1.000
  the current model against the predictions written before the runs: TTFT and TPOT within 0.2%, tok/s within 0.5%
  two more held-out sweeps (512/384 and 1536/192, +-50%; chapter 18), 12 rates: TTFT p50 0.79-1.07, p95 0.72-1.09, TPOT 0.96-1.02, tok/s 0.99-1.01
  512/384 sweep: TTFT p95 0.72 at 2.75 req/s (measured median 224 ms); TTFT p50 0.79 at 3.5 req/s (measured median 1538 ms)
  typical error, all 26 held-out cells: TTFT p50 5.4%, p95 7.1%, TPOT 1.9%, tok/s 0.5%
```

On the three sweeps every TTFT lies within 0.97–1.17, TPOT within
0.97–1.03 and throughput within 2%, below the knee and far past it. The
last two columns test the loop itself: within 4% of the engine's number
of steps and of its mean decodes per step, the simulator makes the same
scheduling decisions. The other two sweeps hold the worst cells, both in
the 512/384 sweep: 0.72 on the p95 at 2.75 req/s, 0.79 on the median at
3.5, where it tipped into saturation. Over all 26 held-out cells the
typical error is 5.4% on median TTFT, 7.1% on p95 and 1.9% on TPOT. What
was fitted: the GPU's constants (chapter 4), kernels timed alone
(chapters 3, 7, 15 and 16) and the 25 ms overhead from two idle probes.
Nothing was fitted to a loaded run.

## Where it breaks

- **Averages inside a step.** A decode step is priced at its requests'
  mean context, and several prompts' chunks as equal chunks at their
  mean size and start. Both are exact only where the price is linear.
- **The knee.** Small step errors grow there into large TTFT errors, so
  a point prediction is worth little; chapter 18 gives intervals.
- **One scheduler.** First come, first served, as vLLM 0.15.1 does it:
  no priorities, no cancellations, one replica.
- **The per-request overhead** is one fitted constant, the same for
  every request (chapter 16).
- **Synthetic traffic.** Every validated run used random-token prompts
  and forced output lengths, on one dense model and one GPU. Real
  traffic has burstier arrivals, long-tailed lengths, shared prefixes
  and replies that end early.

## What you built

- The serving metrics, defined: TTFT, inter-token latency, TPOT,
  throughput, SLO and goodput.
- A discrete-event simulator whose events are engine steps, with one
  price per step and a report of percentiles.
- A step-price cache: 639 graphs for 15,059 steps, its context buckets
  exact where the price is linear in the context.
- Admission by reservation or by pages, chunked prefill under a token
  budget, and the chunk step.
- Benchmark replay, request for request.
- Held-out evidence: TTFT typical error 5.4% (median) and 7.1% (p95),
  TPOT 1.9%, and the engine's schedule within 4%.

## Exercises

1. **Inter-token latency.** Run `steps = []`;
   `simulate(..., trace=steps)`. A request gains a token at the end of
   each step between its first token and its finish (subtract the 25 ms
   overhead from the report's times first). For Table 14.4's workload,
   which budgets keep the p99 gap under 200 ms, and what TTFT do they
   cost?
2. **Burstier traffic.** Rebuild Table 14.6's trace with
   `tools/bench_trace.py --burstiness 0.5` and replay it. How far does
   the goodput peak move?
3. **Bucket size.** Set `KV_BUCKET` to 64 and to 1,024 and rerun Table
   14.1's sweep and Table 14.7. How many graphs are built, and does any
   ratio move?
4. **Priorities.** Give a fifth of the requests a tighter SLO (TTFT at
   most 300 ms) and admit them first. At what rate does each class's
   goodput collapse?

---

*[← Chapter 13: Training]({% post_url 2026-09-29-tinyperf-13-training %}) · [Contents](/series/tinyperf/) · [Chapter 15: What a decode step is made of →]({% post_url 2026-09-29-tinyperf-15-what-a-decode-step-is-made-of %})*
