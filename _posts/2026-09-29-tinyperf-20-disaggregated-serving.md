---
layout: post
title: "Building tinyperf, chapter 20: Disaggregated prefill and decode"
date: 2026-09-29 12:20:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/20-disaggregated-serving/
excerpt: "A GPU that runs prefill and decode makes them take turns: a step that carries part of a new prompt lasts several decode steps (chapter 16), and every running request waits for it. One remedy is structural. A prefill server computes each prompt's KV cache and first token, the cache is shipped to a decode server, and the decode server generates the reply without ever seeing a prompt. Should prefill and decode run on separate GPUs? When does splitting them win, which latency target does it serve, and what does moving the cache cost?"
redirect_from:
  - /tinyperf/perf-modeling/2026/08/31/building-tinyperf-m37.html
  - /tinyperf/perf-modeling/2026/09/22/building-tinyperf-m66.html
  - /tinyperf/perf-modeling/2026/09/24/building-tinyperf-m71.html
---

*[Building tinyperf](/series/tinyperf/) · Part IV: Serving · Code: [`tinyperf/disagg.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/disagg.py), `simulate_disagg`, `kv_transfer_us` and `simulate_replicas`, and [`tinyperf/serving.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/serving.py), `kv_producer_us` and `simulate` · Every table and both plots in this chapter come from `python3 book/scripts/ch20_disagg.py`.*

A GPU that runs prefill and decode makes them take turns: a step that
carries part of a new prompt lasts several decode steps (chapter 16),
and every running request waits for it. One remedy is structural. A
*prefill server* computes each prompt's KV cache and first token, the
cache is shipped to a *decode server*, and the decode server generates
the reply without ever seeing a prompt. Should prefill and decode run on
separate GPUs? When does splitting them win, which latency target does
it serve, and what does moving the cache cost?

The short answer: splitting serves TPOT. The decode GPU is never
interrupted, so the time between tokens is set by the decode batch alone
and its tail stays close to its mean. Everything else costs: the cache
crosses a link (hidden here, because it is sent layer by layer during
the prefill), each request passes two servers and pays an extra step,
saving the cache adds work to each prefill step, and the prompts get one
GPU instead of two. On two RTX A6000s serving Qwen3-8B, a colocated pair
wins TTFT at every load and throughput at high load (below 4 requests
per second both serve the offered load, a tie); one prefill plus one
decode GPU (1P1D) wins TPOT at every load. Which sustains more load
depends on the service-level objective (SLO). The numbers committed
before the runs order the two designs as measured in all 16 TTFT and
TPOT comparisons.

By the end of this chapter you will know:

- what a disaggregated deployment does with a request, and what each
  part costs;
- how tinyperf builds one from two copies of chapter 14's simulator and
  a priced link;
- what saving the cache adds to a prefill step, and how a server whose
  host waits for it holds finished answers for a step;
- how close the model is on two GPUs, and how to choose a split on more.

## One GPU, two kinds of work

Prefill is math-bound and decode streams weights (chapter 6): on an RTX
A6000, a 1,024-token prompt of Qwen3-8B is a step of about 152 ms, a
decode step at batch 1 about 25. With chunked prefill (chapter 14) the
prompt rides beside the decodes in one forward (chapter 16), so every
decoding request waits the whole step.

A disaggregated deployment puts a small router, the *proxy*, in front of
two servers. It sends each request to the prefill server with
`max_tokens=1`, then to the decode server, and streams the decode
server's reply to the client. vLLM moves the cache with a *KV
connector*, a plug-in on both servers. On the prefill server (the *KV
producer*) it *saves* and sends each layer's KV; on the decode server it
receives it. The decode server gets the cache of all but the last
prompt token, so its first step computes one token and samples the
first output. With one GPU per role the deployment is *1P1D*; in
general, xPyD. Figure 20.1 follows four requests through both designs.

![Four requests on two GPUs, colocated and
disaggregated.](/assets/tinyperf-book/ch20-timeline.svg)

*Figure 20.1. Four requests on two GPUs, as the model schedules them.
In the pair, r2's prompt lands on GPU 0 while r0 decodes, and r0's next
token waits for the whole step. In 1P1D the decode GPU runs only short
steps; the 26.9 ms before its first step is the prefill server's 25 ms
per-request overhead plus the 1.86 ms hop through the proxy. Two
answers are held for a step, explained below.*

```
Worked example  Figure 20.1's four requests: 1024 tokens in, 20 out, arriving at 0, 100, 420 and 560 ms, model
  request   arrives ms   colocated: TTFT ms  largest gap ms   1P1D: TTFT ms  largest gap ms
  r0                 0                179.1           164.3           402.6            26.9
  r1               100                179.1           164.3           302.6            26.9
  r2               420                219.7            25.1           436.0            25.9
  r3               560                229.1            25.1           296.0            25.9
  1P1D: the answers of r0 and r2 wait for the next prompt's forward: 161.9 and 161.9 ms
```

In the pair, r0 and r1 each wait 164.3 ms for a token that otherwise
comes every 25. In 1P1D no gap passes 26.9 ms, but first tokens come
later: 296–436 ms against 179–229.

## What splitting costs

With *hop* the proxy's cost per pass, *overhead* a server's per-request
overhead outside its steps (25 ms, chapters 14 and 16) and *wait* the
queueing, including async scheduling's extra step (chapter 16):

```
TTFT, colocated  =  hop + overhead + wait + prefill step
TTFT, 1P1D       =  hop + overhead + wait + prefill step + save  [+ a held step]
                  + hop + overhead + wait + one-token step        [+ transfer not hidden]
TPOT, 1P1D       =  decode steps at the decode server's batch, never a prompt
transfer         =  KV bytes per token × prompt / tp / link bandwidth + link latency
not hidden       =  max(0, transfer − prefill)                    (sent layer by layer)
```

On an idle server:

```
Worked example  One 1024-token request on an idle server, Qwen3-8B, RTX A6000 pair, model, ms
  colocated: hop 1.86 + overhead 25.0 + prefill step 152.2 = 179.1
  1P1D:      hop 1.86 + overhead 25.0 + prefill step 152.2 + save 9.7
             + hop 1.86 + overhead 25.0 + one-token step 24.7 = 240.3
  the difference, 61.2 ms: a second hop and per-request overhead, the save, and the decode server's first step
```

The hop was measured as direct against proxied requests (notes in
`data/validation/predictions_qwen3_8b_rtx_a6000_pd_before_measurement.txt`).
The transfer is missing because it hides. It is the cache's size over
the link; with tensor parallelism each rank ships its shard over its own
link:

```
Table 20.1  Moving one prompt's KV cache: the transfer on three links, and the prefill it can hide behind
                                          transfer ms
  model        tp KB/token  prompt  MB/GPU  PCIe pair  InfiniBand   NVLink  prefill ms, H100
  Qwen3-8B      1      144    1024     151       23.1         3.0     0.34              20.2
  Qwen3-8B      1      144    8192    1208      184.4        24.2     2.69             173.0
  Llama-3-70B   4      320    1024      84       12.8         1.7     0.19              61.6
  Llama-3-70B   4      320    8192     671      102.5        13.4     1.49             457.6
  links: the measured pair 6.55 GB/s; H100 InfiniBand 50 GB/s per GPU, NVLink 450; prefill at H100 datasheet rates (projection)
  Qwen3-8B on the RTX A6000 pair (calibrated): prefill 151.7 ms at 1024 tokens, 1345.3 at 8192; the transfer is 0.15 and 0.14 of it
  weight bytes per cached byte, Llama-3-70B over Qwen3-8B: 3.9 (tp=4 divides cache and math alike)
  a latent cache (chapter 8): DeepSeek-V4-Flash 8300 bytes per token, 18 times less than Qwen3-8B's
  the pair's rate: a 16 MB NCCL send/recv inside a CUDA graph, 2.56 ms: 6.55 GB/s, the model's p2p_bw_gbps
  the device file as shipped: p2p_bw_gbps 0, nvlink_bw_gbps 56; its 1024-token transfer 2.7 ms
```

The transfer and the prefill both grow with the prompt, so their ratio
is roughly a property of model and hardware: cache bytes per token over
the link, against math per token over the GPU. Qwen3-8B's cache crosses
this pair in 0.15 of its prefill time, and InfiniBand on H100s in 3.0 ms
against 20.2. Llama-3-70B does about 4 times the math per cached byte
(tp=4 divides both alike): 1.7 ms against 61.6. A latent cache is
negligible.

The pair's 6.55 GB/s is a send inside a CUDA graph; the device file
lacks it (Where it breaks), so every run here passes
`p2p_bw_gbps=6.55, nvlink_bw_gbps=4.0`. On the connector's own clock:

```
Recorded  The KV transfer on the connector's own clock: 1P1D, 1024-token prompts, 1 and 3 req/s, medians, ms
  per layer: 2.55 and 2.59 (a control message to the decode server 0.58 and 0.57; the NCCL send and the wait for it 1.89 and 1.95)
  the send thread busy per request: 88 and 89; priced by the model: 23.1
  of the 65 ms excess, 36 control messages 21; a layer's 4.2 MB sent at 2.2 GB/s, a 4 MB send/recv alone at 5.3
  the prefill step's wait for its sends to drain: 0.03; the decode server's in-step wait for the KV: 0.01
  not measured, a 128-token prompt: the model's prefill 29.4 and transfer 2.9; 36 control messages alone 20.8
```

The model's price is about four times too low. The control messages
account for 21 ms of the 65 ms excess; the rest is the sends running at
a third of the link's rate while the forward runs beside them (not
profiled). Here it doesn't matter. A layer of a 1,024-token prefill
computes in about 4.2 ms (151.7 over 36 layers) and its send takes 2.6,
so the sends keep pace and finish with the step. At 128 tokens the
control messages alone would take 20.8 ms of a 29.4 ms prefill; whether
the sends still hide is unmeasured.

## The model: two engines and a link

`simulate_disagg` builds the deployment from chapter 14's `simulate`.
Trimmed below: the docstring, most of the signature, the two step
models (`lat_p`, `lat_d`), the settings both pools share (`common`),
and the bookkeeping that copies results back:

```python
def simulate_disagg(p, device, requests, prefill_workers=2, ..., prefill_host_sync=False, ...):
    ...
    order = sorted(requests, key=lambda r: r.arrival_us)
    online_p = (lat_p.calibration.online_overhead_us or 0.0) if lat_p.calibration else 0.0

    # ---- prefill pool: engines whose requests stop after the first token
    pre = {}
    for w in range(prefill_workers):
        mine = order[w::prefill_workers]
        shadow = [Request(r.arrival_us, r.prompt, gen=1, max_gen=1) for r in mine]
        simulate(p, device, shadow, latency_model=lat_p, host_sync_forward=prefill_host_sync, kv_producer=True,
                 ...)
        for r, s in zip(mine, shadow):
            pre[id(r)] = (w, s)

    # ---- transfer: layer-by-layer during the prefill, queued per worker link
    tp_p = lat_p.tp
    link_free = [0.0] * prefill_workers
    arrive_d = {}
    for r in sorted(order, key=lambda r: pre[id(r)][1].ttft_us + r.arrival_us):
        w, s = pre[id(r)]
        engine_done = r.arrival_us + s.ttft_us - online_p        # the prefill's end on the engine
        xfer = kv_transfer_us(p, device, r.prompt, recipe, link=kv_link, tp=tp_p)
        if transfer_overlap:
            start = max(link_free[w], engine_done - lat_p.prefill_us(r.prompt))
            done = max(engine_done, start + xfer)
        else:
            done = max(link_free[w], engine_done) + xfer
        link_free[w] = done
        response = r.arrival_us + s.ttft_us                    # the prefill server's answer reaches the proxy
        arrive_d[id(r)] = max(response + hop_us, done)

    # ---- decode pool: engines whose requests arrive with their prompt's KV
    reports = []
    for w in range(decode_workers):
        mine = order[w::decode_workers]
        shadow = [Request(arrive_d[id(r)], r.prompt, gen=r.gen, max_gen=r.max_gen or r.gen,
                          kv_loaded=max(0, r.prompt - 1)) for r in mine]
        reports.append(simulate(p, device, shadow, latency_model=lat_d, **common))
        for r, s in zip(mine, shadow):
            r.ttft_us = s.arrival_us + s.ttft_us - r.arrival_us + hop_us     # + the front door
            ...
```

The **prefill pool** is `simulate` on copies that stop after one token,
as the proxy asks; `kv_producer=True` charges the save and
`host_sync_forward` the held answers. The **transfer** starts when the
prompt's prefill starts (the answer's time less the prompt's prefill
time), so only what outlasts the prefill delays the request; transfers
from one worker queue on its link. For a held answer, `engine_done` is
the delivery, so the transfer starts a step late: harmless here (23 ms
against 152). The **decode pool** is `simulate` on copies that arrive
once both answer and cache have come, with `kv_loaded` tokens treated as
cached: the first step computes one token, stays in the full CUDA graph
and costs a decode step (chapter 15). `kv_transfer_us` is the formula
(docstring, type annotations and the refusals of a missing link
trimmed):

```python
def kv_transfer_us(p, device, prompt_tokens, recipe=None, link="fabric", tp=1) -> float:
    nbytes = _kv_local_bytes_per_token(p, 1, recipe) * prompt_tokens / tp
    if link == "fabric":
        ...
        return nbytes / (device.ib_bw_gbps * 1e9) * 1e6 + device.ib_latency_us
    if link == "local":
        bw = device.p2p_bw_gbps or device.nvlink_bw_gbps
        ...
        return nbytes / (bw * 1e9) * 1e6 + device.nvlink_hop_latency_us
```

The colocated baseline, `simulate_replicas`, runs n independent
`simulate`s, each on every n-th request, and adds the hop. Round-robin
matters: each GPU sees every other arrival of a Poisson stream, so its
gaps are sums of two exponential gaps, with 0.71 of the spread and
fewer bursts. A random split would leave each GPU a Poisson stream:

```
Table 20.2  Round-robin smooths arrivals: TTFT on one GPU fed Poisson arrivals, and on a pair fed twice
  the rate, split round-robin or at random; model, median over 16 samples of 240 requests per GPU, 1024 in, 128 out, ms
  req/s per GPU  one GPU p50   p95  round-robin p50   p95  random p50   p95
            3.0          344   730              269   510         334   721
            3.5          427  1051              330   659         424  1010
  measured: the colocated pair at 7 req/s, TTFT p50 311, p95 679; one GPU at 3.5 req/s on three traces, p50 424-437, p95 967-1087
```

At 3.5 requests per second per GPU, round-robin cuts the median TTFT by
a quarter and the p95 by a third; a random split changes little.

## The KV producer's step

At every layer, for each prompt the step finishes, the connector
gathers the layer's KV blocks and queues them for a send thread. The
gathers cost more than the sends. `tools/measure_prefill_step.py` sends
k prompts at once behind a 2,048-token prompt, so they share one step,
and times the forward on the engine's clock (chapter 16), on a plain
server and on a pair's prefill server, same GPU:

```
Table 20.3  The KV producer's step, fitted: the same prompts on a plain server and on a 1P1D prefill server,
  forward time on the engine's clock (median of 6-13 steps), RTX A6000, Qwen3-8B
  prompts/step tokens each  plain ms  producer ms  extra ms  model ms
             1         128      32.5         32.7       0.3       9.7   held out
             1         256      51.2         60.5       9.3       9.7
             1         512      79.7         88.0       8.3       9.7
             1        1024     151.3        160.6       9.3       9.7
             1        2048     289.8        298.4       8.6       9.7
             2         256      78.3         92.8      14.5      13.1
             2         512     149.9        164.3      14.5      13.1
             2        1024     283.0        298.1      15.1      13.1
             4         128      78.3         98.6      20.3      19.8   held out
             4         256     149.0        169.8      20.8      19.8
             4         512     280.4        297.7      17.2      19.8
  least squares over the 9 cells of 256 tokens or more: 269 us per layer for the first prompt, 94 for each further one; the calibration: 269 and 94
  model = 36 layers x (269 + 94 x (prompts - 1)) us; the fitted cells within 2.6 ms
```

The extra cost is flat in prompt length, 8.3–9.3 ms for one prompt
from 256 to 2,048 tokens, so it isn't the data. It grows with the
prompts in a step, each costing less than the one before. The gather
indexes the GPU cache with a tensor that lives on the CPU, and copying
that index to the GPU blocks the host until the layer's work is done.
The pattern is consistent with one host round trip per prompt per layer,
the first also leaving the GPU idle until the host launches again (not
profiled). One held-out cell is unexplained: a lone 128-token prompt
pays 0.3 ms.

The price is a flat cost per step (docstring trimmed):

```python
    def kv_producer_us(self, requests: int) -> float:
        coef = self.calibration.kv_producer_layer_us if self.calibration else None
        if not coef or requests <= 0:
            return 0.0
        first, further = coef
        return -(-self.p.n_layers // self.pp) * (first + further * (requests - 1))
```

`simulate(kv_producer=True)` charges each step for the prompts it
finishes, because the connector saves a prompt split across steps once,
with its last chunk.

`kv_producer_layer_us` is *fitted end to end*, the kind of calibration
field chapter 4 warns about: it came from whole forwards of a running
server, a producer's minus a plain server's, not from a kernel timed
alone, so it absorbs whatever else a producer's step does.

## An answer held for a step

Because the host waits at every layer, the prefill server's forward
blocks its engine thread, and under async scheduling a finished answer
then leaves only when the next forward returns: chapter 16's "a host
that waits for its GPU". On a prefill server whose steps run back to
back, it waits for the next prompt's whole prefill: r0 and r2 in Figure
20.1, 161.9 ms each. `simulate(host_sync_forward=True)` plans each step
after the previous one ends, and moves each delivery (the end of
`simulate`):

```python
    if held:
        # a step's tokens reach the client when the NEXT step's forward
        # returns, if that step started as this one ended
        ends = [e for _, e in bounds]

        def delivered(x):
            k = bisect.bisect_left(ends, x - 1e-6)
            if k + 1 < len(bounds) and abs(ends[k] - x) <= 1e-6 and bounds[k + 1][0] == ends[k]:
                return ends[k + 1]
            return x
        for r in pending:
            if r.ttft_us is not None:
                r.ttft_us = delivered(r.arrival_us + r.ttft_us) - r.arrival_us
            if r.finish_us is not None:
                r.finish_us = delivered(r.finish_us)
```

`held` is async scheduling with a host-synchronous forward.

From here the evidence is Qwen3-8B in bf16 on two RTX A6000s under vLLM
0.15.1, both designs behind one proxy (`tools/pd_proxy.py`): a server
per GPU, round-robin; or prefill on GPU 0 and decode on GPU 1. `vllm
bench serve` sent 240 requests of 1,024 random tokens in and 128 out,
Poisson at 1–8 per second, to servers of at most 64 sequences, a
2,048-token budget and top-p sampling. The model is
`test_pd_silicon_envelope`'s set-up in `tests/test_core.py`, on the
benchmark's own trace. The proxy logged each prefill answer in re-runs
at 1, 3 and 5 requests per second, with async scheduling on as deployed
and, as a control, off on the prefill server alone:

```
Table 20.4  The prefill server's answer on the proxy's clock, 1P1D at the sweep's own arrivals, ms:
  measured (as recorded) and today's model, with async scheduling on (as deployed) and off (the control)
  scheduling  req/s   measured p10/p50/p90/p99  model                  held, measured   model
  async on        1   179/185/340/484           187/187/349/535             40 of 241  35 of 240
  async on        3   181/332/668/1035          187/342/653/910            114 of 241 105 of 240
  async on        5   189/613/1203/1526         187/566/1174/1463          186 of 241 173 of 240
  async off       1   178/184/247/452           187/187/245/458              4 of 241          -
  async off       3   180/190/479/745           187/187/471/719             23 of 241          -
  async off       5   183/405/963/1254          187/378/896/1199            77 of 241          -
  model/measured over the quantiles shown: 0.88-1.11
  held, measured: two whole prefill forwards between arrival and answer (at 5 req/s also waits behind a full step);
  model: answers its rule delays, none by construction with async off; the proxy logged 241 requests, the benchmark sent 240
  written down before the control's data were read (not committed), p10/p50/p90/p99: 1 req/s 186/186/275/458; 3 req/s 186/186/477/794; 5 req/s 186/404/1005/1776
  the decode server's share, proxy to first byte, p50 at 1, 3, 5 req/s: measured 83/94/103, model 91/101/112
  end to end with async off, 1, 3, 5 req/s: measured TTFT p50 2%, 30%, 27% below the deployment's; today's model/measured 1.06, 1.03, 0.92
```

At 3 requests per second the median answer takes 332 ms, against 185 at
1, and 114 of 241 answers left after a later forward; today's model is
within 12% at every quantile shown. With async scheduling off the median
falls to 190 ms, and the median TTFT by 30%. The mechanism was found on
these runs, so the table is in-sample. With this connector, run the
prefill server without async scheduling; `prefill_host_sync` defaults to
`False`, a connector that doesn't stall its host. (On vLLM 0.15.1 the
pair also needed small connector patches, in `tools/vllm_patches/`.)

## How close is it: two GPUs, two designs

```
Table 20.5  Two GPUs, two designs: Qwen3-8B on two RTX A6000s under vLLM, 240 requests of 1024 tokens in,
  128 out, measured and model/measured
         colocated pair                                    1P1D
  req/s  TTFT p50  model  TPOT ms  model  tok/s  model    TTFT p50  model  TPOT ms  model  tok/s  model
      1     219.3   1.03    26.63   1.02    126   1.00       272.2   1.05    25.81   1.01    126   1.00
      2     224.8   1.04    29.55   1.02    249   1.00       291.8   1.04    27.34   1.00    248   1.00
      3     229.4   1.03    33.43   1.01    368   1.00       423.6   1.04    28.93   1.00    366   1.00
      4     234.1   1.05    38.57   1.01    483   1.00       525.1   0.95    30.82   0.99    478   1.00
      5     244.2   1.03    45.72   1.00    592   1.00       719.1   0.96    32.60   0.99    584   1.00
      6     265.5   1.03    55.68   1.00    695   1.00      1167.5   0.88    34.17   1.01    682   1.01
      7     311.3   1.03    69.75   1.01    789   1.00      2635.6   0.82    34.80   1.02    732   1.03
      8     459.5   0.99    92.89   1.02    865   1.00      4570.9   0.89    34.88   1.02    740   1.03
  colocated: TTFT p50 0.99-1.05, p95 0.99-1.07; TPOT 1.00-1.02, p95 1.00-1.02; tok/s 1.00-1.00
  1P1D: TTFT p50 0.82-1.05, p95 0.79-1.03; TPOT 0.99-1.02, p95 0.99-1.03; tok/s 1.00-1.03
  1P1D TTFT p50: 0.95-1.05 at 1-5 req/s, 0.82-0.89 at 6-8, where its prefill server saturates
  the lower TTFT p50 and TPOT at each rate, measured and model: the same design in 16 of 16
  the pair's throughput lead, measured: under 0.5% at 1-3 req/s (both serve the offered load: a tie), 1.0-16.9% at 4-8; the model's: 0.9-13.8% at 4-8
  measured TPOT p95 / mean: pair 1.10-1.29, 1P1D 1.02-1.08
  today's model; this sweep has tested every later change
```

![Median TTFT and mean TPOT against load for the colocated pair and
1P1D, model and measured.](/assets/tinyperf-book/ch20-load.svg)

*Figure 20.2. The two designs against load: the model (dashed) and vLLM
(solid lines through the measured points), same trace.*

The pair wins TTFT at every rate: two GPUs can prefill, and nothing
crosses a second server. Throughput is a tie below 4 requests per
second, where both serve the offered load; from 6 the lone prefill GPU
saturates, and 1P1D stops at 740 tokens per second against 865. 1P1D
wins TPOT at every rate, 34.9 ms against 92.9 at 8 per second, and its
p95 TPOT stays within 8% of its mean where the pair's runs 10–29% above.

Table 20.5 is today's model, and this sweep has tested every later
change to it, so it is in-sample. Held out are the numbers committed
before the runs, each the mean over eight other Poisson samples:

```
Recorded  Predictions committed before the runs (held out), each the mean over 8 Poisson samples, against the measurements
  colocated: TTFT p50 0.76-1.02, TPOT 0.81-1.00, tok/s 0.98-1.01; 1P1D: TTFT p50 0.71-1.04, TPOT 0.98-1.00, tok/s 0.98-1.00
  the lower TTFT p50 and TPOT at each rate: the design measured in 16 of 16; tok/s tied at 1 and 2 req/s, the pair ahead above
```

They read 1P1D's median TTFT as low as 0.71: the held answers and the
producer's save were both found on this sweep afterwards.

**Which design wins depends on the SLO.** No single metric changes
winner between 1 and 8 requests per second, but an SLO combines two.
Table 20.6 applies limits to the run as a whole (its p95 TTFT and mean
TPOT), not chapter 14's per-request SLO. We chose these limits after the
runs. They re-read Table 20.5 rather than test anything new: the
committed predictions agree in 16 of 20, today's model (in-sample on
1P1D TTFT) in 19.

```
Table 20.6  Which design sustains more load under run-level limits chosen after the runs: the highest rate
  at which it and every lower rate meet TTFT p95 and mean TPOT limits, pair : 1P1D, measured / today's model
  TTFT p95      TPOT 30 ms    TPOT 35 ms    TPOT 45 ms    TPOT 60 ms    TPOT 75 ms
  0.5 s          2:1 / 2:1     3:1 / 3:1     4:1 / 4:1     5:1 / 5:1     5:1 / 5:1
  1 s            2:3 / 2:3     3:3 / 3:3     4:3 / 4:3     6:3 / 6:3     7:3 / 7:3
  2 s            2:3 / 2:3     3:5 / 3:6     4:5 / 4:6     6:5 / 6:6     7:5 / 7:6
  4 s            2:3 / 2:3     3:7 / 3:6     4:7 / 4:7     6:7 / 6:7     7:7 / 7:7
  today's model names the measured winner, or the tie, in 19 of 20; the miss: TTFT 2 s, TPOT 60 ms (measured pair, model tie)
  the predictions committed before the runs: 16 of 20
```

A tight TTFT limit favours the pair at any TPOT limit; a tight TPOT
limit favours 1P1D once the TTFT limit allows its extra 60 ms and its
queue. Under TTFT p95 2 s and TPOT 35 ms, the pair sustains 3 requests
per second and 1P1D 5.

## Held out: the KV producer on new traces

The save was checked on two sweeps whose predictions were committed
before any ran: the main sweep's shape on a new seed, and varied
lengths:

```
Table 20.7  The KV producer, held out: 1P1D on two new traces, the predictions committed before the runs
  over the measurements, and today's model's TTFT p50 over the measurement
  trace                    req/s  TTFT p50 ms  TTFT p50  TTFT p95   TPOT  tok/s  criteria  today
  1024/128, seed 1             1          272      1.06      1.01   1.01   1.00    within   1.06
                               3          440      1.01      0.99   0.99   1.00    within   1.01
                               5          727      0.96      0.97   0.99   1.00    within   0.96
                               6         1131      0.92      0.95   1.01   1.00    within   0.92
                               7         2562      0.78      0.77   1.02   1.03      miss   0.78
  768/128 +-90%, seed 2        3          325      0.99      1.00   0.99   1.00    within   0.99
                               6          582      0.94      1.00   0.99   1.00    within   0.94
                               8         1003      0.96      0.97   1.03   1.00    within   0.94
                              10         3377      0.97      0.95   1.04   1.01    within   0.93
  768/128 +-90%: prompts 80-1460 tokens, replies 12-243, up to 6 prompts in a prefill step (the fit saw 4)
  committed criteria: (1) TTFT p50 and p95 0.85-1.15, TPOT 0.95-1.05, or (2) at a knee (1% on every step moves
  TTFT p50 over 15%) the measured p50 in that band; (3) tok/s within 3%; (4) prefill steps at their logged
  composition 0.95-1.05 for each prompt count with 10+ steps (recorded: 0.95-1.00)
  8 of 9 cells within; the miss, 1024/128 at 7 req/s, was named in advance as the prefill server's knee
  excluded: a first run on a shared decode GPU (329 and 92 forwards over 50 ms; the re-run 0 and 0): 1 req/s TPOT 0.95 (within), 3 req/s TPOT 0.93 (miss)
  today's model charges the save once per prompt, read off the varied sweep's step logs; without the save: TTFT p50 0.55-1.02
```

Eight of nine cells meet the committed criteria. The miss, 0.78 at the
knee of the 1,024-token shape, was named in advance by the committed
file
(`predictions_qwen3_8b_rtx_a6000_kv_producer_before_measurement.txt` in
`data/validation`): under sustained load, steps carrying two such
prompts had run 2–3% above the fitted cost, and near a knee a few
percent per step is a large error in TTFT (chapter 18). An excluded
first run, whose decode GPU another process shared, had missed on TPOT
at 3 req/s.

## More GPUs: choosing the split

Two GPUs force a 1:1 split, and for 1,024-token prompts with 128-token
replies it starves the prompts: the prefill GPU saturated while the
decode GPU ran batches of about 26 (740 tokens per second × 34.9 ms).
With more GPUs the ratio is a choice:

```
Table 20.8  Choosing the split, a projection: four GPUs like the measured pair, 480 Poisson requests
  workload                 design       TTFT p50     p95  TPOT ms    p95  tok/s
  1024 in, 128 out, 12/s   4 colocated       262     371     56.8   68.7   1410
                           1P3D            14876   30216     27.3   27.4    821
                           2P2D              596    1155     34.9   36.8   1398
                           3P1D             5788   15470     56.8   59.6   1015
  256 in, 512 out, 8/s     4 colocated       151     174     35.8   38.9   3284
                           1P3D              220     296     36.2   38.9   3239
                           2P2D             5867   16055     39.0   40.7   2637
                           3P1D            37122   87049     39.9   40.6   1482
  a well-behaved connector (no host sync); the pair's calibration, link and producer cost; nothing measured
```

With long prompts, 2P2D ships the colocated fleet's tokens within 1% at
0.61 of its TPOT and about half its p95 TPOT, for 2.3 times the median
TTFT; either wrong split loses a quarter to a half of the throughput.
With short prompts and long replies there is little prefill to
interfere, and the colocated fleet wins or ties every column.
Disaggregation pays where prompt steps dominate the decodes' wait, and
only at the right ratio.

## Where it breaks

- **One node, two GPUs, one connector.** Every measurement is Qwen3-8B
  on a PCIe pair under one vLLM version's NCCL connector, to which the
  save and the held answers belong. Nothing beyond 1P1D, over NVLink or
  InfiniBand, or with tensor parallelism in a pool is measured.
- **The transfer is priced as a bulk copy,** four times low, and hides
  only because each layer's send keeps pace with its compute.
- **The device file's link.** `Device.load("rtx_a6000")` has no
  point-to-point rate and a 56 GB/s NVLink field, so it prices this
  pair's transfer at 2.7 ms, not 23.1; pass the measured link.
- **The save is fitted end to end,** idle, at one to four prompts per
  step; under load the prefill steps read up to 5% low, and the prefill
  server's knee 0.78.
- **Replicated caches.** `kv_transfer_us` divides the one-GPU cache by
  tp, but a latent cache is replicated on every rank, and so are KV
  heads when ranks outnumber them: those transfers are priced up to tp
  times low.
- **Routing.** Round-robin only: no load-aware router, no pool resized
  to its load.

## What you built

- A disaggregated deployment from two copies of the serving simulator,
  a KV transfer hidden behind the prefill, a proxy paid twice, and a
  one-token first step on the decode server.
- The KV producer's step: a flat save per prompt per layer, fitted end
  to end, and a host-synchronous forward that holds answers for a step.
- Which design wins as a question of the SLO, and the split as a
  decision the model prices.

## Exercises

1. **Price the control messages.** Give `kv_transfer_us` a cost per
   layer (0.58 ms) and the measured send rate, and find the prompt
   length below which the transfer stops hiding behind a prefill on the
   pair.
2. **Fix the connector.** Move the gather's block indices to the GPU.
   Predict with the model what happens to Table 20.4 and to 1P1D's TTFT
   at 3–5 requests per second, then measure it.
3. **The SLO frontier.** For eight GPUs and your own workload, find the
   P:D ratio that maximizes goodput (chapter 14) when each request must
   see TTFT ≤ 1 s and TPOT ≤ 40 ms. How does it move as prompts
   lengthen?
4. **tp in a pool.** Serve Llama-3-70B as 1P1D with tp=4 per pool on
   H100s over InfiniBand, pricing the transfer by the shard each rank
   actually holds. When does the transfer stop hiding?

---

*[← Chapter 19: Speculative decoding]({% post_url 2026-09-29-tinyperf-19-speculative-decoding %}) · [Contents](/series/tinyperf/) · [Chapter 21: Validating a performance model →]({% post_url 2026-09-29-tinyperf-21-validating-a-performance-model %})*
