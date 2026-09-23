---
layout: post
math: true
title: "Building tinyperf M66: Disaggregation on one node: the frozen model picks the winner, and the prefill server holds its answers"
date: 2026-09-22 15:00:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "One prefill GPU and one decode GPU against two colocated servers, on the same two GPUs, with predictions frozen first: the model named the winning deployment in all 24 comparisons and decode landed within 2%. The one miss, a 27% TTFT gap, came from vLLM's P2P NCCL connector blocking the prefill server's engine thread, so under async scheduling an answer waits for the next step whenever one follows; turning async scheduling off there cut median TTFT by up to 30%."
---

*Milestone 66 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `tinyperf/disagg.py`, `simulate(host_sync_forward=...)` in `serving.py`, `tools/pd_proxy.py`, `tools/pd_timeline.py` · Data: `data/validation/comparison_qwen3_8b_rtx_a6000_pd.txt`, `pd_timeline_qwen3_8b_rtx_a6000.json`.*

Milestone 37 priced disaggregated serving: a prefill pool computes
prompts, ships each request's KV cache to a decode pool, and decode never
sees a prompt. It was never measured, and it could not have been priced
on this box, because it shipped KV only over an inter-node fabric. This
box has two GPUs and a PCIe host bridge between them.

That is still a real deployment question. Take two GPUs. Should they run
two ordinary servers, or one prefill server and one decode server? At
what load, and for which service-level objective?

## The model, rebuilt

Milestone 37's pools were simplified engines written before milestones
54–57 measured what a serving engine does: the 25 ms online overhead, the
async admission delay, the cost of a step that carries prefill, and
CUDA-graph padding. Rather than port those into a second engine, the
pools are now the colocated engine itself:

- **Prefill server.** It is an engine whose requests stop after one
  token, which is exactly what a disaggregation proxy asks of it
  (`max_tokens=1`).
- **Decode server.** It is an engine whose requests arrive with all but
  their last prompt token's KV already loaded (`Request.kv_loaded`). Its
  first step computes that one token.
- **Transfer.** The KV crosses a link — the fabric, or now the node-local
  GPU-to-GPU path (`kv_link="local"`, at the 6.55 GB/s this pair measured
  in milestone 45). It is sent layer by layer during the prefill, so only
  the part that outlasts the prefill shows.
- **Proxy.** Every request pays the proxy's hop twice: once to the
  prefill server, and again on the way to the decode server.

The colocated baseline is `simulate_replicas`: two engines behind the
same round-robin proxy.

## Making vLLM do it

vLLM 0.15.1 ships a point-to-point NCCL connector for exactly this. A
proxy sends each request to the prefill server with `max_tokens=1` and a
request id naming both servers' KV endpoints. The prefill server pushes
the KV layer by layer. The proxy then sends the request to the decode
server and streams its answer back. `tools/pd_proxy.py` does this in
about 150 lines, and also round-robins two plain servers, so both
deployments pay the same hop (1.86 ms, measured).

The first request hung. vLLM 0.15.1 appends a random 8-hex suffix to
every engine request id, and the connector keys every transferred layer by
the suffixed id. The two servers draw different suffixes for the same
request, so the decode server waits forever for KV filed under a name
that never arrives. `tools/vllm_patches/p2p_nccl_request_id` strips the
suffix where the connector builds or releases a tensor id. After that,
2000-token prompts came back in 1.10 s. That matches prefill, plus
transfer, plus decode, with no recompute on the decode side.

## Frozen, then swept

The predictions were committed before either sweep ran. The workload is
Qwen3-8B, 1024-token prompts, 128 output tokens, Poisson arrivals, 240
requests per rate, 1–8 req/s — the milestone-54 sweep, now on two GPUs.
Measured values, with the frozen predictions in brackets:

```
                  colocated pair (round-robin)                    1P1D (prefill GPU + decode GPU)
  rate    TTFT p50 ms    TPOT ms        out tok/s        TTFT p50 ms      TPOT ms        out tok/s
    1     219 (223)      26.6 (26.2)    126 (124)        272 (284)        25.8 (25.7)    126 (124)
    3     229 (233)      33.4 (33.2)    368 (363)        424 (307)        28.9 (28.9)    366 (361)
    5     244 (246)      45.7 (43.9)    592 (585)        719 (524)        32.6 (32.0)    584 (580)
    8     460 (350)      92.9 (74.8)    865 (876)       4571 (3508)       34.9 (34.3)    740 (740)
```

**The winner is called everywhere.** On TTFT, TPOT and throughput, at
every rate, the frozen model named the winning deployment in all 24
comparisons. The colocated pair wins TTFT and throughput: it has two
GPUs that can prefill, and no second hop. 1P1D wins TPOT, because its
decode server never runs a prompt.

**1P1D decode is on the model everywhere.** TPOT is within 0.98–1.00 at
every rate, and saturation throughput is 740 tok/s both measured and
predicted. At 8 req/s the lone prefill GPU is the bottleneck, and the
model knew its rate.

**The colocated pair is on the model to 5 req/s.** Above that it
reproduces the single-GPU knee gap milestone 54 documented (TPOT 0.81 at
8 req/s, i.e. 4 per GPU). The pair also shows a mechanism the model
predicted and silicon confirms: round-robin splits a Poisson stream into
less bursty halves. At 3.5 req/s per GPU the pair holds a 311 ms median
TTFT; a single GPU at 3.5 req/s tipped over in milestone 57, at 4.6 s.

**The crossover lives in the SLO, not the rate.** No single metric
changes winner between 1 and 8 req/s, but an SLO combines two metrics,
and which deployment sustains more load depends on it. Take a TTFT p95
limit of 2 s and a TPOT limit of 35 ms. The pair meets it up to 3 req/s;
1P1D meets it up to 5. With a TTFT limit of 500 ms, the pair wins at
every TPOT limit. Across a grid of 20 SLOs (TTFT p95 0.5–4 s × TPOT
30–75 ms), the frozen model picked the deployment that sustains more load
in 17. The three misses each sit one rate step from a tie.

**One number is wrong.** 1P1D TTFT is on the model at 1–2 req/s. From
3 req/s its median is 0.71–0.77 of measured — a flat 27% miss that the
frozen model has no mechanism for.

## Where the time goes

`tools/pd_proxy.py --timing-log` records, for each request, when it
reached the proxy, when the prefill server answered, and when the decode
server's first byte came back. At 3 req/s, in ms:

```
                         p10   p25   p50   p75   p90   p95   p99
  prefill answer         181   186   332   485   668   764  1035
  decode first byte       80    88    94   103   112   116   124
```

The decode server's share is on the model (94 ms measured, 101 ms
modelled). The whole miss is the prefill server. Its answer takes 332 ms
at the median; the model says 193. The server's own metrics say the
requests barely queue (11 ms), yet the time from scheduling to first
token grows from 194 ms at 1 req/s to 288 ms at 3.

A second sitecustomize, `tools/vllm_patches/p2p_nccl_timing`,
timestamps the connector on the same clock: every layer's send, every
receive, every forward's start and end. Here is the prefill server at
3 req/s:

```
    50.0   request A arrives
    71.3   forward starts (A)          36 layer sends, 73.7 -> 230.3
   202.9   request B arrives
   227.4   forward ends
   229.7   forward starts (B)          36 layer sends, 233.7 -> 389.1
   386.8   forward ends
   390.6   A's answer leaves  (341 ms)
   394.6   B's answer leaves  (192 ms)
```

A's prefill finished at 227 ms, but its answer left at 391 ms, after B's
forward. The answer waited a full step for another request's prompt.

The cause is in the connector. On the prefill server it gathers each
layer's KV with a CPU index tensor (`torch.tensor(block_ids)`). A CUDA
tensor indexed by a CPU tensor copies the index to the device from
pageable memory, and that copy synchronizes the stream. So the host waits
for every layer: the forward holds the engine thread for its whole GPU
time, 157 ms here. Under async scheduling the engine launches step k+1
before it processes step k's output. With a forward that blocks, that
means step k's answers leave only when step k+1's forward returns. This
happens whenever a step follows back to back.

In the model this is `simulate(host_sync_forward=True)`. An arrival
during step k joins step k+1, because the plan is made after k's forward
returns, and step k's tokens are delivered when step k+1 ends. The
mechanism's frequency can be checked directly: the fraction of answers
that left after a later forward is 17% and 47% on silicon at 1 and
3 req/s, and 16% and 45% in the model.

## The control

If the explanation is right, turning async scheduling off on the prefill
server alone should release every answer when its own step ends. Before
reading the control's data, the model predicted the prefill answer times
(`simulate(async_scheduling=False)`):

```
  3 req/s, prefill answer    p10   p25   p50   p75   p90   p95   p99
  predicted                  186   186   186   339   477   582   794
  measured (async off)       180   184   190   338   479   569   745
```

Held answers fell from 40 to 4 of 241 at 1 req/s, and from 114 to 23 at
3 req/s. The server's scheduled-to-first-token time at 3 req/s fell from
288 ms to 199. End to end:

- **TTFT median** fell 30% at 3 req/s (424 → 296 ms) and 27% at 5
  (719 → 526 ms).
- **TTFT p95** fell 23% and 20%.
- **TPOT** was unchanged to 0.03 ms.

On this connector, a prefill server should run with
`--no-async-scheduling`.

## The transfer, priced wrong and hidden

The same clock times the KV transfer: 2.4–2.6 ms per layer (0.6 ms
ZeroMQ handshake, then 1.9 ms of `ncclSend` and a stream sync). That is
about 89 ms per 1024-token request, against the 23 ms priced from the
link's bandwidth.

It does not matter here. The sends pace with the forward's layers (one
layer computes in ~4.3 ms), so they finish about 2 ms after the prefill.
The prefill step's drain wait is 0.03 ms at the median, and the decode
server's in-step KV wait is 0.01 ms. At a short prompt it would matter:
a 128-token prefill takes ~28 ms, and 36 handshakes do not fit behind
it. That regime is unmeasured, and ENVELOPE.md says so.

## What changed, and what did not

- **Mechanisms.** `host_sync_forward` is new in the engine.
  `simulate_disagg` gains `prefill_host_sync` (True prices this
  connector) and `prefill_async_scheduling` (prices the control). Both
  default off, so a well-behaved connector is the default.
- **Post-hoc fit.** With the mechanism on, 1P1D TTFT median is
  1.01–1.08 at 1–5 req/s and p95 1.06–1.17. Across the SLO grid it now
  picks the winner in 19 of 20. This is a post-hoc fit and is labelled
  as one; the frozen file is unchanged.
- **Constants.** None were fitted to a sweep.
- **Milestone 37.** Its regime map no longer reproduces. The colocated
  rows moved with the milestone 54–57 engine, and the rebuilt pools moved
  the disaggregated ones. The regimes, the jitter collapse and the cost
  of the wrong split stand, but the size of disaggregation's win at
  saturation shrank (an erratum is on the post).

## Exercises

1. Price the transfer from the connector's per-layer cost (a handshake
   plus bytes at the measured rate), then find the prompt length below
   which it stops hiding behind the prefill. Measure it with
   `tools/pd_timeline.py`.
2. The model's rate-5 prefill tail is heavier than this trace's (p95
   1297 vs 1102 ms with async off). One trace or a mechanism? Re-run
   the control on three arrival seeds.
3. Fix the connector instead of the flag: index with a device tensor,
   and order the send after the gather with an event. Predict what
   happens to the held fraction, then measure it.
