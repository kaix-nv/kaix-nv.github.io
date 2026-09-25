---
layout: post
math: true
title: "Building tinyperf M71: The KV producer's step: the connector's per-layer save, on the engine's clock"
date: 2026-09-24 18:00:00 -0700
categories: [tinyperf, perf-modeling]
excerpt: "Milestone 70 left a disaggregated pair's prefill server off the model. On the engine's own clock, its forward carries vLLM's KV connector saving every prompt at every layer, a host round trip per prompt per layer, priced as two constants. Frozen and measured on two new traces, 8 of 9 cells land within the stated criteria, the miss being the knee the frozen file named. Along the way, a connector assertion that kills the prefill server, and a shared GPU caught by the step clock."
---

*Milestone 71 of [building an analytical GPU performance model from
scratch](/series/tinyperf/). Code:
[`tinyperf`](https://github.com/kaix-nv/tinyperf) — `kv_producer_layer_us` in `methodology.py`, `simulate(kv_producer=)` in `serving.py`, `tools/vllm_patches/pd_step_timing`, `tools/measure_prefill_step.py` · Data: `data/calibration/kv_producer_rtx_a6000.json`, `data/validation/comparison_qwen3_8b_rtx_a6000_kv_producer.txt`, `engine_steps_m71_qwen3_8b_rtx_a6000.json`.*

Milestone 70 left one sweep off the model. With a disaggregated pair
(one prefill server, one decode server), TTFT above 3 req/s read
0.58–0.87 of measured. Its prefill-server steps had been priced 8% high,
and removing that overcharge exposed a cost the model did not have: the
KV connector's work on the server that produces the cache. This
milestone puts the engine's step clock on that server.

## The prefill server, on its own clock

Python runs only one `sitecustomize`, so `tools/vllm_patches/pd_step_timing`
runs two patches in separate namespaces. One times every engine step on
the GPU; the other pairs the servers and timestamps the connector. The
probe needed one more thing. From idle, the engine schedules the first of
several simultaneous prompts alone. `tools/measure_prefill_step.py
--blocker` therefore sends a 2048-token prompt first, so that the k test
prompts arrive while the engine is busy and share the next step.

I ran the same probe twice on the same GPU: once on a plain server and
once through the pair. Forward time, start to sampler, median of 7:

```
  prompts   prompt     plain    producer   extra
  per step  length      (ms)      (ms)      (ms)
     1        256       51.2      60.4       9.3
     1       1024      151.3     160.6       9.3
     1       2048      289.8     298.4       8.6
     2        256       78.3      92.8      14.5
     2       1024      283.0     298.1      15.1
     4        256      149.0     169.8      20.8
     4        512      280.4     297.7      17.2
```

The extra cost is flat in prompt length and grows with the number of
prompts in the step, each one costing less than the one before.

## Why per prompt

vLLM's P2pNcclConnector saves KV from inside the forward. At every layer,
for each prompt it gathers the layer's KV blocks and queues the copy for
a send thread. The gather is indexed by a tensor that lives on the CPU,
and copying that index to the GPU blocks the host until the layer's work
is done. So every prompt costs a host round trip per layer:

- **The first prompt in a layer** waits for the layer, and the GPU idles
  until the host relaunches.
- **Each further prompt** finds the GPU already idle, and pays only the
  round trip.

The sends themselves are not the main cost. They run on another thread,
paced with the forward. In a 2048-token step the sender is busy 44% of
the time with one prompt and 92% with four. But with one prompt per step
the extra cost stays flat from 256 to 2048 tokens, while NCCL's send time
grows eightfold.

Two constants price this, fitted on the probe: 269 µs per layer for a
step's first prompt and 94 µs for each further one. Over 36 layers that
is 9.7 ms for one prompt, 13.1 for two and 19.8 for four. The fit is
within 2.4 ms of every cell of 256 tokens and more. One cell isn't
explained: a single 128-token prompt reads 0.3 ms.

## Under load

Two diagnostic sweeps ran on the pair with both step clocks. The first
re-ran M66's own trace, and it reproduces M66's measurement to 3% on TTFT
and exactly on TPOT. The second used 512-token prompts, so a step carries
up to four. (The first sweep's GPU telemetry was lost: a stale watcher of
mine launched the run a second time, and I stopped it before its servers
came up.)

Priced as a plain engine, the prefill server's steps read 0.85–0.95 of
measured. With the save, they read 0.95–1.01. One case runs above the fit
under load: two 1024-token prompts cost 20–22 ms, against 13 fitted and
15 idle, with the GPU at its power cap. On the re-run of M66's trace,
TTFT reads 0.94–1.06 up to 5 req/s and 0.80 at 7, where the prefill
server saturates.

## Frozen, then measured

The predictions were pushed before any held-out run. There were two
sweeps on new seeds:

- **H1:** M66's shape (1024-token prompts, 128-token outputs) at
  1–7 req/s.
- **H2:** varied lengths (prompts 80–1460, outputs 12–243) at 3–10 req/s.
  Up to six prompts share a step here, more than the fit ever saw.

The criteria were the same as M70's. The frozen file also named one
expected miss: at the prefill server's knee on the 1024 shape the model
would read low, because of that under-load residual.

Two things went wrong on the way:

- **The prefill server died.** At 6 req/s in H2, vLLM's connector
  asserted that a prompt resuming a chunked prefill brings new KV blocks.
  A resumed chunk that fits in the prompt's last block brings none, and
  varied lengths produce exactly that. `p2p_nccl_request_id` now hands
  the connector an empty block list there, and H2 was run again.
- **GPU 1 was shared.** In H1's first run at 1 and 3 req/s, the decode
  server's forwards ran 2–2.5× normal in bursts: 329 and 92 steps over
  50 ms, with GPU 1's clocks steady. That is another process on the GPU,
  not a slow one. The re-run logged every GPU process every 2 seconds;
  only the pair's two engines appear, and there are no slow forwards.
  The first run's cells are kept but excluded.

```
  predicted / measured                  TTFT p50    TTFT p95    TPOT
  H1, M66's shape, 1-6 req/s            0.92-1.06   0.95-1.01   0.99-1.01
  H1 at 7 req/s (the named knee)        0.78        0.77        1.02
  H2, varied lengths, 3-10 req/s        0.94-0.99   0.95-1.00   0.99-1.04
  M70, the same cells (1-6 and H2)      0.74-1.02
```

Throughput is within 1.00–1.03. Eight of nine cells meet the criteria;
the miss is the one the file predicted. The prefill server's steps, at
the composition the engine recorded, read 0.95–1.00 for every prompt
count with 10 or more steps; the few with five or six prompts, beyond
the fit's four, read 0.97–0.98. The decode server's steps read 0.98–1.02.

## What the run taught about the connector

The connector saves a prompt split across steps only once, with its last
chunk. It sends 36 layers per prompt, so its send count gives the number
saved per step. At 8 req/s, 30 steps that carried three chunks saved only
two prompts. The model had charged every chunk. I corrected that after
the freeze, and report both. H2's TTFT p50 moves from the frozen
0.94–0.99 to 0.93–0.99; H1's prompts never split, so it is unchanged.

## Errata

- **M70:** its 1P1D residual above 3 req/s is the KV producer's
  per-layer save. M66's sweep now reads TTFT 0.95–1.05 through 5 req/s.

## Exercises

1. Move the connector's block indices to the GPU. Does the per-prompt
   host sync, and with it most of this milestone's cost, vanish?
2. Two 1024-token prompts cost 20–22 ms under load and 15 idle. Is it the
   power cap? Pin the clocks, or time the same step at several fixed
   clocks.
3. A single 128-token prompt pays 0.3 ms instead of 9. What is different
   about a step that short?
