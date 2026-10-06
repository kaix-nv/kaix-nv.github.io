---
layout: post
title: "Building tinyperf, appendix A: Using the estimator"
date: 2026-09-29 12:23:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/appendix-a-using-the-estimator/
excerpt: "The chapters built a model that prices a step (predicts the time of one forward pass over a batch) and a simulator that replays a server under load. Planning asks which configuration, how many GPUs, and what a million tokens costs in dollars and joules, at a latency target. How do you answer that with the model, and how far can you trust the answers?"
redirect_from:
  - /tinyperf/perf-modeling/2026/08/22/building-tinyperf-m14.html
  - /tinyperf/perf-modeling/2026/08/23/building-tinyperf-m15.html
  - /tinyperf/perf-modeling/2026/08/28/building-tinyperf-m16.html
  - /tinyperf/perf-modeling/2026/08/30/building-tinyperf-m29.html
  - /tinyperf/perf-modeling/2026/09/08/building-tinyperf-m52.html
---

*[Building tinyperf](/series/tinyperf/) · Appendices · Code: [`tinyperf/sweep.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/sweep.py), `pareto`, `find_knee` and `tip_fraction`; [`tinyperf/serving.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/serving.py), `prediction_interval`, and at a [later commit](https://github.com/kaix-nv/tinyperf/blob/4ae68ec/tinyperf/serving.py) `VLLM` and `recorded_trace`; [`tinyperf/capacity.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/capacity.py), `vllm_kv_pool`; [`tinyperf/energy.py`](https://github.com/kaix-nv/tinyperf/blob/8b7ae99/tinyperf/energy.py), `energy_report` · Every table and both figures in this appendix come from `python3 book/scripts/appA_using.py`.*

The chapters built a model that prices a step (predicts the time of
one forward pass over a batch) and a simulator that replays a server
under load. Planning asks which configuration, how many GPUs, and what
a million tokens costs in dollars and joules, at a latency target. How
do you answer that with the model, and how far can you trust the
answers?

The short answer: price every configuration and keep the ones nothing
beats. Simulate one replica across load, and plan below its knee (the
load where latency turns sharply upward) with margins for the model's
error, the traffic and steady load. Divide GPU-hours by tokens: as the
latency target approaches the time of a one-sequence decode step (the
*wall*), the cost grows as `1 / (1 − wall/target)`. Energy per token
follows the same GPU time. Only the plan on the RTX A6000 rests on
held-out runs, which nothing in the model was fitted to or built
around; other GPUs, prices, energy and speculative decoding (a cheap
draft model proposes tokens that the served model checks in one step;
chapter 19) are projections, unchecked by any measured server.

By the end of this appendix you will know:

- how to sweep configurations and read a Pareto front (the
  configurations no other beats on both metrics);
- how to size a deployment: what caps a replica, how far below its knee
  to plan, how many replicas to buy;
- why tight latency targets cost disproportionately;
- what the power model charges, and where it counts twice;
- when speculative decoding lowers the cost per token;
- which answers rest on held-out runs.

## Sweeps: price everything, keep the front

A sweep lists configurations (`expand`, a Cartesian product), prices
each (`run_sweep`), and keeps the ones no other beats (docstring
trimmed):

```python
def pareto(results: list, x: str, y: str) -> list:
    feasible = sorted((r for r in results if r.get("fits")), key=lambda r: (r[x], -r[y]))
    frontier, best_y = [], float("-inf")
    for r in feasible:
        if r[y] > best_y:
            frontier.append(r)
            best_y = r[y]
    return frontier
```

On this *Pareto front*, sorted by x (smaller is better), a point joins
only if its y (larger is better) beats every point before it: improving
one metric costs the other. The price function is yours; the script's
uses the step model of chapter 14's engine preset (vLLM 0.15.1's
settings in one object), `VLLM.step_model`, which prices kernels
launched from a CUDA graph (recorded once and replayed as one;
chapter 4) and takes a precision recipe, the dtype of each kind of
operator (chapter 10's 8-bit floating-point recipe ships as
`RECIPE_FP8_SERVING`):

```python
def price(cfg):
    recipe = RECIPE_FP8_SERVING if cfg["precision"] == "fp8" else None
    mem = llm_memory(LLAMA, cfg["gpu"], batch=cfg["batch"], context_len=SWEEP_CTX,
                     tp=cfg["tp"], pp=cfg["pp"], recipe=recipe)
    if not mem.fits:
        return {"fits": False}
    tpot_ms = step_model(cfg["gpu"], cfg["tp"], cfg["pp"], cfg["precision"]).decode_us(cfg["batch"], SWEEP_CTX) / 1e3
    world = cfg["tp"] * cfg["pp"]
    return {"fits": True, "tpot_ms": tpot_ms, "per_user": 1e3 / tpot_ms,
            "per_gpu": cfg["batch"] * 1e3 / tpot_ms / world, "world": world}
```

What exceeds 90% of memory (chapter 6's rule) is not priced. The rest,
priced at the calibrated tier (on constants fitted to each GPU;
chapter 4), get *tokens per second per user*, 1000 / TPOT (time per
output token) in ms, and *per GPU*, what the owner pays for. Each price
comes from a step graph, the list of operators one step runs
(chapter 5). Table A.1 sweeps Llama-3-70B's decode at 4,096 tokens of
context: tensor parallelism tp = 1–8 (each GPU holds 1/tp of every
weight matrix; chapter 11), pipeline stages pp = 1–2 (the layers cut
into pp blocks of consecutive layers, each on its own GPUs; chapter 12),
bf16 or chapter 10's FP8 recipe, batch 1–256.

```
Table A.1  A decode sweep's Pareto front: Llama-3-70B on H100 SXMs, context 4096, calibrated tier
  layout         batch  TPOT ms  tok/s per user  tok/s per GPU
  tp8 pp1 fp8        1     9.41           106.2           13.3
  tp8 pp1 fp8        2     9.46           105.7           26.4
  tp8 pp1 fp8        4     9.56           104.6           52.3
  tp8 pp1 fp8        8     9.76           102.5          102.5
  tp8 pp1 fp8       16    10.15            98.5          197.0
  tp8 pp1 fp8       32    10.95            91.3          365.3
  tp8 pp1 fp8       64    12.54            79.7          637.9
  tp8 pp1 fp8      128    15.72            63.6         1017.5
  tp4 pp1 fp8      128    21.77            45.9         1469.9
  tp4 pp1 fp8      256    33.08            30.2         1934.9
  288 configurations (H100, B200 x tp 1-8 x pp 1-2 x bf16, fp8 x batch 1-256); 48 exceed 90% of memory; 240 priced from 372 step graphs
  H100, at least 50 tok/s per user: bf16 at most 446 tok/s per GPU (tp8 pp1 bf16, batch 64); fp8 1018 (tp8 pp1 fp8, batch 128)
  projection: GEMM constants fitted in fp16 on each GPU (chapter 4), FP8 unmeasured
```

![Three Pareto fronts, tokens per second per GPU (log scale) against
tokens per second per user: the B200's highest, then the H100's, then
the H100's in bf16 alone.](/assets/tinyperf-book/appA-front.svg)

*Figure A.1. Pareto fronts from Table A.1's sweep, from throughput at
the left to interactivity at the right; dashed, the H100's best in bf16
alone.*

- **Batch is the exchange rate.** From batch 1 to 128 at tp8 the step
  grows from 9.41 to 15.72 ms, each GPU's throughput from 13.3 to 1,017.5
  tokens per second: the batch shares the weight read (chapter 6).
- **The layout depends on the target.** Eight GPUs give the most tokens
  per user, fewer the most per GPU: at batch 128, going from 8 to 4
  lengthens the step only from 15.72 to 21.77 ms.
- **FP8 is on every point**, with constants fitted on fp16 GEMMs: at 50
  tokens per user or more, bf16's best is 446 tokens per GPU, FP8's
  1,018. And no pipeline: a stage doesn't shorten a weight-bound decode
  step (chapter 12).

## How many GPUs

A deployment is sized in requests, which takes the serving simulator
(chapter 14). From chapter 18: the *knee* is the load at which TTFT
(time to first token) turns sharply upward; ρ is the share of the
GPU's time spent on requests' own work rather than the weight read
every step shares; and the *envelope*, a steady-state model of the
server from mean values, puts the knee where requests in flight fill
the engine's *seats* (`max_num_seqs`, the most requests it runs at
once; 64 here).

**What caps a replica.** The seats, or the KV pool (the cache space, in
tokens) vLLM sizes at start-up (chapter 17):

```
Worked example  What caps one replica: Qwen3-8B on one RTX A6000, vLLM's defaults with 64 seats
  KV pool (vllm_kv_pool, chapter 17): 196,704 tokens; per seat: 3,073 tokens
  1024 in, 128 out: 1,152 tokens; 64 fill 37% of the pool: the seats bind
  4096 in, 512 out: 4,608 tokens; the pool holds 42: the pool binds
  envelope's knee, req/s (chapter 18): 1024/128 4.01 (rho 0.80); 4096/512 0.70 (rho 0.86), with 42 seats 0.65 (rho 0.80)
```

Requests longer than 3,073 tokens fill the pool first. For 4096/512
that matters little: ρ is already 0.86 at the 64-seat knee, and 42
seats move it only to 0.65 requests per second.

**Where to plan.** Table A.2 runs the short workload as chapter 14
checked it against held-out runs: the engine preset with the measured
server's 64 seats (the calibrated A6000, top-p sampling, chunked prefill
of at most 2,048 prompt tokens per step), on the arrivals `vllm bench
serve` (vLLM's load generator) sent in chapter 14's runs, as recorded
(`recorded_trace`) and replayed by `bench_requests`. The script's
`plan_checks`, condensed:

```python
SERVER = VLLM.with_(max_num_seqs=64)                            # the measured server
LAT = SERVER.step_model(QWEN, A6000)
run = lambda reqs, s=1.0: SERVER.simulate(QWEN, A6000, reqs, latency_model=LAT, step_scale=s)
band = LAT.calibration.serving_error_band
SEED0 = recorded_trace(1024, 128, 240)                          # the 240-request trace

point = run(bench_requests(SEED0, rate))
interval = prediction_interval(lambda s: run(bench_requests(SEED0, rate), s), band)["ttft_p95"]
over = tip_fraction(QWEN, A6000, rate, 1000, seeds=40, latency_model=LAT,
                    max_num_seqs=64, max_num_batched_tokens=2048)
steady = [run(poisson_requests(4000, rate, 1024, 128, seed=s)) for s in (1, 2)]
```

`tip_fraction`, the share of random arrival traces whose p95 TTFT
exceeds a threshold, takes `simulate`'s arguments, not the preset, so
the seats and the token budget (the most tokens one step may carry) go
in by vLLM's names, which `simulate` accepts for `max_batch` and
`chunk_tokens`.

```
Table A.2  Planning one replica: Qwen3-8B on one RTX A6000, 1024 tokens in, 128 out; target p95 TTFT <= 1 s and p95 TPOT <= 100 ms
         the 240-request trace                                    40 draws  4,000 requests         measured
  req/s  goodput  p95 TPOT       p95 TTFT ms: point [interval]  over 1 s    p95 TTFT   p95 TPOT  p95 TTFT
   2.50      307      60.6                       634 [575 685]         0     586-623      66-68         -
   2.75      336      69.5                       671 [608 731]         0     650-668      76-80         -
   3.00      363      81.8                       787 [670 894]         6     729-757      93-96       769
   3.25      368      95.7                      990 [820 1082]        13     889-957    117-118       851
   3.50      308     112.4                     1076 [948 1220]        23   2257-2517    130-132      1053
  busiest rate passing: point 3.25, interval 3.0, draws 2.75, long traces 3.0; all four (our rule, unvalidated) 2.75: 8 replicas for 20 req/s (7 at 3.25)
  0 of 40 draws bounds the miss share below 7% (95% confidence); interval: steps x (1 -/+ 0.017), ends widened by 0.072
  measured: vLLM on the 240-request trace (in-sample), 3.25 and 3.5 from Table 18.8's re-run; goodput: requests with TTFT <= 1 s, TPOT <= 100 ms
```

Four checks, each adding margin:

- **The point prediction** says 3.25 requests per second (p95 TTFT
  990 ms), where goodput peaks too (chapter 14). The server measured
  851 ms there and 1,053 at 3.5 (in-sample: the model was built with
  these runs in view).
- **The model's error.** `prediction_interval` (chapter 18) reruns the
  simulation with every step up to 1.7% faster and slower, the model's
  step error on held-out runs, and widens each end by 7.2%, the p95-TTFT
  error left once each run's step error is removed (both 90th
  percentiles). It stays under 1 s up to 3.0. Such intervals held 35 of
  36 held-out values.
- **The traffic.** Future arrivals are another random draw.
  `tip_fraction` simulates 40 more: at 3.0, 6 exceed 1 s; at 2.75 none.
  A maximum depends on how many draws you take: 0 of 40 bounds the miss
  share only below 7%.
- **Steady load.** A 240-request trace starts empty and drains, so
  fewer requests share a step than under steady traffic. On 4,000
  requests the point answer fails, p95 TPOT 117–118 ms at 3.25; 3.0
  passes with little room.

Our rule, which nothing has validated: plan the busiest rate passing
all four. At 2.75 every check has room: the interval tops out at
731 ms, no draw exceeds 1 s, and steady traffic keeps p95 TPOT at
76–80 ms. For 20 requests per second that is 8 replicas, not the point
prediction's 7. Sending each request to a random replica gives each
Poisson arrivals at its share, so one replica's table applies.

## The price of latency

A replica serving λ requests per second of G output tokens, steadily
and below its knee, serves λ·G tokens per second:

```
$ per million tokens = GPUs × $ per GPU-hour / (λ × G × 3600 / 10^6)
```

**Prices.** Each device file (a GPU's rates and sizes, in
`data/devices/`; chapter 2) carries a `usd_per_hour` (0.49 for the RTX
A6000, 2.49 for the H100 SXM, 5.99 for the B200) that the code calls an
"illustrative public-cloud" price, with no source or date. Every cost
scales with it; yours goes in as
`Device.load("h100_sxm", usd_per_hour=2.0)`.

Table A.3 does not call `find_knee`, which keeps the busiest rate at
which 99% of the tokens come from requests meeting both targets, prices
greedy sampling and divides by goodput over a short run; on the A6000
at 100 ms it answers 1.5 requests per second at $0.70 (footer). Table
A.3 reads p95s, as Table A.2 does, from one load sweep per GPU on
Poisson traces long enough for steady load; a target only picks which
runs qualify:

```python
def operating_point(sweep, target_ms):
    ok = [rate for rate, v in sweep.items() if v["ttft95"] <= 1000 and v["tpot95"] <= target_ms]
    return max(ok) if ok else None
```

```
Table A.3  The price of latency: Qwen3-8B, one GPU per replica, 1024 tokens in, 128 out; $ per million output tokens at the busiest rate meeting p95 TTFT <= 1 s and p95 TPOT <= the target
  TPOT target ms     RTX A6000 ($0.49/h)      H100 SXM ($2.49/h)          B200 ($5.99/h)
               5                    wall                    wall      2.21 at 5.89 req/s
               8                    wall     26.88 at 0.20 req/s     0.57 at 22.64 req/s
              10                    wall      1.34 at 4.02 req/s     0.41 at 31.69 req/s
              15                    wall     0.54 at 10.05 req/s     0.29 at 45.28 req/s
              20                    wall     0.38 at 14.08 req/s     0.29 at 45.28 req/s
              25                    none     0.28 at 19.10 req/s     0.29 at 45.28 req/s
              30      3.80 at 0.28 req/s     0.28 at 19.10 req/s     0.29 at 45.28 req/s
              40      1.06 at 1.00 req/s     0.28 at 19.10 req/s     0.29 at 45.28 req/s
              60      0.53 at 2.00 req/s     0.28 at 19.10 req/s     0.29 at 45.28 req/s
             100      0.38 at 2.81 req/s     0.28 at 19.10 req/s     0.29 at 45.28 req/s
             150      0.33 at 3.21 req/s     0.28 at 19.10 req/s     0.29 at 45.28 req/s
  $/Mtok = $/h / (req/s x 128 x 3600 / 10^6), a point prediction; wall (chapter 18's a, top-p, context 1,088): 24.71, 7.54, 4.22 ms; none: no simulated rate meets it
  prices illustrative (usd_per_hour); floors, no TPOT target: 0.677, 0.114, 0.048 GPU-hours per million tokens,
  at 0.80, 0.95, 1.00 x the envelope's knee, p95 TTFT 996, 668, 711 ms
  RTX A6000: at 30 ms 11.4 x its floor; prompts take 50% of its time there; at the planned 2.75 req/s, $0.387
  find_knee, RTX A6000: 100 ms, 1.5 req/s at $0.70; 30 ms, no rate of its list (from 0.25 req/s)
  78 simulations of 240-2,000 Poisson requests, 1,574,352 engine steps, 4,010 step graphs; H100 and B200 projected
```

![Three curves of dollars per million tokens against the p95 TPOT
target, both on log scales: each starts steeply at its GPU's wall, 4.2,
7.5 and 24.7 ms, and flattens to about $0.3 per million
tokens.](/assets/tinyperf-book/appA-price.svg)

*Figure A.2. The price of latency. Each curve starts at its wall
(dotted), falls steeply and turns flat where the TTFT target and the
seats take over. The steps are the grid of simulated rates.*

**The wall.** No request decodes faster than a step with one sequence,
chapter 18's a: 24.71 ms on the A6000, top-p sampler included. No load
meets a tighter target; only a faster GPU, fewer bytes or speculation
moves the wall.

**The steep segment.** In chapter 18's envelope a step with requests
in flight stretches to `t = a / (1 − ρ)`, where ρ = λ·(P·c_p + G·b)
(P prompt tokens per request; b what each decoding sequence adds to a
step, c_p what each prompt token adds). Keeping the step under a target
T caps ρ at 1 − a/T:

```
tokens per second ≤ G · (1 − a/T) / (P·c_p + G·b)
cost per token    ∝ (P·c_p + G·b) / (G · (1 − a/T))
```

Next to a fully busy GPU, a target T multiplies the cost by
`1 / (1 − a/T)`: twice at T = 2a, five times at 1.25a, without bound at
a. The envelope is a mean and the target a 95th percentile, so the
simulated curve rises earlier (at 30 ms the A6000 pays 11.4 times its
floor): the formula gives the shape, the simulator the numbers.

**The floor.** Once the target clears the step at the knee, the TTFT
target and the seats set the rate: $0.33, $0.28 and $0.29 per million
tokens. These are point predictions at 0.80–1.00 of each envelope's
knee, without margin; at Table A.2's planned 2.75 requests per second
the A6000 costs $0.387.

So the cheapest GPU depends on the target: loose targets cost
$0.28–0.33 on all three at these prices, but below 25 ms the A6000
can't serve, nor the H100 below 8. At the A6000's floor, prompts take
half the GPU's time. The step-price cache (step prices stored by shape
and reused; chapter 14) keeps all this cheap: 1.6 million engine steps
took 4,010 step graphs.

## Energy per token

The power model charges four terms per operation, as `energy_report`
computes them row by row:

```
E_op   = executed_flops × pJ_per_flop × dtype_scale + bytes × pJ_per_byte + active_w × t
E_step = sum of E_op + static_w × step time
demand = static_w + E_op / t;   above tdp_w the operation runs demand / tdp_w times longer
```

*Executed* flops include a tile's padding rows (a GEMM runs in
fixed-size output tiles, and those at the matrix's edge are padded;
chapter 3); `dtype_scale` halves an FP8 flop's energy, an assumption.
Datasheets give the limit (the TDP, thermal design power, `tdp_w`), not
the other coefficients; `tools/measure_power.py` measures them:

```
Table A.4  The power model on the RTX A6000: four coefficients from four loads, one load held out (data/power/rtx_a6000_measured.json)
  load (tools/measure_power.py)       measured W          rate   coefficient, or the model
  idle, holding a CUDA context              78.4             -   static_w 78.4 W
  a kernel resident in L2                  145.4             -   active_w 67.0 W
  1 GB copy                                240.8      683 GB/s   pj_per_byte 139.6
  8192^3 fp16 GEMM                         299.5  106.1 TFLOPS   pj_per_flop 1.451: its whole draw above 145.4 W, DRAM included
  held out: GEMV, 1 x 8192 by 8192^2       299.0      695 GB/s   capped; tool 307 W (64-row tile), model 268 W (16-row tile, 178 us)
  the GEMM at the datasheet clock, 7.34 ms: 145.4 + 217.3 FLOPs + 28.2 DRAM = 391 W; without the DRAM term 363 W; held to 300 W, 0.80 of peak (chapter 4)
  the GEMM at the calibrated tier, 9.61 ms: 145.4 + 165.9 FLOPs + 34.2 DRAM = 346 W; 0.93 of the measured 10.30 ms, where, without the DRAM term, 300.3 W
```

Each of the first four loads fixes one coefficient (fitted). But the
energy per FLOP is the GEMM's whole draw above static and active power,
DRAM traffic included, and `energy_report` adds bytes × pJ_per_byte on
top: 28 W counted twice. Without them the GEMM demands 363 W at the
datasheet clock and, held to 300 W, runs at 0.80 of peak (chapter 4):
the power limit is most of the gap to the fitted 0.75. The same double
count, with a price 7% faster than measured, makes the calibrated GEMM
demand 346 W; at the measured time, without the bytes, it is 300.3 W by
construction. So don't cap calibrated prices: their rate already runs
at the limit.

The held-out GEMV, a decode step's shape, streams weights with almost
no useful math yet drew 299 W, at the cap, so its uncapped demand is at
least that. The tool predicts 307 W with a 64-row tile; the model
268 W with its 16-row tile, at best 0.90 (a ratio, here as in every
chapter, is predicted ÷ measured).

```
Table A.5  Energy per token, projected: Qwen3-8B, context 1,088, calibrated tier, kernels in a CUDA graph (J per token)
  GPU        coefficients    decode, 1          8         32         64  prefill, 1024  8 x 2048   watts, decode / prefill
  RTX A6000  measured            6.040      0.821      0.266      0.178         0.0495    0.0484         249-269 / 334-346
  RTX A6000  net                 5.968      0.811      0.261      0.174         0.0452    0.0441         246-262 / 305-315
  H100 SXM   derived             2.084      0.277      0.092      0.058         0.0134    0.0130         293-330 / 518-542
  B200       derived             1.642      0.217      0.069      0.043         0.0090    0.0084         432-497 / 734-815
  net: pj_per_flop 1.152, less the fitting GEMM's DRAM energy; derived: from the TDP, 12% static and 15% active assumed; A6000 prefill held to 300 W: 0.0444, 0.0420
  at Table A.3's floors, weighted by prompt share, prefill held to the limit: RTX A6000 0.185-0.190, H100 SXM 0.046-0.049, B200 0.029-0.030 kWh per million tokens
```

Table A.5 is a projection: no step's energy was measured. The net row
removes the fitting GEMM's DRAM energy from the energy per FLOP, so
each GEMM's bytes count once: decode moves 1–2%, prefill 9%. The
A6000's prefill still reads above 300 W, its calibrated price being
faster than measured.

Batch is the energy lever: on the A6000 a token decoded alone costs
6.04 J, one in a batch of 64 0.178 J, since static and active power are
paid by the second. Power barely moves (249–269 W across decode
batches), so energy per token follows GPU time per token, and a tight
latency target raises it as it raises the price.

## What speculation is worth

At a fixed batch, speculative decoding cuts the cost per token by its
speedup, 2.16 times at one sequence and 1.28 at 64 (Table 19.5). At a
latency target it also lets a larger batch fit; Table A.6 finds the
largest batch meeting each target, and the cheapest depth k, the tokens
drafted per cycle. *i.i.d.* acceptance gives every drafted token the
same chance α of being accepted; "decay 0.8" lowers it to α·0.8^(i−1)
at the i-th. An *MTP head* (multi-token prediction) is one extra layer
some models ship to draft their own next token (chapter 19).

```
Table A.6  Speculation at a latency target: decode steps alone, alpha 0.8, the cheapest depth k (1-8) and batch; $ per million output tokens
  target ms plain: batch       $  i.i.d.: k  batch       $   decay 0.8: $   decay 0.8 / plain
  Qwen3-8B, Qwen3-0.6B draft, RTX A6000 ($0.49/h), context 1024; batch at most 177 plain, 95 with the draft
         15         wall                  5     10   0.201          0.396                   -
         25            3   1.128          2     42   0.081          0.091                0.08
         30           21   0.193          1     53   0.076          0.076                0.40
         40           55   0.099          1     81   0.067          0.067                0.68
         60          128   0.063          1     95   0.065          0.065                1.04
  Qwen3.8-27B with its MTP head, B200 ($5.99/h), context 4096; batch at most 246
          6         wall                  5     42   0.236          0.414                   -
         10         wall                  3     96   0.170          0.213                   -
         12           12   1.658          3    112   0.170          0.194                0.12
         16           54   0.491          3    160   0.159          0.176                0.36
         24          133   0.300          4    246   0.155          0.170                0.57
  no prompts, sampler, seat cap or queue: read its ratios, not its dollars; memory by chapter 6's 90% rule
  plain walls (no sampler): 24.21 and 10.92 ms; the B200's 246 charges the MTP head's cache when unused (255 otherwise)
```

- **It moves the wall**, one step with one sequence: 24.21 ms for
  Qwen3-8B here (no sampler, context 1,024; Table A.3's 24.71 has both)
  and 10.92 ms for Qwen3.8-27B. Speculating, the A6000 serves 15 ms and
  the B200 6.
- **Next to the wall it is worth far more than its speedup.** At 25 ms
  plain decoding fits 3 sequences at $1.128 per million tokens;
  speculation fits 42 at $0.081 (i.i.d.), or $0.091 if acceptance
  decays. The saving is the batch the target allows.
- **At loose targets a separate draft stops paying.** At 60 ms it
  costs 4% more: the verify step's GEMMs (the target checking the
  drafted tokens) pass the ridge point, where they turn math-bound
  (chapter 19), and the draft's weights and cache cap the batch at 95
  against plain decoding's 128. The MTP head, about 5% of a target step
  with a one-layer cache, still saves 43% at 24 ms.
- **Decay matters most where targets are tightest:** at 15 ms the
  i.i.d. formula quotes about half the decaying price.

## How far to trust a plan

Only the A6000's serving answers (Table A.2 and Table A.3's first
column) rest on held-out runs on this GPU and engine: a typical error
(the ratios' usual distance from 1, `exp(mean |ln ratio|) − 1`;
chapter 3) of 7.1% on p95 TTFT and 1.9% on TPOT over 26 cells
(chapter 14), the KV pool within 1% (chapter 17), intervals holding 35
of 36 values (chapter 18); the margin rule is ours. The sweep
(collectives, the calls that combine GPUs' partial results, measured
only on a PCIe pair; chapter 11), energy per token and speculation are
projections.

## Where it breaks

- **`price_serving_config`**, which `find_knee` calls, takes no engine
  preset. It prices greedy sampling only (top-p steps
  are 2–4% longer on the A6000; chapter 18), admits by reservation (a
  request enters only if its prompt and whole reply fit beside the
  others'), which under-batches when the pool binds (chapter 17), and
  divides by a short run's makespan, drain included.
  **`price_decode_config`** at the calibrated tier charges the eager
  launch cost (each kernel issued from Python, 12–25 µs), not a CUDA
  graph's.
- **Other GPUs.** The H100 and B200 have GEMM constants only; a held-out
  B200 decode step of Qwen3.8-27B read 0.878–0.958 (chapter 8). The
  25 ms per-request overhead (time a request spends outside the GPU's
  steps; chapter 14), fitted on one A6000 server, is latency only,
  never capacity, and absent elsewhere: at the B200's 45 requests per
  second it would be 1.1 s of host work per second, untested.
- **Fixed lengths.** A real mix moves both the pool and the knee. And
  **prices** are one number per GPU: no spot pricing, power or cooling.
- **Energy.** The energy per FLOP, fitted on a GEMM's whole draw, makes
  every GEMM's DRAM bytes count twice (the net row is a first-order
  fix); the derived coefficients, giving a peak-rate GEMM the whole
  limit before its bytes, share the flaw; capping calibrated prices
  counts the limit twice.

## What you built

- A sweep and its Pareto front, and a capacity plan with margins for
  the model's error, the traffic and steady load.
- The price of latency: a wall, a steep segment as
  `1 / (1 − wall/target)`, a floor; and what speculation is worth at a
  target.
- Energy per token from four measured coefficients, with two traps:
  DRAM bytes counted twice, and the cap on calibrated prices.

## Exercises

1. **More seats.** Rerun Table A.3's B200 column with 128 and 256 seats
   (`VLLM.with_(max_num_seqs=128)`, and `VLLM` itself: 256 is vLLM's
   default). How far does the floor fall, and does the pool then bind?
2. **Two GPUs per replica.** Plan Table A.2's traffic on pairs of A6000s
   at tp = 2 (chapter 11's measured collectives). Does a pair serve more
   than twice one GPU's safe rate?
3. **Fix the double count.** Refit `pj_per_flop` from Table A.4's GEMM
   net of its own DRAM bytes, rerun the held-out GEMV, and redo Table
   A.5. Which way does the GEMV's 0.90 move?

---

*[← Chapter 22: Case study: one constant, three hidden mechanisms]({% post_url 2026-09-29-tinyperf-22-case-study %}) · [Contents](/series/tinyperf/) · [Appendix b: Glossary and conventions →]({% post_url 2026-09-29-tinyperf-appendix-b-glossary %})*
