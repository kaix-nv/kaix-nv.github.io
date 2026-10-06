---
layout: post
title: "Building tinyperf, appendix B: Glossary and conventions"
date: 2026-09-29 12:24:00 -0700
categories: [tinyperf, perf-modeling]
permalink: /tinyperf/book/appendix-b-glossary/
excerpt: "This appendix collects the chapters' conventions, the symbols they reuse and the terms they define, in their own words, shortened. The script holds all three as data, each item with the chapters it cites and a phrase to find there, and checks them:"
---

*[Building tinyperf](/series/tinyperf/) · Appendices · Every table in this appendix comes from `python3 book/scripts/appB_glossary.py`.*

This appendix collects the chapters' conventions, the symbols they
reuse and the terms they define, in their own words, shortened. The
script holds all three as data, each item with the chapters it cites
and a phrase to find there, and checks them:

```
Table B.1  Every term and symbol here, checked against the chapters it cites
  glossary terms: 144; found in every chapter they cite: 144
  terms in Conventions (labels, tiers, typical error): 12; found: 12
  symbols (Table B.2): 27; found: 27
  this appendix's glossary lists the same terms, citing the same chapters: yes
```

## Conventions

**Ratios.** A ratio is predicted ÷ measured (*model/meas* in tables):
above 1 the model reads *high*, below 1 *low*. A set of ratios is
summarized by its range or its *typical error*,
`exp(mean |ln ratio|) − 1`, the geometric mean distance from 1 either
way (chapters 1 and 3).

**Evidence labels.** Every measured number says what the model had seen
(chapters 1 and 21):

- *held out*: nothing in the model was fitted to it or designed with it
  in view. Chapter 21 asks along which axis (seed, shape, model, GPU);
  chapter 4 marks data "held out from the fit, not from the mechanism".
- *fitted*: it supplied a constant.
- *in-sample*: a mechanism was built with it in view; it stays so.
- *recorded*: stored when measured, not recomputed; it combines with the
  others.
- *projection*: nothing was measured (chapter 21).
- *frozen prediction*: numbers and pass/fail criteria committed before
  the measurement, never edited (chapters 18 and 21).

**Tiers.** Each tier is a *methodology* in the code (chapters 1 and 4):
*speed of light* (SOL), the roofline per operation at datasheet peaks;
*projected* (PROJ), the full mechanism on datasheet rates, for a GPU
nobody has measured; *calibrated* (CAL), the mechanism on constants
fitted to one GPU and software stack.

**Units.** Prose writes µs and ms, script output `us` and `ms`. GB is
decimal; GiB appears where an engine reports binary units (chapter 17),
and chapter 11's message sizes are binary KB. TFLOPS and TFLOP/s both
mean 10¹² FLOPs per second; a MAC is two FLOPs. Prose writes 15,059 and
0.97–1.05, script output 15059 and 0.97-1.05.

**Code.** Front lines link the code at a pinned commit; quotes say what
they trim, and pseudocode is labelled.

**Blocks.** A fenced block starting `Table N.M  Title`, numbered by
first appearance, is printed by the chapter's script
(`book/scripts/chNN_*.py`, or `appX_*.py`), as are *Worked example*,
*Recorded* (numbers stored when measured) and *Check* blocks.
`check_chapters.py` treats a fenced block as output if its first line
starts with "Table" or is printed by the script, then requires every
line of it in the output. A `> Field note` tells a detour once; "(not
profiled)" marks a cause the data fit but no profile confirmed.

## Symbols

Table B.2 spells the symbols as the book's tables do (alpha for α).

```
Table B.2  Symbols the book reuses, and the chapter that defines each
  symbol          meaning                                     chapter
  M, N, K         a GEMM's rows, columns and depth            3
  A, B            a weight GEMM's activations and weights     10
  b               the batch                                   1
  s               a prompt's tokens                           6
  T               the tokens in a step                        9
  kv              the cache length attention reads            6
  h, g, d         query heads, KV heads, head dimension       7
  e, k            experts, and each token's picks             9
  W               a sliding window                            8
  n               the GPUs in a collective                    11
  tp, cp, dp, pp  tensor, context, data, pipeline degrees     12
  ep              expert parallelism: 1, or tp x dp           12
  P               parameters                                  13
  m               micro-batches per step                      13
  F, D, W         forward, input- and weight-gradient chunks  13
  lambda          arrivals per second                         14, 18
  mu              completions per second, seats full          18
  rho             GPU time's share on requests' own work      18
  a               a step with one sequence                    18
  b               each further decoding sequence's cost       18
  c_p             each prompt token's cost                    18
  P, G            a request's prompt tokens, decode steps     18
  S               seats                                       18
  alpha           acceptance rate                             19
  k               drafted tokens per cycle                    19
  E               expected tokens per cycle                   19
  c, v            draft, verify step over a decode step       19
```

Several letters change meaning by chapter; the chapter's definition
wins. **b**: the batch (1, 7, 19); a decoding sequence's cost (18).
**B**: batch rows (2, 9, 15); a GEMM's weight operand (3, 10); a
buffer's bytes (11); the backward chunk D + W (13); requests decoding at
once (18). **P**: softmax probabilities (7); parameters (13); a prefix
hit's cached tokens (17); prompt tokens (18). **k**: the top k tokens
(8); each token's experts (9); the sampler's top-k (15); a step's index
(16); drafted tokens (19). **c**: a context and the MLA latent (8); a
router constant (9); a prefill wave's chunks (12); the draft's cost
ratio (19). **s**: the split-K factor (3); prompt tokens (1, 6); a
stage's index (13); a step scale (18). **m**: query rows (7); rows per
expert (9); micro-batches (13); new tokens per sequence (14). **S**:
scores (7); linear attention's state (8); a prefix hit's new tokens
(17); seats (18). **W**: a window (8); a weight matrix and the
weight-gradient chunk (13). **E**: experts (12); expected tokens (19); a
step's expert part (22). **e**: experts (9); a fraction of peak (13).
**v**: virtual stages (13); the verify ratio (19).

## Glossary

Entries cite the chapters that define each term; `code` is where
tinyperf keeps it.

- **1F1B** (ch. 13). After pp − 1 warm-up forwards, each pipeline stage
  alternates one forward and one backward: GPipe's bubble, pp
  micro-batches held.
- **1P1D** (ch. 20). One prefill GPU and one decode GPU; in general,
  xPyD.
- **2:4 sparsity** (ch. 10). Two of every four weights zero: twice the
  dense rate, 0.5625 of the weight bytes at 16 bits.
- **Absorption rule** (ch. 22). A constant fitted end to end carries
  every other part's error, sign flipped, and fails when the mix of work
  changes.
- **Acceptance rate** (ch. 19). α, the chance that a drafted position is
  accepted: an input the model can't derive.
- **Activations** (ch. 13). What the backward needs from the forward, 24
  bytes per hidden unit per token per layer; recomputation and sequence
  parallelism cut them.
- **Active parameters** (ch. 9). The weights each token uses: in an MoE
  model, the attention and other shared layers plus the k experts it
  picks. A token's math costs what these cost; a step's weight reads
  depend on the experts the whole batch touches.
- **All-gather, all-to-all, reduce-scatter** (ch. 11). See Collective.
- **All-reduce** (ch. 11). Leaves the element-wise sum of every GPU's
  buffer on every GPU.
- **Arithmetic intensity** (ch. 2). FLOPs per byte moved. Below the
  ridge point an operation is memory-bound, above it math-bound.
- **Async scheduling** (ch. 16). vLLM plans step k + 1 while step k
  runs; the model admits an arrival during step k at step k + 2.
- **Attention backend** (ch. 7). The engine's attention kernels
  (FlashAttention-2, or FA2; FlashInfer; Triton): a model parameter,
  `attn_backend`.
- **Attention replicas** (ch. 12). `dp` tensor-parallel groups serving
  their own requests, meeting only in the expert layers.
- **Bound column** (ch. 5). The report's verdict on the term that set an
  operator's price; read it with the runner-up.
- **Bubble** (ch. 12, 13). A pipeline stage's idle time while work
  crosses the others: (pp − 1)(F + B) under GPipe and 1F1B.
- **Calibration** (ch. 4). A GPU's record of four fitted constants
  (math, DRAM, L2, launch) and per-kernel measurements, for one
  software stack.
- **Capacity model** (ch. 6). `max_batch`: the sequences whose cache and
  working memory fit in 90% of memory after the weights.
- **Cascade attention** (ch. 17). Reads a shared prefix once for the
  whole batch, then each suffix; priced as a bracket.
- **Chunk step** (ch. 14). m new tokens per sequence against an
  existing context (`chunk_us`): prefix hits, verify steps.
- **Chunked prefill** (ch. 14). Decodes take one token each of a step's
  token budget; waiting prompts fill the rest in chunks.
- **Collective** (ch. 11). A call all GPUs in a group make together
  (NCCL): all-reduce; all-gather, every piece to all; reduce-scatter,
  one piece's sum to each; all-to-all, a piece from each to each.
- **Colocated** (ch. 20). Prefill and decode on the same GPU: each GPU a
  full server, the baseline disaggregated serving is set against.
- **Context parallelism** (ch. 12). `cp` groups each hold part of every
  sequence: s/cp tokens in prefill, 1/cp of the cache in decode.
- **Control** (ch. 21). The same experiment with one thing changed on
  purpose, whose result you can predict.
- **Crossover** (ch. 2, 10). The fewest rows at which a weight GEMM
  turns math-bound: equal at every compute precision, 3.8 times lower
  with 4-bit weights.
- **CTA** (ch. 3). Cooperative thread array, or thread block: the unit a
  kernel's work is cut into. One CTA computes one tile on one SM.
- **CUDA graph** (ch. 4). Kernel launches recorded once and replayed as
  one, about 3.5 µs a kernel; an engine records one per batch size.
- **CUDA-graph padding** (ch. 9, 15). A step runs in the next captured
  size up (`padded_batch`); GEMMs pay for the padded rows, and an MoE
  router routes their stale tokens.
- **Data parallelism** (ch. 13). `dp` copies of a model in training,
  all-reducing their gradients before each optimizer step.
- **Decay** (ch. 19). Acceptance α · decay^(i−1) at the i-th draft; it
  makes the best depth shallow.
- **Decode attention** (ch. 7). A floor per call
  (`fmha_decode_floor_us`) plus the cache at a fitted rate, at the
  batch's mean context.
- **Decode step** (ch. 1, 6). One token per sequence: every weight read
  once plus each sequence's cache; memory-bound.
- **Device file** (ch. 2). A GPU as about a dozen rates and sizes in
  `data/devices/`; a field earns its place only if a price uses it.
- **Disaggregated serving** (ch. 20). A prefill server computes each
  prompt's cache and ships it to a decode server; a proxy in front costs
  a hop per pass.
- **DRAM** (ch. 2). The GPU's main memory: HBM (high-bandwidth memory)
  on data-center GPUs, GDDR on the RTX A6000. The roofline's memory term
  is DRAM bytes over its bandwidth.
- **Eager** (ch. 4). PyTorch issuing each operation as Python reaches
  it: 12–25 µs per kernel, fitted.
- **Engine preset** (ch. 14). `serving.VLLM`: vLLM 0.15.1's defaults
  under vLLM's names (256 sequences, a 2,048-token budget, paged KV,
  top-p sampling), priced at the calibrated tier with CUDA-graph
  launches. The book's measured server is `VLLM.with_(max_num_seqs=64)`.
- **Engine's clock** (ch. 15, 16). Steps timed by CUDA events on the
  engine's stream, period and contents included; a client's clock
  misreads long steps.
- **Envelope** (ch. 18). A steady-state model of a server: the step
  stretches to a/(1 − ρ), and the knee is where the batch fills the
  seats. Not chapter 1's back of an envelope, nor tests named
  `*_envelope`.
- **Errors that cancel** (ch. 21). Offsetting errors, exposed when a
  correct fix worsens validated numbers: keep the fix, find the other
  error.
- **Expected tokens** (ch. 19). E = (1 − α^(k+1))/(1 − α) per cycle,
  for i.i.d. acceptance.
- **Expert parallelism** (ch. 12). `ep` ranks hold whole experts
  (ep = tp × dp); tokens travel there and back, and the step waits for
  the hot rank (`moe_imbalance`).
- **Experts touched** (ch. 9). The distinct experts a step reads, which
  set decode's cost: e(1 − (1 − k/e)^T) for a fair router; real ones
  touch fewer (measured tables).
- **Fitted end to end** (ch. 4, 20). A calibration field fitted to an
  engine's step times or TTFT, not to its own kernel: one to distrust.
- **FP8, FP4** (ch. 10). 8- and 4-bit floating point. A precision recipe
  assigns them to operators; tensor cores that support them run them at
  higher peak rates than 16-bit formats.
- **Fused attention** (ch. 7). FlashAttention keeps the scores on chip;
  one `FusedAttention` replaces QKᵀ, softmax and PV, at an efficiency of
  0.65.
- **Goodput** (ch. 14). Throughput from the requests that meet the SLO.
- **GQA** (ch. 6, 7). Grouped-query attention: query heads share KV
  heads, so the cache holds fewer; a decode kernel reads each once per
  group (GQA packing).
- **Graph** (ch. 5). The intermediate representation: an ordered list of
  operators, each with a family, tensors (shape and dtype, no data) and
  attributes such as `count`.
- **Greedy, top-p** (ch. 14, 15). Two ways the sampler picks the next
  token: greedy takes the likeliest; top-p draws among the likeliest
  tokens whose probabilities sum to p, which sorts every row of logits.
  vLLM applies a checkpoint's generation config, often top-p, unless a
  request says otherwise.
- **Grouped GEMM** (ch. 9). A batched GEMM over the experts touched,
  each with its own rows: an MoE layer.
- **Held answer** (ch. 20). On a prefill server that blocks its host, an
  answer waiting for the next forward to return (`host_sync_forward`).
- **Hop latency** (ch. 11). One hand-off's fixed cost in a ring.
  Chapter 20's hop is a proxy's cost per pass.
- **Hybrid model** (ch. 8). A model that mixes full-attention layers
  with linear-attention layers, which keep a fixed-size state instead of
  a KV cache.
- **Instrument** (ch. 21). Any means of measuring, which can measure
  something else: a host loop, a client's clock, a profiler.
- **Inter-token latency** (ch. 14). ITL: the gap between consecutive
  output tokens of one request.
- **Knee** (ch. 14, 18). The load where TTFT turns sharply upward (14);
  where a server's requests fill its seats (18).
- **KV cache** (ch. 6). Each past token's key and value per KV head per
  layer (`kv_bytes_per_token`).
- **KV connector** (ch. 20). vLLM's plug-in that moves the cache between
  disaggregated servers.
- **KV pool** (ch. 17). vLLM's 16-token blocks: memory requested, less
  the weights, a profiling pass's peak and memory outside PyTorch.
- **KV producer** (ch. 20). The connector on the prefill server; saving
  each layer's KV adds a fitted cost per layer per prompt.
- **KV transfer** (ch. 20). Cache bytes ÷ tp over the link
  (`kv_transfer_us`), sent layer by layer behind the prefill.
- **L2 cache** (ch. 2, 3). 6–126 MB shared by all SMs; a wave's tiles
  share panels through it. Its bandwidth is on no datasheet.
- **Latent cache** (ch. 8). MLA caches one small latent per token per
  layer for all heads; decode absorbs the up-projections, prefill
  materializes K and V.
- **Launch cost** (ch. 2, 3, 4). A kernel's fixed cost to start and
  finish: 3 µs by default, 12–25 µs eager, 3.5 µs from a CUDA graph.
- **Linear attention** (ch. 8). A fixed-size state instead of a cache;
  the gated delta rule updates it per token, and decode is a recurrent
  step.
- **Little's law** (ch. 18). Requests in a system = arrival rate × the
  time each spends there.
- **LM head** (ch. 6). The vocab × hidden output projection; serving
  prefill runs it on one row per sequence.
- **Lockstep** (ch. 12). Attention replicas step together, padded to the
  largest; an idle one runs a dummy batch of one repeated token.
- **Marlin** (ch. 9, 10). vLLM's MXFP4 kernel for GPUs without a 4-bit
  rate: it unpacks the weights to bf16 inside the GEMM.
- **Mean context** (ch. 7). Decode attention depends on a batch's
  contexts only through their mean, where the simulator prices it.
- **Micro-batch** (ch. 12, 13). A slice of the batch that a pipeline
  keeps in flight, so that its stages work on different slices at once.
- **Mixed step** (ch. 7, 16). Prompt chunks beside decodes: one forward
  over the combined rows, each side's attention, FA2's re-read
  (`mixed_step_us`).
- **Mixture of experts (MoE)** (ch. 9). Feed-forward layers as many
  small experts, and a router (a GEMM and a top-k) sending each token to
  k.
- **MTP head** (ch. 19). A multi-token prediction layer that predicts
  the token after the one just sampled; as a draft, about 5% of a target
  step.
- **MXFP4** (ch. 9, 10). 4-bit weights, 32 per 8-bit scale: 0.53125
  bytes per weight.
- **NCCL** (ch. 11). NVIDIA's collective communications library: the
  all-reduces, all-gathers and point-to-point sends between GPUs.
- **Node edge** (ch. 11). Where NVLink gives way to InfiniBand; a group
  spanning it all-reduces hierarchically.
- **NVLink, InfiniBand** (ch. 2, 11). NVLink is NVIDIA's direct link
  between the GPUs of a node; InfiniBand is the slower network (fabric)
  between nodes. The validated pair has neither: its GPUs talk over
  PCIe.
- **Offloading** (ch. 17). Weights or cache kept in host memory and
  brought over PCIe every step (`host_bw_gbps`).
- **Online, online table** (ch. 9, 22). Read off a server handling a
  stream of requests; the online table counts experts touched that way.
- **Operator family** (ch. 5). An operator's kind, its `op_type` (a
  GEMM, fused attention, a collective, a memory-bound op), which decides
  the function that prices it.
- **Overlap bracket** (ch. 11). Communication between the serial sum and
  perfect hiding; `comm_overlap` declares the fraction hidden.
- **Paged KV** (ch. 14, 17). Admission against the 16-token blocks
  requests fill; when none is free, the newest request is preempted.
- **Pareto front** (app. A). The configurations no other beats on both
  metrics (`pareto`).
- **Pass** (ch. 5, 10). A function that rewrites a graph: fusion, the
  backward, precision tags.
- **Per-request overhead** (ch. 14, 16). `online_overhead_us`, 25 ms
  fitted on an idle server: the path outside the steps, added to TTFT
  and end-to-end time.
- **Pinned test** (ch. 21). A validated grid turned into a test that
  asserts a band; tests also pin directions, controls and known misses.
- **Pipeline fill** (ch. 3). The stages − 1 iterations before a
  kernel's load-compute pipeline runs at rate; these stages are slices
  in shared memory.
- **Pipeline parallelism** (ch. 12). `pp` stages of consecutive layers:
  a memory lever, no help to a weight-bound decode step.
- **Pipeline schedule** (ch. 13). The order of F, D and W chunks: GPipe,
  1F1B, interleaved, zero-bubble (ZB-H1, ZB-H2) and DualPipe, each
  buying a smaller bubble with memory.
- **Precision recipe** (ch. 10). Operator-name prefixes mapped to dtypes
  (`apply_recipe`). `RECIPE_FP4_SERVING`: weight GEMMs fp4, attention
  and cache fp8, head and router 16-bit.
- **Prediction interval** (ch. 18). The simulator rerun across the
  model's step error, widened by the rest (`prediction_interval`): wide
  at the knee.
- **Preemption** (ch. 17). Freeing the newest running request's blocks
  and requeueing it at the front, to resume by recomputation.
- **Prefill** (ch. 1, 6). One pass over the prompt: about 2 FLOPs per
  non-embedding parameter per token, plus attention; math-bound.
- **Prefill-first** (ch. 14). A step that admits requests is a
  dedicated prefill; decodes wait.
- **Prefix cache** (ch. 17). Keeps a shared prefix's blocks; a hit
  prefills only the rest, as a chunk step.
- **Price** (ch. 1). To predict the time of an operation, a step or a
  run; a price is that predicted time.
- **Queue amplification** (ch. 18). Past the knee the queue grows at
  λ − μ, which a step error ε changes by a fraction ε·μ/(λ − μ).
- **Rank** (ch. 12, 13). One GPU of a parallel group, numbered within
  it.
- **Recomputation** (ch. 13, 17). Rerunning a layer's forward in the
  backward instead of storing its activations (13); re-prefilling a
  preempted request (17).
- **Reply position** (ch. 9). How far a step's decodes are into their
  replies; routing spreads as it grows.
- **Re-read** (ch. 7, 16). Beside a chunk, FA2 reads each decode row's
  cache once per query head, less L2 hits (`mixed_reread_us`).
- **Reservation** (ch. 14, 17). Admitting a request only if prompt plus
  `max_gen` fits beside others' reservations: no preemption, but
  phantom room.
- **Residual** (ch. 4, 21). A fit's leftover error: tracking one term's
  share, it points at a constant; changing with shape, at a mechanism.
- **Ridge point** (ch. 2). Peak FLOPs over memory bandwidth, in FLOPs
  per byte: below it an operation is memory-bound, above it math-bound.
- **Ring** (ch. 11). An all-reduce passed round n GPUs: 2(n − 1)/n ·
  B/link plus 2(n − 1) hops and a launch.
- **RMSNorm, RoPE, SwiGLU** (ch. 5). Parts of the transformers modeled
  here: RMSNorm divides each row by its root mean square; RoPE (rotary
  position embeddings) rotates queries and keys by position; SwiGLU, the
  feed-forward activation, multiplies the activated gate half of its
  input by the up half.
- **Roofline** (ch. 1, 2). `max(flops / peak, bytes / bandwidth)`: a
  floor for any kernel reading from DRAM.
- **Row curve** (ch. 4, 15). `dense_gemm_row_factor`: cuBLAS over the
  tile model by rows, carrying the library's tile edges.
- **Sampler** (ch. 15). Picks each next token after the forward: a fixed
  cost plus fp32 passes over the vocabulary per row, by mode.
- **Saturation** (ch. 14, 17). Requests arriving faster than they finish
  (14); the KV pool binding (17).
- **Scheduler** (ch. 5, 14). tinyperf's `execute`, pricing each operator
  and adding (5); an engine's, choosing each step's requests (14).
- **Seat queue** (ch. 18). The requests in an engine beyond its seats,
  waiting for one.
- **Seats** (ch. 18). The most requests an engine runs at once, vLLM's
  `max_num_seqs`.
- **Serving engine** (ch. 1). The software that batches requests and
  runs the model on the GPU: vLLM in this book.
- **Sink** (ch. 7, 8). gpt-oss's learned extra softmax logit per head
  (7); the first tokens a window keeps (8).
- **SLO** (ch. 14). Service-level objective: a latency target per
  request, such as TTFT ≤ 1 s and TPOT ≤ 100 ms.
- **SM** (ch. 2). Streaming multiprocessor, one of a GPU's many
  processors (84 on the RTX A6000); a GPU's peak rates are per-SM rates
  times the SM count.
- **Software stack** (ch. 4). How kernels are issued, eager or graph
  (`for_stack`); the launch cost belongs to it.
- **Sparse and compressed attention** (ch. 8). Attention to the top k
  tokens an indexer picks (DSA), or to entries pooling r tokens each.
- **Speculative decoding** (ch. 19). A draft proposes k tokens and the
  target verifies them in one step; rejection sampling leaves the output
  unchanged.
- **Split-K** (ch. 3). K divided among s thread blocks per tile, their
  fp32 partial sums added by a second kernel.
- **Step** (ch. 1). One forward pass of the model over a batch, the unit
  the simulator prices.
- **Step bias** (ch. 18). The model's step prices over the engine's
  clock; outside the band, it points at a missing mechanism.
- **Step model** (ch. 1, 14). `StepLatencyModel`: prices a step from its
  shape (the batch, its context, any prompt chunk) by building and
  pricing its graph, and caches the prices; the serving simulator calls
  it at every step.
- **Step-price cache** (ch. 14). Step prices keyed by shape, contexts
  priced every 256 tokens and interpolated.
- **Tensor parallelism** (ch. 11). `tp` GPUs each holding 1/tp of every
  weight matrix, with two all-reduces per layer.
- **Tile** (ch. 3). A BM × BN output block computed by one thread block
  (CTA); tile quantization is the padding where a matrix's edge cuts
  one.
- **Tile model** (ch. 3). Chapter 3's GEMM price: tiles and waves, L2
  reuse, split-K and a launch cost, searched over the tiles a library
  offers.
- **Timed alone** (ch. 4). The calibration's rule: each field measured
  on its own kernel, in the engine's configuration, never fitted to the
  numbers it is checked against.
- **Token budget** (ch. 12, 14). The most tokens one step may carry,
  `max_num_batched_tokens`.
- **Top-k** (ch. 8, 9, 15). The k tokens sparse attention keeps (8),
  the experts a router picks (9), the logits a sampler keeps (15).
- **TPOT** (ch. 1, 14). Time per output token: (last − first token
  time)/(n − 1); on a fixed batch, one decode step.
- **Trace** (ch. 14). The requests a benchmark sent: each one's send
  time, prompt length and output length. `recorded_trace` loads the ones
  measured here; `bench_requests` replays one at a rate.
- **Triton** (ch. 7). A Python-based language for GPU kernels; one of
  vLLM's attention backends is written in it.
- **TTFT** (ch. 1, 14). Time to first token, from sending a request,
  waiting included; on a fixed batch, until every prompt has one.
- **Twin** (ch. 8). The same model with one attention design removed,
  priced beside it.
- **Uniform routing** (ch. 9). Every expert equally likely to be picked
  by every token: what a model gets when nobody has measured its
  routing.
- **Verify step** (ch. 19). The target's chunk step over k + 1
  positions: about one decode step below the ridge point.
- **Wall** (app. A). A one-sequence decode step: no tighter TPOT target
  can be met, and cost grows as 1/(1 − wall/target) near it.
- **Wave** (ch. 3, 12). A round of thread blocks, one per SM, as slow as
  its busiest (wave quantization); in chapter 12, chunks crossing
  pipeline stages.
- **Weight-only quantization** (ch. 10). 4-bit weights unpacked to 16
  bits in the kernel, so the math keeps the 16-bit rate.
- **ZeRO** (ch. 13). Sharding Adam's 16 bytes per parameter over
  data-parallel ranks: optimizer state, then gradients, then weights.

---

*[← Appendix a: Using the estimator]({% post_url 2026-09-29-tinyperf-appendix-a-using-the-estimator %}) · [Contents](/series/tinyperf/)*
