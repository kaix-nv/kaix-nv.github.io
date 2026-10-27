---
layout: post
title: "Building tinyperf, appendix B: Glossary and conventions"
date: 2026-09-29 12:25:00 -0700
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
  glossary terms: 146; found in every chapter they cite: 146
  terms in Conventions (labels, tiers, typical error): 12; found: 12
  symbols (Table B.2): 27; found: 27
  this appendix's glossary lists the same terms, citing the same chapters: yes
```

## Conventions

**Ratios.** A ratio is predicted ÷ measured (*model/meas* in tables):
above 1 the model reads *high*, below 1 *low*. A set of ratios is
summarized by its range or its *typical error*,
`exp(mean |ln ratio|) − 1`, the geometric mean distance from 1 either
way (chapters 1 and 4).

**Evidence labels.** Every measured number says what the model had seen
(chapters 1 and 22):

- *held out*: nothing in the model was fitted to it or designed with it
  in view. Chapter 22 asks along which axis (seed, shape, model, GPU);
  chapter 5 marks data "held out from the fit, not from the mechanism".
- *fitted*: it supplied a constant.
- *in-sample*: a mechanism was built with it in view; it stays so.
- *recorded*: stored when measured, not recomputed; it combines with the
  others.
- *projection*: nothing was measured (chapter 22).
- *frozen prediction*: numbers and pass/fail criteria committed before
  the measurement, never edited (chapters 19 and 22).

**Tiers.** Each tier is a *methodology* in the code (chapters 1 and 5):
*speed of light* (SOL), the roofline per operation at datasheet peaks;
*projected* (PROJ), the full mechanism on datasheet rates, for a GPU
nobody has measured; *calibrated* (CAL), the mechanism on constants
fitted to one GPU and software stack.

**Units.** Prose writes µs and ms, script output `us` and `ms`. GB is
decimal; GiB appears where an engine reports binary units (chapter 18),
and chapter 12's message sizes are binary KB. TFLOPS and TFLOP/s both
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
  M, N, K         a GEMM's rows, columns and depth            4
  A, B            a weight GEMM's activations and weights     11
  b               the batch                                   1
  s               a prompt's tokens                           7
  T               the tokens in a step                        10
  kv              the cache length attention reads            7
  h, g, d         query heads, KV heads, head dimension       8
  e, k            experts, and each token's picks             10
  W               a sliding window                            9
  n               the GPUs in a collective                    12
  tp, cp, dp, pp  tensor, context, data, pipeline degrees     13
  ep              expert parallelism: 1, or tp x dp           13
  P               parameters                                  14
  m               micro-batches per step                      14
  F, D, W         forward, input- and weight-gradient chunks  14
  lambda          arrivals per second                         15, 19
  mu              completions per second, seats full          19
  rho             GPU time's share on requests' own work      19
  a               a step with one sequence                    19
  b               each further decoding sequence's cost       19
  c_p             each prompt token's cost                    19
  P, G            a request's prompt tokens, decode steps     19
  S               seats                                       19
  alpha           acceptance rate                             20
  k               drafted tokens per cycle                    20
  E               expected tokens per cycle                   20
  c, v            draft, verify step over a decode step       20
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

- **1F1B** (ch. 14). After pp − 1 warm-up forwards, each pipeline stage
  alternates one forward and one backward: GPipe's bubble, pp
  micro-batches held.
- **1P1D** (ch. 21). One prefill GPU and one decode GPU; in general,
  xPyD.
- **2:4 sparsity** (ch. 11). Two of every four weights zero: twice the
  dense rate, 0.5625 of the weight bytes at 16 bits.
- **Absorption rule** (ch. 23). A constant fitted end to end carries
  every other part's error, sign flipped, and fails when the mix of work
  changes.
- **Acceptance rate** (ch. 20). α, the chance that a drafted position is
  accepted: an input the model can't derive.
- **Activations** (ch. 14). What the backward needs from the forward, 24
  bytes per hidden unit per token per layer; recomputation and sequence
  parallelism cut them.
- **Active parameters** (ch. 10). The weights each token uses: in an MoE
  model, the attention and other shared layers plus the k experts it
  picks. A token's math costs what these cost; a step's weight reads
  depend on the experts the whole batch touches.
- **All-gather, all-to-all, reduce-scatter** (ch. 12). See Collective.
- **All-reduce** (ch. 12). Leaves the element-wise sum of every GPU's
  buffer on every GPU.
- **Arithmetic intensity** (ch. 3). FLOPs per byte moved. Below the
  ridge point an operation is memory-bound, above it math-bound.
- **Async scheduling** (ch. 17). vLLM plans step k + 1 while step k
  runs; the model admits an arrival during step k at step k + 2.
- **Attention backend** (ch. 8). The engine's attention kernels
  (FlashAttention-2, or FA2; FlashInfer; Triton): a model parameter,
  `attn_backend`.
- **Attention replicas** (ch. 13). `dp` tensor-parallel groups serving
  their own requests, meeting only in the expert layers.
- **Bound column** (ch. 6). The report's verdict on the term that set an
  operator's price; read it with the runner-up.
- **Bubble** (ch. 13, 14). A pipeline stage's idle time while work
  crosses the others: (pp − 1)(F + B) under GPipe and 1F1B.
- **Calibration** (ch. 5). A GPU's record of four fitted constants
  (math, DRAM, L2, launch) and per-kernel measurements, for one
  software stack.
- **Capacity model** (ch. 7). `max_batch`: the sequences whose cache and
  working memory fit in 90% of memory after the weights.
- **Cascade attention** (ch. 18). Reads a shared prefix once for the
  whole batch, then each suffix; priced as a bracket.
- **Chunk step** (ch. 15). m new tokens per sequence against an
  existing context (`chunk_us`): prefix hits, verify steps.
- **Chunked prefill** (ch. 15). Decodes take one token each of a step's
  token budget; waiting prompts fill the rest in chunks.
- **Collective** (ch. 12). A call all GPUs in a group make together
  (NCCL): all-reduce; all-gather, every piece to all; reduce-scatter,
  one piece's sum to each; all-to-all, a piece from each to each.
- **Colocated** (ch. 21). Prefill and decode on the same GPU: each GPU a
  full server, the baseline disaggregated serving is set against.
- **Context parallelism** (ch. 13). `cp` groups each hold part of every
  sequence: s/cp tokens in prefill, 1/cp of the cache in decode.
- **Control** (ch. 22). The same experiment with one thing changed on
  purpose, whose result you can predict.
- **Crossover** (ch. 3, 11). The fewest rows at which a weight GEMM
  turns math-bound: equal at every compute precision, 3.8 times lower
  with 4-bit weights.
- **CTA** (ch. 4). Cooperative thread array, or thread block: the unit a
  kernel's work is cut into. One CTA computes one tile on one SM.
- **CUDA graph** (ch. 5). Kernel launches recorded once and replayed as
  one, about 3.5 µs a kernel; an engine records one per batch size.
- **CUDA-graph padding** (ch. 10, 16). A step runs in the next captured
  size up (`padded_batch`); GEMMs pay for the padded rows, and an MoE
  router routes their stale tokens.
- **Data parallelism** (ch. 14). `dp` copies of a model in training,
  all-reducing their gradients before each optimizer step.
- **Decay** (ch. 20). Acceptance α · decay^(i−1) at the i-th draft; it
  makes the best depth shallow.
- **Decode attention** (ch. 8). A floor per call
  (`fmha_decode_floor_us`) plus the cache at a fitted rate, at the
  batch's mean context.
- **Decode step** (ch. 1, 7). One token per sequence: every weight read
  once plus each sequence's cache; memory-bound.
- **Device file** (ch. 3). A GPU as about a dozen rates and sizes in
  `data/devices/`; a field earns its place only if a price uses it.
- **Disaggregated serving** (ch. 21). A prefill server computes each
  prompt's cache and ships it to a decode server; a proxy in front costs
  a hop per pass.
- **DRAM** (ch. 3). The GPU's main memory: HBM (high-bandwidth memory)
  on data-center GPUs, GDDR on the RTX A6000. The roofline's memory term
  is DRAM bytes over its bandwidth.
- **Eager** (ch. 5). PyTorch issuing each operation as Python reaches
  it: 12–25 µs per kernel, fitted.
- **Engine preset** (ch. 15). `serving.VLLM`: vLLM 0.15.1's defaults
  under vLLM's names (256 sequences, a 2,048-token budget, paged KV,
  top-p sampling), priced at the calibrated tier with CUDA-graph
  launches. The book's measured server is `VLLM.with_(max_num_seqs=64)`.
- **Engine's clock** (ch. 16, 17). Steps timed by CUDA events on the
  engine's stream, period and contents included; a client's clock
  misreads long steps.
- **Envelope** (ch. 19). A steady-state model of a server: the step
  stretches to a/(1 − ρ), and the knee is where the batch fills the
  seats. Not chapter 1's back of an envelope, nor tests named
  `*_envelope`.
- **Errors that cancel** (ch. 22). Offsetting errors, exposed when a
  correct fix worsens validated numbers: keep the fix, find the other
  error.
- **Expected tokens** (ch. 20). E = (1 − α^(k+1))/(1 − α) per cycle,
  for i.i.d. acceptance.
- **Expert parallelism** (ch. 13). `ep` ranks hold whole experts
  (ep = tp × dp); tokens travel there and back, and the step waits for
  the hot rank (`moe_imbalance`).
- **Experts touched** (ch. 10). The distinct experts a step reads, which
  set decode's cost: e(1 − (1 − k/e)^T) for a fair router; real ones
  touch fewer (measured tables).
- **Fitted end to end** (ch. 5, 21). A calibration field fitted to an
  engine's step times or TTFT, not to its own kernel: one to distrust.
- **FP8, FP4** (ch. 11). 8- and 4-bit floating point. A precision recipe
  assigns them to operators; tensor cores that support them run them at
  higher peak rates than 16-bit formats.
- **Fused attention** (ch. 8). FlashAttention keeps the scores on chip;
  one `FusedAttention` replaces QKᵀ, softmax and PV, at an efficiency of
  0.65.
- **Goodput** (ch. 15). Throughput from the requests that meet the SLO.
- **GQA** (ch. 7, 8). Grouped-query attention: query heads share KV
  heads, so the cache holds fewer; a decode kernel reads each once per
  group (GQA packing).
- **Graph** (ch. 6). The intermediate representation: an ordered list of
  operators, each with a family, tensors (shape and dtype, no data) and
  attributes such as `count`.
- **Greedy, top-p** (ch. 15, 16). Two ways the sampler picks the next
  token: greedy takes the likeliest; top-p draws among the likeliest
  tokens whose probabilities sum to p, which sorts every row of logits.
  vLLM applies a checkpoint's generation config, often top-p, unless a
  request says otherwise.
- **Grouped GEMM** (ch. 10). A batched GEMM over the experts touched,
  each with its own rows: an MoE layer.
- **Held answer** (ch. 21). On a prefill server that blocks its host, an
  answer waiting for the next forward to return (`host_sync_forward`).
- **Hop latency** (ch. 12). One hand-off's fixed cost in a ring.
  Chapter 21's hop is a proxy's cost per pass.
- **Hybrid model** (ch. 9). A model that mixes full-attention layers
  with linear-attention layers, which keep a fixed-size state instead of
  a KV cache.
- **Instrument** (ch. 22). Any means of measuring, which can measure
  something else: a host loop, a client's clock, a profiler.
- **Inter-token latency** (ch. 15). ITL: the gap between consecutive
  output tokens of one request.
- **Knee** (ch. 15, 19). The load where TTFT turns sharply upward (14);
  where a server's requests fill its seats (18).
- **KV cache** (ch. 7). Each past token's key and value per KV head per
  layer (`kv_bytes_per_token`).
- **KV connector** (ch. 21). vLLM's plug-in that moves the cache between
  disaggregated servers.
- **KV pool** (ch. 18). vLLM's 16-token blocks: memory requested, less
  the weights, a profiling pass's peak and memory outside PyTorch.
- **KV producer** (ch. 21). The connector on the prefill server; saving
  each layer's KV adds a fitted cost per layer per prompt.
- **KV transfer** (ch. 21). Cache bytes ÷ tp over the link
  (`kv_transfer_us`), sent layer by layer behind the prefill.
- **L2 cache** (ch. 3, 4). 6–126 MB shared by all SMs; a wave's tiles
  share panels through it. Its bandwidth is on no datasheet.
- **Latent cache** (ch. 9). MLA caches one small latent per token per
  layer for all heads; decode absorbs the up-projections, prefill
  materializes K and V.
- **Launch cost** (ch. 3, 4, 5). A kernel's fixed cost to start and
  finish: 3 µs by default, 12–25 µs eager, 3.5 µs from a CUDA graph.
- **Linear attention** (ch. 9). A fixed-size state instead of a cache;
  the gated delta rule updates it per token, and decode is a recurrent
  step.
- **Little's law** (ch. 19). Requests in a system = arrival rate × the
  time each spends there.
- **LM head** (ch. 7). The vocab × hidden output projection; serving
  prefill runs it on one row per sequence.
- **Lockstep** (ch. 13). Attention replicas step together, padded to the
  largest; an idle one runs a dummy batch of one repeated token.
- **Marlin** (ch. 10, 11). vLLM's MXFP4 kernel for GPUs without a 4-bit
  rate: it unpacks the weights to bf16 inside the GEMM.
- **Mean context** (ch. 8). Decode attention depends on a batch's
  contexts only through their mean, where the simulator prices it.
- **Micro-batch** (ch. 13, 14). A slice of the batch that a pipeline
  keeps in flight, so that its stages work on different slices at once.
- **Mixed step** (ch. 8, 17). Prompt chunks beside decodes: one forward
  over the combined rows, each side's attention, FA2's re-read
  (`mixed_step_us`).
- **Mixture of experts (MoE)** (ch. 10). Feed-forward layers as many
  small experts, and a router (a GEMM and a top-k) sending each token to
  k.
- **MTP head** (ch. 20). A multi-token prediction layer that predicts
  the token after the one just sampled; as a draft, about 5% of a target
  step.
- **MXFP4** (ch. 10, 11). 4-bit weights, 32 per 8-bit scale: 0.53125
  bytes per weight.
- **NCCL** (ch. 12). NVIDIA's collective communications library: the
  all-reduces, all-gathers and point-to-point sends between GPUs.
- **Node edge** (ch. 12). Where NVLink gives way to InfiniBand; a group
  spanning it all-reduces hierarchically.
- **NVLink, InfiniBand** (ch. 3, 12). NVLink is NVIDIA's direct link
  between the GPUs of a node; InfiniBand is the slower network (fabric)
  between nodes. The validated pair has neither: its GPUs talk over
  PCIe.
- **Offloading** (ch. 18). Weights or cache kept in host memory and
  brought over PCIe every step (`host_bw_gbps`).
- **Online, online table** (ch. 10, 23). Read off a server handling a
  stream of requests; the online table counts experts touched that way.
- **Operator family** (ch. 6). An operator's kind, its `op_type` (a
  GEMM, fused attention, a collective, a memory-bound op), which decides
  the function that prices it.
- **Overlap bracket** (ch. 12). Communication between the serial sum and
  perfect hiding; `comm_overlap` declares the fraction hidden.
- **Paged KV** (ch. 15, 18). Admission against the 16-token blocks
  requests fill; when none is free, the newest request is preempted.
- **Pareto front** (app. A). The configurations no other beats on both
  metrics (`pareto`).
- **Pass** (ch. 6, 11). A function that rewrites a graph: fusion, the
  backward, precision tags.
- **Per-request overhead** (ch. 15, 17). `online_overhead_us`, 25 ms
  fitted on an idle server: the path outside the steps, added to TTFT
  and end-to-end time.
- **Pinned test** (ch. 22). A validated grid turned into a test that
  asserts a band; tests also pin directions, controls and known misses.
- **Pipeline fill** (ch. 4). The stages − 1 iterations before a
  kernel's load-compute pipeline runs at rate; these stages are slices
  in shared memory.
- **Pipeline parallelism** (ch. 13). `pp` stages of consecutive layers:
  a memory lever, no help to a weight-bound decode step.
- **Pipeline schedule** (ch. 14). The order of F, D and W chunks: GPipe,
  1F1B, interleaved, zero-bubble (ZB-H1, ZB-H2) and DualPipe, each
  buying a smaller bubble with memory.
- **Precision recipe** (ch. 11). Operator-name prefixes mapped to dtypes
  (`apply_recipe`). `RECIPE_FP4_SERVING`: weight GEMMs fp4, attention
  and cache fp8, head and router 16-bit.
- **Prediction interval** (ch. 19). The simulator rerun across the
  model's step error, widened by the rest (`prediction_interval`): wide
  at the knee.
- **Preemption** (ch. 18). Freeing the newest running request's blocks
  and requeueing it at the front, to resume by recomputation.
- **Prefill** (ch. 1, 7). One pass over the prompt: about 2 FLOPs per
  non-embedding parameter per token, plus attention; math-bound.
- **Prefill-first** (ch. 15). A step that admits requests is a
  dedicated prefill; decodes wait.
- **Prefix cache** (ch. 18). Keeps a shared prefix's blocks; a hit
  prefills only the rest, as a chunk step.
- **Price** (ch. 1). To predict the time of an operation, a step or a
  run; a price is that predicted time.
- **Queue amplification** (ch. 19). Past the knee the queue grows at
  λ − μ, which a step error ε changes by a fraction ε·μ/(λ − μ).
- **Rank** (ch. 13, 14). One GPU of a parallel group, numbered within
  it.
- **Recomputation** (ch. 14, 18). Rerunning a layer's forward in the
  backward instead of storing its activations (13); re-prefilling a
  preempted request (17).
- **Reply position** (ch. 10). How far a step's decodes are into their
  replies; routing spreads as it grows.
- **Re-read** (ch. 8, 17). Beside a chunk, FA2 reads each decode row's
  cache once per query head, less L2 hits (`mixed_reread_us`).
- **Reservation** (ch. 15, 18). Admitting a request only if prompt plus
  `max_gen` fits beside others' reservations: no preemption, but
  phantom room.
- **Residual** (ch. 5, 22). A fit's leftover error: tracking one term's
  share, it points at a constant; changing with shape, at a mechanism.
- **Ridge point** (ch. 3). Peak FLOPs over memory bandwidth, in FLOPs
  per byte: below it an operation is memory-bound, above it math-bound.
- **Ring** (ch. 12). An all-reduce passed round n GPUs: 2(n − 1)/n ·
  B/link plus 2(n − 1) hops and a launch.
- **RMSNorm, RoPE, SwiGLU** (ch. 6). Parts of the transformers modeled
  here: RMSNorm divides each row by its root mean square; RoPE (rotary
  position embeddings) rotates queries and keys by position; SwiGLU, the
  feed-forward activation, multiplies the activated gate half of its
  input by the up half.
- **Roofline** (ch. 1, 3). `max(flops / peak, bytes / bandwidth)`: a
  floor for any kernel reading from DRAM.
- **Row curve** (ch. 5, 16). `dense_gemm_row_factor`: cuBLAS over the
  tile model by rows, carrying the library's tile edges.
- **Sampler** (ch. 16). Picks each next token after the forward: a fixed
  cost plus fp32 passes over the vocabulary per row, by mode.
- **Saturation** (ch. 15, 18). Requests arriving faster than they finish
  (14); the KV pool binding (17).
- **Scheduler** (ch. 6, 15). tinyperf's `execute`, pricing each operator
  and adding (5); an engine's, choosing each step's requests (14).
- **Seat queue** (ch. 19). The requests in an engine beyond its seats,
  waiting for one.
- **Seats** (ch. 19). The most requests an engine runs at once, vLLM's
  `max_num_seqs`.
- **Serving engine** (ch. 1). The software that batches requests and
  runs the model on the GPU: vLLM in this book.
- **Sink** (ch. 8, 9). gpt-oss's learned extra softmax logit per head
  (7); the first tokens a window keeps (8).
- **SLO** (ch. 15). Service-level objective: a latency target per
  request, such as TTFT ≤ 1 s and TPOT ≤ 100 ms.
- **SM** (ch. 3). Streaming multiprocessor, one of a GPU's many
  processors (84 on the RTX A6000); a GPU's peak rates are per-SM rates
  times the SM count.
- **Software stack** (ch. 5). How kernels are issued, eager or graph
  (`for_stack`); the launch cost belongs to it.
- **Sparse and compressed attention** (ch. 9). Attention to the top k
  tokens an indexer picks (DSA), or to entries pooling r tokens each.
- **Speculative decoding** (ch. 20). A draft proposes k tokens and the
  target verifies them in one step; rejection sampling leaves the output
  unchanged.
- **Split-K** (ch. 4). K divided among s thread blocks per tile, their
  fp32 partial sums added by a second kernel.
- **Step** (ch. 1). One forward pass of the model over a batch, the unit
  the simulator prices.
- **Step bias** (ch. 19). The model's step prices over the engine's
  clock; outside the band, it points at a missing mechanism.
- **Step model** (ch. 1, 2, 15). `StepLatencyModel`: prices a step from its
  shape (the batch, its context, any prompt chunk) by building and
  pricing its graph, and caches the prices; the serving simulator calls
  it at every step.
- **Step shape** (ch. 2). What a step's price depends on, given the
  model and its split across GPUs: the phase, the batch, and a prefill's
  prompt length or a decode's context.
- **Step-price cache** (ch. 15). Step prices keyed by shape, contexts
  priced every 256 tokens and interpolated.
- **Tensor parallelism** (ch. 12). `tp` GPUs each holding 1/tp of every
  weight matrix, with two all-reduces per layer.
- **Tile** (ch. 4). A BM × BN output block computed by one thread block
  (CTA); tile quantization is the padding where a matrix's edge cuts
  one.
- **Tile model** (ch. 4). Chapter 4's GEMM price: tiles and waves, L2
  reuse, split-K and a launch cost, searched over the tiles a library
  offers.
- **Timed alone** (ch. 5). The calibration's rule: each field measured
  on its own kernel, in the engine's configuration, never fitted to the
  numbers it is checked against.
- **Token budget** (ch. 13, 15). The most tokens one step may carry,
  `max_num_batched_tokens`.
- **Top-k** (ch. 9, 10, 16). The k tokens sparse attention keeps (8),
  the experts a router picks (9), the logits a sampler keeps (15).
- **TPOT** (ch. 1, 15). Time per output token: (last − first token
  time)/(n − 1); on a fixed batch, one decode step.
- **Trace** (ch. 15). The requests a benchmark sent: each one's send
  time, prompt length and output length. `recorded_trace` loads the ones
  measured here; `bench_requests` replays one at a rate.
- **Triton** (ch. 8). A Python-based language for GPU kernels; one of
  vLLM's attention backends is written in it.
- **TTFT** (ch. 1, 15). Time to first token, from sending a request,
  waiting included; on a fixed batch, until every prompt has one.
- **Twin** (ch. 9). The same model with one attention design removed,
  priced beside it.
- **Uniform routing** (ch. 10). Every expert equally likely to be picked
  by every token: what a model gets when nobody has measured its
  routing.
- **Verify step** (ch. 20). The target's chunk step over k + 1
  positions: about one decode step below the ridge point.
- **Wall** (app. A). A one-sequence decode step: no tighter TPOT target
  can be met, and cost grows as 1/(1 − wall/target) near it.
- **Wave** (ch. 4, 13). A round of thread blocks, one per SM, as slow as
  its busiest (wave quantization); in chapter 13, chunks crossing
  pipeline stages.
- **Weight-only quantization** (ch. 11). 4-bit weights unpacked to 16
  bits in the kernel, so the math keeps the 16-bit rate.
- **Write-back** (ch. 16). The GEMMs after a linear core slowing by
  about the time to write its new state to memory, consistent with that
  state leaving the cache while they run.
- **ZeRO** (ch. 14). Sharding Adam's 16 bytes per parameter over
  data-parallel ranks: optimizer state, then gradients, then weights.

---

*[← Appendix a: Using the estimator]({% post_url 2026-09-29-tinyperf-appendix-a-using-the-estimator %}) · [Contents](/series/tinyperf/)*
