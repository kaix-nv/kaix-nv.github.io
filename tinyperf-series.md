---
layout: page
title: Building tinyperf
permalink: /series/tinyperf/
---

A book in 22 chapters on building an analytical GPU performance model
for LLMs from scratch: price a kernel from a GPU's datasheet, then a
model, a parallel layout and a serving engine under load, and check
every step against a real GPU. Each chapter asks one question, builds
the mechanism that answers it, shows the code and the evidence, and
says where it breaks. Every number in it is printed by a script in the
repository and labelled by what the model had seen.

## Part I: Pricing one kernel

- Chapter 1: [What a performance model is for]({% post_url 2026-09-29-tinyperf-01-what-a-performance-model-is-for %})
- Chapter 2: [A GPU as a handful of rates]({% post_url 2026-09-29-tinyperf-02-a-gpu-as-a-handful-of-rates %})
- Chapter 3: [Pricing a GEMM]({% post_url 2026-09-29-tinyperf-03-pricing-a-gemm %})
- Chapter 4: [Meeting a real GPU: calibration and its tiers]({% post_url 2026-09-29-tinyperf-04-meeting-a-real-gpu %})

## Part II: One model on one GPU

- Chapter 5: [Graphs and the scheduler]({% post_url 2026-09-29-tinyperf-05-graphs-and-the-scheduler %})
- Chapter 6: [A transformer: prefill, decode and memory]({% post_url 2026-09-29-tinyperf-06-a-transformer-prefill-decode-and-memory %})
- Chapter 7: [Attention]({% post_url 2026-09-29-tinyperf-07-attention %})
- Chapter 8: [Attention variants]({% post_url 2026-09-29-tinyperf-08-attention-variants %})
- Chapter 9: [Mixture of experts]({% post_url 2026-09-29-tinyperf-09-mixture-of-experts %})
- Chapter 10: [Precision and sparsity as passes]({% post_url 2026-09-29-tinyperf-10-precision-and-sparsity %})

## Part III: Many GPUs

- Chapter 11: [Collectives and tensor parallelism]({% post_url 2026-09-29-tinyperf-11-collectives-and-tensor-parallelism %})
- Chapter 12: [Parallel layouts]({% post_url 2026-09-29-tinyperf-12-parallel-layouts %})
- Chapter 13: [Training]({% post_url 2026-09-29-tinyperf-13-training %})

## Part IV: Serving

- Chapter 14: [A serving simulator]({% post_url 2026-09-29-tinyperf-14-a-serving-simulator %})
- Chapter 15: [What a decode step is made of]({% post_url 2026-09-29-tinyperf-15-what-a-decode-step-is-made-of %})
- Chapter 16: [The mixed step]({% post_url 2026-09-29-tinyperf-16-the-mixed-step %})
- Chapter 17: [Memory under load]({% post_url 2026-09-29-tinyperf-17-memory-under-load %})
- Chapter 18: [The knee]({% post_url 2026-09-29-tinyperf-18-the-knee %})
- Chapter 19: [Speculative decoding]({% post_url 2026-09-29-tinyperf-19-speculative-decoding %})
- Chapter 20: [Disaggregated prefill and decode]({% post_url 2026-09-29-tinyperf-20-disaggregated-serving %})

## Part V: Knowing it's right

- Chapter 21: [Validating a performance model]({% post_url 2026-09-29-tinyperf-21-validating-a-performance-model %})
- Chapter 22: [Case study: one constant, three hidden mechanisms]({% post_url 2026-09-29-tinyperf-22-case-study %})

## Appendices

- Appendix a: [Using the estimator]({% post_url 2026-09-29-tinyperf-appendix-a-using-the-estimator %})
- Appendix b: [Glossary and conventions]({% post_url 2026-09-29-tinyperf-appendix-b-glossary %})

[Source code](https://github.com/kaix-nv/tinyperf)
