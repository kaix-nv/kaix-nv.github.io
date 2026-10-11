---
layout: page
title: Building tinyperf
permalink: /series/tinyperf/
---

A book in 23 chapters on building an analytical GPU performance model
for LLMs from scratch: price a kernel from a GPU's datasheet, then a
model, a parallel layout and a serving engine under load, and check
every step against a real GPU. Each chapter asks one question, builds
the mechanism that answers it, shows the code and the evidence, and
says where it breaks. Every number in it is printed by a script in the
repository and labelled by what the model had seen.

## Part I: Pricing one kernel

- Chapter 1: [What a performance model is for]({% post_url 2026-09-29-tinyperf-01-what-a-performance-model-is-for %})
- Chapter 2: [How tinyperf is designed]({% post_url 2026-09-29-tinyperf-02-how-tinyperf-is-designed %})
- Chapter 3: [A GPU as a handful of rates]({% post_url 2026-09-29-tinyperf-03-a-gpu-as-a-handful-of-rates %})
- Chapter 4: [Pricing a GEMM]({% post_url 2026-09-29-tinyperf-04-pricing-a-gemm %})
- Chapter 5: [Meeting a real GPU: calibration and its tiers]({% post_url 2026-09-29-tinyperf-05-meeting-a-real-gpu %})

## Part II: One model on one GPU

- Chapter 6: [Graphs and the scheduler]({% post_url 2026-09-29-tinyperf-06-graphs-and-the-scheduler %})
- Chapter 7: [A transformer: prefill, decode and memory]({% post_url 2026-09-29-tinyperf-07-a-transformer-prefill-decode-and-memory %})
- Chapter 8: [Attention]({% post_url 2026-09-29-tinyperf-08-attention %})
- Chapter 9: [Attention variants]({% post_url 2026-09-29-tinyperf-09-attention-variants %})
- Chapter 10: [Mixture of experts]({% post_url 2026-09-29-tinyperf-10-mixture-of-experts %})
- Chapter 11: [Precision and sparsity as passes]({% post_url 2026-09-29-tinyperf-11-precision-and-sparsity %})

## Part III: Many GPUs

- Chapter 12: [Collectives and tensor parallelism]({% post_url 2026-09-29-tinyperf-12-collectives-and-tensor-parallelism %})
- Chapter 13: [Parallel layouts]({% post_url 2026-09-29-tinyperf-13-parallel-layouts %})
- Chapter 14: [Training]({% post_url 2026-09-29-tinyperf-14-training %})

## Part IV: Serving

- Chapter 15: [A serving simulator]({% post_url 2026-09-29-tinyperf-15-a-serving-simulator %})
- Chapter 16: [What a decode step is made of]({% post_url 2026-09-29-tinyperf-16-what-a-decode-step-is-made-of %})
- Chapter 17: [The mixed step]({% post_url 2026-09-29-tinyperf-17-the-mixed-step %})
- Chapter 18: [Memory under load]({% post_url 2026-09-29-tinyperf-18-memory-under-load %})
- Chapter 19: [The knee]({% post_url 2026-09-29-tinyperf-19-the-knee %})
- Chapter 20: [Speculative decoding]({% post_url 2026-09-29-tinyperf-20-speculative-decoding %})
- Chapter 21: [Disaggregated prefill and decode]({% post_url 2026-09-29-tinyperf-21-disaggregated-serving %})

## Part V: Knowing it's right

- Chapter 22: [Validating a performance model]({% post_url 2026-09-29-tinyperf-22-validating-a-performance-model %})
- Chapter 23: [Case study: one constant, three hidden mechanisms]({% post_url 2026-09-29-tinyperf-23-case-study %})

## Appendices

- Appendix a: [Using the estimator]({% post_url 2026-09-29-tinyperf-appendix-a-using-the-estimator %})
- Appendix b: [Glossary and conventions]({% post_url 2026-09-29-tinyperf-appendix-b-glossary %})

[Source code](https://github.com/kaix-nv/tinyperf)
