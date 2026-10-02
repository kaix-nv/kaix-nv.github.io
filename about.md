---
layout: page
title: About
permalink: /about/
---

These are engineering notes from building systems to understand how they
work. Articles are grouped into focused series, with an emphasis on first
principles, minimal implementations, correctness, and reproducible
measurements.

There are two series so far:

- [**Building tinyperf**](/series/tinyperf/), a book in 22 chapters and two
  appendices: an analytical GPU performance model for LLMs, built from
  scratch. It prices kernels from a GPU's datasheet, then models, parallel
  layouts and a serving engine under load, and checks each mechanism
  against a real GPU, with predictions written down before the
  measurements they are compared with.
- [**Building tinyserve**](/series/tinyserve/), an educational LLM inference
  engine developed one milestone at a time: model execution, KV caching,
  batching, paged memory, scheduling, streaming, and modern serving
  optimizations.

Between series, standalone notes take up a single topic, such as
[linear attention from first principles]({% post_url 2026-09-06-linear-attention-and-replayssm %}).

The code for both series is on GitHub
([tinyperf](https://github.com/kaix-nv/tinyperf),
[tinyserve](https://github.com/kaix-nv/tinyserve)).
