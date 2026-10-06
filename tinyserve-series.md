---
layout: page
title: Building tinyserve
permalink: /series/tinyserve/
---

A book in 18 chapters: build a minimal, complete modern LLM serving engine
to understand how it works. Start with a generated token and the tools to
reason about its cost, then develop model architecture, the serving runtime,
execution optimization, and distributed serving.

Each chapter follows a concrete example through the mechanism, implementation,
and evidence. Attention and feed-forward networks/MoE are model architecture;
memory ownership and scheduling belong to the runtime. Performance analysis
accompanies both. This is an educational implementation, not a production
deployment guide.

[Start with Chapter 1]({% post_url 2026-10-06-tinyserve-01-first-token %}) ·
[Milestone archive](/series/tinyserve/milestones/)

{% for part in site.data.tinyserve_book.parts %}
<h2 id="{{ part.id }}">Part {{ part.number }}: {{ part.title | escape }}</h2>
<ol start="{% assign part_chapters = site.data.tinyserve_book.chapters | where: 'part', part.id %}{{ part_chapters.first.number }}">
  {% for chapter in part_chapters %}
    {% assign chapter_post = site.posts | where: "slug", chapter.slug | first %}
  <li>
    <p><a href="{{ chapter_post.url | relative_url }}">{{ chapter.title | escape }}</a><br>
    {{ chapter.question | escape }}</p>
  </li>
  {% endfor %}
</ol>
{% endfor %}

## How to read the evidence

Sparse attention is a background and design chapter, not a claim of a
qualified Tinyserve sparse backend. Other chapters distinguish supported
paths, opt-in experiments, and failed performance or numerical gates.
Historical benchmark results retain their original model, workload, and
hardware boundaries; this editorial rewrite does not introduce new timings.

The [milestone archive](/series/tinyserve/milestones/) keeps the original
48 development posts and URLs, including detailed experiments and negative
results. Read the chapters for the concepts and the archive for their history.

[Source code](https://github.com/kaix-nv/tinyserve) is currently in an
access-controlled repository. The chapters include the examples and diagrams
needed to follow the explanations without repository access.
