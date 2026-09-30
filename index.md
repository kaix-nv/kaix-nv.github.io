---
layout: default
---

Notes from building systems to understand how they work. Each series
builds one system from first principles, in code you can run, checked for
correctness and measured on real hardware, mistakes included.

## Series

### [Building tinyperf: an analytical GPU performance model](/series/tinyperf/)

A book in 22 chapters and two appendices. Price a GEMM from a GPU's
datasheet, then a transformer, a parallel layout and a serving engine under
load; check each mechanism against a real GPU, and label every number by
what the model had seen before it was measured.

[Start with chapter 1]({% post_url 2026-09-29-tinyperf-01-what-a-performance-model-is-for %}) ·
[Contents](/series/tinyperf/) ·
[Source code](https://github.com/kaix-nv/tinyperf)

### [Building tinyserve: an LLM serving engine](/series/tinyserve/)

A minimal, complete modern LLM serving engine, one milestone at a time.
Each milestone begins with a measured limitation, adds one serving
technique, proves the outputs stay correct and measures the result.

[Start with M0]({% post_url 2026-07-14-building-tinyserve-m0 %}) ·
[All posts](/series/tinyserve/) ·
[Source code](https://github.com/kaix-nv/tinyserve)

## Latest notes

{% assign date_format = site.minima.date_format | default: "%b %-d, %Y" -%}
{%- assign shown = 0 -%}
{%- assign book_listed = false -%}
<ul class="post-list">
{%- for post in site.posts -%}
  {%- if shown >= 6 %}{% break %}{% endif -%}
  {%- if post.categories contains "tinyperf" -%}
    {%- unless book_listed %}
  <li>
    <span class="post-meta">{{ post.date | date: date_format }}</span>
    <h3><a class="post-link" href="{{ '/series/tinyperf/' | relative_url }}">Building tinyperf, the book: 22 chapters and two appendices</a></h3>
    <p>The milestone posts, rewritten as a book: each chapter asks one question, builds the mechanism that answers it, shows the code and the evidence, and says where it breaks. Links to the old posts lead to the chapters that replace them.</p>
  </li>
      {%- assign book_listed = true -%}
      {%- assign shown = shown | plus: 1 -%}
    {%- endunless -%}
  {%- else %}
  <li>
    <span class="post-meta">{{ post.date | date: date_format }}</span>
    <h3><a class="post-link" href="{{ post.url | relative_url }}">{{ post.title | escape }}</a></h3>
    <p>{{ post.excerpt | strip_html | normalize_whitespace | truncatewords: 40 }}</p>
  </li>
    {%- assign shown = shown | plus: 1 -%}
  {%- endif -%}
{%- endfor %}
</ul>

<p class="rss-subscribe">Subscribe <a href="{{ '/feed.xml' | relative_url }}">via RSS</a>.</p>
