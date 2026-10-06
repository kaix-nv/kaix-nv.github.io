---
layout: page
title: Building tinyserve
permalink: /series/tinyserve/
---

Build a minimal, complete modern LLM serving engine from scratch, one measured
milestone at a time.

The series follows milestone order. A completed investigation can report a
failed optimization; each article states its own correctness and performance
boundary.

<ul class="post-list">
  {% for slug in site.data.tinyserve_order %}
    {% assign post = site.posts | where: "slug", slug | first %}
    {% if post %}
    <li>
      <h3><a class="post-link" href="{{ post.url | relative_url }}">{{ post.title | escape }}</a></h3>
      <p>{{ post.excerpt | strip_html | normalize_whitespace }}</p>
    </li>
    {% endif %}
  {% endfor %}
</ul>

[Source code](https://github.com/kaix-nv/tinyserve)
