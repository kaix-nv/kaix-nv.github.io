---
layout: page
title: Building tinyserve
permalink: /series/tinyserve/
---

Build a minimal, complete modern LLM serving engine from scratch, one measured
milestone at a time.

{% assign tinyserve_posts = site.categories.tinyserve %}
<ul class="post-list">
  {% for post in tinyserve_posts reversed %}
    <li>
      <h3><a class="post-link" href="{{ post.url | relative_url }}">{{ post.title | escape }}</a></h3>
      <p>{{ post.excerpt | strip_html | normalize_whitespace }}</p>
    </li>
  {% endfor %}
</ul>

[Source code](https://github.com/kaix-nv/tinyserve)
