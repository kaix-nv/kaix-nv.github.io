---
layout: page
title: Building tinyperf
permalink: /series/tinyperf/
---

Build an analytical GPU performance model from scratch — the kind GPU
vendors use to predict workload performance on hardware that doesn't exist
yet. Each milestone adds one modeling mechanism, validates it against
public datasheet anchors, and ships as its own pull request.

{% assign tinyperf_posts = site.categories.tinyperf %}
<ul class="post-list">
  {% for post in tinyperf_posts reversed %}
    <li>
      <h3><a class="post-link" href="{{ post.url | relative_url }}">{{ post.title | escape }}</a></h3>
      <p>{{ post.excerpt | strip_html | normalize_whitespace }}</p>
    </li>
  {% endfor %}
</ul>

[Source code](https://github.com/kaix-nv/tinyperf)
