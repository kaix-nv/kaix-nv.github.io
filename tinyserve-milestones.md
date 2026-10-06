---
layout: page
title: Tinyserve milestone archive
permalink: /series/tinyserve/milestones/
---

These 48 posts preserve the development history, including experiments,
failed gates, and intermediate designs. Milestone numbers are not the book's
reading order. For a connected explanation, start with the
[18-chapter book](/series/tinyserve/).

Existing article URLs and fragment links remain unchanged. A milestone can
contribute to more than one chapter, so the archive does not automatically
redirect readers to a single replacement.

<ul class="post-list">
{% for slug in site.data.tinyserve_order %}
  {% assign post = site.posts | where: "slug", slug | first %}
  <li>
    <h3><a class="post-link" href="{{ post.url | relative_url }}">{{ post.title | escape }}</a></h3>
    <p>{{ post.excerpt | strip_html | normalize_whitespace }}</p>
  </li>
{% endfor %}
</ul>
