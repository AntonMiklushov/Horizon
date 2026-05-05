---
layout: default
title: "{{ digest.title }}"
date: {{ digest.date }}
lang: {{ digest.language }}
selected_items: {{ digest.items|length }}
total_fetched: {{ digest.total_fetched }}
---

{{ markdown }}
