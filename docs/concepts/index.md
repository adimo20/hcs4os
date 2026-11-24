# How the components work

The library is five layers, each one replaceable on its own:

```
  _data/*.xml|xlsx|csv|json          raw published documentation
            │        ▲
            ▼        │
  dataloaders/  ── to_records() ──►  list[dict] ──► list[Code]      format layer
            │        │                                              (how to parse)
            ▼        │
  classification_system/ ── get_prefixes() ──►  hierarchy           structure layer
            │        │                                              (how codes nest)
            │        └── taxonomy_refinement/ ── writes back a richer dataset
            ▼
  _vector_database/ ── index() / search_category() ──► candidates   retrieval layer
            │                                          (how codes are found)
            ▼
  agents/  ReAct over tools        ─┐
  rag/     retrieve + CoT           ├──►  code + explanation        reasoning layer
  rag/     rewrite + retrieve + CoT ┘
```

Wiring between the layers happens in one table: `classification_system/mapping.py`.
The retrieval layer is skipped entirely by the hierarchy-navigating agent, which
reaches the classification system directly through its four tools.
