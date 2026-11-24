# Adapting the content

## A. Update or replace a source file

Drop the new file into `hcs4os/_data/` and point the `data_path` of the relevant
`mapping.py` entry at it. Nothing else changes as long as the format is unchanged —
e.g. refreshing `sea2021.xml` from the Klassifikationsserver is a one-line edit.

If the _code strings_ change shape in the new edition (extra level, different
separator), also revisit that system's `get_prefixes`.

## B. Add a classification system that is a Klassifikationsserver XML

The common case — no loader work needed.

1. Download the "Gliederung mit Erläuterung" XML into `hcs4os/_data/`.
2. Add a system class in `classification_system/systems/ClassificationSystems.py`:

   ```python
   @register("CPA")
   class ClassificationSystemCPA(ClassificationSystem):
       def get_root_categories(self) -> list[dict]:
           return [{"code": c.code, "description": c.description}
                   for c in self.codes if len(c.code) == 2]

       def get_prefixes(self, code: str) -> list[str]:
           return [code[:i] for i in range(2, len(code) + 1)]
   ```

3. Add a `mapping.py` entry pointing at `"CPA"` + `"KLASS_SERVER"` + the file.
4. Sanity-check the hierarchy — this is the step that actually catches mistakes:

   ```python
   cs = get_classification_system("CPA_2026")
   print(len(cs.codes), len(cs.get_root_categories()))
   print(cs.get_code_trace("<a known deep code>"))   # every level, no gaps
   print(cs.get_children("<a known mid-level code>"))
   ```

   An empty `get_children` on a non-leaf, or a trace with missing rungs, means
   `get_prefixes` produces strings that do not exist as codes.

## C. Add a source in a new format

Write a loader in `hcs4os/dataloaders/loaders/`, implementing only `to_records()`
and returning `Code`-shaped dicts:

```python
from ..BaseDataLoader import ClassificationLoader
from ..registry import register

@register("MY_FORMAT")
class MyLoader(ClassificationLoader):
    def to_records(self) -> list[dict]:
        return [{"code": ..., "description": ..., "level": ...,
                 "detailled_description": ..., "details": {...}}, ...]
```

Import it from `hcs4os/dataloaders/__init__.py` so the `@register` decorator runs,
then reference `"MY_FORMAT"` from `mapping.py`.

For one-off or generated content (LLM-rephrased descriptions, a subset, a merged
system), skip the loader entirely: dump a JSON list of `Code`-shaped dicts into
`_data/` and use `"loader_name": "JSON"` — that is what `SEA_2021_REPHRASED` and
`NAF_rev2` do.

## D. Change what the agent reads and says

Prompts live in signature docstrings, tool wording in the agent's `registry.py`. In
increasing order of intrusiveness:

1. **Tool wording** (navigation agent) — edit the `get_children_doc` / `get_code_doc`
   / `get_parent_doc` / `get_root_categories_doc` entries in
   [HierarchicalNavigation/registry.py](https://github.com/adimo20/hcs4os/blob/master/hcs4os/agents/HierarchicalNavigation/registry.py).
   These are what the model sees when deciding which tool to call. The RAG agent has
   the single `search_category_tool_doc`.
2. **Search behaviour and the I/O contract** — edit the `dspy.Signature`. Its
   docstring is the system prompt (descent protocol, tie-breaking rules, residual-code
   policy, target level); its fields are the contract. They are deliberately in one
   file, since renaming an output field changes what the prompt can refer to. Field
   _names_ are part of the prompt, so keep them descriptive.
3. **Retrieval behaviour** — in `RAGAgent` the shortlist length `k` is chosen by the
   model per call and steered by the signature docstring (edit the "use a k large
   enough" guidance there); in `RAGChainOfThought` you pass `k` and `where` yourself.
   `store.index(codes_to_include=[...])` narrows what is in the collection at all.
4. **The embedding backend** — the store's `embedding_function_type` and `model_name`
   are usually the biggest single lever on retrieval quality. A generic multilingual
   MiniLM is a weak retriever for German product text, and a domain- or
   language-matched model is worth more than any prompt edit above. Since the store is
   injected, swapping it is a change at the call site rather than inside a classifier —
   just index into a fresh `collection_name`, because a collection is only meaningful
   with the model that built it.
5. **The query itself** — switch `RAGChainOfThought` for `RRR_RAGChainOfThought`, or
   edit the rephrasing signature in [rag/registry.py](https://github.com/adimo20/hcs4os/blob/master/hcs4os/rag/registry.py). When
   retrieval misses because the item's wording is nothing like the labels, this beats
   every prompt edit above it; `rephrased_query` on the prediction tells you whether
   the rewrite is helping.

## E. Add an agent for another classification

Every classifier is already generic over the classification — the work is a
configuration entry, not new control flow. Keyed by the dataset name that `mapping.py`
uses (e.g. `"COICOP_2018"`, the same string you pass as `classification_name`), add:

- a `dspy.Signature` whose **docstring** describes _that_ tree (its level names, its
  code shape, an example path, the target level) and whose fields are named for the
  new domain, then
- an entry wherever that classifier reads its config, with keys matching the base
  constructor's parameters:

| classifier                                    | signatures live in                   | config lives in                             | keys                                                                                         |
| --------------------------------------------- | ------------------------------------ | ------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `HierarchicalNavigationAgent`                 | `HierarchicalNavigation/registry.py` | same file (`registry`)                      | `signature`, `get_root_categories_doc`, `get_children_doc`, `get_code_doc`, `get_parent_doc` |
| `RAGAgent`                                    | `RAG/registry.py`                    | same file (`registry`)                      | `signature`, `search_category_tool_doc`                                                      |
| `RAGChainOfThought` / `RRR_RAGChainOfThought` | `rag/registry.py`                    | `RAGChainOfThought.CONFIGS` in `rag/RAG.py` | `signature`, `query_field`, `context_field`                                                  |

For `RRR_RAGChainOfThought` also add a rephrasing signature to `rag/registry.py` and
register it in `QueryRephraser.mapping` — without one, construction fails the
assertion `No rephrasing signature for '<key>'`.

The tool methods, indexing and retrieval code need no changes — that is the point of
the uniform `ClassificationSystem` interface. Widen the `Literal` type hint on the
subclass's `classification_name` too; it is not enforced, but it is what readers trust.

## F. Try a prompt without touching the registry

Instantiate the `Base…` class directly and pass the prose in:

```python
from hcs4os.agents.RAG.BaseAgentRAG import BaseRAGAgent
from hcs4os.classification_system import get_classification_system
from hcs4os._vector_database.vector_store import VectorStore

wz = get_classification_system("WZ_2008")

store = VectorStore(
    collection_name="wz_level3",
    chromadb_path="./data/chroma",
    embedding_function_type="sentence_transformers",
    model_name="deutsche-telekom/gbert-large-paraphrase-cosine",
)
# fill the store yourself — restrict it to the level you predict
store.index(classification_system=wz,
            codes_to_include=[c.code for c in wz.codes if c.level == 3])

agent = BaseRAGAgent(
    classification_name="WZ_2008",        # any key in mapping.py
    signature=MyWzSignature,              # its __doc__ is the system prompt
    search_category_tool_doc="...",
    store=store,
    api_key=..., model_name=...,
)
```

This is the path for classifications the registries do not cover yet, and for
A/B-testing wording. Promote a prompt into the registry once it is worth keeping.

## G. Improve the descriptions the retriever sees

If inner nodes of a classification have thin labels, generate better ones rather than
fighting it with prompts:

```python
from hcs4os.taxonomy_refinement.TaxonomyRefinement import ClassificationsSystemAugmenter

aug = ClassificationsSystemAugmenter("SEA_2021", api_key=..., model=...)
aug.enrich_classification_system(start_level=5)   # walks upward from level 5
```

Then dump `_lookup` to JSON in `_data/`, add a `mapping.py` entry with
`"loader_name": "JSON"`, and record the model and date under `metadata` — the
`SEA_2021` / `SEA_2021_REPHRASED` pair is the worked example. Keeping both entries is
what lets you measure whether the augmentation helped.
