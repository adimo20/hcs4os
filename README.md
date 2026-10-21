# hcs4os — Hierarchical Classification Systems for Official Statistics

`hcs4os` turns the published documentation of official statistical classifications
(COICOP, SEA 2021, WZ 2008, KldB 2010, ICATUS 2016, GP, NST, …) into a uniform,
navigable Python object, and puts LLM agents on top of it that classify a free-text
item against the real classification instead of recalling a code from memory.

Four classifier styles solve that same task from different directions:

- **`HierarchicalNavigationAgent`** (`dspy.ReAct`) walks the tree top-down with tools
  — roots, children, parent, code — one level at a time.
- **`RAGAgent`** (`dspy.ReAct`) embeds every code once into a vector store and gives
  the model a single `search_category` tool to pull candidates by semantic similarity
  and reason its way to one of them, re-querying when the notes point elsewhere.
- **`RAGChainOfThought`** (`dspy.ChainOfThought`) does one retrieval and one reasoning
  pass over the candidates — no loop, no tool calls, the cheap baseline.
- **`RRR_RAGChainOfThought`** adds a Rewrite step in front of that baseline: a small
  model first restates the raw item in the vocabulary of the classification's labels,
  and only that rewritten query is embedded.

All four end at the same `ClassificationSystem` and emit the same output fields
(`<system>_code` + `explaination`), which is what makes them comparable on the same
data. Each is a thin subclass of a `Base…` class that takes its prompts and signature
as constructor arguments; the subclass only supplies a registered configuration. The
base variants work with any classification in `mapping.py`, and with custom ones.

Alongside them, **`ClassificationsSystemAugmenter`** works on the classification
itself rather than on items: it writes the missing prose descriptions for parent
categories from their children, and the result is saved back into `_data/` as a new
dataset.

Two things follow from the design:

- Every system — regardless of whether its source is an XML file from the German
  Klassifikationsserver, a Eurostat `.xlsx`, a UN `.csv`, or a hand-written JSON —
  is exposed through the same four operations: `get_root_categories`,
  `get_children`, `get_code`, `get_parent`.
- Those four operations are exactly what the agents build on. Adding a
  classification system automatically makes it agent-ready.

---

## Installation

Installation via pip:

```bash
pip install git+https://github.com/adimo20/hcs4os.git
```

Requires Python ≥ 3.11. Runtime dependencies: `pandas`, `openpyxl`, `dspy`,
`chromadb` and `sentence-transformers` (the last two only matter for the
retrieval-based classifiers). The classification source files ship inside the package
(`hcs4os/_data/`), so no classification data is downloaded at runtime — the embedding
model is pulled from Hugging Face on first use and cached.

---

## Quick start

### Navigating a classification system

```python
from hcs4os.classification_system import get_classification_system

sea = get_classification_system("SEA_2021")

sea.get_root_categories()[:2]
# [{'code': '00', 'description': 'Einnahmen der privaten Haushalte'},
#  {'code': '01', 'description': 'Nahrungsmittel und alkoholfreie Getränke'}]

sea.get_children("0111")
# [Code(code='0111 1', description='Getreide', level=4, ...), ...]

sea.get_code("0111 1").to_dict()
sea.get_parent("0111 1")          # -> dict of the parent code
sea.get_code_trace("0111 1")
# [('01', 'Nahrungsmittel und alkoholfreie Getränke'),
#  ('011', 'Nahrungsmittel'),
#  ('0111', 'Getreide und Getreideerzeugnisse'),
#  ('0111 1', 'Getreide')]
```

### Classifying by walking the hierarchy

```python
import os
from hcs4os.agents.HierarchicalNavigation import HierarchicalNavigationAgent

agent = HierarchicalNavigationAgent(
    classification_name="COICOP_2018",
    api_key=os.getenv("MISTRAL_API_KEY"),
    model_name="mistral/mistral-small-latest",
)

result = agent.agent(input_expense="Basmati rice 1kg bag")
print(result.coicop_code, result.explaination)
```

> Note: `HierarchicalNavigationAgent` subclasses `dspy.Module` but does not define
> `forward()`, so call the inner `.agent` as shown rather than the module itself.

### Classifying by retrieval

```python
import os
from hcs4os.agents.RAG.RAGAgent import RAGAgent

agent = RAGAgent(
    classification_name="COICOP_2018",
    embedding_model_name="sentence-transformers/all-MiniLM-L12-v2",
    collection_name="coicop_full",
    api_key=os.getenv("MISTRAL_API_KEY"),
    model_name="mistral/mistral-small-latest",
    chromadb_path="./data/chroma",         # where the persistent collection lives
    create_new_collection=True,            # index the classification on construction
)

result = agent.agent(input_expense="Basmati rice 1kg bag")   # input_activity for ICATUS
print(result.coicop_code, result.explaination)               # sea_code for SEA
```

> Note: like `HierarchicalNavigationAgent`, `RAGAgent` subclasses `dspy.Module` but
> does not define `forward()`, so call the inner `.agent` as shown rather than the
> module itself.

The first call embeds the classification and persists it under `chromadb_path`
(`./data/chroma` by default); later runs reuse the collection (pass
`create_new_collection=False` to skip re-indexing).

### Classifying by retrieval, without the loop

```python
from hcs4os.rag.RAG import RAGChainOfThought

rag = RAGChainOfThought(
    classification_name="SEA_2021",
    embedding_model_name="deutsche-telekom/gbert-large-paraphrase-cosine",
    collection_name="sea_2021",
    api_key=os.getenv("MISTRAL_API_KEY"),
    model_name="mistral/mistral-small-latest",
)

result = rag(query="frisches Brot vom Bäcker", k=10)
print(result.sea_code, result.explaination)

# restrict the candidates to one level of the hierarchy
result = rag(query="Beitrag zur Krankenversicherung", k=10, where={"level": 5})
```

This one _does_ define `forward()`, so it is called directly. `k` is an argument
rather than something the model decides, and `where` is passed through to chroma as a
metadata filter — `{"level": 5}` is the useful case, since it forces candidates to the
depth actually being predicted.

### Classifying with a rewrite step (RRR)

```python
from hcs4os.rag.RAG import RRR_RAGChainOfThought

rrr = RRR_RAGChainOfThought(          # same arguments as RAGChainOfThought
    classification_name="COICOP_2018",
    embedding_model_name="sentence-transformers/all-MiniLM-L12-v2",
    collection_name="coicop_full",
    api_key=os.getenv("MISTRAL_API_KEY"),
    model_name="mistral/mistral-small-latest",
)

result = rrr(query="BASMATI REIS 1KG BTL", k=10)
print(result.rephrased_query)   # 'rice, milled' — what was actually embedded
print(result.coicop_code, result.explaination)
```

Drop-in replacement for `RAGChainOfThought`: identical constructor, identical
`forward(query, k, where=None)`, one extra field on the prediction
(`rephrased_query`) recording what the retrieval actually searched for.

The model string of every classifier is passed straight to `dspy.LM`, i.e. any
LiteLLM-supported provider works (`openai/…`, `anthropic/…`, `ollama_chat/…`); pass
`api_base=` for self-hosted endpoints.

---

## How the components work

The library is four layers, each one replaceable on its own:

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
  agents/  ReAct over tools        ─┐
  rag/     retrieve + CoT           ├──►  code + explanation        reasoning layer
  rag/     rewrite + retrieve + CoT ┘
```

Wiring between the layers happens in one table: `classification_system/mapping.py`.

### 1. `Code` — the common currency ([Code.py](hcs4os/_shared/Code.py))

A dataclass every source is normalised into:

| field                   | meaning                                                          |
| ----------------------- | ---------------------------------------------------------------- |
| `code`                  | the code string, e.g. `"01.1.1.1"`, `"0111 1"`                   |
| `description`           | short label, e.g. `"Cereals (ND)"`                               |
| `level`                 | depth in the hierarchy                                           |
| `detailled_description` | long prose definition (COICOP intro notes, ICATUS definition, …) |
| `details`               | free-form dict: `includes` / `excludes` / `keywords` / `context` |

`details` is deliberately unconstrained — each classification publishes different
kinds of notes, and the agents just read whatever is there. `Code.from_dict()`
silently drops unknown keys, so a loader may emit extra columns without breaking.

### 2. Data loaders — _how to read a file_ ([dataloaders/](hcs4os/dataloaders))

`ClassificationLoader` (in [BaseDataLoader.py](hcs4os/dataloaders/BaseDataLoader.py))
is a template: it checks the path exists and converts records to `Code` objects.
Subclasses implement exactly one method, `to_records() -> list[dict]`.

Each loader registers itself under a string ID with the `@register` decorator, and
`get_loader(name, path)` resolves it. Currently registered:

| ID             | Source format                         | Notes                                                                            |
| -------------- | ------------------------------------- | -------------------------------------------------------------------------------- |
| `KLASS_SERVER` | Claset XML (klassifikationsserver.de) | The workhorse — parses labels, keywords, inclusions/exclusions; drops empty keys |
| `COICOP`       | UN COICOP 2018 `.xlsx`                | Expects columns `code, title, intro, includes, alsoIncludes, excludes`           |
| `ICATUS`       | UNSD Time-Use `.csv`                  | Flattens the three code columns `PID/ID/CID` into one code                       |
| `NACE`         | Eurostat NACE Rev. 2.1 `.xlsx`        | Registered but not yet referenced from `mapping.py`                              |
| `JSON`         | pre-built list of `Code`-shaped dicts | Used for the LLM-rephrased variants; the escape hatch for any custom pipeline    |

The `KLASS_SERVER` loader is the one to reuse: every German classification
downloadable from the Klassifikationsserver in "Gliederung mit Erläuterung" format
parses without a single new line of code.

### 3. Classification systems — _how codes nest_ ([classification_system/](hcs4os/classification_system))

`ClassificationSystem` (in
[BaseClassificationSystem.py](hcs4os/classification_system/BaseClassificationSystem.py))
provides all navigation for free. On construction it loads the codes and builds two
indices:

- `_lookup` — `code → Code`
- `_children_register` — `code → list[Code]`, computed by testing every pair with
  `_is_child()`

The whole hierarchy is derived from **one abstract method**:

```python
def get_prefixes(self, code: str) -> list[str]: ...
```

It returns the chain of ancestors of a code, ending with the code itself. `_is_child`
then simply says: _parent is the second-to-last prefix of the child_. Get this right
and `get_children`, `get_parent`, and `get_code_trace` all follow.

Subclasses also implement `get_root_categories()` — the entry point the agents start
from — which is usually just "all codes whose string has length 1 or 2".

Why this indirection: code strings look regular but are not. Compare the
implementations in
[ClassificationSystems.py](hcs4os/classification_system/systems/ClassificationSystems.py):

- **COICOP** is dot-separated → split on `"."` (`01` → `01.1` → `01.1.1`).
- **ICATUS / KldB / VUL** are pure prefix codes → every string prefix is an ancestor.
- **SEA 2021** is a prefix code _with a formatting quirk_: from digit 5 on, a space
  is inserted (`0111` → `0111 1`), so `get_prefixes` re-inserts it.
- **NAF rev. 2** (French) inserts a dot after the 2nd digit instead (`43` → `43.3` →
  `43.34`), so `get_prefixes` strips the dot and re-inserts it at a fixed position.
- **WZ 2008** carries trailing dots that must be stripped, and de-duplicated.
- **GP** jumps levels — only lengths 2, 3, 4, 6, 7, 9, 10, 11 are real codes.
- **NST** has only two meaningful levels.

`SEA_NS` is the same SEA hierarchy without the space quirk. It is registered but no
longer referenced from `mapping.py` — the JSON-sourced SEA variant now keeps the
official spacing and uses the plain `SEA` class.

Systems register under a string ID the same way loaders do.

### 4. `mapping.py` — the wiring table

[mapping.py](hcs4os/classification_system/mapping.py) is the only place the three
layers meet. One entry per _available dataset_:

```python
"SEA_2021": {
    "classification_system": "SEA",             # @register ID in systems/
    "loader_name":           "KLASS_SERVER",    # @register ID in dataloaders/
    "data_path":             make_path("sea2021.xml"),
    "metadata": {"url": "https://klassifikationsserver.de/..."},
},
```

`get_classification_system("SEA_2021")` looks the key up here, builds the loader,
and hands it to the system class. Note the deliberate split: `classification_system`
and `loader_name` are independent, so the same hierarchy logic can be fed from a
different source. That is exactly what the `_REPHRASED` pairs do — `SEA_2021` parses
the official XML, `SEA_2021_REPHRASED` loads a JSON variant of the same hierarchy
whose descriptions were generated by `taxonomy_refinement/`, and both run through the
same `SEA` class. For generated datasets `metadata` carries the provenance:

```python
"metadata": {
    "url":   "https://klassifikationsserver.de/klassService/thyme/variant/sea_2021",
    "model": "mistral/zai-glm-5-2",
    "date":  "15.09.2026",
},
```

Currently mapped keys:

`NAF_rev2`, `ICATUS_2016`, `SEA_2021`, `SEA_2021_REPHRASED`, `COICOP_2018`,
`WZ_2008`, `KLDB_2010`, `KLDB_2010_REPHRASED`, `EAV`, `VUL`, `NST`, `GP`

### 5. ReAct agents ([agents/](hcs4os/agents))

Both agents follow the same layout: a `Base…` class holding all the machinery, and a
thin subclass that supplies nothing but a configuration. The subclass is the whole
file:

```python
class RAGAgent(BaseRAGAgent):
    CONFIGS = registry                     # from ./registry.py

    def __init__(self, classification_name, **kwargs):
        if classification_name not in self.CONFIGS:
            raise ValueError(f"Unknown classification system: {classification_name}")
        super().__init__(classification_name=classification_name,
                         **self.CONFIGS[classification_name], **kwargs)
```

So the registry entry is literally a bundle of keyword arguments for the base
constructor, and its keys are the base's parameter names — `signature` plus
`get_root_categories_doc` / `get_children_doc` / `get_code_doc` / `get_parent_doc`
for the navigation agent, `signature` plus `search_category_tool_doc` for the RAG one.
Rename a base parameter and the registry must follow.

There is no longer a `system_prompt` key, and nothing mutates `signature.__doc__` at
construction: **each system prompt lives in its `dspy.Signature` docstring**, where
dspy reads it anyway. Prompt and I/O contract are now edited in one place.

`classification_name` is passed straight through to `get_classification_system`, so
it is the dataset key `mapping.py` uses, and both registries are keyed on those same
strings — `HierarchicalNavigation/registry.py` on `COICOP_2018`, `ICATUS_2016`,
`NAF_rev2`, `SEA_2021` and `SEA_2021_REPHRASED`, `RAG/registry.py` on the same set
minus `NAF_rev2`. A key that is not in the registry raises
`ValueError: Unknown classification system: …` before any model or embedding work
starts.

**[HierarchicalNavigationAgent](hcs4os/agents/HierarchicalNavigation/HierarchicalNavigationAgent.py)** —
`dspy.ReAct` over four thin tool methods that forward to the classification system
and convert `Code` objects to dicts. Its system prompt lays out a descent protocol:
start at the roots, compare every child, _follow the `excludes` pointers instead of
forcing a fit_, verify the leaf, backtrack if excluded, and never produce a code that
did not come from a tool.

**[RAGAgent](hcs4os/agents/RAG/RAGAgent.py)** — retrieve, then choose. Three parts:

- **Indexing.** On construction (when `create_new_collection=True`) `index()` walks
  every `Code` in the classification, embeds its **`description` only** as the
  document, and puts the rest of the record into the chroma metadata under
  `id_<code>`: `code`, `level` and `detailled_description` as they are, and the
  nested `details` dict JSON-dumped into a single string, because chroma accepts only
  flat scalars. So the embedding sees the short label alone, while the long notes ride
  along and reach the model through the tool result. `RAGAgent` indexes every code
  with no level filtering; `BaseRAGAgent` can narrow that with `codes_to_include`.
- **[vector_database.py](hcs4os/agents/RAG/vector_database.py)** wraps a persistent
  chroma collection (cosine space, sentence-transformers embeddings). `VectorStore`
  exposes `create_collection` (a batched `add`), and the underlying
  `collection.query` is used directly for search.
- **Retrieval and selection.** The agent is a `dspy.ReAct` over one tool,
  `search_category(query, k)`, which embeds the query, pulls the `k` nearest codes,
  and returns each one's flattened metadata with its `description` joined back on.
  The signature docstring drives the loop: search with a focused query, read the
  includes/excludes notes on the candidates, refine the query and search again when
  the notes point elsewhere, and never emit a code the tool did not return.

`RAGAgent` does not define `forward()`, so it is called as `agent.agent(...)`. Its
signature outputs are the domain code (`coicop_code` / `icatus_code` / `sea_code`) and
`explaination`.

Indexing runs whenever `create_new_collection=True`; pass `False` to reuse an
already-populated collection. Because the ids are deterministic (`id_<code>`),
re-indexing the same collection re-sends the same ids — chroma skips them with a
warning rather than duplicating — so it is wasted work, not corruption; point later
runs at `create_new_collection=False` or a fresh `collection_name`.

#### The `Base…` variants

[BaseAgent.py](hcs4os/agents/HierarchicalNavigation/BaseAgent.py) and
[BaseAgentRAG.py](hcs4os/agents/RAG/BaseAgentRAG.py) hold the machinery: the LM, the
classification system, the tool methods, the indexing, and the `dspy.ReAct`. Every
registered agent is one of these with a `CONFIGS` lookup in front, so instantiating a
base class directly is the same code path, just with the prose passed in by hand.

Use them to run a classification the registry does not cover, or to A/B a prompt
without editing shared files; the registry subclasses stay the right home for a prompt
you intend to keep. `classification_name` still goes to `get_classification_system`,
so the classification itself must be in `mapping.py` either way.

`BaseRAGAgent` additionally takes `codes_to_include: list[str] | None` — index only
those codes. Restricting the collection to the level actually being predicted, or to
one branch, is the cheapest available lever on retrieval quality, because it removes
whole classes of wrong answers before the model ever sees them.

### 6. Retrieval without the loop ([rag/](hcs4os/rag))

[RAG.py](hcs4os/rag/RAG.py) holds `RAGChainOfThought`: same indexing and same
`search_category`, but a single `dspy.ChainOfThought` instead of a ReAct loop. It
retrieves once and reasons once.

The differences that matter in practice:

- It defines `forward(query, k, where=None)`, so it is callable directly.
- `k` is set by the caller, not chosen by the model — retrieval depth becomes an
  experimental parameter rather than a model decision.
- `where` is forwarded to chroma as a metadata filter (`{"level": 5}`).
- The candidates are rendered as indented JSON into a `retrieved_candidates` input
  field, and the whole reasoning protocol lives in the signature's own docstring.
- One retrieval, one answer: it cannot recover from a bad shortlist the way the ReAct
  agent can by re-querying. That is the trade for costing a single model call.

[BaseRAG.py](hcs4os/rag/BaseRAG.py) is its configurable twin: `BaseRAGCoT` takes the
`signature` plus `query_field` and `context_field` naming which of its input fields
receive the item and the retrieved candidates, and asserts at construction that both
names really exist on the signature.

`RAGChainOfThought` is the registered subclass, and because the base needs the field
names too, its `CONFIGS` carries three things per classification rather than one:

```python
CONFIGS = {
    "COICOP_2018": {"signature":     CoicopRAGChainOfThoughtSignature,
                    "query_field":   "input_expense",
                    "context_field": "retrieved_candidates"},
    ...   # ICATUS_2016, SEA_2021, SEA_2021_REPHRASED
}
```

#### Rewrite-Retrieve-Read

`RRR_RAGChainOfThought` subclasses `RAGChainOfThought` and puts a rewrite step in
front of retrieval. The motivation is a property of the index: only each code's short
`description` label is embedded — `"Eggplants (aubergines), fresh or chilled"`,
`"Steinobstkonserven, z.B. Pflaumen-, … Pfirsichkonserven"`, `"Eating meals/snack"`.
Real items do not look like that. Receipt shorthand, brand names, bank-statement text
and diary abbreviations sit far from those labels in embedding space, so retrieval
misses before the reasoning model ever gets a chance.

Measured on the COICOP collection, the raw receipt line `"BASMATI REIS 1KG BTL"`
retrieves `01.1.1.1.5` Millet, `12.2.9.1` Remittances fees and `01.1.1.3` Bread —
rice is nowhere in the top 3. Rewritten to `"rice, milled"`, the same store returns
`01.1.1.1.2` Rice first.

The three steps:

1. **Rewrite** — `self.rephrasing_module`, a `QueryRephraser`, restates the item in
   the vocabulary of the labels. It explicitly does **not** classify: the signatures
   forbid emitting a code or choosing between categories.
2. **Retrieve** — the rewritten query is embedded and the `k` nearest codes are pulled
   from the store.
3. **Read** — the inherited `ChainOfThought` classifies the **original** item over
   those candidates. The rewrite only steers retrieval; it never replaces what the
   model has to classify, so detail the rephraser dropped is not lost from the
   decision.

`forward(query, k, where=None)` matches the parent exactly, and the prediction carries
one extra field, `rephrased_query`, so you can see and log what was actually searched
for. That makes `RRR_RAGChainOfThought` a drop-in A/B against `RAGChainOfThought` on
the same collection.

**`QueryRephraser`** is a standalone `dspy.Module` and usable on its own. It picks a
signature from its own `mapping` (`COICOP_2018`, `ICATUS_2016`, `SEA_2021`,
`SEA_2021_REPHRASED`), runs `dspy.Predict`, and hides which input field that signature
uses — callers always pass `query=`:

```python
from hcs4os.rag.RAG import QueryRephraser

rephraser = QueryRephraser(lm=cheap_lm, classification_name="SEA_2021")
rephraser(query="LASTSCHRIFT ALLIANZ KFZ").rephrased_query
# intended shape of the output: 'Beitrag Kfz-Haftpflichtversicherung'
```

It runs inside `with dspy.context(lm=self.lm)`, so it uses the LM handed to its
constructor regardless of what is globally configured. If the model returns an empty
rewrite, the raw query is used instead.

The rephrasing prompts are the most classification-specific prose in the library,
because what has to survive the rewrite differs per system. COICOP and SEA keep
placement-driving attributes — state or form (fresh, frozen, canned, ready-made), good
vs. service, consumed at home vs. served, new vs. second-hand — and drop brand, shop,
quantity and price. SEA additionally has to pin down the direction of the money flow,
since an insurance premium (Abteilung 16) and a household appliance (05) are different
parts of the system, and it answers in German to match the German labels. ICATUS
instead has to surface the economic nature of the activity and who it was done for,
because its titles are organised by that rather than by what the person did. All three
end with the same guard: do not force a category term you are unsure of, because a
wrong one pulls retrieval into the wrong branch.

Three things to know before building on it:

- The class docstring lists `rephraser_lm` and `retrieve_with_original` parameters.
  **Neither is implemented.** `__init__` only forwards `*args, **kwargs` to
  `RAGChainOfThought`, so passing `rephraser_lm=` raises
  `TypeError: BaseRAGCoT.__init__() got an unexpected keyword argument`, and
  `forward` retrieves with the rewritten query alone — there is no merge with the raw
  query's hits. To use a cheaper rewrite model today, assign
  `rrr.rephrasing_module.lm = my_lm` after construction.
- Retrieval depends entirely on one rewrite. A bad rewrite is unrecoverable, where
  plain `RAGChainOfThought` at least embeds the item's own words. This is the risk
  `retrieve_with_original` was meant to hedge.
- `QueryRephraser.forward` ends in a bare `print(rephrased)`, so every call writes to
  stdout. Noisy in batch evaluation; `rephrased_query` on the prediction carries the
  same value.

### 7. Taxonomy refinement ([taxonomy_refinement/](hcs4os/taxonomy_refinement))

Published classifications document their leaves well and their inner nodes barely at
all — a division often has nothing but a two-word label. That hurts retrieval most,
because a sparse label is a weak embedding.

`ClassificationsSystemAugmenter` fills those gaps. `enrich_classification_system(start_level)`
walks the hierarchy upward from `start_level`, and for each parent asks the model for
a prose description given its own label plus the labels of its direct children — on
the premise that a parent's meaning is the union of what its children cover. The
`GenerateParentDescription` signature pins that down: characterise the category as a
whole rather than concatenating the children, stay within what the labels support,
and answer in the language of the input.

Two things to know before using it:

- It writes into `classification_system._lookup` only. `codes` and
  `_children_register` still hold the old objects, so re-read results through
  `get_code`, not by iterating `codes`.
- The result is meant to be dumped to JSON in `_data/` and registered as its own
  `mapping.py` entry with a `JSON` loader — that is how `SEA_2021_REPHRASED` and
  `sea_2021_augmented.json` came to exist. Keep the generated dataset beside the
  original rather than replacing it, so the two remain comparable.

---

## Adapting the content

### A. Update or replace a source file

Drop the new file into `hcs4os/_data/` and point the `data_path` of the relevant
`mapping.py` entry at it. Nothing else changes as long as the format is unchanged —
e.g. refreshing `sea2021.xml` from the Klassifikationsserver is a one-line edit.

If the _code strings_ change shape in the new edition (extra level, different
separator), also revisit that system's `get_prefixes`.

### B. Add a classification system that is a Klassifikationsserver XML

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

### C. Add a source in a new format

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
`KLDB_2010_REPHRASED` do.

### D. Change what the agent reads and says

Prompts live in signature docstrings, tool wording in the agent's `registry.py`. In
increasing order of intrusiveness:

1. **Tool wording** (navigation agent) — edit the `get_children_doc` / `get_code_doc`
   / `get_parent_doc` / `get_root_categories_doc` entries in
   [HierarchicalNavigation/registry.py](hcs4os/agents/HierarchicalNavigation/registry.py).
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
   `codes_to_include` (on `BaseRAGAgent` / `BaseRAGCoT`) narrows what is indexed at
   all. `embedding_model_name` is usually the biggest single lever on retrieval
   quality — a generic multilingual MiniLM is a weak retriever for German product
   text, and a domain- or language-matched model is worth more than any prompt edit.
   Changing it means indexing into a fresh `collection_name`, since a collection is
   only meaningful with the model that built it.
4. **The query itself** — switch `RAGChainOfThought` for `RRR_RAGChainOfThought`, or
   edit the rephrasing signature in [rag/RAG.py](hcs4os/rag/RAG.py). When retrieval
   misses because the item's wording is nothing like the labels, this beats every
   prompt edit above it; `rephrased_query` on the prediction tells you whether the
   rewrite is helping.

### E. Add an agent for another classification

Every classifier is already generic over the classification — the work is a
configuration entry, not new control flow. Keyed by the dataset name that `mapping.py`
uses (e.g. `"COICOP_2018"`, the same string you pass as `classification_name`), add:

- a `dspy.Signature` whose **docstring** describes _that_ tree (its level names, its
  code shape, an example path, the target level) and whose fields are named for the
  new domain, then
- an entry wherever that classifier reads its config, with keys matching the base
  constructor's parameters:

| classifier                                  | config lives in                           | keys                                                                                     |
| ------------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------- |
| `HierarchicalNavigationAgent`               | `HierarchicalNavigation/registry.py`       | `signature`, `get_root_categories_doc`, `get_children_doc`, `get_code_doc`, `get_parent_doc` |
| `RAGAgent`                                  | `RAG/registry.py`                          | `signature`, `search_category_tool_doc`                                                  |
| `RAGChainOfThought` / `RRR_RAGChainOfThought` | `RAGChainOfThought.CONFIGS` in `rag/RAG.py` | `signature`, `query_field`, `context_field`                                              |

For `RRR_RAGChainOfThought` also add a rephrasing signature to `QueryRephraser`'s
`mapping` in the same file — without one, construction fails the assertion
`No rephrasing signature for '<key>'`.

The tool methods, indexing and retrieval code need no changes — that is the point of
the uniform `ClassificationSystem` interface. Widen the `Literal` type hint on the
subclass's `classification_name` too; it is not enforced, but it is what readers trust.

### F. Try a prompt without touching the registry

Instantiate the `Base…` class directly and pass the prose in:

```python
from hcs4os.agents.RAG.BaseAgentRAG import BaseRAGAgent
from hcs4os.classification_system import get_classification_system

wz = get_classification_system("WZ_2008")

agent = BaseRAGAgent(
    classification_name="WZ_2008",        # any key in mapping.py
    signature=MyWzSignature,              # its __doc__ is the system prompt
    search_category_tool_doc="...",
    codes_to_include=[c.code for c in wz.codes if c.level == 3],
    embedding_model_name="deutsche-telekom/gbert-large-paraphrase-cosine",
    collection_name="wz_level3",
    api_key=..., model_name=...,
)
```

This is the path for classifications the registries do not cover yet, and for
A/B-testing wording. Promote a prompt into the registry once it is worth keeping.

### G. Improve the descriptions the retriever sees

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

## License

Copyright (c) 2026 Adrian Montag
Licensed under the EUPL v. 1.2
