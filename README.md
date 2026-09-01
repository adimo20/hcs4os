# hcs4os — Hierarchical Classification Systems for Official Statistics

`hcs4os` turns the published documentation of official statistical classifications
(COICOP, SEA 2021, WZ 2008, KldB 2010, ICATUS 2016, GP, NST, …) into a uniform,
navigable Python object, and puts LLM agents on top of it that classify a free-text
item against the real classification instead of recalling a code from memory.

Two agents solve that same task from opposite directions:

* **`HierarchicalNavigationAgent`** walks the tree top-down with tools — roots,
  children, parent, code — one level at a time.
* **`RAGAgent`** embeds every code once into a vector store and gives the model a
  single `search_category` retrieval tool to pull candidates by semantic
  similarity and reason its way to one of them.

Both are `dspy.ReAct` agents that end at the same `ClassificationSystem` and emit the
same output fields (`<system>_code` + `explaination`), which is what makes them
comparable on the same data.

Two things follow from the design:

* Every system — regardless of whether its source is an XML file from the German
  Klassifikationsserver, a Eurostat `.xlsx`, a UN `.csv`, or a hand-written JSON —
  is exposed through the same four operations: `get_root_categories`,
  `get_children`, `get_code`, `get_parent`.
* Those four operations are exactly what the agents build on. Adding a
  classification system automatically makes it agent-ready.

---

## Installation

```bash
pip install -e .
```

```bash
pip install git+https://github.com/adimo20/hcs4os.git
```

Requires Python ≥ 3.11. Runtime dependencies: `pandas`, `openpyxl`, `dspy`,
`chromadb` and `sentence-transformers` (the last two only matter for `RAGAgent`).
The classification source files ship inside the package (`hcs4os/_data/`), so no
classification data is downloaded at runtime — the embedding model for `RAGAgent`
is pulled from Hugging Face on first use and cached.

For local development the repo assumes a venv at `.venv/`:

```bash
source .venv/Scripts/activate
```

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
    classification_name="COICOP",          # or "ICATUS"
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
from hcs4os.agents.RAG import RAGAgent

agent = RAGAgent(
    classification_name="COICOP_2018",     # or "ICATUS_2016" — the dataset/registry key
    embedding_model_name="sentence-transformers/all-MiniLM-L12-v2",
    collection_name="coicop_full",
    api_key=os.getenv("MISTRAL_API_KEY"),
    model_name="mistral/mistral-small-latest",
    chromadb_path="./data/chroma",         # where the persistent collection lives
    create_new_collection=True,            # index the classification on construction
)

result = agent.agent(input_expense="Basmati rice 1kg bag")   # input_activity for ICATUS
print(result.coicop_code, result.explaination)
```

> Note: like `HierarchicalNavigationAgent`, `RAGAgent` subclasses `dspy.Module` but
> does not define `forward()`, so call the inner `.agent` as shown rather than the
> module itself.

The first call embeds the classification and persists it under `chromadb_path`
(`./data/chroma` by default); later runs reuse the collection (pass
`create_new_collection=False` to skip re-indexing).

The model string of either agent is passed straight to `dspy.LM`, i.e. any
LiteLLM-supported provider works (`openai/…`, `anthropic/…`, `ollama_chat/…`); pass
`api_base=` for self-hosted endpoints.

---

## How the components work

The library is four layers, each one replaceable on its own:

```
  _data/*.xml|xlsx|csv|json          raw published documentation
            │
            ▼
  dataloaders/  ── to_records() ──►  list[dict] ──► list[Code]      format layer
            │                                                       (how to parse)
            ▼
  classification_system/ ── get_prefixes() ──►  hierarchy           structure layer
            │                                                       (how codes nest)
            ▼
  agents/       ── dspy.ReAct over 4 tools ──►  code + explanation  reasoning layer
```

Wiring between the layers happens in one table: `classification_system/mapping.py`.

### 1. `Code` — the common currency ([Code.py](hcs4os/_shared/Code.py))

A dataclass every source is normalised into:

| field                   | meaning                                                          |
| ----------------------- | ---------------------------------------------------------------- |
| `code`                  | the code string, e.g. `"01.1.1.1"`, `"0111 1"`                    |
| `description`           | short label, e.g. `"Cereals (ND)"`                                |
| `level`                 | depth in the hierarchy                                            |
| `detailled_description` | long prose definition (COICOP intro notes, ICATUS definition, …)  |
| `details`               | free-form dict: `includes` / `excludes` / `keywords` / `context`  |

`details` is deliberately unconstrained — each classification publishes different
kinds of notes, and the agents just read whatever is there. `Code.from_dict()`
silently drops unknown keys, so a loader may emit extra columns without breaking.

### 2. Data loaders — *how to read a file* ([dataloaders/](hcs4os/dataloaders))

`ClassificationLoader` (in [BaseDataLoader.py](hcs4os/dataloaders/BaseDataLoader.py))
is a template: it checks the path exists and converts records to `Code` objects.
Subclasses implement exactly one method, `to_records() -> list[dict]`.

Each loader registers itself under a string ID with the `@register` decorator, and
`get_loader(name, path)` resolves it. Currently registered:

| ID             | Source format                          | Notes                                                                            |
| -------------- | -------------------------------------- | -------------------------------------------------------------------------------- |
| `KLASS_SERVER` | Claset XML (klassifikationsserver.de)  | The workhorse — parses labels, keywords, inclusions/exclusions; drops empty keys |
| `COICOP`       | UN COICOP 2018 `.xlsx`                 | Expects columns `code, title, intro, includes, alsoIncludes, excludes`           |
| `ICATUS`       | UNSD Time-Use `.csv`                   | Flattens the three code columns `PID/ID/CID` into one code                       |
| `NACE`         | Eurostat NACE Rev. 2.1 `.xlsx`         | Registered but not yet referenced from `mapping.py`                              |
| `JSON`         | pre-built list of `Code`-shaped dicts  | Used for the LLM-rephrased variants; the escape hatch for any custom pipeline    |

The `KLASS_SERVER` loader is the one to reuse: every German classification
downloadable from the Klassifikationsserver in "Gliederung mit Erläuterung" format
parses without a single new line of code.

### 3. Classification systems — *how codes nest* ([classification_system/](hcs4os/classification_system))

`ClassificationSystem` (in
[BaseClassificationSystem.py](hcs4os/classification_system/BaseClassificationSystem.py))
provides all navigation for free. On construction it loads the codes and builds two
indices:

* `_lookup` — `code → Code`
* `_children_register` — `code → list[Code]`, computed by testing every pair with
  `_is_child()`

The whole hierarchy is derived from **one abstract method**:

```python
def get_prefixes(self, code: str) -> list[str]: ...
```

It returns the chain of ancestors of a code, ending with the code itself. `_is_child`
then simply says: *parent is the second-to-last prefix of the child*. Get this right
and `get_children`, `get_parent`, and `get_code_trace` all follow.

Subclasses also implement `get_root_categories()` — the entry point the agents start
from — which is usually just "all codes whose string has length 1 or 2".

Why this indirection: code strings look regular but are not. Compare the
implementations in
[ClassificationSystems.py](hcs4os/classification_system/systems/ClassificationSystems.py):

* **COICOP** is dot-separated → split on `"."` (`01` → `01.1` → `01.1.1`).
* **ICATUS / KldB / VUL** are pure prefix codes → every string prefix is an ancestor.
* **SEA 2021** is a prefix code *with a formatting quirk*: from digit 5 on, a space
  is inserted (`0111` → `0111 1`), so `get_prefixes` re-inserts it.
* **WZ 2008** carries trailing dots that must be stripped, and de-duplicated.
* **GP** jumps levels — only lengths 2, 3, 4, 6, 7, 9, 10, 11 are real codes.
* **NST** has only two meaningful levels.

`SEA_NS` is the same SEA hierarchy without the space quirk, for the JSON-sourced
rephrased variant.

Systems register under a string ID the same way loaders do.

### 4. `mapping.py` — the wiring table

[mapping.py](hcs4os/classification_system/mapping.py) is the only place the three
layers meet. One entry per *available dataset*:

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
different source — that is exactly what `KLDB_2010` vs. `KLDB_2010_REPHRASED` do.

Currently mapped keys:

`ICATUS_2016`, `SEA_2021`, `SEA_2021_REPHRASED`, `COICOP_2018`, `WZ_2008`,
`KLDB_2010`, `KLDB_2010_REPHRASED`, `EAV`, `VUL`, `NST`, `GP`

### 5. Agents ([agents/](hcs4os/agents))

Both agents follow the same layout: a thin `dspy.Module` next to a `registry.py`
holding all the prose (system prompts, signatures, tool descriptions) keyed by
classification name. Prompts live in the registry, not in the agent, so retargeting
an agent is an edit to data rather than to control flow. On construction each agent
looks its `classification_name` up in its own `registry.py`, assigns the matching
`dspy.Signature`, sets that signature's `__doc__` to the `system_prompt`, and builds
a `dspy.ReAct` whose tool descriptions come from the same registry entry.

`classification_name` is passed straight through to `get_classification_system`, so
`RAGAgent` takes the dataset key that `mapping.py` uses — `"COICOP_2018"` or
`"ICATUS_2016"` — and its `RAG/registry.py` is keyed on those same strings.

**[HierarchicalNavigationAgent](hcs4os/agents/HierarchicalNavigation/HierarchicalNavigationAgent.py)** —
`dspy.ReAct` over four thin tool methods that forward to the classification system
and convert `Code` objects to dicts. Its system prompt lays out a descent protocol:
start at the roots, compare every child, *follow the `excludes` pointers instead of
forcing a fit*, verify the leaf, backtrack if excluded, and never produce a code that
did not come from a tool.

**[RAGAgent](hcs4os/agents/RAG/RAGAgent.py)** — retrieve, then choose. Three parts:

* **Indexing.** On construction (when `create_new_collection=True`) `index()` walks
  every `Code` in the classification, embeds its `description` as the document, and
  flattens the rest of the record — `level`, `detailled_description`, and each
  `details` note — into the chroma metadata (nested keys are joined, e.g.
  `details_excludes`). Every code is indexed; there is no per-level filtering.
* **[vector_database.py](hcs4os/agents/RAG/vector_database.py)** wraps a persistent
  chroma collection (cosine space, sentence-transformers embeddings). `VectorStore`
  exposes `create_collection` (a batched `add`), and the underlying
  `collection.query` is used directly for search.
* **Retrieval and selection.** The agent is a `dspy.ReAct` over one tool,
  `search_category(query, k)`, which embeds the query, pulls the `k` nearest codes,
  and returns each one's flattened metadata with its `description` joined back on.
  The `system_prompt` drives the loop: search with a focused query, read the
  includes/excludes notes on the candidates, refine the query and search again when
  the notes point elsewhere, and never emit a code the tool did not return.

`RAGAgent` does not define `forward()`, so it is called as `agent.agent(...)`. Its
signature outputs are the domain code (`coicop_code` / `icatus_code`) and
`explaination`.

Indexing runs whenever `create_new_collection=True`; pass `False` to reuse an
already-populated collection. Because the ids are deterministic (`id_<code>`),
re-indexing the same collection re-sends the same ids — chroma skips them with a
warning rather than duplicating — so it is wasted work, not corruption; point later
runs at `create_new_collection=False` or a fresh `collection_name`.

---

## Adapting the content

### A. Update or replace a source file

Drop the new file into `hcs4os/_data/` and point the `data_path` of the relevant
`mapping.py` entry at it. Nothing else changes as long as the format is unchanged —
e.g. refreshing `sea2021.xml` from the Klassifikationsserver is a one-line edit.

If the *code strings* change shape in the new edition (extra level, different
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

Everything the model reads lives in the agent's `registry.py`, keyed by
classification name. In increasing order of intrusiveness:

1. **Tool wording** (navigation agent) — edit the `get_children` / `get_code` /
   `get_parent` / `get_root_categories` entries in
   [HierarchicalNavigation/registry.py](hcs4os/agents/HierarchicalNavigation/registry.py).
   These are what the model sees when deciding which tool to call.
2. **Search behaviour** — edit the `system_prompt` entry (descent protocol,
   tie-breaking rules, residual-code policy, target level).
3. **Retrieval behaviour** (RAG agent) — the shortlist length `k` is chosen by the
   model per call and steered by the `system_prompt` (edit the "use a k large enough"
   guidance there). `embedding_model_name` is usually the biggest single lever on
   retrieval quality (a generic multilingual MiniLM is a weak retriever for German
   product text; a domain- or language-matched model is worth more than any prompt
   edit); changing it means indexing into a fresh `collection_name`, since a
   collection is only meaningful with the model that built it.
4. **Input/output contract** — edit the `dspy.Signature` to rename fields, or add
   outputs (a confidence score, the runner-up code). Field *names* are part of the
   prompt, so they carry meaning: keep them descriptive.

### E. Add an agent for another classification

Both agents are already generic over the classification — the work is a registry
entry, not new control flow. Add an entry to the relevant agent's `registry.py`,
keyed by the dataset name that `mapping.py` uses (e.g. `"COICOP_2018"`, the same
string you pass as `classification_name`):

* a `dspy.Signature` with input/output fields named for the new domain,
* a `system_prompt` describing *that* tree (its level names, its code shape, an
  example path), and
* the tool descriptions for that agent — the four navigation tools
  (`get_root_categories` / `get_children` / `get_code` / `get_parent`) for the
  hierarchical agent, or the single `search_category` description for the RAG agent.

The tool methods and retrieval code need no changes — that is the point of the
uniform `ClassificationSystem` interface.

---

## Rough edges worth knowing

* In the COICOP block of
  [HierarchicalNavigation/registry.py](hcs4os/agents/HierarchicalNavigation/registry.py),
  the `get_children` and `get_code` descriptions are swapped: each tool is described
  to the model as the other one. The ICATUS block is correct.
* Neither agent defines `forward()`, so both must be called as `agent.agent(...)`,
  not as the module itself.
* `RAGAgent`'s `classification_name` is type-hinted `Literal["COICOP", "ICATUS"]`,
  but the values that actually work are the dataset keys `"COICOP_2018"` /
  `"ICATUS_2016"` — they must resolve in both `mapping.py` (for
  `get_classification_system`) and `RAG/registry.py` (for the signature and prompts).
  The hint is stale; the two agents' registries are also keyed differently
  (`HierarchicalNavigation` on `"COICOP"` / `"ICATUS"`, `RAG` on the dataset keys).
* `_children_register` is built by comparing every code against every other one, so
  construction is O(n²). Fine for COICOP (871 codes) or ICATUS (230); noticeably slow
  for SEA 2021 (2,654), KldB 2010 (2,177) and especially GP 2026 (11,989 codes, a
  multi-minute `get_classification_system` call). Instantiate once and reuse.
* `Code.detailled_description` and the `explaination` output field of every agent
  signature contain typos that are part of the public API surface — renaming them is
  a breaking change.
* Each agent only knows the classifications keyed in its own `registry.py`
  (`RAG/registry.py`: `COICOP_2018`, `ICATUS_2016`), even though the library ships
  eleven. Adding one is a single registry entry, see *Adapting the content → E*.

## License

Copyright (c) 2026 Adrian Montag
Licensed under the EUPL v. 1.2
