# Bringing your own classification

Everything in the built-in catalogue — COICOP, ICATUS, SEA, … — is wired together
through three small registries and one plain dictionary. All four are ordinary
importable Python objects, so you can register your own classification **from your own
script, against the installed package**, without forking or editing anything inside
`hcs4os`. Once registered, your classification is indistinguishable from a built-in
one: the same `get_classification_system(...)`, the same navigation, the same vector
store.

You supply three things and tie them together with a fourth:

| You write                   | Answers the question                  | Contract                                                   |
| --------------------------- | ------------------------------------- | ---------------------------------------------------------- |
| a **loader**                | _how do I read my file?_              | one method: `to_records() -> list[dict]`                   |
| a **classification system** | _how do my codes nest?_               | one method: `get_prefixes(code)` (+ `get_root_categories`) |
| a **mapping entry**         | _what is this dataset called?_        | a dict tying the loader + system + file together           |
| (`Code`)                    | the uniform record every code becomes | provided — you just emit its fields                        |

The worked example below registers the UN **HS (Harmonized System)** commodity codes
and searches them, end to end.

## 0. Install with an embedding backend

The core install does parsing and navigation only. The vector store is optional and
pulls its embedding backend from an extra — pick the one you'll use:

```bash
pip install "hcs4os[local]"    # sentence-transformers, runs in-process, offline
# or
pip install "hcs4os[ollama]"   # embeddings from a (self-hosted) Ollama server
# or
pip install "hcs4os[openai]"   # embeddings from the OpenAI API
```

## 1. Teach it to read your file — a loader

A loader subclasses
[`ClassificationLoader`](https://github.com/adimo20/hcs4os/blob/master/hcs4os/dataloaders/BaseDataLoader.py) and implements exactly
one method, `to_records()`, returning a list of `Code`-shaped dicts. The base class
has already stored the source path as `self.path` (a `pathlib.Path`) and checked that
it exists; how you parse it is entirely up to you.

```python
import pandas as pd
from hcs4os.dataloaders.BaseDataLoader import ClassificationLoader
from hcs4os.dataloaders.registry import register as register_loader
from hcs4os._shared.Code import Code

@register_loader("HS_CODE")                 # <- the id you'll reference from the mapping, for convinience
class HSCodeLoader(ClassificationLoader):
    def to_records(self) -> list[dict]:
        df = pd.read_csv(self.path, dtype=str)
        return df.apply(
            lambda row: Code(
                code=row["id"],
                description=row["text"],
                level=row["aggrlevel"],
            ).to_dict(),
            axis=1,
        ).to_list()
```

[`Code`](https://github.com/adimo20/hcs4os/blob/master/hcs4os/_shared/Code.py) has five fields — `code`, `description`, `level`,
`detailled_description`, and `details` (a free-form dict for includes / excludes /
keywords / …). Fill only what you have: anything you omit defaults to empty, and any
extra keys you emit are dropped silently. Building each record with `Code(...).to_dict()`
is the safe way to get a correctly shaped dict, but a hand-built dict with the same
keys works too.

## 2. Teach it how codes nest — a classification system

A classification system subclasses
[`ClassificationSystem`](https://github.com/adimo20/hcs4os/blob/master/hcs4os/classification_system/BaseClassificationSystem.py) and
implements two methods. Everything else — `get_children`, `get_parent`, `get_code`,
`get_code_trace` — is derived for free from the first one:

```python
from hcs4os.classification_system.BaseClassificationSystem import ClassificationSystem
from hcs4os.classification_system.registry import register as register_classification_system

@register_classification_system("HS_CODE")
class ClassificationSystemHSCode(ClassificationSystem):
    def get_root_categories(self) -> list[dict]:
        # the entry points: HS chapters are the 2-digit codes
        return [{"code": c.code, "description": c.description}
                for c in self.codes if len(c.code) == 2]

    def get_prefixes(self, code: str) -> list[str]:
        # HS codes step two digits per level: "010121" -> ["01", "0101", "010121"]
        return [code[:i] for i in range(2, len(code) + 1, 2)]
```

`get_prefixes` is the only real logic you write, and it defines the entire hierarchy.
The rule the rest of the library builds on: **return the chain of ancestors of a code,
ending with the code itself; the parent of any code is then its second-to-last
prefix.** Get that right and parents, children and traces all fall out automatically.
HS codes are fixed-width prefix codes that grow two digits per level, so the ancestors
are simply the 2-, 4- and 6-digit slices. `get_root_categories` returns the top-level
entries as `{code, description}` dicts — where a top-down search (or a human) would
start.

> How your codes are strung together is the one thing only you know. Dot-separated
> (`01.1.1`), fixed-width prefixes (HS), irregular level jumps — each is just a
> different `get_prefixes`. After writing it, sanity-check with
> `cs.get_code_trace("<a deep code>")`: every rung should be present with no gaps.

## 3. Name the dataset — a mapping entry

The `mapping` dict is the lookup table that `get_classification_system(name)` consults.
It is an ordinary dict; add your entry to it at runtime:

```python
from hcs4os.classification_system.mapping import mapping

mapping["HS_CODE_2022"] = {
    "classification_system": "HS_CODE",   # the register_classification_system id from step 2
    "loader_name":           "HS_CODE",   # the register_loader id from step 1
    "data_path":             r"C:\data\hscode_unsd.csv",
    "metadata":              {"url": "https://unstats.un.org/..."},
}
```

The key (`"HS_CODE_2022"`) is the name you ask for later. `classification_system` and
`loader_name` are independent ids on purpose — the same hierarchy logic can be fed from
a different file or format just by pointing a new mapping key at a different loader.

## 4. Build it, index it, search it

From here your classification behaves exactly like a built-in one. Build the navigable
object, embed every code once into a persistent vector store, then query it:

```python
from hcs4os.classification_system import get_classification_system
from hcs4os._vector_database.vector_store import VectorStore

hs = get_classification_system("HS_CODE_2022")   # navigable immediately
# hs.get_root_categories(); hs.get_children("01"); hs.get_code_trace("010121"); ...

store = VectorStore(
    collection_name="HS_CODE_2022",
    chromadb_path=r"C:\data\chroma",
    embedding_function_type="ollama",            # or "sentence_transformers" / "openai"
    url="http://192.168.178.48:11434",           # backend-specific kwargs pass straight through
    model_name="qwen3-embedding:0.6b",
)

store.index(classification_system=hs)            # one-time: embeds every code, persists to disk

hits = store.search_category(
    "Ishihara color-blindness test book, leather cover",
    k=20,
    where={"level": "6"},                         # optional metadata filter
)
```

`index()` walks every `Code`, embeds its `description`, and stores the remaining
fields (`level`, `detailled_description`, `details`) as metadata beside it; pass
`codes_to_include=[...]` to index only a subset of codes. `search_category()` returns
the `k` nearest codes, each as a dict of that metadata with its `description` joined
back on.

## Running it again

Indexing is a one-time cost: the collection is written under `chromadb_path` and
reused on later runs. Each code is stored under a deterministic id, so re-running
`index()` on a populated collection is a harmless no-op (chroma skips existing ids with
a warning) — but wasted work, so on later runs just construct the `VectorStore` with
the same `collection_name` and go straight to `search_category`. A collection is only
meaningful with the embedding model that built it; to change `model_name` or backend,
index into a fresh `collection_name`.

## Three things that catch people out

- **Registration happens on import.** The `@register_*` decorators and the
  `mapping[...] = ` assignment only take effect when that code actually runs, so the
  module that defines your loader and system and adds the mapping entry must be
  imported _before_ you call `get_classification_system`. In a single script, keep
  steps 1–3 above the first use; in a package, import that module early.
- **Metadata filters match the stored type.** Whatever type a field has when it leaves
  your loader is what lands in the vector-store metadata. The HS loader reads the CSV
  with `dtype=str`, so `level` is the string `"6"` — hence `where={"level": "6"}`, not
  `{"level": 6}`. Store levels as ints and you filter with ints. (Nested dicts like
  `details` are serialized to a JSON string on the way in, so they come back but are
  not practically filterable.)
- **The embedding extra must match the backend.** `embedding_function_type="ollama"`
  needs `hcs4os[ollama]`, `"sentence_transformers"` needs `hcs4os[local]`, and so on;
  otherwise construction raises an `ImportError` naming the extra to install.

## Driving an LLM agent over it

Navigation and retrieval above work for any registered classification with zero extra
wiring. The two LLM classifiers (`RAGAgent` and the ReAct navigator) additionally need
a prose entry — a signature, a system prompt, tool descriptions — keyed by your dataset
name in the agent's own `registry.py`, because those are classification-specific. That
step is covered in _Adapting the content → E_; everything in this section is
independent of it.
