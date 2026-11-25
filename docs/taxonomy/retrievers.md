# Retrievers

The retriever is what the RAG-based approaches use to find the categories most relevant
to an input text. You build it and fill it; classifiers only query it.

## Why semantic search

The text you classify is rarely written in the language of your taxonomy. Receipt texts
carry brand names and abbreviations, survey responses carry whatever the respondent
typed, and the correct category's official label may share not one word with either.
Keyword search fails here by construction: a query saying *beverage* does not match a
label saying *drink*.

Dense retrieval compares meaning instead of wording. Categories and queries are encoded
into embeddings by the same model, and similarity in that vector space decides the match.
Because categories are embedded independently of the query, the whole taxonomy can be
embedded once and reused, and a classification costs one embedding pass plus a
nearest-neighbour search.

Two practical consequences:

- **The embedding model is the largest single lever on retrieval quality.** A generic
  multilingual model is a weak retriever for short domain-specific texts, and a model
  matched to the language and register of your data is worth more than any prompt edit
  further down the pipeline. Where you have labelled data, fine-tuning the embedding
  model on your own text-to-label pairs is the most effective intervention available.
- **A collection is only meaningful with the model that built it.** Both sides of the
  comparison have to come from the same embedding space, so changing the model or backend
  means indexing into a *fresh* collection name rather than re-indexing an existing one.

## How it works

`Retriever` is the abstract interface the classifiers are typed against — four methods:
`__init__`, `create_collection`, `index` and `search_category`. `VectorStore` is the
implementation that ships with the package, backed by a persistent ChromaDB client with
cosine distance.

Three embedding backends are available, selected by `embedding_function_type`, and any
further keyword arguments are forwarded untouched to the backend:

| `embedding_function_type` | Runs                            | Install extra   |
| ------------------------- | ------------------------------- | --------------- |
| `"sentence_transformers"` | in-process, offline after first download | `rac4os[local]`  |
| `"ollama"`                | against a self-hosted server    | `rac4os[ollama]` |
| `"openai"`                | against the hosted API          | `rac4os[openai]` |

```python
from rac4os._vector_database.vector_store import VectorStore

store = VectorStore(
    collection_name="spend_2026_level3",
    chromadb_path="./chroma",
    embedding_function_type="sentence_transformers",
    model_name="sentence-transformers/all-MiniLM-L6-v2",   # backend-specific kwarg
)

store.index(
    classification_system=cs,
    codes_to_include=[c.code for c in cs.codes if c.level == 3],
)
```

`index()` walks the classification system, embeds each category's `description`, and
stores the remaining fields as metadata beside it. `codes_to_include` restricts what goes
in at all — see the note on target levels below. Indexing is a one-time cost: the
collection is written under `chromadb_path` and reused on later runs, so a second run
should construct the same `VectorStore` and go straight to searching.

**Store injection.** The classifiers never index anything. They take a `Retriever` that
you have already built and filled, which is what lets one collection serve several
classifiers and lets you swap the embedding backend at the call site instead of inside a
classifier.

## Example output

```pycon
>>> hits = store.search_category("two boxes of A4 paper for the printer", k=3)
>>> print(json.dumps(hits, indent=2, ensure_ascii=False))
[
  {
    "code": "101010",
    "description": "Printing and copying paper",
    "level": 3,
    "detailled_description": "",
    "details": "{\"includes\": [\"copier paper\", \"inkjet paper\", \"continuous forms\"]}"
  },
  {
    "code": "101020",
    "description": "Hygiene paper",
    "level": 3,
    "detailled_description": "",
    "details": "{\"includes\": [\"paper towels\", \"toilet paper\", \"facial tissues\"]}"
  },
  {
    "code": "102010",
    "description": "Pens and pencils",
    "level": 3,
    "detailled_description": "",
    "details": "{\"includes\": [\"ballpoint pens\", \"pencils\", \"markers\"]}"
  }
]
```

Note what the query and the top hit have in common lexically: *paper*, and nothing else.
*A4*, *boxes* and *printer* appear in no label; *copying* appears in no query. A dense
retriever ranks it first regardless, which is the entire argument of this page in one
result.

Note also the second hit. *Hygiene paper* is semantically adjacent and wrong, and no
embedding model will reliably separate the two on label text alone. Resolving that is the
generator's job, and what the generator needs in order to do it is the subject of
[Candidate Augmentation](augmentation.md).

## Restricting the collection

Indexing only the codes you are actually willing to predict is the cheapest available
improvement to retrieval quality: it removes whole classes of wrong answer before any
model sees them. Two restrictions are worth making deliberately:

- **To the target level.** If you classify at level 3, index level 3. A collection
  containing every level competes broad inner codes against the specific codes you want.
- **To the codes that occur in practice.** Published taxonomies contain categories that
  never appear in your data. Excluding them narrows the candidate space without losing
  anything you could have predicted.

Both are applied when you fill the store, via `codes_to_include`, not on the classifier.

A metadata filter can also be applied per query, which is useful for restricting a single
call to one branch:

```python
store.search_category("docking station", k=5, where={"level": 3})
```

`where` matches against the stored metadata, and the stored type is whatever your loader
emitted — if `level` left your loader as the string `"3"`, the filter must be `{"level":
"3"}`. Nested dicts such as `details` are serialised to a JSON string on the way in, so
they come back intact but are not practically filterable.

## Using a different retriever entirely

`Retriever` is an abstract base class, and the classifiers only ever call
`search_category(query, k, where)` on it. A FAISS index, a managed vector database, a
hybrid sparse-dense setup, or a thin wrapper around an existing search service at your
institution can all be substituted by implementing the four methods — no classifier
changes, because none of them know what they are talking to.
