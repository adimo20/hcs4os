# The vector store

This package owns everything between the classification and the retrieval-based
classifiers. It used to live inside `agents/RAG/`; pulling it out made it shareable by
`agents/RAG/` and `rag/` alike, and turned the embedding backend into a choice rather
than a hard-coded import.

**The retrieval contract.** [Retriever.py](https://github.com/adimo20/hcs4os/blob/master/hcs4os/_vector_database/Retriever.py) is an
ABC declaring `create_collection`, `index` and `search_category` (plus `__init__`, so
a subclass must define its own), and every classifier types its `store` parameter as
`Retriever`, not as `VectorStore`. So
the chroma store is one implementation rather than the only one: anything satisfying
that contract can be passed in, which is the seam for swapping in a different
retrieval approach (a BM25 or hybrid index, a managed vector service, a reranking
wrapper around another retriever) without touching the classifiers. Only
`search_category` is called at inference time; `index` and `create_collection` matter
only when you build the collection.

[vector_store.py](https://github.com/adimo20/hcs4os/blob/master/hcs4os/_vector_database/vector_store.py) holds `VectorStore`, the
chroma implementation of that contract.

**Pluggable embedding backends.** `get_ef()` is a small factory over three chroma
embedding functions, selected by `embedding_function_type` (`EFType`):

| type                    | runs                    | typical kwargs                                    |
| ----------------------- | ----------------------- | ------------------------------------------------- |
| `sentence_transformers` | in-process, local model | `model_name`, `device`, `normalize_embeddings`    |
| `ollama`                | HTTP, self-hosted       | `url`, `model_name`, `timeout`                    |
| `openai`                | HTTP, hosted API        | `api_key`, `model_name`, `api_base`, `dimensions` |

Everything beyond `embedding_function_type` is collected as `**embedding_kwargs` and
handed straight to that function, so the store never has to know what any backend
accepts. An unknown type fails fast with
`ValueError: Unknown embedding function: 'cohere'`. The collection itself is always a
persistent chroma collection in cosine space.

**`store.index(classification_system, codes_to_include=None)`** fills it. Every `Code`
is embedded by its **`description` only**; the rest of the record goes into the chroma
metadata under the id `id_<code>` — `code`, `level` and `detailled_description` as
they are, and the nested `details` dict JSON-dumped into one string, because chroma
accepts only flat scalars. So the embedding sees the short label alone, while the long
includes/excludes notes ride along and reach the model with the candidate.
`codes_to_include=None` indexes everything; a list of code strings narrows it.

**`store.search_category(query, k, where=None)`** embeds the query, pulls the `k`
nearest codes, and joins each one's `description` back onto its metadata.
`where` is a chroma metadata filter over the flat fields — `{"level": 5}` being the
one that matters, since it confines candidates to the depth being predicted. The
JSON-dumped `details` cannot be filtered on, only read.

One store, one embedding model, one collection: a collection is only meaningful
together with the model that built it, so changing `embedding_function_type` or
`model_name` means a fresh `collection_name`.
