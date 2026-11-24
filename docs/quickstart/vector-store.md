# Building a vector store

Every retrieval-based classifier takes a `VectorStore` you index yourself, so the
embedding backend is a decision you make once and can reuse across classifiers:

```python
from hcs4os._vector_database.vector_store import VectorStore

# local model, downloaded and run in-process
store = VectorStore(
    collection_name="coicop_full",
    chromadb_path="./data/chroma",            # where the persistent collection lives
    embedding_function_type="sentence_transformers",
    model_name="sentence-transformers/all-MiniLM-L12-v2",
)

# or a self-hosted Ollama server
store = VectorStore(
    collection_name="sea_2021",
    chromadb_path="./data/chroma",
    embedding_function_type="ollama",
    url="http://localhost:11434",
    model_name="mxbai-embed-large",
)

# or OpenAI/OpenAI compatible endpoint
store = VectorStore(
    collection_name="coicop_openai",
    chromadb_path="./data/chroma",
    embedding_function_type="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small",
    embedding_batch_size=64,
)
```

Everything after `embedding_function_type` is `**embedding_kwargs`, forwarded
untouched to the matching chroma embedding function — so whatever that backend accepts
works here. `embedding_batch_size` (default `16`) controls how many documents go per
`collection.add` call; raise it for a local model, keep it low for a remote API.

Fill the store once, from the classification you will query — this is what the
retrieval classifiers search.

```python
from hcs4os.classification_system import get_classification_system

store.index(classification_system=get_classification_system("COICOP_2018"))
# store.index(classification_system=..., codes_to_include=[...])   # optional: index a subset
```

`index()` embeds every code's `description` and persists the collection under
`chromadb_path`, so it is a one-time cost: on later runs rebuild the same
`VectorStore(collection_name=…)` and go straight to querying. Re-running `index()` on a
populated collection is a harmless no-op (ids are deterministic), and a collection is
only meaningful with the embedding model that built it — change the backend or
`model_name` and you index into a fresh `collection_name`.
