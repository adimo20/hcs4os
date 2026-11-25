# Installation

We recommend **Python 3.12+**.

## Install with pip

```bash
pip install git+https://github.com/adimo20/rac4os.git
```

The core install covers only parsing and navigating taxonomies. Its dependencies are
`pandas`, `openpyxl` and `dspy`.

## Install with an embedding backend

The retrieval-based approaches ([linear RAG](../linear-rag/usage.md),
[Rewrite-Retrieve-Read](../linear-rag/rrr.md), [agentic RAG](../agentic-rag/usage.md))
additionally need a vector-store backend, which is available as an optional extra. Each
extra installs `chromadb` together with one embedding backend. The extras are kept
separate because the embedding backends' dependencies are large — install the one you
will use:

```bash
pip install "rac4os[local]  @ git+https://github.com/adimo20/rac4os.git"   # chromadb + sentence-transformers (in-process)
```

```bash
pip install "rac4os[ollama] @ git+https://github.com/adimo20/rac4os.git"   # chromadb + ollama (self-hosted HTTP)
```

```bash
pip install "rac4os[openai] @ git+https://github.com/adimo20/rac4os.git"   # chromadb + openai (hosted API)
```

| Extra      | `embedding_function_type` | Embeddings computed        |
| ---------- | ------------------------- | -------------------------- |
| `local`    | `"sentence_transformers"` | in-process, on your machine |
| `ollama`   | `"ollama"`                | by a self-hosted server     |
| `openai`   | `"openai"`                | by the hosted API           |

!!! tip

    The extra and the backend have to match. Constructing a `VectorStore` with
    `embedding_function_type="ollama"` without `rac4os[ollama]` installed raises an
    `ImportError` naming the extra to install.

Installing more than one extra is fine; they differ only in the embedding library they
pull in, and the backend is chosen per store at runtime.

## Editable install

To work on the package itself:

```bash
git clone https://github.com/adimo20/rac4os.git
```

```bash
pip install -e ".[local]"
```

## The language model

`rac4os` does not install or host a language model. The classifiers construct a
`dspy.LM` from the `model_name`, `api_key` and optional `api_base` you pass them, so any
provider reachable through [LiteLLM](https://docs.litellm.ai/docs/providers) model
strings works — a hosted API, a self-hosted endpoint, or a local server:

```python
classifier = SomeClassifier(
    classification_name="SPEND_2026",
    model_name="openai/gpt-4o-mini",   # provider/model
    api_key="...",
    api_base=None,                      # set for a self-hosted or compatible endpoint
    ...
)
```

Nothing is sent anywhere until you call a classifier.

## Building the documentation

```bash
pip install -e . -r docs/requirements.txt
```

```bash
mkdocs serve
```
