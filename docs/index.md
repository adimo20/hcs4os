# hcs4os

`hcs4os` turns the published documentation of official statistical classifications
(COICOP 2018, SEA 2021, WZ 2008, KldB 2010, ICATUS 2016, NAF rév. 2) into a uniform,
navigable Python object, and puts LLM agents and RAG-based approaches on top of it that classify a free-text
item against the classification systems descriptions. Any further classification can
be added without touching the library — see _Bringing your own classification_ below.

Four classifier styles solve that same task from different directions:

- **`HierarchicalNavigationAgent`** (`dspy.ReAct`) walks the tree top-down with tools
  — roots, children, parent, code — one level at a time.
- **`RAGAgent`** (`dspy.ReAct`) searches a vector store of the whole classification
  through a single `search_category` tool, pulling candidates by semantic similarity
  and reasoning its way to one of them, re-querying when the notes point elsewhere.
- **`RAGChainOfThought`** (`dspy.ChainOfThought`) does one retrieval and one reasoning
  pass over the candidates — no loop, no tool calls, the cheap baseline.
- **`RRR_RAGChainOfThought`** adds a Rewrite step in front of that baseline: a small
  model first restates the raw item in the vocabulary of the classification's labels,
  and only that rewritten query is embedded.

All four end at the same `ClassificationSystem` and emit the same output fields
(`<system>_code` + `explanation`), which is what makes them comparable on the same
data. Each is a thin subclass of a `Base…` class that takes its prompts and signature
as constructor arguments; the subclass only supplies a registered configuration. The
base variants work with any classification in `mapping.py`, and with custom ones. The
three retrieval-based ones take a `VectorStore` you build yourself, so the embedding
backend — a local `sentence-transformers` model, a self-hosted Ollama server, or
OpenAI — is independent of the reasoning model and shared across classifiers.

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

## Installation

Installation via pip:

```bash
pip install git+https://github.com/adimo20/hcs4os.git
```

Requires Python ≥ 3.12. The core install is parsing and navigation only — its
dependencies are `pandas`, `openpyxl` and `dspy`. The retrieval-based classifiers need
a vector-store backend, which ships as an optional extra (each pulls in `chromadb`
plus one embedding backend) — install the one you will use:

```bash
pip install "hcs4os[local]  @ git+https://github.com/adimo20/hcs4os.git"   # chromadb + sentence-transformers (in-process)
pip install "hcs4os[ollama] @ git+https://github.com/adimo20/hcs4os.git"   # chromadb + ollama (self-hosted HTTP)
pip install "hcs4os[openai] @ git+https://github.com/adimo20/hcs4os.git"   # chromadb + openai (hosted API)
```

The classification source files ship inside the package (`hcs4os/_data/`), so no
classification data is downloaded at runtime. A local `sentence-transformers`
embedding model is pulled from Hugging Face on first use and cached; the `openai` and
`ollama` embedding backends call out over HTTP instead and download nothing.
