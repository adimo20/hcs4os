# rac4os — Retrieval Augmented Classification for Official Statistics

**📖 Documentation: <https://adimo20.github.io/rac4os/>**

`rac4os` is an experimental prototype for applying Retrieval-Augmented Generation (RAG)
to classification tasks based on hierarchical statistical taxonomies. It is aimed at
practitioners in official statistics, the social sciences and economics who want to
classify free-text data against statistical taxonomies with RAG-based and agentic
approaches. The package provides modular, adaptable components for modelling each stage
of a RAG classification pipeline — you can use every component as it is, or adapt and
extend it for new use cases.

Three approaches to classification are available:

- **agentic hierarchy traversal** — walks your taxonomy top-down through four navigation
  tools; no retriever and no embeddings required.
- **linear RAG** — one retrieval call and one chain-of-thought generation per item. Comes
  in two variants: a naïve RAG implementation and a more advanced
  Rephrase-Retrieve-Response RAG, which rewrites the query in the vocabulary of your
  labels before retrieving.
- **agentic RAG** — exposes the retriever to the model as a tool, so it can reformulate
  its query and search again when the candidates do not fit.

All three are built on two shared components you can adapt: a **classification system**
(the uniform interface to your taxonomy) and a **retriever** (semantic search over your
categories). Both are built around _your_ classification — you describe it once, and
every approach works on it. Worked entries for several official classifications (COICOP,
SEA, WZ, KldB, ICATUS, NAF) ship with the package as examples.

## Install

```bash
pip install "rac4os[local] @ git+https://github.com/adimo20/rac4os.git"
```

Python ≥ 3.12. The core install covers parsing and navigating taxonomies only; the
retrieval-based approaches need one embedding backend — `[local]`
(sentence-transformers, in-process), `[ollama]` or `[openai]`.

## A first look

```python
from rac4os.classification_system import get_classification_system

cs = get_classification_system("SPEND_2026")       # a taxonomy you registered
cs.get_children("1010")
# [Code(code='101010', description='Printing and copying paper', level=3, ...), ...]
cs.get_code_trace("101010")
# [('10', 'Office supplies and consumables'), ('1010', 'Paper and paper products'), ...]
```

## Where to go next

|                                                                              |                                             |
| ---------------------------------------------------------------------------- | ------------------------------------------- |
| [Quickstart](https://adimo20.github.io/rac4os/getting-started/quickstart/)   | each approach run once, end to end          |
| [Your Own Taxonomy](https://adimo20.github.io/rac4os/taxonomy/overview/)     | describe your classification in three steps |
| [Linear RAG](https://adimo20.github.io/rac4os/linear-rag/usage/)             | retrieve once, reason once                  |
| [Agentic RAG](https://adimo20.github.io/rac4os/agentic-rag/usage/)           | the retriever as a tool                     |
| [Hierarchy Traversal](https://adimo20.github.io/rac4os/traversal/usage/)     | classify by descending the hierarchy        |
| [Package Reference](https://adimo20.github.io/rac4os/reference/classifiers/) | generated from the source                   |

## License

Copyright (c) 2026 Adrian Montag
Licensed under the EUPL v. 1.2

## Citation

```bibtex
@software{montag_rac4os,
  author  = {Montag, Adrian},
  title   = {{rac4os}: Retrieval Augmented Classification for Official Statistics},
  version = {0.2.10},
  year    = {2026},
  url     = {https://github.com/adimo20/rac4os},
  license = {EUPL-1.2}
}
```
