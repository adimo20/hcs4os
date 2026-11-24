# hcs4os — Hierarchical Classification Systems for Official Statistics

**📖 Documentation: <https://adimo20.github.io/hcs4os/>**

`hcs4os` turns the published documentation of official statistical classifications
(COICOP 2018, SEA 2021, WZ 2008, KldB 2010, ICATUS 2016, NAF rév. 2) into a uniform,
navigable Python object, and puts LLM agents and RAG-based approaches on top of it
that classify a free-text item against the classification systems descriptions. Any
further classification can be added without touching the library.

Four classifier styles solve that same task from different directions:

- **`HierarchicalNavigationAgent`** (`dspy.ReAct`) walks the tree top-down with tools
  — roots, children, parent, code — one level at a time.
- **`RAGAgent`** (`dspy.ReAct`) searches a vector store of the whole classification
  through a single `search_category` tool, re-querying when the notes point elsewhere.
- **`RAGChainOfThought`** (`dspy.ChainOfThought`) does one retrieval and one reasoning
  pass over the candidates — no loop, no tool calls, the cheap baseline.
- **`RRR_RAGChainOfThought`** adds a Rewrite step in front of that baseline: a small
  model first restates the raw item in the vocabulary of the classification's labels,
  and only that rewritten query is embedded.

All four end at the same `ClassificationSystem` and emit the same output fields
(`<system>_code` + `explanation`), which is what makes them comparable on the same
data.

## Install

```bash
pip install "hcs4os[local] @ git+https://github.com/adimo20/hcs4os.git"
```

Python ≥ 3.12. The core install is parsing and navigation only; the retrieval
classifiers need one embedding backend — `[local]` (sentence-transformers,
in-process), `[ollama]` or `[openai]`.

## A first look

```python
from hcs4os.classification_system import get_classification_system

sea = get_classification_system("SEA_2021")
sea.get_children("0111")
# [Code(code='0111 1', description='Getreide', level=4, ...), ...]
sea.get_code_trace("0111 1")
# [('01', 'Nahrungsmittel und alkoholfreie Getränke'), ('011', 'Nahrungsmittel'), ...]
```

## Where to go next

|                                                                                                    |                                                                       |
| -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| [Quick start](https://adimo20.github.io/hcs4os/quickstart/)                                        | runnable examples for each classifier style                           |
| [Bringing your own classification](https://adimo20.github.io/hcs4os/guides/custom-classification/) | register any classification without editing the library               |
| [Concepts](https://adimo20.github.io/hcs4os/concepts/)                                             | how the loaders, hierarchy, vector store and classifiers fit together |
| [Prompt reference](https://adimo20.github.io/hcs4os/prompts/)                                      | the exact system prompts each classifier sends                        |
| [API reference](https://adimo20.github.io/hcs4os/api/)                                             | generated from the source                                             |

## License

Copyright (c) 2026 Adrian Montag
Licensed under the EUPL v. 1.2

## Citation

```bibtex
@software{montag_hcs4os,
  author  = {Montag, Adrian},
  title   = {{hcs4os}: Hierarchical Classification Systems for Official Statistics},
  version = {0.2.9},
  year    = {2026},
  url     = {https://github.com/adimo20/hcs4os},
  license = {EUPL-1.2}
}
```
