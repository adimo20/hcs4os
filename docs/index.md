# rac4os

## Overview

`rac4os` is an experimental prototype for applying Retrieval-Augmented Generation (RAG) to classification tasks based on official statistical taxonomies. It is aimed at practitioners in official statistics, the social sciences and economics who want to use RAG-based and agentic approaches to classify data against common statistical taxonomies. The package provides modular, adaptable components for modelling each stage of a RAG classification pipeline. You can use every component as it is or adapt and extend it for new use cases.

`rac4os` offers three approaches to classification: **agentic hierarchy traversal**, **agentic RAG** and **linear RAG**. Linear RAG comes in two variants, a naïve RAG implementation and a more advanced Rephrase-Retrieve-Response RAG.

All three approaches are built on two shared components, you can adapt and customize:

- **Classification system**: a shared interface for working with the statistical taxonomy. It stores metadata about the taxonomy's codes and handles code lookups. It also supports more advanced operations, such as finding the child codes of a given higher-level code.
- **Retriever**: used by the RAG-based approaches to find the taxonomy codes most relevant to an input text through semantic search.

## Installation

Installation via pip:

```bash
pip install git+https://github.com/adimo20/hcs4os.git
```

Requires Python ≥ 3.12. The core install covers only parsing and navigating taxonomies. The retrieval-based classifiers also need a vector-store backend, which is available as an optional extra. Each extra installs `chromadb` together with one embedding backend. The extras are kept separate because the embedding backends' dependencies are large. To install the extra you need, run:

```bash
pip install "hcs4os[local]  @ git+https://github.com/adimo20/hcs4os.git"   # chromadb + sentence-transformers (in-process)
pip install "hcs4os[ollama] @ git+https://github.com/adimo20/hcs4os.git"   # chromadb + ollama (self-hosted HTTP)
pip install "hcs4os[openai] @ git+https://github.com/adimo20/hcs4os.git"   # chromadb + openai (hosted API)
```
