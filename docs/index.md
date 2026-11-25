# rac4os Documentation

`rac4os` is an experimental prototype for applying Retrieval-Augmented Generation (RAG)
to classification tasks based on hierarchical statistical taxonomies. It is aimed at
practitioners in official statistics, the social sciences and economics who want to use
RAG-based and agentic approaches to classify data against statistical taxonomies. The
package provides modular, adaptable components for modelling each stage of a RAG
classification pipeline. You can use every component as it is or adapt and extend it for
new use cases.

`rac4os` offers three approaches to classification: **agentic hierarchy traversal**
([quickstart](getting-started/quickstart.md#hierarchy-traversal-agent)), **linear RAG**
([quickstart](getting-started/quickstart.md#linear-rag)) and **agentic RAG**
([quickstart](getting-started/quickstart.md#agentic-rag)). Linear RAG comes in two
variants, a naïve RAG implementation and a more advanced Rephrase-Retrieve-Response RAG.

All three approaches are built on two shared components you can adapt and customize:

- **Classification system**: a shared interface for working with the statistical
  taxonomy. It stores metadata about the taxonomy's codes and handles code lookups. It
  also supports more advanced operations, such as finding the child codes of a given
  higher-level code.
- **Retriever**: used by the RAG-based approaches to find the taxonomy codes most
  relevant to an input text through semantic search.

Both components are built around **your** taxonomy. You describe a classification once —
how its source file is read and how its codes nest — and all three approaches work on it
without further wiring. Worked entries for several official classifications (COICOP, SEA,
WZ, KldB, ICATUS, NAF) ship with the package as examples, but none of them is a
precondition for anything.

Don't hesitate to open an issue on the [repository](https://github.com/adimo20/rac4os)
if something is broken or if you have further questions.

## Usage

!!! tip "See also"

    See the [Quickstart](getting-started/quickstart.md) for more quick information on how
    to use `rac4os`.

### Installation

You can install `rac4os` using pip:

```bash
pip install "rac4os[local] @ git+https://github.com/adimo20/rac4os.git"
```

Requires Python ≥ 3.12. The core install covers only parsing and navigating taxonomies;
the retrieval-based approaches need one embedding backend, which comes as an optional
extra. See [Installation](getting-started/installation.md) for the other backends.

### Describing your taxonomy

```python
from rac4os.classification_system import get_classification_system

cs = get_classification_system("SPEND_2026")       # a taxonomy you registered

cs.get_root_categories()
# [{'code': '10', 'description': 'Office supplies and consumables'},
#  {'code': '20', 'description': 'IT equipment and services'}]

cs.get_code_trace("101010")
# [('10', 'Office supplies and consumables'),
#  ('1010', 'Paper and paper products'),
#  ('101010', 'Printing and copying paper')]
```

### Classifying an item

```python
prediction = classifier(query="two boxes of A4 paper for the printer", k=3)

prediction.spend_code
# '101010'
prediction.explanation
# 'Chosen 101010 because the item is copying paper, which that category covers; ...'
```

## What Next?

Consider reading one of the following sections to answer the related question:

**Your own taxonomy:**

- How do I describe my classification to `rac4os`? → [Your Own Taxonomy > Overview](taxonomy/overview.md)
- How do I read my own source file? → [Your Own Taxonomy > Data Loaders](taxonomy/loaders.md)
- How do I say how my codes nest? → [Your Own Taxonomy > Classification Systems](taxonomy/systems.md)
- How do I refer to a dataset by name? → [Your Own Taxonomy > Datasets and the Registry](taxonomy/datasets.md)
- How do I make my codes searchable? → [Your Own Taxonomy > Retrievers](taxonomy/retrievers.md)
- What should the model see about each candidate? → [Your Own Taxonomy > Candidate Augmentation](taxonomy/augmentation.md)

**Hierarchy traversal:**

- How do I use the hierarchy traversal agent? → [Hierarchy Traversal Agent > Usage](traversal/usage.md)
- What does it actually do? → [Hierarchy Traversal Agent > How It Works](traversal/how-it-works.md)
- How do I run it on my taxonomy? → [Hierarchy Traversal Agent > Adapting Overview](traversal/adapting.md)
- What prompt does it send? → [Hierarchy Traversal Agent > Prompt Overview](traversal/prompts.md)

**Linear RAG:**

- How do I use linear RAG? → [Linear RAG > Usage](linear-rag/usage.md)
- What does it actually do? → [Linear RAG > How It Works](linear-rag/how-it-works.md)
- What is Rephrase-Retrieve-Response? → [Linear RAG > Rewrite-Retrieve-Read](linear-rag/rrr.md)
- How do I run it on my taxonomy? → [Linear RAG > Adapting Overview](linear-rag/adapting.md)
- What prompt does it send? → [Linear RAG > Prompt Overview](linear-rag/prompts.md)

**Agentic RAG:**

- How do I use agentic RAG? → [Agentic RAG > Usage](agentic-rag/usage.md)
- What does the agent decide? → [Agentic RAG > How It Works](agentic-rag/how-it-works.md)
- How do I run it on my taxonomy? → [Agentic RAG > Adapting Overview](agentic-rag/adapting.md)
- What prompt does it send? → [Agentic RAG > Prompt Overview](agentic-rag/prompts.md)

**Utilities:**

- The inner categories of my taxonomy have thin descriptions. → [Taxonomy Refinement > Usage](refinement/usage.md)

## Which approach should I use?

| Approach                                       | Retrieval                          | Loop | Needs a retriever | Model calls per item |
| ---------------------------------------------- | ---------------------------------- | ---- | ----------------- | -------------------- |
| [Hierarchy traversal](traversal/usage.md)      | none — walks your taxonomy         | yes  | no                | several              |
| [Linear RAG](linear-rag/usage.md)              | once, on the raw item              | no   | yes               | one                  |
| [Rewrite-Retrieve-Read](linear-rag/rrr.md)     | once, on a rewritten query         | no   | yes               | two                  |
| [Agentic RAG](agentic-rag/usage.md)            | as often as the agent calls for it | yes  | yes               | several              |

Start with [linear RAG](linear-rag/usage.md): one retrieval call and one generation call
per item, and the baseline the others have to beat. If retrieval misses because your
items are worded nothing like your labels, add the [rewrite step](linear-rag/rrr.md). If
retrieval misses and you cannot improve the retriever, let the
[agent](agentic-rag/usage.md) re-query. If your taxonomy's structure carries more signal
than its labels — or you have no embedding backend at all —
[traverse the hierarchy](traversal/usage.md) instead.

## Citing

If you use `rac4os` in published work, please cite it:

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
