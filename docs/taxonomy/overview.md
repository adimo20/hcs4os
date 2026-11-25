# Overview

This section is the main path through the package: how you describe a classification of
your own, so that every approach can work on it. It is shared by all three approaches —
each of their *Adapting Overview* pages points back here.

## What `rac4os` assumes about your taxonomy

A statistical taxonomy is a structured set of categories presented as a hierarchy, where
the hierarchy is reflected in the codes themselves. `rac4os` assumes exactly that and
nothing more:

- **Codes carry their position.** `101010` is a child of `1010` because of how the two
  strings relate, not because an edge is stored somewhere. This is what lets the whole
  hierarchy be derived from one function.
- **Inner categories are assignable.** You may classify to a leaf or to an inner node —
  full-depth or partial-depth labelling. Which one you do changes what the model should
  see about each candidate (see [Candidate Augmentation](augmentation.md)) but not how
  you describe the taxonomy.
- **Categories have a short label, and possibly more.** A label is what gets embedded;
  anything further — inclusions, exclusions, keywords, notes — is kept alongside it and
  is what resolves the ambiguous cases.

Nothing requires your codes to be numeric, fixed-width, dot-separated, German, or
published by a statistical institute. An internal product catalogue, a spend
classification, an in-house coding scheme all qualify.

## Components

| Component                                   | Answers                                   | What you write                                         |
| ------------------------------------------- | ----------------------------------------- | ------------------------------------------------------ |
| [Data loader](loaders.md)                   | how is my source file read?               | one method: `to_records() -> list[dict]`               |
| [Classification system](systems.md)         | how do my codes nest?                     | `get_prefixes(code)` and `get_root_categories()`       |
| [Dataset entry](datasets.md)                | what is this dataset called?              | a dict tying loader, system and file together          |
| [Retriever](retrievers.md)                  | how are candidate codes found?            | nothing — construct a `VectorStore` and fill it        |
| [Candidate augmentation](augmentation.md)   | what does the model see about a candidate? | what you put in `Code.details`                        |
| `Code`                                      | the record every category becomes         | provided — you emit its fields                         |

Two methods and a dict are the whole obligation. Everything downstream is derived from
them.

## How the layers fit together

```
  your file: .csv | .xlsx | .xml | .json       the published or internal documentation
            │
            ▼
  a data loader ── to_records() ──► list[dict] ──► list[Code]      format layer
            │                                                       (how it is parsed)
            ▼
  a classification system ── get_prefixes() ──► hierarchy          structure layer
            │        ▲                                              (how codes nest)
            │        └── taxonomy refinement ── writes back a richer dataset
            ▼
  a retriever ── index() / search_category() ──► candidates        retrieval layer
            │                                                       (how codes are found)
            ▼
  hierarchy traversal agent    ─┐
  linear RAG (+ rewrite step)   ├──►  code + explanation           reasoning layer
  agentic RAG                   ┘
```

Each layer is replaceable on its own. The
[hierarchy traversal agent](../traversal/usage.md) skips the retrieval layer entirely and
reaches the classification system through its four operations; the RAG-based approaches
take a retriever you have built and filled yourself.

## The four operations every approach builds on

Once your taxonomy is described, it answers four questions, and every approach in the
package is written against these rather than against any particular classification:

| Operation             | Question                              |
| --------------------- | ------------------------------------- |
| `get_root_categories` | where does a top-down search start?   |
| `get_children`        | what falls under this code?           |
| `get_parent`          | what is this code a special case of?  |
| `get_code`            | what does this code say about itself? |

Adding a classification therefore makes it agent-ready automatically. There is no
per-approach wiring beyond a prompt.

## Steps

1. [Data Loaders](loaders.md) — turn your source format into `Code` records.
2. [Classification Systems](systems.md) — say how your codes nest; the four operations
   follow.
3. [Datasets and the Registry](datasets.md) — give the dataset a name.
4. [Retrievers](retrievers.md) — embed your categories once. Only for the RAG-based
   approaches.
5. [Candidate Augmentation](augmentation.md) — decide what the model reads about each
   candidate.

Then pick an approach: [hierarchy traversal](../traversal/usage.md),
[linear RAG](../linear-rag/usage.md) or [agentic RAG](../agentic-rag/usage.md).

## The registry

Steps 1–3 use three small registries and a plain dictionary, and they exist for one
reason: so that a dataset can be referred to by a name. Once `"SPEND_2026"` means *this
file, read by this loader, nested by these rules*, every component takes that one string
and needs nothing else — and you can keep variants of the same taxonomy side by side (a
raw and a [refined](../refinement/usage.md) edition, two vintages, a subset) and switch
between them by changing a string.

Nothing in the package requires it. `ClassificationSystem` subclasses can be constructed
directly from a loader instance, and the `Base…` variant of every classifier takes its
configuration as plain constructor arguments. The registry saves you from passing the
same objects around; it is a convenience, not a gate.

!!! note "Bundled datasets"

    The package ships worked entries for several official classifications (COICOP 2018,
    SEA 2021, WZ 2008, KldB 2010, ICATUS 2016, NAF rév. 2), with their source
    documentation under `rac4os/_data/`. They are reference implementations of the steps
    above: useful to run against before your own taxonomy is wired up, and useful as
    examples when your source format resembles one of theirs. No approach treats them
    differently from a taxonomy you register yourself.

## The example used throughout

To keep every example runnable without any of the bundled data, this documentation
registers a small internal spend taxonomy from a CSV file — three levels, two digits per
level. The [Quickstart](../getting-started/quickstart.md) has the file and the full
registration script, and every output shown in this section was produced from it.
