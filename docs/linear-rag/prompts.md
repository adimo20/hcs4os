# Prompt Overview

In `rac4os` a prompt is not a string in a config file. Each classifier is driven by a
`dspy.Signature` whose **docstring is the system prompt** and whose **fields are the
input/output contract**. Both live in one place, because renaming an output field changes
what the prompt is allowed to refer to.

What you see on this page is rendered from the installed package, so it is exactly what
the model receives — it cannot drift from the code.

## Signature table

| Dataset               | Signature                              |
| --------------------- | -------------------------------------- |
| `COICOP_2018`         | `CoicopRAGChainOfThoughtSignature`     |
| `ICATUS_2016`         | `IcatusRAGChainOfThoughtSignature`     |
| `SEA_2021`            | `SeaRAGChainOfThoughtSignature`        |
| `SEA_2021_REPHRASED`  | `SeaRAGChainOfThoughtSignature`        |

The rewrite step has its own signatures — see [Rephrasing signatures](#rephrasing-signatures).

Each entry in `RAGChainOfThought.CONFIGS` bundles the signature with the two field names
the pipeline needs:

```python
"SEA_2021": {
    "signature": registry.SeaRAGChainOfThoughtSignature,
    "query_field": "input_expense",
    "context_field": "retrieved_candidates",
}
```

## Commonly used instructions

The bundled docstrings are written around a few recurring elements. They are worth
copying into a signature of your own:

| Element              | What it does                                                                |
| -------------------- | --------------------------------------------------------------------------- |
| Role and task        | names the taxonomy and the unit being classified                            |
| Reasoning protocol   | the order in which candidates are compared, and what breaks a tie           |
| Hierarchy guidance   | tells the model to judge on `includes` rather than on the short label       |
| Target level         | states which level a code may be emitted at                                 |
| Grounding constraint | forbids emitting a code that was not in the candidate set                   |
| Output discipline    | what belongs in the code field, what belongs in the explanation             |

Two variants of the hierarchy guidance exist in the bundled prompts, matching the two
[augmentation setups](../taxonomy/augmentation.md): one instructs the model to base its
judgment on the `includes` of each candidate, the other states that the description is
the only anchor available. They differ in that passage and nowhere else.

## Writing your own

```python
import dspy


class SpendRAGCoTSignature(dspy.Signature):
    """Classify an internal spend item into the SPEND taxonomy.

    You are given one item as recorded by a member of staff, plus candidate categories
    retrieved from the taxonomy. Choose exactly one.

    Reasoning protocol:
    - Read every candidate before judging, comparing the item against the `includes`
      entries, which enumerate what a category actually covers.
    - Where two candidates both fit, prefer the one whose `includes` names the item's
      concrete form.
    - Never emit a code that is not in the candidate list.

    The target level is level 3.
    """

    input_item: str = dspy.InputField(desc="the item text to classify, as recorded")
    retrieved_candidates: str = dspy.InputField(desc="candidate categories as JSON")
    spend_code: str = dspy.OutputField(desc="the code of the chosen category")
    explanation: str = dspy.OutputField(desc="why this category and not the others")
```

!!! tip

    Field names are part of the prompt. `spend_code` tells the model more than `code`
    does, and `input_item` more than `query`. Rename fields when you adapt a signature to
    a new taxonomy rather than keeping another classification's vocabulary.

Changing a prompt needs no registry edit at all — pass the signature straight to
`BaseRAGCoT` and A/B two versions in one script. Promote it into `CONFIGS` once it is
worth keeping.

## Bundled signatures

### Chain-of-thought signatures

::: rac4os.rag.registry.CoicopRAGChainOfThoughtSignature

::: rac4os.rag.registry.IcatusRAGChainOfThoughtSignature

::: rac4os.rag.registry.SeaRAGChainOfThoughtSignature

### Rephrasing signatures

Used by [Rewrite-Retrieve-Read](rrr.md). These do not classify — they restate the item in
the vocabulary of the labels.

::: rac4os.rag.registry.CoicopRephrasingSignature

::: rac4os.rag.registry.IcatusRephrasingSignature

::: rac4os.rag.registry.SeaRephrasingSignature
