# Prompt Overview

An agentic classifier reads two pieces of prose: the **signature docstring**, which is the
system prompt, and the **tool description**, which is what it sees when deciding how to
call the retriever. Both are rendered below from the installed package, so they are
exactly what the model receives.

## Signature table

| Dataset               | Signature                            |
| --------------------- | ------------------------------------ |
| `COICOP_2018`         | `CoicopRAGSearchAgentSignature`      |
| `ICATUS_2016`         | `IcatusRAGSearchAgentSignature`      |
| `SEA_2021`            | `SeaRAGSearchAgentSignature`         |
| `SEA_2021_REPHRASED`  | `SeaRAGSearchAgentSignature`         |

Each entry in `RAGAgent.CONFIGS` bundles the signature with the tool description:

```python
"SEA_2021": {
    "signature": registry.SeaRAGSearchAgentSignature,
    "search_category_tool_doc": "...",
}
```

## Commonly used instructions

| Element              | What it does                                                        |
| -------------------- | ------------------------------------------------------------------- |
| Role and task        | names the taxonomy and the unit being classified                    |
| Search protocol      | how to word the first query                                          |
| Re-query trigger     | when to search again instead of answering                            |
| Stop condition       | when to stop searching and commit                                    |
| Hierarchy guidance   | judge on what a category includes, not on its short label            |
| Target level         | which level a code may be emitted at                                 |
| Grounding constraint | never emit a code the tool did not return                            |

The re-query trigger and the stop condition are what distinguish an agentic prompt from a
linear one. Without the first, the agent searches once and behaves like a linear pipeline
with extra latency; without the second, it keeps searching after it already has the
answer.

## The tool description

`search_category_tool_doc` is passed to `dspy.Tool` as the description of the store's
`search_category` method. It should state:

- what the tool searches,
- what `query` should look like,
- that `k` exists and roughly how to choose it,
- what the result contains,
- that calling it again with different wording is allowed.

```python
search_category_tool_doc = (
    "Search the SPEND taxonomy for candidate categories by semantic similarity. "
    "Arguments: `query`, a short description of the product or service; `k`, how many "
    "candidates to return. Returns a list of categories with their code, description "
    "and `includes`. Call it again with different wording whenever the returned "
    "categories do not cover the item."
)
```

## Writing your own

```python
import dspy


class SpendRAGAgentSignature(dspy.Signature):
    """Classify an internal spend item into the SPEND taxonomy by searching it.

    Search protocol:
    - Start with a focused query naming the generic product or service, not the brand,
      the quantity or the supplier.
    - Read the `includes` of every candidate returned before judging.
    - If none fits, search again with different wording — a synonym, a more generic
      term, or the function of the item rather than its name.
    - Stop as soon as one candidate clearly covers the item.
    - Never emit a code the search tool did not return.

    The target level is level 3.
    """

    input_item: str = dspy.InputField(desc="the item text to classify, as recorded")
    spend_code: str = dspy.OutputField(desc="the code of the chosen category")
    explanation: str = dspy.OutputField(desc="why this category and not the others")
```

!!! tip

    An agentic signature has no candidate input field — candidates arrive as observations
    in the trajectory. If you adapt a [linear signature](../linear-rag/prompts.md), delete
    its context field.

## Bundled signatures

::: rac4os.agents.RAG.registry.CoicopRAGSearchAgentSignature

::: rac4os.agents.RAG.registry.IcatusRAGSearchAgentSignature

::: rac4os.agents.RAG.registry.SeaRAGSearchAgentSignature
