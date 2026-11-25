# Adapting Overview

Running agentic RAG on a taxonomy of your own needs the same taxonomy and retriever as
[linear RAG](../linear-rag/adapting.md), and two pieces of prose instead of three
configuration values.

## Components

| Component             | What you supply                                      | Page                                                  |
| --------------------- | ---------------------------------------------------- | ----------------------------------------------------- |
| Taxonomy              | a loader, a nesting rule, a dataset name             | [Overview](../taxonomy/overview.md)                   |
| Retriever             | a `VectorStore`, filled by you                       | [Retrievers](../taxonomy/retrievers.md)               |
| Signature             | the system prompt (docstring) and the I/O fields     | [Prompt Overview](prompts.md)                          |
| Tool description      | `search_category_tool_doc`                           | this page                                              |
| Candidate shape       | what each `Code` carries into an observation          | [Candidate Augmentation](../taxonomy/augmentation.md) |

The first two are shared with the linear approach — the same store can serve both, which
is what makes comparing them on the same data straightforward.

## Signature

An agentic signature has **no candidate input field**. The candidates arrive as
observations, so the only input is the item:

```python
import dspy


class SpendRAGAgentSignature(dspy.Signature):
    """Classify an internal spend item into the SPEND taxonomy by searching it.

    Search protocol:
    - Start with a focused query naming the generic product or service, not the brand,
      the quantity or the supplier.
    - Read the `includes` of every candidate returned; they enumerate what a category
      actually covers, and the short description alone is often broader than it looks.
    - If none of the candidates fits, search again with different wording — a synonym, a
      more generic term, or the function of the item rather than its name.
    - Stop as soon as one candidate clearly covers the item. Do not keep searching for
      confirmation.
    - Never emit a code the search tool did not return.

    The target level is level 3.
    """

    input_item: str = dspy.InputField(desc="the item text to classify, as recorded")
    spend_code: str = dspy.OutputField(desc="the code of the chosen category")
    explanation: str = dspy.OutputField(desc="why this category and not the others")
```

Two instructions carry most of the weight here and have no counterpart in the linear
prompt: **when to re-query** and **when to stop**. Without the first the agent behaves
like a linear pipeline with extra steps; without the second it keeps searching after it
already has the answer, which is the common way an agentic run becomes expensive.

## Tool description

`search_category_tool_doc` is what the model reads when deciding how to call the
retriever. It has to say what the tool searches, what the arguments mean, and that
calling it again is allowed:

```python
search_category_tool_doc = (
    "Search the SPEND taxonomy for candidate categories by semantic similarity. "
    "Arguments: `query`, a short description of the product or service; `k`, how many "
    "candidates to return — use 5 to 10 for a broad first look and fewer when you are "
    "confirming a specific guess. Returns a list of categories with their code, "
    "description and `includes`. Call it again with different wording whenever the "
    "returned categories do not cover the item."
)
```

Mentioning `k` explicitly matters: the model chooses it, and a tool description that does
not mention it tends to produce a single conservative search with a small candidate set.

## Retriever

Identical to the [linear setup](../linear-rag/adapting.md#retriever) — build it, fill it
once, restrict it to the codes you are willing to predict:

```python
store.index(
    classification_system=cs,
    codes_to_include=[c.code for c in cs.codes if c.level == 3],
)
```

Restriction matters more here than in the linear case, because the agent will happily
search several times within whatever space you gave it. Codes you would never assign are
candidates it can waste steps on.

## Configuration

Either pass the two prose pieces to the base class:

```python
from rac4os.agents.RAG.BaseAgentRAG import BaseRAGAgent

agent = BaseRAGAgent(
    classification_name="SPEND_2026",
    api_key="...",
    model_name="openai/gpt-4o-mini",
    signature=SpendRAGAgentSignature,
    search_category_tool_doc=search_category_tool_doc,
    store=store,
)
```

…or register them once:

```python
from rac4os.agents.RAG.RAGAgent import RAGAgent

RAGAgent.CONFIGS["SPEND_2026"] = {
    "signature": SpendRAGAgentSignature,
    "search_category_tool_doc": search_category_tool_doc,
}

agent = RAGAgent(
    classification_name="SPEND_2026",
    api_key="...", model_name="openai/gpt-4o-mini", store=store,
)
```

The keys are the base constructor's parameter names. An unregistered name raises
`ValueError: Unknown classification system: …` before any model or embedding work starts.

## End-to-End Example

Assuming `cs` and `store` from
[Linear RAG > Adapting Overview](../linear-rag/adapting.md#end-to-end-example):

```python
from rac4os.agents.RAG.BaseAgentRAG import BaseRAGAgent

agent = BaseRAGAgent(
    classification_name="SPEND_2026",
    api_key="...",
    model_name="openai/gpt-4o-mini",
    signature=SpendRAGAgentSignature,
    search_category_tool_doc=search_category_tool_doc,
    store=store,
)

for item in ["docking station for the new laptops",
             "two boxes of A4 paper for the printer",
             "annual antivirus renewal"]:
    p = agent(input_item=item, max_iters=5)
    steps = sum(1 for key in p.trajectory if key.startswith("tool_name_"))
    print(f"{item!r:45} -> {p.spend_code}  ({steps} steps)")
```

Logging the step count from the start is worth the two lines: it is what tells you whether
the loop is earning its cost on your data, or whether a
[linear pipeline](../linear-rag/usage.md) would have reached the same answer in one call.

## Choosing a model

The constraint specific to this approach is structured-output reliability. The agent must
emit a well-formed tool call at every step, and models that are fine in the linear
pipeline can fail here often enough to be unusable. Test a candidate model on a handful
of items first and watch for parser errors rather than wrong codes — they are a different
failure, and they do not improve with prompt changes.
