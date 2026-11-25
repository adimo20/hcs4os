# Adapting Overview

This is the shortest adaptation path in the package: a described taxonomy, a signature,
and four tool descriptions. No retriever, no embeddings.

## Components

| Component              | What you supply                                  | Page                                              |
| ---------------------- | ------------------------------------------------ | ------------------------------------------------- |
| Taxonomy               | a loader, a nesting rule, a dataset name         | [Overview](../taxonomy/overview.md)               |
| Signature              | the system prompt (docstring) and the I/O fields | [Prompt Overview](prompts.md)                      |
| Four tool descriptions | what each navigation tool is for                 | this page                                          |

## Taxonomy

Everything this approach needs is the four navigation operations, so the only requirement
is that your [classification system](../taxonomy/systems.md) answers them correctly:

```python
from rac4os.classification_system import get_classification_system

cs = get_classification_system("SPEND_2026")

cs.get_root_categories()          # the agent's entry point
cs.get_children("10")             # one level down
cs.get_code("101010")             # what a category covers
cs.get_parent("101010")           # back out of a branch
```

!!! warning "Verify the hierarchy first"

    This approach fails silently on a wrong nesting rule. If `get_children` returns an
    empty list for a category that has children, the agent sees a dead end and will
    classify into whatever it reached last. Run the check from
    [Classification Systems](../taxonomy/systems.md#checking-it-before-you-trust-it)
    before you spend model calls:

    ```python
    print(cs.get_code_trace("<a known deep code>"))   # every level, no gaps
    print(cs.get_children("<a known mid-level code>"))
    ```

## Signature

A traversal signature has no candidate field — the categories arrive as tool
observations. What it does need is the descent protocol:

```python
import dspy


class SpendNavigationSignature(dspy.Signature):
    """Classify an internal spend item into the SPEND taxonomy by descending it.

    Descent protocol:
    - Call `get_root_categories` first and choose the top-level category whose purpose
      the item serves.
    - At each level, call `get_children` and compare the item against *every* child
      before choosing one. Do not take the first plausible option.
    - Use `get_code` to check what a candidate actually includes before committing to
      it; a short label is often broader or narrower than it appears.
    - If nothing at a level fits, call `get_parent` and try a sibling branch instead of
      forcing the item into the best of a bad set.
    - Descend until you reach a level 3 code. Never emit a code that did not come from a
      tool call.
    """

    input_item: str = dspy.InputField(desc="the item text to classify, as recorded")
    spend_code: str = dspy.OutputField(desc="the code of the chosen category")
    explanation: str = dspy.OutputField(
        desc="the path taken through the taxonomy and why each turn was chosen"
    )
```

The two instructions that matter most are *compare every child* and *backtrack when
nothing fits*. They are what address this approach's
[failure mode](how-it-works.md#the-failure-mode); without them an early wrong turn
becomes a wrong answer.

Asking for the path in the `explanation` is also worth doing: it makes the output readable
as a coding justification rather than a bare code.

## Tool descriptions

Four strings, one per tool. They are what the model reads when deciding which tool to
call, so each should say what the tool returns *and when to use it*:

```python
docs = {
    "get_root_categories_doc": (
        "Return the top-level categories of the SPEND taxonomy as code and "
        "description. Call this first, before anything else."
    ),
    "get_children_doc": (
        "Return the child categories of a code, with their descriptions and what they "
        "include. Use it to descend exactly one level at a time."
    ),
    "get_code_doc": (
        "Return the full record of a single code: its description, its level and what "
        "it includes. Use it to verify a candidate before answering."
    ),
    "get_parent_doc": (
        "Return the parent of a code. Use it to back out of a branch when none of its "
        "children fits the item."
    ),
}
```

!!! tip

    Describe your levels in your own vocabulary. If your taxonomy calls them divisions,
    groups and classes, say so in the tool descriptions and in the signature — the model
    is navigating your structure, and naming it the way your documentation does makes the
    descent easier to follow and easier to debug.

## Configuration

Pass the five pieces to the base class:

```python
from rac4os.agents.HierarchicalNavigation.BaseAgent import BaseHierarchicalNavigationAgent

agent = BaseHierarchicalNavigationAgent(
    classification_name="SPEND_2026",
    api_key="...",
    model_name="openai/gpt-4o-mini",
    signature=SpendNavigationSignature,
    **docs,
)
```

…or register them once and construct by name:

```python
from rac4os.agents.HierarchicalNavigation.HierarchicalNavigationAgent import (
    HierarchicalNavigationAgent,
)

HierarchicalNavigationAgent.CONFIGS["SPEND_2026"] = {
    "signature": SpendNavigationSignature,
    **docs,
}

agent = HierarchicalNavigationAgent(
    classification_name="SPEND_2026",
    api_key="...", model_name="openai/gpt-4o-mini",
)
```

The keys are the base constructor's parameter names, so a registry entry is a bundle of
keyword arguments and nothing more.

## End-to-End Example

```python
import dspy
from rac4os.classification_system import get_classification_system
from rac4os.agents.HierarchicalNavigation.BaseAgent import BaseHierarchicalNavigationAgent

cs = get_classification_system("SPEND_2026")           # registered earlier
assert cs.get_children("1010"), "nesting rule is wrong"

agent = BaseHierarchicalNavigationAgent(
    classification_name="SPEND_2026",
    api_key="...",
    model_name="openai/gpt-4o-mini",
    signature=SpendNavigationSignature,
    **docs,
)

for item in ["two boxes of A4 paper for the printer",
             "docking station for the new laptops"]:
    p = agent(input_item=item, max_iters=8)
    path = [p.trajectory[k] for k in p.trajectory if k.startswith("tool_args_")]
    print(f"{item!r:45} -> {p.spend_code}   via {path}")
```

## Next steps

- Compare it against [linear RAG](../linear-rag/adapting.md) on the same taxonomy. The
  output contract is identical, so you can run both over the same items and diff the
  codes — where they disagree is where your taxonomy's structure and its labels pull in
  different directions.
- If the agent keeps choosing the wrong branch at the top level, the problem is usually
  that your top-level categories separate by purpose rather than by kind. That is a case
  for retrieval, not for a longer prompt.
