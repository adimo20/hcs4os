# Usage

`HierarchicalNavigationAgent` · `BaseHierarchicalNavigationAgent`

The hierarchy traversal agent classifies by walking your taxonomy instead of searching
it. It starts at the top-level categories, compares the item against the children of the
branch it chose, descends, and verifies — using the four navigation operations your
[classification system](../taxonomy/systems.md) already provides.

It needs no retriever, no embeddings and no vector-store extra.

!!! tip "See also"

    [How It Works](how-it-works.md) for the descent and its failure mode,
    [Adapting Overview](adapting.md) for running it on your own taxonomy.

## Characteristics

| | |
| --- | --- |
| Retrieval | none — traverses the hierarchy |
| Model calls per item | several (one per level, plus verification) |
| Needs a retriever | no |
| Tools | `get_root_categories`, `get_children`, `get_parent`, `get_code` |
| Output | your code field + `explanation` + `trajectory` |

## Running it

```python
import dspy
from rac4os.agents.HierarchicalNavigation.BaseAgent import BaseHierarchicalNavigationAgent


class SpendNavigationSignature(dspy.Signature):
    """Classify an internal spend item into the SPEND taxonomy by descending it.

    Start at the top-level categories, compare the item against every child of the
    branch you choose, and descend one level at a time until you reach a level 3 code.
    Never emit a code that did not come from a tool call.
    """

    input_item: str = dspy.InputField(desc="the item text to classify, as recorded")
    spend_code: str = dspy.OutputField(desc="the code of the chosen category")
    explanation: str = dspy.OutputField(desc="the path taken and why")


agent = BaseHierarchicalNavigationAgent(
    classification_name="SPEND_2026",
    api_key="...",
    model_name="openai/gpt-4o-mini",
    api_base=None,
    signature=SpendNavigationSignature,
    get_root_categories_doc=(
        "Return the top-level categories of the SPEND taxonomy. Call this first."
    ),
    get_children_doc=(
        "Return the child categories of a code. Use it to descend one level at a time."
    ),
    get_code_doc=(
        "Return the full record of one code, including what it includes. Use it to "
        "verify a candidate before answering."
    ),
    get_parent_doc=(
        "Return the parent of a code. Use it to back out of a branch that turned out "
        "to be wrong."
    ),
)

prediction = agent(input_item="two boxes of A4 paper for the printer")

prediction.spend_code
# '101010'
```

### With a registered configuration

```python
from rac4os.agents.HierarchicalNavigation.HierarchicalNavigationAgent import (
    HierarchicalNavigationAgent,
)

HierarchicalNavigationAgent.CONFIGS["SPEND_2026"] = {
    "signature": SpendNavigationSignature,
    "get_root_categories_doc": "...",
    "get_children_doc": "...",
    "get_code_doc": "...",
    "get_parent_doc": "...",
}

agent = HierarchicalNavigationAgent(
    classification_name="SPEND_2026",
    api_key="...", model_name="openai/gpt-4o-mini",
)
```

Note what is missing from the constructor: no `store`. This is the only approach that
works on a taxonomy you have merely described, without indexing anything.

## Capping the loop

A descent needs roughly as many steps as your taxonomy has levels, plus a verification
call or two. The underlying `dspy.ReAct` permits 20 by default, which is generous:

```python
prediction = agent(input_item="two boxes of A4 paper for the printer", max_iters=8)
```

Set the cap to about twice your depth. An agent that exceeds that is backtracking in
circles, and the extra steps rarely produce a better answer.

## Reading the trajectory

```python
prediction.trajectory["tool_name_0"], prediction.trajectory["tool_args_0"]
# ('get_root_categories', {})

prediction.trajectory["observation_0"]
# [{'code': '10', 'description': 'Office supplies and consumables'},
#  {'code': '20', 'description': 'IT equipment and services'}]

prediction.trajectory["tool_name_1"], prediction.trajectory["tool_args_1"]
# ('get_children', {'code': '10'})

prediction.trajectory["observation_1"]
# [{'code': '1010', 'description': 'Paper and paper products', 'level': 2, ...},
#  {'code': '1020', 'description': 'Writing and drawing instruments', 'level': 2, ...}]

prediction.trajectory["tool_name_2"], prediction.trajectory["tool_args_2"]
# ('get_children', {'code': '1010'})

prediction.spend_code
# '101010'
```

The trajectory is a readable audit trail: it shows the path through your taxonomy, which
is exactly the justification a human coder would give. Where the agent went wrong is
visible as the level at which it took the wrong branch.

!!! note

    The tool calls above are the real shape of what the four tools return for the example
    taxonomy; which calls a given model makes is up to the model.

## When to use it

- You have no embedding backend, or cannot install one.
- Your taxonomy's structure carries more signal than its labels — branches are
  well-separated and the top-level distinctions are easy.
- You want an auditable path rather than a similarity score.
- You are classifying at a shallow target level, where the descent is short.

Its cost grows with depth and with the branching factor: every level is a model call, and
every call carries that level's children in the context. For a deep taxonomy with wide
levels, a [RAG-based approach](../linear-rag/usage.md) is cheaper and usually better.
