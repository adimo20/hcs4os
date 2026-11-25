# How It Works

## Classifying by descent

A statistical taxonomy is a hierarchy of categories that get progressively more specific,
and a human coder uses that: they decide the broad group first, then the subgroup, then
the category. The traversal agent does the same thing. Instead of finding candidates by
similarity, it uses the structure your
[classification system](../taxonomy/systems.md) already exposes.

```
get_root_categories()  ──►  choose a branch
        │
        ▼
get_children(branch)   ──►  compare the item against every child, choose one
        │
        ▼                        (repeat per level)
get_code(candidate)    ──►  verify what the category actually covers
        │
        ▼
code + explanation
```

The agent is a `dspy.ReAct` over four tools that forward directly to the classification
system and convert `Code` objects to dicts:

| Tool                  | Forwards to                      | Used for                                |
| --------------------- | -------------------------------- | --------------------------------------- |
| `get_root_categories` | `cs.get_root_categories()`       | where to start                          |
| `get_children`        | `cs.get_children(code)`          | descending one level                    |
| `get_code`            | `cs.get_code(code)`              | verifying a candidate before committing |
| `get_parent`          | `cs.get_parent(code)`            | backing out of a wrong branch           |

Because the tools are thin wrappers, the agent's view of your taxonomy is exactly what you
described — including whatever you put in `details`, which is what the verification step
reads.

## Why a loop rather than one call

The alternative is putting the whole taxonomy in one prompt. That works, and it costs the
entire code index in tokens for every item — which does not scale with the size of the
classification. Descending instead means each call carries one level's children, so the
context stays small and roughly constant regardless of how large the taxonomy is.

The exchange is the number of calls. One classification is several model calls, and each
one depends on the previous choice, which is the structural weakness below.

## The failure mode

**A wrong turn early is unrecoverable unless the agent backtracks.** If the item belongs
under *Office supplies* and the agent descends into *IT equipment*, nothing further down
that branch will fit, and the correct category is never shown to it. This is the
hierarchical analogue of the
[retrieval bottleneck](../linear-rag/how-it-works.md#the-retrieval-bottleneck): an early
decision made with the least information constrains everything that follows.

Two things mitigate it, and both belong in the [prompt](prompts.md):

- **Compare before descending.** The instruction to read *every* child of a level before
  choosing, rather than taking the first plausible one, is what prevents most wrong turns.
- **Backtrack explicitly.** `get_parent` exists so that an agent which finds nothing
  fitting at a level can go back up and try a sibling branch. An agent that is not told
  it may do this will instead force the item into the best of the wrong options.

Where a taxonomy's top-level distinctions are genuinely hard — and in consumption or
occupation classifications they often are, because an item's purpose rather than its
nature decides the branch — this approach is at its weakest, and retrieval, which is not
committed to a branch at all, is at its strongest.

## Components

| Component                | Role                                               | Where it comes from                               |
| ------------------------ | -------------------------------------------------- | ------------------------------------------------- |
| classification system    | the taxonomy; the agent's entire environment       | [Classification Systems](../taxonomy/systems.md)  |
| four tool descriptions   | what the model reads when choosing a tool          | [Prompt Overview](prompts.md)                      |
| signature                | the system prompt (docstring) and the I/O contract | [Prompt Overview](prompts.md)                      |
| `dspy.ReAct`             | the loop                                           | built from signature + four tools                  |
| LM                       | `model_name`, `api_key`, `api_base`                | [Installation](../getting-started/installation.md) |

There is no retriever, no embedding model and no collection — so there is also nothing to
index, nothing to keep in sync with an embedding model, and no vector-store extra to
install.

## What `forward` does

```python
def forward(self, **kwargs):
    missing = set(self.signature.input_fields) - kwargs.keys()
    if missing:
        raise ValueError(f"Missing input fields: {sorted(missing)}")

    with dspy.context(lm=self.lm, track_usage=True):
        return self.agent(**kwargs)
```

Input fields are validated up front, and the LM is pinned per call so the agent uses the
model it was constructed with regardless of global `dspy` state.
