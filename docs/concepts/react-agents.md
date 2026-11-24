# ReAct agents

Both agents follow the same layout: a `Base…` class holding all the machinery, and a
thin subclass that supplies nothing but a configuration. The subclass is the whole
file:

```python
class RAGAgent(BaseRAGAgent):
    CONFIGS = registry                     # from ./registry.py

    def __init__(self, classification_name, **kwargs):
        if classification_name not in self.CONFIGS:
            raise ValueError(f"Unknown classification system: {classification_name}")
        super().__init__(classification_name=classification_name,
                         **self.CONFIGS[classification_name], **kwargs)
```

So the registry entry is literally a bundle of keyword arguments for the base
constructor, and its keys are the base's parameter names — `signature` plus
`get_root_categories_doc` / `get_children_doc` / `get_code_doc` / `get_parent_doc`
for the navigation agent, `signature` plus `search_category_tool_doc` for the RAG one.
Rename a base parameter and the registry must follow.

There is no longer a `system_prompt` key, and nothing mutates `signature.__doc__` at
construction: **each system prompt lives in its `dspy.Signature` docstring**, where
dspy reads it anyway. Prompt and I/O contract are now edited in one place.

`classification_name` is passed straight through to `get_classification_system`, so
it is the dataset key `mapping.py` uses, and both registries are keyed on those same
strings — `HierarchicalNavigation/registry.py` on `COICOP_2018`, `ICATUS_2016`,
`NAF_rev2`, `SEA_2021` and `SEA_2021_REPHRASED`, `RAG/registry.py` on the same set
minus `NAF_rev2`. A key that is not in the registry raises
`ValueError: Unknown classification system: …` before any model or embedding work
starts.

**[HierarchicalNavigationAgent](https://github.com/adimo20/hcs4os/blob/master/hcs4os/agents/HierarchicalNavigation/HierarchicalNavigationAgent.py)** —
`dspy.ReAct` over four thin tool methods that forward to the classification system
and convert `Code` objects to dicts. Its system prompt lays out a descent protocol:
start at the roots, compare every child, _follow the `excludes` pointers instead of
forcing a fit_, verify the leaf, backtrack if excluded, and never produce a code that
did not come from a tool.

**[RAGAgent](https://github.com/adimo20/hcs4os/blob/master/hcs4os/agents/RAG/RAGAgent.py)** — retrieve, then choose. It takes a
ready-made, already-indexed `store` (see section 5) rather than embedding parameters:
it never indexes anything itself — filling the store with `store.index(...)` is the
caller's job — and is a `dspy.ReAct` over a single tool, the store's own
`search_category`, handed to the model under `search_category_tool_doc`. The signature
docstring drives the loop — search with a focused query, read the includes/excludes
notes on the candidates, refine the query and search again when the notes point
elsewhere, and never emit a code the tool did not return. Its outputs are the domain
code (`coicop_code` / `icatus_code` / `sea_code`) and `explanation`.

### The `Base…` variants

[BaseAgent.py](https://github.com/adimo20/hcs4os/blob/master/hcs4os/agents/HierarchicalNavigation/BaseAgent.py) and
[BaseAgentRAG.py](https://github.com/adimo20/hcs4os/blob/master/hcs4os/agents/RAG/BaseAgentRAG.py) hold the machinery: the LM, the
classification system, the tool methods, and the `dspy.ReAct`. Every registered agent
is one of these with a `CONFIGS` lookup in front, so instantiating a base class
directly is the same code path, just with the prose passed in by hand.

Use them to run a classification the registry does not cover, or to A/B a prompt
without editing shared files; the registry subclasses stay the right home for a prompt
you intend to keep. `classification_name` still goes to `get_classification_system`,
so the classification itself must be in `mapping.py` either way.

Restricting the collection to the level actually being predicted, or to one branch, is
the cheapest available lever on retrieval quality — it removes whole classes of wrong
answers before the model ever sees them. That restriction is applied when you fill the
store (`store.index(codes_to_include=[...])`), not on the agent. `BaseRAGAgent` still
accepts a `codes_to_include` argument for symmetry with the other classifiers, but
since the agent no longer does its own indexing the argument is currently inert —
narrow the collection at `store.index` instead.

### `forward()` and usage tracking

Both base agents define `forward(**kwargs)`, so the module is callable directly and
`agent.agent(...)` is no longer needed. It does two things beyond delegating to the
`dspy.ReAct`:

- **Validates the input fields.** `set(signature.input_fields) - kwargs.keys()` is
  checked first, so a typo'd or forgotten field raises
  `ValueError: Missing input fields: ['input_expense']` immediately instead of
  surfacing as a confusing dspy error several calls later.
- **Pins the LM and tracks usage.** The body runs inside
  `dspy.context(lm=self.lm, track_usage=True)`, so each classifier uses the LM it was
  constructed with regardless of global dspy state, and `prediction.get_lm_usage()`
  carries the token counts. The flip side: `dspy.configure(lm=…)` at call time has no
  effect — assign `agent.lm` instead.
