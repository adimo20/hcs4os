# How It Works

## From a pipeline to an agent

[Linear RAG](../linear-rag/how-it-works.md) is bound by its initial query: retrieval runs
once, before any reasoning, and if the correct category is not in the candidate set the
generation cannot recover it.

Agentic RAG removes that constraint by not invoking the retriever automatically. The
model receives the item and decides what to do: which query to search with, how many
candidates to ask for, whether what came back is good enough, and whether to search
again with different wording. Retrieval becomes an action rather than a stage.

```
item ──► reason ──► search(query, k) ──► observe candidates ──┐
            ▲                                                 │
            └──────────── not good enough, re-query ──────────┘
                                   │
                              good enough
                                   │
                                   ▼
                          code + explanation
```

## The loop

The implementation is a `dspy.ReAct` over a single tool: the store's own
`search_category`. ReAct interleaves reasoning and action — the model writes a thought,
picks a tool and its arguments, sees the result, and reasons again with that result in
hand. The alternative, chain-of-thought, commits to its entire reasoning path before
seeing any result, which is cheaper and gives it no way to react to what retrieval
returns.

Each step appends four entries to the trajectory:

| Entry             | Content                                       |
| ----------------- | --------------------------------------------- |
| `thought_{i}`     | why the agent is about to do what it does     |
| `tool_name_{i}`   | the tool chosen — here always `search_category` (or `finish`) |
| `tool_args_{i}`   | the query and `k` the model decided on        |
| `observation_{i}` | what the retriever returned                   |

The loop ends when the agent calls `finish`, when the step cap is reached, or when the
context window is exhausted.

## What this buys and what it costs

**Buys:** iterative query refinement. The agent can reformulate when the first query was
badly worded, expand when the results are too narrow, and in effect compensate for a
retriever that is not good enough to be trusted with a single shot — which is exactly the
situation when you have no labelled data to fine-tune an embedding model with. It also
produces a readable trace of how it got there.

**Costs:** every step is a model call *and* a retrieval call, and classification becomes
a multi-turn generation. Expect latency on the order of tens of seconds per item rather
than one round trip. That rules the approach out for interactive use — classifying while
a respondent types — and leaves it well suited to batch pre-coding in a human-in-the-loop
workflow, where it reduces manual effort rather than competing with real-time
expectations.

A second cost is reliability. The loop depends on the model emitting well-formed tool
calls at every step. Models that handle the linear prompt comfortably can still fail here,
and those failures look like errors in instruction-following rather than mistakes about
the classification.

## Components

| Component                  | Role                                                 | Where it comes from                               |
| -------------------------- | ---------------------------------------------------- | ------------------------------------------------- |
| classification system      | the taxonomy, loaded from `classification_name`      | [Classification Systems](../taxonomy/systems.md)  |
| retriever                  | exposed to the model as the `search_category` tool   | [Retrievers](../taxonomy/retrievers.md)           |
| `search_category_tool_doc` | the tool description the model reads before calling  | [Prompt Overview](prompts.md)                      |
| signature                  | the system prompt (docstring) and the I/O contract   | [Prompt Overview](prompts.md)                      |
| `dspy.ReAct`               | the loop                                             | built from signature + tool                        |
| LM                         | `model_name`, `api_key`, `api_base`                  | [Installation](../getting-started/installation.md) |

Note the asymmetry with the linear approach: there is no `context_field`, because the
candidates do not arrive as an input field. They arrive as observations inside the
trajectory, which also means the agent sees whatever shape `search_category` returns — the
`Code` record as a dict, including whatever you put in
[`details`](../taxonomy/augmentation.md).

## What `forward` does

```python
def forward(self, **kwargs):
    missing = set(self.signature.input_fields) - kwargs.keys()
    if missing:
        raise ValueError(f"Missing input fields: {sorted(missing)}")

    with dspy.context(lm=self.lm, track_usage=True):
        return self.agent(**kwargs)
```

Two things beyond delegating to the loop: input fields are validated up front, so a
forgotten field fails immediately; and the LM is pinned per call, so the agent uses the
model it was constructed with regardless of global `dspy` state. The flip side is that
`dspy.configure(lm=...)` at call time has no effect — assign `agent.lm` instead.
