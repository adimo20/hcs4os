# Usage

`RAGAgent` · `BaseRAGAgent`

Agentic RAG hands the retriever to the model as a tool. Instead of retrieving once on the
raw item, the agent decides when to search, with what query and with what *k*, reads what
came back, and searches again when the results do not fit.

!!! tip "See also"

    [How It Works](how-it-works.md) for what the agent decides and what it costs,
    [Adapting Overview](adapting.md) for running it on your own taxonomy.

## Characteristics

| | |
| --- | --- |
| Retrieval | as often as the agent calls for it |
| Model calls per item | several (one per step, plus the final answer) |
| Needs a retriever | yes, filled by you |
| `k` chosen by | the model, per call |
| Output | your code field + `explanation` + `trajectory` |

## Running it

```python
import dspy
from rac4os.agents.RAG.BaseAgentRAG import BaseRAGAgent


class SpendRAGAgentSignature(dspy.Signature):
    """Classify an internal spend item into the SPEND taxonomy by searching it.

    Search with a focused query describing the generic product or service, read the
    `includes` of every candidate returned, and search again with different wording when
    none of them fits. Never emit a code the search tool did not return.
    """

    input_item: str = dspy.InputField(desc="the item text to classify, as recorded")
    spend_code: str = dspy.OutputField(desc="the code of the chosen category")
    explanation: str = dspy.OutputField(desc="why this category and not the others")


agent = BaseRAGAgent(
    classification_name="SPEND_2026",
    api_key="...",
    model_name="openai/gpt-4o-mini",
    api_base=None,
    signature=SpendRAGAgentSignature,
    search_category_tool_doc=(
        "Search the SPEND taxonomy for candidate categories. Pass `query`, a short "
        "description of the product or service, and `k`, the number of candidates to "
        "return. Call it again with different wording if the results do not fit."
    ),
    store=store,
)

prediction = agent(input_item="docking station for the new laptops")

prediction.spend_code
# '201020'
```

Note that there is no `query=` argument and no `k=`: the input is passed under your
signature's own field name, and the agent chooses its queries itself.

### With a registered configuration

```python
from rac4os.agents.RAG.RAGAgent import RAGAgent

RAGAgent.CONFIGS["SPEND_2026"] = {
    "signature": SpendRAGAgentSignature,
    "search_category_tool_doc": "Search the SPEND taxonomy ...",
}

agent = RAGAgent(
    classification_name="SPEND_2026",
    api_key="...", model_name="openai/gpt-4o-mini", store=store,
)
```

## Capping the loop

The underlying `dspy.ReAct` allows up to 20 steps by default. For classification that is
far more than useful — a handful of searches is normally where the agent either has the
answer or is going in circles. Cap it per call:

```python
prediction = agent(input_item="docking station for the new laptops", max_iters=5)
```

The cap is the main cost control you have. Each step is a model call plus a retrieval
call, so an uncapped agent on an ambiguous item is the worst case for both latency and
spend.

## Reading the trajectory

The prediction carries the agent's whole trace, which is the main reason to run this
approach rather than a linear one:

```python
prediction.trajectory["thought_0"]
# 'The item is a docking station, a computer peripheral. I will search for that.'

prediction.trajectory["tool_name_0"]
# 'search_category'

prediction.trajectory["tool_args_0"]
# {'query': 'docking station', 'k': 5}

prediction.trajectory["observation_0"]
# [{'code': '201020', 'description': 'Peripherals', 'level': 3,
#   'details': '{"includes": ["keyboards", "mice", "monitors", "docking stations"]}'},
#  {'code': '201010', 'description': 'Notebooks and tablets', ...}]

prediction.spend_code
# '201020'
prediction.explanation
# 'Chosen 201020: docking stations are listed among its includes. 201010 covers the
#  laptops themselves, not accessories for them.'
```

!!! note

    The thoughts and the explanation above are illustrative. The observations are the real
    shape of what `search_category` returns for the example taxonomy.

Keys are numbered per step: `thought_0`, `tool_name_0`, `tool_args_0`, `observation_0`,
then `_1`, and so on. Three things are worth logging from it:

- **the number of steps** — how often the agent needed more than one search;
- **`tool_args_*`** — the queries it invented, which is the cheapest insight you will get
  into how your items differ from your labels;
- **whether the correct code ever appeared in an observation** — if it did and the agent
  still answered wrongly, that is a prompt problem, not a retrieval problem.

## Common pitfalls

- **Structured-output failures.** The agent depends on the model following `dspy`'s
  structured generation scheme for tool calls. Smaller or heavily quantized models fail
  at this and produce parser errors that are not classification errors. If a model throws
  these frequently, it is not usable in this approach even if it performs well in the
  [linear](../linear-rag/usage.md) one.
- **`codes_to_include` is inert.** The agent does not index, so narrowing happens at
  `store.index(codes_to_include=[...])`.
- **The tool description does real work.** `search_category_tool_doc` is what the model
  reads when deciding how to call the retriever — including that it may pass `k`. A vague
  tool description produces one timid search and no re-query.
