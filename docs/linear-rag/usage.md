# Usage

`RAGChainOfThought` · `BaseRAGCoT`

Linear RAG retrieves once and reasons once: the item is embedded, the _k_ nearest
categories are pulled from your [retriever](../taxonomy/retrievers.md), and a
chain-of-thought generation picks one of them. It is one retrieval call and one model
call per item, which makes it the approach to try first.

!!! tip "See also"

    [How It Works](how-it-works.md) for what the pipeline does and where it breaks,
    [Adapting Overview](adapting.md) for running it on your own taxonomy.

## Characteristics

|                      |                                               |
| -------------------- | --------------------------------------------- |
| Retrieval            | once, on the raw item                         |
| Model calls per item | one                                           |
| Needs a retriever    | yes, filled by you                            |
| `k` chosen by        | you, per call                                 |
| Output               | your code field + `explanation` + `reasoning` |

## Running it

```python
import dspy
from rac4os.rag.BaseRAG import BaseRAGCoT

classifier = BaseRAGCoT(
    classification_name="SPEND_2026",          # a registered dataset name
    api_key="...",
    model_name="openai/gpt-4o-mini",
    api_base=None,                              # set for a self-hosted endpoint
    signature=SpendRAGCoTSignature,             # its docstring is the system prompt
    query_field="input_item",                   # which input field is the item
    context_field="retrieved_candidates",       # which input field is the candidate list
    store=store,                                # a filled Retriever
)

prediction = classifier(query="two boxes of A4 paper for the printer", k=3)

prediction.spend_code
# '101010'
```

`forward(query, k, where=None)` performs the retrieval, serialises the candidates to
JSON, and passes item and context to the generation under the two field names you
declared. Both `k` and the optional metadata filter are yours to choose per call.

### With a registered configuration

If you would rather name the configuration once and reuse it, put it in `CONFIGS` and
construct by name:

```python
from rac4os.rag.RAG import RAGChainOfThought

RAGChainOfThought.CONFIGS["SPEND_2026"] = {
    "signature": SpendRAGCoTSignature,
    "query_field": "input_item",
    "context_field": "retrieved_candidates",
}

classifier = RAGChainOfThought(
    classification_name="SPEND_2026",
    api_key="...", model_name="openai/gpt-4o-mini", store=store,
)
```

Both forms are the same code path. `RAGChainOfThought` is `BaseRAGCoT` with a `CONFIGS`
lookup in front of it.

## Restricting the candidate space

Narrow retrieval to one branch or one level per call with a metadata filter:

```python
prediction = classifier(
    query="docking station for the new laptops",
    k=5,
    where={"level": 3},
)
```

`where` is passed straight to the store and matches the metadata as your loader emitted
it — if `level` left your loader as a string, filter with `{"level": "3"}`. Restricting
the collection _as a whole_ happens when you fill it, with
`store.index(codes_to_include=[...])`.

## Reading the prediction

```python
prediction = classifier(query="two boxes of A4 paper for the printer", k=3)

prediction.spend_code
# '101010'

print(prediction.reasoning)
# The item is paper bought for use in a printer. Candidate 101010 "Printing and copying
# paper" lists copier paper and inkjet paper under includes, which is what A4 paper for a
# printer is. Candidate 101020 "Hygiene paper" covers towels and tissues, a different
# purpose despite also being paper. Candidate 102010 covers writing instruments.

print(prediction.explanation)
# Chosen 101010 because the item is copying paper, explicitly covered by that category's
# includes. 101020 was rejected because it covers hygiene paper, and 102010 because it
# covers writing instruments.

prediction.get_lm_usage()
# {'openai/gpt-4o-mini': {'prompt_tokens': 712, 'completion_tokens': 188, ...}}
```

`reasoning` comes from the chain-of-thought and is the model's working; `explanation` is
a declared output field and is the justification you keep next to the code. In a
production coding context the second one is what a human reviewer reads.

!!! note

    The reasoning and explanation above are illustrative — the shape of what a generation
    returns, not a measured result. The retrieved candidates and all non-model output
    shown in these docs are real output from the example taxonomy.

## Batch classification

There is no batch API; a loop is the intended usage, and `dspy` handles one call at a
time:

```python
rows = []
for item in items:
    p = classifier(query=item, k=5)
    rows.append({
        "item": item,
        "code": p.spend_code,
        "explanation": p.explanation,
        "usage": p.get_lm_usage(),
    })
```

Keep the usage counts: they are the only cheap signal on what a configuration costs
before you run it at volume. If you want to apply batch processing adapt [`dpsy.Parallel`](https://dspy.ai/current/api/modules/Parallel/).
All classifiers of `rac4os` are dspy Modules that can be parallelised by the common dspy approach.

## Common pitfalls

- **An empty candidate list** almost always means an empty or wrongly named collection.
  `BaseRAGCoT` never indexes — see [Retrievers](../taxonomy/retrievers.md).
- **`codes_to_include` on the constructor is inert.** It is accepted for symmetry with
  the other classifiers; restrict the candidate space at `store.index` instead.
- **A missing input field raises immediately.** `forward` checks the signature's input
  fields first, so a typo surfaces as
  `ValueError: Missing input fields: ['input_item']` rather than as an obscure failure
  later.
- **`forward` here does not pin the LM.** `BaseRAGCoT` relies on the `dspy.configure`
  call made in its constructor, so if you run several classifiers with different LMs in
  one process, construct each one immediately before use.
