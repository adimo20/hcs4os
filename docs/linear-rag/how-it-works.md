# How It Works

## The pipeline

Retrieval-Augmented Generation gives a language model access to knowledge it does not
have in its weights — here, your classification. In its naïve form the pipeline has two
components and runs each of them once:

```
item ──► embed ──► nearest k categories ──► JSON context ──► chain-of-thought ──► code + explanation
             │                                      │
        your retriever                         your signature
```

1. **Index** (once, up front). Every category is embedded into a vector store. No
   chunking happens — a category label is already a short text, and splitting it would
   destroy the unit of retrieval.
2. **Retrieve.** The item is embedded with the same model and the _k_ nearest categories
   are returned.
3. **Augment.** Those candidates are serialised to JSON and placed in the signature's
   context field, next to the item. What each candidate looks like is
   [yours to decide](../taxonomy/augmentation.md).
4. **Read.** A `dspy.ChainOfThought` reasons over the candidates and emits your code
   field plus an explanation.

The cost per item is a function of _k_, not of the size of your taxonomy, and nothing
needs labelled data: the knowledge base is built from the classification's own
documentation.

## Why chain-of-thought

Choosing between candidate categories is a comparison under exclusion rules, not a
lookup. The model has to weigh a candidate against the others, notice that a more
specific category exists, or see that an apparently fitting label is excluded by its own
notes. Reasoning before answering suits that, and it leaves a trace a reviewer can audit
— which is why `explanation` is a declared output field rather than an afterthought.

Chain-of-thought produces its whole reasoning path in one pass, before any observation
comes back. That is exactly what makes it cheap, and exactly what distinguishes it from
the [agentic](../agentic-rag/how-it-works.md) variant.

## The retrieval bottleneck

The structural weakness is worth understanding before you choose this approach:
**the pipeline is bound by its initial query.** Retrieval happens once, on the raw item,
before any reasoning. If the correct category is not among the _k_ retrieved, no amount
of generator quality recovers it — the answer is not in the room.

In practice this, rather than generation quality, is usually what limits a RAG
classification pipeline. Two consequences:

- **The retriever's accuracy at _k_ is a ceiling on the pipeline.** Measure it on its own
  before touching any prompt. Whether the correct code is present in the candidate set
  tells you immediately whether you have a retrieval problem or a generation problem, and
  the two are fixed by completely different work.
- **Raising _k_ trades one error for another.** A larger candidate set is more likely to
  contain the correct category and gives the model more opportunity to pick a wrong one.
  It also spends context, which competes with
  [how richly each candidate is described](../taxonomy/augmentation.md).

The two other approaches in the package exist to address this bottleneck from different
directions: [Rewrite-Retrieve-Read](rrr.md) improves the single query,
[agentic RAG](../agentic-rag/how-it-works.md) allows more than one.

## Components

| Component                       | Role                                                   | Where it comes from                                |
| ------------------------------- | ------------------------------------------------------ | -------------------------------------------------- |
| classification system           | the taxonomy; loaded from `classification_name`        | [Classification Systems](../taxonomy/systems.md)   |
| retriever                       | finds the candidates                                   | [Retrievers](../taxonomy/retrievers.md)            |
| signature                       | the system prompt and the I/O contract                 | [Prompt Overview](prompts.md)                      |
| `query_field` / `context_field` | which input field holds the item, which the candidates | your signature's field names                       |
| `dspy.ChainOfThought`           | the generation                                         | built from the signature                           |
| LM                              | `model_name`, `api_key`, `api_base`                    | [Installation](../getting-started/installation.md) |

The classification system is loaded even though the linear pipeline never traverses it:
it is there so that the same `classification_name` means the same thing across all
approaches, and so you can enrich candidates from the hierarchy yourself.

## What `forward` does

```python
def forward(self, query: str, k: int, where: dict | None = None):
    retrieved_context = self.store.search_category(query, k, where=where)
    retrieved_context_str = json.dumps(retrieved_context, indent=4, ensure_ascii=False)
    return self.CoT(**{
        self.query_field: query,
        self.context_field: retrieved_context_str,
    })
```

That is the entire pipeline — the three runtime steps are four lines, because everything
specific to your taxonomy lives in the store you passed and the signature you wrote.
