# Rewrite-Retrieve-Read

`RRR_RAGChainOfThought` · `QueryRephraser`

Also referred to as Rephrase-Retrieve-Response. This is linear RAG with one step added in
front of retrieval, and it addresses the most common reason retrieval misses: the item is
worded nothing like your labels.

## Why rewrite the query

Dense retrieval compares meaning, but it still compares *an item as recorded* against
*a label as published*, and those are written by different people for different purposes.
`2x A4 Papier Kopierer` and *Printing and copying paper* are the same thing in two
registers. Abbreviations, brand names, pack sizes, local shorthand and typos all push the
query's embedding away from the label's.

Rewriting the query before embedding it is a pre-retrieval step: a model restates the raw
item in the vocabulary the taxonomy uses, and only that restatement is embedded. It keeps
the single-retrieval cost structure of [linear RAG](how-it-works.md) — one extra, cheap
model call — while removing the wording mismatch that the retriever cannot see past.

## The three steps

```
item ──► rewrite ──► rewritten query ──► embed ──► nearest k ──┐
  │                                                            │
  └────────────────── the ORIGINAL item ──────────────────────► chain-of-thought ──► code
```

1. **Rewrite.** A `QueryRephraser` turns the item into a retrieval query in the
   vocabulary of your labels. It does not classify.
2. **Retrieve.** The rewritten query is embedded and the *k* nearest categories are
   pulled.
3. **Read.** The chain-of-thought classifies the **original** item over those candidates.

Step 3 is the part worth noticing. The rewrite only steers retrieval; it never replaces
the item the model has to classify, so detail the rephraser dropped — a brand, a quantity,
a qualifier that decides between two categories — is not lost on the way.

## Running it

The rephraser takes its own LM, which can be smaller and cheaper than the one doing the
classification:

```python
import dspy
from rac4os.rag.RAG import RAGChainOfThought, QueryRephraser, RRR_RAGChainOfThought


class SpendRephrasingSignature(dspy.Signature):
    """Restate an internal spend item in the vocabulary of the SPEND taxonomy labels.

    Expand abbreviations, drop quantities and brand names, and name the generic product
    or service in the wording a category label would use. Do not classify, do not guess
    a code, and do not add information the item does not contain.
    """

    input_item: str = dspy.InputField(desc="the item text as recorded")
    rephrased_query: str = dspy.OutputField(
        desc="a short retrieval query using the vocabulary of the category labels"
    )


RAGChainOfThought.CONFIGS["SPEND_2026"] = {
    "signature": SpendRAGCoTSignature,
    "query_field": "input_item",
    "context_field": "retrieved_candidates",
}

rephraser = QueryRephraser(
    lm=dspy.LM(api_key="...", model="openai/gpt-4o-mini"),   # the small model
    classification_name=None,                                 # not using the bundled map
    signature=SpendRephrasingSignature,
)

classifier = RRR_RAGChainOfThought(
    rephraser,                                   # first positional argument
    classification_name="SPEND_2026",
    api_key="...",
    model_name="openai/gpt-4o-mini",             # the classifying model
    store=store,
)

prediction = classifier(query="2x A4 Papier Kopierer", k=3)
```

`RRR_RAGChainOfThought` subclasses `RAGChainOfThought`, so it reads its signature and
field names from `CONFIGS` — hence the entry above. The rephraser is passed separately
because it is a module with its own LM and its own prompt.

!!! tip

    `QueryRephraser` accepts either a `classification_name` that is in its bundled
    mapping, or an explicit `signature`. For your own taxonomy, pass
    `classification_name=None` and your signature; the input field name is read off the
    signature, so the caller never has to know it.

## Reading the prediction

The prediction carries everything linear RAG returns plus the rewritten query:

```python
prediction = classifier(query="2x A4 Papier Kopierer", k=3)

prediction.rephrased_query
# 'printing and copying paper, A4 sheets for an office copier'

prediction.spend_code
# '101010'

prediction.explanation
# 'Chosen 101010 because the item is copying paper for a printer; 101020 covers hygiene
#  paper and was rejected.'
```

!!! note

    Illustrative output — the shape of the prediction, not a measured result.

`rephrased_query` is the diagnostic that makes this approach worth its extra call. Log
it. If the rewrites read like labels from your taxonomy, the step is doing its job; if
they read like paraphrases of the input, or invent specifics the item never contained,
the rephrasing prompt needs work rather than the classification prompt.

## When to use it

Reach for the rewrite step when:

- retrieval misses on items whose wording is far from your labels — abbreviations, codes,
  brand names, another language register;
- you cannot fine-tune the embedding model, so the mismatch cannot be fixed on the
  retrieval side;
- latency and cost matter, so the [agentic](../agentic-rag/usage.md) loop is too
  expensive.

Skip it when retrieval already finds the right category reliably. It adds a model call
and one more prompt to maintain, and a rephraser that drops a decisive detail can make
retrieval worse than the raw query.

!!! warning "`retrieve_with_original`"

    Earlier drafts of this documentation described an option to retrieve with both the
    original and the rewritten query and merge the candidate sets. It is not implemented;
    retrieval uses the rewritten query only.
