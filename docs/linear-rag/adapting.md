# Adapting Overview

This page is the end-to-end route from "I have a classification" to "linear RAG runs on
it". Nothing in it requires editing the package.

## Why adapt?

Every bundled configuration is a configuration, not control flow. The pipeline itself —
retrieve, serialise, reason — is generic over the taxonomy, so running it on a
classification of your own is a matter of supplying five things, two of which you already
have if you followed [Your Own Taxonomy](../taxonomy/overview.md).

## Components

| Component                | What you supply                                            | Page                                              |
| ------------------------ | ---------------------------------------------------------- | ------------------------------------------------- |
| Taxonomy                 | a loader, a nesting rule, a dataset name                   | [Overview](../taxonomy/overview.md)               |
| Retriever                | a `VectorStore`, filled by you                             | [Retrievers](../taxonomy/retrievers.md)           |
| Candidate representation | what each candidate carries into the prompt                | [Candidate Augmentation](../taxonomy/augmentation.md) |
| Signature                | the system prompt (docstring) and the I/O fields           | [Prompt Overview](prompts.md)                      |
| Configuration            | `signature`, `query_field`, `context_field`                | this page                                          |

## Taxonomy

Linear RAG resolves `classification_name` through the same registry as everything else,
so any dataset name you have registered works:

```python
from rac4os.classification_system import get_classification_system

cs = get_classification_system("SPEND_2026")
```

The pipeline never traverses the hierarchy, but it loads it, which means you can use it
to enrich candidates yourself (see *Candidate representation* below).

## Retriever

Build the store, fill it once, and restrict it to the codes you are willing to predict:

```python
from rac4os._vector_database.vector_store import VectorStore

store = VectorStore(
    collection_name="spend_2026_level3",
    chromadb_path="./chroma",
    embedding_function_type="sentence_transformers",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

store.index(
    classification_system=cs,
    codes_to_include=[c.code for c in cs.codes if c.level == 3],
)
```

Restricting the collection to the target level is the cheapest improvement available to a
RAG pipeline: it removes whole classes of wrong answer before the model sees anything.

## Candidate representation

What the model reads about each candidate comes from the `Code` records your loader
produced, not from the classifier. If your source documentation has inclusions or
exclusions, put them in `Code.details` and they will be in the context. If it does not,
derive them from the hierarchy:

```python
candidates = store.search_category(item, k=5)
for c in candidates:
    c["includes"] = [ch.description for ch in cs.get_children(c["code"])]
```

See [Candidate Augmentation](../taxonomy/augmentation.md) for why this matters more than
anything else on this page when you classify to inner categories.

## Signature

One signature per taxonomy. The docstring is the system prompt; the fields are the
contract:

```python
import dspy


class SpendRAGCoTSignature(dspy.Signature):
    """Classify an internal spend item into the SPEND taxonomy.

    You are given one item as recorded by a member of staff, plus candidate categories
    retrieved from the taxonomy. Choose exactly one.

    Reasoning protocol:
    - Read every candidate before judging. Compare the item against the `includes`
      entries, which enumerate what a category actually covers; the short description
      alone is often broader than it appears.
    - Where two candidates both fit, prefer the one whose `includes` names the item's
      concrete form.
    - Never emit a code that is not in the candidate list. If none fits, choose the
      closest and say in the explanation why the fit is poor.

    The target level is level 3. Do not emit level 1 or level 2 codes.
    """

    input_item: str = dspy.InputField(
        desc="the item text to classify, as recorded"
    )
    retrieved_candidates: str = dspy.InputField(
        desc="candidate categories as a JSON list of objects with code, description "
             "and includes"
    )
    spend_code: str = dspy.OutputField(
        desc="the code of the chosen category, exactly as it appears in the candidates"
    )
    explanation: str = dspy.OutputField(
        desc="why this category was chosen and why the competing candidates were not"
    )
```

Three rules of thumb:

- **Name the output field for your taxonomy** (`spend_code`, not `code`). Field names are
  part of the prompt the model sees.
- **State the target level explicitly** if you classify to inner categories. A model given
  a mixed-level candidate set will otherwise answer at whatever level looks most specific.
- **Keep prompt and augmentation in agreement.** A docstring that tells the model to judge
  on `includes` is misleading if your candidates carry no `includes`.

## Configuration

Either pass the three configuration values directly to the base class:

```python
from rac4os.rag.BaseRAG import BaseRAGCoT

classifier = BaseRAGCoT(
    classification_name="SPEND_2026",
    api_key="...",
    model_name="openai/gpt-4o-mini",
    signature=SpendRAGCoTSignature,
    query_field="input_item",
    context_field="retrieved_candidates",
    store=store,
)
```

…or register them once and construct by name:

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

The keys are the base constructor's parameter names — the registry entry is literally a
bundle of keyword arguments. `query_field` and `context_field` must name input fields
that exist on the signature; both are asserted at construction, so a rename fails
immediately rather than at the first call.

Registering is also what [Rewrite-Retrieve-Read](rrr.md) needs, since it reads its
configuration from the same `CONFIGS`.

## End-to-End Example

```python
import dspy, pandas as pd
from rac4os.dataloaders.BaseDataLoader import ClassificationLoader
from rac4os.dataloaders.registry import register as register_loader
from rac4os._shared.Code import Code
from rac4os.classification_system.BaseClassificationSystem import ClassificationSystem
from rac4os.classification_system.registry import register as register_system
from rac4os.classification_system.mapping import mapping
from rac4os.classification_system import get_classification_system
from rac4os._vector_database.vector_store import VectorStore
from rac4os.rag.BaseRAG import BaseRAGCoT


# 1. taxonomy -------------------------------------------------------------
@register_loader("SPEND_CSV")
class SpendLoader(ClassificationLoader):
    def to_records(self) -> list[dict]:
        df = pd.read_csv(self.path, dtype=str).fillna("")
        return [
            Code(code=r["code"], description=r["label"], level=int(r["level"]),
                 details={"includes": r["includes"].split(";") if r["includes"] else []}
                 ).to_dict()
            for _, r in df.iterrows()
        ]


@register_system("SPEND")
class SpendTaxonomy(ClassificationSystem):
    def get_root_categories(self) -> list[dict]:
        return [{"code": c.code, "description": c.description}
                for c in self.codes if len(c.code) == 2]

    def get_prefixes(self, code: str) -> list[str]:
        return [code[:i] for i in range(2, len(code) + 1, 2)]


mapping["SPEND_2026"] = {
    "classification_system": "SPEND",
    "loader_name": "SPEND_CSV",
    "data_path": "spend.csv",
    "metadata": {"url": "internal", "version": "2026"},
}

cs = get_classification_system("SPEND_2026")

# 2. retriever ------------------------------------------------------------
store = VectorStore(
    collection_name="spend_2026_level3",
    chromadb_path="./chroma",
    embedding_function_type="sentence_transformers",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
store.index(classification_system=cs,
            codes_to_include=[c.code for c in cs.codes if c.level == 3])

# 3. signature ------------------------------------------------------------
class SpendRAGCoTSignature(dspy.Signature):
    """Classify an internal spend item into the SPEND taxonomy. Choose exactly one of
    the retrieved candidates, judging on their `includes` entries, and never emit a code
    that is not among them. The target level is level 3."""

    input_item: str = dspy.InputField(desc="the item text to classify, as recorded")
    retrieved_candidates: str = dspy.InputField(desc="candidate categories as JSON")
    spend_code: str = dspy.OutputField(desc="the code of the chosen category")
    explanation: str = dspy.OutputField(desc="why this category and not the others")


# 4. classify -------------------------------------------------------------
classifier = BaseRAGCoT(
    classification_name="SPEND_2026",
    api_key="...",
    model_name="openai/gpt-4o-mini",
    signature=SpendRAGCoTSignature,
    query_field="input_item",
    context_field="retrieved_candidates",
    store=store,
)

prediction = classifier(query="two boxes of A4 paper for the printer", k=3)
print(prediction.spend_code, "|", prediction.explanation)
# 101010 | Chosen 101010 because the item is copying paper, which that category covers ...
```

## Next steps

- Compare against [agentic RAG](../agentic-rag/adapting.md) on the same taxonomy and the
  same store — the output contract is identical, so only the classifier changes.
- If your inner categories have thin descriptions, generate better ones with
  [Taxonomy Refinement](../refinement/usage.md) and register the result as a second
  dataset.
