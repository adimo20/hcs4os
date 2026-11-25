# Quickstart

This page runs each of the three approaches once, against a taxonomy of your own. The
taxonomy used throughout the documentation is a small internal spend classification read
from a CSV file — three levels, two digits per level — the kind of in-house taxonomy that
has no published documentation at all.

!!! tip "See also"

    [Your Own Taxonomy > Overview](../taxonomy/overview.md) explains the three steps
    below in detail. This page is the short version.

## Your taxonomy

Save this as `spend.csv`:

```csv
code,label,level,includes
10,Office supplies and consumables,1,
1010,Paper and paper products,2,
101010,Printing and copying paper,3,copier paper;inkjet paper;continuous forms
101020,Hygiene paper,3,paper towels;toilet paper;facial tissues
1020,Writing and drawing instruments,2,
102010,Pens and pencils,3,ballpoint pens;pencils;markers
20,IT equipment and services,1,
2010,Computing hardware,2,
201010,Notebooks and tablets,3,laptops;tablets;convertibles
201020,Peripherals,3,keyboards;mice;monitors;docking stations
2020,Software and licences,2,
202010,Standard software licences,3,office suites;antivirus subscriptions
```

Describe it once — a loader that reads the file, a class that says how the codes nest,
and a name to refer to it by:

```python
import pandas as pd
from rac4os.dataloaders.BaseDataLoader import ClassificationLoader
from rac4os.dataloaders.registry import register as register_loader
from rac4os._shared.Code import Code
from rac4os.classification_system.BaseClassificationSystem import ClassificationSystem
from rac4os.classification_system.registry import register as register_system
from rac4os.classification_system.mapping import mapping
from rac4os.classification_system import get_classification_system


@register_loader("SPEND_CSV")
class SpendLoader(ClassificationLoader):
    def to_records(self) -> list[dict]:
        df = pd.read_csv(self.path, dtype=str).fillna("")
        return [
            Code(
                code=row["code"],
                description=row["label"],
                level=int(row["level"]),
                details={"includes": row["includes"].split(";") if row["includes"] else []},
            ).to_dict()
            for _, row in df.iterrows()
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

print(len(cs.codes))
# 12
print(cs.get_root_categories())
# [{'code': '10', 'description': 'Office supplies and consumables'}, {'code': '20', 'description': 'IT equipment and services'}]
print(cs.get_code_trace("101010"))
# [('10', 'Office supplies and consumables'), ('1010', 'Paper and paper products'), ('101010', 'Printing and copying paper')]
```

Your taxonomy is now navigable, and that is all the hierarchy traversal agent needs.

## A retriever

The two RAG-based approaches additionally need your categories embedded into a vector
store. You build and fill it yourself; classifiers never index:

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
    codes_to_include=[c.code for c in cs.codes if c.level == 3],   # only the target level
)

store.search_category("two boxes of A4 paper for the printer", k=3)
# [{'code': '101010', 'description': 'Printing and copying paper', 'level': 3, ...},
#  {'code': '101020', 'description': 'Hygiene paper', 'level': 3, ...},
#  {'code': '102010', 'description': 'Pens and pencils', 'level': 3, ...}]
```

Indexing is a one-time cost — the collection is persisted under `chromadb_path` and
reused on later runs.

## A signature

Every approach is driven by a `dspy.Signature`: its **docstring is the system prompt**
and its fields are the input/output contract. One signature per taxonomy and approach:

```python
import dspy


class SpendRAGCoTSignature(dspy.Signature):
    """Classify an internal spend item into the SPEND taxonomy.

    You are given one item as it was recorded by a member of staff, plus a list of
    candidate categories retrieved from the taxonomy. Choose exactly one.

    - Read every candidate before judging, and compare the item against the `includes`
      entries, which enumerate what each category actually covers.
    - Where two candidates both fit, prefer the one whose `includes` names the item's
      concrete form.
    - Never emit a code that is not in the candidate list.
    """

    input_item: str = dspy.InputField(desc="the item text to classify, as recorded")
    retrieved_candidates: str = dspy.InputField(desc="candidate categories as JSON")
    spend_code: str = dspy.OutputField(desc="the code of the chosen category")
    explanation: str = dspy.OutputField(desc="why this category and not the others")
```

## Linear RAG

One retrieval call, one generation call.

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

prediction = classifier(query="two boxes of A4 paper for the printer", k=3)

prediction.spend_code
# '101010'
prediction.reasoning
# 'The item is paper bought for use in a printer. Candidate 101010 lists copier paper ...'
prediction.explanation
# 'Chosen 101010 because the item is copying paper, explicitly covered by that ...'
```

→ [Linear RAG](../linear-rag/usage.md)

## Rewrite-Retrieve-Read

A small model first restates the item in the vocabulary of your labels; only that
rewritten query is embedded. The item the classifier reasons about stays the original.

```python
from rac4os.rag.RAG import RAGChainOfThought, QueryRephraser, RRR_RAGChainOfThought

class SpendRephrasingSignature(dspy.Signature):
    """Restate an internal spend item in the vocabulary of the SPEND taxonomy labels."""
    input_item: str = dspy.InputField(desc="the item text as recorded")
    rephrased_query: str = dspy.OutputField(desc="a retrieval query using category wording")

RAGChainOfThought.CONFIGS["SPEND_2026"] = {
    "signature": SpendRAGCoTSignature,
    "query_field": "input_item",
    "context_field": "retrieved_candidates",
}

rephraser = QueryRephraser(
    lm=dspy.LM(api_key="...", model="openai/gpt-4o-mini"),
    classification_name=None,
    signature=SpendRephrasingSignature,
)

classifier = RRR_RAGChainOfThought(
    rephraser,
    classification_name="SPEND_2026",
    api_key="...",
    model_name="openai/gpt-4o-mini",
    store=store,
)

prediction = classifier(query="2x A4 Papier Kopierer", k=3)

prediction.rephrased_query
# 'printing and copying paper, A4 sheets for office copier'
prediction.spend_code
# '101010'
```

→ [Rewrite-Retrieve-Read](../linear-rag/rrr.md)

## Agentic RAG

The retriever becomes a tool. The agent decides when to search, with what query, and
whether to search again.

```python
from rac4os.agents.RAG.BaseAgentRAG import BaseRAGAgent

class SpendRAGAgentSignature(dspy.Signature):
    """Classify an internal spend item into the SPEND taxonomy by searching it.

    Search with a focused query, read the `includes` of every candidate returned, and
    search again with different wording when the results do not fit. Never emit a code
    the search tool did not return.
    """
    input_item: str = dspy.InputField(desc="the item text to classify, as recorded")
    spend_code: str = dspy.OutputField(desc="the code of the chosen category")
    explanation: str = dspy.OutputField(desc="why this category and not the others")

agent = BaseRAGAgent(
    classification_name="SPEND_2026",
    api_key="...",
    model_name="openai/gpt-4o-mini",
    signature=SpendRAGAgentSignature,
    search_category_tool_doc=(
        "Search the SPEND taxonomy for candidate categories. Pass a short descriptive "
        "query and k, the number of candidates to return."
    ),
    store=store,
)

prediction = agent(input_item="docking station for the new laptops")

prediction.spend_code
# '201020'
prediction.trajectory["tool_args_0"]
# {'query': 'docking station', 'k': 5}
```

→ [Agentic RAG](../agentic-rag/usage.md)

## Hierarchy traversal agent

No retriever and no embeddings — the agent descends your taxonomy level by level through
four tools.

```python
from rac4os.agents.HierarchicalNavigation.BaseAgent import BaseHierarchicalNavigationAgent

agent = BaseHierarchicalNavigationAgent(
    classification_name="SPEND_2026",
    api_key="...",
    model_name="openai/gpt-4o-mini",
    signature=SpendRAGAgentSignature,          # same I/O contract, no candidate field
    get_root_categories_doc="Return the top-level categories of the taxonomy.",
    get_children_doc="Return the child categories of a code. Use it to descend one level.",
    get_code_doc="Return the full record of one code, including what it includes.",
    get_parent_doc="Return the parent of a code. Use it to back out of a wrong branch.",
)

prediction = agent(input_item="two boxes of A4 paper for the printer")

prediction.spend_code
# '101010'
```

→ [Hierarchy Traversal Agent](../traversal/usage.md)

## Next Steps

- Describe your own taxonomy properly: [Your Own Taxonomy > Overview](../taxonomy/overview.md)
- Decide what the model should see about each candidate:
  [Candidate Augmentation](../taxonomy/augmentation.md) — at non-leaf target levels this
  matters more than the choice of approach
- Read the prompts the bundled classifiers send: e.g.
  [Linear RAG > Prompt Overview](../linear-rag/prompts.md)
