# Candidate Augmentation

Augmentation is the step between retrieval and generation: the candidates are combined
with the item and an instruction, and the result is what the model actually reads. It
looks like plumbing, and when you classify to inner categories it is the most
consequential decision on your list.

## Why it matters

An inner category's label was never written to enumerate what belongs to it — the
enumeration lives in its children. Show a model only the label and it has less
information than your taxonomy actually holds, and precisely the part that decides the
hard cases is missing.

A category labelled *Bread and baked goods* gives no hint that a filled wafer bar belongs
to it, while a sibling labelled *Chocolate, cocoa and cocoa-based products* looks like the
obvious choice. Only the children of the first — which list filled bars containing biscuit
or wafer — settle it. The label does not merely underdetermine the answer; it points the
wrong way.

So append what a candidate covers, not just what it is called.

## Two setups

The bundled prompts are written for two augmentation setups, and they are worth naming
because the signature docstrings refer to them.

**Without hierarchical context.** Each candidate is presented as its code and its official
description. This is the conventional representation, and it is all that is available when
the target level *is* the leaf level.

```json
[
  {"code": "101010", "description": "Printing and copying paper"},
  {"code": "101020", "description": "Hygiene paper"}
]
```

**With hierarchical context.** The same representation plus an `includes` field listing
the descriptions of the child codes subsumed under each candidate.

```json
[
  {"code": "101010", "description": "Printing and copying paper",
   "includes": ["copier paper", "inkjet paper", "continuous forms"]},
  {"code": "101020", "description": "Hygiene paper",
   "includes": ["paper towels", "toilet paper", "facial tissues"]}
]
```

The second setup is what makes a query such as *two boxes of A4 paper for the printer*
decidable: *copier paper* is in the first candidate's `includes`, and nothing in the
second's comes near it. The generator is no longer guessing at the extent of a category;
it is reading it.

Two practical consequences:

- **The two setups need slightly different instructions.** A model given `includes`
  should be told to judge on it; a model given only descriptions should be told that the
  description is its only anchor. The bundled signatures for the linear approaches come
  in these variants, differing only in that passage.
- **Hierarchical context costs tokens.** The `includes` list of a broad inner code can be
  long, and it is multiplied by the number of candidates retrieved. This is the one place
  where retrieval depth and context richness trade against each other directly.

## How it works

Where augmentation happens differs by approach, and this is the main structural
difference between them:

| Approach                                                             | Who builds the context                                                |
| -------------------------------------------------------------------- | --------------------------------------------------------------------- |
| [Linear RAG](../linear-rag/usage.md) and [RRR](../linear-rag/rrr.md) | the pipeline: retrieved candidates are serialised to JSON and passed as one input field |
| [Agentic RAG](../agentic-rag/usage.md)                          | the agent: each tool result enters the trajectory as an observation    |
| [Hierarchy traversal](../traversal/usage.md)          | the agent: each tool result is one level of the tree                   |

In the linear approaches the candidate list is serialised with
`json.dumps(..., indent=4, ensure_ascii=False)` and placed in the signature's context
field, so what the model reads is the JSON above. In the agentic approaches the shape is
whatever `search_category` or the navigation tools return, which is the `Code` record as a
dict.

That means **the representation is determined by what you put in the vector store and in
the `Code` record**, not by the classifier. Which brings up the one thing to get right:

```python
Code(
    code="101010",
    description="Printing and copying paper",     # embedded, and shown as the label
    details={"includes": ["copier paper", ...]},  # shown as hierarchical context
)
```

If your source documentation has inclusions, exclusions or introductory notes, putting
them in `details` is what makes the second setup available to you. If it does not have
them — and many internal taxonomies and some official ones do not, or have them only
unsystematically at the target level — the children of each code can serve the same
purpose, since `get_children` is available for every registered taxonomy:

```python
candidate = cs.get_code("101010").to_dict()
candidate["includes"] = [c.description for c in cs.get_children("101010")]
```

Where the inner codes themselves have thin descriptions, the
[taxonomy refinement](../refinement/usage.md) module generates prose for
them from their children and writes the result back as a new dataset — which is the same
idea applied one step earlier, to what gets *embedded* rather than to what gets shown.

## Example output

The effect on the input the model receives, for the same query and the same retrieved
candidates. Without hierarchical context:

```json
{
  "input_item": "two boxes of A4 paper for the printer",
  "retrieved_candidates": [
    {"code": "101010", "description": "Printing and copying paper"},
    {"code": "101020", "description": "Hygiene paper"},
    {"code": "102010", "description": "Pens and pencils"}
  ]
}
```

With it:

```json
{
  "input_item": "two boxes of A4 paper for the printer",
  "retrieved_candidates": [
    {"code": "101010", "description": "Printing and copying paper",
     "includes": ["copier paper", "inkjet paper", "continuous forms"]},
    {"code": "101020", "description": "Hygiene paper",
     "includes": ["paper towels", "toilet paper", "facial tissues"]},
    {"code": "102010", "description": "Pens and pencils",
     "includes": ["ballpoint pens", "pencils", "markers"]}
  ]
}
```

Same retrieval, same model, same instruction. The second input contains the word that
decides the case.
