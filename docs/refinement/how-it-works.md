# How It Works

## Why refine a taxonomy

Retrieval compares an item against what a category *says about itself*. When a category's
published label is a few words long — and inner categories often are — there is very
little for an embedding model to match on, and very little for a generator to judge by.

The information is not missing from the taxonomy, though. It is in the children: a
parent's meaning is the union of the concepts its children cover, and together they
delimit its scope. Refinement makes that implicit information explicit by writing it out
as prose, so that the category becomes describable on its own.

This is the same idea as [candidate augmentation](../taxonomy/augmentation.md), applied
one step earlier. Augmentation shows the children to the model at classification time;
refinement turns them into a description once, up front, and stores it.

## The walk

`enrich_classification_system(start_level)` works upward, one level at a time:

```
for lvl in start_level … 1:
    collect the codes at that level
    for each of them:
        find its parent
        collect ALL children of that parent
        remove those siblings from the remaining work for this level
        generate a description for the parent from (parent label, children labels)
```

Two details follow from that loop:

- **Each parent is processed once, not once per child.** As soon as a parent is handled,
  all of its children are removed from the level's worklist.
- **It walks upward, so refinement compounds.** Levels closer to the leaves are described
  first, which means that by the time a higher level is reached, the evidence used for it
  can already include descriptions written in the previous round.

## What generates the text

A `dspy.ChainOfThought` over `GenerateParentDescription`, which receives one
`parent_child_pairs` dict — the parent as `code`, `description`, `level`, and its direct
children in the same shape — and returns `detailed_description`.

The signature's instructions are what keep the output usable rather than a concatenation
of labels. It asks for a description that

- characterises the parent as a whole, as the general concept it captures;
- reflects the full scope implied by the children, so it is clear what belongs;
- stays faithful to the given labels, inventing no subcategories, products or
  distinctions;
- is written in the language of the input labels;
- stands on its own, without the reader needing to see the child list.

The faithfulness constraint is the important one: the module's purpose is to surface
information the taxonomy already contains, not to extend the taxonomy.

::: rac4os.taxonomy_refinement.TaxonomyRefinement.GenerateParentDescription

## Where the result goes

Generated text is written to the in-memory lookup as `detailled_description` on the parent
code, and `get_updated_classification_system(path)` dumps every record in the lookup to
JSON. That file is a complete dataset — a `Code`-shaped list — which is why the
[`"JSON"` loader](../taxonomy/loaders.md#loading-something-that-is-not-a-file-format)
reads it back with no parsing work.

Only the lookup is updated during a run; the `codes` list and the children register are
not rebuilt. Treat the written file as the output, and load it as a fresh dataset rather
than continuing with the live object.

## Components

| Component                      | Role                                                 |
| ------------------------------ | ---------------------------------------------------- |
| classification system          | provides parents, children and their labels          |
| `GenerateParentDescription`    | the prompt and the I/O contract                      |
| `dspy.ChainOfThought`          | the generation                                       |
| LM                             | `model`, `api_key`                                   |
| `get_updated_classification_system` | writes the new dataset                          |

!!! note

    This module calls `dspy.configure(lm=...)` globally in its constructor and does not
    pin its LM per call. Construct it in a script of its own, or construct it immediately
    before use, rather than alongside classifiers that expect a different model.
