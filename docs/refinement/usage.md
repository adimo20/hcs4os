# Usage

`ClassificationsSystemAugmenter`

Taxonomy refinement does not classify items. It works on the taxonomy itself: it writes a
detailed description for each inner category from the labels of its children, and saves
the result as a new dataset you can register alongside the original.

Use it when the categories you classify into have labels too thin to retrieve against.

!!! tip "See also"

    [How It Works](how-it-works.md) for what it generates and how it walks the hierarchy.

## Running it

```python
from rac4os.taxonomy_refinement.TaxonomyRefinement import ClassificationsSystemAugmenter

augmenter = ClassificationsSystemAugmenter(
    classification_name="SPEND_2026",     # a registered dataset name
    api_key="...",
    model="openai/gpt-4o-mini",
)

augmenter.enrich_classification_system(start_level=3)   # walk upward from level 3
augmenter.get_updated_classification_system("spend_2026_refined.json")
```

`enrich_classification_system(start_level)` walks the hierarchy upward from
`start_level`, and for each parent it reaches, generates a `detailled_description` from
the parent's own label plus the labels of its direct children. `start_level` should be
the deepest level you want to use as evidence — usually the leaf level.

`get_updated_classification_system(path)` writes every code, enriched and not, as a JSON
list of `Code`-shaped records.

## Example output

The console trace shows which parent is being processed, one per group of siblings:

```text
['101010', '101020', '102010', '201010', '201020', '202010']
Processing: Code(code='1010', description='Paper and paper products', level=2, ...)
Processing: Code(code='1020', description='Writing and drawing instruments', level=2, ...)
Processing: Code(code='2010', description='Computing hardware', level=2, ...)
Processing: Code(code='2020', description='Software and licences', level=2, ...)
```

A refined record gains prose in `detailled_description` while everything else stays as it
was:

```json
{
    "code": "1010",
    "description": "Paper and paper products",
    "level": 2,
    "detailled_description": "Paper-based consumables purchased for office use, covering both paper intended for printing and copying — such as copier and inkjet paper and continuous forms — and hygiene paper products such as paper towels, toilet paper and facial tissues.",
    "details": {"includes": []}
}
```

!!! note

    The generated prose above is illustrative; the record structure and the console trace
    are real output from the example taxonomy.

## Registering the result

Refinement produces a *second* dataset, not a replacement. Write it next to your source
file, register it with the `"JSON"` loader and the same structure class, and record what
produced it:

```python
from rac4os.classification_system.mapping import mapping

mapping["SPEND_2026_REFINED"] = {
    "classification_system": "SPEND",          # same nesting rules
    "loader_name": "JSON",                      # reads Code-shaped records
    "data_path": "spend_2026_refined.json",
    "metadata": {
        "url": "internal",
        "model": "openai/gpt-4o-mini",
        "date": "2026-11-25",
    },
}
```

Keeping both entries registered is the point: it lets you run the same approach against
the raw and the refined taxonomy and see whether the refinement helped. The `metadata` is
what keeps the refined dataset reproducible — without the model and the date it becomes
unexplainable within weeks.

The bundled `SEA_2021` / `SEA_2021_REPHRASED` pair is exactly this arrangement.

## Using the refined descriptions

Nothing downstream changes except the dataset name — and what you choose to embed:

```python
cs = get_classification_system("SPEND_2026_REFINED")

store = VectorStore(
    collection_name="spend_2026_refined_level2",   # a NEW collection
    chromadb_path="./chroma",
    embedding_function_type="sentence_transformers",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
store.index(classification_system=cs,
            codes_to_include=[c.code for c in cs.codes if c.level == 2])
```

!!! warning

    `index()` embeds `description`, not `detailled_description`. Refinement therefore
    improves what the *generator* reads about a candidate out of the box; to make the
    richer text drive *retrieval*, move it into `description` when you write the JSON, or
    in the loader that reads it back.

## Caveats

- **Generated descriptions are generated.** They are derived from the labels you already
  have, and the signature instructs the model not to invent subcategories, but this is a
  model writing prose about your taxonomy. Review a sample before relying on it, and keep
  the provenance in `metadata`.
- **Errors are swallowed per category.** A failed generation is printed and skipped, and
  the walk continues — so a run can complete with some categories unenriched. Check for
  empty `detailled_description` values in the output before registering it.
- **Only `_lookup` is updated in memory.** The in-memory `codes` list and the children
  register are not refreshed, so use the written JSON rather than the live object after a
  run.
