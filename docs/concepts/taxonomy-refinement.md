# Taxonomy refinement

Published classifications document their leaves well and their inner nodes barely at
all — a division often has nothing but a two-word label. That hurts retrieval most,
because a sparse label is a weak embedding.

`ClassificationsSystemAugmenter` fills those gaps. `enrich_classification_system(start_level)`
walks the hierarchy upward from `start_level`, and for each parent asks the model for
a prose description given its own label plus the labels of its direct children — on
the premise that a parent's meaning is the union of what its children cover. The
`GenerateParentDescription` signature pins that down: characterise the category as a
whole rather than concatenating the children, stay within what the labels support,
and answer in the language of the input.

Two things to know before using it:

- It writes into `classification_system._lookup` only. `codes` and
  `_children_register` still hold the old objects, so re-read results through
  `get_code`, not by iterating `codes`.
- The result is meant to be dumped to JSON in `_data/` and registered as its own
  `mapping.py` entry with a `JSON` loader — that is how `SEA_2021_REPHRASED` and
  `sea_2021_augmented.json` came to exist. Keep the generated dataset beside the
  original rather than replacing it, so the two remain comparable.
