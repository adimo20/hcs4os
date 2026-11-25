# Data Loaders

## The concept

Published classification documentation arrives in whatever format the publishing
institution chose: an XML export from a classification server, a Eurostat spreadsheet, a
UN CSV table, a PDF appendix someone has already transcribed. Internal taxonomies
usually arrive as a spreadsheet or a database export. None of these formats agree on
anything, and none of them is the format a classifier wants to work with.

The loader layer exists to absorb that variation in one place. Its job is to turn a
source file into a list of uniform records, and its contract is deliberately narrow: one
method, no state, no knowledge of the hierarchy. Whether your codes nest two digits at a
time or are separated by dots is not a question about the file format, so it is not the
loader's concern — that belongs to the [next step](systems.md).

## The record: `Code`

Every category in every taxonomy becomes a `Code`, a dataclass with five fields:

| Field                   | Holds                                                                 |
| ----------------------- | --------------------------------------------------------------------- |
| `code`                  | the code string itself, e.g. `"101010"`                               |
| `description`           | the short official label — **this is what gets embedded**             |
| `level`                 | the depth of the code in the hierarchy, as an int                     |
| `detailled_description` | longer prose about what belongs in the category, where it exists      |
| `details`               | a free-form dict for everything else: includes, excludes, keywords, … |

Fill what you have. Anything you omit defaults to empty, and extra keys you emit are
dropped silently rather than raising. The split between the last two fields is worth
getting right: `description` is the text the retriever indexes, so it should be the label
a human would recognise, while `details` is where the material that resolves ambiguity
belongs — the inclusions and exclusions that say what the label alone does not.

## How it works

A loader subclasses `ClassificationLoader` and implements `to_records()`, returning a
list of `Code`-shaped dicts. The base class has already resolved the source path into
`self.path` (a `pathlib.Path`) and verified that it exists; `load()`, which the rest of
the package calls, maps your records through `Code.from_dict`.

```python
import pandas as pd
from rac4os.dataloaders.BaseDataLoader import ClassificationLoader
from rac4os.dataloaders.registry import register as register_loader
from rac4os._shared.Code import Code


@register_loader("SPEND_CSV")
class SpendLoader(ClassificationLoader):
    def to_records(self) -> list[dict]:
        df = pd.read_csv(self.path, dtype=str).fillna("")
        return [
            Code(
                code=row["code"],
                description=row["label"],
                level=int(row["level"]),
                details={
                    "includes": row["includes"].split(";") if row["includes"] else []
                },
            ).to_dict()
            for _, row in df.iterrows()
        ]
```

Building each record with `Code(...).to_dict()` is the safe way to get a correctly shaped
dict — a hand-written dict with the same keys works too, but you lose the type checking.

The `@register_loader("SPEND_CSV")` decorator is the convenience discussed in
[Naming the dataset](datasets.md): it lets a dataset entry refer to this loader by
string. You can skip it and instantiate `SpendLoader("spend.csv")` yourself.

## Example output

```pycon
>>> loader = SpendLoader("spend.csv")
>>> codes = loader.load()
>>> len(codes)
12
>>> codes[2]
Code(code='101010', description='Printing and copying paper', level=3,
     detailled_description='',
     details={'includes': ['copier paper', 'inkjet paper', 'continuous forms']})
```

That is the whole output of the format layer: a flat list of `Code` objects, in file
order, with no hierarchy yet. Two checks are worth running before moving on — that
`len(codes)` matches the number of categories you expect, and that `level` arrives as an
`int` rather than the string the file gave you, because the level is what you will later
filter on when restricting a retriever to a single target level.

## Loading something that is not a file format

Not every dataset deserves a parser. For generated or one-off content — model-refined
descriptions, a merged taxonomy, a subset restricted to the codes that actually occur in
your data — the pragmatic path is to write a JSON list of `Code`-shaped dicts and load it
with the bundled `"JSON"` loader, which does nothing but read and forward it. The
[taxonomy refinement](../refinement/usage.md) module produces exactly such
a file.
