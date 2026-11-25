# Datasets and the Registry

## The concept

A loader and a classification system describe a *kind* of thing: this format, these
nesting rules. What a classifier needs is a *particular* thing: this file, read that way,
nested those ways. The dataset entry is where those three facts are tied together and
given a name.

The name is the point. Once `"SPEND_2026"` means *spend.csv, read by SpendLoader, nested
by SpendTaxonomy*, every component in the package takes that one string and needs nothing
else. It also lets you keep variants of the same taxonomy side by side — a raw and a
[refined](../refinement/usage.md) edition, two vintages, a subset
restricted to the codes that occur in your data — and switch between them by changing a
string rather than rewiring a pipeline. Keeping both editions registered is what makes it
possible to tell whether a change to the taxonomy's descriptions helped.

## How it works

`mapping` is an ordinary dictionary, and `get_classification_system(name)` is the
function that consults it: it looks up the entry, asks the loader registry for the named
loader, hands it the path, and constructs the named classification system around the
result.

```python
from rac4os.classification_system.mapping import mapping

mapping["SPEND_2026"] = {
    "classification_system": "SPEND",      # the @register_system id
    "loader_name":           "SPEND_CSV",  # the @register_loader id
    "data_path":             "spend.csv",
    "metadata":              {"url": "internal", "version": "2026"},
}
```

Then:

```pycon
>>> from rac4os.classification_system import get_classification_system
>>> cs = get_classification_system("SPEND_2026")
>>> len(cs.codes)
12
```

`classification_system` and `loader_name` are separate ids on purpose: the same nesting
rules can be fed from a different file or a different format by pointing a new entry at a
different loader. That is how a refined edition of a taxonomy is registered — same
structure class, different loader and file.

`metadata` is not read by the package. It is for you: where the source came from, which
edition it is, and — for a generated dataset — which model produced it and when. A
refined dataset without that provenance becomes unreproducible within weeks.

## Example output

The full three-step setup, in one script, is all that stands between a CSV file and a
navigable taxonomy:

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
                details={
                    "includes": row["includes"].split(";") if row["includes"] else []
                },
            ).to_dict()
            for _, row in df.iterrows()
        ]


@register_system("SPEND")
class SpendTaxonomy(ClassificationSystem):
    def get_root_categories(self) -> list[dict]:
        return [
            {"code": c.code, "description": c.description}
            for c in self.codes
            if len(c.code) == 2
        ]

    def get_prefixes(self, code: str) -> list[str]:
        return [code[:i] for i in range(2, len(code) + 1, 2)]


mapping["SPEND_2026"] = {
    "classification_system": "SPEND",
    "loader_name": "SPEND_CSV",
    "data_path": "spend.csv",
    "metadata": {"url": "internal", "version": "2026"},
}

cs = get_classification_system("SPEND_2026")
print(len(cs.codes), cs.get_root_categories())
```

```text
12 [{'code': '10', 'description': 'Office supplies and consumables'}, {'code': '20', 'description': 'IT equipment and services'}]
```

From here your taxonomy is indistinguishable from a bundled one: the same navigation, the
same retriever, the same four classification approaches.

!!! warning "Registration happens on import"

    The `@register_*` decorators and the `mapping[...] =` assignment take effect only
    when that code actually runs. The module that defines your loader and your structure
    class, and adds the dataset entry, must therefore be imported *before* the first call
    to `get_classification_system`. In a single script, keep the three steps above the
    first use; in a package, import that module in your `__init__.py`.

    An unknown name raises `ValueError: Unknown format 'SPEND_2026'` and lists what is
    registered — which in practice means the registering module never ran.

## Without the registry

Nothing above is mandatory. A classification system can be constructed directly from a
loader instance:

```python
cs = SpendTaxonomy(SpendLoader("spend.csv"))
```

This is equivalent to the registered path and is the shorter route when you have exactly
one taxonomy and no need to name it. What you give up is the ability to refer to the
dataset by string, which the `Base…` classifiers and the refinement module use when they
construct a classification system on your behalf.

## The bundled entries

The package ships dataset entries for several official classifications, with their source
documentation included under `rac4os/_data/`:

| Name                  | Source format                           |
| --------------------- | --------------------------------------- |
| `COICOP_2018`         | Eurostat/UN structure spreadsheet       |
| `SEA_2021`            | Klassifikationsserver XML               |
| `SEA_2021_REPHRASED`  | JSON, refined descriptions              |
| `WZ_2008`             | Klassifikationsserver XML               |
| `KLDB_2010`           | Klassifikationsserver XML               |
| `ICATUS_2016`         | UN CSV table                            |
| `NAF_rev2`            | JSON                                    |

They are worked examples of the three steps on this page, nothing more. If your source is
a Klassifikationsserver XML export, the `"KLASS_SERVER"` loader already reads it and you
can skip [step 1](loaders.md) entirely — that is the one case where a bundled entry saves
you real work rather than merely illustrating it.
