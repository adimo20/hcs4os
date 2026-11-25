# Classification Systems

## The concept

In a statistical taxonomy the hierarchy is reflected in the codes themselves — that is
what distinguishes it from an arbitrary set of labels. `101010` is a child of `1010`
because of how the two strings relate, not because an edge was stored somewhere. The
structure is latent in the notation.

This has a convenient consequence. The entire hierarchy can be reconstructed from a
single function: given a code, return the chain of its ancestors. Children, parents,
sibling sets and the path from a root to a leaf all follow from it, and they follow in
one pass over the codes, so building the structure is linear in the number of categories
rather than quadratic.

What that function looks like is the one thing only you know about your taxonomy. Codes
that grow a fixed number of digits per level, dot-separated codes, codes that mix letters
and digits, codes whose levels are irregular — each is simply a different implementation
of the same contract.

## How it works

A classification system subclasses `ClassificationSystem` and implements two methods:

```python
from rac4os.classification_system.BaseClassificationSystem import ClassificationSystem
from rac4os.classification_system.registry import register as register_system


@register_system("SPEND")
class SpendTaxonomy(ClassificationSystem):
    def get_root_categories(self) -> list[dict]:
        # the entry points of a top-down search: the two-digit codes
        return [
            {"code": c.code, "description": c.description}
            for c in self.codes
            if len(c.code) == 2
        ]

    def get_prefixes(self, code: str) -> list[str]:
        # two digits per level: "101010" -> ["10", "1010", "101010"]
        return [code[:i] for i in range(2, len(code) + 1, 2)]
```

**The rule `get_prefixes` must satisfy: return the ancestors of the code in order, ending
with the code itself.** The parent of a code is then its second-to-last prefix, and that
single convention is what the base class uses to build its children register and to
answer every navigation call. Get it right and `get_children`, `get_parent`, `get_code`
and `get_code_trace` all work; get it wrong and they fail in the two recognisable ways
described below.

`get_root_categories` returns the top-level entries as `{code, description}` dicts. It is
the only place where "where does a search start?" is answered, and it is the first thing
a hierarchy-traversing agent calls.

Dot-separated codes, for comparison, need no slicing at all:

```python
def get_prefixes(self, code: str) -> list[str]:
    parts = code.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts) + 1)]
```

## Example output

With the loader from the [previous step](loaders.md) and the dataset entry from the
[next one](datasets.md) in place:

```pycon
>>> from rac4os.classification_system import get_classification_system
>>> cs = get_classification_system("SPEND_2026")
>>> len(cs.codes)
12

>>> cs.get_root_categories()
[{'code': '10', 'description': 'Office supplies and consumables'},
 {'code': '20', 'description': 'IT equipment and services'}]

>>> for c in cs.get_children("1010"):
...     print(c)
Code(code='101010', description='Printing and copying paper', level=3, detailled_description='', details={'includes': ['copier paper', 'inkjet paper', 'continuous forms']})
Code(code='101020', description='Hygiene paper', level=3, detailled_description='', details={'includes': ['paper towels', 'toilet paper', 'facial tissues']})

>>> cs.get_parent("101010")
Code(code='1010', description='Paper and paper products', level=2, detailled_description='', details={'includes': []})

>>> cs.get_code_trace("101010")
[('10', 'Office supplies and consumables'),
 ('1010', 'Paper and paper products'),
 ('101010', 'Printing and copying paper')]
```

`get_code` returns the record itself, which is what a classifier shows a model when it
asks about one specific category:

```pycon
>>> print(cs.get_code("101020").to_json())
{
    "code": "101020",
    "description": "Hygiene paper",
    "level": 3,
    "detailled_description": "",
    "details": {
        "includes": [
            "paper towels",
            "toilet paper",
            "facial tissues"
        ]
    }
}
```

## Checking it before you trust it

`get_code_trace` is the diagnostic worth running on a deep code the moment you have
written `get_prefixes`. It walks the whole ancestor chain and silently skips the rungs
that do not exist as codes, which makes a wrong prefix rule immediately visible:

```python
print(len(cs.codes), len(cs.get_root_categories()))
print(cs.get_code_trace("<a known deep code>"))   # every level, no gaps
print(cs.get_children("<a known mid-level code>"))
```

Two failure modes account for nearly everything that goes wrong here:

- **A trace with missing rungs** means `get_prefixes` produces strings that do not exist
  as codes in your data — usually a separator that the file writes but the code strings
  do not, or the reverse.
- **An empty `get_children` on a code that certainly has children** is the same problem
  one level up: the parent string your prefix logic derives is not the parent string your
  file contains. Compare `cs.get_prefixes("<child code>")[-2]` against the parent code as
  it appears in the file.

Both are notation problems rather than data problems, and both are silent — nothing
raises, you simply get an empty list. Hence the check.

!!! warning "Codes are strings"

    Leading zeros are load-bearing in most taxonomies, and `01` is not `1`. Read your
    source with explicit string types (`dtype=str` in pandas) rather than letting the
    parser infer them. A code column silently converted to integers is the most common
    cause of a hierarchy that builds without error and then answers nothing.

## Adding categories at runtime

`add_code(code: dict, parent: str | None = None)` inserts a single category into an
already-built system, registering it under `parent` when one is given. It exists for
patching — a category your source file is missing, a provisional code — and updates the
lookup and the children register without re-reading the file. For anything systematic,
change the loader instead, so that the next run reproduces it.
