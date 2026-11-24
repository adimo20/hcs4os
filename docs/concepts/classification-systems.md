# Classification systems — how codes nest

`ClassificationSystem` (in
[BaseClassificationSystem.py](https://github.com/adimo20/hcs4os/blob/master/hcs4os/classification_system/BaseClassificationSystem.py))
provides all navigation for free. On construction it loads the codes and builds two
indices:

- `_lookup` — `code → Code`
- `_children_register` — `code → list[Code]`

Both are built in a single pass over the codes: each code's own prefix chain names its
parent, so it appends itself to that parent's list. That makes construction linear in
the number of codes — loading the largest shipped classification (SEA 2021, 2,654
codes) takes well under a tenth of a second, and what time loading does cost is
spent in the loader parsing the source file, not in building the hierarchy.

The whole hierarchy is derived from **one abstract method**:

```python
def get_prefixes(self, code: str) -> list[str]: ...
```

It returns the chain of ancestors of a code, ending with the code itself. The
second-to-last prefix is the parent. Get this right and `get_children`, `get_parent`,
and `get_code_trace` all follow.

Lookups are forgiving in one direction only: `get_children` on an unknown code returns
`[]`, the same as for a leaf, while `get_code` raises `ValueError`. So an empty child
list never tells you whether you reached a leaf or mistyped the code — check with
`get_code` if that distinction matters.

Subclasses also implement `get_root_categories()` — the entry point the agents start
from — which is usually just "all codes whose string has length 1 or 2".

Why this indirection: code strings look regular but are not. Compare the
implementations in
[ClassificationSystems.py](https://github.com/adimo20/hcs4os/blob/master/hcs4os/classification_system/systems/ClassificationSystems.py):

- **COICOP** is dot-separated → split on `"."` (`01` → `01.1` → `01.1.1`).
- **ICATUS / KldB** are pure prefix codes → every string prefix is an ancestor.
- **SEA 2021** is a prefix code _with a formatting quirk_: from digit 5 on, a space
  is inserted (`0111` → `0111 1`), so `get_prefixes` re-inserts it.
- **NAF rév. 2** (French) inserts a dot after the 2nd digit instead (`43` → `43.3` →
  `43.34`), so `get_prefixes` strips the dot and re-inserts it at a fixed position.
- **WZ 2008** carries trailing dots that must be stripped, and de-duplicated.

Six classes are registered — `COICOP`, `ICATUS`, `KLDB`, `SEA`, `WZ`, `NAF_rev2` —
under a string ID the same way loaders are.
