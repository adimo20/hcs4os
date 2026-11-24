# `Code` — the common currency

A dataclass every source is normalised into:

| field                   | meaning                                                          |
| ----------------------- | ---------------------------------------------------------------- |
| `code`                  | the code string, e.g. `"01.1.1.1"`, `"0111 1"`                   |
| `description`           | short label, e.g. `"Cereals (ND)"`                               |
| `level`                 | depth in the hierarchy                                           |
| `detailled_description` | long prose definition (COICOP intro notes, ICATUS definition, …) |
| `details`               | free-form dict: `includes` / `excludes` / `keywords` / `context` |

`details` is deliberately unconstrained — each classification publishes different
kinds of notes, and the agents just read whatever is there. `Code.from_dict()`
silently drops unknown keys, so a loader may emit extra columns without breaking.
