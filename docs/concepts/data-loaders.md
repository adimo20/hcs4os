# Data loaders — how to read a file

`ClassificationLoader` (in [BaseDataLoader.py](https://github.com/adimo20/hcs4os/blob/master/hcs4os/dataloaders/BaseDataLoader.py))
is a template: it checks the path exists and converts records to `Code` objects.
Subclasses implement exactly one method, `to_records() -> list[dict]`.

Each loader registers itself under a string ID with the `@register` decorator, and
`get_loader(name, path)` resolves it. Currently registered:

| ID             | Source format                         | Notes                                                                            |
| -------------- | ------------------------------------- | -------------------------------------------------------------------------------- |
| `KLASS_SERVER` | Claset XML (klassifikationsserver.de) | The workhorse — parses labels, keywords, inclusions/exclusions; drops empty keys |
| `COICOP`       | UN COICOP 2018 `.xlsx`                | Expects columns `code, title, intro, includes, alsoIncludes, excludes`           |
| `ICATUS`       | UNSD Time-Use `.csv`                  | Flattens the three code columns `PID/ID/CID` into one code                       |
| `NACE`         | Eurostat NACE Rev. 2.1 `.xlsx`        | Registered but not yet referenced from `mapping.py`                              |
| `JSON`         | pre-built list of `Code`-shaped dicts | Used for the LLM-rephrased variants; the escape hatch for any custom pipeline    |

The `KLASS_SERVER` loader is the one to reuse: every German classification
downloadable from the Klassifikationsserver in "Gliederung mit Erläuterung" format
parses without a single new line of code.
