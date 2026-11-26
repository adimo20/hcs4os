from ..BaseDataLoader import ClassificationLoader
from ..registry import register
import json

@register("JSON")
class PredefinedJsonLoader(ClassificationLoader):
    
  """
  Loader for a classification that is already in `Code` shape, registered under "JSON".

  This is the loader to use for your own taxonomy once you have converted it once, and
  the one that reads back what `ClassificationsSystemAugmenter` writes: both are a flat
  JSON list of `Code`-shaped objects, so there is no parsing left to do.
  """
  def to_records(
        self
    )->list[dict]:

    """
    Read the JSON file and return its contents unchanged.

    The file must hold a list of `Code`-shaped objects. Nothing is validated here; keys
    that `Code` does not define are dropped later, by `Code.from_dict`.

    Returns:
        list[dict]: the records as they stand in the file.
    """
    with open(f"{self.path}", "r", encoding = "utf-8") as f:
        all_labels = json.loads(f.read())
    
    return all_labels

if __name__ == "__main__":
    import sys

    codes = PredefinedJsonLoader(sys.argv[1]).load()
    print(codes[:10])

