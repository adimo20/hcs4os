from ..BaseDataLoader import ClassificationLoader
from ..registry import register
import json

@register("JSON")
class PredefinedJsonLoader(ClassificationLoader):
    
  def to_records(
        self
    )->list[dict]:

    with open(f"{self.path}", "r", encoding = "utf-8") as f:
        all_labels = json.loads(f.read())
    
    return all_labels

if __name__ == "__main__":
    import sys

    codes = PredefinedJsonLoader(sys.argv[1]).load()
    print(codes[:10])

