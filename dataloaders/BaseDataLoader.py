# dataloaders/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from Code import Code

class ClassificationLoader(ABC):
    """Template: subclasses provide read_raw() + to_records(); base handles the rest."""
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Classification source not found: {self.path}")

    @abstractmethod
    def to_records(self) -> list[dict]:
        """Return a list of Code-shaped dicts. This is the ONLY thing subclasses implement."""
        ...

    def load(self) -> list[Code]:
        return [Code.from_dict(r) for r in self.to_records()]