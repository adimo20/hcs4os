# dataloaders/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from .._shared.Code import Code

class ClassificationLoader(ABC):
    """Template: subclasses provide read_raw() + to_records(); base handles the rest."""
    def __init__(self, path: str | Path) -> None:
        """
        Stores the source path and checks that it exists.

        Failing here rather than at `to_records` means a wrong path is reported before any
        parsing work or model call happens.

        Raises:
            FileNotFoundError: if the path does not exist.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Classification source not found: {self.path}")

    @abstractmethod
    def to_records(self) -> list[dict]:
        """Return a list of Code-shaped dicts. This is the ONLY thing subclasses implement."""
        ...

    def load(self) -> list[Code]:
        """
        Parse the source and return it as `Code` records.

        Calls the subclass's `to_records` and converts each dict with `Code.from_dict`, which
        drops keys `Code` does not define and leaves missing ones at their defaults.

        Returns:
            list[Code]: one record per category in the source.
        """
        return [Code.from_dict(r) for r in self.to_records()]