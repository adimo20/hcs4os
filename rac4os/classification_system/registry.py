from collections.abc import Callable
from .BaseClassificationSystem import ClassificationSystem
from .mapping import mapping
from ..dataloaders.registry import get_loader

_REGISTRY_CS: dict[str, type[ClassificationSystem]] = {}



def register(name: str) -> Callable:
    """
    Class decorator that registers a `ClassificationSystem` implementation under a name.

    The name is the one a dataset refers to in `mapping` through its
    `classification_system` key, which is how `get_classification_system` finds the class
    to build.

    Parameters:
        name (str): key to register the decorated class under.

    Returns:
        Callable: the decorator, which registers the class and returns it unchanged.
    """
    def deco(cls):
        """
        Registers the decorated class under the captured name.

        Parameters:
            cls (type[ClassificationSystem]): the class being decorated.

        Returns:
            type[ClassificationSystem]: the same class, unmodified.
        """
        _REGISTRY_CS[name] = cls
        return cls
    return deco

def get_classification_system(name: str) -> ClassificationSystem:
    """
    Build the classification system for a registered dataset name.

    Looks the name up in `mapping`, builds the loader named there for the data path given
    there, and hands that loader to the registered `ClassificationSystem` subclass. This
    is purely a convenience: it exists so a dataset can be referred to by one name instead
    of wiring a loader and a system class together at every call site. Constructing the
    two by hand stays equally valid and is what you do for a taxonomy you have not
    registered.

    Parameters:
        name (str): a key of `mapping`, such as "COICOP_2018", or one you registered yourself.

    Returns:
        ClassificationSystem: a system built from that dataset, ready to query.

    Raises:
        ValueError: if the name is not in `mapping`, or if the class it names was never registered.
    """
    try:
        cls = _REGISTRY_CS[mapping[name]["classification_system"]]
        loader = get_loader(
            mapping[name]["loader_name"],
            mapping[name]["data_path"]
        )
        return cls(loader)
    except KeyError:
        raise ValueError(f"Unknown format {name!r}; known: {sorted(_REGISTRY_CS)}")
        return
    