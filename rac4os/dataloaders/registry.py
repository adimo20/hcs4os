from collections.abc import Callable
from .BaseDataLoader import ClassificationLoader

_REGISTRY_DL: dict[str, type[ClassificationLoader]] = {}

def register(name: str) -> Callable:
    
    """
    Class decorator that registers a `ClassificationLoader` implementation under a name.

    The name is what a dataset refers to in `mapping` through its `loader_name` key, and
    what `get_loader` resolves.

    Parameters:
        name (str): key to register the decorated class under.

    Returns:
        Callable: the decorator, which registers the class and returns it unchanged.
    """
    def deco(cls):
        """
        Registers the decorated class under the captured name.

        Parameters:
            cls (type[ClassificationLoader]): the class being decorated.

        Returns:
            type[ClassificationLoader]: the same class, unmodified.
        """
        _REGISTRY_DL[name] = cls
        return cls
    return deco

def get_loader(name: str, path) -> ClassificationLoader:
    """
    Build a registered loader for a source path.

    Parameters:
        name (str): a registered loader name, e.g. "JSON", "COICOP" or one of your own.
        path (str|Path): the file the loader should read.

    Returns:
        ClassificationLoader: the loader, constructed and ready for `load`.

    Raises:
        ValueError: if no loader is registered under that name.
    """
    try:
        cls = _REGISTRY_DL[name]
    except KeyError:
        raise ValueError(f"Unknown format {name!r}; known: {sorted(_REGISTRY_DL)}")
    return cls(path)