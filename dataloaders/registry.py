from .BaseDataLoader import ClassificationLoader

_REGISTRY_DL: dict[str, type[ClassificationLoader]] = {}

def register(name: str):
    
    def deco(cls):
        _REGISTRY_DL[name] = cls
        return cls
    return deco

def get_loader(name: str, path) -> ClassificationLoader:
    try:
        cls = _REGISTRY_DL[name]
    except KeyError:
        raise ValueError(f"Unknown format {name!r}; known: {sorted(_REGISTRY_DL)}")
    return cls(path)