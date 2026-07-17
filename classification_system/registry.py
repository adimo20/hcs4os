from .BaseClassificationSystem import ClassificationSystem
from .mapping import mapping
from dataloaders.registry import get_loader

_REGISTRY_CS: dict[str, type[ClassificationSystem]] = {}



def register(name: str):
    def deco(cls):
        _REGISTRY_CS[name] = cls
        return cls
    return deco

def get_classification_system(name: str) -> ClassificationSystem:
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
    