"""Packaging promises: slim core install, bundled data, Python version."""
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from hcs4os.classification_system.mapping import mapping

ROOT = Path(__file__).parent.parent


OPTIONAL = ("chromadb", "sentence_transformers", "ollama", "openai", "torch")


def _loaded_optional_modules(statement: str) -> set[str]:
    """Run `statement` in a fresh interpreter and report which optional modules it loaded."""
    code = (f"{statement}\nimport sys\n"
            f"print(','.join(m for m in {OPTIONAL!r} if m in sys.modules))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, f"`{statement}` failed:\n{result.stderr}"
    out = result.stdout.strip().splitlines()
    return set(out[-1].split(",")) - {""} if out else set()


@pytest.mark.parametrize("statement", [
    "import hcs4os",
    "from hcs4os.classification_system import get_classification_system",
    "from hcs4os.agents.HierarchicalNavigation import HierarchicalNavigationAgent",
])
def test_core_imports_do_not_need_optional_extras(statement):
    """The core install has no chromadb/torch, so hcs4os itself must not import them.

    dspy (via litellm) may import some of these on its own when they happen to be
    installed; that is not hcs4os's doing, so whatever `import dspy` loads is subtracted.
    """
    from_dspy = _loaded_optional_modules("import dspy")
    added_by_hcs4os = _loaded_optional_modules(statement) - from_dspy
    assert added_by_hcs4os == set()


def test_optional_extras_are_declared():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    assert {"local", "ollama", "openai"} <= set(extras)
    for name in ("local", "ollama", "openai"):
        assert "chromadb" in extras[name], f"extra {name!r} must work on its own"
    core = pyproject["project"]["dependencies"]
    assert not {"chromadb", "sentence-transformers", "torch"} & set(core)


def test_requires_python_matches_classifiers():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    minimum = project["requires-python"].removeprefix(">=")
    versions = [c.rsplit(" ", 1)[-1] for c in project["classifiers"]
                if c.startswith("Programming Language :: Python :: 3.")]
    assert min(versions, key=lambda v: tuple(map(int, v.split(".")))) == minimum


@pytest.mark.parametrize("name", list(mapping))
def test_every_mapped_data_file_is_bundled(name):
    path = Path(mapping[name]["data_path"])
    assert path.exists(), f"{name}: {path} missing"
    assert path.suffix.lstrip(".") in {"txt", "json", "csv", "xml", "xlsx"}, \
        "file type must be listed in [tool.setuptools.package-data]"


@pytest.mark.parametrize("name", list(mapping))
def test_mapping_entries_reference_registered_components(name):
    from hcs4os.classification_system.registry import _REGISTRY_CS
    from hcs4os.dataloaders.registry import _REGISTRY_DL
    entry = mapping[name]
    assert entry["classification_system"] in _REGISTRY_CS
    assert entry["loader_name"] in _REGISTRY_DL
    assert isinstance(entry["metadata"], dict)