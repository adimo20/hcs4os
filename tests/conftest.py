"""
Shared fixtures for the hcs4os test suite.

Design rules for the suite:
- No network and no real LLM in the default run. LLM calls are replaced by
  fakes, embeddings by a deterministic bag-of-words embedding.
- Every registration (loader, classification system, mapping entry) is undone
  after the test via `monkeypatch`, so tests never leak global state.
- Slow tests (GP takes over a minute to load) only run with --runslow.
- Tests that need a live Ollama / LLM are marked `integration` and only run
  when the matching environment variables are set (see test_integration.py).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

RESOURCES = Path(__file__).parent / "resources"
FICTION_JSON = RESOURCES / "fiction.json"


# ---------------------------------------------------------------------------
# command-line options and markers
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run tests marked slow (e.g. loading GP)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes more than a few seconds; run with --runslow")
    config.addinivalue_line("markers", "integration: needs a live embedding server or LLM")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow; use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# ---------------------------------------------------------------------------
# a small custom classification, registered without touching package files
# ---------------------------------------------------------------------------

@pytest.fixture
def fiction_records() -> list[dict]:
    return json.loads(FICTION_JSON.read_text(encoding="utf-8"))


@pytest.fixture
def register_fiction(monkeypatch):
    """Registers loader 'FICTION', system 'FICTION' and mapping key 'FICTION_TEST'.

    Everything is undone after the test. Returns the mapping key.
    """
    from hcs4os.dataloaders import registry as dl_registry
    from hcs4os.dataloaders.BaseDataLoader import ClassificationLoader
    from hcs4os.classification_system import registry as cs_registry
    from hcs4os.classification_system import mapping as mapping_module
    from hcs4os.classification_system.BaseClassificationSystem import ClassificationSystem

    class FictionLoader(ClassificationLoader):
        def to_records(self):
            return json.loads(Path(self.path).read_text(encoding="utf-8"))

    class FictionSystem(ClassificationSystem):
        def get_root_categories(self):
            return [{"code": c.code, "description": c.description}
                    for c in self.codes if len(c.code) == 1]

        def get_prefixes(self, code: str):
            return [code[:i] for i in range(1, len(code) + 1)]

    monkeypatch.setitem(dl_registry._REGISTRY_DL, "FICTION", FictionLoader)
    monkeypatch.setitem(cs_registry._REGISTRY_CS, "FICTION", FictionSystem)
    monkeypatch.setitem(mapping_module.mapping, "FICTION_TEST", {
        "classification_system": "FICTION",
        "loader_name": "FICTION",
        "data_path": FICTION_JSON,
        "metadata": {},
    })
    return "FICTION_TEST"


@pytest.fixture
def fiction(register_fiction):
    from hcs4os.classification_system import get_classification_system
    return get_classification_system(register_fiction)


# ---------------------------------------------------------------------------
# vector store with a deterministic offline embedding
# ---------------------------------------------------------------------------

def _make_fake_ef_class():
    import numpy as np
    from chromadb.api.types import EmbeddingFunction

    class FakeEmbeddingFunction(EmbeddingFunction):
        """Hashes words into 64 buckets: same words -> similar vectors. Offline and fast."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, input):
            out = []
            for text in input:
                v = np.zeros(64)
                for word in text.lower().replace(",", " ").split():
                    v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
                out.append((v + 1e-6).tolist())
            return out

        @staticmethod
        def name():
            return "hcs4os-test-fake"

        def get_config(self):
            return {}

        @staticmethod
        def build_from_config(config):
            return FakeEmbeddingFunction()

    return FakeEmbeddingFunction


@pytest.fixture
def fake_ef(monkeypatch):
    """Makes embedding_function_type='fake' available in VectorStore."""
    pytest.importorskip("chromadb", reason="vector store needs an extra: pip install 'hcs4os[ollama]'")
    import hcs4os._vector_database.vector_store as vs
    cls = _make_fake_ef_class()
    monkeypatch.setitem(vs._EMEBEDDIG_FUNCTIONS, "fake", cls)
    return cls


@pytest.fixture
def make_store(fake_ef, tmp_path):
    """Factory for fresh, empty VectorStores in a temporary folder."""
    from hcs4os._vector_database.vector_store import VectorStore

    def _make(collection_name: str = "test_collection") -> VectorStore:
        return VectorStore(
            collection_name=collection_name,
            chromadb_path=str(tmp_path / "chroma"),
            embedding_function_type="fake",
        )
    return _make


@pytest.fixture
def fiction_store(make_store, fiction):
    store = make_store("fiction_test")
    store.index(classification_system=fiction)
    return store


# ---------------------------------------------------------------------------
# fakes for dspy modules, so no LLM is ever called
# ---------------------------------------------------------------------------

class FakeModule:
    """Stands in for dspy.ChainOfThought / Predict / ReAct.

    Records every call (kwargs and the dspy LM active at call time) and
    returns a fixed dspy.Prediction.
    """

    def __init__(self, **outputs):
        self.outputs = outputs
        self.calls: list[dict] = []
        self.lms_seen: list = []

    def __call__(self, **kwargs):
        import dspy
        self.calls.append(kwargs)
        self.lms_seen.append(dspy.settings.lm)
        return dspy.Prediction(**self.outputs)


@pytest.fixture
def fake_module():
    return FakeModule
