"""VectorStore: embedding backend selection, indexing and retrieval (offline)."""
import json

import pytest

pytest.importorskip("chromadb", reason="vector store needs an extra: pip install 'hcs4os[ollama]'")

import hcs4os._vector_database.vector_store as vs  # noqa: E402
from hcs4os._vector_database.vector_store import VectorStore, get_ef  # noqa: E402


# ---------------------------------------------------------------------------
# get_ef
# ---------------------------------------------------------------------------

def test_get_ef_unknown_type_raises_value_error():
    with pytest.raises(ValueError, match="Unknown embedding function: 'cohere'"):
        get_ef("cohere")  # type: ignore[arg-type]


def test_get_ef_forwards_kwargs(fake_ef):
    ef = get_ef("fake", model_name="m", url="http://x")  # type: ignore[arg-type]
    assert ef.kwargs == {"model_name": "m", "url": "http://x"}


@pytest.mark.parametrize("backend, extra", [
    ("ollama", "ollama"), ("openai", "openai"), ("sentence_transformers", "local"),
])
@pytest.mark.parametrize("error", [
    ImportError("No module named 'x'"),
    ValueError("The x python package is not installed. Please install it"),
])
def test_get_ef_missing_package_points_to_the_right_extra(monkeypatch, backend, extra, error):
    def raises(**kwargs):
        raise error
    monkeypatch.setitem(vs._EMEBEDDIG_FUNCTIONS, backend, raises)
    with pytest.raises(ImportError, match=rf'pip install "hcs4os\[{extra}\]"'):
        get_ef(backend)


def test_get_ef_other_value_errors_are_not_rewritten(monkeypatch):
    def raises(**kwargs):
        raise ValueError("model_name must be set")
    monkeypatch.setitem(vs._EMEBEDDIG_FUNCTIONS, "ollama", raises)
    with pytest.raises(ValueError, match="model_name must be set"):
        get_ef("ollama")


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def test_new_store_is_empty(make_store):
    store = make_store()
    assert store.collection.count() == 0
    assert store.embedding_function_type == "fake"
    assert store.embedding_batch_size == 16


def test_collection_uses_cosine_distance(make_store):
    store = make_store()
    assert store.collection.configuration["hnsw"]["space"] == "cosine"


def test_store_persists_between_instances(make_store, fiction):
    make_store("persisted").index(classification_system=fiction)
    assert make_store("persisted").collection.count() == len(fiction.codes)


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

def test_index_all_codes(fiction_store, fiction):
    assert fiction_store.collection.count() == len(fiction.codes)


def test_index_subset(make_store, fiction):
    store = make_store()
    level3 = [c.code for c in fiction.codes if c.level == 3]
    store.index(classification_system=fiction, codes_to_include=level3)
    assert store.collection.count() == len(level3)


def test_index_ids_are_deterministic(fiction_store):
    got = fiction_store.collection.get(ids=["id_111"])
    assert got["ids"] == ["id_111"]


def test_index_embeds_only_the_description(fiction_store):
    got = fiction_store.collection.get(ids=["id_111"], include=["documents", "metadatas"])
    assert got["documents"] == ["Jedi and light side Force users"]
    assert "description" not in got["metadatas"][0]


def test_index_flattens_details_to_json(fiction_store):
    meta = fiction_store.collection.get(ids=["id_111"], include=["metadatas"])["metadatas"][0]
    assert meta["code"] == "111"
    assert meta["level"] == 3
    details = json.loads(meta["details"])
    assert "Darth Vader (112)" in details["excludes"]
    assert "Jedi" in details["keywords"]


def test_reindexing_does_not_duplicate(fiction_store, fiction):
    fiction_store.index(classification_system=fiction)
    assert fiction_store.collection.count() == len(fiction.codes)


@pytest.mark.parametrize("batch_size", [1, 5, 100])
def test_batch_size_does_not_change_result(fake_ef, tmp_path, fiction, batch_size):
    store = VectorStore("batched", str(tmp_path), embedding_function_type="fake",
                        embedding_batch_size=batch_size)
    store.index(classification_system=fiction)
    assert store.collection.count() == len(fiction.codes)


# ---------------------------------------------------------------------------
# search_category
# ---------------------------------------------------------------------------

def test_search_returns_k_records_with_description(fiction_store):
    results = fiction_store.search_category("Jedi", k=3)
    assert len(results) == 3
    for r in results:
        assert {"code", "description", "level", "details"} <= set(r)


def test_search_finds_obvious_match_first(fiction_store):
    assert fiction_store.search_category("Droids", k=1)[0]["code"] == "131"


def test_search_where_filters_by_level(fiction_store):
    results = fiction_store.search_category("characters", k=5, where={"level": 3})
    assert results and all(r["level"] == 3 for r in results)


def test_search_on_subset_only_returns_indexed_codes(make_store, fiction):
    store = make_store()
    store.index(classification_system=fiction, codes_to_include=["111", "112"])
    codes = {r["code"] for r in store.search_category("anything", k=2)}
    assert codes == {"111", "112"}
