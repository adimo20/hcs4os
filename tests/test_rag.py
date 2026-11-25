"""rag/: BaseRAGCoT, RAGChainOfThought, QueryRephraser and RRR_RAGChainOfThought.

The LLM is never called: the dspy modules inside the classifiers are replaced
by FakeModule (see conftest.py), which records what it was called with.
"""
import json

import dspy
import pytest

pytest.importorskip("chromadb", reason="RAG needs an extra: pip install 'rac4os[ollama]'")

from rac4os.rag import registry as rag_registry  # noqa: E402
from rac4os.rag.BaseRAG import BaseRAGCoT  # noqa: E402
from rac4os.rag.RAG import QueryRephraser, RAGChainOfThought, RRR_RAGChainOfThought  # noqa: E402


class FictionSignature(dspy.Signature):
    """Classify a fictional character into one code from the candidates."""
    character: str = dspy.InputField()
    retrieved_candidates: str = dspy.InputField()
    fiction_code: str = dspy.OutputField()
    explanation: str = dspy.OutputField()


class FictionRephrasing(dspy.Signature):
    """Rewrite a character description into category vocabulary."""
    character: str = dspy.InputField()
    rephrased_query: str = dspy.OutputField()


@pytest.fixture
def fiction_rag(register_fiction, fiction_store):
    return BaseRAGCoT(
        classification_name=register_fiction,
        api_key=None,                       # local models need no key since v0.2.8
        model_name="ollama_chat/test-model",
        signature=FictionSignature,
        query_field="character",
        context_field="retrieved_candidates",
        store=fiction_store,
    )


# ---------------------------------------------------------------------------
# BaseRAGCoT construction
# ---------------------------------------------------------------------------

def test_construction_without_api_key(fiction_rag):
    assert fiction_rag.lm.model == "ollama_chat/test-model"


def test_construction_does_not_index(register_fiction, make_store):
    store = make_store()
    BaseRAGCoT(classification_name=register_fiction, api_key=None, model_name="openai/x",
               signature=FictionSignature, query_field="character",
               context_field="retrieved_candidates", store=store)
    assert store.collection.count() == 0


@pytest.mark.parametrize("query_field, context_field", [
    ("not_a_field", "retrieved_candidates"),
    ("character", "not_a_field"),
])
def test_field_names_must_exist_on_signature(register_fiction, make_store, query_field, context_field):
    with pytest.raises(AssertionError):
        BaseRAGCoT(classification_name=register_fiction, api_key=None, model_name="openai/x",
                   signature=FictionSignature, query_field=query_field,
                   context_field=context_field, store=make_store())


def test_model_name_is_required(register_fiction, make_store):
    with pytest.raises(AssertionError):
        BaseRAGCoT(classification_name=register_fiction, api_key=None, model_name=None,  # type: ignore[arg-type]
                   signature=FictionSignature, query_field="character",
                   context_field="retrieved_candidates", store=make_store())


# ---------------------------------------------------------------------------
# BaseRAGCoT.forward
# ---------------------------------------------------------------------------

def test_forward_passes_query_and_candidates(fiction_rag, fake_module):
    fiction_rag.CoT = fake_module(fiction_code="131", explanation="it is a droid")
    prediction = fiction_rag(query="Droids", k=3)

    assert prediction.fiction_code == "131"
    call = fiction_rag.CoT.calls[0]
    assert call["character"] == "Droids"
    candidates = json.loads(call["retrieved_candidates"])
    assert len(candidates) == 3
    assert candidates[0]["code"] == "131"


def test_forward_candidates_keep_umlauts(fiction_rag, fake_module):
    fiction_rag.CoT = fake_module(fiction_code="221", explanation="")
    fiction_rag(query="Nazgûl", k=1)
    assert "Nazgûl" in fiction_rag.CoT.calls[0]["retrieved_candidates"]   # ensure_ascii=False


def test_forward_where_filter(fiction_rag, fake_module):
    fiction_rag.CoT = fake_module(fiction_code="111", explanation="")
    fiction_rag(query="Jedi", k=4, where={"level": 3})
    candidates = json.loads(fiction_rag.CoT.calls[0]["retrieved_candidates"])
    assert all(c["level"] == 3 for c in candidates)


# ---------------------------------------------------------------------------
# RAGChainOfThought (registered configurations)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, config", list(RAGChainOfThought.CONFIGS.items()))
def test_registered_configs_are_consistent(name, config):
    inputs = config["signature"].input_fields
    assert config["query_field"] in inputs
    assert config["context_field"] in inputs
    assert "explanation" in config["signature"].output_fields
    assert config["signature"].__doc__.strip()


def test_unknown_classification_raises(make_store):
    with pytest.raises(ValueError, match="Unknown classification system"):
        RAGChainOfThought(classification_name="WZ_2008", api_key=None,  # type: ignore[arg-type]
                          model_name="openai/x", store=make_store())


def test_registered_classifier_end_to_end(make_store, fake_module):
    from rac4os.classification_system import get_classification_system
    store = make_store("coicop")
    store.index(classification_system=get_classification_system("COICOP_2018"))
    rag = RAGChainOfThought(classification_name="COICOP_2018", api_key=None,
                            model_name="openai/x", store=store)
    rag.CoT = fake_module(coicop_code="01.1.1.1", explanation="rice")
    prediction = rag(query="rice", k=5)
    assert prediction.coicop_code == "01.1.1.1"
    assert set(rag.CoT.calls[0]) == {"input_expense", "retrieved_candidates"}


# ---------------------------------------------------------------------------
# QueryRephraser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["COICOP_2018", "ICATUS_2016", "SEA_2021", "SEA_2021_REPHRASED"])
def test_rephraser_picks_registered_signature(name):
    r = QueryRephraser(lm=dspy.LM("openai/x"), classification_name=name)
    assert r.signature is r.mapping[name]
    assert "rephrased_query" in r.signature.output_fields


def test_rephraser_custom_signature():
    r = QueryRephraser(lm=dspy.LM("openai/x"), classification_name=None, signature=FictionRephrasing)
    assert r.signature is FictionRephrasing
    assert r.input_field == "character"


def test_rephraser_needs_name_or_signature():
    with pytest.raises(AssertionError, match="No rephrasing signature"):
        QueryRephraser(lm=dspy.LM("openai/x"), classification_name="WZ_2008")


def test_rephraser_registered_name_wins_over_custom_signature():
    """Documented behaviour (README): signature= is only for unregistered classifications."""
    r = QueryRephraser(lm=dspy.LM("openai/x"), classification_name="COICOP_2018",
                       signature=FictionRephrasing)
    assert r.signature is rag_registry.CoicopRephrasingSignature


def test_rephraser_feeds_query_to_signature_input_and_uses_own_lm(fake_module):
    lm = dspy.LM("openai/rephraser")
    r = QueryRephraser(lm=lm, classification_name=None, signature=FictionRephrasing)
    r.rephrase = fake_module(rephrased_query="  Jedi and light side Force users  ")
    out = r(query="green guy with a lightsaber")
    assert out.rephrased_query == "Jedi and light side Force users"      # stripped
    assert r.rephrase.calls[0] == {"character": "green guy with a lightsaber"}
    assert r.rephrase.lms_seen[0] is lm


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_rephraser_falls_back_to_raw_query(fake_module, empty):
    r = QueryRephraser(lm=dspy.LM("openai/x"), classification_name=None, signature=FictionRephrasing)
    r.rephrase = fake_module(rephrased_query=empty)
    assert r(query="raw item").rephrased_query == "raw item"


# ---------------------------------------------------------------------------
# RRR_RAGChainOfThought
# ---------------------------------------------------------------------------

@pytest.fixture
def coicop_rrr(make_store, fake_module):
    from rac4os.classification_system import get_classification_system
    store = make_store("coicop_rrr")
    store.index(classification_system=get_classification_system("COICOP_2018"))
    rephraser = QueryRephraser(lm=dspy.LM("openai/small"), classification_name="COICOP_2018")
    rephraser.rephrase = fake_module(rephrased_query="Rice")
    rrr = RRR_RAGChainOfThought(rephraser, classification_name="COICOP_2018",
                                api_key=None, model_name="openai/large", store=store)
    rrr.CoT = fake_module(coicop_code="01.1.1.1.2", explanation="")
    return rrr


def test_rrr_retrieves_with_rephrased_query(coicop_rrr, monkeypatch):
    seen = []
    original = coicop_rrr.store.search_category
    monkeypatch.setattr(coicop_rrr.store, "search_category",
                        lambda q, k, where=None: seen.append(q) or original(q, k, where=where))
    coicop_rrr(query="BASMATI REIS 1KG BTL", k=5)
    assert seen == ["Rice"]


def test_rrr_classifies_the_original_query(coicop_rrr):
    coicop_rrr(query="BASMATI REIS 1KG BTL", k=5)
    assert coicop_rrr.CoT.calls[0]["input_expense"] == "BASMATI REIS 1KG BTL"


def test_rrr_records_rephrased_query_on_prediction(coicop_rrr):
    prediction = coicop_rrr(query="BASMATI REIS 1KG BTL", k=5)
    assert prediction.rephrased_query == "Rice"
    assert prediction.coicop_code == "01.1.1.1.2"


def test_rrr_read_step_uses_its_own_lm(coicop_rrr):
    coicop_rrr(query="x", k=3)
    assert coicop_rrr.CoT.lms_seen[0] is coicop_rrr.lm
    assert coicop_rrr.rephrasing_module.rephrase.lms_seen[0] is coicop_rrr.rephrasing_module.lm
