"""Integration tests against a live embedding server and LLM.

Skipped unless configured, so the default `pytest` run stays offline:

    rac4os_OLLAMA_URL=http://192.168.178.48:11434
    rac4os_EMBEDDING_MODEL=qwen3-embedding:0.6b
    rac4os_LLM_MODEL=ollama_chat/qwen3:4b          # optional, for the classifier test
    pytest -m integration

The variables can also live in a .env file (needs python-dotenv).
"""
import os

import pytest

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

pytest.importorskip("chromadb")

OLLAMA_URL = os.getenv("rac4os_OLLAMA_URL")
EMBEDDING_MODEL = os.getenv("rac4os_EMBEDDING_MODEL")
LLM_MODEL = os.getenv("rac4os_LLM_MODEL")
API_KEY = os.getenv("rac4os_TEST_API_KEY")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (OLLAMA_URL and EMBEDDING_MODEL),
                       reason="set rac4os_OLLAMA_URL and rac4os_EMBEDDING_MODEL"),
]


@pytest.fixture
def ollama_fiction_store(tmp_path, fiction):
    from rac4os._vector_database.vector_store import VectorStore
    store = VectorStore(
        collection_name="fiction_integration",
        chromadb_path=str(tmp_path / "chroma"),
        embedding_function_type="ollama",
        url=OLLAMA_URL,
        model_name=EMBEDDING_MODEL,
    )
    store.index(classification_system=fiction)
    return store


@pytest.mark.parametrize("input_text, expected_code", [
    ("Member of the Jedi high council", "111"),
    ("Sith Lord", "112"),
    ("astromech robot", "131"),
    ("halfling from the Shire", "211"),
    ("wizard with a grey staff", "231"),
])
def test_retrieval_recall_at_5(ollama_fiction_store, input_text, expected_code):
    results = ollama_fiction_store.search_category(input_text, k=5, where={"level": 3})
    assert expected_code in [r["code"] for r in results]


@pytest.mark.skipif(not LLM_MODEL, reason="set rac4os_LLM_MODEL")
@pytest.mark.parametrize("input_text, expected_code", [
    ("Darth Vader Sith Lord", "112"),      
    ("Han Solo Rebellion", "121"),           
])
def test_rag_classifier_end_to_end(ollama_fiction_store, register_fiction, input_text, expected_code):
    import dspy
    from rac4os.rag.BaseRAG import BaseRAGCoT

    class Sig(dspy.Signature):
        """Classify the fictional character into exactly one level-3 code from the
        candidates. Read the includes/excludes notes; never invent a code."""
        character: str = dspy.InputField()
        retrieved_candidates: str = dspy.InputField()
        fiction_code: str = dspy.OutputField()
        explanation: str = dspy.OutputField()

    rag = BaseRAGCoT(classification_name=register_fiction, api_key=API_KEY, model_name=LLM_MODEL,
                     api_base=None, signature=Sig, query_field="character",
                     context_field="retrieved_candidates", store=ollama_fiction_store)
    prediction = rag(query=input_text, k=8, where={"level": 3})
    assert prediction.fiction_code.strip() == expected_code
