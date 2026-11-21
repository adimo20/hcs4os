"""agents/: HierarchicalNavigationAgent and RAGAgent (ReAct), without calling an LLM."""
import inspect

import dspy
import pytest

from hcs4os.agents.HierarchicalNavigation import HierarchicalNavigationAgent
from hcs4os.agents.HierarchicalNavigation.BaseAgent import BaseHierarchicalNavigationAgent
from hcs4os.agents.HierarchicalNavigation.registry import registry as nav_registry
from hcs4os.classification_system.mapping import mapping


# ---------------------------------------------------------------------------
# registries: every entry must fit its base constructor
# ---------------------------------------------------------------------------

def _required_config_keys(base_cls) -> set[str]:
    params = inspect.signature(base_cls.__init__).parameters
    generic = {"self", "classification_name", "api_key", "model_name", "api_base",
               "store", "codes_to_include"}
    return {p for p in params if p not in generic}


@pytest.mark.parametrize("name, config", list(nav_registry.items()))
def test_navigation_registry_entries_match_constructor(name, config):
    assert name in mapping, "registry key must be a dataset key in mapping.py"
    assert set(config) == _required_config_keys(BaseHierarchicalNavigationAgent)
    assert all(isinstance(v, str) and v.strip() for k, v in config.items() if k.endswith("_doc"))
    assert config["signature"].__doc__.strip()
    assert "explaination" in config["signature"].output_fields


def test_rag_agent_registry_entries_match_constructor():
    pytest.importorskip("chromadb")
    from hcs4os.agents.RAG.BaseAgentRAG import BaseRAGAgent
    from hcs4os.agents.RAG.registry import registry as rag_agent_registry
    for name, config in rag_agent_registry.items():
        assert name in mapping
        assert set(config) == _required_config_keys(BaseRAGAgent)
        assert config["search_category_tool_doc"].strip()
        assert "explaination" in config["signature"].output_fields


# ---------------------------------------------------------------------------
# HierarchicalNavigationAgent
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nav_agent():
    return HierarchicalNavigationAgent(classification_name="COICOP_2018",
                                       api_key=None, model_name="openai/test")


def test_navigation_unknown_classification_raises():
    with pytest.raises(ValueError, match="Unknown classification system"):
        HierarchicalNavigationAgent(classification_name="WZ_2008",  # type: ignore[arg-type]
                                    api_key=None, model_name="openai/x")


def test_navigation_construction_without_api_key(nav_agent):
    assert nav_agent.lm.model == "openai/test"


def test_navigation_agent_has_four_tools(nav_agent):
    assert {"get_root_categories", "get_children", "get_parent", "get_code"} <= set(nav_agent.agent.tools)


def test_tool_get_root_categories(nav_agent):
    roots = nav_agent.get_root_categories_tool()
    assert {"code": "01", "description": roots[0]["description"]} == roots[0]
    assert len(roots) == 15


def test_tool_get_children_returns_dicts(nav_agent):
    children = nav_agent.get_children_tool("01.1.1")
    assert all(isinstance(c, dict) for c in children)
    assert children[0]["code"] == "01.1.1.1"


def test_tool_get_code_returns_dict(nav_agent):
    assert nav_agent.get_code_tool("01.1.1.1")["code"] == "01.1.1.1"


def test_tool_get_code_unknown_raises(nav_agent):
    with pytest.raises(ValueError):
        nav_agent.get_code_tool("99.9")


def test_tool_get_parent(nav_agent):
    assert nav_agent.get_parent_tool("01.1.1.1")["code"] == "01.1.1"


def test_navigation_forward_validates_input_fields(nav_agent):
    with pytest.raises(ValueError, match=r"Missing input fields: \['input_expense'\]"):
        nav_agent(wrong_field="rice")


def test_navigation_forward_delegates_with_own_lm(nav_agent, fake_module, monkeypatch):
    fake = fake_module(coicop_code="01.1.1.1", explaination="")
    monkeypatch.setattr(nav_agent, "agent", fake)
    dspy.configure(lm=dspy.LM("openai/some-other-model"))
    prediction = nav_agent(input_expense="rice")
    assert prediction.coicop_code == "01.1.1.1"
    assert fake.calls == [{"input_expense": "rice"}]
    assert fake.lms_seen[0] is nav_agent.lm


def test_base_navigation_agent_with_custom_prompt(fiction, register_fiction):
    class Sig(dspy.Signature):
        """Walk the tree and classify the character."""
        character: str = dspy.InputField()
        fiction_code: str = dspy.OutputField()

    agent = BaseHierarchicalNavigationAgent(
        classification_name=register_fiction, api_key=None, model_name="openai/x",
        signature=Sig, get_root_categories_doc="roots", get_children_doc="children",
        get_code_doc="code", get_parent_doc="parent",
    )
    assert [r["code"] for r in agent.get_root_categories_tool()] == ["1", "2"]
    assert agent.agent.tools["get_children"].desc == "children"


# ---------------------------------------------------------------------------
# RAGAgent
# ---------------------------------------------------------------------------

@pytest.fixture
def rag_agent(make_store):
    pytest.importorskip("chromadb")
    from hcs4os.agents.RAG.RAGAgent import RAGAgent
    return RAGAgent(classification_name="ICATUS_2016", api_key=None,
                    model_name="openai/test", store=make_store("icatus_agent"))


def test_rag_agent_unknown_classification_raises(make_store):
    from hcs4os.agents.RAG.RAGAgent import RAGAgent
    with pytest.raises(ValueError, match="Unknown classification system"):
        RAGAgent(classification_name="WZ_2008", api_key=None,  # type: ignore[arg-type]
                 model_name="openai/x", store=make_store())


def test_rag_agent_does_not_index_on_construction(rag_agent):
    assert rag_agent.store.collection.count() == 0


def test_rag_agent_search_tool_is_the_stores_search(rag_agent):
    tool = rag_agent.agent.tools["search_category"]
    assert tool.func == rag_agent.store.search_category
    assert tool.desc.strip()


def test_rag_agent_search_tool_works_after_indexing(rag_agent):
    rag_agent.store.index(classification_system=rag_agent.classification_system)
    results = rag_agent.agent.tools["search_category"].func(query="Eating meals", k=3)
    assert len(results) == 3


def test_rag_agent_forward_validates_input_fields(rag_agent):
    with pytest.raises(ValueError, match=r"Missing input fields: \['input_activity'\]"):
        rag_agent(input_expense="cooking")


def test_rag_agent_forward_delegates_with_own_lm(rag_agent, fake_module, monkeypatch):
    fake = fake_module(icatus_code="110", explaination="")
    monkeypatch.setattr(rag_agent, "agent", fake)
    dspy.configure(lm=dspy.LM("openai/some-other-model"))
    assert rag_agent(input_activity="cooking dinner").icatus_code == "110"
    assert fake.lms_seen[0] is rag_agent.lm
