from typing import Literal, Any
import dspy
from ...classification_system import get_classification_system
import warnings


RegisteredClassifications = Literal["COICOP_2018", "ICATUS_2016", "NAF_rev2", "SEA_2021"]

class BaseHierarchicalNavigationAgent(dspy.Module):
    
    """
    Classifies an item by walking the taxonomy top-down instead of retrieving candidates.

    A `dspy.ReAct` agent is given four navigation tools over your classification system
    (`get_root_categories`, `get_children`, `get_parent`, `get_code`) and reasons its way
    from the first level down to a code, deciding at each step which branch to descend.
    Nothing is embedded and no vector store is involved, so this is the approach that works
    on a taxonomy you have only just described.

    The tool descriptions are passed in rather than hardcoded, because what a tool should
    say about itself depends on how your codes are written: the agent reads them to decide
    which tool to call next.

    Parameters:
        classification_name (str): dataset name resolved through `get_classification_system`.
        api_key (str): credential for the LM provider.
        model_name (str): LiteLLM model identifier, e.g. "openai/gpt-4o-mini".
        signature (type[dspy.Signature]): the I/O contract; its docstring is the system prompt.
        get_root_categories_doc (str): tool description for listing the first level.
        get_children_doc (str): tool description for descending one level.
        get_code_doc (str): tool description for reading a single code.
        get_parent_doc (str): tool description for stepping back up one level.
        api_base (str|None): base URL of a self-hosted or proxied endpoint; None uses the provider's own.

    Attributes:
        classification_system (ClassificationSystem): the taxonomy being navigated.
        agent (dspy.ReAct): the reasoning loop over the four tools.
        lm (dspy.LM): the language model this module pins for its own calls.
    """
    def __init__(
        self,
        classification_name:RegisteredClassifications,
        api_key:str,
        model_name:str,
        signature:type[dspy.Signature],
        get_root_categories_doc:str,
        get_children_doc:str,
        get_code_doc:str,
        get_parent_doc:str,
        api_base:str|None=None,
    )->None:
        
        """
        Builds the LM, loads the classification system and wires up the four navigation tools.

        Note that this configures `dspy` globally as well as storing the LM on the instance, so
        constructing a second classifier with a different model changes the global default. The
        `forward` methods pin their own LM, so already-constructed classifiers keep working.
        """
        assert model_name is not None
        assert classification_name is not None
        
        self.signature:type[dspy.Signature] = signature
        self.get_root_categories_doc:str = get_root_categories_doc
        self.get_children_doc:str = get_children_doc
        self.get_code_doc:str = get_code_doc
        self.get_parent_doc:str = get_parent_doc
        
        
        self.lm = dspy.LM(
            api_key=api_key,
            api_base=api_base,
            model=model_name
        )
        dspy.configure(
            lm=self.lm,
            track_usage=True
        )
        
        self.classification_name = classification_name
        self.classification_system = get_classification_system(
            classification_name
        )
        self.agent = dspy.ReAct(
            self.signature, 
            tools=[
                dspy.Tool(
                    self.get_root_categories_tool,
                    name="get_root_categories",
                    desc=self.get_root_categories_doc, 
                ),
                dspy.Tool(
                    self.get_children_tool, 
                    name="get_children",
                    desc=self.get_children_doc, 
                ),
                dspy.Tool(
                    self.get_parent_tool, 
                    name="get_parent",
                    desc=self.get_parent_doc  
                ),
                dspy.Tool(
                    self.get_code_tool, 
                    name="get_code",
                    desc=self.get_code_doc  
                ),
            ],
        )
    
    def get_code_tool(self, code: str) -> dict:
        """
        Tool: read the full record of one code.

        Parameters:
            code (str): the code to look up.

        Returns:
            dict: the `Code` record as a dict.
        """
        return self.classification_system.get_code(code).to_dict()

    def get_children_tool(self, code: str) -> list[dict]:
        """
        Tool: list the categories directly below a code, which is how the agent descends.

        Parameters:
            code (str): the parent code.

        Returns:
            list[dict]: one dict per direct child; empty for a leaf.
        """
        return [c.to_dict() for c in self.classification_system.get_children(code)]

    def get_parent_tool(self, code: str) -> dict:
        """
        Tool: step back up one level, which is how the agent recovers from a wrong branch.

        Parameters:
            code (str): the code whose parent is wanted.

        Returns:
            dict: the parent's `Code` record as a dict.
        """
        return self.classification_system.get_parent(code).to_dict() # type: ignore

    def get_root_categories_tool(self) -> list[dict]:
        """
        Tool: list the first level of the taxonomy, where every traversal starts.

        Returns:
            list[dict]: one `code`/`description` dict per first-level category.
        """
        return self.classification_system.get_root_categories() 
    
    def forward(self, **kwargs: Any) -> dspy.Prediction:
        """
        Classify one item.

        The signature's input fields are checked against the keyword arguments first, so a
        misnamed field surfaces immediately instead of as an obscure failure once the
        trajectory is already running. The call is wrapped in a `dspy.context` that pins this
        module's own LM, which makes it safe to hold several classifiers with different models
        in one process.

        Parameters:
            **kwargs (Any): one keyword argument per input field of the signature.

        Returns:
            dspy.Prediction: the signature's output fields, plus the ReAct trajectory under `thought_{i}`, `tool_name_{i}`, `tool_args_{i}` and `observation_{i}`.

        Raises:
            ValueError: if an input field declared by the signature was not passed.
        """
        missing = set(self.signature.input_fields) - kwargs.keys()
        if missing:
            raise ValueError(f"Missing input fields: {sorted(missing)}")

        with dspy.context(lm=self.lm, track_usage=True):
            return self.agent(**kwargs)
      