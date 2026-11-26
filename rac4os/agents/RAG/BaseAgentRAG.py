from typing import Any
import dspy
from ...classification_system import get_classification_system
from ..._vector_database.Retriever import Retriever

class BaseRAGAgent(dspy.Module):
    
    """
    Classifies an item with the retriever exposed to the model as a tool.

    A `dspy.ReAct` agent gets one tool, `search_category`, and decides for itself how often
    and with which query to use it. Where the linear pipeline is bound by its single
    initial query, this agent can reformulate and search again when the candidates it got
    back do not fit, which is what makes it the approach to reach for when retrieval is the
    part that fails.

    The cost of that is variance: how many model calls an item takes is decided by the
    model, not by you.

    Parameters:
        classification_name (str): dataset name resolved through `get_classification_system`.
        api_key (str): credential for the LM provider.
        model_name (str): LiteLLM model identifier, e.g. "openai/gpt-4o-mini".
        signature (type[dspy.Signature]): the I/O contract; its docstring is the system prompt.
        search_category_tool_doc (str): tool description the agent reads to decide how to query the store. Say what a good query looks like for your labels here.
        store (Retriever): a filled retriever; this class never indexes.
        codes_to_include (list[str]|None): kept for symmetry with the other classifiers and otherwise unused. Restrict the candidate space when you fill the store, with `store.index(codes_to_include=[...])`.
        api_base (str|None): base URL of a self-hosted or proxied endpoint; None uses the provider's own.

    Attributes:
        classification_system (ClassificationSystem): the taxonomy, loaded so that a dataset name means the same thing across all approaches.
        store (Retriever): the retriever behind the tool.
        agent (dspy.ReAct): the reasoning loop over the search tool.
        lm (dspy.LM): the language model this module pins for its own calls.
    """
    def __init__(
        self,
        classification_name:str,
        api_key:str,
        model_name:str,
        signature:type[dspy.Signature],
        search_category_tool_doc:str,
        store:Retriever,
        codes_to_include:list[str]|None=None,
        api_base:str|None=None           
    )->None:
        """
        Builds the LM, loads the classification system and wraps the store's `search_category`
        into the agent's single tool.

        Note that this configures `dspy` globally as well as storing the LM on the instance, so
        constructing a second classifier with a different model changes the global default.
        `forward` pins its own LM, so already-constructed classifiers keep working.
        """
        super().__init__()


        assert model_name is not None
        assert classification_name is not None
        assert signature is not None
        
        self.signature = signature
        self.search_category_tool_doc = search_category_tool_doc
        self.codes_to_include = codes_to_include

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

        self.store = store
        
        self.signature = self.signature
        
        self.agent = dspy.ReAct(
            self.signature,  # type: ignore
            tools=[
                dspy.Tool(
                    self.store.search_category,
                    name="search_category",
                    desc= self.search_category_tool_doc
                )
            ],
        )
        
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
        
        
        
            
       
