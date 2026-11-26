import dspy
from .._vector_database.Retriever import Retriever
from ..classification_system.registry import get_classification_system
from typing import Literal
import json
    
    
class BaseRAGCoT(dspy.Module):
    
    """
    Classifies an item by retrieving once and reasoning once.

    The item is embedded, the k nearest categories are pulled from the retriever,
    serialised to JSON and placed in the signature's context field next to the item, and a
    `dspy.ChainOfThought` picks one of them. One retrieval call and one model call per
    item, which makes it the cheapest of the three approaches and the one to try first.

    Its structural weakness is worth knowing before you choose it: retrieval happens once,
    on the raw item, before any reasoning. If the correct category is not among the k
    retrieved, no amount of generator quality recovers it. The retriever's accuracy at k is
    therefore a ceiling on the whole pipeline, and measuring it on its own tells you
    whether you have a retrieval problem or a generation problem.

    The classification system is loaded even though the linear pipeline never traverses it,
    so that one dataset name means the same thing across all approaches and so you can
    enrich candidates from the hierarchy yourself.

    Parameters:
        classification_name (str): dataset name resolved through `get_classification_system`.
        api_key (str): credential for the LM provider.
        model_name (str): LiteLLM model identifier, e.g. "openai/gpt-4o-mini".
        signature (dspy.Signature): the I/O contract; its docstring is the system prompt.
        query_field (str): name of the signature input field that holds the item.
        context_field (str): name of the signature input field that holds the candidate list.
        store (Retriever): a filled retriever; this class never indexes.
        codes_to_include (list[str]|None): accepted for symmetry with the other classifiers and otherwise unused. Restrict the candidate space when you fill the store, with `store.index(codes_to_include=[...])`.
        api_base (str|None): base URL of a self-hosted or proxied endpoint; None uses the provider's own.

    Attributes:
        classification_system (ClassificationSystem): the taxonomy behind `classification_name`.
        store (Retriever): the retriever the candidates come from.
        CoT (dspy.ChainOfThought): the generation step.
        lm (dspy.LM): the language model configured for this module.
    """
    def __init__(
        self,
        classification_name,
        api_key:str,
        model_name:str,
        signature:dspy.Signature,
        query_field:str,
        context_field:str,
        store: Retriever,        
        codes_to_include:list[str]|None=None,
        api_base:str|None=None                      
    )->None:

        """
        Builds the LM, loads the classification system and the chain-of-thought step.

        `query_field` and `context_field` are asserted against the signature's declared input
        fields, so a name that does not exist fails here rather than at the first call.

        Unlike the agentic classifiers, `forward` does not pin an LM per call: this module
        relies on the global `dspy.configure` made here. If you run several classifiers with
        different models in one process, construct each one immediately before use.
        """
        assert model_name is not None
        assert query_field is not None
        assert context_field is not None       
        assert query_field in list(signature.input_fields.keys()) # type: ignore
        assert context_field in list(signature.input_fields.keys())  # type: ignore
        
        
        self.codes_to_include = codes_to_include
        self.signature:dspy.Signature = signature
        self.query_field = query_field
        self.context_field = context_field


        self.lm = dspy.LM(
            api_key=api_key,
            api_base=api_base,
            model=model_name
        )
        dspy.configure(
            lm=self.lm,
            track_usage=True
        )

        print("Loading classification system")
        self.classification_name = classification_name
        self.classification_system = get_classification_system(
            classification_name
        )

        print("Creating VectorDB")        
        self.store = store
        
        self.CoT = dspy.ChainOfThought(
            signature=self.signature # type: ignore 
        )
      
    
    def forward(self, query: str, k: int, where: dict|None = None) -> dspy.Prediction:
        """
        Retrieve candidates for one item and classify it over them.

        Parameters:
            query (str): the item to classify, used both as the retrieval query and as the item the generator sees.
            k (int): how many candidates to retrieve. Raising it makes the correct category more likely to be present and gives the model more chances to pick a wrong one, and it spends context that competes with how richly each candidate is described.
            where (dict|None): metadata filter passed straight to the store, e.g. `{"level": 3}` to retrieve only one level. Matches the metadata as your loader emitted it, so a level that left the loader as a string has to be filtered as one.

        Returns:
            dspy.Prediction: the signature's output fields, plus `reasoning` from the chain of thought.
        """
        retrieved_context = self.store.search_category(query, k, where=where)
        retrieved_context_str = json.dumps(retrieved_context, indent=4, ensure_ascii=False)
        return self.CoT(**{
            self.query_field: query,
            self.context_field: retrieved_context_str,
        })
        
        
        
            
        

