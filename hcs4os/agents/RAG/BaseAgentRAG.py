import dspy
from ...classification_system import get_classification_system
from ..._vector_database.vector_store import VectorStore

class BaseRAGAgent(dspy.Module):
    
    def __init__(
        self,
        classification_name:str,
        api_key:str,
        model_name:str,
        signature:dspy.Signature,
        search_category_tool_doc:str,
        store:VectorStore,
        codes_to_include:list[str]|None=None,
        api_base:str|None=None,        
        create_new_collection:bool=True,             
    )->None:

        assert api_key is not None
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
        
        if create_new_collection:
            self.store.index(
                codes_to_include=self.codes_to_include,
                classification_system=self.classification_system
            )

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
        
        
        
            
       
