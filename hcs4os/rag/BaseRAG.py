import dspy
from .._vector_database.vector_store import VectorStore
from ..classification_system.registry import get_classification_system
from typing import Literal
import json
    
    
class BaseRAGCoT(dspy.Module):
    
    def __init__(
        self,
        classification_name,
        api_key:str,
        model_name:str,
        signature:dspy.Signature,
        query_field:str,
        context_field:str,
        store: VectorStore,        
        codes_to_include:list[str]|None=None,
        api_base:str|None=None,
        create_new_collection:bool=True                          
    )->None:

        assert api_key is not None
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
        
        if create_new_collection:
            print("Indexing classification system")
            self.store.index(
                self.codes_to_include,
                self.classification_system
            )
        print("Create vector store")
        
        self.CoT = dspy.ChainOfThought(
            signature=self.signature # type: ignore 
        )
      
    
    def forward(self, query: str, k: int, where: dict|None = None):
        retrieved_context = self.store.search_category(query, k, where=where)
        retrieved_context_str = json.dumps(retrieved_context, indent=4, ensure_ascii=False)
        return self.CoT(**{
            self.query_field: query,
            self.context_field: retrieved_context_str,
        })
        
        
        
            
        

