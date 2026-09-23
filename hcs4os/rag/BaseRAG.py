import dspy
from ..agents.RAG.vector_database import VectorStore
from ..classification_system.registry import get_classification_system
from typing import Literal
import json
    
    
class BaseRAGCoT(dspy.Module):
    
    def __init__(
        self,
        embedding_model_name:str,
        collection_name:str,
        classification_name:Literal["COICOP_2018", "ICATUS_2016"],
        api_key:str,
        model_name:str,
        signature:dspy.Signature,
        query_field:str,
        context_field:str,
        codes_to_include:list[str]|None=None,
        api_base:str|None=None,
        chromadb_path:str="./data/chroma",
        create_new_collection:bool=True,  
        include_hierarchy=False,           
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
        self.include_hierarchy = include_hierarchy


        self.lm = dspy.LM(
            api_key=api_key,
            api_base=api_base,
            model=model_name,
            model_type="chat",
            temperature=0,
            max_tokens=2048,     
            timeout=120,       
            num_retries=3,
        )
        dspy.configure(lm=self.lm, max_errors=10000)

        print("Loading classification system")
        self.classification_name = classification_name
        self.classification_system = get_classification_system(
            classification_name
        )

        print("Creating VectorDB")
        self.store = VectorStore(
            collection_name=collection_name,
            model_name=embedding_model_name,
            chromadb_path=chromadb_path,
        )
        if create_new_collection:
            print("Indexing classification system")
            self.index(self.codes_to_include)

        
        self.CoT = dspy.ChainOfThought(
            signature=self.signature # type: ignore 
        )
      
    
    def index(
        self,
        codes_to_include:list[str]|None
    )->None:
        
        def flatten(c: dict) -> dict:
            return {
                k: (v if isinstance(v, (str, int, float, bool, type(None))) else json.dumps(v, ensure_ascii=False)) # this is sketchy dont know why this is the fix, for chroma raising an error, chroma dont likes when the meta dict itself contains dicts
                # also makes it not possible to later search those dumped categories
                for k, v in c.items()
                if k != "description"
            }
        
        if codes_to_include is None:
            codes = [c.to_dict() for c in self.classification_system.codes]
        else:
            codes = [c.to_dict() for c in self.classification_system.codes if c.code in codes_to_include]
            
        
        ids = [f"id_{c["code"]}" for c in codes]
        documents = [c["description"] for c in codes]
        metadatas = [flatten(c=c) for c in codes]
        
        self.store.create_collection(
            ids=ids,
            documents=documents,
            metadatas=metadatas # type: ignore
        )

        
    def search_category(
        self,
        query:str,
        k:int,
        where:dict|None=None
    ):
        result = self.store.collection.query(
            query_texts=[query],
            n_results=k,
            where=where
        )

        if result is not None:
            
            metas = result["metadatas"][0] 
            docs = result["documents"][0]

            output = []

            for d, m in zip(docs, metas):

                curr_code = {
                    "code":m["code"],
                    "description":d
                }
                if self.include_hierarchy:
                    
                    includes_info = json.loads(m["details"]) if isinstance(m["details"], str) else m["details"]
                    
                    curr_code["includes"] = list(set(includes_info.get("includes",[])))

                output.append(curr_code)

            return output
        else:
            raise Exception("Error occured while searching for code")
    
    def forward(self, query: str, k: int, where: dict|None = None):
        retrieved_context = self.search_category(query, k, where=where)
        retrieved_context_str = json.dumps(retrieved_context, indent=4, ensure_ascii=False)
        return self.CoT(**{
            self.query_field: query,
            self.context_field: retrieved_context_str,
        })
        
        
        
            
        

