from inspect import signature
from typing import Literal
import dspy
from ...classification_system import get_classification_system
from .vector_database import VectorStore

class RAGAgent(dspy.Module):
    
    def __init__(
        self,
        embedding_model_name:str,
        collection_name:str,
        classification_name:Literal["COICOP", "ICATUS"],
        api_key:str,
        model_name:str,
        api_base:str|None=None,
        chromadb_path:str="./data/chroma",
        create_new_collection:bool=True,             
    )->None:

        assert api_key is not None
        assert model_name is not None
        assert classification_name is not None

        self.lm = dspy.LM(
            api_key=api_key,
            api_base=api_base,
            model=model_name
        )
        dspy.configure(lm=self.lm)

        self.classification_name = classification_name
        self.classification_system = get_classification_system(
            classification_name
        )

        self.store = VectorStore(
            collection_name=collection_name,
            model_name=embedding_model_name,
            chromadb_path=chromadb_path,
        )
        if create_new_collection:
            self.index()
        
        self.signature.__doc__ = ""
        
        self.agent = dspy.ReAct(
             signature=signature,
             tools=[dspy.Tool(
                 self.search_category,
                 desc="pass"
             )]
         )
        
    def index(
        self
    )->None:
        
        def flatten(c: dict) -> dict:
            meta = {}
            for k, v in c.items():
                if k == "description":
                    continue
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        meta[f"{k}_{sub_k}"] = sub_v
                else:
                    meta[k] = v
            return meta
        
        codes = [c.to_dict() for c in self.classification_system.codes]
        
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
        k:int
    ):
        result = self.store.collection.query(
            query_texts=[query],
            n_results=k
        )
        if result is not None:
            output = result["metadatas"][0] # type: ignore
            docs = result["documents"][0] # type: ignore
            for o, d in zip(output, docs):
                o["description"] = d # type: ignore
            return output
        else:
            raise Exception("Error occured while searching for code")
        
        
        
            
        

