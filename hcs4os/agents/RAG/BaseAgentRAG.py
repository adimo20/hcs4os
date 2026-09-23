import dspy
from ...classification_system import get_classification_system
from .vector_database import VectorStore
import json
dspy.settings.configure(adapter=dspy.ChatAdapter())
class BaseRAGAgent(dspy.Module):
    
    def __init__(
        self,
        embedding_model_name:str,
        collection_name:str,
        classification_name:str,
        api_key:str,
        model_name:str,
        signature:dspy.Signature,
        search_category_tool_doc:str,
        codes_to_include:list[str]|None=None,
        api_base:str|None=None,
        chromadb_path:str="./data/chroma",
        create_new_collection:bool=True, 
        include_hierarchy=False            
    )->None:

        assert api_key is not None
        assert model_name is not None
        assert classification_name is not None
        assert signature is not None
        
        self.signature = signature
        self.search_category_tool_doc = search_category_tool_doc
        self.codes_to_include = codes_to_include
        self.include_hierarchy = include_hierarchy

        self.lm = dspy.LM(
            api_key=api_key,
            api_base=api_base,
            model=model_name,
            model_type="chat",
            max_tokens=2048,   
            temperature=0,  
            timeout=120,       
            num_retries=3
        )
        dspy.configure(
            lm=self.lm,
            adapter=dspy.ChatAdapter(), 
            provide_traceback=False,# Turn on for debugging when using quanitzed models, there are frequently appearing errors, due to the json scheme,
            disable_history=False,
            max_errors=10000,
            max_iters=5
        )

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
            self.index(codes_to_include=self.codes_to_include)

        self.signature = self.signature
        
        self.agent = dspy.ReAct(
            self.signature,  # type: ignore
            tools=[
                dspy.Tool(
                    self.search_category,
                    name="search_category",
                    desc= self.search_category_tool_doc
                )
            ],
            max_iters=5
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
        
        
        
            
        

