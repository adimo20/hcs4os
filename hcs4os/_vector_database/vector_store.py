from typing import Literal
import json
import chromadb
from ..classification_system.BaseClassificationSystem import ClassificationSystem
from chromadb.utils.embedding_functions import( 
    SentenceTransformerEmbeddingFunction, 
    OllamaEmbeddingFunction,
    OpenAIEmbeddingFunction
)

EFType = Literal["openai", "ollama", "sentence_transformers"]

_EMEBEDDIG_FUNCTIONS = {
    "openai":OpenAIEmbeddingFunction,
    "ollama":OllamaEmbeddingFunction,
    "sentence_transformers":SentenceTransformerEmbeddingFunction
}

def get_ef(embedding_function_type: EFType, **kwargs):
    try:
        return _EMEBEDDIG_FUNCTIONS[embedding_function_type](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown embedding function: {embedding_function_type!r}") from None

class VectorStore:
    """
    Interface for filling and querying the chromaDB (persistent) client.

    Parameters:
        collection_name (str) - name of the collection to create/load
        model_name (str) - sentence-transformers model used to embed documents
            and queries. Both sides go through the same model, so a collection
            is only meaningful together with the model that built it.
        chromadb_path (str) - path of the persistent client to load or create
    """

    def __init__(
        self, 
        collection_name: str, 
        chromadb_path: str, 
        embedding_function_type:Literal["openai", "ollama", "sentence_transformers"], 
        embedding_batch_size:int=16,
        **embedding_kwargs) -> None:
        
        self.collection_name = collection_name
        self.chromadb_path = chromadb_path
        self.embedding_function_type = embedding_function_type
        self.embedding_batch_size = embedding_batch_size
        self.chroma_client = chromadb.PersistentClient(path=chromadb_path)
        print("Setting up embedding function and collection!")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=get_ef(embedding_function_type=self.embedding_function_type, **embedding_kwargs), # type: ignore
            configuration={"hnsw": {"space": "cosine"}},
        )

    def create_collection(
        self,
        ids:list[str],
        documents:list[str],
        metadatas:list[dict]
    )->None:
        
        batch_size = self.embedding_batch_size
        
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size], # type: ignore
            )
        print("Sucessfully indexed classification system")
        
    def index(
        self,
        codes_to_include:list[str]|None,
        classification_system:ClassificationSystem 
    )->None:
        
        def flatten(c: dict) -> dict:
            return {
                k: (v if isinstance(v, (str, int, float, bool, type(None))) else json.dumps(v, ensure_ascii=False)) # this is sketchy dont know why this is the fix, for chroma raising an error, chroma dont likes when the meta dict itself contains dicts
                # also makes it not possible to later search those dumped categories
                for k, v in c.items()
                if k != "description"
            }
        
        if codes_to_include is None:
            codes = [c.to_dict() for c in classification_system.codes]
        else:
            codes = [c.to_dict() for c in classification_system.codes if c.code in codes_to_include]
            
        
        ids = [f"id_{c["code"]}" for c in codes]
        documents = [c["description"] for c in codes]
        metadatas = [flatten(c=c) for c in codes]
        
        self.create_collection(
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
        result = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where
        )
        if result is not None:
            output = result["metadatas"][0] # type: ignore
            docs = result["documents"][0] # type: ignore
            for o, d in zip(output, docs):
                o["description"] = d # type: ignore
            return output
        else:
            raise Exception("Error occured while searching for code")

