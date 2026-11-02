from typing import Literal

import chromadb
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
        **embedding_kwargs) -> None:
        self.collection_name = collection_name
        self.chromadb_path = chromadb_path
        self.embedding_function_type = embedding_function_type
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
        metadatas:list[dict],
        batch_size:int=32
    )->None:
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size], # type: ignore
            )
        print("Sucessfully indexed classification system")