import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


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

    def __init__(self, collection_name: str, model_name: str, chromadb_path: str) -> None:
        self.collection_name = collection_name
        self.model_name = model_name
        self.chromadb_path = chromadb_path
        self.chroma_client = chromadb.PersistentClient(path=chromadb_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=SentenceTransformerEmbeddingFunction(model_name), # type: ignore
            configuration={"hnsw": {"space": "cosine"}},
        )

    def create_collection(
        self,
        ids:list[str],
        documents:list[str],
        metadatas:list[dict],
        batch_size:int=512
    )->None:
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size], # type: ignore
            )
        print("Sucessfully indexed classification system")