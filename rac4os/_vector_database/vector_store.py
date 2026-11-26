from typing import Any, Literal
import json
import chromadb
from .Retriever import Retriever
from ..classification_system.BaseClassificationSystem import ClassificationSystem
from chromadb.api.types import EmbeddingFunction
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

_EXTRA = {"sentence_transformers": "local", "ollama": "ollama", "openai": "openai"}

def get_ef(embedding_function_type: EFType, **kwargs) -> EmbeddingFunction:
    """
    Takes an embedding function type, maps the embedding function type 
    to the correct interface and returns the loaded ef interface, given the 
    defined kwargs.
    
    Paramters:
        embedding_function_type (Literal["openai", "ollama", "sentence_transformers"]): tpye of embedding function to use.
        **kwargs (Any): arguments used to specify what model and paramters to use when loading the embedding function. Must correspond to what the chromaDB embedding function takes as input
    
    Returns:
        embedding function

    """
    try:
        ef_cls = _EMEBEDDIG_FUNCTIONS[embedding_function_type]
    except KeyError:
        raise ValueError(f"Unknown embedding function: {embedding_function_type!r}") from None
    try:
        return ef_cls(**kwargs)
    except (ImportError, ValueError) as e:
        if "not installed" in str(e) or isinstance(e, ImportError):
            raise ImportError(
                f"If you want to use the {embedding_function_type} please install rac4os[{_EXTRA[embedding_function_type]}]"
                f'pip install "rac4os[{_EXTRA[embedding_function_type]}]"'
            ) from e
        raise

class VectorStore(Retriever):
    """
    Interface for filling and querying the chromaDB (persistent) client.
    Builds a chromaDB collection from a predefined ClassificationSystem

    Parameters:
        collection_name (str): name of the collection to create/load
        chromadb_path (str): path of the persistent client to load or create
        embedding_function_type (str): which backend embeds documents and
            queries - "openai", "ollama" or "sentence_transformers".
        embedding_batch_size (int): how many documents go per `collection.add`
            call. Raise it for a local model, keep it low for a remote API.
        **embedding_kwargs (Any): forwarded untouched to the chosen embedding
            function, e.g. `model_name`, `url`, `api_key`. Both sides of a
            collection go through the same model, so a collection is only
            meaningful together with the model that built it.
    """

    def __init__(
        self, 
        collection_name: str, 
        chromadb_path: str, 
        embedding_function_type:Literal["openai", "ollama", "sentence_transformers"], 
        embedding_batch_size:int=16,
        **embedding_kwargs) -> None:
        
        """
        Opens the persistent chromaDB client and gets or creates the collection.

        The embedding function is built here from `embedding_function_type` and
        `embedding_kwargs` and attached to the collection, so queries are embedded with the
        same model that indexed it. The collection is created with cosine distance.

        Nothing is indexed by constructing a store: an existing collection is reused as it
        stands, and a new one starts empty until `index` is called. This is why pointing at the
        wrong `collection_name` shows up as an empty candidate list rather than as an error.
        """
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
        
        """
        Fills the chromaDB clients collection with the contents, that will be searchable afterwards.
        
        Paramters:
            ids (list[str]): ids of the given tuple
            documents (list[str]): documents to be indexed/vectorised
            metadatas (list[dict[str, Any]]): metadata of the documents to be inserted into the db
        """
        
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
        classification_system:ClassificationSystem,
        codes_to_include:list[str]|None=None,
        **kwargs
    )->None:
        
        """
        Interface for indexing a `ClassificationSystem`. Takes a `ClassificationSystem` object and inserts it into 
        a chromaDB collection of the persistent client. All the codes of the given classification system will be index, 
        based on their description, which will serve as documents, all other information associated with code, will be
        stored in the metadata of the tuples. If you only want to index a certain subset of the classification system use
        codes_to_include, and insert a list of codes to index. One thing is important to notice at this point. Because chromaDB
        does not allow, metadata entries themselves to be dictionaries, all dicts will be dumped to a json-string.
        
        Parameters:
            classification_system (ClassificationSystem): classification system to be index in the vector store
            codes_to_include (list[str]|None): list of codes to be indexed, when this paramter is left blank all entries of the classification system will indexed.
            **kwargs (Any): additional key word args, at this points they will be ignored.
        
        Returns:
            None
            
         
        """
        
        def flatten(c: dict) -> dict:
            """
            Reduce one code record to chromaDB-storable metadata.

            chromaDB rejects metadata values that are themselves dicts, so anything that is not a
            scalar is dumped to a JSON string. The side effect is that the contents of those dumped
            fields cannot be filtered on afterwards with `where`. `description` is left out because
            it is indexed as the document rather than as metadata, and `search_category` puts it
            back on the result.

            Parameters:
                c (dict): a `Code` record as a dict.

            Returns:
                dict: the same record with `description` removed and every non-scalar value JSON-encoded.
            """
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
        where:dict[str, Any]|None=None
    )->list[dict[str, Any]]:
        
        """
        Function for searching the indexed chromaDB collection of the classification system.
        
        Paramters:
            query (str): query to search the collection
            k (int): number of DB entries to be returned
            where (dict[str,Any]|None): optional metadata filter which can be used to set contraints to the returned results. See chromaDB documentation for references.
        
        Returns:
            list of k codes that are the result of the similiarity search.
        
        """
        
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

