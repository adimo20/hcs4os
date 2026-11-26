from abc import ABC, abstractmethod
from typing import Any
from ..classification_system.BaseClassificationSystem import ClassificationSystem

class Retriever(ABC):
    
    """
    Abstract class that integrates with all `rac4os` classifiers that build on retrieval. If you want to use a custom retrieval approach that is not supported 
    by `rac4os` use this class and implement your custom logic here. If the approach does not need to build a custom, vector database just do not implement the
    related methods, just implement the search_category method. This is the only method that is required for the retrieval based classification approaches.
    """
    
    @abstractmethod
    def __init__(self, **kwargs) -> None:
        """
        Set up whatever the backend needs to answer `search_category`.

        Nothing in the package calls this with a fixed argument list, so an implementation is
        free to take whatever its backend needs: a path and a collection name for a local
        store, a URL and a key for a hosted search service.

        Parameters:
            **kwargs (Any): whatever your implementation requires.
        """
        ...
    
    @abstractmethod
    def create_collection(self,
        ids:list[str],
        documents:list[str],
        metadatas:list[dict]
    )->None:
        """
        Write a batch of documents into the backend.

        Only needed when your retriever builds its own index. A wrapper around an existing
        search service can leave this unimplemented.

        Parameters:
            ids (list[str]): unique id per document.
            documents (list[str]): the texts that get embedded and searched over.
            metadatas (list[dict]): one metadata dict per document, carrying everything about the code that is not the searched text.
        """
        ...
        
    @abstractmethod
    def index(
        self,
        classification_system:ClassificationSystem,
        codes_to_include:list[str]|None=None,   
        **kwargs       
    )->None:
        """
        Fill the retriever from a classification system.

        Only needed when your retriever builds its own index. This is the method that decides
        what a category's searchable text is: the choice of what to embed, and what to keep
        alongside as metadata, is the single biggest lever on retrieval quality.

        Parameters:
            classification_system (ClassificationSystem): the taxonomy to index.
            codes_to_include (list[str]|None): restrict indexing to these codes; None indexes every code in the system.
            **kwargs (Any): implementation-specific options.
        """
        ...
        
    @abstractmethod
    def search_category(
        self,
        query:str,
        k:int,
        where:dict|None=None
    )->Any:
        """
        Return the k categories closest to a query.

        This is the only method the retrieval-based classifiers actually call, so it is the
        only one you have to implement. Each returned candidate is serialised into the
        generator's prompt as it comes back from here, which means the shape you return is the
        shape the model reasons over.

        Parameters:
            query (str): the text to search for.
            k (int): how many candidates to return.
            where (dict|None): optional metadata filter, e.g. restricting results to one level or one branch.

        Returns:
            Any: k candidates, each carrying at least the code and the text that was matched.
        """
        ...
        