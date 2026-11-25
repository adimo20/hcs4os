from abc import ABC, abstractmethod
from typing import Any
from ..classification_system.BaseClassificationSystem import ClassificationSystem

class Retriever(ABC):
    
    @abstractmethod
    def __init__(self, **kwargs) -> None:
        ...
    
    @abstractmethod
    def create_collection(self,
        ids:list[str],
        documents:list[str],
        metadatas:list[dict]
    ):
        ...
        
    @abstractmethod
    def index(
        self,
        classification_system:ClassificationSystem,
        codes_to_include:list[str]|None=None,   
        **kwargs       
    ):
        ...
        
    @abstractmethod
    def search_category(
        self,
        query:str,
        k:int,
        where:dict|None=None
    )->Any:
        ...
        