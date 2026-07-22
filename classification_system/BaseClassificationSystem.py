from abc import ABC, abstractmethod
from dataloaders.BaseDataLoader import ClassificationLoader
from Code import Code



class ClassificationSystem(ABC):

    def __init__(
        self,
        loader:ClassificationLoader
    )->None:

        self.loader = loader
        self.codes = self.loader.load()
        self._children_register = {}
        self.__initialise()

    def __initialise(
        self
    )->None:

        self._lookup: dict[str, Code] = {c.code: c for c in self.codes}
        for parent in self.codes:
            self._children_register[parent.code] = [
                code for code in self.codes 
                if self._is_child(parent.code, code.code)
            ]

    @abstractmethod
    def get_prefixes(
        self,
        code:str
    )->list[str]:
        ...
        
    
    @abstractmethod
    def get_root_categories(
        self,        
    )->list[dict]:
        ...

    def _is_child(
        self,
        parent:str,
        potential_child:str
    )->bool:
        potential_child_prefixes = self.get_prefixes(potential_child)
        num_prefixes = len(potential_child_prefixes)
        if num_prefixes < 2: return False
        return parent == potential_child_prefixes[num_prefixes-2]        
        
    
    def get_code(
        self,
        code:str
    )->Code:

        try:
            return self._lookup[code]
        except KeyError as e:
            raise ValueError(
                f"Code {code!r} is not in the classification system. "
                f"Check if your formatting was correct."
            ) from e
    
    def get_children(
        self,
        code:str
    )->list[Code]:

        try:
            return self._children_register[code]
        except Exception as e:
            print(f"Code {code!r} hat no children.")
            raise ValueError(f"Code {code!r} hat no children. Exeption: {e}")

    def get_parent(
        self, 
        parent:str
    ):
        prefixes = self.get_prefixes(parent)
        num_prefixes = len(prefixes)
        if num_prefixes > 1:
            return self.get_code(
                code=prefixes[num_prefixes-2]
            ).to_dict()
        

    def get_code_trace(
        self,
        code:str
    ):
        prefixes = self.get_prefixes(code=code)
        trace = []
        for prefix in prefixes:
            try:
                trace.append((prefix,self._lookup[prefix].description))
            except (ValueError, KeyError):
                continue
        return trace
           