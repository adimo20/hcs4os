from abc import ABC, abstractmethod
from dataloaders.BaseDataLoader import ClassificationLoader
from Code import Code



class ClassificationSystem(ABC):

    """
    Central class to organise and retrieve informations from the hierarchical classification system. It receives 
    a list of Code objects as inputs and makes them searchable through it's methods. Additionally it standardises the
    codes to a format, where only letters and numbers are used, indicating that all special characters and spaces will be
    removed because the do not have any semantic meaning inside of the classification systems. 

    **Semantic Meaning of Codes in classifcation systems**

    Hierarchical classification systems are usually structed into certain very generic and general top level division, 
    that devide themselfs into more and more specific sub- and sub-sub-groups. Example from the COICOP:

```markdown
    `01 FOOD AND NON-ALCOHOLIC BEVERAGES` - Level 1
    `011 FOOD` - Level 2
    `0111 Cereals and cereal products (ND)` - LEvel 3
    `01111 Cereals (ND)` - Level 4
```

    This indicates a hierarchical tree-like structure, where:
    * 011 is the `child` of 01 
    * 01 is the `parent` of 011
    And it is also meaning that the parent category contains all the elements that deeper down that hierarchy sharing the same root nodes.
    Usually the condition applied to identify a child-parent relation between code is if "XX" -> "XXY" = Shared root.
    """

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
    def _is_child(
        self,
        parent:str,
        potential_child:str
    )->bool:
        ...
    
    def get_code(
        self,
        code:str
    )->Code:
        """
        Looks up a code inside of the classification system and returns the details in form the custom datatype Code. 
        Applies preprocessing and code normalisation before lookup so we do not miss a code just due to not aligned 
        code formatting. 
        Args: 
            code (str) - e.g. 01111
        """
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
        """
        Collects a list of all child categories for a given parent.
        Parameters:
            parent (str): The code you want to explore the children of (e.g., '01' or '011').
        Returns:
            List of child categories: Code
        """

        try:
            return self._children_register[code]
        except Exception as e:
            print(f"Code {code!r} hat no children.")
            raise ValueError(f"Code {code!r} hat no children. Exeption: {e}")

    @abstractmethod
    def get_prefixes(
        self,
        code:str
    )->list[str]:
        ...

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
                # In case a certain code is not inside the classification system/or a trace cannot be identified, because of missing parent elements
                # the codes that are not inside the system will be skipped silently without breaking the flow. 
                continue
        return trace
           