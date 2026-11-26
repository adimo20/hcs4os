from abc import ABC, abstractmethod
from ..dataloaders.BaseDataLoader import ClassificationLoader
from .._shared.Code import Code



class ClassificationSystem(ABC):

    """
    The uniform interface every `rac4os` classifier is typed against. A classification
    system wraps a loaded taxonomy and answers the structural questions the classifiers
    ask of it: what the first level is, what a code means, and what sits below or above it.

    Subclasses implement `get_prefixes` and `get_root_categories` and nothing else. The
    lookup, the parent/child register, `get_code`, `get_children`, `get_parent` and
    `get_code_trace` are all derived from those two, so describing your own taxonomy comes
    down to answering two questions about how its codes are written.

    Parameters:
        loader (ClassificationLoader): loader that yields the `Code` records this system is built from.

    Attributes:
        loader (ClassificationLoader): the loader that was passed in.
        codes (list[Code]): every record the loader returned.
    """
    def __init__(
        self,
        loader:ClassificationLoader
    )->None:

        """
        Loads the taxonomy through the loader and builds the internal registers.

        Both happen eagerly, so constructing a classification system reads and parses the
        source file.
        """
        self.loader = loader
        self.codes = self.loader.load()
        self.__initialise()

    def __initialise(
        self
    )->None:

        """
        Builds the two registers the derived accessors rely on.

        `_lookup` maps a code to its `Code` record for direct access, and
        `_children_register` maps a code to the records directly below it. The second is
        filled by asking `get_prefixes` for each code's second-to-last prefix, which is that
        code's parent.
        """
        self._lookup: dict[str, Code] = {c.code: c for c in self.codes}
        self._children_register:dict[str, list[Code]] = {c.code: [] for c in self.codes}
        
        for child in self.codes:
            
            prefixes = self.get_prefixes(code=child.code)
            num_prefixes = len(prefixes)
            
            if (
                num_prefixes >= 2 and
                prefixes[num_prefixes-2] in self._children_register
            ):
                self._children_register[prefixes[num_prefixes-2]].append(child)

    @abstractmethod
    def get_prefixes(
        self,
        code:str
    )->list[str]:
        """
        Return every prefix of a code, from the broadest level down to the code itself.

        This is the one method that encodes how your taxonomy writes its codes: separators,
        fixed-width levels, padding. The last element is the code itself and the
        second-to-last is its parent, which is what the parent/child register is built from.

        Parameters:
            code (str): a code of this classification system.

        Returns:
            list[str]: the code's prefixes, broadest first, ending with the code itself.
        """
        ...
    
    
    @abstractmethod
    def get_root_categories(
        self,        
    )->list[dict]:
        """
        Return the first level of the taxonomy.

        The hierarchy traversal agent starts here and descends, so this is the entry point
        into your classification. Each entry carries at least `code` and `description`.

        Returns:
            list[dict]: the categories at the first level of the hierarchy.
        """
        ...

    def _is_child(
        self,
        parent:str,
        potential_child:str
    )->bool:
        """
        Whether one code sits directly below another.

        Compares the parent against the second-to-last prefix of the candidate, so only
        direct children count and a grandchild returns False.

        Parameters:
            parent (str): the code to test against.
            potential_child (str): the code that may sit directly below it.

        Returns:
            bool: True if `potential_child` is a direct child of `parent`.
        """
        potential_child_prefixes = self.get_prefixes(potential_child)
        num_prefixes = len(potential_child_prefixes)
        if num_prefixes < 2: return False
        return parent == potential_child_prefixes[num_prefixes-2]        
      
    def add_code(
        self, 
        code:dict,
        parent:str|None=None
    ):
        
        """
        Add a single code to the live system without rebuilding it.

        Useful for patching in a category the source documentation omits. Only the lookup and
        the parent's child list are touched; `self.codes` stays as the loader returned it, so
        the addition does not survive into a system built freshly from the same source.

        Parameters:
            code (dict): a `Code`-shaped dict; keys that `Code` does not define are dropped.
            parent (str|None): code to file the new entry under. Left out, the code is reachable through `get_code` but appears in no `get_children` result.
        """
        _code = Code.from_dict(code)
        self._lookup[_code.code] = _code
        if parent:
            if parent in self._children_register.keys(): 
                self._children_register[parent].append(_code)
            else: 
                self._children_register[parent] =[_code]
        
    
    def get_code(
        self,
        code:str
    )->Code:

        """
        Look up the full record for a code.

        Parameters:
            code (str): the code to look up, written the way the classification system writes it.

        Returns:
            Code: the record for that code.

        Raises:
            ValueError: if the code is not part of this classification system. This is far more often a formatting mismatch, such as a missing separator, than a genuinely absent category.
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
        Return the categories directly below a code.

        Parameters:
            code (str): the parent code.

        Returns:
            list[Code]: the direct children; an empty list for a leaf or for a code the system does not know.
        """
        try:
            return self._children_register.get(code, [])
        except Exception as e:
            print(f"Code {code!r} hat no children.")
            raise ValueError(f"Code {code!r} hat no children. Exeption: {e}")

    def get_parent(
        self, 
        parent:str
    )->Code|None:
        
        """
        Return the category directly above a code.

        Parameters:
            parent (str): the code whose parent is wanted.

        Returns:
            Code|None: the parent record, or None if the code is already at the first level.
        """
        prefixes = self.get_prefixes(parent)
        num_prefixes = len(prefixes)
        if num_prefixes > 1:
            return self.get_code(
                code=prefixes[num_prefixes-2]
            )
        

    def get_code_trace(
        self,
        code:str
    )->list[tuple[str, str]]:
        """
        Return the path from the first level down to a code.

        Each step is a `(code, description)` pair, so the trace reads as the chain of
        categories a human coder would have walked through. Prefixes that are not themselves
        categories of the system are skipped rather than raising, which keeps the trace usable
        for taxonomies whose levels are not all populated.

        Parameters:
            code (str): the code whose path is wanted.

        Returns:
            list[tuple[str, str]]: `(code, description)` pairs, broadest first, ending with the code itself.
        """
        prefixes = self.get_prefixes(code=code)
        trace = []
        for prefix in prefixes:
            try:
                trace.append((prefix,self._lookup[prefix].description))
            except (ValueError, KeyError):
                continue
        return trace
           