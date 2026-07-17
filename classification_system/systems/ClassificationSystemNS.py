from ..BaseClassificationSystem import ClassificationSystem
from ..registry import register

@register("SEA_NS")
class ClassificationSystemSEA_NS(ClassificationSystem):

    def _is_child(
        self,
        parent,
        potential_child
    )->bool:
        return parent == potential_child[:len(potential_child)-1]

    def get_code_trace(
        self,
        code:str
    ):
        
        trace = []
        for i in range(2,len(code)+1):            
          
            try:
                c = code[:i]
                trace.append((c,self._lookup[c].description))
            except (ValueError, KeyError):
                # In case a certain code is not inside the classification system/or a trace cannot be identified, because of missing parent elements
                # the codes that are not inside the system will be skipped silently without breaking the flow. 
                continue
        return trace

    