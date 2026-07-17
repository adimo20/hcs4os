from ..BaseClassificationSystem import ClassificationSystem
from ..registry import register

@register("SEA")
class ClassificationSystemSEA(ClassificationSystem):

    def _is_child(
        self,
        parent,
        potential_child
    )->bool:
        parent = parent.replace(" ", "")
        potential_child = potential_child.replace(" ", "")
        return parent == potential_child[:len(potential_child)-1]

    def get_code_trace(
        self,
        code:str
    ):
        code = code.replace(" ", "")

        trace = []
        for i in range(2,len(code)+1):
            
            if i <= 4:
                c = code[:i]
            else: 
                c = code[:4] + " " + code[4:i]
            
            try:
                trace.append((c,self._lookup[c].description))
            except (ValueError, KeyError):
                # In case a certain code is not inside the classification system/or a trace cannot be identified, because of missing parent elements
                # the codes that are not inside the system will be skipped silently without breaking the flow. 
                continue
        return trace

    