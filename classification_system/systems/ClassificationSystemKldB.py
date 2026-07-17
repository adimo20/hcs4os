from ..BaseClassificationSystem import ClassificationSystem
from ..registry import register

@register("KLDB")
class ClassificationSystemKldB(ClassificationSystem):

    def _is_child(
        self,
        parent,
        potential_child
    )->bool:
        return parent == potential_child[:len(potential_child)-1]
    
    def get_prefixes(self, code: str) -> list[str]:
        code = code.replace(" ", "")
        return [code[:i] for i in range(2,len(code)+1)]
   

