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

    def __get_prefixes(self, code: str):
        return [code[:i] for i in range(2,len(code)+1)]
    


    