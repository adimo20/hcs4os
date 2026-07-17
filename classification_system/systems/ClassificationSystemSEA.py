from ..BaseClassificationSystem import ClassificationSystem
from ..registry import register

@register("SEA")
class ClassificationSystemSEA(ClassificationSystem):

    def _is_child(
        self, 
        parent:str, 
        potential_child:str
    )->bool:
        parent = parent.replace(" ", "")
        potential_child = potential_child.replace(" ", "")
        return parent == potential_child[:len(potential_child)-1]

    def get_prefixes(self, code: str) -> list[str]:
        code = code.replace(" ", "")
        prefixes = []
        for i in range(2,len(code)+1):
            if i <= 4:
                prefixes.append(code[:i])
            else: 
                prefixes.append(code[:4] + " " + code[4:i])
        return prefixes
    