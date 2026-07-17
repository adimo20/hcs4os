from ..BaseClassificationSystem import ClassificationSystem
from ..registry import register
# Bug here --> 
@register("PS")
class ClassificationSystemPS(ClassificationSystem):

    def _is_child(
        self,
        parent:str,
        potential_child:str
    )->bool:
        pc_sub_codes = potential_child.split(".")
        parent_sub_codes = parent.split(".")
        return parent_sub_codes == pc_sub_codes[:len(pc_sub_codes)-1]

    def get_prefixes(self, code: str) -> list[str]:
        sub_codes = code.split(".")
        return [".".join(sub_codes[:i]) for i in range(1,len(sub_codes)+1)]
    


    