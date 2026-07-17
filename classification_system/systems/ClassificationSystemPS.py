from ..BaseClassificationSystem import ClassificationSystem
from ..registry import register

@register("PS")
class ClassificationSystemPS(ClassificationSystem):

    def _is_child(
        self,
        parent,
        potential_child
    )->bool:
        pc_sub_codes = potential_child.split(".")
        parent_sub_codes = parent.split(".")
        return parent_sub_codes == pc_sub_codes[:len(pc_sub_codes)-1]

    def get_code_trace(
        self,
        code:str
    ):
        sub_codes = code.split(".")
        trace = []
        for i in range(1,len(sub_codes)+1):
            c = ".".join(sub_codes[:i])
            try:
                trace.append((c,self._lookup[c].description))
            except (ValueError, KeyError):
                # In case a certain code is not inside the classification system/or a trace cannot be identified, because of missing parent elements
                # the codes that are not inside the system will be skipped silently without breaking the flow. 
                continue
        return trace

    