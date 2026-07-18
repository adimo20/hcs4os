from ..BaseClassificationSystem import ClassificationSystem
from ..registry import register
import re

@register("KLDB")
class ClassificationSystemKldB(ClassificationSystem):
    
    def get_prefixes(self, code: str) -> list[str]:
        code = code.replace(" ", "")
        return [code[:i] for i in range(1,len(code)+1)]
   

@register("SEA_NS")
class ClassificationSystemSEA_NS(ClassificationSystem):

    def get_prefixes(self, code: str):
        return [code[:i] for i in range(2,len(code)+1)]
    
@register("WZ")
class ClassificationSystemWZ(ClassificationSystem):

    def get_prefixes(self, code: str) -> list[str]:
        prefixes = []
        for i in range(2, len(code)+1):
            curr = re.sub(r"\.$", "", code[:i])
            if curr not in prefixes:
                prefixes.append(curr)           
                       
        return prefixes

@register("SEA")
class ClassificationSystemSEA(ClassificationSystem):

    def get_prefixes(self, code: str) -> list[str]:
        
        code = code.replace(" ", "")
        prefixes = []
        for i in range(2,len(code)+1):
            if i <= 4:
                prefixes.append(code[:i])
            else: 
                prefixes.append(code[:4] + " " + code[4:i])
        return prefixes
    