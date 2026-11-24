from ..BaseClassificationSystem import ClassificationSystem
from ..registry import register
import re

@register("NAF_rev2")
class ClassificationSystemNAF(ClassificationSystem):

    def get_root_categories(self) -> list[dict]:
        return [
            {"code":c.code, "description":c.description} for c in self.codes if len(c.code) == 2
        ]

    def get_prefixes(self, code: str) -> list[str]:

        code = code.replace(".", "")
        prefixes = []
        for i in range(2,len(code)+1):
            if i <= 2:
                prefixes.append(code[:i])
            else:
                prefixes.append(code[:2]+"."+code[2:i])
        
        return prefixes
        
        


@register("ICATUS")
class ClassificationSystemICATUS(ClassificationSystem):

    def get_root_categories(self) -> list[dict]:
        return [
            {"code":c.code, "description":c.description} for c in self.codes if len(c.code) == 1
        ]

    def get_prefixes(self, code:str)->list[str]:
        return [code[:i] for i in range(1, len(code)+1)]



@register("COICOP")
class ClassificationSystemCOICOP(ClassificationSystem):

    def get_root_categories(self) -> list[dict]:
        return [
            {"code":c.code, "description":c.description} for c in self.codes if len(c.code) == 2
        ]

    def get_prefixes(self, code: str) -> list[str]:        
        codes = code.split(".")
        return [".".join(codes[:i]) for i in range(1,len(codes)+1)]


@register("KLDB")
class ClassificationSystemKldB(ClassificationSystem):

    def get_root_categories(self) -> list[dict]:
        return [
            {"code":c.code, "description":c.description} for c in self.codes if len(c.code) == 1
        ]

    def get_prefixes(self, code: str) -> list[str]:
        code = code.replace(" ", "")
        return [code[:i] for i in range(1,len(code)+1)]
   
    
@register("WZ")
class ClassificationSystemWZ(ClassificationSystem):
    
    def get_root_categories(self) -> list[dict]:
        """This is generelly not right because its A,B,C and so on but for the moment it should be fine"""
        return [
            {"code":c.code, "description":c.description} for c in self.codes if len(c.code) == 2
        ]


    def get_prefixes(self, code: str) -> list[str]:
        prefixes = []
        for i in range(2, len(code)+1):
            curr = re.sub(r"\.$", "", code[:i])
            if curr not in prefixes:
                prefixes.append(curr)           
                       
        return prefixes

@register("SEA")
class ClassificationSystemSEA(ClassificationSystem):
    
    def get_root_categories(self) -> list[dict]:
        return [
            {"code":c.code, "description":c.description} for c in self.codes if len(c.code) == 2
        ]
    
    def get_prefixes(self, code: str) -> list[str]:
        
        code = code.replace(" ", "")
        prefixes = []
        for i in range(2,len(code)+1):
            if i <= 4:
                prefixes.append(code[:i])
            else: 
                prefixes.append(code[:4] + " " + code[4:i])
        return prefixes
    
