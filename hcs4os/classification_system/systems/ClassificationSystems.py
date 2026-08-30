from ..BaseClassificationSystem import ClassificationSystem
from ..registry import register
import re

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
   

@register("SEA_NS")
class ClassificationSystemSEA_NS(ClassificationSystem):

    def get_root_categories(self) -> list[dict]:
        return [
            {"code":c.code, "description":c.description} for c in self.codes if len(c.code) == 2
        ]

    
    def get_prefixes(self, code: str):
        return [code[:i] for i in range(2,len(code)+1)]
    
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
    
    
@register("EAV")
class ClassificationSystemEAV(ClassificationSystem):
    
    def get_root_categories(self) -> list[dict]:
        return [
            {
                "code":c.code,
                "description":c.description
                } for c in self.codes if len(c.code) == 2
        ]
    
    def get_prefixes(self, code: str) -> list[str]:
                        
        return [code[:2], code[:4], code]


@register("VUL")
class ClassificationSystemVUL(ClassificationSystem):
    
    def get_root_categories(self) -> list[dict]:
        return [
            {
                "code":c.code,
                "description":c.description
                } for c in self.codes if len(c.code) == 1
        ]
    
    def get_prefixes(self, code: str) -> list[str]:
                        
        return [code[:i] for i in range(1,len(code)+1)]


@register("NST")
class ClassificationSystemNST(ClassificationSystem):
    
    def get_root_categories(self) -> list[dict]:
        return [
            {
                "code":c.code,
                "description":c.description
                } for c in self.codes if len(c.code) == 2
        ]
    
    def get_prefixes(self, code: str) -> list[str]:
        if len(code) > 2:   
            return [code[:2], code[:3]]
        elif len(code) == 2:
            return [code]
        else:
            raise ValueError(f"The code {code} is not in the classification system")
        

@register("GP")
class ClassificationSystemGP(ClassificationSystem):
    
    def get_root_categories(self) -> list[dict]:
        return [
            {
                "code":c.code,
                "description":c.description
                } for c in self.codes if len(c.code) == 2
        ]
    
    def get_prefixes(self, code: str) -> list[str]:
        n = len(code)
        if n == 4:
            prefixes_4 = [code[:i] for i in [2,3,4]]
            return prefixes_4
        elif n <= 7:
            prefixes_4 = [code[:i] for i in [2,3,4]]
            prefixes_5_7 = [code[:i] for i in [6,7] if i <= n ]
            return prefixes_4 + prefixes_5_7
        else:
            prefixes_4 = [code[:i] for i in [2,3,4]]
            prefixes_5_7 = [code[:i] for i in [6,7] if i <= n ]
            prefixes_9_11 = [code[:i] for i in [9,10,11] if i <= n ]
            return prefixes_4 + prefixes_5_7 + prefixes_9_11