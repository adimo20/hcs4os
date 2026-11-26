from ..BaseClassificationSystem import ClassificationSystem
from ..registry import register
import re

@register("NAF_rev2")
class ClassificationSystemNAF(ClassificationSystem):

    """
    Classification system for the French NAF rév. 2, registered under "NAF_rev2".

    NAF codes are written as a two-digit division followed by dot-separated digits, so
    `01` sits above `01.1`, which sits above `01.11`.
    """
    def get_root_categories(self) -> list[dict]:
        """
        Return the first level of the hierarchy.

        Returns:
            list[dict]: one `code`/`description` dict per two-digit division.
        """
        return [
            {"code":c.code, "description":c.description} for c in self.codes if len(c.code) == 2
        ]

    def get_prefixes(self, code: str) -> list[str]:

        """
        Return the prefixes of a NAF code, broadest first.

        The dot is stripped before the prefixes are cut and re-inserted after the second
        digit, so `0111` and `01.11` are treated as the same code and either spelling can be
        passed in.

        Parameters:
            code (str): a NAF code.

        Returns:
            list[str]: the code's prefixes, ending with the code itself.
        """
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

    """
    Classification system for ICATUS 2016, registered under "ICATUS".

    ICATUS codes are plain digit strings whose length is their level, so each additional
    digit descends one level: `1` above `11` above `111`.
    """
    def get_root_categories(self) -> list[dict]:
        """
        Return the first level of the hierarchy.

        Returns:
            list[dict]: one `code`/`description` dict per single-digit major division.
        """
        return [
            {"code":c.code, "description":c.description} for c in self.codes if len(c.code) == 1
        ]

    def get_prefixes(self, code:str)->list[str]:
        """
        Return the prefixes of a ICATUS code, broadest first.

        One prefix per leading digit, since a code's length is its level.

        Parameters:
            code (str): a ICATUS code.

        Returns:
            list[str]: the code's prefixes, ending with the code itself.
        """
        return [code[:i] for i in range(1, len(code)+1)]



@register("COICOP")
class ClassificationSystemCOICOP(ClassificationSystem):

    """
    Classification system for COICOP 2018, registered under "COICOP".

    COICOP codes are dot-separated groups of digits and the number of groups is the level,
    so `01` sits above `01.1`, which sits above `01.1.1`.
    """
    def get_root_categories(self) -> list[dict]:
        """
        Return the first level of the hierarchy.

        Returns:
            list[dict]: one `code`/`description` dict per two-digit division.
        """
        return [
            {"code":c.code, "description":c.description} for c in self.codes if len(c.code) == 2
        ]

    def get_prefixes(self, code: str) -> list[str]:        
        """
        Return the prefixes of a COICOP code, broadest first.

        The code is split on the dot and rejoined one group at a time, so the separator is
        preserved in every prefix.

        Parameters:
            code (str): a COICOP code.

        Returns:
            list[str]: the code's prefixes, ending with the code itself.
        """
        codes = code.split(".")
        return [".".join(codes[:i]) for i in range(1,len(codes)+1)]


@register("KLDB")
class ClassificationSystemKldB(ClassificationSystem):

    """
    Classification system for the KldB 2010, registered under "KLDB".

    KldB codes are digit strings whose length is their level. Published codes are often
    padded with spaces to a fixed width; the spaces are stripped before the hierarchy is
    derived.
    """
    def get_root_categories(self) -> list[dict]:
        """
        Return the first level of the hierarchy.

        Returns:
            list[dict]: one `code`/`description` dict per single-digit Berufsbereich.
        """
        return [
            {"code":c.code, "description":c.description} for c in self.codes if len(c.code) == 1
        ]

    def get_prefixes(self, code: str) -> list[str]:
        """
        Return the prefixes of a KldB code, broadest first.

        Spaces are removed first, so a code padded to the full width yields the same prefixes
        as its unpadded form.

        Parameters:
            code (str): a KldB code.

        Returns:
            list[str]: the code's prefixes, ending with the code itself.
        """
        code = code.replace(" ", "")
        return [code[:i] for i in range(1,len(code)+1)]
   
    
@register("WZ")
class ClassificationSystemWZ(ClassificationSystem):
    
    """
    Classification system for the WZ 2008, registered under "WZ".

    WZ codes are two-digit divisions with dot-separated digits below them. Trailing dots
    that appear in the published notation are dropped, so a prefix is never left ending in
    a separator.
    """
    def get_root_categories(self) -> list[dict]:
        """This is generelly not right because its A,B,C and so on but for the moment it should be fine"""
        return [
            {"code":c.code, "description":c.description} for c in self.codes if len(c.code) == 2
        ]


    def get_prefixes(self, code: str) -> list[str]:
        """
        Return the prefixes of a WZ code, broadest first.

        Each prefix has a trailing dot stripped, and duplicates that arise from doing so are
        dropped, so a code and the prefix that differs from it only by a separator do not both
        appear.

        Parameters:
            code (str): a WZ code.

        Returns:
            list[str]: the code's prefixes, ending with the code itself.
        """
        prefixes = []
        for i in range(2, len(code)+1):
            curr = re.sub(r"\.$", "", code[:i])
            if curr not in prefixes:
                prefixes.append(curr)           
                       
        return prefixes

@register("SEA")
class ClassificationSystemSEA(ClassificationSystem):
    
    """
    Classification system for the SEA 2021, registered under "SEA".

    SEA codes follow COICOP down to four digits and then continue after a space, so `01`
    sits above `0111`, which sits above `0111 100`. Spaces are stripped before the
    hierarchy is derived and re-inserted after the fourth digit.
    """
    def get_root_categories(self) -> list[dict]:
        """
        Return the first level of the hierarchy.

        Returns:
            list[dict]: one `code`/`description` dict per two-digit division.
        """
        return [
            {"code":c.code, "description":c.description} for c in self.codes if len(c.code) == 2
        ]
    
    def get_prefixes(self, code: str) -> list[str]:
        
        """
        Return the prefixes of a SEA code, broadest first.

        Spaces are removed first; prefixes up to four digits are cut directly, and longer ones
        get a space re-inserted after the fourth digit, which is the notation the published
        SEA uses.

        Parameters:
            code (str): a SEA code.

        Returns:
            list[str]: the code's prefixes, ending with the code itself.
        """
        code = code.replace(" ", "")
        prefixes = []
        for i in range(2,len(code)+1):
            if i <= 4:
                prefixes.append(code[:i])
            else: 
                prefixes.append(code[:4] + " " + code[4:i])
        return prefixes
    
