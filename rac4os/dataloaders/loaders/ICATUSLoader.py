from ..BaseDataLoader import ClassificationLoader
from ..registry import register
import pandas as pd

@register("ICATUS")
class ICATUSLaoder(ClassificationLoader):
    
    """
    Loader for the UNSD ICATUS 2016 time-use statistics csv, registered under "ICATUS".

    The source spreads the code over three columns, one per level, with the deeper ones
    empty for higher categories. The loader collapses them into a single code and derives
    the level from its length.
    """
    def to_records(self) -> list[dict]:
        
        
        """
        Read the csv and return `Code`-shaped dicts.

        Returns:
            list[dict]: one dict per ICATUS category, with `includes`, `excludes` and `keywords` kept under `details`.
        """
        def map_code(PID, ID, CID):
            """
            Pick the deepest of the three code columns that is populated.

            Parameters:
                PID: major-division code, set for every row.
                ID: second-level code, empty for major divisions.
                CID: third-level code, empty above it.

            Returns:
                the most specific code present on the row.
            """
            if pd.isna(ID) and pd.isna(CID):
                return PID
            elif pd.isna(CID):
                return ID
            else:
                return CID
        
        df = pd.read_csv(self.path)
        df.columns = ['PID', 'ID', 'CID', 'description', 'detailled_description', 'includes', 'excludes', 'keywords']
        
            

        df["code"] = df.apply(lambda row: str(int(map_code(row["PID"], row["ID"], row["CID"]))),axis=1)
        codes = df.to_dict(orient="records")
        records = [
            {
                "code":c["code"], 
                "description":c["description"],
                "detailled_description":c["detailled_description"],
                "level":len(c["code"]),
                "details":{
                    "includes":c["includes"],
                    "excludes":c["excludes"],
                    "keywords":c["keywords"]
                }    
             } for c in codes
        ]
        
        return records
