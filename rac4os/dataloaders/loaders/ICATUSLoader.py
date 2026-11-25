from ..BaseDataLoader import ClassificationLoader
from ..registry import register
import pandas as pd

@register("ICATUS")
class ICATUSLaoder(ClassificationLoader):
    
    def to_records(self) -> list[dict]:
        
        
        def map_code(PID, ID, CID):
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
