import pandas as pd # type: ignore
from ..registry import register
from ..BaseDataLoader import ClassificationLoader

# File used has been downloaded from here: https://showvoc.op.europa.eu/#/datasets/ESTAT_Statistical_Classification_of_Economic_Activities_in_the_European_Community_Rev._2.1._%28NACE_2.1%29/downloads
# Filename: NACE_Rev2.1_Structure_Explanatory_Notes_EN.xlsx

REQUIRED_COLS = [
    "CODE", 
    "HEADING", 
    "PARENT_CODE", 
    "Includes", 
    "LEVEL", 
    "IncludesAlso", 
    "Excludes"
]
        
@register("NACE")
class NaceLoader(ClassificationLoader):

    def to_records(
        self        
    )->list[dict]:
        """Loads the official NACE documentation from an xlsx file and transforms it into a structure we can load with Code/ClassificationSystem"""
        
        df = pd.read_excel(self.path, dtype=str).fillna("")
        
        df = df[REQUIRED_COLS]
        df.columns = ["code", "description", "parent_code", "Includes", "level", "IncludesAlso", "Excludes"]
       
        df["details"] = df.apply(lambda row: 
            {
                "parent_code" : row["parent_code"], 
                "includes" :  row["Includes"], 
                "alsoIcludes" : row["IncludesAlso"], 
                "excludes" : row["Excludes"]
            }, axis=1)

        df = df[["code", "description", "level", "details"]]
        df["level"] = df.level.astype(int)
        nace_dict = df.to_dict(orient="index")
        nace_list = [nace_dict[k] for k in nace_dict.keys()]
        return nace_list

