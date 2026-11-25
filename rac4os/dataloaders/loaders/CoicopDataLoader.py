from ..BaseDataLoader import ClassificationLoader
from ..registry import register
import pandas as pd # type: ignore
import re

@register("COICOP")
class CoicopDataLoader(ClassificationLoader):
        
    def to_records(
        self
    )->list[dict]:
        
        coicop_df = pd.read_excel(self.path, dtype=str).fillna("")
        for col in coicop_df.columns:
            coicop_df[col] = coicop_df[col].apply(lambda s: re.sub(r"(\xa00)|(_x000D_\n)|(\xa0)", " ",s))
        coicop_df["details"] = coicop_df.apply(
            lambda row: {
                "includes":row["includes"],
                "alsoIncludes": row["alsoIncludes"],
                "excludes":row["excludes"]
            },axis=1)

        coicop_df = coicop_df.drop(
            [
                "includes",
                "alsoIncludes",
                "excludes"
            ], axis=1)

        coicop_df.columns = ["code", "description", "detailled_description", "details"]
        coicop_df["level"] = coicop_df.code.apply(lambda s: len(s.split("."))) # int
        return coicop_df.to_dict(orient="records") 

