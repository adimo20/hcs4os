from ..BaseDataLoader import ClassificationLoader
from ..registry import register
import pandas as pd # type: ignore
import re

@register("COICOP")
class CoicopDataLoader(ClassificationLoader):
        
    """
    Loader for the official COICOP 2018 structure as published in xlsx, registered under
    "COICOP".

    The sheet's three note columns (`includes`, `alsoIncludes`, `excludes`) are folded into
    the `details` dict rather than dropped, because the exclusion notes are what settle most
    of the hard cases between neighbouring categories. The level is derived from the code
    itself, by counting its dot-separated groups.
    """
    def to_records(
        self
    )->list[dict]:
        
        """
        Read the xlsx and return `Code`-shaped dicts.

        Non-breaking spaces and the `_x000D_` carriage returns that the published file contains
        are replaced with ordinary spaces first; left in, they end up in the embedded text and
        in the prompt.

        Returns:
            list[dict]: one dict per COICOP category, with `includes`, `alsoIncludes` and `excludes` kept under `details`.
        """
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

