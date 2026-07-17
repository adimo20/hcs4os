from dataclasses import dataclass, field, fields, asdict
import re
import json

@dataclass
class Code:
    """
    The code class works a the base datatype for storing, retrieving and working with 
    codes from the classification systems. More or less every hierarchical classification
    system in official statistics, like NACE or COICOP haven a extensive documentation available, 
    what always contains 
    * `code` - Code in Form of digitis, e.g. 01111
    * `description` - overall description of the content of the code, e.g. Cereals (ND)
    * `level` - the level of the code inside the systems hierarchy, e.g. the corresponding level for the code 01111 is 4.
    * `detailled_description` - A more detailled description of what should be classified inside of a certain category, can be found in e.g. the COICOP documentation under introductory_notes
    * `details` - here will be all other details stored, that come on top of all the previous informations, e.g. explicit exclusion, inclusion, etc. 
    """
    code: str = field(default_factory=str)
    description: str = field(default_factory=str)
    level: int = field(default_factory=int)
    detailled_description: str = field(default_factory=str)
    details: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict):
        """
        Loads in a code form a dictionary and saves it as a Code object. 
        All keys that don't exist in the dict but that exist in the Code
        object will be left blank. It is imprtant that the codes inserted 
        here match the required datarypes defined, otherwise it will break here
        or deeper down the pipeline.
        """
        valid_fields: set[str] = {f.name for f in fields(cls)}
        cleaned_data: dict = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**cleaned_data)

    def to_json(self, indent: int=4) -> str:
        """Converts the dataclass instance to a JSON string."""
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)
    
    def to_dict(self) -> dict:
        """Converts the dataclass instance to a JSON string."""
        return asdict(self)
  
  