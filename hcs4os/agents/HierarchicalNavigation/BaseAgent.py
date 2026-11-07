from typing import Literal, Any
import dspy
from ...classification_system import get_classification_system
import warnings


RegisteredClassifications = Literal["COICOP_2018", "ICATUS_2016", "NAF_rev2", "SEA_2021"]

class BaseHierarchicalNavigationAgent(dspy.Module):
    
    def __init__(
        self,
        classification_name:RegisteredClassifications,
        api_key:str,
        model_name:str,
        signature:type[dspy.Signature],
        get_root_categories_doc:str,
        get_children_doc:str,
        get_code_doc:str,
        get_parent_doc:str,
        api_base:str|None=None,
    )->None:
        
        assert api_key is not None
        assert model_name is not None
        assert classification_name is not None
        
        self.signature:type[dspy.Signature] = signature
        self.get_root_categories_doc:str = get_root_categories_doc
        self.get_children_doc:str = get_children_doc
        self.get_code_doc:str = get_code_doc
        self.get_parent_doc:str = get_parent_doc
        
        
        self.lm = dspy.LM(
            api_key=api_key,
            api_base=api_base,
            model=model_name
        )
        dspy.configure(
            lm=self.lm,
            track_usage=True
        )
        
        self.classification_name = classification_name
        self.classification_system = get_classification_system(
            classification_name
        )
        self.agent = dspy.ReAct(
            self.signature, 
            tools=[
                dspy.Tool(
                    self.get_root_categories_tool,
                    name="get_root_categories",
                    desc=self.get_root_categories_doc, 
                ),
                dspy.Tool(
                    self.get_children_tool, 
                    name="get_children",
                    desc=self.get_children_doc, 
                ),
                dspy.Tool(
                    self.get_parent_tool, 
                    name="get_parent",
                    desc=self.get_parent_doc  
                ),
                dspy.Tool(
                    self.get_code_tool, 
                    name="get_code",
                    desc=self.get_code_doc  
                ),
            ],
        )
    
    def get_code_tool(self, code: str) -> dict:
        return self.classification_system.get_code(code).to_dict()

    def get_children_tool(self, code: str) -> list[dict]:
        return [c.to_dict() for c in self.classification_system.get_children(code)]

    def get_parent_tool(self, code: str) -> dict:
        return self.classification_system.get_parent(code).to_dict() # type: ignore

    def get_root_categories_tool(self) -> list[dict]:
        return self.classification_system.get_root_categories() 
    
    def forward(self, **kwargs: Any) -> dspy.Prediction:
        missing = set(self.signature.input_fields) - kwargs.keys()
        if missing:
            raise ValueError(f"Missing input fields: {sorted(missing)}")

        with dspy.context(lm=self.lm, track_usage=True):
            return self.agent(**kwargs)
      