from typing import Literal
import dspy
from ...classification_system import get_classification_system
from pathlib import Path
import warnings
from .registry import tool_descriptions

warnings.filterwarnings(
  "ignore", 
  category=UserWarning, 
  module="pydantic"
)

class HierarchicalNavigationAgent(dspy.Module):
    
    def __init__(
        self,
        classification_name:Literal["COICOP", "ICATUS"],
        api_key:str,
        model_name:str,
        api_base:str|None=None,
    )->None:
        
        assert api_key is not None
        assert model_name is not None
        assert classification_name is not None
        
        
        self.lm = dspy.LM(
            api_key=api_key,
            api_base=api_base,
            model=model_name
        )
        dspy.configure(lm=self.lm)
        
        self.coicop = get_classification_system("COICOP_2018")     
        self.classification_name = classification_name
        self.signature = tool_descriptions.get(self.classification_name).get("signature") # type: ignore
        self.signature.__doc__ = tool_descriptions.get(self.classification_name).get("system_prompt")  # type: ignore
        self.agent = dspy.ReAct(
            self.signature,  # type: ignore
            tools=[
                dspy.Tool(
                    self.get_root_categories_tool,
                    name="get_root_categories",
                    desc=tool_descriptions.get(self.classification_name).get("get_root_categories")  # type: ignore
                ),
                dspy.Tool(
                    self.get_children_tool, 
                    name="get_children",
                    desc=tool_descriptions.get(self.classification_name).get("get_children")),  # type: ignore
                dspy.Tool(
                    self.get_parent_tool, 
                    name="get_parent",
                    desc=tool_descriptions.get(self.classification_name).get("get_parent")  # type: ignore
                    ),
                dspy.Tool(
                    self.get_code_tool, 
                    name="get_code",
                    desc=tool_descriptions.get(self.classification_name).get("get_code")  # type: ignore
                    ),
            ],
        )
    
    def get_code_tool(self, code: str) -> dict:
        return self.coicop.get_code(code).to_dict()

    def get_children_tool(self, code: str) -> list[dict]:
        return [c.to_dict() for c in self.coicop.get_children(code)]

    def get_parent_tool(self, code: str) -> dict:
        return self.coicop.get_parent(code).to_dict() # type: ignore

    def get_root_categories_tool(self) -> list[dict]:
        return self.coicop.get_root_categories() 
      