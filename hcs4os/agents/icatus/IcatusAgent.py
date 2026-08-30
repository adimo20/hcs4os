import dspy
from ...classification_system import get_classification_system
from pathlib import Path
import warnings

warnings.filterwarnings(
  "ignore", 
  category=UserWarning, 
  module="pydantic"
)


def create_system_prompt(
  target_level:int=3
)->str:
  
  return f"""
      Classify a time-use activity into a single ICATUS 2016 code by searching
      the classification hierarchy top-down, using the available tools rather
      than relying on memorized codes.

      ICATUS 2016 (International Classification of Activities for Time-Use
      Statistics) is a tree. The top level is a set of nine major divisions
      (e.g. "1" Employment and related activities, "9" Self-care and
      maintenance), and each code has children one level deeper
      (major division -> division -> group), e.g. 1 -> 11 -> 110. A valid final
      answer is a real code that exists in the system; prefer the most specific
      (deepest, leaf) code that correctly covers the activity.

      Tools:
      - get_root_categories_tool(): list the top-level major divisions. Start here.
      - get_children_tool(code): list the direct child codes one level down, each
        with its title and includes/excludes notes.
      - get_code_tool(code): read one code's full record (title, definition, and
        'includes' / 'excludes' / 'examples' notes).
      - get_parent_tool(code): move one level up to re-read broader context or
        reconsider a branch.

      Reasoning protocol:
      1. Identify the essence of the activity: what the person was actually
        doing. Note any detail that affects placement (whether it was done for
        pay or profit, for own final use, as an unpaid service for the
        household, as volunteering, or as a personal activity; and for whom the
        activity was performed).
      2. Call get_root_categories() and choose the single best-fitting major
        division. If two major divisions seem plausible, note the alternative
        to revisit later.
      3. Descend one level at a time with get_children on the current code.
        At each level:
          - Compare the activity against every child's title and notes.
          - Read the 'includes' and 'examples' notes to confirm a match.
          - Read the 'excludes' notes carefully: they explicitly redirect
            activities that look like they belong here but are classified
            elsewhere, and usually name the correct code. Follow those
            pointers instead of forcing a fit.
          - Pick the best-matching child and repeat.
      4. Continue descending until you reach a leaf (get_children returns an
        empty list) or until no deeper code fits better than the current one.
      5. Before committing, verify the chosen code with get_code and check that
        its notes do not exclude this activity. If they do, backtrack using
        get_parent or return to a noted alternative branch and search again.
      6. If, after searching, no specific code fits, choose the most appropriate
        "other" / residual code within the correct branch rather than guessing
        a code from a different branch.

      Rules:
      - Never invent or guess a code from memory. Every code in the final answer
        and in your reasoning must have been returned by a tool.
      - The final icatus_code must be an exact code string that the tools
        returned (e.g. "110"), not a paraphrase or a made-up variant.
      - The target level for the classification of the activity is **{str(target_level)}**,
      always try to find a code from level **{str(target_level)}** the level of a code
      is returned by get_code.
      - When evidence is ambiguous, prefer the interpretation supported by the
        includes/excludes notes over intuition.      

      Output:
      - icatus_code: the single most specific ICATUS code that correctly
        classifies the activity.
      - explaination: a concise justification tracing the path taken (major
        division -> ... -> final code) and citing the decisive includes/excludes
        note(s) that determined the choice, including any branch you rejected
        and why.
      """
    

class ICATUSHierarchicalSearchAgentSignature(dspy.Signature):
    
    input_activity: str = dspy.InputField(
        desc="A time-use activity to classify, e.g. a diary line item or a short description of an activity a person spent time on."
    )
    icatus_code: str = dspy.OutputField(
        desc="The single most specific ICATUS code that correctly covers the activity, exactly as returned by the tools (e.g. '110')."
    )
    explaination: str = dspy.OutputField(
        desc="Concise reasoning tracing the top-down path to the code and citing the includes/excludes notes that justified it, plus any rejected alternative."
    )    
    
def load_doc_strings(tool_name:str):
  path=path = Path(__file__).parent / "tool_descriptions" / f"{tool_name}.txt"
  with open(path, "r", encoding="utf-8") as f:
    return f.read() 
  
  
class ICATUSHierarchicalSearchAent(dspy.Module):
    
    def __init__(
        self,
        api_key:str,
        model_name:str,
        api_base:str|None=None,
        target_level:int=3
    )->None:
        
        assert api_key is not None
        assert model_name is not None
        
        
        self.lm = dspy.LM(
            api_key=api_key,
            api_base=api_base,
            model=model_name
        )
        dspy.configure(lm=self.lm)
        
        self.icatus = get_classification_system("ICATUS_2016")     
        self.signature = ICATUSHierarchicalSearchAgentSignature
        self.signature.__doc__ = create_system_prompt(target_level=target_level)
        self.agent = dspy.ReAct(
            self.signature,
            tools=[
                dspy.Tool(self.get_root_categories_tool, name="get_root_categories",
                          desc=load_doc_strings("get_root_categories")),
                dspy.Tool(self.get_children_tool, name="get_children",
                          desc=load_doc_strings("get_children")),
                dspy.Tool(self.get_parent_tool, name="get_parent",
                          desc=load_doc_strings("get_parent")),
                dspy.Tool(self.get_code_tool, name="get_code",
                          desc=load_doc_strings("get_code")),
            ],
        )
    
    def get_code_tool(self, code: str) -> dict:
        return self.icatus.get_code(code).to_dict()

    def get_children_tool(self, code: str) -> list[dict]:
        return [c.to_dict() for c in self.icatus.get_children(code)]

    def get_parent_tool(self, code: str) -> dict:
        return self.icatus.get_parent(code).to_dict() # type: ignore

    def get_root_categories_tool(self) -> list[dict]:
        return self.icatus.get_root_categories()