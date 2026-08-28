import dspy
from ...classification_system import get_classification_system
from pathlib import Path

class CoicopHierarchicalSearchAgentSignature(dspy.Signature):
    """
    Classify a household expense into a single COICOP code by searching the
    classification hierarchy top-down, using the available tools rather than
    relying on memorized codes.

    COICOP is a tree. The top level is a set of divisions (e.g. "01" Food and
    non-alcoholic beverages), and each code has children one level deeper
    (division -> group -> class -> subclass), e.g. 01 -> 01.1 -> 01.1.1 ->
    01.1.1.1. A valid final answer is a real code that exists in the system;
    prefer the most specific (deepest, leaf) code that correctly covers the
    expense.

    Tools:
    - get_root_categories_tool(): list the top-level divisions. Start here.
    - get_children_tool(code): list the direct child codes one level down, each
      with its description and includes/excludes notes.
    - get_code_tool(code): read one code's full record (description, elaborated
      definition, and 'includes' / 'alsoIncludes' / 'excludes' notes).
    - get_parent_tool(code): move one level up to re-read broader context or
      reconsider a branch.

    Reasoning protocol:
    1. Identify the essence of the expense: what good or service was actually
       purchased. Note any detail that affects placement (state/form of the
       item, purpose, whether it is a good vs. a service).
    2. Call get_root_categories() and choose the single best-fitting division.
       If two divisions seem plausible, note the alternative to revisit later.
    3. Descend one level at a time with get_children on the current code.
       At each level:
         - Compare the expense against every child's description and notes.
         - Read the 'includes' and 'alsoIncludes' notes to confirm a match.
         - Read the 'excludes' notes carefully: they explicitly redirect
           items that look like they belong here but are classified
           elsewhere, and usually name the correct code. Follow those
           pointers instead of forcing a fit.
         - Pick the best-matching child and repeat.
    4. Continue descending until you reach a leaf (get_children returns an
       empty list) or until no deeper code fits better than the current one.
    5. Before committing, verify the chosen code with get_code and check that
       its notes do not exclude this expense. If they do, backtrack using
       get_parent or return to a noted alternative branch and search again.
    6. If, after searching, no specific code fits, choose the most appropriate
       "other" / residual code within the correct branch rather than guessing
       a code from a different branch.

    Rules:
    - Never invent or guess a code from memory. Every code in the final answer
      and in your reasoning must have been returned by a tool.
    - The final coicop_code must be an exact code string that the tools
      returned (e.g. "01.1.1.1"), not a paraphrase or a made-up variant.
    - If it is possible always return a least a level 4 code, that has 5 digits
    - When evidence is ambiguous, prefer the interpretation supported by the
      includes/excludes notes over intuition.

    Output:
    - coicop_code: the single most specific COICOP code that correctly
      classifies the expense.
    - explaination: a concise justification tracing the path taken (division
      -> ... -> final code) and citing the decisive includes/excludes note(s)
      that determined the choice, including any branch you rejected and why.
    """
    input_expense: str = dspy.InputField(
        desc="A household expense to classify, e.g. a receipt line item or a short description of a purchased good or service."
    )
    coicop_code: str = dspy.OutputField(
        desc="The single most specific COICOP code that correctly covers the expense, exactly as returned by the tools (e.g. '01.1.1.1')."
    )
    explaination: str = dspy.OutputField(
        desc="Concise reasoning tracing the top-down path to the code and citing the includes/excludes notes that justified it, plus any rejected alternative."
    )    
    
def load_doc_strings(tool_name:str):
  path=path = Path(__file__).parent / "tool_descriptions" / f"{tool_name}.txt"
  with open(path, "r", encoding="utf-8") as f:
    return f.read() 
  
  
class CoicopHierarchicalSearchAent(dspy.Module):
    
    def __init__(
        self,
        api_key:str,
        model_name:str,
        api_base:str|None=None
    )->None:
        
        assert api_key is not None
        assert model_name is not None
        
        
        self.lm = dspy.LM(
            api_key=api_key,
            api_base=api_base,
            model=model_name
        )
        dspy.configure(lm=self.lm)
        
        self.coicop = get_classification_system("COICOP_2018")     
        
        self.agent = dspy.ReAct(
            CoicopHierarchicalSearchAgentSignature,
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
        return self.coicop.get_code(code).to_dict()

    def get_children_tool(self, code: str) -> list[dict]:
        return [c.to_dict() for c in self.coicop.get_children(code)]

    def get_parent_tool(self, code: str) -> dict:
        return self.coicop.get_parent(code).to_dict() # type: ignore

    def get_root_categories_tool(self) -> list[dict]:
        return self.coicop.get_root_categories() 
      