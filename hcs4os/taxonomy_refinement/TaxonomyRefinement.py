import dspy
from ..classification_system.registry import get_classification_system
import json

class GenerateParentDescription(dspy.Signature):
    """Generate a detailed description for a parent category in a hierarchical
    classification system , based on its own label and the
    labels of its direct children.

    The classification system is a directed acyclic graph in which each node is a
    code. Parent nodes represent broader, more general concepts; their children
    represent progressively more specific concepts. A parent's meaning is the
    union of the concepts covered by its children: every child falls under the
    parent, and together the children delimit the parent's scope.

    Write the detailed description so that it:
    - Characterizes the parent category as a whole (the general concept it captures),
      not merely a concatenation of the children.
    - Reflects the full scope implied by the children, making clear what kinds of
      items belong to this category.
    - Stays faithful to the given labels: do not invent subcategories, products, or
      distinctions that are not supported by the parent or child descriptions.
    - Is written in the same language as the input labels.
    - Is a coherent, self-contained prose description usable on its own, without
      requiring the reader to see the child list.
    """

    parent_child_pairs: dict = dspy.InputField(
        desc="A dict with two keys. 'parent': a dict describing the category to "
             "describe, with keys 'code' (str), 'description' (str, its short "
             "official label), and 'level' (int). 'children': a list of dicts, each "
             "a direct child of the parent, with the same 'code', 'description', and "
             "'level' keys. The children collectively delimit the parent's scope: "
             "every child falls under the parent, and together they define what the "
             "parent covers. Use the parent label as the core concept and the child "
             "labels as evidence of the parent's full extent."
    )
    detailed_description: str = dspy.OutputField(
        desc="A detailed, coherent description of the parent category, in the "
             "language of the input labels."
    )

class ClassificationsSystemAugmenter:
    
    def __init__(
        self,
        classification_name,
        api_key,
        model
    ):
        self.classification_name = classification_name
        self.classification_system = get_classification_system(self.classification_name)
        self.generator = dspy.ChainOfThought(GenerateParentDescription)
        
        self.api_key = api_key
        self.model = model
        
        lm = dspy.LM(
            api_key=self.api_key,
            model=self.model
        )
        dspy.configure(lm=lm)
    
    def enrich_classification_system(self, start_level:int)->None:
      
      """
      After this is through: lookup is updated but _children_register and codes are not
      """
      
      for lvl in range(start_level, 0, -1):
        
        level_x_codes = [code.code for code in self.classification_system.codes if code.level == lvl]
        print(level_x_codes)
        for current_child in level_x_codes:
          
          if current_child not in level_x_codes: continue
          
          parent = self.classification_system.get_parent(current_child)
          if parent is None: continue
          parent_code = parent.get("code")
          if parent_code is None: continue
          
          children = self.classification_system.get_children(parent_code)
          children_codes = [c.code for c in children]
          
          level_x_codes = [codex for codex in level_x_codes if codex not in children_codes]
          
          pair = {
            "parent":parent,
            "children":[c.to_dict() for c in children]   
          }
          
          try:
            print(f"Processing: {parent}")
            
            result = self.generator(
                parent_child_pairs=pair
            )
            self.classification_system._lookup[parent_code].detailled_description = result.detailed_description  
          
          except Exception as e:
            print(f"Error while processing: {parent}")
            print(e)
            continue
            
      def get_updated_classification_system(self, path):
        codes_new = [self.classification_system._lookup[k].to_dict() for k in self.classification_system._lookup.keys()]
        with open(path, "w", encoding="utf-8") as f:
          f.write(json.dumps(codes_new, indent=4, ensure_ascii=False))
        
        
