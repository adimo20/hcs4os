# Prompt reference

Every classifier's system prompt lives in the docstring of its `dspy.Signature`,
where dspy reads it. The pages below render those docstrings straight from the
installed package, so what you see here is exactly what the model receives — it
cannot drift from the code.

Three families of prompt:

| family | used by | what it asks the model to do |
| --- | --- | --- |
| [Hierarchical navigation](navigation.md) | `HierarchicalNavigationAgent` | descend the tree one level at a time, following `excludes` pointers |
| [Retrieval](retrieval.md) | `RAGAgent`, `RAGChainOfThought` | choose from retrieved candidates, never from memory |
| [Rephrasing](rephrasing.md) | `QueryRephraser` (RRR) | restate the item in the classification's vocabulary, without classifying |

The prompts are deliberately classification-specific: what has to survive a
rewrite, which notes decide placement, and which level to target all differ
between COICOP, SEA, ICATUS and NAF.
