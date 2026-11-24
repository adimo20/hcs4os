# Classifying by walking the hierarchy

```python
import os
from hcs4os.agents.HierarchicalNavigation import HierarchicalNavigationAgent

agent = HierarchicalNavigationAgent(
    classification_name="COICOP_2018",
    api_key=os.getenv("MISTRAL_API_KEY"),
    model_name="mistral/mistral-small-latest",
)

result = agent(input_expense="Basmati rice 1kg bag")
print(result.coicop_code, result.explanation)
print(result.get_lm_usage())          # token usage, tracked by default
```
