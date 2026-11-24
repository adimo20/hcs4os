# Classifying by retrieval

```python
import os
from hcs4os.agents.RAG.RAGAgent import RAGAgent

agent = RAGAgent(
    classification_name="COICOP_2018",
    api_key=os.getenv("MISTRAL_API_KEY"),
    model_name="mistral/mistral-small-latest",
    store=store,                           # the indexed store from above
)

result = agent(input_expense="Basmati rice 1kg bag")   # input_activity for ICATUS
print(result.coicop_code, result.explanation)         # sea_code for SEA
```

`RAGAgent` does not build or fill the store — it queries the one you pass in, so index
it once (above) before the first call. The agent wraps the store's `search_category`
as a single ReAct tool and re-queries as it reasons.
