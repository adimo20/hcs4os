# Classifying with a rewrite step (RRR)

```python
import dspy
from hcs4os.rag.RAG import RRR_RAGChainOfThought, QueryRephraser

rrr = RRR_RAGChainOfThought(
    # the rewrite step is injected, so it can run on a smaller, cheaper model
    rephraser=QueryRephraser(
        lm=dspy.LM(model="mistral/mistral-small-latest", api_key=os.getenv("MISTRAL_API_KEY")),
        classification_name="COICOP_2018",
    ),
    classification_name="COICOP_2018",
    api_key=os.getenv("MISTRAL_API_KEY"),
    model_name="mistral/mistral-large-latest",
    store=store,
)

result = rrr(query="BASMATI REIS 1KG BTL", k=10)
print(result.rephrased_query)   # 'rice, milled' — what was actually embedded
print(result.coicop_code, result.explanation)
```

Otherwise a drop-in replacement for `RAGChainOfThought`: identical
`forward(query, k, where=None)`, one extra field on the prediction
(`rephrased_query`) recording what the retrieval actually searched for.

The model string of every classifier is passed straight to `dspy.LM`, i.e. any
LiteLLM-supported provider works (`openai/…`, `anthropic/…`, `ollama_chat/…`); pass
`api_base=` for self-hosted endpoints. The reasoning LM and the embedding backend are
chosen independently — a local MiniLM store with a hosted reasoning model, or an
OpenAI store with a local Ollama reasoner, are both fine.

Usage tracking is on for every classifier: `prediction.get_lm_usage()` returns the
per-model token counts for that call. Note that `forward()` pins the classifier's own
`self.lm` through `dspy.context`, so a global `dspy.configure(lm=…)` at call time is
ignored — to swap the model after construction, set the `.lm` attribute.
