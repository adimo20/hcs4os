# Classifying by retrieval, without the loop

```python
from hcs4os.rag.RAG import RAGChainOfThought

rag = RAGChainOfThought(
    classification_name="SEA_2021",
    api_key=os.getenv("MISTRAL_API_KEY"),
    model_name="mistral/mistral-small-latest",
    store=store,
)

result = rag(query="frisches Brot vom Bäcker", k=10)
print(result.sea_code, result.explanation)

# restrict the candidates to one level of the hierarchy
result = rag(query="Beitrag zur Krankenversicherung", k=10, where={"level": 5})
```

`k` is an argument rather than something the model decides, and `where` is passed
through to chroma as a metadata filter — `{"level": 5}` is the useful case, since it
forces candidates to the depth actually being predicted.
