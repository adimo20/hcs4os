# Retrieval without the loop

The module is laid out like `agents/`: [registry.py](https://github.com/adimo20/hcs4os/blob/master/hcs4os/rag/registry.py) holds all
the prose — the three ChainOfThought signatures and the three rephrasing signatures —
and [RAG.py](https://github.com/adimo20/hcs4os/blob/master/hcs4os/rag/RAG.py) holds only the classes that use them.

`RAGChainOfThought` uses the same store and the same `search_category`, but a single
`dspy.ChainOfThought` instead of a ReAct loop. It retrieves once and reasons once.

The differences that matter in practice:

- `k` is set by the caller, not chosen by the model — retrieval depth becomes an
  experimental parameter rather than a model decision.
- `where` is forwarded to chroma as a metadata filter (`{"level": 5}`).
- The candidates are rendered as indented JSON into a `retrieved_candidates` input
  field, and the whole reasoning protocol lives in the signature's own docstring.
- One retrieval, one answer: it cannot recover from a bad shortlist the way the ReAct
  agent can by re-querying. That is the trade for costing a single model call.

[BaseRAG.py](https://github.com/adimo20/hcs4os/blob/master/hcs4os/rag/BaseRAG.py) is its configurable twin: `BaseRAGCoT` takes the
`signature` plus `query_field` and `context_field` naming which of its input fields
receive the item and the retrieved candidates, and asserts at construction that both
names really exist on the signature. Like the agents, it takes an injected `store`.

`RAGChainOfThought` is the registered subclass, and because the base needs the field
names too, its `CONFIGS` carries three things per classification rather than one:

```python
CONFIGS = {
    "COICOP_2018": {"signature":     registry.CoicopRAGChainOfThoughtSignature,
                    "query_field":   "input_expense",
                    "context_field": "retrieved_candidates"},
    ...   # ICATUS_2016, SEA_2021, SEA_2021_REPHRASED
}
```

Its `forward(query, k, where=None)` differs from the agents' in one respect worth
knowing: it calls `self.CoT` without a `dspy.context` wrapper, so unlike the ReAct
agents and unlike `RRR_RAGChainOfThought` it does not pin its own LM per call and
relies on the `dspy.configure` done at construction.

### Rewrite-Retrieve-Read

`RRR_RAGChainOfThought` subclasses `RAGChainOfThought` and puts a rewrite step in
front of retrieval. The motivation is a property of the index: only each code's short
`description` label is embedded — `"Eggplants (aubergines), fresh or chilled"`,
`"Steinobstkonserven, z.B. Pflaumen-, … Pfirsichkonserven"`, `"Eating meals/snack"`.
Real items do not look like that. Receipt shorthand, brand names, bank-statement text
and diary abbreviations sit far from those labels in embedding space, so retrieval
misses before the reasoning model ever gets a chance.

Measured on the COICOP collection, the raw receipt line `"BASMATI REIS 1KG BTL"`
retrieves `01.1.1.1.5` Millet, `12.2.9.1` Remittances fees and `01.1.1.3` Bread —
rice is nowhere in the top 3. Rewritten to `"rice, milled"`, the same store returns
`01.1.1.1.2` Rice first.

The three steps:

1. **Rewrite** — `self.rephrasing_module`, a `QueryRephraser`, restates the item in
   the vocabulary of the labels. It explicitly does **not** classify: the signatures
   forbid emitting a code or choosing between categories.
2. **Retrieve** — the rewritten query is embedded and the `k` nearest codes are pulled
   from the store.
3. **Read** — the inherited `ChainOfThought` classifies the **original** item over
   those candidates. The rewrite only steers retrieval; it never replaces what the
   model has to classify, so detail the rephraser dropped is not lost from the
   decision.

`forward(query, k, where=None)` matches the parent's, and the prediction carries one
extra field, `rephrased_query`, so you can see and log what was actually searched for.
That makes `RRR_RAGChainOfThought` a drop-in A/B against `RAGChainOfThought` on the
same collection. The Read step runs under `dspy.context(lm=self.lm, track_usage=True)`.

**The rephraser is injected.** `RRR_RAGChainOfThought(rephraser, *args, **kwargs)`
takes a built `QueryRephraser` as its first argument, so the rewrite step carries its
own LM and the two halves of the pipeline can run on different models — a small cheap
model to restate the item, a stronger one to decide the code:

```python
rrr = RRR_RAGChainOfThought(
    rephraser=QueryRephraser(lm=cheap_lm, classification_name="COICOP_2018"),
    classification_name="COICOP_2018",
    model_name="mistral/mistral-large-latest",   # the Read step
    api_key=..., store=store,
)
```

**`QueryRephraser`** is a standalone `dspy.Module`, usable on its own. It resolves a
signature from its `mapping` (`COICOP_2018`, `ICATUS_2016`, `SEA_2021`,
`SEA_2021_REPHRASED`), runs `dspy.Predict`, and hides which input field that signature
uses — callers always pass `query=`:

```python
from hcs4os.rag.RAG import QueryRephraser

rephraser = QueryRephraser(lm=cheap_lm, classification_name="SEA_2021")
rephraser(query="LASTSCHRIFT ALLIANZ KFZ").rephrased_query
# intended shape of the output: 'Beitrag Kfz-Haftpflichtversicherung'
```

It runs inside `with dspy.context(lm=self.lm)`, so it uses the LM handed to its
constructor regardless of what is globally configured. If the model returns an empty
rewrite, the raw query is used instead.

For a classification with no registered rephrasing signature, pass your own as
`signature=` and leave `classification_name=None`:

```python
QueryRephraser(lm=cheap_lm, classification_name=None, signature=MyWzRephrasingSignature)
```

Whichever input field that signature declares first is the one the query is fed to,
so callers still only pass `query=`. Supplying neither a registered name nor a
`signature` fails at construction with
`AssertionError: No rephrasing signature for '<name>'`. Note that a registered name
takes priority: pass both and the registered signature wins, so `signature=` is for
classifications the registry does not cover rather than for overriding one it does.

The rephrasing prompts are the most classification-specific prose in the library,
because what has to survive the rewrite differs per system. COICOP and SEA keep
placement-driving attributes — state or form (fresh, frozen, canned, ready-made), good
vs. service, consumed at home vs. served, new vs. second-hand — and drop brand, shop,
quantity and price. SEA additionally has to pin down the direction of the money flow,
since an insurance premium (Abteilung 16) and a household appliance (05) are different
parts of the system, and it answers in German to match the German labels. ICATUS
instead has to surface the economic nature of the activity and who it was done for,
because its titles are organised by that rather than by what the person did. All three
end with the same guard: do not force a category term you are unsure of, because a
wrong one pulls retrieval into the wrong branch.
