from typing import Literal
import json
import dspy
from .BaseRAG import BaseRAGCoT
from . import registry
    

class RAGChainOfThought(BaseRAGCoT):
    
    CONFIGS = {
        "COICOP_2018":{
            "signature":registry.CoicopRAGChainOfThoughtSignature,
            "query_field":"input_expense",
            "context_field":"retrieved_candidates"
        },
        "ICATUS_2016":{
            "signature":registry.IcatusRAGChainOfThoughtSignature,
            "query_field":"input_activity",
            "context_field":"retrieved_candidates"
        },    
        "SEA_2021":{
            "signature":registry.SeaRAGChainOfThoughtSignature,
            "query_field":"input_expense",
            "context_field":"retrieved_candidates"
        },
        "SEA_2021_REPHRASED":{
            "signature":registry.SeaRAGChainOfThoughtSignature,
            "query_field":"input_expense",
            "context_field":"retrieved_candidates"
        },
    }
    
    def __init__(
        self,
        classification_name:Literal["COICOP_2018", "ICATUS_2016", "SEA_2021", "SEA_2021_REPHRASED"],
        **kwargs,
    ) -> None:
        
        if classification_name not in self.CONFIGS:
            raise ValueError(f"Unknown classification system: {classification_name}")

        super().__init__(
            classification_name=classification_name,
            **self.CONFIGS[classification_name],
            **kwargs,
        )
        

# ---------------------------------------------------------------------------
# Rewrite step: QueryRephraser
# ---------------------------------------------------------------------------


class QueryRephraser(dspy.Module):
    """
    Rewrites a raw item into a retrieval query in the vocabulary of the
    classification's labels. It does not classify.

    Picks the rephrasing signature for `classification_name` from
    `self.mapping` and runs it on its own `lm`, which can be the same LM as
    the Read step or a smaller, cheaper one.

    Called as `rephraser(query=...)`; returns a `dspy.Prediction` with
    `rephrased_query`. The input field name of the signature
    (`input_expense` / `input_activity`) is handled internally, so callers
    never need to know it.
    """

    def __init__(
        self,
        lm: dspy.LM,
        classification_name: str|None,
        signature: dspy.Signature|None=None,
    ) -> None:
        super().__init__()
        self.lm = lm
        self.mapping = {
            "COICOP_2018": registry.CoicopRephrasingSignature,
            "ICATUS_2016": registry.IcatusRephrasingSignature,
            "SEA_2021": registry.SeaRephrasingSignature,
            "SEA_2021_REPHRASED": registry.SeaRephrasingSignature,
        }
        
        assert classification_name in self.mapping or signature is not None, (
            f"No rephrasing signature for '{classification_name}'"
        )
        
        self.classification_name = classification_name
        self.signature = self.mapping.get(classification_name, signature)
        
        self.input_field = next(iter(self.signature.input_fields))
        self.rephrase = dspy.Predict(self.signature)

    def forward(self, query: str) -> dspy.Prediction:
        with dspy.context(lm=self.lm):
            prediction = self.rephrase(**{self.input_field: query})
        rephrased = (prediction.rephrased_query or "").strip()
        print(rephrased)
        return dspy.Prediction(rephrased_query=rephrased or query)


# ---------------------------------------------------------------------------
# Rewrite-Retrieve-Read pipeline
# ---------------------------------------------------------------------------


class RRR_RAGChainOfThought(RAGChainOfThought):
    """
    Rewrite-Retrieve-Read on top of `RAGChainOfThought`.

    1. Rewrite - `self.rephrasing_module` (a `QueryRephraser`) turns the raw
       item into a query in the vocabulary of the classification's labels.
    2. Retrieve - the rephrased query is embedded and the k nearest codes are
       pulled from the vector store (optionally merged with the hits for the
       original query, see `retrieve_with_original`).
    3. Read - the parent's ChainOfThought classifies the ORIGINAL item over
       the retrieved candidates. The rewrite only steers retrieval; it never
       replaces the item the model has to classify, so detail the rephraser
       dropped is not lost.

    Parameters (on top of `RAGChainOfThought`):
        rephraser_lm (dspy.LM | None) - LM for the rewrite step; None reuses
            the Read step's `self.lm`.
        retrieve_with_original (bool) - also retrieve with the raw query and
            interleave both result lists (rephrased first, deduplicated by
            code), capped at k. Guards against a bad rewrite while keeping
            the context the same size as the baseline.
    """

    def __init__(
        self,
        rephraser: QueryRephraser,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.rephrasing_module = rephraser

    
    def forward(self, query: str, k: int, where: dict | None = None):
        rephrased_query = self.rephrasing_module(query=query).rephrased_query
        retrieved_context = self.store.search_category(rephrased_query, k, where=where)
        retrieved_context_str = json.dumps(
            retrieved_context, indent=4, ensure_ascii=False
        )
        with dspy.context(lm=self.lm, track_usage=True):
            prediction = self.CoT(**{
                self.query_field: query,
                self.context_field: retrieved_context_str,
            })
        # Keep the rewritten query for logging and info later
        prediction.rephrased_query = rephrased_query
        return prediction
