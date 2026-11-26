from typing import Literal
import json
import dspy
from .BaseRAG import BaseRAGCoT
from . import registry
    

class RAGChainOfThought(BaseRAGCoT):
    
    """
    `BaseRAGCoT` with the bundled configurations in front of it.

    `CONFIGS` holds a signature and the two field names per shipped classification, so only
    the store, the credentials and the model have to be supplied. For your own taxonomy
    either add an entry to `CONFIGS` or construct `BaseRAGCoT` directly and pass the
    signature and field names yourself; both are the same code path.

    Parameters:
        classification_name (str): a key of `CONFIGS`.
        **kwargs (Any): forwarded to `BaseRAGCoT`, e.g. `store`, `api_key`, `model_name`.
    """
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
        
        """
        Looks the configuration up in `CONFIGS` and hands it to `BaseRAGCoT`.

        Anything you pass yourself is forwarded unchanged, so a bundled signature or field name
        can be overridden per call.

        Raises:
            ValueError: if `classification_name` has no entry in `CONFIGS`.
        """
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
        """
        Picks the rephrasing signature and builds the one-shot predictor that runs it.

        Either `classification_name` must be a key of the internal mapping or an explicit
        `signature` must be given; passing `classification_name=None` together with your own
        signature is how you rephrase for a taxonomy that is not bundled. The signature's first
        input field is read off here, so callers never have to know what it is called.

        Raises:
            AssertionError: if neither a known classification name nor a signature was given.
        """
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
        """
        Rewrite one item into a retrieval query.

        Runs on this module's own LM through a `dspy.context`, so the rewrite can use a smaller
        and cheaper model than the classification step. An empty rewrite falls back to the
        original item, so the pipeline can never end up retrieving on an empty query.

        Parameters:
            query (str): the raw item.

        Returns:
            dspy.Prediction: carries `rephrased_query`; the original item if the rewrite came back empty.
        """
        with dspy.context(lm=self.lm):
            prediction = self.rephrase(**{self.input_field: query})
        rephrased = (prediction.rephrased_query or "").strip()
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
       pulled from the vector store.
    3. Read - the parent's ChainOfThought classifies the ORIGINAL item over
       the retrieved candidates. The rewrite only steers retrieval; it never
       replaces the item the model has to classify, so detail the rephraser
       dropped is not lost.

    Parameters (on top of `RAGChainOfThought`):
        rephraser_lm (dspy.LM | None) - LM for the rewrite step; None reuses
            the Read step's `self.lm`.        
    """

    def __init__(
        self,
        rephraser: QueryRephraser,
        *args,
        **kwargs,
    ) -> None:
        """
        Builds the linear pipeline and puts the rewrite step in front of it.

        Parameters:
            rephraser (QueryRephraser): the rewrite step.
            *args (Any): forwarded to `RAGChainOfThought`.
            **kwargs (Any): forwarded to `RAGChainOfThought`, e.g. `classification_name`, `store`, `api_key`, `model_name`.
        """
        super().__init__(*args, **kwargs)

        self.rephrasing_module = rephraser

    
    def forward(
        self, 
        query: str, 
        k: int, 
        where: dict | None = None
    ) -> dspy.Prediction:
        """
        Rewrite the item, retrieve on the rewritten query, then classify the original item.

        The rewrite steers retrieval only. The generator still sees the item as it came in, so
        detail the rephraser dropped cannot be lost from the decision itself. The query that
        was actually retrieved on is attached to the prediction as `rephrased_query`, which is
        what you inspect when a result looks wrong: it separates a bad rewrite from a bad
        choice among good candidates.

        Parameters:
            query (str): the item to classify.
            k (int): how many candidates to retrieve.
            where (dict|None): metadata filter passed straight to the store.

        Returns:
            dspy.Prediction: the signature's output fields, `reasoning`, and `rephrased_query`.
        """
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