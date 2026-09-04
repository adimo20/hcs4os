import dspy
from ..agents.RAG.vector_database import VectorStore
from ..classification_system.registry import get_classification_system
from typing import Literal
import json

class CoicopRAGChainOfThoughtSignature(dspy.Signature):
    """
    Classify a household expense into a single COICOP code by reasoning over a
    set of pre-retrieved candidate codes, rather than relying on memorized codes.

    COICOP is a hierarchy of codes (division -> group -> class -> subclass),
    e.g. 01 -> 01.1 -> 01.1.1 -> 01.1.1.1. A valid final answer is a real code
    present in the retrieved candidates; prefer the most specific code that
    correctly covers the expense.

    Reasoning protocol:
    1. Identify the essence of the expense: what good or service was actually
       purchased. Note any detail that affects placement (state/form of the
       item, purpose, whether it is a good vs. a service).
    2. Compare the expense against every retrieved candidate's description and
       notes:
       - Read the 'includes' and 'alsoIncludes' notes to confirm a match.
       - Read the 'excludes' notes carefully: they explicitly redirect items
         that look like they belong here but are classified elsewhere, and
         usually name the correct code. If the redirected code is itself among
         the candidates, prefer it.
    3. Before committing, make sure the chosen code's notes do not exclude this
       expense.
    4. If no candidate fits specifically, choose the most appropriate "other" /
       residual candidate within the correct branch rather than guessing a code
       from a different branch.

    Rules:
    - Never invent or guess a code from memory. The final coicop_code must be an
      exact code string that appears in retrieved_candidates, not a paraphrase
      or a made-up variant.
    - The target level for the classification of the expense is **4**; always
      try to find a level-4 code among the candidates.
    - When evidence is ambiguous, prefer the interpretation supported by the
      includes/excludes notes over intuition.
    """

    input_expense: str = dspy.InputField(
        desc="A household expense to classify, e.g. a receipt line item or a short description of a purchased good or service."
    )
    retrieved_candidates: str = dspy.InputField(
        desc="The pre-retrieved candidate COICOP records most semantically similar to the expense, each with its code, description, level, and 'includes'/'alsoIncludes'/'excludes' notes. Choose the final code only from these records."
    )
    coicop_code: str = dspy.OutputField(
        desc="The single most specific COICOP code from retrieved_candidates that correctly covers the expense, exactly as given (e.g. '01.1.1.1')."
    )
    explaination: str = dspy.OutputField(
        desc="Concise reasoning explaining which retrieved candidate was chosen and why, citing the includes/excludes notes that justified it, plus any rejected alternative."
    )


class IcatusRAGChainOfThoughtSignature(dspy.Signature):
    """
    Classify a time-use activity into a single ICATUS 2016 code by reasoning over
    a set of pre-retrieved candidate codes, rather than relying on memorized
    codes.

    ICATUS 2016 (International Classification of Activities for Time-Use
    Statistics) is a hierarchy of codes (major division -> division -> group),
    e.g. 1 -> 11 -> 110. A valid final answer is a real code present in the
    retrieved candidates; prefer the most specific code that correctly covers
    the activity.

    Reasoning protocol:
    1. Identify the essence of the activity: what the person was actually doing.
       Note any detail that affects placement (whether it was done for pay or
       profit, for own final use, as an unpaid service for the household, as
       volunteering, or as a personal activity; and for whom it was performed).
    2. Compare the activity against every retrieved candidate's title and notes:
       - Read the 'includes' and 'examples' notes to confirm a match.
       - Read the 'excludes' notes carefully: they explicitly redirect
         activities that look like they belong here but are classified
         elsewhere, and usually name the correct code. If the redirected code is
         itself among the candidates, prefer it.
    3. Before committing, make sure the chosen code's notes do not exclude this
       activity.
    4. If no candidate fits specifically, choose the most appropriate "other" /
       residual candidate within the correct branch rather than guessing a code
       from a different branch.

    Rules:
    - Never invent or guess a code from memory. The final icatus_code must be an
      exact code string that appears in retrieved_candidates, not a paraphrase
      or a made-up variant.
    - The target level for the classification of the activity is **3**; always
      try to find a level-3 code among the candidates.
    - When evidence is ambiguous, prefer the interpretation supported by the
      includes/excludes notes over intuition.
    """

    input_activity: str = dspy.InputField(
        desc="A time-use activity to classify, e.g. a diary line item or a short description of an activity a person spent time on."
    )
    retrieved_candidates: str = dspy.InputField(
        desc="The pre-retrieved candidate ICATUS records most semantically similar to the activity, each with its code, title, level, and 'includes'/'excludes'/'examples' notes. Choose the final code only from these records."
    )
    icatus_code: str = dspy.OutputField(
        desc="The single most specific ICATUS code from retrieved_candidates that correctly covers the activity, exactly as given (e.g. '110')."
    )
    explaination: str = dspy.OutputField(
        desc="Concise reasoning explaining which retrieved candidate was chosen and why, citing the includes/excludes notes that justified it, plus any rejected alternative."
    )

mapping = {
    "COICOP_2018":CoicopRAGChainOfThoughtSignature,
    "ICATUS_2016":IcatusRAGChainOfThoughtSignature
}

class RAGChainOfThought(dspy.Module):
    
    def __init__(
        self,
        embedding_model_name:str,
        collection_name:str,
        classification_name:Literal["COICOP_2018", "ICATUS_2016"],
        api_key:str,
        model_name:str,
        api_base:str|None=None,
        chromadb_path:str="./data/chroma",
        create_new_collection:bool=True,             
    )->None:

        assert api_key is not None
        assert model_name is not None
        assert classification_name is not None and classification_name in mapping.keys()

        self.lm = dspy.LM(
            api_key=api_key,
            api_base=api_base,
            model=model_name
        )
        dspy.configure(lm=self.lm)

        self.classification_name = classification_name
        self.classification_system = get_classification_system(
            classification_name
        )

        self.store = VectorStore(
            collection_name=collection_name,
            model_name=embedding_model_name,
            chromadb_path=chromadb_path,
        )
        if create_new_collection:
            self.index()

        
        self.CoT = dspy.ChainOfThought(
            mapping[self.classification_name]
        )
        
    def index(
        self
    )->None:
        
        def flatten(c: dict) -> dict:
            meta = {}
            for k, v in c.items():
                if k == "description":
                    continue
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        meta[f"{k}_{sub_k}"] = sub_v
                else:
                    meta[k] = v
            return meta
        
        codes = [c.to_dict() for c in self.classification_system.codes]
        
        ids = [f"id_{c["code"]}" for c in codes]
        documents = [c["description"] for c in codes]
        metadatas = [flatten(c=c) for c in codes]
        
        self.store.create_collection(
            ids=ids,
            documents=documents,
            metadatas=metadatas # type: ignore
        )
        
    def search_category(
        self,
        query:str,
        k:int
    ):
        result = self.store.collection.query(
            query_texts=[query],
            n_results=k
        )
        if result is not None:
            output = result["metadatas"][0] # type: ignore
            docs = result["documents"][0] # type: ignore
            for o, d in zip(output, docs):
                o["description"] = d # type: ignore
            return output
        else:
            raise Exception("Error occured while searching for code")
    
    def forward(self, query:str, k:int):
        
        retrieved_context = self.search_category(query, k)
        retrieved_context_str = json.dumps(retrieved_context, indent=4, ensure_ascii=False)
        self.CoT(query, retrieved_context_str)
        
        
        
            
        

