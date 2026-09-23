from hcs4os.rag.BaseRAG import BaseRAGCoT
import os
import dspy
import json
from helpers import harmonize_code, load_train_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_id")
parser.add_argument("--embed_model_name")
parser.add_argument("--classification_name")
parser.add_argument("--collection_name")
parser.add_argument("--output_folder")
parser.add_argument("--k", type=int)
parser.add_argument("--include_hierarchy")
parser.add_argument("--path_test_data", default="/posit_share/home/montag-a/Adrian_Masterarbeit/data/training_data/rag_sub_sample2.json")
args = parser.parse_args()

print(args.model_id)
print(args.embed_model_name)
print(args.classification_name)
print(args.collection_name)
print(args.output_folder)
print(args.path_test_data)
print(args.include_hierarchy)
print(args.k)




train_data = load_train_split()
# Filtering out the relevant codes, so that we do not expose the model to the full classification index, only 
# those that are relevant for our classification. This removes a certain level of complexity, will be added
# while creating the vector db only those codes, will be indexed.
relevant_codes = list(set([example["label"] for example in train_data]))
relevant_codes = [harmonize_code(c) for c in relevant_codes]
del(train_data)


#with open("/posit_share/home/montag-a/Adrian_Masterarbeit/data/training_data/rag_sub_sample.json","r", encoding="utf-8") as f:
with open(args.path_test_data,"r", encoding="utf-8") as f:
    test_data = json.loads(f.read())

class SeaRAGChainOfThoughtSignature(dspy.Signature):
    """
    Classify a private-household income or expenditure item into a single SEA
    2021 code by reasoning over a set of pre-retrieved candidate codes, rather
    than relying on memorized codes.

    The inputs you will receive are **always** german language.

    The SEA 2021 (Systematik der Einnahmen und Ausgaben der privaten
    Haushalte, Statistisches Bundesamt) is a hierarchy of numeric codes across
    THREE parts: Abteilung 00 (Einnahmen / household income); Abteilungen
    01-15 (Verwendungszwecke des Individualkonsums / individual consumption,
    mirroring COICOP 2018; households 01-13, private non-profit institutions
    14, government 15); and Abteilung 16 (Ausgaben ohne Individualkonsum /
    non-consumption expenditure: taxes, social- and private-insurance
    contributions, membership fees, donations, loan repayment/interest, and
    formation of tangible/financial wealth). Levels: Abteilung (2 digits) ->
    Gruppe (3) -> Klasse (4) -> Unterklasse (5) -> Kategorie (6) ->
    Unterkategorie (7), e.g. 01 -> 011 -> 0111 -> 0111 1 -> 0111 10 ->
    0111 101, written with a space after the 4th digit. A valid final answer is
    a real code present in the retrieved candidates; prefer the most specific
    code that correctly covers the item.

    Reasoning protocol:
    1. Identify the essence of the item. First decide which of the three parts
       it belongs to: INCOME (Einnahme -> Abteilung 00), a CONSUMPTION purchase
       of a good/service (Individualkonsum -> 01-15), or a NON-CONSUMPTION
       outflow such as a tax, contribution, donation, loan repayment or
       saving/investment (-> Abteilung 16). This choice is the most
       consequential (e.g. an insurance premium is 16, a household appliance is
       05). Note any consumption detail that affects placement: home vs.
       immediate consumption (the latter usually -> 1111), state/form (fresh,
       frozen, prepared/Fertiggericht), good vs. service.
    2. Compare the item against every retrieved candidate's Bezeichnung and
       notes:
       - Read the **includes** notes to confirm a match.
       - The **inlcudes** section will give you most insight into what to classify into 
       given category.
    3. Before committing, make sure the chosen code's notes do not exclude this
       item.
    4. If no candidate fits specifically, choose the most appropriate
       (residual) candidate within the correct branch rather than guessing a code from a different branch.

    Rules:
    - Never invent or guess a code from memory. The final sea_code must be an
      exact code string that appears in retrieved_candidates, not a paraphrase
      or a made-up variant, formatted with the space after the 4th digit
      (e.g. '0111 1').
    - The target level for the classification of the item is **4** (the
      Klasse); always try to find a level-4 code among the candidates,
      but there my also be expenses where it is plausible to assign an level **3** (Gruppe).
      When a level **3** is more plausible than a level **4** code select the level 3 code.
    - When evidence is ambiguous, prefer the interpretation supported by the
      **includes** notes over intuition.
    """

    input_expense: str = dspy.InputField(
    desc="A private-household income or expenditure item to classify, e.g. a receipt line item, a Haushaltsbuch entry, or a short description of a purchased good/service or a source of income."
    )
    retrieved_candidates: str = dspy.InputField(
    desc="The pre-retrieved candidate SEA 2021 records most semantically similar to the item, each with its code, Bezeichnung (label), level, and 'Eingeschlossen'/'Ausgeschlossen' notes. Choose the final code only from these records."
    )
    sea_code: str = dspy.OutputField(
    desc="The single most specific SEA 2021 code from retrieved_candidates that correctly covers the item, exactly as given (e.g. '0111 101')."
    )
    explaination: str = dspy.OutputField(
    desc="Concise reasoning explaining which retrieved candidate was chosen and why, citing the Eingeschlossen/Ausgeschlossen notes that justified it, plus any rejected alternative."
    )

rag = BaseRAGCoT(
    embedding_model_name=args.embed_model_name,
    collection_name=args.collection_name,
    classification_name=args.classification_name,
    codes_to_include=relevant_codes,
    query_field="input_expense",
    context_field="retrieved_candidates",
    api_key=os.getenv("API_KEY"),
    api_base=os.getenv("API_BASE"),
    model_name=f"openai/{args.model_id}",
    signature=SeaRAGChainOfThoughtSignature,
    create_new_collection=False,
    include_hierarchy=bool(args.include_hierarchy)   
)
print("Creating eval dataset!")

test_dataset_dspy = [
    dspy.Example({
        "input_expense":example["anchor"],
        "retrieved_candidates":json.dumps(rag.search_category(query=example["anchor"], k=int(args.k)), indent=4, ensure_ascii=False),
        "sea_code":harmonize_code(example["label"])
    }).with_inputs("input_expense", "retrieved_candidates") for example in test_data
]

def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    return int(example.sea_code == pred.sea_code)



evaluate = dspy.Evaluate(
    devset=test_dataset_dspy,
    metric=metric,
    num_threads=4,
    display_table=True,
    display_progress=True
)

results = evaluate(rag.CoT)

output = {}
i = 0
for example, response, _ in results.results:
    output.update({
        i:{
            "example":example.toDict(),
            "response":response.toDict()
        }
    })

    output[i]["example"]["retrieved_candidates"] = json.loads(output[i]["example"]["retrieved_candidates"])

    i+=1
    
#with open(f"data/eval_rag0/{sys.argv[2]}/{model_id.replace("/", "_")}.json", "w", encoding="utf-8") as f:
with open(f"{args.output_folder}/{args.model_id.replace("/", "_")}.json", "w", encoding="utf-8") as f:

    f.write(
        json.dumps(output, indent=4, ensure_ascii=False)
    )

