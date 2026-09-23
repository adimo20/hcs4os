from hcs4os.agents.RAG.BaseAgentRAG import BaseRAGAgent
import os
import dspy
from helpers import harmonize_code, load_train_split
from dotenv import load_dotenv
import json
import argparse
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--model_id")
parser.add_argument("--embed_model_name")
parser.add_argument("--classification_name")
parser.add_argument("--collection_name")
parser.add_argument("--output_folder")
parser.add_argument("--include_hierarchy")
parser.add_argument("--path_test_data", default="/posit_share/home/montag-a/Adrian_Masterarbeit/data/training_data/rag_sub_sample2.json")


args = parser.parse_args()

print(args.model_id)
print(args.embed_model_name)
print(args.classification_name)
print(args.collection_name)
print(args.output_folder)
print(args.include_hierarchy)
print(args.path_test_data)


train_data = load_train_split()
# Filtering out the relevant codes, so that we do not expose the model to the full classification index, only 
# those that are relevant for our classification. This removes a certain level of complexity, will be added
# while creating the vector db only those codes, will be indexed.
relevant_codes = list(set([example["label"] for example in train_data]))
relevant_codes = [harmonize_code(c) for c in relevant_codes]
del(train_data)


with open(args.path_test_data,"r", encoding="utf-8") as f:
    test_data = json.loads(f.read())

class SeaRAGSearchAgentSignature(dspy.Signature):
    """"
    Classify a private-household income or expenditure item into a single SEA
    2021 code by semantically retrieving candidate codes with the search tool.

    The inputs you will receive are **always** german language.


    The SEA 2021 (Systematik der Einnahmen und Ausgaben der privaten
    Haushalte, Statistisches Bundesamt) is a hierarchy of numeric codes across
    THREE parts:
    - Abteilung 00: Einnahmen der privaten Haushalte (household income).
    - Abteilungen 01-15: Verwendungszwecke des Individualkonsums (individual
    consumption; mirrors COICOP 2018). Households = 01-13, private
    non-profit institutions = 14, government = 15.
    - Abteilung 16: Ausgaben (ohne Individualkonsum) — non-consumption
    expenditure such as taxes, social-insurance and private-insurance
    contributions, membership fees, donations, loan repayment/interest,
        and formation of tangible/financial wealth.
    Codes are written witha SPACE after the 4th digit (e.g. "0111 1").
    A valid final answer is a real code that exists in the system; prefer 
    the most specific code that correctly covers the item.

    Tool:
    - search_category(query, k): retrieve the k SEA codes whose descriptions
        are most semantically similar to the query. Each result comes with its
        description and an includes notes.

    Reasoning protocol:
    1. Identify the essence of the item. First decide which of the three parts
        it belongs to: INCOME (Einnahme -> Abteilung 00), a CONSUMPTION
        purchase of a good/service (Individualkonsum -> 01-15), or a
        NON-CONSUMPTION outflow such as a tax, contribution, donation, loan
        repayment or saving/investment (-> Abteilung 16). This choice is the
        most consequential (e.g. an insurance PREMIUM is 16, a household
        appliance is 05). Note any consumption detail that drives placement:
        home vs. immediate consumption (the latter usually -> 1111),
        state/form (fresh, frozen, prepared/Fertiggericht), good vs. service.
    2. Call search_category with a focused query describing the item. Bias the
        query toward the part identified in step 1, and use a k large enough to
        see several plausible candidates (e.g. 5-10). Use German query terms only 
        because they match the classification's language best.
    3. Read each candidate's **description** to judge whether it plausibly
          covers the item; the description states which goods and services the
          code includes, so weigh the fit against it rather than the
          Bezeichnung (label) alone, which is often broad.
        - Shortlist the candidates whose descriptions genuinely cover the item,
          then choose the one that covers it most specifically. Where two
          candidates fit, prefer the one at the target level (see Rules) and the
          one in the part identified in step 1.
        - Reject a candidate whose description points to a different kind of
          item even if its label looks similar, a described mismatch outweighs
          a superficial label match.
          refine the query (different wording, the redirected category, or a broader/narrower term) and 
          search again — even if the first result already looks correct, use the second search to confirm or challenge it.
    4. Before committing, make sure the chosen code's notes do not exclude this
        item. If they do, search again for the redirected code.
    5. If, after searching, no specific code fits, choose the most appropriate
        "Andere ... , a.n.g." (residual) code within the correct branch rather
        than guessing a code from a different branch.

    Rules:
    - Never invent or guess a code from memory. Every code in the final answer
        and in your reasoning must have been returned by the search tool.
    - The final sea_code must be an exact code string that the tool returned,
        formatted with the space after the 4th digit (e.g. "0111 1"), not a
        paraphrase or a made-up variant.
    - The target level for the classification of the item is **5-digits**;
      always try to find a 5-digit code among the candidates, but there my also 
      be expenses where it is plausible to assign an **4-digits** or **3-digits** code.
    - When evidence is ambiguous, prefer the interpretation supported by the
        includes notes over intuition.

    Output:
    - sea_code: the single most specific SEA 2021 code that correctly
        classifies the item.
    - explaination: a concise justification naming the decisive retrieved
        candidate and the includes note(s) that determined
        the choice, including any candidate you rejected and why.
    """    
    input_expense: str = dspy.InputField(
    desc="A private-household income or expenditure item to classify, e.g. a receipt line item, a Haushaltsbuch entry, or a short description of a purchased good/service or a source of income."
    )
    sea_code: str = dspy.OutputField(
    desc="The single most specific SEA 2021 code that correctly covers the item, exactly as returned by the tools (e.g. '0111 1'). Written with a space after the 4th digit."
    )
    explaination: str = dspy.OutputField(
    desc="Concise reasoning explaining which retrieved candidate was chosen and why, citing the includes notes that justified it, plus any rejected alternative."
    )


search_category_doc = """
    Semantically retrieve the classification codes most similar to a query.

    Use this to find candidate SEA 2021 codes for an item: given a natural
    language description, it returns the k codes whose descriptions are closest
    in the vector space, each with its full record so you can compare
    candidates and pick the best match. German queries match the
    classification's language best.

    Args:
        query: A natural language description of the item to classify,
        e.g. "frisches Brot vom Bäcker" or "Beitrag zur Krankenversicherung".
        k: The number of candidate codes to retrieve, e.g. 5.

    Returns:
        A list of up to k records, each a dict with the code, its descriptionand 
        and an 'includes' notes that disambiguate what belongs under it.
    """


test_dataset_dspy = [
    dspy.Example({
        "input_expense":example["anchor"],
        "sea_code":harmonize_code(example["label"])
    }).with_inputs("input_expense") for example in test_data
]


def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    return int(example.sea_code == pred.sea_code)


rag = BaseRAGAgent(
    embedding_model_name=args.embed_model_name,
    collection_name=args.collection_name,
    classification_name=args.classification_name,
    api_key=os.getenv("API_KEY"),
    api_base=os.getenv("API_BASE"),
    model_name=f"openai/{args.model_id}",
    signature=SeaRAGSearchAgentSignature, 
    search_category_tool_doc=search_category_doc,
    create_new_collection=False,
    include_hierarchy=bool(args.include_hierarchy)   
       
)

evaluate = dspy.Evaluate(
    devset=test_dataset_dspy,
    metric=metric,
    num_threads=4,
    display_table=True,
    display_progress=True
)

results = evaluate(rag.agent)


output = {}
i = 0
for example, response, _ in results.results:
    output.update({
        i:{
            "example":example.toDict(),
            "response":response.toDict()
        }
    })
    i+=1
    
with open(f"{args.output_folder}/{args.model_id.replace("/", "_")}.json", "w", encoding="utf-8") as f:
    f.write(
        json.dumps(output, indent=4, ensure_ascii=False)
    )
