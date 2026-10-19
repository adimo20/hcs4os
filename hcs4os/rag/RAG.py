from typing import Literal
import json
import dspy
from .BaseRAG import BaseRAGCoT

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
    
    
class SeaRAGChainOfThoughtSignature(dspy.Signature):
    """
    Classify a private-household income or expenditure item into a single SEA
    2021 code by reasoning over a set of pre-retrieved candidate codes, rather
    than relying on memorized codes.

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
       - Read the 'Eingeschlossen sind' (includes) notes to confirm a match.
       - Read the 'Ausgeschlossen sind' (excludes) notes carefully: they
         explicitly redirect items that look like they belong here but are
         classified elsewhere, and usually name the correct code in
         parentheses. If the redirected code is itself among the candidates,
         prefer it.
    3. Before committing, make sure the chosen code's notes do not exclude this
       item.
    4. If no candidate fits specifically, choose the most appropriate
       "Andere ... , a.n.g." (residual) candidate within the correct branch
       rather than guessing a code from a different branch.

    Rules:
    - Never invent or guess a code from memory. The final sea_code must be an
      exact code string that appears in retrieved_candidates, not a paraphrase
      or a made-up variant, formatted with the space after the 4th digit
      (e.g. '0111 101').
    - The target level for the classification of the item is **5** (the
      Unterklasse); always try to find a level-5 code among the candidates,
      descending to Kategorie/Unterkategorie (levels 6-7) only when the
      evidence clearly supports a more specific base unit.
    - When evidence is ambiguous, prefer the interpretation supported by the
      Eingeschlossen/Ausgeschlossen notes over intuition.
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
    

class RAGChainOfThought(BaseRAGCoT):
    
    CONFIGS = {
        "COICOP_2018":{
            "signature":CoicopRAGChainOfThoughtSignature,
            "query_field":"input_expense",
            "context_field":"retrieved_candidates"
        },
        "ICATUS_2016":{
            "signature":IcatusRAGChainOfThoughtSignature,
            "query_field":"input_activity",
            "context_field":"retrieved_candidates"
        },    
        "SEA_2021":{
            "signature":SeaRAGChainOfThoughtSignature,
            "query_field":"input_expense",
            "context_field":"retrieved_candidates"
        },
        "SEA_2021_REPHRASED":{
            "signature":SeaRAGChainOfThoughtSignature,
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
# Rewrite step: rephraser signatures
#
# The vector store only embeds each code's short `description` label
# (e.g. "Eggplants (aubergines), fresh or chilled", "Steinobstkonserven, z.B.
# Pflaumen-, ... Pfirsichkonserven", "Eating meals/snack"). Raw items (receipt
# lines, brands, diary shorthand) sit far from those labels in embedding space.
# The rephraser translates the item into the vocabulary of the labels; it does
# NOT classify. The original item still goes to the Read step unchanged.
# ---------------------------------------------------------------------------


class CoicopRephrasingSignature(dspy.Signature):
    """
    Rewrite a household expense into a search query that reads like a COICOP
    2018 category title, so that semantic search over those titles retrieves
    the right candidate codes. You are NOT classifying the expense: never
    output a code and never decide between categories.

    The search index contains only the short English titles of the COICOP
    codes, for example "Eggplants (aubergines), fresh or chilled",
    "Ice, ice cream and sorbets", "School uniforms", "Repair of electric
    appliances for personal care", "Life and accident insurance". Raw expense
    text — receipt abbreviations, brand names, shop wording, quantities — is
    far from these titles in embedding space. Your job is to translate the
    expense into the vocabulary of these titles.

    Rewriting protocol:
    1. Work out what was actually bought. Expand receipt shorthand and
       abbreviations (e.g. "BIO VOLLM 3,8%" -> organic whole milk) and replace
       brand or product names with the generic good or service they stand for
       (e.g. "Nutella" -> chocolate hazelnut spread; "Netflix" -> video
       streaming subscription service). Translate non-English input into
       English.
    2. Name it with the plain generic term an official classification would
       use. Where you are certain of the broader product group, put it first
       and the specific item after it (e.g. "vegetables, fresh or chilled:
       eggplant").
    3. Keep every attribute that changes where the item is classified:
       - state or form: fresh, chilled, frozen, dried, preserved/canned,
         ready-made meal;
       - good vs. service: purchase vs. repair, rental, hire, installation,
         maintenance, insurance, subscription, fee;
       - consumed at home vs. served in a restaurant, café or canteen;
       - new vs. second-hand;
       - intended user or purpose where stated (children, pets, vehicle,
         school, sport, medical use).
    4. Drop everything that does not change placement: brand, shop, quantity,
       pack size, price, date, promotional or marketing wording.

    Rules:
    - Output one short noun phrase (about 2-12 words) in English. No full
      sentences, no explanations, no code numbers, no "(ND)"/"(S)" suffixes.
    - Do not force a category word you are unsure of: a wrong broader term
      pulls retrieval into the wrong branch. When in doubt, describe the item
      itself generically and leave the category out.
    - Do not add facts that are not in, or clearly implied by, the input.
    - If the input is already a clean generic description, return it almost
      unchanged.
    """

    input_expense: str = dspy.InputField(
        desc="A raw household expense as it appears in the data, e.g. a receipt line item or a short free-text description of a purchased good or service."
    )
    rephrased_query: str = dspy.OutputField(
        desc="A short English noun phrase naming the generic good or service in the wording of a COICOP title, keeping placement-relevant attributes (form, good vs. service, at home vs. away, purpose). No codes."
    )


class IcatusRephrasingSignature(dspy.Signature):
    """
    Rewrite a time-use activity into a search query that reads like an ICATUS
    2016 activity title, so that semantic search over those titles retrieves
    the right candidate codes. You are NOT classifying the activity: never
    output a code and never decide between categories.

    The search index contains only the short English titles of the ICATUS
    codes, for example "Eating meals/snack", "Travelling and commuting for
    employment", "Agriculture, forestry, fishing and mining for own final
    use", "Providing paid business and professional services in households
    and household enterprises", "Travelling time related to unpaid volunteer,
    trainee and other unpaid work". These titles are organised less by WHAT
    the person did than by the ECONOMIC NATURE of the activity and FOR WHOM it
    was done. Raw diary entries ("did the dishes", "drove Tom to school",
    "Mathe-Nachhilfe gegeben") are far from these titles in embedding space.
    Your job is to translate the entry into their vocabulary.

    Rewriting protocol:
    1. Work out what the person was actually doing. Expand shorthand and
       translate non-English input into English.
    2. Make the economic nature explicit whenever the input states or clearly
       implies it: for pay or profit (employment, own business), for own
       final use (producing goods or services for one's own household),
       unpaid domestic service or unpaid caregiving for household or family
       members, unpaid volunteer work for others, learning, or a personal /
       leisure / self-care activity.
    3. Make the beneficiary explicit when stated: oneself, a child of the
       household, a dependent or non-dependent adult household or family
       member, another household, the community or an organisation.
    4. If the entry is travel, say that it is travelling and what it was
       related to (e.g. "travelling time related to childcare: taking child
       to school").
    5. Drop names of people and places, times, durations and other detail
       that does not change placement.

    Rules:
    - Output one short phrase (about 3-15 words) in English, in the style of
      an ICATUS title. No full sentences, no explanations, no code numbers.
    - Do not invent a pay status or beneficiary that is not stated or
      clearly implied; a wrong one pulls retrieval into the wrong major
      division. When unknown, describe the activity itself and leave it out.
    - If the input is already a clean generic description, return it almost
      unchanged.
    """

    input_activity: str = dspy.InputField(
        desc="A raw time-use activity as it appears in the data, e.g. a diary line item or a short free-text description of what a person spent time on."
    )
    rephrased_query: str = dspy.OutputField(
        desc="A short English phrase in the style of an ICATUS title, stating the activity and, where known, its economic nature (paid, own use, unpaid care, volunteering, leisure) and beneficiary. No codes."
    )


class SeaRephrasingSignature(dspy.Signature):
    """
    Rewrite a private-household income or expenditure item into a German
    search query that reads like an SEA 2021 Bezeichnung (category label), so
    that semantic search over those labels retrieves the right candidate
    codes. You are NOT classifying the item: never output a code and never
    decide between categories.

    The search index contains only the German labels (Bezeichnungen) of the
    SEA 2021 codes, for example "Steinobstkonserven, z.B. Pflaumen-,
    Mirabellen-, Kirsch-, Pfirsich- und Aprikosenkonserven", "Gartenmöbel
    (ohne Sonnenschirme ...), z.B. Gartenbänke, Balkonmöbel",
    "Versicherungsdienstleistungen", "Kredite für Hochzeiten, Familienfeiern
    u.Ä.", "Gastronomie- und Beherbergungsdienstleistungen". Many labels name
    a generic group followed by typical examples ("z.B. ..."). Raw items —
    receipt abbreviations, brand names, bank-statement text, quantities — are
    far from these labels in embedding space. Your job is to translate the
    item into their vocabulary.

    Rewriting protocol:
    1. Work out what the item is. Expand receipt and bank-statement shorthand
       (e.g. "BIO VOLLM 3,8%" -> Bio-Vollmilch; "LASTSCHRIFT ALLIANZ KFZ" ->
       Beitrag Kfz-Haftpflichtversicherung) and replace brand or product names
       with the generic good or service (e.g. "Nutella" ->
       Nuss-Nougat-Creme; "Netflix" -> Video-Streaming-Abonnement). Translate
       non-German input into German.
    2. Make the direction of the money flow explicit when it is clear:
       an Einnahme (income: Lohn/Gehalt, Rente, Kindergeld, Mieteinnahme,
       Zinsen, ...), a Konsumausgabe for a good or service, or an Ausgabe
       ohne Konsum (Steuer, Sozialversicherungs- oder Versicherungsbeitrag,
       Mitgliedsbeitrag, Spende, Tilgung oder Zinsen eines Kredits,
       Sparen/Geldanlage). Use the wording of that area (e.g.
       "Beitrag zur ...", "Einnahmen aus ...", "Tilgung ...").
    3. Name it with the generic term an official German classification would
       use. Where you are certain of the broader group, write it first and
       the specific item after it, mirroring the "Gruppe, z.B. Artikel"
       pattern of the labels (e.g. "Steinobstkonserven, Pfirsichkonserve").
    4. Keep every attribute that changes placement:
       - Zustand/Form: frisch, gekühlt, tiefgefroren, getrocknet, Konserve,
         Fertiggericht;
       - Ware vs. Dienstleistung: Kauf vs. Reparatur, Miete, Leihe,
         Installation, Wartung, Gebühr, Abonnement;
       - Verzehr zu Hause vs. im Restaurant, Café, Imbiss oder in der Kantine;
       - neu vs. gebraucht;
       - Verwendungszweck oder Nutzer, wenn genannt (Kinder, Haustier,
         Fahrzeug, Schule, Sport, Gesundheit).
    5. Drop everything that does not change placement: brand, shop, quantity,
       pack size, price, date, IBAN, reference numbers, promotional wording.

    Rules:
    - Output one short German noun phrase (about 2-12 words). No full
      sentences, no explanations, no code numbers.
    - Do not force a group word or a money-flow direction you are unsure of:
      a wrong one pulls retrieval into the wrong Abteilung. When in doubt,
      describe the item itself generically and leave it out.
    - Do not add facts that are not in, or clearly implied by, the input.
    - If the input is already a clean generic German description, return it
      almost unchanged.
    """

    input_expense: str = dspy.InputField(
        desc="A raw private-household income or expenditure item as it appears in the data, e.g. a receipt line, a Haushaltsbuch entry, a bank-statement line, or a short free-text description."
    )
    rephrased_query: str = dspy.OutputField(
        desc="A short German noun phrase in the wording of an SEA 2021 Bezeichnung, keeping placement-relevant attributes (money-flow direction, form, good vs. service, at home vs. away, purpose). No codes."
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
        classification_name: str,
    ) -> None:
        super().__init__()
        self.lm = lm
        self.mapping = {
            "COICOP_2018": CoicopRephrasingSignature,
            "ICATUS_2016": IcatusRephrasingSignature,
            "SEA_2021": SeaRephrasingSignature,
            "SEA_2021_REPHRASED": SeaRephrasingSignature,
        }
        assert classification_name in self.mapping, (
            f"No rephrasing signature for '{classification_name}'"
        )

        self.classification_name = classification_name
        self.signature = self.mapping[classification_name]
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
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.rephrasing_module = QueryRephraser(
            self.lm,
            classification_name=self.classification_name,
        )

    
    def forward(self, query: str, k: int, where: dict | None = None):
        rephrased_query = self.rephrasing_module(query=query).rephrased_query
        retrieved_context = self.search_category(rephrased_query, k, where=where)
        retrieved_context_str = json.dumps(
            retrieved_context, indent=4, ensure_ascii=False
        )
        prediction = self.CoT(**{
            self.query_field: query,
            self.context_field: retrieved_context_str,
        })
        # Keep the rewritten query for logging and info later
        prediction.rephrased_query = rephrased_query
        return prediction