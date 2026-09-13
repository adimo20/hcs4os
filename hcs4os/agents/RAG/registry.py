import dspy


class CoicopRAGSearchAgentSignature(dspy.Signature):

    input_expense: str = dspy.InputField(
        desc="A household expense to classify, e.g. a receipt line item or a short description of a purchased good or service."
    )
    coicop_code: str = dspy.OutputField(
        desc="The single most specific COICOP code that correctly covers the expense, exactly as returned by the tools (e.g. '01.1.1.1')."
    )
    explaination: str = dspy.OutputField(
        desc="Concise reasoning explaining which retrieved candidate was chosen and why, citing the includes/excludes notes that justified it, plus any rejected alternative."
    )


class IcatusRAGSearchAgentSignature(dspy.Signature):

    input_activity: str = dspy.InputField(
        desc="A time-use activity to classify, e.g. a diary line item or a short description of an activity a person spent time on."
    )
    icatus_code: str = dspy.OutputField(
        desc="The single most specific ICATUS code that correctly covers the activity, exactly as returned by the tools (e.g. '110')."
    )
    explaination: str = dspy.OutputField(
        desc="Concise reasoning explaining which retrieved candidate was chosen and why, citing the includes/excludes notes that justified it, plus any rejected alternative."
    )


class SeaRAGSearchAgentSignature(dspy.Signature):

    input_expense: str = dspy.InputField(
        desc="A private-household income or expenditure item to classify, e.g. a receipt line item, a Haushaltsbuch entry, or a short description of a purchased good/service or a source of income."
    )
    sea_code: str = dspy.OutputField(
        desc="The single most specific SEA 2021 code that correctly covers the item, exactly as returned by the tools (e.g. '0111 101'). Written with a space after the 4th digit."
    )
    explaination: str = dspy.OutputField(
        desc="Concise reasoning explaining which retrieved candidate was chosen and why, citing the Eingeschlossen/Ausgeschlossen notes that justified it, plus any rejected alternative."
    )

tool_descriptions = {
    "COICOP_2018": {
        "system_prompt": """
        Classify a household expense into a single COICOP code by semantically
        retrieving candidate codes with the search tool, then reasoning over the
        retrieved records to pick the best match, rather than relying on
        memorized codes.

        COICOP is a hierarchy of codes (division -> group -> class -> subclass),
        e.g. 01 -> 01.1 -> 01.1.1 -> 01.1.1.1. A valid final answer is a real
        code that exists in the system; prefer the most specific code that
        correctly covers the expense.

        Tool:
        - search_category(query, k): retrieve the k COICOP codes whose
            descriptions are most semantically similar to the query. Each result
            comes with its description and includes/alsoIncludes/excludes notes.

        Reasoning protocol:
        1. Identify the essence of the expense: what good or service was actually
            purchased. Note any detail that affects placement (state/form of the
            item, purpose, whether it is a good vs. a service).
        2. Call search_category with a focused query describing the expense. Use a
            k large enough to see several plausible candidates (e.g. 5-10).
        3. Compare the expense against every retrieved candidate's description and
            notes:
            - Read the 'includes' and 'alsoIncludes' notes to confirm a match.
            - Read the 'excludes' notes carefully: they explicitly redirect items
                that look like they belong here but are classified elsewhere, and
                usually name the correct code.
        4. If no retrieved candidate fits well, or the excludes notes point
            elsewhere, refine the query (different wording, the redirected
            category, or a broader/narrower term) and search again.
        5. Before committing, make sure the chosen code's notes do not exclude
            this expense. If they do, search again for the redirected code.
        6. If, after searching, no specific code fits, choose the most appropriate
            "other" / residual code within the correct branch rather than guessing
            a code from a different branch.

        Rules:
        - Never invent or guess a code from memory. Every code in the final answer
            and in your reasoning must have been returned by the search tool.
        - The final coicop_code must be an exact code string that the tool
            returned (e.g. "01.1.1.1"), not a paraphrase or a made-up variant.
        - The target level for the classification of the expense is **4**;
            always try to find a code from level **4**.
        - When evidence is ambiguous, prefer the interpretation supported by the
            includes/excludes notes over intuition.

        Output:
        - coicop_code: the single most specific COICOP code that correctly
            classifies the expense.
        - explaination: a concise justification naming the decisive retrieved
            candidate and the includes/excludes note(s) that determined the
            choice, including any candidate you rejected and why.
        """,

        "search_category": """
        Semantically retrieve the classification codes most similar to a query.

        Use this to find candidate COICOP codes for an expense: given a natural
        language description, it returns the k codes whose descriptions are
        closest in the vector space, each with its full record so you can compare
        candidates and pick the best match.

        Args:
            query: A natural language description of the expense to classify,
                e.g. "fresh bread from a bakery".
            k: The number of candidate codes to retrieve, e.g. 5.

        Returns:
            A list of up to k records, each a dict with the code, its
            description, level, and the 'includes' / 'alsoIncludes' / 'excludes'
            notes that disambiguate what belongs under it.
        """,
        "signature": CoicopRAGSearchAgentSignature
    },
    "ICATUS_2016": {
        "system_prompt": """
        Classify a time-use activity into a single ICATUS 2016 code by
        semantically retrieving candidate codes with the search tool, then
        reasoning over the retrieved records to pick the best match, rather than
        relying on memorized codes.

        ICATUS 2016 (International Classification of Activities for Time-Use
        Statistics) is a hierarchy of codes (major division -> division ->
        group), e.g. 1 -> 11 -> 110. A valid final answer is a real code that
        exists in the system; prefer the most specific code that correctly covers
        the activity.

        Tool:
        - search_category(query, k): retrieve the k ICATUS codes whose titles are
            most semantically similar to the query. Each result comes with its
            title and includes/excludes/examples notes.

        Reasoning protocol:
        1. Identify the essence of the activity: what the person was actually
            doing. Note any detail that affects placement (whether it was done for
            pay or profit, for own final use, as an unpaid service for the
            household, as volunteering, or as a personal activity; and for whom
            the activity was performed).
        2. Call search_category with a focused query describing the activity. Use
            a k large enough to see several plausible candidates (e.g. 5-10).
        3. Compare the activity against every retrieved candidate's title and
            notes:
            - Read the 'includes' and 'examples' notes to confirm a match.
            - Read the 'excludes' notes carefully: they explicitly redirect
                activities that look like they belong here but are classified
                elsewhere, and usually name the correct code.
        4. If no retrieved candidate fits well, or the excludes notes point
            elsewhere, refine the query (different wording, the redirected
            category, or a broader/narrower term) and search again.
        5. Before committing, make sure the chosen code's notes do not exclude
            this activity. If they do, search again for the redirected code.
        6. If, after searching, no specific code fits, choose the most appropriate
            "other" / residual code within the correct branch rather than guessing
            a code from a different branch.

        Rules:
        - Never invent or guess a code from memory. Every code in the final answer
            and in your reasoning must have been returned by the search tool.
        - The final icatus_code must be an exact code string that the tool
            returned (e.g. "110"), not a paraphrase or a made-up variant.
        - The target level for the classification of the activity is **3**;
            always try to find a code from level **3**.
        - When evidence is ambiguous, prefer the interpretation supported by the
            includes/excludes notes over intuition.

        Output:
        - icatus_code: the single most specific ICATUS code that correctly
            classifies the activity.
        - explaination: a concise justification naming the decisive retrieved
            candidate and the includes/excludes note(s) that determined the
            choice, including any candidate you rejected and why.
        """,

        "search_category": """
        Semantically retrieve the classification codes most similar to a query.

        Use this to find candidate ICATUS codes for an activity: given a natural
        language description, it returns the k codes whose titles are closest in
        the vector space, each with its full record so you can compare candidates
        and pick the best match.

        Args:
            query: A natural language description of the activity to classify,
                e.g. "preparing dinner for the household".
            k: The number of candidate codes to retrieve, e.g. 5.

        Returns:
            A list of up to k records, each a dict with the code, its title,
            level, and the 'includes' / 'excludes' / 'examples' notes that
            disambiguate what belongs under it.
        """,
        "signature": IcatusRAGSearchAgentSignature
    },
    "SEA_2021": {
        "system_prompt": """
        Classify a private-household income or expenditure item into a single SEA
        2021 code by semantically retrieving candidate codes with the search tool,
        then reasoning over the retrieved records to pick the best match, rather
        than relying on memorized codes.

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
        Levels: Abteilung (2 digits) -> Gruppe (3) -> Klasse (4) ->
        Unterklasse (5) -> Kategorie (6) -> Unterkategorie (7), e.g.
        01 -> 011 -> 0111 -> 0111 1 -> 0111 10 -> 0111 101. Codes are written with
        a SPACE after the 4th digit (e.g. "0111 101"). A valid final answer is a
        real code that exists in the system; prefer the most specific code that
        correctly covers the item.

        Tool:
        - search_category(query, k): retrieve the k SEA codes whose descriptions
            are most semantically similar to the query. Each result comes with its
            Bezeichnung (label) and Eingeschlossen/Ausgeschlossen notes.

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
            see several plausible candidates (e.g. 5-10). German query terms match
            the classification's language best.
        3. Compare the item against every retrieved candidate's Bezeichnung and
            notes:
            - Read 'Eingeschlossen sind' (includes) to confirm a match.
            - Read 'Ausgeschlossen sind' (excludes) carefully: these notes
                explicitly redirect items that look like they belong here but are
                classified elsewhere, naming the correct code in parentheses.
        4. If no retrieved candidate fits well, or the Ausgeschlossen notes point
            elsewhere, refine the query (different wording, the redirected
            category, or a broader/narrower term) and search again.
        5. Before committing, make sure the chosen code's notes do not exclude this
            item. If they do, search again for the redirected code.
        6. If, after searching, no specific code fits, choose the most appropriate
            "Andere ... , a.n.g." (residual) code within the correct branch rather
            than guessing a code from a different branch.

        Rules:
        - Never invent or guess a code from memory. Every code in the final answer
            and in your reasoning must have been returned by the search tool.
        - The final sea_code must be an exact code string that the tool returned,
            formatted with the space after the 4th digit (e.g. "0111 101"), not a
            paraphrase or a made-up variant.
        - The target level for the classification is **5** (the Unterklasse), the
            depth targeted by the EVS from 2023; always try to reach a level-**5**
            code, descending to Kategorie/Unterkategorie (levels 6-7) only when the
            evidence clearly supports a more specific base unit.
        - When evidence is ambiguous, prefer the interpretation supported by the
            Eingeschlossen/Ausgeschlossen notes over intuition.

        Output:
        - sea_code: the single most specific SEA 2021 code that correctly
            classifies the item.
        - explaination: a concise justification naming the decisive retrieved
            candidate and the Eingeschlossen/Ausgeschlossen note(s) that determined
            the choice, including any candidate you rejected and why.
        """,

        "search_category": """
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
            A list of up to k records, each a dict with the code, its Bezeichnung
            (label), level, and the 'Eingeschlossen' / 'Ausgeschlossen' notes that
            disambiguate what belongs under it.
        """,
        "signature": SeaRAGSearchAgentSignature
    }
}
