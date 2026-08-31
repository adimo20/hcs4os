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
    }
}
