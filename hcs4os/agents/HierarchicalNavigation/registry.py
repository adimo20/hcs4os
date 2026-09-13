import dspy


class CoicopHierarchicalSearchAgentSignature(dspy.Signature):
    
    input_expense: str = dspy.InputField(
        desc="A household expense to classify, e.g. a receipt line item or a short description of a purchased good or service."
    )
    coicop_code: str = dspy.OutputField(
        desc="The single most specific COICOP code that correctly covers the expense, exactly as returned by the tools (e.g. '01.1.1.1')."
    )
    explaination: str = dspy.OutputField(
        desc="Concise reasoning tracing the top-down path to the code and citing the includes/excludes notes that justified it, plus any rejected alternative."
    )    
    

class IcatusHierarchicalSearchAgentSignature(dspy.Signature):
    
    input_activity: str = dspy.InputField(
        desc="A time-use activity to classify, e.g. a diary line item or a short description of an activity a person spent time on."
    )
    icatus_code: str = dspy.OutputField(
        desc="The single most specific ICATUS code that correctly covers the activity, exactly as returned by the tools (e.g. '110')."
    )
    explaination: str = dspy.OutputField(
        desc="Concise reasoning tracing the top-down path to the code and citing the includes/excludes notes that justified it, plus any rejected alternative."
    )    
 
class NafHierarchicalSearchAgentSignature(dspy.Signature):

    input_activity: str = dspy.InputField(
        desc="An economic activity to classify, e.g. a short description of what a business or establishment does (its main activity)."
    )
    naf_code: str = dspy.OutputField(
        desc="The single most specific NAF code that correctly covers the activity, exactly as returned by the tools (e.g. '10.11A')."
    )
    explaination: str = dspy.OutputField(
        desc="Concise reasoning tracing the top-down path to the code and citing the includes/excludes notes that justified it, plus any rejected alternative."
    )

class SeaHierarchicalSearchAgentSignature(dspy.Signature):

    input_expense: str = dspy.InputField(
        desc="A household income or expenditure item to classify, e.g. a receipt line item, a Haushaltsbuch entry, or a short description of a purchased good/service or a source of income."
    )
    sea_code: str = dspy.OutputField(
        desc="The single most specific SEA 2021 code that correctly covers the item, exactly as returned by the tools (e.g. '0111 101'). Written with a space after the 4th digit."
    )
    explaination: str = dspy.OutputField(
        desc="Concise reasoning tracing the top-down path to the code and citing the Eingeschlossen/Ausgeschlossen notes that justified it, plus any rejected alternative."
    )

tool_descriptions = {
    "COICOP_2018":{
        "system_prompt":"""
        Classify a household expense into a single COICOP code by searching the
        classification hierarchy top-down, using the available tools rather than
        relying on memorized codes.

        COICOP is a tree. The top level is a set of divisions (e.g. "01" Food and
        non-alcoholic beverages), and each code has children one level deeper
        (division -> group -> class -> subclass), e.g. 01 -> 01.1 -> 01.1.1 ->
        01.1.1.1. A valid final answer is a real code that exists in the system;
        prefer the most specific (deepest, leaf) code that correctly covers the
        expense.

        Tools:
        - get_root_categories_tool(): list the top-level divisions. Start here.
        - get_children_tool(code): list the direct child codes one level down, each
            with its description and includes/excludes notes.
        - get_code_tool(code): read one code's full record (description, elaborated
            definition, and 'includes' / 'alsoIncludes' / 'excludes' notes).
        - get_parent_tool(code): move one level up to re-read broader context or
            reconsider a branch.

        Reasoning protocol:
        1. Identify the essence of the expense: what good or service was actually
            purchased. Note any detail that affects placement (state/form of the
            item, purpose, whether it is a good vs. a service).
        2. Call get_root_categories() and choose the single best-fitting division.
            If two divisions seem plausible, note the alternative to revisit later.
        3. Descend one level at a time with get_children on the current code.
            At each level:
            - Compare the expense against every child's description and notes.
            - Read the 'includes' and 'alsoIncludes' notes to confirm a match.
            - Read the 'excludes' notes carefully: they explicitly redirect
                items that look like they belong here but are classified
                elsewhere, and usually name the correct code. Follow those
                pointers instead of forcing a fit.
            - Pick the best-matching child and repeat.
        4. Continue descending until you reach a leaf (get_children returns an
            empty list) or until no deeper code fits better than the current one.
        5. Before committing, verify the chosen code with get_code and check that
            its notes do not exclude this expense. If they do, backtrack using
            get_parent or return to a noted alternative branch and search again.
        6. If, after searching, no specific code fits, choose the most appropriate
            "other" / residual code within the correct branch rather than guessing
            a code from a different branch.

        Rules:
        - Never invent or guess a code from memory. Every code in the final answer
            and in your reasoning must have been returned by a tool.
        - The final coicop_code must be an exact code string that the tools
            returned (e.g. "01.1.1.1"), not a paraphrase or a made-up variant.
        - The target level for the classification of the expense is **4**,
        always try to find a code from level **4** the level of a code
        is returned by get_code.
        - When evidence is ambiguous, prefer the interpretation supported by the
            includes/excludes notes over intuition.      

        Output:
        - coicop_code: the single most specific COICOP code that correctly
            classifies the expense.
        - explaination: a concise justification tracing the path taken (division
            -> ... -> final code) and citing the decisive includes/excludes note(s)
            that determined the choice, including any branch you rejected and why.
        """,
        
        "get_children":"""
        Retrieve the full record for a single classification code.

        Use this to inspect one specific code in detail — its description,
        level in the hierarchy, and (for leaf/lower levels) an elaborated
        description plus 'includes', 'alsoIncludes' and 'excludes' notes that
        disambiguate what belongs under it. The 'excludes' notes are especially
        useful: they point to the correct sibling or related code when an item
        looks like it fits here but doesn't.

        Args:
            code: The classification code, e.g. "01.1.1".

        Returns:
            A Code object with fields:
                - code: the code string (e.g. "01.1.1")
                - description: short label (e.g. "Cereals and cereal products (ND)")
                - level: depth in the hierarchy (e.g. 2 = group, 3, 4 = subclass)
                - detailled_description: longer prose definition (may be empty)
                - details: dict with 'includes', 'alsoIncludes', 'excludes' notes
                    (may be empty strings at higher levels)
        """,
        
        "get_code":"""
        List the direct child codes one level below the given code.

        Use this to drill down the hierarchy: given a code you've decided the
        item falls under, this returns the candidate sub-codes to choose from
        next. Each child comes with its full record (description, includes/
        excludes notes), so you can compare siblings and pick the best match
        without additional lookups. Returns an empty list for leaf codes that
        have no children.

        Args:
            code: The parent classification code, e.g. "01.1.1".

        Returns:
            A list of Code objects for the direct children (e.g. "01.1.1.1",
            "01.1.1.2", ...), each with the same fields as get_code returns.
        """,
        "get_parent":"""
        Retrieve the immediate parent (one level up) of the given code.

        Use this to move back up the hierarchy — to reconsider a branch, read
        the broader category's definition for context, or verify that a code
        sits under the intended higher-level group. Returns the parent's full
        record.

        Args:
            code: The classification code whose parent you want, e.g. "01.1.1".

        Returns:
            A Code object for the parent (e.g. "01.1"), with the same fields as
            get_code returns. The scope note on a parent often summarises what
            the whole branch covers and excludes.""",
        "get_root_categories":"""
        List the top-level divisions of the classification system.

        Use this as the entry point when starting a classification: it returns
        the broadest categories (e.g. COICOP divisions 01–15), from which you
        select the most appropriate branch and then descend using get_children.

        Returns:
            A list of dicts, each with:
                - code: the top-level code (e.g. "01")
                - description: its label (e.g. "Food and non-alcoholic beverages")
        """,
        "signature":CoicopHierarchicalSearchAgentSignature
    },
    "ICATUS_2016":{
        "system_prompt":"""
        Classify a time-use activity into a single ICATUS 2016 code by searching
        the classification hierarchy top-down, using the available tools rather
        than relying on memorized codes.

        ICATUS 2016 (International Classification of Activities for Time-Use
        Statistics) is a tree. The top level is a set of nine major divisions
        (e.g. "1" Employment and related activities, "9" Self-care and
        maintenance), and each code has children one level deeper
        (major division -> division -> group), e.g. 1 -> 11 -> 110. A valid final
        answer is a real code that exists in the system; prefer the most specific
        (deepest, leaf) code that correctly covers the activity.

        Tools:
        - get_root_categories_tool(): list the top-level major divisions. Start here.
        - get_children_tool(code): list the direct child codes one level down, each
            with its title and includes/excludes notes.
        - get_code_tool(code): read one code's full record (title, definition, and
            'includes' / 'excludes' / 'examples' notes).
        - get_parent_tool(code): move one level up to re-read broader context or
            reconsider a branch.

        Reasoning protocol:
        1. Identify the essence of the activity: what the person was actually
            doing. Note any detail that affects placement (whether it was done for
            pay or profit, for own final use, as an unpaid service for the
            household, as volunteering, or as a personal activity; and for whom the
            activity was performed).
        2. Call get_root_categories() and choose the single best-fitting major
            division. If two major divisions seem plausible, note the alternative
            to revisit later.
        3. Descend one level at a time with get_children on the current code.
            At each level:
            - Compare the activity against every child's title and notes.
            - Read the 'includes' and 'examples' notes to confirm a match.
            - Read the 'excludes' notes carefully: they explicitly redirect
                activities that look like they belong here but are classified
                elsewhere, and usually name the correct code. Follow those
                pointers instead of forcing a fit.
            - Pick the best-matching child and repeat.
        4. Continue descending until you reach a leaf (get_children returns an
            empty list) or until no deeper code fits better than the current one.
        5. Before committing, verify the chosen code with get_code and check that
            its notes do not exclude this activity. If they do, backtrack using
            get_parent or return to a noted alternative branch and search again.
        6. If, after searching, no specific code fits, choose the most appropriate
            "other" / residual code within the correct branch rather than guessing
            a code from a different branch.

        Rules:
        - Never invent or guess a code from memory. Every code in the final answer
            and in your reasoning must have been returned by a tool.
        - The final icatus_code must be an exact code string that the tools
            returned (e.g. "110"), not a paraphrase or a made-up variant.
        - The target level for the classification of the activity is **3**,
        always try to find a code from level **3** the level of a code
        is returned by get_code.
        - When evidence is ambiguous, prefer the interpretation supported by the
            includes/excludes notes over intuition.      

        Output:
        - icatus_code: the single most specific ICATUS code that correctly
            classifies the activity.
        - explaination: a concise justification tracing the path taken (major
            division -> ... -> final code) and citing the decisive includes/excludes
            note(s) that determined the choice, including any branch you rejected
            and why.
        """,
        "get_children":"""
        List the direct child codes one level below the given code.

        Use this to drill down the hierarchy: given a code you've decided the
        activity falls under, this returns the candidate sub-codes to choose
        from next. Each child comes with its full record (title, definition,
        includes/excludes notes), so you can compare siblings and pick the best
        match without additional lookups. Returns an empty list for group-level
        (leaf) codes that have no children.

        The hierarchy has three levels: major division (1 digit, e.g. "1") ->
        division (2 digits, e.g. "11") -> group (3 digits, e.g. "110").

        Args:
            code: The parent classification code, e.g. "11".

        Returns:
            A list of Code objects for the direct children (e.g. "110",
            "111", ...), each with the same fields as get_code returns.
        """,
        "get_code":"""
        Retrieve the full record for a single classification code.

        Use this to inspect one specific code in detail — its title, level in
        the hierarchy, and the definition plus 'includes', 'excludes' and
        'examples' notes that disambiguate what belongs under it. The
        'excludes' notes are especially useful: they point to the correct
        sibling or related code when an activity looks like it fits here but
        doesn't.

        The hierarchy has three levels: major division (1 digit, e.g. "1") ->
        division (2 digits, e.g. "11") -> group (3 digits, e.g. "110").

        Args:
            code: The classification code, e.g. "110".

        Returns:
            A Code object with fields:
                - code: the code string (e.g. "110")
                - description: short label / title (e.g. "Employment in
                    corporations, government and non-profit institutions")
                - level: depth in the hierarchy (1 = major division,
                    2 = division, 3 = group)
                - detailled_description: longer prose definition (may be empty)
                - details: dict with 'includes', 'excludes' and 'examples' notes
                    (may be empty strings at higher levels)
        """,
        "get_parent":"""
        Retrieve the immediate parent (one level up) of the given code.

        Use this to move back up the hierarchy — to reconsider a branch, read
        the broader category's definition for context, or verify that a code
        sits under the intended higher-level division. Returns the parent's
        full record.

        The hierarchy has three levels: major division (1 digit, e.g. "1") ->
        division (2 digits, e.g. "11") -> group (3 digits, e.g. "110").

        Args:
            code: The classification code whose parent you want, e.g. "110".

        Returns:
            A Code object for the parent (e.g. "11"), with the same fields as
            get_code returns. The definition on a parent often summarises what
            the whole branch covers and excludes.
        """,
        "get_root_categories":"""
        List the top-level major divisions of the classification system.

        Use this as the entry point when starting a classification: it returns
        the broadest categories (the nine ICATUS 2016 major divisions, coded
        1-9, e.g. "1" Employment and related activities, "9" Self-care and
        maintenance), from which you select the most appropriate branch and
        then descend using get_children.

        Returns:
            A list of dicts, each with:
                - code: the top-level code (e.g. "1")
                - description: its label (e.g. "Employment and related activities")
        """,
        "signature":IcatusHierarchicalSearchAgentSignature
    }, 
    "NAF_rev2": {
        "system_prompt": """
        Classify an economic activity into a single NAF code by searching the
        classification hierarchy top-down, using the available tools rather than
        relying on memorized codes.

        The NAF (Nomenclature d'Activités Française, rév. 2) is a tree. The top
        level is a set of 21 sections coded with a single letter (e.g. "A"
        Agriculture, sylviculture et pêche; "C" Industrie manufacturière), and
        each code has children one level deeper:
        section -> division (2 digits) -> group (3 chars) -> class (4 chars) ->
        sub-class (5 chars), e.g. C -> 10 -> 10.1 -> 10.11 -> 10.11A. The first
        four levels are identical to the European NACE; the fifth level
        (sub-class) is the French-specific detail. A valid final answer is a real
        code that exists in the system; prefer the most specific (deepest, leaf)
        code that correctly covers the activity.

        Tools:
        - get_root_categories_tool(): list the top-level sections (A-U). Start here.
        - get_children_tool(code): list the direct child codes one level down, each
            with its label and includes/excludes notes.
        - get_code_tool(code): read one code's full record (label, elaborated
            definition, and 'comprend' (includes) / 'comprend aussi' /
            'ne comprend pas' (excludes) notes).
        - get_parent_tool(code): move one level up to re-read broader context or
            reconsider a branch.

        Reasoning protocol:
        1. Identify the essence of the activity: what good is produced or what
            service is rendered. Note any detail that affects placement — whether
            it is production, wholesale/retail trade, repair, or a service;
            whether the good is transformed or merely resold; the material or
            sector concerned. NAF classifies the ACTIVITY (what is done), not the
            product itself (that is the CPF's role).
        2. Call get_root_categories() and choose the single best-fitting section.
            Watch the classic ambiguities: manufacturing (C) vs. wholesale/retail
            trade (G); repair vs. manufacture; a support service vs. the core
            activity it supports. If two sections seem plausible, note the
            alternative to revisit later.
        3. Descend one level at a time with get_children on the current code.
            At each level:
            - Compare the activity against every child's label and notes.
            - Read the 'comprend' / 'comprend aussi' notes to confirm a match.
            - Read the 'ne comprend pas' notes carefully: they explicitly redirect
                activities that look like they belong here but are classified
                elsewhere, and usually name the correct code. Follow those
                pointers instead of forcing a fit.
            - Pick the best-matching child and repeat.
        4. Continue descending until you reach a sub-class leaf (get_children
            returns an empty list) or until no deeper code fits better than the
            current one.
        5. Before committing, verify the chosen code with get_code and check that
            its notes do not exclude this activity. If they do, backtrack using
            get_parent or return to a noted alternative branch and search again.
        6. If, after searching, no specific code fits, choose the most appropriate
            "autres" / residual (".9" or "Z") code within the correct branch
            rather than guessing a code from a different branch.

        Rules:
        - Never invent or guess a code from memory. Every code in the final answer
            and in your reasoning must have been returned by a tool.
        - The final naf_code must be an exact code string that the tools returned
            (e.g. "10.11A"), not a paraphrase or a made-up variant.
        - The target level for the classification is **5** (the sub-class, the
            French-specific leaf level); always try to find a level-**5** code —
            the level of a code is returned by get_code.
        - When evidence is ambiguous, prefer the interpretation supported by the
            includes/excludes notes over intuition.

        Output:
        - naf_code: the single most specific NAF code that correctly classifies
            the activity.
        - explaination: a concise justification tracing the path taken (section
            -> division -> group -> class -> sub-class) and citing the decisive
            includes/excludes note(s) that determined the choice, including any
            branch you rejected and why.
        """,

        "get_children": """
        List the direct child codes one level below the given code.

        Use this to drill down the hierarchy: given a code you've decided the
        activity falls under, this returns the candidate sub-codes to choose from
        next. Each child comes with its full record (label, comprend/ne comprend
        pas notes), so you can compare siblings and pick the best match without
        additional lookups. Returns an empty list for sub-class (leaf) codes that
        have no children.

        The hierarchy has five levels: section (1 letter, e.g. "C") ->
        division (2 digits, e.g. "10") -> group (3 chars, e.g. "10.1") ->
        class (4 chars, e.g. "10.11") -> sub-class (5 chars, e.g. "10.11A").

        Args:
            code: The parent classification code, e.g. "10.11".

        Returns:
            A list of Code objects for the direct children (e.g. "10.11A",
            "10.11B", ...), each with the same fields as get_code returns.
        """,

        "get_code": """
        Retrieve the full record for a single classification code.

        Use this to inspect one specific code in detail — its label, level in the
        hierarchy, and the definition plus 'comprend' (includes), 'comprend aussi'
        and 'ne comprend pas' (excludes) notes that disambiguate what belongs
        under it. The 'ne comprend pas' notes are especially useful: they point to
        the correct sibling or related code when an activity looks like it fits
        here but doesn't.

        The hierarchy has five levels: section (1 letter, e.g. "C") ->
        division (2 digits, e.g. "10") -> group (3 chars, e.g. "10.1") ->
        class (4 chars, e.g. "10.11") -> sub-class (5 chars, e.g. "10.11A").

        Args:
            code: The classification code, e.g. "10.11A".

        Returns:
            A Code object with fields:
                - code: the code string (e.g. "10.11A")
                - description: short label (e.g. "Transformation et conservation
                    de la viande de boucherie")
                - level: depth in the hierarchy (1 = section, 2 = division,
                    3 = group, 4 = class, 5 = sub-class)
                - detailled_description: longer prose definition (may be empty)
                - details: dict with 'comprend', 'comprend aussi', 'ne comprend
                    pas' notes (may be empty strings at higher levels)
        """,

        "get_parent": """
        Retrieve the immediate parent (one level up) of the given code.

        Use this to move back up the hierarchy — to reconsider a branch, read the
        broader category's definition for context, or verify that a code sits
        under the intended higher-level group. Returns the parent's full record.

        The hierarchy has five levels: section (1 letter, e.g. "C") ->
        division (2 digits, e.g. "10") -> group (3 chars, e.g. "10.1") ->
        class (4 chars, e.g. "10.11") -> sub-class (5 chars, e.g. "10.11A").

        Args:
            code: The classification code whose parent you want, e.g. "10.11A".

        Returns:
            A Code object for the parent (e.g. "10.11"), with the same fields as
            get_code returns. The scope note on a parent often summarises what the
            whole branch covers and excludes.
        """,

        "get_root_categories": """
        List the top-level sections of the classification system.

        Use this as the entry point when starting a classification: it returns the
        broadest categories (the 21 NAF sections, coded A-U, e.g. "A" Agriculture,
        sylviculture et pêche; "C" Industrie manufacturière; "G" Commerce;
        réparation d'automobiles et de motocycles), from which you select the most
        appropriate branch and then descend using get_children.

        Returns:
            A list of dicts, each with:
                - code: the top-level code (e.g. "C")
                - description: its label (e.g. "Industrie manufacturière")
        """,
        "signature": NafHierarchicalSearchAgentSignature
    },
    "SEA_2021": {
        "system_prompt": """
        Classify a private-household income or expenditure item into a single SEA
        2021 code by searching the classification hierarchy top-down, using the
        available tools rather than relying on memorized codes.

        The SEA 2021 (Systematik der Einnahmen und Ausgaben der privaten
        Haushalte, Statistisches Bundesamt) is a tree with THREE parts, each under
        its own top-level Abteilung:
        - Abteilung 00: Einnahmen der privaten Haushalte (household income).
        - Abteilungen 01-15: Verwendungszwecke des Individualkonsums (individual
            consumption; this part mirrors COICOP 2018). Households = 01-13,
            private non-profit institutions = 14, government = 15.
        - Abteilung 16: Ausgaben (ohne Individualkonsum) — non-consumption
            expenditure such as taxes, social-insurance and private-insurance
            contributions, membership fees, donations, loan repayment/interest,
            and formation of tangible/financial wealth.

        Codes are numeric and positional, up to 7 digits, with these levels:
        Abteilung (2 digits, e.g. "01") -> Gruppe (3, e.g. "011") ->
        Klasse (4, e.g. "0111") -> Unterklasse (5, e.g. "0111 1") ->
        Kategorie (6, e.g. "0111 10") -> Unterkategorie (7, e.g. "0111 101").
        Codes are written with a SPACE after the 4th digit (e.g. "0111 101").
        A valid final answer is a real code that exists in the system; prefer the
        most specific (deepest) code that correctly covers the item.

        Tools:
        - get_root_categories_tool(): list the top-level Abteilungen. Start here.
        - get_children_tool(code): list the direct child codes one level down, each
            with its Bezeichnung (label) and Eingeschlossen/Ausgeschlossen notes.
        - get_code_tool(code): read one code's full record (Bezeichnung, elaborated
            definition, and 'Eingeschlossen sind' (includes) / 'Ausgeschlossen
            sind' (excludes) notes).
        - get_parent_tool(code): move one level up to re-read broader context or
            reconsider a branch.

        Reasoning protocol:
        1. Identify the essence of the item. First decide which of the three parts
            it belongs to: is it INCOME (Einnahme -> Abteilung 00), a CONSUMPTION
            purchase of a good/service (Individualkonsum -> 01-15), or a
            NON-CONSUMPTION outflow such as a tax, contribution, donation, loan
            repayment or saving/investment (-> Abteilung 16)? This choice is the
            most consequential; note any detail that affects it (e.g. an insurance
            PREMIUM is 16, but a household appliance is 05).
        2. Within consumption, note details that drive placement: whether the good
            is for home consumption vs. immediate consumption in a
            restaurant/cafe (the latter usually -> 1111), its state/form (fresh,
            frozen, prepared/Fertiggericht), whether it is a good or a service,
            and its durability.
        3. Call get_root_categories() and choose the single best-fitting Abteilung.
            If two Abteilungen seem plausible, note the alternative to revisit.
        4. Descend one level at a time with get_children on the current code.
            At each level:
            - Compare the item against every child's Bezeichnung and notes.
            - Read 'Eingeschlossen sind' to confirm a match.
            - Read 'Ausgeschlossen sind' carefully: these notes explicitly
                redirect items that look like they belong here but are classified
                elsewhere, and name the correct code in parentheses (e.g.
                "... (0119 1)"). Follow those pointers instead of forcing a fit.
            - Pick the best-matching child and repeat.
        5. Continue descending until you reach a leaf (get_children returns an
            empty list) or until no deeper code fits better than the current one.
        6. Before committing, verify the chosen code with get_code and check that
            its notes do not exclude this item. If they do, backtrack using
            get_parent or return to a noted alternative branch and search again.
        7. If, after searching, no specific code fits, choose the most appropriate
            "Andere ... , a.n.g." (residual) code within the correct branch rather
            than guessing a code from a different branch.

        Rules:
        - Never invent or guess a code from memory. Every code in the final answer
            and in your reasoning must have been returned by a tool.
        - The final sea_code must be an exact code string that the tools returned,
            formatted with the space after the 4th digit (e.g. "0111 101"), not a
            paraphrase or a made-up variant.
        - The target level for the classification is **4** (the Klasse), the
            depth targeted by the EVS from 2023; always try to reach a level-**4**
            code. Descend further to Kategorie/Unterkategorie (levels 6-7) when the
            evidence clearly supports a more specific base unit and full depth is
            wanted. The level of a code is returned by get_code.
        - When evidence is ambiguous, prefer the interpretation supported by the
            Eingeschlossen/Ausgeschlossen notes over intuition.

        Output:
        - sea_code: the single most specific SEA 2021 code that correctly
            classifies the item.
        - explaination: a concise justification tracing the path taken (Abteilung
            -> Gruppe -> Klasse -> ... -> final code) and citing the decisive
            Eingeschlossen/Ausgeschlossen note(s) that determined the choice,
            including any branch you rejected and why.
        """,

        "get_children": """
        List the direct child codes one level below the given code.

        Use this to drill down the hierarchy: given a code you've decided the item
        falls under, this returns the candidate sub-codes to choose from next. Each
        child comes with its full record (Bezeichnung, Eingeschlossen/
        Ausgeschlossen notes), so you can compare siblings and pick the best match
        without additional lookups. Returns an empty list for leaf codes.

        The hierarchy has up to six levels below the top:
        Abteilung (2 digits, e.g. "01") -> Gruppe (3, e.g. "011") ->
        Klasse (4, e.g. "0111") -> Unterklasse (5, e.g. "0111 1") ->
        Kategorie (6, e.g. "0111 10") -> Unterkategorie (7, e.g. "0111 101").

        Args:
            code: The parent classification code, e.g. "0111".

        Returns:
            A list of Code objects for the direct children (e.g. "0111 1",
            "0111 2", ...), each with the same fields as get_code returns.
        """,

        "get_code": """
        Retrieve the full record for a single classification code.

        Use this to inspect one specific code in detail — its Bezeichnung (label),
        level in the hierarchy, and the definition plus 'Eingeschlossen sind'
        (includes) and 'Ausgeschlossen sind' (excludes) notes that disambiguate
        what belongs under it. The 'Ausgeschlossen sind' notes are especially
        useful: they point to the correct sibling or related code (given in
        parentheses) when an item looks like it fits here but doesn't.

        The hierarchy has up to six levels below the top:
        Abteilung (2 digits, e.g. "01") -> Gruppe (3, e.g. "011") ->
        Klasse (4, e.g. "0111") -> Unterklasse (5, e.g. "0111 1") ->
        Kategorie (6, e.g. "0111 10") -> Unterkategorie (7, e.g. "0111 101").

        Args:
            code: The classification code, e.g. "0111 101".

        Returns:
            A Code object with fields:
                - code: the code string (e.g. "0111 101")
                - description: short label (e.g. "Reis, z.B. Reis im Kochbeutel")
                - level: depth in the hierarchy (2 = Abteilung, 3 = Gruppe,
                    4 = Klasse, 5 = Unterklasse, 6 = Kategorie,
                    7 = Unterkategorie)
                - detailled_description: longer prose definition (may be empty)
                - details: dict with 'Eingeschlossen'/'Ausgeschlossen' notes
                    (may be empty strings at higher levels)
        """,

        "get_parent": """
        Retrieve the immediate parent (one level up) of the given code.

        Use this to move back up the hierarchy — to reconsider a branch, read the
        broader category's definition for context, or verify that a code sits under
        the intended higher-level Gruppe/Klasse. Returns the parent's full record.

        The hierarchy has up to six levels below the top:
        Abteilung (2 digits, e.g. "01") -> Gruppe (3, e.g. "011") ->
        Klasse (4, e.g. "0111") -> Unterklasse (5, e.g. "0111 1") ->
        Kategorie (6, e.g. "0111 10") -> Unterkategorie (7, e.g. "0111 101").

        Args:
            code: The classification code whose parent you want, e.g. "0111 101".

        Returns:
            A Code object for the parent (e.g. "0111 1"), with the same fields as
            get_code returns. The scope note on a parent often summarises what the
            whole branch covers and excludes.
        """,

        "get_root_categories": """
        List the top-level Abteilungen of the classification system.

        Use this as the entry point when starting a classification: it returns the
        broadest categories across the SEA 2021's three parts — Abteilung 00
        (Einnahmen), Abteilungen 01-15 (Verwendungszwecke des Individualkonsums,
        e.g. "01" Nahrungsmittel und alkoholfreie Getränke, "07" Verkehr), and
        Abteilung 16 (Ausgaben ohne Individualkonsum) — from which you select the
        most appropriate branch and then descend using get_children.

        Returns:
            A list of dicts, each with:
                - code: the top-level code (e.g. "01")
                - description: its label (e.g. "Nahrungsmittel und alkoholfreie
                    Getränke")
        """,
        "signature": SeaHierarchicalSearchAgentSignature
    }   
    
}
