doc_strings = {
    "server_name":"MCP-Server für die Klassifikation der Einnahmen und Ausgabem der privaten Haushalte (SEA)",
    ###########################################################################################################
    ###########################################################################################################
    "server_instructions":"",
    ###########################################################################################################
    ###########################################################################################################
    "get_root_categories":"""
    Returns the top-level divisions (root categories) of the classification system.
    
    WHEN TO USE THIS:
    - Use this as your starting point when you have a completely new item to classify and you don't know which general category it belongs to.
    - Use this to understand the highest-level structure of the classification.
    
    Args:
        None: no input required.
        
    Returns:
        list[dict]: A list of dictionaries containing the top-level 'codes' and its overarching 'description'.
    """,
    ###########################################################################################################
    ###########################################################################################################
    "get_children":"""
    Collects a list of direct child categories for a given parent code within the SEA system.
    
    WHEN TO USE THIS:
    - Use this to drill down hierarchically into the classification tree. 
    - Once you have identified a broad category, use this tool to find the next level of specificity.
    - Repeat this process until you reach the lowest level (leaf node) that accurately describes the product.
    
    Args:
        parent_code (str): The classification code you want to explore the children
        
    Returns:
        list[str]: A list of JSON strings detailing the child categories, their codes, and descriptions.
    """,
    ###########################################################################################################
    ###########################################################################################################
    "get_parent":"""
    Retrieves the immediate parent category for a given overly specific code.
    
    Args:
        specific_code (str): The overly specific classification code you want to abstract 
                             upwards from (e.g., '03121' or '01141'). Do not include trailing zeros.
        
    Returns:
        str: A JSON string detailing the broader parent category, its code, and description.
    """,
    ###########################################################################################################
    ###########################################################################################################
    "get_code_specification":"""
    Generates a comprehensive, definitive Markdown report for specific SEA classification codes.
    
    WHEN TO USE THIS:
    - Use this only with codes **that you do not already have seen a detailled descritpiton** when using a semantic or fulltext search.
    - Use this to VERIFY if a product belongs in a specific category.
    - Use this when you need the official rules, inclusions, and exclusions for a specific code.
    - If you are debating between two or more codes, pass them both in the list to compare their exact specifications.
    
    This method retrieves the exact hierarchical trace (path) through the classification system, 
    and detailed descriptive texts to help make a final classification decision and understand the semantic meaning of a code.

    Args:
        list_of_codes (list[str]): A list of code strings to retrieve detailed information for. 
            Must always be a list, even for a single code (e.g., ['01111'] or ['01111', '01211']).

    Returns:
        dict
    """
}