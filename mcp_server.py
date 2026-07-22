from fastmcp import FastMCP
from dotenv import load_dotenv
from typing import Any
import os

from classification_system.registry import get_classification_system

load_dotenv()

CLASSIFICATION_NAME = os.getenv("CLASSIFICATION_NAME", "hierarchical Classification")
CLASSIFICATION_ID = os.getenv("CLASSIFICATION_ID")


try:
    if CLASSIFICATION_ID is None:
        raise ValueError("Could't find a valid classification id in environment variable. Make sure to set CLASSIFICATION_ID.")
    classification = get_classification_system(name=CLASSIFICATION_ID)
except Exception as e:
    raise Exception(f"Could not load classification system: {e}")

hints = {
    "readOnlyHint": True,
    "idempotentHint": True,
    "openWorldHint": False,
}


mcp: FastMCP[Any] = FastMCP(
    name="MCP-Server for the Classification of products according to the SEA (Systematik der Einnahmen und Ausgaben der Privaten Haushalte)"
)

@mcp.tool(annotations=hints)
def get_root_categories() -> list[dict]:
    """
    Returns the top-level divisions (root categories) of the classification system.
    
    WHEN TO USE THIS:
    - Use this as your starting point when you have a completely new item to classify and you don't know which general category it belongs to.
    - Use this to understand the highest-level structure of the classification.
    
    Args:
        None: no input required.
        
    Returns:
        list[dict]: A list of dictionaries containing the top-level 'codes' and its overarching 'description'.
    """
    return classification.get_root_categories()

@mcp.tool(annotations=hints)
def get_children(parent_code: str) -> list[dict]:
    """
    Collects a list of direct child categories for a given parent code within the SEA system.
    
    WHEN TO USE THIS:
    - Use this to drill down hierarchically into the classification tree. 
    - Once you have identified a broad category, use this tool to find the next level of specificity.
    - Repeat this process until you reach the lowest level (leaf node) that accurately describes the product.
    
    Args:
        parent_code (str): The classification code you want to explore the children
        
    Returns:
        list[str]: A list of JSON strings detailing the child categories, their codes, and descriptions.
    """
    children: list = classification.get_children(
        code=parent_code
    )
    children_json = [{
            "code":c.code,
            "description":c.description
        } for c in children]
    
    return children_json

@mcp.tool(annotations=hints)
def get_parent(parent_code: str) -> dict|None:
    """
    Retrieves the immediate parent category for a given overly specific code.
    
    Args:
        specific_code (str): The overly specific classification code you want to abstract 
                             upwards from (e.g., '03121' or '01141'). Do not include trailing zeros.
        
    Returns:
        str: A JSON string detailing the broader parent category, its code, and description.
    """
    parent = classification.get_parent(
        parent=parent_code
    )
    return parent


@mcp.tool(annotations=hints)
def get_code_specification(list_of_codes: list[str]) -> list[dict]:
    """
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
        str: A formatted Markdown string containing comprehensive summaries for all requested valid codes, 
             separated by horizontal rules (---).
    """
    codes = [
        classification.get_code(
           code=code
        ).to_dict() for code in list_of_codes
    ]
    return codes

if __name__ == "__main__":
    
    transport = os.getenv("MCP_TRANSPORT_METHOD", "sse")
    if transport not in ("stdio", "http", "sse", "streamable-http"):
        raise ValueError(f"Invalid transport: {transport}")

    mcp.run(
        transport=transport,
        host=os.getenv("HOST"),
        port=os.getenv("PORT", 8000),
        
    )