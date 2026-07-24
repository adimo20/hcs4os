from fastmcp import FastMCP
from dotenv import load_dotenv
from typing import Any
import os
import json

from classification_system.registry import get_classification_system
from tool_descriptions import doc_strings

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
    name=doc_strings["server_name"],
    instructions=doc_strings["server_instructions"]
)

@mcp.tool(annotations=hints, description=doc_strings["get_root_categories"])
def get_root_categories() -> list[dict]:
    
    return classification.get_root_categories()

@mcp.tool(annotations=hints, description=doc_strings["get_children"])
def get_children(parent_code: str) -> list[dict]:
    
    children: list = classification.get_children(
        code=parent_code
    )
    children_json = [{
            "code":c.code,
            "description":c.description
        } for c in children]
    
    return children_json

@mcp.tool(annotations=hints, description=doc_strings["get_parent"])
def get_parent(parent_code: str) -> dict|None:
    
    parent = classification.get_parent(
        parent=parent_code
    )
    return parent


@mcp.tool(annotations=hints, description=doc_strings["get_code_specification"])
def get_code_specification(list_of_codes: list[str]) -> list[dict]:
    
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
        port=int(os.getenv("PORT", 8000)),
        
    )