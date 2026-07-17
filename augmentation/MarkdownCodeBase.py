from abc import ABC, abstractmethod
from dataclasses import dataclass
from Code import Code

@dataclass
class MarkdownCodeBaseConfig:
    code_header:str = "Code"
    description_header:str = "Titel"
    detailled_description_header:str = "Detailierte Beschreibung"
    examples_header:str = "Beispiele"
    trace_header: str = "Pfad der Klassifikation"



class MarkdownCodeBase(ABC):
    
    def __init__(
        self,
        taxanomy_of_labels:list[str],
        config:MarkdownCodeBaseConfig
    )->None:
        
        self.taxanomy_of_labels = taxanomy_of_labels
        self.config = config

    def header_plus_content(
        self,
        header:str,
        content:str,
        header_degree:int=2,
        highlight_content:bool=False
    )->str:
        """
        Creates a markdown element, that consists out of a header and a content element, 
        you can highlight the content and or steer the degree of of the header

        Args:
            header (str) - Name of the markdown Element you want to create, e.g. ## Detailled Description of the
            Code
            content (str) - The content body you want to show
            header_degree (int) - Degree of the the header, # or ## ...
            highlight_content (bool) - Indicates if you want to highlight the content body, default is false
        Returns:
            str: markdown formatted string
        """
        content_formatted:str = f"**{content}**" if highlight_content else content
        hashes = "#"*header_degree
        return f"{hashes} {header}\n{content_formatted}\n"
    
    
    def generate_examples_part(
        self,
        examples:list[str]
    )->str:
        """
        Generates a string containing a markdown list of examples you want to show to the agents, e.g.:
        Args:
            examples (list[str]) - Examples you want to show to the model
        Returns:
            str
        Example:
            ["Käse", "Milch"] --> "## Beispiele \n* Käse \n* Milch\n"

        """
        joined_examples = "\n".join([f"* {ex}" for ex in examples])
        return f"## {self.config.examples_header} \n{joined_examples}"
    

    def format_traces_to_markdown(
        self,
        trace:list[tuple[str,str]]
    )->str:
        """
        Creates a formatted markdown list from the output of the ClassificationSystem().get_code_trace() function
        Args:
            trace (list[tuple[str,str]]) - output of the ClassificationSystem().get_code_trace(), in Form of e.g. 
            [("01", "FOOD AND NON-ALCOHOLIC BEVERAGES"), ("011", "FOOD"), ...]
        Returns:
            str: markdown formatted string
        Example:
           output: "* `Abteilung 01`: **FOOD AND NON-ALCOHOLIC BEVERAGES** <br> \n* `Gruppe 011`: **FOOD** <br> \n ..."
        """
        i = 0
        formatted_traces = ""
        for code, title in trace:
            formatted_traces += f"`{self.taxanomy_of_labels[i]} {code}`: **{title}** <br> \n" 
            # Output looks like: `Abteilung 01`: **FOOD AND NON-ALCOHOLIC BEVERAGES** <br> \n and so on for the whole trace
            i+=1
        return formatted_traces

    def code_to_markdown(
        self,
        code:Code,
        trace:list[tuple],
        examples:list[str]|None=None   
    )->str:

        """
        Create the whole code summary by joining the different markdown parts. Customizable to the needs of a classification
        system
        It's possible to cut out or add new components, as long as you return a string
        Args:        
            code (Code) - important the input used here is type Code, custom dataclass
            trace (list[tuple[str, str]])
            examples (list[str]|None) - List of examples you want to add to your code summary, can be left None, in this
            case no examples will be shown
            classification_name (str) - Name of the classification system you want to use, will be inserted into the
            markdown string
        Returns:
            str: comprehensive markdown formatted string that is understandable for agents
        """

        conf:list[tuple[str, str, bool]] = [
            (
                self.config.description_header,
                code.description,
                False
                
            ),(
                self.config.code_header,
                code.code,
                True
            ),(
                self.config.detailled_description_header,
                code.detailled_description,
                False
            ),(
                self.config.trace_header,
                self.format_traces_to_markdown(trace=trace),
                False
            )
        ]
        
        code_markdown_format:str = "\n".join(
            [
                self.header_plus_content(header=h, content=c, highlight_content=highlight)
                for h, c, highlight in conf
            ]
        )
         
        if examples is not None and examples != []:      
            code_markdown_format = code_markdown_format + self.generate_examples_part(
                examples=examples
            )

        return code_markdown_format 
    
if __name__ == "__main__":
    from classification_system.registry import get_classification_system

    cs = get_classification_system("KLDB_2010_REPHRASED")
    code = cs.get_code("111")
    trace = cs.get_code_trace("111")
    
    formatter = MarkdownCodeBase(
        taxanomy_of_labels = [
            "Abteilung",
            "Gruppe",
            "Klasse",
            "Unterklasse",
            "Kategorie",
            "Unterkategorie"
        ],
        config=MarkdownCodeBaseConfig()
    )

    print(code)
    print(
        formatter.code_to_markdown(
            code=code,
            trace=trace
        )
    )
