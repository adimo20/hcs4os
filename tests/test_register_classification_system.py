import pytest
from pathlib import Path
from dotenv import load_dotenv


def test_registering_classification_system():
    # Imports for the dataloader
    from hcs4os.dataloaders.BaseDataLoader import ClassificationLoader
    from hcs4os.dataloaders.registry import register as register_loader
    from hcs4os._shared.Code import Code
    # Imports for the classification system
    from hcs4os.classification_system.BaseClassificationSystem import ClassificationSystem
    from hcs4os.classification_system.registry import register as register_classification_system, get_classification_system
    from hcs4os.classification_system.mapping import mapping
    # General import to load data
    import json

    
    @register_loader("FICTION")
    class HSCodeLoader(ClassificationLoader):
        
        def to_records(self):
            
            with open(self.path, "r", encoding="utf-8") as f:
                codes = json.loads(f.read())
            
            return codes

    @register_classification_system("FICTION")
    class ClassificationSystemHSCode(ClassificationSystem):
        
        def get_root_categories(self):
            return [c.to_dict() for c in self.codes if len(c.code) == 1]    
        
        def get_prefixes(self, code:str):
            return [code[:i] for i in range(len(code)+1) if i > 0]  
    
    mapping["FICTION_TEST"] = {
            "classification_system":"FICTION", 
            "loader_name":"FICTION",
            "data_path": Path(__file__).parent / "resources" / "fiction.json", 
            "metadata":{}
    }
    
    fiction = get_classification_system("FICTION_TEST")

@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Membeder of the Jedi high council", "111"),
        ("Sith Lord", "112")
    ],
)
def test_registering_classification_system_store_integration(input_text, expected_output):
    # Imports for the dataloader
    from hcs4os.dataloaders.BaseDataLoader import ClassificationLoader
    from hcs4os.dataloaders.registry import register as register_loader
    from hcs4os._shared.Code import Code
    # Imports for the classification system
    from hcs4os.classification_system.BaseClassificationSystem import ClassificationSystem
    from hcs4os.classification_system.registry import register as register_classification_system, get_classification_system
    from hcs4os.classification_system.mapping import mapping, make_path
    # General import to load data
    import json
    from hcs4os._vector_database.vector_store import VectorStore

    
    @register_loader("FICTION")
    class HSCodeLoader(ClassificationLoader):
        
        def to_records(self):
            
            with open(self.path, "r", encoding="utf-8") as f:
                codes = json.loads(f.read())
            
            return codes

    @register_classification_system("FICTION")
    class ClassificationSystemHSCode(ClassificationSystem):
        
        def get_root_categories(self):
            return [c.to_dict() for c in self.codes if len(c.code) == 1]    
        
        def get_prefixes(self, code:str):
            return [code[:i] for i in range(len(code)+1) if i > 0]  
    
    mapping["FICTION_TEST"] = {
            "classification_system":"FICTION", 
            "loader_name":"FICTION",
            "data_path": Path(__file__).parent / "resources" / "fiction.json", 
            "metadata":{}
    }
    

    fiction = get_classification_system("FICTION_TEST")

    store = VectorStore(
        collection_name="FICTION_TEST",
        chromadb_path=rf"C:\Users\adrim\projects\ABC\data\chroma",
        embedding_function_type="ollama",
        url="http://192.168.178.48:11434",
        model_name="qwen3-embedding:0.6b",
    )

    store.index(
        classification_system=fiction
    )
    
    result = store.search_category(input_text, k=5, where={"level":3})
    codes = [c.get("code") for c in result]
    assert expected_output in codes
    
    
    
    
    