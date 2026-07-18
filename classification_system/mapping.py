from pathlib import Path

def make_path(filename:str):
    return Path(__file__).resolve().parent.parent / "data" / filename

mapping = {
    "SEA_2021":{
        "classification_system":"SEA", # Decorater-ID within classification system, which implements the given logic
        "loader_name":"KLASS_SERVER", # Decorater-ID within dataloaders, which implements the given logic
        "data_path":make_path("sea2021.xml") # Softpath to the xml or ... file which contains the classification system
    },
    "WZ_2008":{
        "classification_system":"WZ",      
        "loader_name":"KLASS_SERVER",
        "data_path":make_path("wz2008.xml")
    },
    "KLDB_2010":{
        "classification_system":"KLDB",      
        "loader_name":"KLASS_SERVER",
        "data_path":make_path("kldb_2010.xml")
    },
    "KLDB_2010_REPHRASED":{
        "classification_system":"KLDB",      
        "loader_name":"JSON",
        "data_path":make_path("classification_system_KldB_2010.json")
    }
    ,
    "SEA_2021_REPHRASED":{
        "classification_system":"SEA_NS",      
        "loader_name":"JSON",
        "data_path":make_path("classification_system_SEA_2021.json")
    }

}
