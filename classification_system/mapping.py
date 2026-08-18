from pathlib import Path

def make_path(filename:str):
    return Path(__file__).resolve().parent.parent / "data" / filename

mapping = {
    "SEA_2021":{
        "classification_system":"SEA", # Decorater-ID within classification system, which implements the given logic
        "loader_name":"KLASS_SERVER", # Decorater-ID within dataloaders, which implements the given logic
        "data_path":make_path("sea2021.xml"), # Softpath to the xml or ... file which contains the classification system
        "metadata":{
            "url":"https://klassifikationsserver.de/klassService/thyme/variant/sea_2021",
        }
    },
    "WZ_2008":{
        "classification_system":"WZ",      
        "loader_name":"KLASS_SERVER",
        "data_path":make_path("wz2008.xml"),
        "metadata":{
            "url":"https://klassifikationsserver.de/klassService/thyme/variant/wz2008"
        }
    },
    "KLDB_2010":{
        "classification_system":"KLDB",      
        "loader_name":"KLASS_SERVER",
        "data_path":make_path("kldb_2010.xml"),
        "metadata":{
            "url":"https://klassifikationsserver.de/klassService/thyme/variant/kldb2010"
        }
    },
    "KLDB_2010_REPHRASED":{
        "classification_system":"KLDB",      
        "loader_name":"JSON",
        "data_path":make_path("classification_system_KldB_2010.json"),
        "metadata":{}
    }
    ,
    "SEA_2021_REPHRASED":{
        "classification_system":"SEA_NS",      
        "loader_name":"JSON",
        "data_path":make_path("classification_system_SEA_2021.json"),
        "metadata":{}
    },
    "EAV":{
        "classification_system":"EAV",      
        "loader_name":"KLASS_SERVER",
        "data_path":make_path("EAV2026.xml"),
        "metadata":{}
    },
    "VUL":{
        "classification_system":"VUL",      
        "loader_name":"KLASS_SERVER",
        "data_path":make_path("VUL_2025-2026-03-13-Gliederung_mit_Erläuterung.xml"),
        "metadata":{}
    },
    "NST":{
        "classification_system":"NST",      
        "loader_name":"KLASS_SERVER",
        "data_path":make_path("NST_2007-2024-08-27-Gliederung_mit_Erläuterung.xml"),
        "metadata":{}
        
    },
    "GP":{
        "classification_system":"GP",      
        "loader_name":"KLASS_SERVER",
        "data_path":make_path("GP_2026-2026-04-10-Structure.xml"),
        "metadata":{}
        
    }

}
