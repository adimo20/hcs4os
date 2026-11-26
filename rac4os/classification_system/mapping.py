from pathlib import Path

def make_path(filename:str):
    """
    Resolve a filename to the bundled data file of that name.

    Parameters:
        filename (str): name of a file in the package's `_data` directory.

    Returns:
        Path: absolute path to that file inside the installed package.
    """
    return Path(__file__).resolve().parent.parent / "_data" / filename

mapping = {
    "NAF_rev2":{
        "classification_system":"NAF_rev2", 
        "loader_name":"JSON", 
        "data_path":make_path("NAF_INSEE.json"), 
        "metadata":{
            "url":"https://unstats.un.org/unsd/demographic-social/time-use/icatus-2016/tableview",
        }
    },
    "ICATUS_2016":{
        "classification_system":"ICATUS", 
        "loader_name":"ICATUS", 
        "data_path":make_path("UNSD_Time_Use_Statistics.csv"), 
        "metadata":{
            "url":"https://unstats.un.org/unsd/demographic-social/time-use/icatus-2016/tableview",
        }
    },
    "SEA_2021_REPHRASED":{
        "classification_system":"SEA", 
        "loader_name":"JSON", 
        "data_path":make_path("sea_2021_augmented.json"), 
        "metadata":{
            "url":"https://klassifikationsserver.de/klassService/thyme/variant/sea_2021",
            "model":"mistral/zai-glm-5-2",
            "date":"15.09.2026"
        }
    },
    "SEA_2021":{
        "classification_system":"SEA", 
        "loader_name":"KLASS_SERVER", 
        "data_path":make_path("sea2021.xml"), 
        "metadata":{
            "url":"https://klassifikationsserver.de/klassService/thyme/variant/sea_2021",
        }
    },
    "COICOP_2018":{
        "classification_system":"COICOP", 
        "loader_name":"COICOP", 
        "data_path":make_path("COICOP_2018_English_structure.xlsx"), 
        "metadata":{
            "url":"",
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
    }
}
