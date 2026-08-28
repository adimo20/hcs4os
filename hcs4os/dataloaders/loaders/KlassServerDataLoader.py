from ..BaseDataLoader import ClassificationLoader
from ..registry import register
import xml.etree.ElementTree as ET

@register("KLASS_SERVER")
class KlassServerDataloader(ClassificationLoader):
    
  def __get_detail_dict(self, item):

    """
    Constructs a dictionary for all the relevant sea categories with the given class names from the xml file.
    Parameters:
        item:xml.etree.ElementTree.Element
    Returns:
        dict
    """

    return {
        "code":item.get("id"),
        "description":item.find(".//Label[@qualifier='Usual']").find("LabelText").text,
        "level":int(item.get("idLevel")),
        "details":{
            "keywords":[
                i.find(".//PropertyText[@type='Content']").text  if i.find(".//PropertyText[@type='Content']") is not None else ""   for i in item.findall(".//Property[@name='Keyword']")],
            "ExplanatoryNote":{
                "exclusions":[ex.find(".//PropertyQualifier[@name='Exclusions']").find("PropertyText").text if ex.find(".//PropertyQualifier[@name='Exclusions']") is not None else "" for ex in item.findall(".//Property[@name='ExplanatoryNote']")],
                "explicit_inclusion":[ex.find(".//PropertyQualifier[@name='CentralContent']").find("PropertyText").text if ex.find(".//PropertyQualifier[@name='CentralContent']") is not None else "" for ex in item.findall(".//Property[@name='ExplanatoryNote']")]
            },
            "context":item.find(".//Label[@qualifier='Context']").find("LabelText").text if item.find(".//Label[@qualifier='Context']") is not None else "",
        }
    }

  def __parse_xml(self, path):
    """
    Parses the sea-documentation xml file and extracts all the relevant information.
    Parameters:
      path:str
    Returns:
      list
    """
    tree = ET.parse(path)
    root = tree.getroot()
    classification = root.find("Classification")
    
    if classification is None:
       raise ValueError("Could not find any classification codes inside the file!")
    
    labels = classification.findall("Item")
    all_labels = [self.__get_detail_dict(l_) for l_ in labels]
    return all_labels

  def __clean_dict(self, all_labels):

    """
    Cleans the dictionary of empty values.
    Parameters:
      all_labels:list
    Returns:
      list
    """

    for label in all_labels:
      for key in list(label["details"].keys()):
        if key == "ExplanatoryNote":
          for key2 in list(label["details"]["ExplanatoryNote"].keys()):
            if label["details"]["ExplanatoryNote"][key2] == []:
              del label["details"]["ExplanatoryNote"][key2]
        if label["details"][key] == [""] or label["details"][key] == "" or label["details"][key] == [] or label["details"][key] == {}:
          del label["details"][key]
      if label["details"] == {}:
        del label["details"]

    return all_labels

  def to_records(
        self
    )->list[dict]:

    all_labels = self.__parse_xml(self.path)
    all_labels = self.__clean_dict(all_labels)
    return all_labels

if __name__ == "__main__":
    import sys

    codes = KlassServerDataloader(sys.argv[1]).load()
    print(codes[:10])

