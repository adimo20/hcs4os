import pytest
from hcs4os.classification_system.registry import get_classification_system


@pytest.mark.parametrize(
    "input_code, formatted_output",
    [
        ("1", ["11", "12", "13", "14", "15", "16", "17", "18"]),
        ("2", ["21", "22", "23", "24", "25"]),
        ("3", ["31", "32", "33", "34", "35", "36", "37", "38", "39"]),
        ("4", ["41", "42", "43", "44", "49"]),
        ("5", ["51", "52", "53", "54", "59"]),
        ("6", ["61", "62", "63", "64", "69"]),
        ("7", ["71", "72", "73", "74", "75", "79"]),
        ("8", ["81", "82", "83", "84", "85", "86", "89"]),
        ("9", ["91", "92", "93", "94", "95", "99"]),
        ("12", ["121", "122", "123", "124", "125", "126", "127", "128", "129"]),
        ("13", ["131", "132", "133", "134", "135", "136", "139"]),
        ("14", ["141", "142"]),
        ("15", ["150"]),
        ("16", ["160"]),
        ("17", ["170"]),
        ("150", []),
        ("160", []),
    ],
)
def test_classification_system_icatus(input_code, formatted_output):
    icatus = get_classification_system("ICATUS_2016")
    codes = icatus.get_children(input_code)
    codes = [c.code for c in codes]
    assert codes == formatted_output

@pytest.mark.parametrize(
    "input_code, formatted_output",
    [
        # Divisions -> groups
        ("01", ["01.1", "01.2", "01.3"]),
        ("02", ["02.1", "02.2", "02.3", "02.4"]),
        ("12", ["12.1", "12.2"]),
        ("01.1", ["01.1.1", "01.1.2", "01.1.3", "01.1.4",
                  "01.1.5", "01.1.6", "01.1.7", "01.1.8", "01.1.9"]),
        ("07.1", ["07.1.1", "07.1.2", "07.1.3", "07.1.4"]),
        ("01.1.1", ["01.1.1.1", "01.1.1.2", "01.1.1.3",
                    "01.1.1.4", "01.1.1.5", "01.1.1.9"]),
        # Leaf subclass -> no children
        ("02.1.1.0", []),
        ("12.6.1", []),
    ],
)
def test_classification_system_coicop(input_code, formatted_output):
    coicop = get_classification_system("COICOP_2018")
    codes = coicop.get_children(input_code)
    codes = [c.code for c in codes]
    assert codes == formatted_output

