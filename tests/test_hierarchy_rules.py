"""get_prefixes of every built-in system, as pure functions (no data loaded).

These are the rules the whole hierarchy is derived from, so each code-format
quirk documented in the README gets a test here.
"""
import pytest

from hcs4os.classification_system.registry import _REGISTRY_CS


def prefixes(system_id: str, code: str) -> list[str]:
    cls = _REGISTRY_CS[system_id]
    instance = object.__new__(cls)          # skip __init__: no file is loaded
    return instance.get_prefixes(code)


@pytest.mark.parametrize("system_id, code, expected", [
    # COICOP: dot-separated
    ("COICOP", "01.1.1.1", ["01", "01.1", "01.1.1", "01.1.1.1"]),
    ("COICOP", "01", ["01"]),
    ("ICATUS", "110", ["1", "11", "110"]),
    ("KLDB", "01104", ["0", "01", "011", "0110", "01104"]),
    ("KLDB", "011 04", ["0", "01", "011", "0110", "01104"]),        # spaces are removed
    # SEA: space re-inserted after the 4th digit
    ("SEA", "0111 101", ["01", "011", "0111", "0111 1", "0111 10", "0111 101"]),
    ("SEA", "0111101", ["01", "011", "0111", "0111 1", "0111 10", "0111 101"]),
    # NAF: dot after the 2nd digit
    ("NAF_rev2", "05.20Z", ["05", "05.2", "05.20", "05.20Z"]),
    # WZ: trailing dots stripped and de-duplicated
    ("WZ", "01.11.0", ["01", "01.1", "01.11", "01.11.0"]),
])
def test_get_prefixes(system_id, code, expected):
    assert prefixes(system_id, code) == expected



@pytest.mark.parametrize("system_id", ["COICOP", "ICATUS", "KLDB", "SEA", "NAF_rev2", "WZ"])
def test_root_codes_have_a_single_prefix(system_id):
    root = {"COICOP": "01", "ICATUS": "1", "KLDB": "1",
            "SEA": "01", "NAF_rev2": "01", "WZ": "01"}[system_id]
    assert prefixes(system_id, root) == [root]
