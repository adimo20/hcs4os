"""Code: the dataclass every classification is normalised into."""
import json

from hcs4os._shared.Code import Code


def test_defaults_are_empty():
    c = Code()
    assert (c.code, c.description, c.level, c.detailled_description, c.details) == ("", "", 0, "", {})


def test_details_default_is_not_shared_between_instances():
    a, b = Code(), Code()
    a.details["includes"] = "x"
    assert b.details == {}


def test_from_dict_keeps_known_fields():
    c = Code.from_dict({"code": "01.1", "description": "Food", "level": 2,
                        "detailled_description": "long", "details": {"includes": "bread"}})
    assert c.code == "01.1"
    assert c.level == 2
    assert c.details == {"includes": "bread"}


def test_from_dict_silently_drops_unknown_keys():
    c = Code.from_dict({"code": "1", "description": "x", "parent_code": "0", "foo": 42})
    assert c.code == "1"
    assert not hasattr(c, "parent_code")


def test_from_dict_missing_keys_use_defaults():
    c = Code.from_dict({"code": "1"})
    assert c.description == "" and c.level == 0 and c.details == {}


def test_to_dict_round_trip():
    original = Code(code="0111 1", description="Getreide", level=4, details={"keywords": ["Reis"]})
    assert Code.from_dict(original.to_dict()) == original


def test_to_json_is_valid_json_and_keeps_umlauts():
    c = Code(code="1", description="Gemüse, frisch")
    text = c.to_json()
    assert "Gemüse" in text                      # ensure_ascii=False
    assert json.loads(text)["description"] == "Gemüse, frisch"


def test_to_json_indent():
    assert "\n  " in Code(code="1").to_json(indent=2)
