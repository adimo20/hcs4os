"""Classification systems: navigation API, every bundled dataset, registration."""
import pytest

from rac4os.classification_system import registry as cs_registry
from rac4os.classification_system.mapping import mapping
from rac4os.classification_system.registry import get_classification_system

ALL_KEYS = [k for k in mapping]

@pytest.fixture(scope="module")
def systems():
    """Load each bundled system once per module."""
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = get_classification_system(name)
        return cache[name]
    return get


# ---------------------------------------------------------------------------
# children of known codes (existing tests)
# ---------------------------------------------------------------------------

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
def test_classification_system_icatus(systems, input_code, formatted_output):
    codes = [c.code for c in systems("ICATUS_2016").get_children(input_code)]
    assert codes == formatted_output


@pytest.mark.parametrize(
    "input_code, formatted_output",
    [
        ("01", ["01.1", "01.2", "01.3"]),
        ("02", ["02.1", "02.2", "02.3", "02.4"]),
        ("12", ["12.1", "12.2"]),
        ("01.1", ["01.1.1", "01.1.2", "01.1.3", "01.1.4",
                  "01.1.5", "01.1.6", "01.1.7", "01.1.8", "01.1.9"]),
        ("07.1", ["07.1.1", "07.1.2", "07.1.3", "07.1.4"]),
        ("01.1.1", ["01.1.1.1", "01.1.1.2", "01.1.1.3",
                    "01.1.1.4", "01.1.1.5", "01.1.1.9"]),
        ("02.1.1.0", []),
        ("12.6.1", []),
    ],
)
def test_classification_system_coicop(systems, input_code, formatted_output):
    codes = [c.code for c in systems("COICOP_2018").get_children(input_code)]
    assert codes == formatted_output


def test_classification_system_sea_children_keep_the_space(systems):
    children = [c.code for c in systems("SEA_2021").get_children("0111")]
    assert children and all(c.startswith("0111 ") for c in children)


# ---------------------------------------------------------------------------
# navigation API, on the bundled SEA and on the custom fiction system
# ---------------------------------------------------------------------------

def test_get_code_returns_code_object(systems):
    c = systems("SEA_2021").get_code("0111 1")
    assert c.description == "Getreide"
    assert c.level == 4


def test_get_code_unknown_raises_value_error(systems):
    with pytest.raises(ValueError, match="not in the classification system"):
        systems("SEA_2021").get_code("99999")


def test_get_children_of_unknown_code_is_empty(systems):
    assert systems("SEA_2021").get_children("not-a-code") == []


def test_get_parent_of_root_is_none(systems):
    assert systems("SEA_2021").get_parent("01") is None


def test_get_code_trace(systems):
    assert systems("SEA_2021").get_code_trace("0111 1") == [
        ("01", "Nahrungsmittel und alkoholfreie Getränke"),
        ("011", "Nahrungsmittel"),
        ("0111", "Getreide und Getreideerzeugnisse"),
        ("0111 1", "Getreide"),
    ]


def test_get_root_categories_shape(systems):
    roots = systems("SEA_2021").get_root_categories()
    assert roots[0] == {"code": "00", "description": "Einnahmen der privaten Haushalte"}
    assert all(set(r) == {"code", "description"} for r in roots)


def test_add_code_with_parent(fiction):
    fiction.add_code({"code": "114", "description": "Grey Jedi", "level": 3}, parent="11")
    assert fiction.get_code("114").description == "Grey Jedi"
    assert "114" in [c.code for c in fiction.get_children("11")]


def test_add_code_with_new_parent_key(fiction):
    fiction.add_code({"code": "999", "description": "x"}, parent="99")
    assert [c.code for c in fiction.get_children("99")] == ["999"]


# ---------------------------------------------------------------------------
# every bundled dataset loads and is internally consistent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_KEYS)
def test_every_mapped_system_loads(systems, name):
    cs = systems(name)
    assert len(cs.codes) > 0
    assert len(cs.get_root_categories()) > 0


@pytest.mark.parametrize("name", ALL_KEYS)
def test_codes_are_unique(systems, name):
    codes = [c.code for c in systems(name).codes]
    assert len(codes) == len(set(codes))



# ---------------------------------------------------------------------------
# lookup and registration
# ---------------------------------------------------------------------------

def test_unknown_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown format"):
        get_classification_system("DOES_NOT_EXIST")


def test_custom_classification_can_be_registered(fiction):
    assert [r["code"] for r in fiction.get_root_categories()] == ["1", "2"]
    assert [c.code for c in fiction.get_children("22")] == ["221", "222", "223"]
    assert fiction.get_code_trace("223") == [
        ("2", "Lord of the Rings characters"),
        ("22", "Servants of the Shadow"),
        ("223", "Traitors and the corrupted"),
    ]


def test_registration_does_not_leak_between_tests():
    """Runs after the fixture-based tests: the fiction entries must be gone again."""
    assert "FICTION_TEST" not in mapping
    assert "FICTION" not in cs_registry._REGISTRY_CS


def test_register_decorator_adds_system(monkeypatch):
    from rac4os.classification_system.BaseClassificationSystem import ClassificationSystem
    monkeypatch.setattr(cs_registry, "_REGISTRY_CS", dict(cs_registry._REGISTRY_CS))

    @cs_registry.register("MY_SYSTEM")
    class MySystem(ClassificationSystem):
        def get_prefixes(self, code):
            return [code]

        def get_root_categories(self):
            return []

    assert cs_registry._REGISTRY_CS["MY_SYSTEM"] is MySystem


def test_bundled_icatus_class_works_for_fiction(monkeypatch):
    """Prefix codes need no new class: the ICATUS system + JSON loader is enough."""
    from pathlib import Path
    FICTION_JSON = Path(__file__).parent / "resources" / "fiction.json"
    monkeypatch.setitem(mapping, "FICTION_VIA_ICATUS", {
        "classification_system": "ICATUS", "loader_name": "JSON",
        "data_path": FICTION_JSON, "metadata": {},
    })
    cs = get_classification_system("FICTION_VIA_ICATUS")
    assert [c.code for c in cs.get_children("1")] == ["11", "12", "13"]
