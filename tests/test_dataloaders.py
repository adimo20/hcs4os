"""Data loaders: registry, base-class behaviour and every built-in loader."""
import json

import pytest

from hcs4os._shared.Code import Code
from hcs4os.classification_system.mapping import make_path
from hcs4os.dataloaders import registry as dl_registry
from hcs4os.dataloaders.BaseDataLoader import ClassificationLoader
from hcs4os.dataloaders.registry import get_loader, register


# ---------------------------------------------------------------------------
# registry and base class
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["KLASS_SERVER", "COICOP", "ICATUS", "NACE", "JSON"])
def test_builtin_loaders_are_registered(name):
    assert name in dl_registry._REGISTRY_DL


def test_unknown_loader_raises_value_error_listing_known_loaders():
    with pytest.raises(ValueError, match="Unknown format 'NOPE'") as exc:
        get_loader("NOPE", make_path("sea2021.xml"))
    assert "KLASS_SERVER" in str(exc.value)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Classification source not found"):
        get_loader("JSON", tmp_path / "does_not_exist.json")


def test_register_decorator_adds_loader(monkeypatch):
    monkeypatch.setattr(dl_registry, "_REGISTRY_DL", dict(dl_registry._REGISTRY_DL))

    @register("MY_FORMAT")
    class MyLoader(ClassificationLoader):
        def to_records(self):
            return [{"code": "1", "description": "x", "level": 1}]

    assert dl_registry._REGISTRY_DL["MY_FORMAT"] is MyLoader


def test_load_converts_records_to_code_objects(tmp_path):
    path = tmp_path / "codes.json"
    path.write_text(json.dumps([{"code": "1", "description": "a", "level": 1, "extra": "dropped"}]))
    codes = get_loader("JSON", path).load()
    assert codes == [Code(code="1", description="a", level=1)]


def test_loader_can_not_be_instantiated_without_to_records(tmp_path):
    with pytest.raises(TypeError):
        ClassificationLoader(tmp_path)  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# built-in loaders on the bundled source files
# ---------------------------------------------------------------------------

def test_klass_server_loader_parses_sea():
    codes = get_loader("KLASS_SERVER", make_path("sea2021.xml")).load()
    by_code = {c.code: c for c in codes}
    assert len(codes) > 2000
    assert by_code["01"].description == "Nahrungsmittel und alkoholfreie Getränke"
    assert by_code["0111 1"].level == 4
    assert all(isinstance(c.level, int) for c in codes)


def test_klass_server_loader_drops_empty_detail_keys():
    for c in get_loader("KLASS_SERVER", make_path("sea2021.xml")).load():
        for value in c.details.values():
            assert value not in ("", [], [""], {})


def test_klass_server_loader_keeps_exclusions():
    codes = get_loader("KLASS_SERVER", make_path("sea2021.xml")).load()
    with_exclusions = [c for c in codes
                       if c.details.get("ExplanatoryNote", {}).get("exclusions")]
    assert with_exclusions, "expected some SEA codes with 'Ausgeschlossen sind' notes"


def test_coicop_loader():
    codes = get_loader("COICOP", make_path("COICOP_2018_English_structure.xlsx")).load()
    by_code = {c.code: c for c in codes}
    assert by_code["01"].level == 1
    assert by_code["01.1.1.1"].level == 4
    assert set(by_code["01.1.1.1"].details) == {"includes", "alsoIncludes", "excludes"}
    assert "\xa0" not in by_code["01.1.1.1"].description


def test_icatus_loader_flattens_three_code_columns():
    codes = get_loader("ICATUS", make_path("UNSD_Time_Use_Statistics.csv")).load()
    by_code = {c.code: c for c in codes}
    assert len(codes) == 230
    assert by_code["1"].level == 1 and by_code["11"].level == 2 and by_code["110"].level == 3
    assert set(by_code["110"].details) == {"includes", "excludes", "keywords"}


def test_nace_loader():
    codes = get_loader("NACE", make_path("NACE_Rev2.1_Structure_Explanatory_Notes_EN.xlsx")).load()
    assert len(codes) > 500
    assert all(isinstance(c.level, int) for c in codes)
    assert all("parent_code" in c.details for c in codes)


def test_json_loader_on_bundled_rephrased_sea():
    codes = get_loader("JSON", make_path("sea_2021_augmented.json")).load()
    assert len(codes) > 2000
    assert any(c.detailled_description for c in codes)
