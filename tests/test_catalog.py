"""Catalog tests (T03): the living room matches Tech Spec 2.1, and a bad file fails loudly."""
import copy

import pytest
import yaml

from spacegen.catalog import CatalogError, load_catalog, load_room_catalog


@pytest.fixture(scope="module")
def living_room():
    return load_room_catalog("living_room")


def test_living_room_has_six_slots_in_layout_order(living_room):
    assert living_room.num_slots == 6
    assert [s.name for s in living_room.slots] == [
        "sofa", "tv_unit", "coffee_table", "bookshelf", "armchair", "side_table"]
    assert [s.index for s in living_room.slots] == list(range(6))


def test_only_sofa_and_tv_unit_are_mandatory(living_room):
    assert [s.name for s in living_room.slots if s.mandatory] == ["sofa", "tv_unit"]


def test_flags_match_the_tech_spec(living_room):
    symmetry = {s.name: s.rot_symmetry for s in living_room.slots}
    assert symmetry == {"sofa": 1, "tv_unit": 1, "coffee_table": 2, "bookshelf": 1,
                        "armchair": 1, "side_table": 4}
    assert [s.name for s in living_room.slots if not s.needs_access] == ["tv_unit"]
    assert living_room.groups == ()


def test_variants_and_default(living_room):
    sofa = living_room.slot(0)
    assert sofa.variant().id == "sofa_3seater"
    two_seater = sofa.variant("sofa_2seater")
    assert (two_seater.w, two_seater.d, two_seater.h, two_seater.price) == (1.60, 0.90, 0.85, 19000)
    assert two_seater.area == pytest.approx(1.44)
    with pytest.raises(KeyError):
        sofa.variant("sofa_9seater")


def test_slot_lookup_by_name_or_index(living_room):
    assert living_room.slot("armchair") is living_room.slot(4)
    with pytest.raises(KeyError):
        living_room.slot("piano")


def test_unknown_room_type():
    with pytest.raises(KeyError):
        load_room_catalog("kitchen")


# --------------------------------------------------------------------------- invalid files

def _slot(index, name):
    variant = {"id": f"{name}_v", "w": 1.0, "d": 0.5, "h": 0.5, "price": 100}
    return {"slot": index, "name": name, "mandatory": False, "rot_symmetry": 1,
            "needs_access": True, "variants": [variant]}


VALID = {"room": {"slots": [_slot(0, "a"), _slot(1, "b"), _slot(2, "c")], "groups": []}}


def _load(tmp_path, catalog):
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(catalog), encoding="utf-8")
    return load_catalog(path)


def test_identical_slots_can_be_grouped(tmp_path):
    catalog = copy.deepcopy(VALID)
    catalog["room"]["groups"] = [[1, 2]]
    assert _load(tmp_path, catalog)["room"].groups == ((1, 2),)


def _renumber_slot(c):
    c["room"]["slots"][1]["slot"] = 5


def _bad_symmetry(c):
    c["room"]["slots"][0]["rot_symmetry"] = 3


def _zero_width(c):
    c["room"]["slots"][0]["variants"][0]["w"] = 0


def _repeat_name(c):
    c["room"]["slots"][2]["name"] = "a"


def _repeat_variant_id(c):
    c["room"]["slots"][2]["variants"][0]["id"] = "a_v"


def _drop_flag(c):
    del c["room"]["slots"][0]["needs_access"]


def _group_different_sizes(c):
    c["room"]["groups"] = [[1, 2]]
    c["room"]["slots"][2]["variants"][0]["w"] = 2.0


def _group_unknown_slot(c):
    c["room"]["groups"] = [[1, 7]]


def _slot_in_two_groups(c):
    c["room"]["groups"] = [[0, 1], [1, 2]]


@pytest.mark.parametrize("break_catalog, message", [
    (_renumber_slot, "must equal its position"),
    (_bad_symmetry, "rot_symmetry must be 1, 2 or 4"),
    (_zero_width, "positive sizes"),
    (_repeat_name, "slot names must be unique"),
    (_repeat_variant_id, "variant ids must be unique"),
    (_drop_flag, "missing or unexpected field"),
    (_group_different_sizes, "differ in size or flags"),
    (_group_unknown_slot, "valid slot numbers"),
    (_slot_in_two_groups, "more than one group"),
])
def test_invalid_files_fail_loudly(tmp_path, break_catalog, message):
    catalog = copy.deepcopy(VALID)
    break_catalog(catalog)
    with pytest.raises(CatalogError, match=message):
        _load(tmp_path, catalog)
