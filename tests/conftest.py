"""Shared fixtures: the real catalog and rules, and a hand-built valid living room."""
import pytest

from spacegen.catalog import load_room_catalog
from spacegen.layout import make_layout
from spacegen.rules import load_rules

# Slot numbers of the living room (tests/test_catalog.py checks this order).
SOFA, TV_UNIT, COFFEE_TABLE, BOOKSHELF, ARMCHAIR, SIDE_TABLE = range(6)

# A 5.0 x 4.0 m room with the door in the middle of the west wall: door centre (0, 2.0),
# clearance zone x 0 to 0.9, y 1.55 to 2.45. Every item has its default variant.
GOOD_ITEMS = {
    "sofa": (2.8, 3.55, 2),  # against the north wall, facing south
    "tv_unit": (2.8, 0.2, 0),  # against the south wall, facing the sofa 2.7 m away
    "coffee_table": (2.8, 2.425, 0),  # 0.40 m in front of the sofa
    "bookshelf": (4.85, 1.5, 3),  # against the east wall, facing west
    "armchair": (4.2, 2.6, 3),  # at 90 degrees to the sofa, beside the coffee table
    "side_table": (1.475, 3.775, 0),  # 0.05 m from the sofa's west end
}


@pytest.fixture(scope="session")
def catalog():
    return load_room_catalog("living_room")


@pytest.fixture(scope="session")
def rules():
    return load_rules()


@pytest.fixture
def good_layout(catalog):
    return make_layout(catalog, 5.0, 4.0, "W", 0.5, GOOD_ITEMS)
