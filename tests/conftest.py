"""Shared fixtures: the real catalog and rules, and a hand-built valid living room."""
import pytest

from spacegen.catalog import load_room_catalog
from spacegen.layout import make_layout
from spacegen.rules import load_rules
from tests.layouts import GOOD_ITEMS


@pytest.fixture(scope="session")
def catalog():
    return load_room_catalog("living_room")


@pytest.fixture(scope="session")
def rules():
    return load_rules()


@pytest.fixture
def good_layout(catalog):
    return make_layout(catalog, 5.0, 4.0, "W", 0.5, GOOD_ITEMS)
