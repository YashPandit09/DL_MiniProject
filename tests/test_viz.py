"""Plot tests (T06): any layout JSON renders to a PNG, and broken checks are named on the item."""
import json

from matplotlib.colors import to_hex

from tests.layouts import ARMCHAIR, SOFA
from spacegen.layout import layout_to_dict
from spacegen.viz import CRITICAL, main, plot_layout


def _labels(fig):
    return [t.get_text() for t in fig.axes[0].texts]


def _red_patches(fig):
    return [p for p in fig.axes[0].patches if to_hex(p.get_edgecolor()) == CRITICAL]


def test_layout_json_renders_to_png(tmp_path, good_layout, catalog, rules):
    source = tmp_path / "layout.json"
    source.write_text(json.dumps(layout_to_dict(good_layout, catalog, rules.door.width)), encoding="utf-8")
    target = tmp_path / "layout.png"
    assert main([str(source), str(target)]) == 0
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_valid_layout_shows_every_item_and_no_red(good_layout, catalog, rules):
    fig = plot_layout(good_layout, catalog, rules)
    assert {"sofa", "tv unit", "coffee table", "bookshelf", "armchair", "side table"} <= set(_labels(fig))
    assert "H4 reachable: pass" in fig.get_suptitle()
    assert _red_patches(fig) == []


def test_broken_check_is_named_on_the_item_and_in_the_title(good_layout, catalog, rules):
    walled = good_layout.without(ARMCHAIR).with_item(SOFA, center=(4.15, 1.5), rot=3)
    fig = plot_layout(walled, catalog, rules)
    assert "bookshelf\n(H4)" in _labels(fig)
    assert "H4 reachable: fail (bookshelf)" in fig.get_suptitle()
    assert len(_red_patches(fig)) == 1
