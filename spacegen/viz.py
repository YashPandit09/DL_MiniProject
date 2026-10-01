"""2D floor plans of layouts (T06): walls, door, clearance zone, walkable area and furniture.

Render a saved layout JSON:  python -m spacegen.viz layout.json layout.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from spacegen import geometry
from spacegen.catalog import RoomCatalog, load_room_catalog
from spacegen.layout import Layout, layout_from_dict
from spacegen.rules import CheckResult, Rules, check_layout, door_geometry, load_rules, reachability

# Colours from the data-viz reference palette (light mode). Every item is named by a direct
# label, so all furniture shares one hue; the status red is kept for items that break a
# hard check, always together with the check's name in the label.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
WALKABLE = "#f0efec"
FURNITURE_FILL = "#cde2fb"
FURNITURE_EDGE = "#2a78d6"
FRONT_EDGE = "#1c5cab"
CRITICAL = "#d03b3b"

# Sequential blue ramp (steps 100 to 700 of the reference palette), starting at the surface
# so that zero coverage disappears into the background.
COVERAGE = LinearSegmentedColormap.from_list("coverage", [
    SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

_WALL_ENDS = {"S": ((0, 0), (1, 0)), "E": ((1, 0), (1, 1)), "N": ((1, 1), (0, 1)), "W": ((0, 1), (0, 0))}


def plot_layout(layout: Layout, catalog: RoomCatalog, rules: Rules, title: str | None = None,
                show_walkable: bool = True) -> Figure:
    """A figure with one floor plan. Matplotlib's object API is used, so no global state."""
    fig = Figure(figsize=(1.6 + 1.3 * layout.width, 2.4 + 1.3 * layout.depth), dpi=150, facecolor=SURFACE,
                 layout="constrained")
    ax = fig.add_subplot()
    result = draw_layout(ax, layout, catalog, rules, show_walkable)
    fig.suptitle(title or _summary(layout, catalog, result), x=0.02, ha="left", fontsize=9, color=INK,
                 linespacing=1.4)
    legend = [Patch(facecolor=WALKABLE, edgecolor=INK_MUTED, linewidth=0.5, label="walkable from the door"),
              Patch(facecolor="none", edgecolor=INK_MUTED, hatch="///", label="door clearance zone"),
              Line2D([], [], color=FRONT_EDGE, linewidth=3, label="front of an item")]
    if not result.valid:
        legend.append(Patch(facecolor=FURNITURE_FILL, edgecolor=CRITICAL, linewidth=2, label="breaks a hard check"))
    fig.legend(handles=legend, loc="outside lower center", ncol=2, frameon=False, fontsize=7,
               labelcolor=INK_SECONDARY)
    return fig


def draw_layout(ax: Axes, layout: Layout, catalog: RoomCatalog, rules: Rules,
                show_walkable: bool = True) -> CheckResult:
    """Draw one floor plan onto existing axes; returns the hard-check result it shows."""
    result = check_layout(layout, catalog, rules)
    w, d = layout.width, layout.depth
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.3, w + 0.3)
    ax.set_ylim(-0.3, d + 0.3)
    ax.set_aspect("equal")

    if show_walkable:
        reach = reachability(layout, catalog, rules)
        ax.imshow(reach.reached.T, origin="lower", extent=(0, w, 0, d), interpolation="nearest",
                  cmap=ListedColormap([SURFACE, WALKABLE]), vmin=0, vmax=1, zorder=0)

    door = door_geometry(w, d, layout.door_wall, layout.door_offset, rules.door)
    lo, _ = geometry.box_bounds(door.zone_center, door.zone_size)
    ax.add_patch(Rectangle(lo, *door.zone_size, facecolor="none", edgecolor=INK_MUTED, hatch="///",
                           linewidth=0.6, zorder=1))
    _draw_walls(ax, layout, door, rules.door.width)

    broken = _broken_checks(result)
    eff = layout.eff_size
    for k in np.flatnonzero(layout.mask):
        slot = catalog.slots[k]
        lo, _ = geometry.box_bounds(layout.center[k], eff[k])
        failing = broken.get(int(k), [])
        ax.add_patch(Rectangle(lo, *eff[k], facecolor=FURNITURE_FILL, linewidth=2.0 if failing else 1.0,
                               edgecolor=CRITICAL if failing else FURNITURE_EDGE, zorder=2))
        if slot.rot_symmetry == 1:  # symmetric items have no meaningful front
            _draw_front(ax, layout.center[k], eff[k], int(layout.rot[k]))
        label = slot.name.replace("_", " ") + (f"\n({', '.join(failing)})" if failing else "")
        ax.text(*layout.center[k], label, ha="center", va="center", fontsize=6.5, color=INK, zorder=4,
                rotation=90 if eff[k, 0] < eff[k, 1] else 0)  # run the label along a tall, thin item

    ax.tick_params(labelsize=7, colors=INK_MUTED, length=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return result


def plot_raster(raster, canvas: float, title: str | None = None) -> Figure:
    """The four channels of one evaluator input (T07) side by side, in meters on the canvas."""
    from spacegen.raster import CHANNELS

    image = np.asarray(raster.detach().cpu() if hasattr(raster, "detach") else raster)
    fig = Figure(figsize=(12, 3.6), dpi=150, facecolor=SURFACE, layout="constrained")
    for c, (ax, name) in enumerate(zip(fig.subplots(1, len(CHANNELS)), CHANNELS)):
        channel = image[c].T  # rows are y, so the picture matches the floor plan
        shown = ax.imshow(channel, origin="lower", extent=(0, canvas, 0, canvas), cmap=COVERAGE,
                          vmin=0, vmax=max(1.0, float(channel.max())), interpolation="nearest")
        ax.set_title(f"{c}: {name}", fontsize=8, color=INK, loc="left")
        ax.tick_params(labelsize=6, colors=INK_MUTED, length=2)
        for spine in ax.spines.values():
            spine.set_color(INK_MUTED)
            spine.set_linewidth(0.5)
        bar = fig.colorbar(shown, ax=ax, shrink=0.8)
        bar.ax.tick_params(labelsize=6, colors=INK_MUTED)
        bar.outline.set_visible(False)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=9, color=INK)
    return fig


def _draw_walls(ax: Axes, layout: Layout, door, door_width: float) -> None:
    room = layout.room
    for wall, (start, end) in _WALL_ENDS.items():
        a, b = np.array(start) * room, np.array(end) * room
        if wall == layout.door_wall:  # leave the door opening out of the wall
            along = (b - a) / np.linalg.norm(b - a)
            segments = [(a, door.center - along * door_width / 2), (door.center + along * door_width / 2, b)]
            ax.text(*(door.center - door.inward * 0.15), "door", ha="center", va="center",
                    fontsize=6.5, color=INK_SECONDARY)
        else:
            segments = [(a, b)]
        for p, q in segments:
            ax.plot([p[0], q[0]], [p[1], q[1]], color=INK, linewidth=2.5, solid_capstyle="projecting", zorder=3)


def _draw_front(ax: Axes, center: np.ndarray, eff: np.ndarray, rot: int) -> None:
    """A thick line along the item's front face."""
    middle = geometry.front_point(center, eff, rot)
    along = np.array([1.0, 0.0]) if rot % 2 == 0 else np.array([0.0, 1.0])
    half = eff[0 if rot % 2 == 0 else 1] / 2
    p, q = middle - along * half, middle + along * half
    ax.plot([p[0], q[0]], [p[1], q[1]], color=FRONT_EDGE, linewidth=2.5, solid_capstyle="butt", zorder=3)


def _broken_checks(result: CheckResult) -> dict[int, list[str]]:
    """Slot -> the hard checks it breaks."""
    broken: dict[int, list[str]] = {}
    for check, slots in [("H1", result.items_out_of_room),
                         ("H2", sorted({k for pair in result.overlapping_pairs for k in pair})),
                         ("H3", result.items_in_door_zone),
                         ("H4", result.unreachable_items)]:
        for k in slots:
            broken.setdefault(int(k), []).append(check)
    return broken


def _summary(layout: Layout, catalog: RoomCatalog, result: CheckResult) -> str:
    def status(ok: bool, slots) -> str:
        return "pass" if ok else "fail (" + ", ".join(catalog.slots[k].name.replace("_", " ") for k in slots) + ")"

    room = layout.room_type.replace("_", " ")
    checks = [("H1 in room", result.in_room, result.items_out_of_room),
              ("H2 no overlap", result.no_overlap, sorted({k for p in result.overlapping_pairs for k in p})),
              ("H3 door clear", result.door_clear, result.items_in_door_zone),
              ("H4 reachable", result.reachable, result.unreachable_items)]
    lines = [f"{room}, {layout.width:.2f} x {layout.depth:.2f} m (axes in meters), "
             f"door on the {layout.door_wall} wall"]
    lines += [f"{name}: {status(ok, slots)}" for name, ok, slots in checks]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a layout JSON as a floor-plan PNG.")
    parser.add_argument("layout_json", type=Path)
    parser.add_argument("png", type=Path)
    args = parser.parse_args(argv)
    data = json.loads(args.layout_json.read_text(encoding="utf-8"))
    catalog = load_room_catalog(data["room"]["type"])
    fig = plot_layout(layout_from_dict(data, catalog), catalog, load_rules())
    fig.savefig(args.png, facecolor=SURFACE)
    print(f"wrote {args.png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
