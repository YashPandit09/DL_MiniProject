"""Rule engine (Tech Spec Section 3): the rules config, door geometry and hard checks H1 to H3.

configs/rules.yaml holds every threshold; all of them are our own design assumptions.
Reachability (H4) is added in T08 and the quality score in T09.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from spacegen import geometry
from spacegen.layout import Layout
from spacegen.paths import CONFIG_DIR

RULES_PATH = CONFIG_DIR / "rules.yaml"

# Walls in the order of the rotation classes: the N wall's outward normal is facing class 0.
WALLS = ("N", "E", "S", "W")

# Walking the walls counter-clockwise (Tech Spec Section 1): the start corner as a fraction
# of (W, D), the direction along the wall, and the inward normal.
_WALL_FRAME = {
    "S": ((0, 0), (1, 0), (0, 1)),
    "E": ((1, 0), (0, 1), (-1, 0)),
    "N": ((1, 1), (-1, 0), (0, -1)),
    "W": ((0, 1), (0, -1), (1, 0)),
}

_QUALITY_TERMS = {"align", "relations", "circulation", "space"}


class RulesError(ValueError):
    """The rules file breaks one of the checks in load_rules()."""


@dataclass(frozen=True)
class DoorRules:
    width: float
    clearance_depth: float
    corner_margin: float  # m_d


@dataclass(frozen=True)
class HardCheckRules:
    in_room_tol: float
    max_pair_overlap: float
    door_overlap_tol: float


@dataclass(frozen=True)
class ReachabilityRules:
    cell: float
    min_path_width: float
    access_offset: float
    access_tolerance: float


@dataclass(frozen=True)
class QualityRules:
    weights: dict[str, float]
    wall_distance: float
    space_band: tuple[float, float]


@dataclass(frozen=True)
class RoomRules:
    width: tuple[float, float]  # sampling range of W
    depth: tuple[float, float]  # sampling range of D
    against_wall: tuple[str, ...]
    relations: dict


@dataclass(frozen=True)
class Rules:
    door: DoorRules
    hard_checks: HardCheckRules
    reachability: ReachabilityRules
    f_max: float
    quality: QualityRules
    rooms: dict[str, RoomRules]


def load_rules(path: Path = RULES_PATH) -> Rules:
    """Read and check the rules file."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    try:
        quality = dict(raw["quality"])
        rules = Rules(
            door=DoorRules(**raw["door"]),
            hard_checks=HardCheckRules(**raw["hard_checks"]),
            reachability=ReachabilityRules(**raw["reachability"]),
            f_max=raw["feasibility"]["f_max"],
            quality=QualityRules(weights=dict(quality.pop("weights")),
                                 space_band=tuple(quality.pop("space_band")), **quality),
            rooms={name: RoomRules(width=tuple(r["width"]), depth=tuple(r["depth"]),
                                   against_wall=tuple(r["against_wall"]), relations=dict(r["relations"]))
                   for name, r in raw["rooms"].items()},
        )
    except (KeyError, TypeError) as e:
        raise RulesError(f"missing or unexpected field ({e})") from e
    _check(rules)
    return rules


def _check(rules: Rules) -> None:
    door = rules.door
    if min(door.width, door.clearance_depth) <= 0:
        raise RulesError("door width and clearance depth must be positive")
    if door.corner_margin < door.width / 2:
        raise RulesError("door.corner_margin must be at least half the door width")
    weights = rules.quality.weights
    if set(weights) != _QUALITY_TERMS or min(weights.values()) < 0 or abs(sum(weights.values()) - 1) > 1e-9:
        raise RulesError(f"quality.weights needs the terms {sorted(_QUALITY_TERMS)}, non-negative, summing to 1")
    _check_range("quality.space_band", rules.quality.space_band)
    for name, room in rules.rooms.items():
        _check_range(f"{name}.width", room.width)
        _check_range(f"{name}.depth", room.depth)
        if min(room.width[0], room.depth[0]) < 2 * door.corner_margin:
            raise RulesError(f"{name}: the shortest wall must fit a door (at least {2 * door.corner_margin} m)")
        for relation, value in room.relations.items():
            if isinstance(value, list):
                _check_range(f"{name}.relations.{relation}", value)


def _check_range(name: str, value) -> None:
    if len(value) != 2 or not 0 <= value[0] < value[1]:
        raise RulesError(f"{name} must be [low, high] with 0 <= low < high, got {list(value)}")


# --------------------------------------------------------------------------- door

@dataclass(frozen=True, eq=False)
class DoorGeometry:
    center: np.ndarray  # middle of the door opening, on the wall
    inward: np.ndarray  # unit vector from the door into the room
    zone_center: np.ndarray  # clearance zone: door width along the wall, clearance depth into the room
    zone_size: np.ndarray  # its extent along x and y


def door_geometry(width: float, depth: float, wall: str, offset: float, door: DoorRules) -> DoorGeometry:
    """The door and its clearance zone (Tech Spec Section 1).

    The door centre lies s = m_d + (L - 2 m_d) * o from the wall's start corner, so it is
    at least m_d from both corners, and its edges at least m_d - width/2.
    """
    if wall not in _WALL_FRAME:
        raise ValueError(f"door wall must be one of {WALLS}, got {wall!r}")
    if not 0.0 <= offset <= 1.0:
        raise ValueError(f"door offset must be in [0, 1], got {offset}")
    length = width if wall in ("S", "N") else depth
    if length < 2 * door.corner_margin:
        raise ValueError(f"the {wall} wall ({length} m) is too short for a door (needs {2 * door.corner_margin} m)")
    start, along, inward = (np.array(v, dtype=float) for v in _WALL_FRAME[wall])
    s = door.corner_margin + (length - 2 * door.corner_margin) * offset
    center = start * np.array([width, depth]) + along * s
    zone_size = np.abs(along) * door.width + np.abs(inward) * door.clearance_depth
    return DoorGeometry(center, inward, center + inward * door.clearance_depth / 2, zone_size)


# --------------------------------------------------------------------------- hard checks

@dataclass(frozen=True)
class CheckResult:
    """Which hard checks a layout passes, with the slots that break each one."""

    items_out_of_room: tuple[int, ...]  # H1
    overlapping_pairs: tuple[tuple[int, int], ...]  # H2
    items_in_door_zone: tuple[int, ...]  # H3

    @property
    def in_room(self) -> bool:
        return not self.items_out_of_room

    @property
    def no_overlap(self) -> bool:
        return not self.overlapping_pairs

    @property
    def door_clear(self) -> bool:
        return not self.items_in_door_zone

    @property
    def valid(self) -> bool:
        """H1 to H3 hold (reachability, H4, joins in T08)."""
        return self.in_room and self.no_overlap and self.door_clear


def check_layout(layout: Layout, rules: Rules) -> CheckResult:
    """Run the hard checks H1 to H3 (Tech Spec 3.1) on the present items."""
    limits = rules.hard_checks
    present, eff = layout.mask, layout.eff_size

    inside = geometry.inside_room(layout.center, eff, layout.room, tol=limits.in_room_tol)
    out_of_room = np.flatnonzero(present & ~inside)

    pair_area = geometry.pairwise_overlap_area(layout.center, eff, present)
    first, second = np.nonzero(np.triu(pair_area > limits.max_pair_overlap, k=1))

    door = door_geometry(layout.width, layout.depth, layout.door_wall, layout.door_offset, rules.door)
    zone_area = geometry.overlap_area(layout.center, eff, door.zone_center, door.zone_size)
    in_zone = np.flatnonzero(present & (zone_area > limits.door_overlap_tol))

    return CheckResult(
        items_out_of_room=tuple(out_of_room.tolist()),
        overlapping_pairs=tuple(zip(first.tolist(), second.tolist())),
        items_in_door_zone=tuple(in_zone.tolist()),
    )
