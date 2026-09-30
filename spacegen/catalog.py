"""Furniture catalog: slots, variants, prices and per-slot flags (Tech Spec Section 2.1).

configs/catalog.yaml lists each room type's slots in layout-vector order. The values are
illustrative placeholders (assumption), not retail data. load_catalog() checks the file,
so a mistake in it fails loudly instead of producing a subtly wrong dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from spacegen.paths import CONFIG_DIR

CATALOG_PATH = CONFIG_DIR / "catalog.yaml"


class CatalogError(ValueError):
    """The catalog file breaks one of the rules load_catalog() checks."""


@dataclass(frozen=True)
class Variant:
    id: str
    w: float  # width along the front face (m)
    d: float  # depth, front to back (m)
    h: float  # height (m)
    price: int  # INR

    @property
    def area(self) -> float:
        """Footprint area in m^2."""
        return self.w * self.d


@dataclass(frozen=True)
class Slot:
    index: int  # position in the layout vector
    name: str
    mandatory: bool
    rot_symmetry: int  # 1, 2 or 4 (Tech Spec Section 1)
    needs_access: bool  # False: skipped by the reachability check H4
    variants: tuple[Variant, ...]

    def variant(self, variant_id: str | None = None) -> Variant:
        """The variant with this id, or the default (first) one."""
        if variant_id is None:
            return self.variants[0]
        for v in self.variants:
            if v.id == variant_id:
                return v
        raise KeyError(f"slot {self.name!r} has no variant {variant_id!r}")


@dataclass(frozen=True)
class RoomCatalog:
    room_type: str
    slots: tuple[Slot, ...]
    groups: tuple[tuple[int, ...], ...]  # interchangeable slots

    @property
    def num_slots(self) -> int:
        """K in the Tech Spec: the number of slots in the layout vector."""
        return len(self.slots)

    def slot(self, key: int | str) -> Slot:
        """A slot by index or by name."""
        if isinstance(key, int):
            return self.slots[key]
        for s in self.slots:
            if s.name == key:
                return s
        raise KeyError(f"{self.room_type} has no slot {key!r}")


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, RoomCatalog]:
    """Read and check every room type in the catalog file."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    catalogs = {room_type: _parse_room(room_type, spec) for room_type, spec in raw.items()}
    ids = [v.id for c in catalogs.values() for s in c.slots for v in s.variants]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise CatalogError(f"variant ids must be unique, repeated: {duplicates}")
    return catalogs


def load_room_catalog(room_type: str, path: Path = CATALOG_PATH) -> RoomCatalog:
    """The catalog of one room type."""
    catalogs = load_catalog(path)
    if room_type not in catalogs:
        raise KeyError(f"no room type {room_type!r} in {path}; found {sorted(catalogs)}")
    return catalogs[room_type]


def _parse_room(room_type: str, spec: dict) -> RoomCatalog:
    slots = tuple(_parse_slot(f"{room_type}.slots[{i}]", i, s) for i, s in enumerate(spec["slots"]))
    names = [s.name for s in slots]
    if len(set(names)) != len(names):
        raise CatalogError(f"{room_type}: slot names must be unique, got {names}")
    groups = tuple(tuple(g) for g in spec.get("groups") or [])
    _check_groups(room_type, slots, groups)
    return RoomCatalog(room_type, slots, groups)


def _parse_slot(where: str, position: int, spec: dict) -> Slot:
    try:
        variants = tuple(Variant(**v) for v in spec["variants"])
        slot = Slot(index=spec["slot"], name=spec["name"], mandatory=bool(spec["mandatory"]),
                    rot_symmetry=spec["rot_symmetry"], needs_access=bool(spec["needs_access"]),
                    variants=variants)
    except (KeyError, TypeError) as e:
        raise CatalogError(f"{where}: missing or unexpected field ({e})") from e
    if slot.index != position:
        raise CatalogError(f"{where}: slot number {slot.index} must equal its position {position}")
    if slot.rot_symmetry not in (1, 2, 4):
        raise CatalogError(f"{where}: rot_symmetry must be 1, 2 or 4, got {slot.rot_symmetry}")
    if not variants:
        raise CatalogError(f"{where}: needs at least one variant")
    for v in variants:
        if min(v.w, v.d, v.h) <= 0 or v.price < 0:
            raise CatalogError(f"{where}: variant {v.id!r} needs positive sizes and a price of at least 0")
    return slot


def _check_groups(room_type: str, slots: tuple[Slot, ...], groups: tuple[tuple[int, ...], ...]) -> None:
    """Grouped slots must be identical apart from their names, so swapping them changes nothing."""
    seen: set[int] = set()
    for group in groups:
        if len(group) < 2 or any(k not in range(len(slots)) for k in group):
            raise CatalogError(f"{room_type}: group {list(group)} needs two or more valid slot numbers")
        if seen & set(group):
            raise CatalogError(f"{room_type}: slots {sorted(seen & set(group))} are in more than one group")
        seen |= set(group)
        first = slots[group[0]]
        for k in group[1:]:
            other = slots[k]
            same_sizes = [(v.w, v.d, v.h) for v in other.variants] == [(v.w, v.d, v.h) for v in first.variants]
            same_flags = (other.rot_symmetry, other.needs_access) == (first.rot_symmetry, first.needs_access)
            if not (same_sizes and same_flags):
                raise CatalogError(f"{room_type}: slots {group[0]} and {k} are grouped but differ in size or flags")
