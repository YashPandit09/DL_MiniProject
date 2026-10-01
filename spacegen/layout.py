"""A furniture layout as plain arrays, one row per catalog slot (Tech Spec 2.3), in meters."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from spacegen import geometry
from spacegen.catalog import RoomCatalog


@dataclass(frozen=True, eq=False)
class Layout:
    room_type: str
    width: float  # W, extent along x
    depth: float  # D, extent along y
    door_wall: str  # N, E, S or W
    door_offset: float  # o in [0, 1] (Tech Spec Section 1)
    center: np.ndarray  # (K, 2) item centres
    rot: np.ndarray  # (K,) rotation classes 0 to 3
    size: np.ndarray  # (K, 2) catalog (w, d) of each item; 0 for absent slots
    mask: np.ndarray  # (K,) True where the slot is present
    variant_ids: tuple  # (K,) catalog variant ids; None for absent slots

    @property
    def room(self) -> np.ndarray:
        """(W, D)."""
        return np.array([self.width, self.depth])

    @property
    def eff_size(self) -> np.ndarray:
        """(K, 2) footprint extents along x and y after rotation."""
        return geometry.effective_size(self.size, self.rot)

    def with_item(self, slot: int, center=None, rot: int | None = None) -> Layout:
        """A copy with one present item moved and/or turned."""
        if not self.mask[slot]:
            raise ValueError(f"slot {slot} is absent")
        new_center, new_rot = self.center.copy(), self.rot.copy()
        if center is not None:
            new_center[slot] = center
        if rot is not None:
            new_rot[slot] = _check_rot(rot)
        return dataclasses.replace(self, center=new_center, rot=new_rot)

    def without(self, slot: int) -> Layout:
        """A copy with one slot marked absent."""
        mask = self.mask.copy()
        mask[slot] = False
        return dataclasses.replace(self, mask=mask)


def make_layout(catalog: RoomCatalog, width: float, depth: float, door_wall: str,
                door_offset: float, items: dict[str, tuple[float, float, int]],
                variants: dict[str, str] | None = None) -> Layout:
    """Build a layout from {slot name: (x, y, rot)}; slots not named are absent.

    `variants` maps slot names to variant ids; other present slots get their default variant.
    """
    variants = variants or {}
    unused = set(variants) - set(items)
    if unused:
        raise KeyError(f"variants given for slots that are not in the layout: {sorted(unused)}")
    k = catalog.num_slots
    center, size = np.zeros((k, 2)), np.zeros((k, 2))
    rot, mask = np.zeros(k, dtype=np.int64), np.zeros(k, dtype=bool)
    ids: list[str | None] = [None] * k
    for name, (x, y, r) in items.items():
        slot = catalog.slot(name)
        variant = slot.variant(variants.get(name))
        center[slot.index] = (x, y)
        rot[slot.index] = _check_rot(r)
        size[slot.index] = (variant.w, variant.d)
        mask[slot.index] = True
        ids[slot.index] = variant.id
    return Layout(catalog.room_type, float(width), float(depth), door_wall, float(door_offset),
                  center, rot, size, mask, tuple(ids))


def _check_rot(rot: int) -> int:
    if rot not in range(4):
        raise ValueError(f"rotation class must be 0, 1, 2 or 3, got {rot}")
    return int(rot)
