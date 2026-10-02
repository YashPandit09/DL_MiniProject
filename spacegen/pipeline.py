"""Inference: from a room request to layouts (Tech Spec 5). T21 adds M1 sampling; T25 adds the
feasibility check, latent optimization, checks, ranking and the top 3.

CVAESampler is M1 (Tech Spec 4.3): z ~ N(0, I), decoded with the CVAE, positions clamped to the
room, each rotation the arg-max of its logits in canonical form. It has the baselines'
sample(cond, n, rng) interface, so spacegen.evaluate runs it like B1, B2 and G0.
"""
from __future__ import annotations

import numpy as np
import torch

from spacegen.baselines import Samples
from spacegen.catalog import RoomCatalog
from spacegen.dataset import LayoutBatch, decode_targets, encode_conditions, stack_layouts
from spacegen.generator import Condition
from spacegen.layout import make_layout
from spacegen.models.cvae import CVAE


def condition_batch(cond: Condition, n: int, catalog: RoomCatalog) -> LayoutBatch:
    """n copies of the room and its items, positions not yet known (all zero)."""
    blank = make_layout(catalog, cond.width, cond.depth, cond.door_wall, cond.door_offset,
                        {name: (0.0, 0.0, 0) for name in cond.items}, cond.items)
    return stack_layouts([blank] * n, catalog)


class CVAESampler:
    """M1: layouts decoded from prior samples of the CVAE (BatchNorm in inference mode)."""

    name = "M1"

    def __init__(self, model: CVAE, catalog: RoomCatalog, device: str | torch.device = "cpu"):
        self.model, self.catalog, self.device = model.to(device).eval(), catalog, device

    def sample(self, cond: Condition, n: int, rng: np.random.Generator) -> Samples:
        blank = condition_batch(cond, n, self.catalog)
        c = torch.as_tensor(encode_conditions(blank, self.catalog), device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(int(rng.integers(2 ** 62)))
        positions, logits = self.model.generate(c, generator)
        decoded = decode_targets(positions, logits, blank, self.catalog)
        return Samples([decoded.layout(i, self.catalog) for i in range(n)], n)
