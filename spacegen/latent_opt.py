"""Latent optimization, M2 (T23, Tech Spec 5.2): repair decoded layouts by moving z.

The decoder is frozen and in eval() mode; only z moves, as a batch of independent problems:

  L_c(z) = lambda_overlap * sum_{i<j} pen_ij^2               (penetration depth, margin mu)
         + lambda_room    * sum_i out_i                       (protrusion out of the room)
         + lambda_door    * sum_i pen(i, door zone)^2
         + lambda_pin     * sum_{k pinned} ||p_k - p_k*||^2     (meters)
         + lambda_anchor  * 0.5 * ||z - z0||^2                (anchor to the candidate's start)
         - lambda_surrogate * q_hat                           (M3 only: a learned score)

Rotations are fixed before optimizing: the decoder's arg-max at z0, overwritten by a pin's
facing. Adam runs for `steps` steps; a candidate stops (its z is kept) once its constraint
terms, the anchor not counted, fall below `tolerance`. Adam is not monotone, so success is
judged by the final loss against the initial one.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from spacegen import geometry
from spacegen.config import DEFAULT_CONFIG, load_config
from spacegen.dataset import unpack_conditions
from spacegen.models.cvae import CVAE


@dataclass(frozen=True)
class LatentOptConfig:
    steps: int = 150
    lr: float = 0.05
    lambda_overlap: float = 10.0
    lambda_room: float = 10.0
    lambda_door: float = 10.0
    lambda_pin: float = 20.0
    lambda_anchor: float = 0.05
    lambda_surrogate: float = 0.0
    margin: float = 0.05
    tolerance: float = 1e-4


def load_latent_opt_config(path: Path = DEFAULT_CONFIG, **overrides) -> LatentOptConfig:
    return LatentOptConfig(**{**load_config(path)["latent_opt"], **overrides})


@dataclass(frozen=True)
class Problem:
    """What the candidates must satisfy, as tensors on the decoder's device (B candidates, K slots)."""
    c: torch.Tensor  # (B, 7 + 3K) condition vectors
    zone_center: torch.Tensor  # (B, 2) door clearance zone
    zone_size: torch.Tensor  # (B, 2)
    pin_mask: torch.Tensor | None = None  # (B, K) True for pinned items
    pin_center: torch.Tensor | None = None  # (B, K, 2) requested centres (m)
    pin_rot: torch.Tensor | None = None  # (B, K) requested facing, -1 to keep the decoder's


@dataclass
class Optimized:
    z: torch.Tensor  # (B, latent) the returned latent codes
    positions: torch.Tensor  # (B, K, 2) normalized centres decoded from z (unclamped)
    rot: torch.Tensor  # (B, K) the fixed rotation classes
    initial: torch.Tensor  # (B,) L_c at z0
    final: torch.Tensor  # (B,) L_c at the returned z
    steps: torch.Tensor  # (B,) steps each candidate took


def constraint_terms(positions: torch.Tensor, rot: torch.Tensor, problem: Problem, num_slots: int,
                     margin: float) -> dict[str, torch.Tensor]:
    """The constraint terms of L_c before weighting, one value per candidate (B,)."""
    room, mask, size = unpack_conditions(problem.c, num_slots)
    present = mask > 0.5
    center = positions * room[:, None, :]
    eff = geometry.effective_size(size, rot)
    terms = {
        "overlap": geometry.overlap_penalty(center, eff, margin=margin, mask=present),
        "room": (geometry.out_of_room(center, eff, room) * present).sum(dim=1),
        "door": (geometry.penetration_depth(center, eff, problem.zone_center[:, None, :], problem.zone_size[:, None, :],
                                            margin=margin) ** 2 * present).sum(dim=1),
    }
    if problem.pin_mask is not None:
        terms["pin"] = (((center - problem.pin_center) ** 2).sum(dim=-1) * problem.pin_mask).sum(dim=1)
    else:
        terms["pin"] = torch.zeros_like(terms["room"])
    return terms


def total_loss(terms: dict[str, torch.Tensor], z: torch.Tensor, z0: torch.Tensor, config: LatentOptConfig,
               surrogate: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(L_c per candidate, the constraint part alone) from the terms of constraint_terms()."""
    constraint = (config.lambda_overlap * terms["overlap"] + config.lambda_room * terms["room"]
                  + config.lambda_door * terms["door"] + config.lambda_pin * terms["pin"])
    loss = constraint + config.lambda_anchor * 0.5 * ((z - z0) ** 2).sum(dim=1)
    if surrogate is not None:
        loss = loss - config.lambda_surrogate * surrogate
    return loss, constraint


def initial_rotations(model: CVAE, z0: torch.Tensor, problem: Problem) -> torch.Tensor:
    """The decoder's arg-max rotation at z0, with pinned facings written over it."""
    with torch.no_grad():
        _, logits = model.decode(z0, problem.c)
    rot = logits.argmax(dim=-1)
    if problem.pin_rot is not None:
        rot = torch.where(problem.pin_rot >= 0, problem.pin_rot, rot)
    return rot


def optimize(model: CVAE, z0: torch.Tensor, problem: Problem, config: LatentOptConfig,
             surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None) -> Optimized:
    """Run latent optimization from z0 (B, latent). `surrogate(positions, rot)` returns a score
    per candidate to reward (M3); it is used only when lambda_surrogate > 0."""
    model.eval()
    k = model.config.num_slots
    rot = initial_rotations(model, z0, problem)
    use_surrogate = surrogate is not None and config.lambda_surrogate > 0

    def evaluate(z: torch.Tensor):
        positions, _ = model.decode(z, problem.c)
        terms = constraint_terms(positions, rot, problem, k, config.margin)
        score = surrogate(positions, rot) if use_surrogate else None
        loss, constraint = total_loss(terms, z, z0, config, score)
        return positions, loss, constraint

    z = z0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=config.lr)
    done = torch.zeros(len(z0), dtype=torch.bool, device=z0.device)
    kept = z0.detach().clone()  # z of the candidates that have stopped
    steps = torch.zeros(len(z0), dtype=torch.long, device=z0.device)
    initial = None
    for step in range(config.steps + 1):
        _, loss, constraint = evaluate(z)
        if initial is None:
            initial = loss.detach()
        newly = (constraint.detach() < config.tolerance) & ~done
        kept[newly] = z.detach()[newly]
        done = done | newly
        if done.all() or step == config.steps:
            break
        (gradient,) = torch.autograd.grad(loss.sum(), z)  # gradients for z only; the decoder stays frozen
        z.grad = gradient * (~done)[:, None]  # stopped candidates keep their z (Adam would still move it)
        optimizer.step()
        steps += (~done).long()
    with torch.no_grad():
        final_z = torch.where(done[:, None], kept, z.detach())
        positions, final, _ = evaluate(final_z)
    return Optimized(final_z, positions, rot, initial, final, steps)


def snap_pins(positions: torch.Tensor, problem: Problem) -> torch.Tensor:
    """Pinned items placed exactly on their requested centres (Tech Spec 5.1, step 6)."""
    if problem.pin_mask is None:
        return positions
    room, _, _ = unpack_conditions(problem.c, positions.shape[1])
    pinned = problem.pin_center / room[:, None, :]
    return torch.where(problem.pin_mask[..., None], pinned, positions)
