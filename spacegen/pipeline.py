"""Inference: from a room request to layouts (Tech Spec 5).

generate() runs the whole pipeline (T25): check the request, choose variants within the budget
and the footprint limit f_max, sample N layouts, keep those that pass H1 to H4, rank them with the
CNN evaluator and return a diverse top 3. Run it from the command line:
  python run.py generate --width 5 --depth 4 --door W --offset 0.5 --items sofa,tv_unit,coffee_table

CVAESampler is M1 (Tech Spec 4.3): z ~ N(0, I), decoded with the CVAE, positions clamped to the
room, each rotation the arg-max of its logits in canonical form. LatentOptSampler is M2: the
same draws, repaired by latent optimization (spacegen.latent_opt) before decoding. Both have
the baselines' sample(cond, n, rng) interface, so spacegen.evaluate runs them like B1, B2, G0.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from spacegen.baselines import Samples
from spacegen.catalog import RoomCatalog
from spacegen.config import DEFAULT_CONFIG, load_config
from spacegen.dataset import LayoutBatch, decode_targets, encode_conditions, stack_layouts
from spacegen.generator import Condition
from spacegen.latent_opt import LatentOptConfig, Problem, optimize
from spacegen.layout import Layout, make_layout
from spacegen.metrics import layout_distance
from spacegen.models.cvae import CVAE, CVAEConfig
from spacegen.models.evaluator import Evaluator, EvaluatorConfig
from spacegen.quality import quality_score
from spacegen.raster import RasterConfig, rasterize_layouts
from spacegen.rules import Rules, check_layout, door_geometry, footprint_ratio, reachability


def condition_batch(cond: Condition, n: int, catalog: RoomCatalog) -> LayoutBatch:
    """n copies of the room and its items, positions not yet known (all zero)."""
    blank = make_layout(catalog, cond.width, cond.depth, cond.door_wall, cond.door_offset,
                        {name: (0.0, 0.0, 0) for name in cond.items}, cond.items)
    return stack_layouts([blank] * n, catalog)


def load_cvae_run(run_dir: Path, device: str | torch.device = "cpu") -> CVAE:
    """The checkpoint of a train_cvae run, in eval() mode."""
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    model = CVAE(CVAEConfig(**record["cvae"]))
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval()


def load_evaluator_run(run_dir: Path, device: str | torch.device = "cpu") -> tuple[Evaluator, RasterConfig]:
    """The checkpoint of a train_evaluator run and the raster settings it was trained on."""
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = EvaluatorConfig(**{**record["evaluator"], "channels": tuple(record["evaluator"]["channels"])})
    raster = RasterConfig(**record["raster"])
    model = Evaluator(config, pixels=raster.pixels)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval(), raster


class CVAESampler:
    """M1: layouts decoded from prior samples of the CVAE (BatchNorm in inference mode)."""

    name = "M1"

    def __init__(self, model: CVAE, catalog: RoomCatalog, device: str | torch.device = "cpu"):
        self.model, self.catalog, self.device = model.to(device).eval(), catalog, device

    def sample(self, cond: Condition, n: int, rng: np.random.Generator) -> Samples:
        blank = condition_batch(cond, n, self.catalog)
        c = torch.as_tensor(encode_conditions(blank, self.catalog), device=self.device)
        positions, logits = self.model.generate(c, _torch_generator(rng, self.device))
        decoded = decode_targets(positions, logits, blank, self.catalog)
        return Samples([decoded.layout(i, self.catalog) for i in range(n)], n)


class LatentOptSampler:
    """M2: the M1 draws, each repaired by latent optimization before it is decoded."""

    name = "M2"

    def __init__(self, model: CVAE, catalog: RoomCatalog, rules: Rules, config: LatentOptConfig,
                 device: str | torch.device = "cpu"):
        self.model, self.catalog, self.rules, self.config = model.to(device).eval(), catalog, rules, config
        self.device = device

    def sample(self, cond: Condition, n: int, rng: np.random.Generator) -> Samples:
        blank = condition_batch(cond, n, self.catalog)
        problem = problem_for(cond, blank, self.catalog, self.rules, self.device)
        z0 = torch.randn(n, self.model.config.latent, generator=_torch_generator(rng, self.device), device=self.device)
        result = optimize(self.model, z0, problem, self.config)
        decoded = decode_targets(result.positions.clamp(0.0, 1.0), F.one_hot(result.rot, 4), blank, self.catalog)
        return Samples([decoded.layout(i, self.catalog) for i in range(n)], n)


def problem_for(cond: Condition, blank: LayoutBatch, catalog: RoomCatalog, rules: Rules,
                device: str | torch.device) -> Problem:
    """The latent-optimization problem of a room: its condition vectors and its door zone."""
    door = door_geometry(cond.width, cond.depth, cond.door_wall, cond.door_offset, rules.door)
    n = len(blank)
    c = torch.as_tensor(encode_conditions(blank, catalog), device=device)

    def repeat(values) -> torch.Tensor:
        return torch.as_tensor(np.asarray(values, dtype=np.float32), device=device).expand(n, -1)

    return Problem(c, repeat(door.zone_center), repeat(door.zone_size))


# --------------------------------------------------------------------------- the request (T25)

@dataclass(frozen=True)
class PipelineConfig:
    candidates: int = 64  # N
    latent_opt: bool = True  # M2; False gives M1
    top_k: int = 3
    tau_div: float = 0.3  # minimum layout distance between picks (m)
    relax: float = 0.25  # share by which tau_div shrinks when too few qualify ...
    relax_times: int = 2  # ... up to this many times


def load_pipeline_config(path: Path = DEFAULT_CONFIG, **overrides) -> PipelineConfig:
    return PipelineConfig(**{**load_config(path)["pipeline"], **overrides})


@dataclass(frozen=True)
class Request:
    """What a user asks for: the room, its door, the furniture and an optional budget (INR)."""
    width: float
    depth: float
    door_wall: str
    door_offset: float
    items: dict[str, str | None]  # slot name -> variant id, or None to let the pipeline choose
    budget: float | None = None


@dataclass
class Candidate:
    layout: Layout
    quality: float  # rule quality score S
    evaluator_score: float | None  # the CNN's predicted score (ranks the candidates)
    evaluator_valid: float | None  # the CNN's P(valid), shown next to the checker's verdict


@dataclass
class PipelineResult:
    request: Request
    condition: Condition | None  # the room with the variants chosen; None if turned away
    top: list[Candidate] = field(default_factory=list)
    message: str | None = None  # why nothing (or fewer than top_k) came back
    warnings: list[str] = field(default_factory=list)
    candidates: int = 0
    valid: int = 0
    tau_used: float | None = None
    seconds: dict[str, float] = field(default_factory=dict)


def check_request(request: Request, catalog: RoomCatalog, rules: Rules) -> tuple[list[str], list[str]]:
    """(errors that stop the request, warnings): unknown or missing items, a door that does not
    fit, a room outside the training ranges (out of distribution, Tech Spec 10)."""
    errors, warnings = [], []
    names = {slot.name for slot in catalog.slots}
    errors += [f"unknown item {name!r}" for name in request.items if name not in names]
    errors += [f"the {slot.name.replace('_', ' ')} is required" for slot in catalog.slots
               if slot.mandatory and slot.name not in request.items]
    for name, variant in request.items.items():
        if name in names and variant is not None and variant not in {v.id for v in catalog.slot(name).variants}:
            errors.append(f"unknown variant {variant!r} for the {name.replace('_', ' ')}")
    if min(request.width, request.depth) <= 0:
        errors.append("the room needs a positive width and depth")
    else:
        try:
            door_geometry(request.width, request.depth, request.door_wall, request.door_offset, rules.door)
        except ValueError as error:
            errors.append(str(error))
        room = rules.rooms[catalog.room_type]
        if not (room.width[0] <= request.width <= room.width[1] and room.depth[0] <= request.depth <= room.depth[1]):
            warnings.append(f"the room is outside the training ranges (W {room.width[0]:g} to {room.width[1]:g} m, "
                            f"D {room.depth[0]:g} to {room.depth[1]:g} m): expect lower validity")
    return errors, warnings


def choose_variants(request: Request, catalog: RoomCatalog, rules: Rules) -> tuple[Condition | None, str | None]:
    """The feasibility pre-check and variant selection (Tech Spec 3.5 and 5.1, step 2).

    Every combination of variants for the items left open is tried; those within the budget and
    the footprint limit f_max qualify, and the one with the largest total footprint wins (a
    comfort proxy), the cheaper one on a tie. None and a message with a suggestion otherwise.
    """
    options = [[catalog.slot(name).variant(v)] if v else list(catalog.slot(name).variants)
               for name, v in request.items.items()]
    names = list(request.items)
    combos = list(itertools.product(*options))
    price = [sum(v.price for v in combo) for combo in combos]
    area = [sum(v.area for v in combo) for combo in combos]
    ratio = [footprint_ratio(a, request.width, request.depth, rules.door) for a in area]
    fits = [i for i in range(len(combos)) if ratio[i] <= rules.f_max
            and (request.budget is None or price[i] <= request.budget)]
    if fits:
        best = max(fits, key=lambda i: (area[i], -price[i]))
        cond = Condition(request.width, request.depth, request.door_wall, request.door_offset,
                         {name: v.id for name, v in zip(names, combos[best])})
        return cond, None
    if request.budget is not None and min(price) > request.budget:
        return None, (f"the cheapest choice costs {min(price):,} INR, over the budget of {request.budget:,.0f} INR: "
                      "remove an optional item or raise the budget")
    optional = [n for n in names if not catalog.slot(n).mandatory]
    hint = f"remove an optional item (for example the {optional[0].replace('_', ' ')})" if optional else "use a bigger room"
    return None, (f"the furniture covers {min(ratio):.0%} of the free floor at best, above the {rules.f_max:.0%} "
                  f"that can usually be furnished: {hint} or choose a smaller variant")


def select_diverse(layouts: list[Layout], k: int, tau: float, relax: float, times: int) -> tuple[list[int], float]:
    """Greedy top-k over layouts already sorted best first: a layout joins if it is at least tau
    from every pick (Tech Spec 5.3); tau shrinks by `relax` up to `times` times if too few join."""
    picks: list[int] = []
    threshold = tau
    for attempt in range(times + 1):
        threshold = tau * (1 - relax) ** attempt
        picks = []
        for i, layout in enumerate(layouts):
            if all(layout_distance(layout, layouts[j]) >= threshold for j in picks):
                picks.append(i)
                if len(picks) == k:
                    return picks, threshold
    return picks, threshold


def generate(request: Request, catalog: RoomCatalog, rules: Rules, cvae: CVAE, rng: np.random.Generator,
             evaluator: Evaluator | None = None, raster: RasterConfig | None = None,
             config: PipelineConfig = PipelineConfig(), latent: LatentOptConfig = LatentOptConfig(),
             device: str | torch.device = "cpu") -> PipelineResult:
    """The inference pipeline of Tech Spec 5.1: check the request, choose variants, sample N
    layouts (M2 or M1), keep those that pass H1 to H4, rank them by the evaluator's score (or the
    rule score without an evaluator) and pick a diverse top k."""
    start = time.perf_counter()
    errors, warnings = check_request(request, catalog, rules)
    if errors:
        return PipelineResult(request, None, message="; ".join(errors), warnings=warnings)
    cond, message = choose_variants(request, catalog, rules)
    if cond is None:
        return PipelineResult(request, None, message=message, warnings=warnings)
    timings = {"check": time.perf_counter() - start}

    start = time.perf_counter()
    sampler = (LatentOptSampler(cvae, catalog, rules, latent, device) if config.latent_opt
               else CVAESampler(cvae, catalog, device))
    layouts = sampler.sample(cond, config.candidates, rng).layouts
    timings["sample"] = time.perf_counter() - start

    start = time.perf_counter()
    valid, quality = [], []
    for layout in layouts:
        reach = reachability(layout, catalog, rules)
        if check_layout(layout, catalog, rules, reach=reach).valid:  # the checker is the final authority
            valid.append(layout)
            quality.append(quality_score(layout, catalog, rules, reach=reach).total)
    timings["checks"] = time.perf_counter() - start

    start = time.perf_counter()
    scores = probabilities = [None] * len(valid)
    if evaluator is not None and valid:
        with torch.no_grad():
            logit, score = evaluator.eval()(rasterize_layouts(valid, catalog, rules, raster, device))
        scores, probabilities = score.cpu().tolist(), torch.sigmoid(logit).cpu().tolist()
    ranking = sorted(range(len(valid)), key=lambda i: -(scores[i] if scores[i] is not None else quality[i]))
    picks, tau = select_diverse([valid[i] for i in ranking], config.top_k, config.tau_div, config.relax,
                                config.relax_times)
    top = [Candidate(valid[ranking[i]], quality[ranking[i]], scores[ranking[i]], probabilities[ranking[i]])
           for i in picks]
    timings["rank"] = time.perf_counter() - start
    note = None
    if len(top) < config.top_k:
        note = (f"only {len(top)} distinct valid layouts out of {config.candidates} candidates"
                + ("" if valid else ": remove an optional item or try another door position"))
    return PipelineResult(request, cond, top, note, warnings, len(layouts), len(valid), tau if top else None,
                          {name: round(value, 3) for name, value in timings.items()})


def _torch_generator(rng: np.random.Generator, device) -> torch.Generator:
    """A torch generator seeded from the numpy one, so a seeded run draws the same z."""
    return torch.Generator(device=device).manual_seed(int(rng.integers(2 ** 62)))


def main(argv: list[str] | None = None) -> int:
    import argparse

    from spacegen.catalog import load_room_catalog
    from spacegen.latent_opt import load_latent_opt_config
    from spacegen.layout import layout_to_dict
    from spacegen.paths import REPORTS_DIR, RUNS_DIR
    from spacegen.rules import load_rules
    from spacegen.viz import SURFACE, plot_layout

    parser = argparse.ArgumentParser(description="Generate the top layouts for one room (the T25 pipeline).")
    parser.add_argument("--width", type=float, required=True, help="W in meters")
    parser.add_argument("--depth", type=float, required=True, help="D in meters")
    parser.add_argument("--door", required=True, choices=["N", "E", "S", "W"], help="wall with the door")
    parser.add_argument("--offset", type=float, default=0.5, help="door position along its wall, 0 to 1")
    parser.add_argument("--items", required=True,
                        help="comma-separated slots, each optionally name:variant, e.g. sofa:sofa_2seater,tv_unit")
    parser.add_argument("--budget", type=float, default=None, help="INR")
    parser.add_argument("--cvae", type=Path, default=RUNS_DIR / "cvae" / "default")
    parser.add_argument("--evaluator", type=Path, default=RUNS_DIR / "evaluator" / "e9a",
                        help="trained evaluator run; rank by the rule score if it does not exist")
    parser.add_argument("--no-latent-opt", action="store_true", help="M1 instead of M2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "demo")
    parser.add_argument("--device", default="cpu", help="cpu is faster than a GPU for one request")
    args = parser.parse_args(argv)

    items = dict((entry.split(":") + [None])[:2] for entry in args.items.split(","))
    request = Request(args.width, args.depth, args.door, args.offset, items, args.budget)
    catalog, rules = load_room_catalog("living_room"), load_rules()
    cvae = load_cvae_run(args.cvae, args.device)
    evaluator, raster = load_evaluator_run(args.evaluator, args.device) if args.evaluator.exists() else (None, None)
    result = generate(request, catalog, rules, cvae, np.random.default_rng(args.seed), evaluator, raster,
                      load_pipeline_config(latent_opt=not args.no_latent_opt), load_latent_opt_config(), args.device)
    for warning in result.warnings:
        print("warning:", warning)
    if result.condition is None:
        print("turned away:", result.message)
        return 1
    print(f"variants: {result.condition.items}")
    print(f"{result.valid} of {result.candidates} candidates valid; seconds {result.seconds} "
          f"(total {sum(result.seconds.values()):.2f})")
    args.out.mkdir(parents=True, exist_ok=True)
    for rank, candidate in enumerate(result.top, start=1):
        metrics = {"rank": rank, "valid": True, "quality": round(candidate.quality, 4),
                   "evaluator_score": None if candidate.evaluator_score is None else round(candidate.evaluator_score, 4),
                   "evaluator_valid": None if candidate.evaluator_valid is None else round(candidate.evaluator_valid, 4)}
        meta = {"method": "M1" if args.no_latent_opt else "M2", "seed": args.seed, "cvae": str(args.cvae)}
        data = layout_to_dict(candidate.layout, catalog, rules.door.width, metrics, meta)
        (args.out / f"top{rank}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        title = (f"Top {rank} of {len(result.top)}: rule quality {candidate.quality:.2f}"
                 + ("" if candidate.evaluator_score is None else f", evaluator score {candidate.evaluator_score:.2f}"))
        plot_layout(candidate.layout, catalog, rules, title=title).savefig(args.out / f"top{rank}.png", facecolor=SURFACE)
        print(f"top {rank}: quality {candidate.quality:.3f}, evaluator {metrics['evaluator_score']}, "
              f"P(valid) {metrics['evaluator_valid']}")
    if result.message:
        print("note:", result.message)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
