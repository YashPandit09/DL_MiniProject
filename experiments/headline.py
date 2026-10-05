"""Headline experiments, first pass (T28: E1, T30: E10; Tech Spec 6 and 7).

python run.py e1     B1, B2, G0, M1 and M2 on the E1 rooms: reports/tables/e1.csv and figures
python run.py e10    M1, M2 and G0 on the four test sets: reports/tables/e10.csv and its figure

--cvae RUN chooses the CVAE (default runs/cvae/default) and --evaluator RUN the ranker (default
runs/evaluator/e9a). Before Gate 2 both experiments run on the default configuration to debug the
pipeline; T33b reruns them on the frozen configuration for the headline numbers.

E1   The evaluation protocol of spacegen.evaluate: the same 500 rooms and 64 raw samples per room
     for every method. B1, B2 and G0 draw from one generator in that order, as `python run.py
     baselines` does, so their rows reproduce reports/tables/baselines.csv (timings aside). M1 and
     M2 start from the same generator seed, so M2 repairs exactly M1's draws. Every sampler runs
     on the CPU, so the costs compare. Added to the spec's metrics (T28): the mean rule quality of
     the diverse top 3 of each room (Tech Spec 5.3), with the valid layouts ranked by the CNN
     evaluator (what the pipeline shows), by the exact rule score (the reference) and in random
     order (no ranking). The ranking is not timed.
E10  M1, M2 and G0 on the E1 rooms (in distribution) and on the first 500 rooms of each held-out
     set: interpolation (22 to 26 m^2), unseen combination (above 32 m^2) and out of range (W 7 to
     8 m, D 6 to 7 m). G0, added to the spec's M1 and M2, shows how hard each set is by itself:
     room size alone moves the quality score, through circulation for example.

Per-room rows are kept in runs/headline/<cvae run>/<set>/<method>.csv and reused while the inputs
(inputs.json: dataset, models, seed, rooms, samples, latent-optimization and top-3 settings) stay
the same; --fresh recomputes them, which a change to the code needs. So E10 takes the
in-distribution rows from E1, and an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spacegen.catalog import RoomCatalog, load_room_catalog
from spacegen.config import load_config
from spacegen.dataset import load_layouts
from spacegen.evaluate import (EvaluationConfig, baseline_samplers, evaluate, evaluation_rooms, generator_sampler,
                               load_evaluation_config, summarize)
from spacegen.generator import Condition
from spacegen.latent_opt import LatentOptConfig, load_latent_opt_config
from spacegen.layout import Layout
from spacegen.models.cvae import CVAE
from spacegen.models.evaluator import Evaluator
from spacegen.paths import DATA_DIR, REPORTS_DIR, RUNS_DIR
from spacegen.pipeline import (CVAESampler, LatentOptSampler, PipelineConfig, load_cvae_run, load_evaluator_run,
                               load_pipeline_config, select_diverse)
from spacegen.provenance import dataset_hash, write_run_record
from spacegen.raster import RasterConfig, rasterize_layouts
from spacegen.rules import Rules, load_rules
from spacegen.seed import set_seed

SETS = ("in_distribution", "interpolation", "unseen_combination", "out_of_range")
METHODS = ("B1", "B2", "G0", "M1", "M2")
E10_METHODS = ("M1", "M2", "G0")
ORDERS = {"evaluator": "quality_top3", "rule": "quality_top3_rule", "random": "quality_top3_random"}


class TopThree:
    """The diverse top k of a room's valid layouts (Tech Spec 5.3) in three orders: the CNN
    evaluator's score (what the pipeline shows), the exact rule score (the reference) and a random
    order (no ranking). Gives the mean rule quality of each top k; the evaluator's is NaN without
    an evaluator. The random order draws from its own generator, not the sampler's."""

    def __init__(self, config: PipelineConfig, rng: np.random.Generator, catalog: RoomCatalog, rules: Rules,
                 evaluator: Evaluator | None = None, raster: RasterConfig | None = None,
                 device: str | torch.device = "cpu"):
        self.config, self.rng, self.catalog, self.rules = config, rng, catalog, rules
        self.evaluator, self.raster, self.device = evaluator, raster, device

    def __call__(self, layouts: list[Layout], quality: np.ndarray) -> dict[str, float]:
        row = {f"top_{order}": np.nan for order in ORDERS}
        if not layouts:
            return row
        orders = {"rule": np.argsort(-quality, kind="stable"), "random": self.rng.permutation(len(layouts))}
        if self.evaluator is not None:
            with torch.no_grad():
                _, score = self.evaluator(rasterize_layouts(layouts, self.catalog, self.rules, self.raster,
                                                            self.device))
            orders["evaluator"] = np.argsort(-score.cpu().numpy(), kind="stable")
        for order, ranked in orders.items():
            picks, _ = select_diverse([layouts[i] for i in ranked], self.config.top_k, self.config.tau_div,
                                      self.config.relax, self.config.relax_times)
            row[f"top_{order}"] = float(quality[ranked[picks]].mean())
        return row


class Headline:
    """E1 and E10 for one CVAE, with the per-room rows of every method and test set cached in `cache`."""

    def __init__(self, data_dir: Path, cache: Path, seed: int, config: EvaluationConfig, cvae: CVAE,
                 catalog: RoomCatalog, rules: Rules, latent: LatentOptConfig, pipeline: PipelineConfig,
                 evaluator: Evaluator | None = None, raster: RasterConfig | None = None,
                 rank_device: str | torch.device = "cpu", inputs: dict | None = None, fresh: bool = False,
                 log=print):
        self.data_dir, self.cache, self.seed, self.config = data_dir, cache, seed, config
        self.cvae = cvae.cpu().eval()  # every sampler runs on the CPU, so the costs compare
        self.catalog, self.rules, self.latent, self.pipeline = catalog, rules, latent, pipeline
        self.evaluator, self.raster, self.rank_device, self.log = evaluator, raster, rank_device, log
        inputs = json.loads(json.dumps({**(inputs or {}), "seed": seed, "rooms": config.rooms,
                                        "samples": config.samples, "latent_opt": dataclasses.asdict(latent),
                                        "pipeline": dataclasses.asdict(pipeline)}))
        stored = cache / "inputs.json"
        if fresh or not stored.exists() or json.loads(stored.read_text(encoding="utf-8")) != inputs:
            for old in cache.glob("*/*.csv"):  # made with other inputs, or a fresh start was asked for
                old.unlink()
            cache.mkdir(parents=True, exist_ok=True)
            stored.write_text(json.dumps(inputs, indent=2), encoding="utf-8", newline="\n")

    def rooms(self, set_name: str) -> tuple[list[Condition], pd.Series]:
        """The rooms of a test set, and the G0 reference diversity per room (the E1 rooms only)."""
        if set_name == "in_distribution":
            return evaluation_rooms(self.data_dir, self.config.rooms, np.random.default_rng(self.seed), self.catalog)
        batch = load_layouts(self.data_dir / f"held_out_{set_name}.npz")
        count = min(self.config.rooms, len(batch))
        return [Condition.of(batch.layout(i, self.catalog), self.catalog) for i in range(count)], pd.Series(dtype=float)

    def per_room(self, set_name: str, method: str) -> pd.DataFrame:
        """The per-room rows of a method on a test set (spacegen.evaluate.evaluate), computed once."""
        path = self.cache / set_name / f"{method}.csv"
        if not path.exists():
            if set_name == "in_distribution" and method in ("B1", "B2", "G0"):
                self._baselines()
            else:
                self._run(set_name, method)
        return pd.read_csv(path)

    def summary(self, set_name: str, method: str, reference: pd.Series) -> dict:
        """The summary row of spacegen.evaluate, plus the quality of the top 3 in each order (mean over
        the rooms with a valid layout)."""
        rooms = self.per_room(set_name, method)
        row = summarize(method, rooms, reference, prefiltered=method == "G0")
        return {**row, **{column: rooms[f"top_{order}"].mean() for order, column in ORDERS.items()}}

    def e1(self) -> pd.DataFrame:
        """One row per method on the E1 rooms."""
        _, reference = self.rooms("in_distribution")
        table = pd.DataFrame([self.summary("in_distribution", method, reference) for method in METHODS])
        first = ["method", "rooms", "samples", "rvr", "mean_overlap", "reachability", "quality", *ORDERS.values()]
        return table[first + [c for c in table.columns if c not in first]]

    def e10(self) -> pd.DataFrame:
        """One row per test set and method (M1, M2, G0), with the set's mean room area and item count."""
        rows = []
        for set_name in SETS:
            conditions, reference = self.rooms(set_name)
            context = {"set": set_name, "area_m2": float(np.mean([c.area for c in conditions])),
                       "items": float(np.mean([len(c.items) for c in conditions]))}
            for method in E10_METHODS:
                row = self.summary(set_name, method, reference)
                rows.append({**context, **{k: v for k, v in row.items() if not k.startswith("diversity")}})
        return pd.DataFrame(rows)

    def _baselines(self) -> None:
        """B1, B2 and G0 on the E1 rooms, drawing from one generator in the order of `python run.py baselines`."""
        rng = np.random.default_rng(self.seed)
        conditions, _ = evaluation_rooms(self.data_dir, self.config.rooms, rng, self.catalog)
        for name, sampler, prefiltered in baseline_samplers(self.data_dir, self.catalog, self.rules):
            self._evaluate("in_distribution", name, sampler, conditions, rng, prefiltered)

    def _run(self, set_name: str, method: str) -> None:
        conditions, _ = self.rooms(set_name)
        if method == "G0":
            sampler, prefiltered, salt = generator_sampler(self.catalog, self.rules), True, 1
        elif method == "M1":
            sampler, prefiltered, salt = CVAESampler(self.cvae, self.catalog), False, 0
        elif method == "M2":  # the same seed as M1: M2 repairs M1's draws
            sampler, prefiltered, salt = LatentOptSampler(self.cvae, self.catalog, self.rules, self.latent), False, 0
        else:
            raise ValueError(f"{method} is evaluated only on the in-distribution rooms")
        self._evaluate(set_name, method, sampler, conditions, self._rng(set_name, salt), prefiltered)

    def _evaluate(self, set_name: str, method: str, sampler, conditions: list[Condition], rng: np.random.Generator,
                  prefiltered: bool) -> None:
        start = time.perf_counter()
        top = TopThree(self.pipeline, self._rng(set_name, 2, METHODS.index(method)), self.catalog, self.rules,
                       self.evaluator, self.raster, self.rank_device)
        rows = evaluate(sampler, conditions, self.config.samples, rng, self.catalog, self.rules, prefiltered, top)
        path = self.cache / set_name / f"{method}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(path, index=False, lineterminator="\n")
        self.log(f"{set_name}, {method}: {len(rows)} rooms in {time.perf_counter() - start:.0f} s, "
                 f"raw valid {rows['valid'].sum() / rows['attempts'].sum():.1%}")

    def _rng(self, set_name: str, *salt: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, SETS.index(set_name), *salt])


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E1 and E10, first pass on the default configuration (T28, T30).")
    parser.add_argument("experiments", nargs="+", choices=["e1", "e10"])
    parser.add_argument("--cvae", type=Path, default=RUNS_DIR / "cvae" / "default")
    parser.add_argument("--evaluator", type=Path, default=RUNS_DIR / "evaluator" / "e9a",
                        help="ranks the top 3; without it only the rule-score and random orders are reported")
    parser.add_argument("--data", type=Path, default=None, help="default: data/<version from configs/default.yaml>")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="for the evaluator's ranking only; the samplers always run on the CPU")
    parser.add_argument("--fresh", action="store_true", help="recompute the cached per-room rows")
    args = parser.parse_args(argv)
    raw = load_config()
    seed = raw["seed"]
    set_seed(seed)
    data_dir = args.data or DATA_DIR / raw["dataset"]["version"]
    catalog, rules = load_room_catalog("living_room"), load_rules()
    has_evaluator = (args.evaluator / "model.pt").exists()
    evaluator, raster = load_evaluator_run(args.evaluator, args.device) if has_evaluator else (None, None)
    latent, pipeline, config = load_latent_opt_config(), load_pipeline_config(), load_evaluation_config()
    cache = RUNS_DIR / "headline" / args.cvae.name
    inputs = {"dataset_hash": dataset_hash(data_dir), "cvae": _file_hash(args.cvae / "model.pt"),
              "evaluator": _file_hash(args.evaluator / "model.pt") if has_evaluator else None}
    headline = Headline(data_dir, cache, seed, config, load_cvae_run(args.cvae), catalog, rules, latent, pipeline,
                        evaluator, raster, args.device, inputs, args.fresh)
    write_run_record(cache, seed, data_dir, task="T28 and T30: first pass of E1 and E10", cvae_run=str(args.cvae),
                     evaluator_run=str(args.evaluator) if has_evaluator else None, ranking_device=str(args.device),
                     evaluation=dataclasses.asdict(config), latent_opt=dataclasses.asdict(latent),
                     pipeline=dataclasses.asdict(pipeline))
    from experiments.figures import FIGURES

    tables_dir, figures_dir = REPORTS_DIR / "tables", REPORTS_DIR / "figures"
    for name in args.experiments:
        table = headline.e1() if name == "e1" else headline.e10()
        tables_dir.mkdir(parents=True, exist_ok=True)
        table.round(5).to_csv(tables_dir / f"{name}.csv", index=False, lineterminator="\n")
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(table.round(4).to_string(index=False))
        figures_dir.mkdir(parents=True, exist_ok=True)
        for figure_name, figure in FIGURES[name](table, cache).items():
            figure.savefig(figures_dir / f"{name}_{figure_name}.png", facecolor="#fcfcfb")
        print(f"wrote reports/tables/{name}.csv and its figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
