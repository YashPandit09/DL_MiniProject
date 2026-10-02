"""Screening runs for the CVAE ablations (T26, Tech Spec 6): one factor at a time, one seed each.

python run.py screen e2          (or e3a, e3b, all)

  E2   position loss: MSE, MAE, Huber (delta 0.01, 0.05, 0.1) on the clean data and on the
       outlier variant (4% of training layouts with one item moved to a random spot), plus MSE
       with the overlap penalty
  E3a  hidden activation with BatchNorm on (the deployed setting): decides the activation
  E3b  the same activations with BatchNorm off at depth 2 and 6, plus BatchNorm-on references
       for Sigmoid and ReLU at depth 6. These run a fixed 60 epochs, so that gradient norms
       exist at epochs 1, 10 and 50 for every run

Every run is the default configuration with a few overrides. Runs with the same overrides are
trained once (runs/cvae/screen/<overrides>/) and shared between experiments, and a finished run
is not trained again. Each experiment writes reports/tables/<experiment>.csv and its figures.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from spacegen.config import load_config
from spacegen.models.cvae import ACTIVATIONS
from spacegen.paths import DATA_DIR, REPORTS_DIR, RUNS_DIR
from spacegen.train_cvae import load_configs, train_cvae

HUBER = (0.01, 0.05, 0.1)
FIXED_60 = ["cvae_training.early_stopping=false", "cvae_training.max_epochs=60"]
OUTLIERS = "cvae_training.outliers=0.04"
OVERLAP = "cvae.lambda_overlap=1.0"  # (T26) the overlap term, on the scale of the reconstruction loss


def _losses(extra: list[str], suffix: str) -> list[tuple[str, list[str]]]:
    return ([(f"mse{suffix}", extra), (f"mae{suffix}", ["cvae.position_loss=mae", *extra])]
            + [(f"huber {d:g}{suffix}", ["cvae.position_loss=huber", f"cvae.huber_delta={d}", *extra]) for d in HUBER])


EXPERIMENTS: dict[str, list[tuple[str, list[str]]]] = {
    "e2": _losses([], "") + _losses([OUTLIERS], ", outliers") + [("mse, overlap term", [OVERLAP])],
    "e3a": [(name, [f"cvae.activation={name}"]) for name in ACTIVATIONS],
    "e3b": [(f"{name}, depth {depth}", [f"cvae.activation={name}", f"cvae.depth={depth}", "cvae.batch_norm=false",
                                         *FIXED_60])
            for depth in (2, 6) for name in ACTIVATIONS]
           + [(f"{name}, depth 6, BatchNorm on", [f"cvae.activation={name}", "cvae.depth=6", *FIXED_60])
              for name in ("sigmoid", "relu")],
}


def effective(overrides: list[str]) -> list[str]:
    """The overrides that actually change the default configuration."""
    defaults = dict(zip(("cvae", "cvae_training"), (dataclasses.asdict(c) for c in load_configs())))
    changed = []
    for override in overrides:
        key, _, value = override.partition("=")
        section, _, field = key.partition(".")
        if yaml.safe_load(value) != defaults[section][field]:
            changed.append(override)
    return changed


def run_name(overrides: list[str]) -> str:
    """A folder name for a set of overrides: the same settings always map to the same run."""
    parts = sorted(o.split(".", 1)[1].replace("=", "-") for o in effective(overrides))
    return "__".join(parts) or "default"


def run_experiment(name: str, data_dir: Path, seed: int, device: str, runs_dir: Path = RUNS_DIR / "cvae" / "screen",
                   settings: list[tuple[str, list[str]]] | None = None, base: list[str] = (),
                   log=print) -> pd.DataFrame:
    """Train what is missing, then collect one row per setting of the experiment."""
    rows = []
    for label, overrides in settings or EXPERIMENTS[name]:
        run_dir = runs_dir / run_name([*base, *overrides])
        if not (run_dir / "summary.json").exists():
            log(f"{name}: training {label} -> {run_dir.name}")
            model, training = load_configs(overrides=[*base, *overrides])
            train_cvae(data_dir, run_dir, seed, device, model, training, log=lambda message: None)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        epochs = pd.read_csv(run_dir / "log.csv")
        m1 = summary.get("m1_check") or {}
        final = epochs.iloc[-1]
        rows.append({"experiment": name, "setting": label, "run": run_dir.name,
                     "epochs": summary["epochs_run"], "selected_epoch": summary["selected_epoch"],
                     "val_loss": summary["selected_validation_loss"], "val_position": summary["selected_position_loss"],
                     "val_rotation": summary["selected_rotation_loss"], "val_kl": summary["selected_kl"],
                     "active_units": summary["active_units"],
                     "position_error_median": summary["position_error_median"],
                     "position_error_mean": summary["position_error_mean"],
                     "m1_rvr": m1.get("rvr"), "m1_quality": m1.get("quality"),
                     "dead_share": float(final.filter(like="val_dead_").mean()),
                     "seconds": float(epochs["seconds"].sum())})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CVAE screening experiments (E2, E3a, E3b).")
    parser.add_argument("experiments", nargs="+", choices=[*EXPERIMENTS, "all"])
    parser.add_argument("--data", type=Path, default=None, help="default: data/<version from configs/default.yaml>")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    raw = load_config()
    data_dir = args.data or DATA_DIR / raw["dataset"]["version"]
    names = list(EXPERIMENTS) if "all" in args.experiments else args.experiments
    tables_dir, figures_dir = REPORTS_DIR / "tables", REPORTS_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        table = run_experiment(name, data_dir, raw["seed"], args.device)
        table.round(5).to_csv(tables_dir / f"{name}.csv", index=False, lineterminator="\n")
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(table.drop(columns=["experiment", "run"]).round(4).to_string(index=False))
        from experiments.figures import FIGURES
        for figure_name, figure in FIGURES[name](table, RUNS_DIR / "cvae" / "screen").items():
            figures_dir.mkdir(parents=True, exist_ok=True)
            figure.savefig(figures_dir / f"{name}_{figure_name}.png", facecolor="#fcfcfb")
        print(f"wrote reports/tables/{name}.csv and its figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
