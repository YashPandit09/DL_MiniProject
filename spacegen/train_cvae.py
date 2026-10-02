"""Training the CVAE on Set A (T21, Tech Spec 4.1).

python run.py train-cvae                                    the default configuration
python run.py train-cvae --name e5-beta1 --set cvae_training.beta_target=1.0 --set cvae.latent=4

- KL annealing: beta rises linearly from 0 at epoch 1 to beta_target at epoch anneal_epochs + 1.
- The validation loss is always computed with beta_target and the same draw of z every epoch,
  so epochs compare fairly. Checkpoint selection and the patience counter start only once
  annealing is over; otherwise the rising beta would stop training early (Tech Spec 4.1).
- log.csv, one row per epoch: the loss terms, the validation position error in meters (comparable
  across position losses), the per-layer gradient norms (mean over the epoch's steps), the share of dead units per hidden layer (units silent for every validation
  sample) and the active latent units (KL above 0.01 nats on validation).
- A run writes runs/cvae/<name>/: run.json (seed, configs, git commit, dataset hash),
  log.csv, model.pt (the selected checkpoint) and summary.json, which includes a first RVR of
  M1 on check_rooms test rooms (the E1 numbers come later, on the frozen configuration).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from spacegen.catalog import RoomCatalog, load_room_catalog
from spacegen.config import DEFAULT_CONFIG, load_config
from spacegen import geometry
from spacegen.dataset import (LayoutBatch, TrainingTensors, load_layouts, minibatches, split_targets, training_tensors,
                              unpack_conditions)
from spacegen.evaluate import evaluate
from spacegen.generator import Condition
from spacegen.models.cvae import CVAE, CVAEConfig, CVAEOutput, active_units, cvae_loss
from spacegen.paths import DATA_DIR, RUNS_DIR
from spacegen.pipeline import CVAESampler
from spacegen.provenance import write_run_record
from spacegen.rules import load_rules
from spacegen.seed import set_seed
from spacegen.splits import SPLITS

OPTIMIZERS = ("adam", "sgd", "sgd_momentum", "rmsprop")
_ACTIVATION_TYPES = (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.GELU, nn.Tanh, nn.Sigmoid)


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str = "adam"
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 0.0
    batch: int = 256
    max_epochs: int = 200
    beta_target: float = 0.1
    anneal_epochs: int = 20
    patience: int = 15
    early_stopping: bool = True
    check_rooms: int = 100
    outliers: float = 0.0  # E2: share of training layouts with one item moved to a random spot

    def __post_init__(self):
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")


def load_configs(path: Path = DEFAULT_CONFIG, overrides: list[str] = ()) -> tuple[CVAEConfig, TrainingConfig]:
    """The cvae and cvae_training sections, with overrides such as "cvae.activation=elu"."""
    raw = load_config(path)
    sections = {"cvae": dict(raw["cvae"]), "cvae_training": dict(raw["cvae_training"])}
    for override in overrides:
        key, _, value = override.partition("=")
        section, _, field = key.partition(".")
        if section not in sections or field not in sections[section] or not value:
            raise ValueError(f"override {override!r} must look like cvae.<field>=<value> or cvae_training.<field>=<value>")
        sections[section][field] = yaml.safe_load(value)
    return CVAEConfig(**sections["cvae"]), TrainingConfig(**sections["cvae_training"])


def beta_at(epoch: int, training: TrainingConfig) -> float:
    """KL weight in a (1-based) epoch: 0 at epoch 1, rising linearly to beta_target at anneal_epochs + 1."""
    if training.anneal_epochs <= 0:
        return training.beta_target
    return training.beta_target * min(1.0, (epoch - 1) / training.anneal_epochs)


@dataclass
class EarlyStopping:
    """Patience on the validation loss, counted only from epoch `start` (the first one after annealing)."""
    patience: int
    start: int
    best: float = float("inf")
    best_epoch: int | None = None

    def update(self, epoch: int, loss: float) -> tuple[bool, bool]:
        """(this epoch is the new best, stop now). Epochs before `start` are neither."""
        if epoch < self.start:
            return False, False
        if loss < self.best:
            self.best, self.best_epoch = loss, epoch
            return True, False
        return False, epoch - self.best_epoch >= self.patience


def make_optimizer(parameters, training: TrainingConfig) -> torch.optim.Optimizer:
    if training.optimizer == "adam":
        return torch.optim.Adam(parameters, lr=training.lr, weight_decay=training.weight_decay)
    if training.optimizer == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=training.lr, weight_decay=training.weight_decay)
    momentum = training.momentum if training.optimizer == "sgd_momentum" else 0.0
    return torch.optim.SGD(parameters, lr=training.lr, momentum=momentum, weight_decay=training.weight_decay)


def layer_names(model: CVAE) -> tuple[list[tuple[str, nn.Linear]], list[tuple[str, nn.Module]]]:
    """Readable names for the linear layers (gradient norms) and the activations (dead units)."""
    linear, activations = [], []
    for stack in ("encoder", "decoder"):
        modules = getattr(model, stack)
        linear += [(f"{stack}_{i}", m) for i, m in enumerate((m for m in modules if isinstance(m, nn.Linear)), 1)]
        activations += [(f"{stack}_{i}", m)
                        for i, m in enumerate((m for m in modules if isinstance(m, _ACTIVATION_TYPES)), 1)]
        if stack == "encoder":
            linear += [("to_mu", model.to_mu), ("to_logvar", model.to_logvar)]
    return linear + [("head_pos", model.head_pos), ("head_rot", model.head_rot)], activations


@torch.no_grad()
def validate(model: CVAE, data: TrainingTensors, eps: torch.Tensor, beta: float) -> dict[str, float]:
    """Loss terms at `beta` with the fixed noise `eps`, active units, and dead units per layer."""
    model.eval()
    outputs: dict[str, torch.Tensor] = {}
    _, activations = layer_names(model)
    hooks = [module.register_forward_hook(lambda m, i, out, name=name: outputs.__setitem__(name, out))
             for name, module in activations]
    try:
        mu, logvar = model.encode(data.x, data.c)
        z = mu + torch.exp(0.5 * logvar) * eps
        positions, logits = model.decode(z, data.c)
    finally:
        for hook in hooks:
            hook.remove()
    loss = cvae_loss(CVAEOutput(positions, logits, mu, logvar, z), data, beta, model.config)
    error = position_errors(positions, data, model.config.num_slots)
    dead = {f"dead_{name}": float((out.abs() <= 1e-8).all(dim=0).float().mean()) for name, out in outputs.items()}
    return {"loss": loss.total.item(), "recon": loss.recon.item(), "position": loss.position.item(),
            "rotation": loss.rotation.item(), "kl": loss.kl.item(), "position_m": float(error.mean()),
            "active_units": active_units(mu, logvar), **dead}


def position_errors(positions: torch.Tensor, data: TrainingTensors, num_slots: int) -> np.ndarray:
    """Distance in meters between predicted and true centres, one value per present item."""
    target, _ = split_targets(data.x)
    room, _, _ = unpack_conditions(data.c, num_slots)
    distance = ((positions - target) * room[:, None, :]).norm(dim=-1)
    return distance[data.mask > 0.5].detach().cpu().numpy()


@torch.no_grad()
def reconstruction_errors(model: CVAE, data: TrainingTensors) -> dict[str, float]:
    """Position error in meters with z = mu (E2), overall and split at 0.05 from the walls (E7)."""
    model.eval()
    out = model(data.x, data.c, sample=False)
    target, _ = split_targets(data.x)
    present = data.mask > 0.5
    near = ((target < 0.05) | (target > 0.95)).any(dim=-1)[present].cpu().numpy()
    error = position_errors(out.positions.clamp(0.0, 1.0), data, model.config.num_slots)
    return {"position_error_median": float(np.median(error)), "position_error_mean": float(error.mean()),
            "position_error_near_walls": float(error[near].mean()) if near.any() else None,
            "position_error_rest": float(error[~near].mean())}


def with_outliers(batch: LayoutBatch, share: float, rng: np.random.Generator,
                  catalog: RoomCatalog) -> tuple[LayoutBatch, np.ndarray]:
    """E2's outlier variant (Tech Spec 2.4): in a share of the layouts, one present item moves to a
    uniformly random spot where it still fits in the room. Returns the batch and the changed rows."""
    rows = np.sort(rng.choice(len(batch), size=round(share * len(batch)), replace=False))
    center = batch.center.copy()
    half = geometry.effective_size(batch.sizes(catalog), batch.rot) / 2
    for row in rows:
        k = rng.choice(np.flatnonzero(batch.mask[row]))
        center[row, k] = rng.uniform(half[row, k], batch.room[row] - half[row, k])
    return dataclasses.replace(batch, center=center), rows


def train_cvae(data_dir: Path, out_dir: Path, seed: int, device: str | torch.device,
               model_config: CVAEConfig | None = None, training: TrainingConfig | None = None,
               catalog: RoomCatalog | None = None, log=print) -> pd.DataFrame:
    """Train on the Set A training split; returns the per-epoch log."""
    set_seed(seed)
    if model_config is None or training is None:
        defaults = load_configs()
        model_config, training = model_config or defaults[0], training or defaults[1]
    catalog = catalog or load_room_catalog("living_room")
    write_run_record(out_dir, seed, data_dir, task="T21 CVAE training", device=str(device),
                     cvae=dataclasses.asdict(model_config), cvae_training=dataclasses.asdict(training))
    set_a = load_layouts(data_dir / "set_a.npz")
    with np.load(data_dir / "splits.npz") as splits:
        rows = {part: splits[f"set_a_{part}"] for part in SPLITS}
    train_layouts, outlier_rows = set_a.subset(rows["train"]), np.array([], dtype=int)
    if training.outliers > 0:  # E2: only the training split changes; validation and test stay clean
        train_layouts, outlier_rows = with_outliers(train_layouts, training.outliers, np.random.default_rng(seed),
                                                    catalog)
    train = training_tensors(train_layouts, catalog, device)
    val = training_tensors(set_a.subset(rows["validation"]), catalog, device)

    model = CVAE(model_config).to(device)
    optimizer = make_optimizer(model.parameters(), training)
    val_eps = torch.randn(len(val), model_config.latent, generator=torch.Generator().manual_seed(seed)).to(device)
    shuffle = torch.Generator().manual_seed(seed)
    stopper = EarlyStopping(training.patience, start=training.anneal_epochs + 1)
    linear, _ = layer_names(model)
    history, best_state = [], None
    for epoch in range(1, training.max_epochs + 1):
        beta = beta_at(epoch, training)
        model.train()
        terms = torch.zeros(5, device=device)
        grad_norms = torch.zeros(len(linear), device=device)
        steps = 0
        start = time.perf_counter()
        for index in minibatches(len(train), training.batch, shuffle, device, drop_last=True):
            batch = train[index]
            loss = cvae_loss(model(batch.x, batch.c), batch, beta, model.config)
            optimizer.zero_grad()
            loss.total.backward()
            grad_norms += torch.stack([module.weight.grad.norm() for _, module in linear])
            optimizer.step()
            terms += torch.stack([loss.total, loss.position, loss.rotation, loss.kl, loss.overlap]).detach()
            steps += 1
        train_terms = (terms / steps).tolist()
        seconds = time.perf_counter() - start
        validation = validate(model, val, val_eps, training.beta_target)
        improved, stop = stopper.update(epoch, validation["loss"])
        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "beta": beta, "seconds": round(seconds, 2),
                        **dict(zip(("train_loss", "train_position", "train_rotation", "train_kl", "train_overlap"),
                                   train_terms)),
                        **{f"val_{k}": v for k, v in validation.items()}, "best": improved,
                        **{f"grad_{name}": value / steps for (name, _), value in zip(linear, grad_norms.tolist())}})
        if epoch == 1 or epoch % 10 == 0 or improved or stop:
            log(f"epoch {epoch}: beta {beta:.3f}; train {train_terms[0]:.4f}; validation {validation['loss']:.4f} "
                f"(position {validation['position']:.4f}, rotation {validation['rotation']:.3f}, KL {validation['kl']:.2f}); "
                f"active units {validation['active_units']}{'; best' if improved else ''}")
        if stop and training.early_stopping:
            break

    selected = stopper.best_epoch if training.early_stopping and stopper.best_epoch else len(history)
    if selected == len(history):  # fixed epochs (E6), or no epoch after annealing: keep the final model
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    table = pd.DataFrame(history)
    table.to_csv(out_dir / "log.csv", index=False, lineterminator="\n")
    torch.save(best_state, out_dir / "model.pt")
    model.load_state_dict(best_state)
    chosen = table.loc[table["epoch"] == selected].iloc[0]
    summary = {"epochs_run": len(history), "stopped_early": len(history) < training.max_epochs,
               "selected_epoch": int(selected), "selected_validation_loss": float(chosen["val_loss"]),
               "selected_position_loss": float(chosen["val_position"]),
               "selected_rotation_loss": float(chosen["val_rotation"]), "selected_kl": float(chosen["val_kl"]),
               "active_units": int(chosen["val_active_units"]),
               "parameters": sum(p.numel() for p in model.parameters()), "outlier_layouts": len(outlier_rows),
               **reconstruction_errors(model, val)}
    if training.check_rooms > 0:
        check = _m1_check(model, set_a, rows["test"], training.check_rooms, seed, catalog, device)
        summary["m1_check"] = check
        quality = "none valid" if check["quality"] is None else f"{check['quality']:.3f}"
        log(f"M1 on {check['rooms']} test rooms x 64 samples: RVR {check['rvr']:.1%}, quality of valid layouts {quality}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    return table


def _m1_check(model: CVAE, set_a, test_rows: np.ndarray, n_rooms: int, seed: int, catalog: RoomCatalog,
              device) -> dict:
    """A first raw valid rate of M1: 64 prior samples for each of n_rooms Set A test rooms."""
    rng = np.random.default_rng(seed)
    rooms = [Condition.of(set_a.layout(int(row), catalog), catalog)
             for row in rng.choice(test_rows, size=min(n_rooms, len(test_rows)), replace=False)]
    table = evaluate(CVAESampler(model, catalog, device), rooms, 64, rng, catalog, load_rules())
    valid = int(table["valid"].sum())
    return {"rooms": len(rooms), "samples": int(table["attempts"].sum()), "rvr": valid / int(table["attempts"].sum()),
            "quality": float(table["quality"].sum() / valid) if valid else None,
            "mean_overlap": float(table["overlap"].sum() / table["returned"].sum())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the CVAE on Set A.")
    parser.add_argument("--name", default="default", help="run folder under runs/cvae/")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="override a config value, e.g. cvae.activation=elu (repeatable)")
    parser.add_argument("--seed", type=int, default=None, help="default: the seed in configs/default.yaml")
    parser.add_argument("--data", type=Path, default=None, help="default: data/<version from configs/default.yaml>")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    raw = load_config()
    model_config, training = load_configs(overrides=args.overrides)
    data_dir = args.data or DATA_DIR / raw["dataset"]["version"]
    out_dir = RUNS_DIR / "cvae" / args.name
    seed = raw["seed"] if args.seed is None else args.seed
    train_cvae(data_dir, out_dir, seed, args.device, model_config, training)
    print(json.dumps(json.loads((out_dir / "summary.json").read_text(encoding="utf-8")), indent=2))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
