"""Figures of the experiments, from their tables and run logs: the screening (T26, T27, T27b,
T29: E2 to E8) and the first pass of the headline experiments (T28: E1, T30: E10).

Colours follow the reference palette (categorical slots in fixed order); every chart with two or
more series has a legend, values are labelled where few enough to read, and text uses ink colours.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from experiments.screening import HUBER, TARGET_LOSS
from spacegen.viz import CONTEXT, INK, INK_MUTED, INK_SECONDARY, SURFACE, _quiet_axes

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
LOSSES = ("mse", "mae", *(f"huber {d:g}" for d in HUBER))


def _figure(width: float, height: float, title: str) -> Figure:
    fig = Figure(figsize=(width, height), dpi=150, facecolor=SURFACE, layout="constrained")
    fig.suptitle(title, x=0.01, ha="left", fontsize=9, color=INK, linespacing=1.4)
    return fig


def _legend(ax, **options) -> None:
    if ax.get_legend_handles_labels()[0]:  # a panel can be empty when an experiment has no run for it
        ax.legend(frameon=False, fontsize=6.5, labelcolor=INK_SECONDARY, **options)


def _legend_below(fig: Figure, ax) -> None:
    """A legend under the panels, where it cannot cover a bar or its label."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False, fontsize=7,
               labelcolor=INK_SECONDARY)


def _bars(ax, labels, groups: dict[str, list[float]], fmt: str, colors=None) -> None:
    """Grouped bars: one colour per group (or `colors`, one entry per group: a colour, or a list
    with a colour per bar), each bar labelled with its value."""
    width = 0.8 / len(groups)
    x = np.arange(len(labels))
    for i, (name, values) in enumerate(groups.items()):
        heights = [np.nan if v is None else v for v in values]
        color = SERIES[i] if colors is None else colors[i]
        bars = ax.bar(x + (i - (len(groups) - 1) / 2) * width, heights, width * 0.92, color=color, label=name)
        ax.bar_label(bars, labels=["" if np.isnan(h) else format(h, fmt) for h in heights], padding=2, fontsize=5.5,
                     color=INK_SECONDARY)
    ax.set_xticks(x, labels)
    _quiet_axes(ax)


# --------------------------------------------------------------------------- E2

def e2_losses() -> Figure:
    """The position losses and their gradients against the error (Appendix D)."""
    e = np.linspace(-0.3, 0.3, 601)
    curves = {"MSE": (e ** 2, 2 * e), "MAE": (np.abs(e), np.sign(e))}
    for d in HUBER:
        inside = np.abs(e) <= d
        curves[f"Huber {d:g}"] = (np.where(inside, 0.5 * e ** 2, d * (np.abs(e) - 0.5 * d)),
                                  np.where(inside, e, d * np.sign(e)))
    fig = _figure(10, 3.8, "E2: position losses (left) and their gradients (right) against the error of one coordinate\n"
                           "MSE's gradient grows with the error, so outliers dominate; MAE and Huber cap it")
    loss_ax, grad_ax = fig.subplots(1, 2)
    for color, (name, (loss, gradient)) in zip(SERIES, curves.items()):
        loss_ax.plot(e, loss, color=color, linewidth=2, label=name)
        grad_ax.plot(e, gradient, color=color, linewidth=2, label=name)
    loss_ax.set(xlabel="error (normalized position)", ylabel="loss")
    grad_ax.set(xlabel="error (normalized position)", ylabel="d loss / d error", ylim=(-1.2, 1.2))
    for ax in (loss_ax, grad_ax):
        _quiet_axes(ax)
    _legend(loss_ax, loc="upper center")
    return fig


def e2_results(table: pd.DataFrame) -> Figure:
    """Position error and M1's raw validity per loss, clean data against the outlier variant."""
    rows = table.set_index("setting")
    labels = [*LOSSES, "mse, overlap term"]

    def column(name: str, suffix: str) -> list[float | None]:
        return [rows[name].get(f"{label}{suffix}") if f"{label}{suffix}" in rows.index else None for label in labels]

    fig = _figure(11, 4.2, "E2: position loss, clean training data vs 4% outliers (one seed per setting)\n"
                           "mean position error on validation (z = mu, meters) and M1 raw valid rate on 100 test rooms")
    error_ax, rvr_ax = fig.subplots(1, 2)
    _bars(error_ax, labels, {"clean": column("position_error_mean", ""),
                             "4% outliers": column("position_error_mean", ", outliers")}, ".3f")
    error_ax.set(ylabel="mean position error (m)")
    _bars(rvr_ax, labels, {"clean": [None if v is None else 100 * v for v in column("m1_rvr", "")],
                           "4% outliers": [None if v is None else 100 * v for v in column("m1_rvr", ", outliers")]}, ".1f")
    rvr_ax.set(ylabel="M1 raw valid (%)")
    for ax in (error_ax, rvr_ax):
        ax.tick_params(axis="x", labelrotation=20)
    _legend_below(fig, error_ax)
    return fig


def e2_curves(table: pd.DataFrame, runs: Path) -> Figure:
    """Validation position error per epoch for each loss (clean data)."""
    fig = _figure(7.5, 4.0, "E2: validation position error during training (clean data; the first 20 epochs anneal beta)")
    ax = fig.add_subplot()
    rows = table.set_index("setting")
    for color, label in zip(SERIES, LOSSES):
        if label in rows.index:
            epochs = pd.read_csv(runs / rows.loc[label, "run"] / "log.csv")
            ax.plot(epochs["epoch"], epochs["val_position_m"], color=color, linewidth=2, label=label)
    ax.set(xlabel="epoch", ylabel="position error, sampled z (m)")
    _quiet_axes(ax)
    _legend(ax, loc="upper right")
    return fig


# --------------------------------------------------------------------------- E3a

def e3a_results(table: pd.DataFrame) -> Figure:
    """Validation loss and M1 raw validity per activation, BatchNorm on."""
    fig = _figure(10, 3.8, "E3a: hidden activation with BatchNorm on (the deployed setting), one seed each\n"
                           "lower validation loss is better; M1 raw valid rate on 100 test rooms")
    loss_ax, rvr_ax = fig.subplots(1, 2)
    _bars(loss_ax, table["setting"], {"validation loss": table["val_loss"].tolist()}, ".3f")
    loss_ax.set(ylabel="validation loss at the selected epoch", ylim=(0, None))
    _bars(rvr_ax, table["setting"], {"M1 raw valid": (100 * table["m1_rvr"]).tolist()}, ".1f")
    rvr_ax.set(ylabel="M1 raw valid (%)")
    return fig


# --------------------------------------------------------------------------- E3b

def _gradient_layers(columns) -> list[str]:
    return [c for c in columns if c.startswith("grad_")]


def e3b_gradients(table: pd.DataFrame, runs: Path) -> Figure:
    """Per-layer gradient norms at epochs 1, 10 and 50, depth 6: BatchNorm off, and the BN-on references."""
    deep = table[table["setting"].str.contains("depth 6")]
    columns = 4
    rows_count = int(np.ceil(len(deep) / columns))
    fig = _figure(13, 2.9 * rows_count + 0.6,
                  "E3b: gradient norm of each linear layer (input side on the left), depth 6, mean over an epoch's steps\n"
                  "BatchNorm off unless stated; vanishing gradients show as norms falling towards the input layers")
    axes = np.atleast_1d(fig.subplots(rows_count, columns, sharey=True)).ravel()
    for ax, (_, row) in zip(axes, deep.iterrows()):
        epochs = pd.read_csv(runs / row["run"] / "log.csv").set_index("epoch")
        layers = _gradient_layers(epochs.columns)
        for color, epoch in zip(SERIES, (1, 10, 50)):
            if epoch in epochs.index:
                ax.plot(range(1, len(layers) + 1), epochs.loc[epoch, layers], color=color, linewidth=1.8,
                        marker="o", markersize=3, label=f"epoch {epoch}")
        ax.set_yscale("log")
        ax.set_title(row["setting"], fontsize=7.5, color=INK, loc="left")
        ax.set_xticks([1, layers.index("grad_decoder_1") + 1, len(layers)], ["encoder in", "decoder in", "heads"])
        _quiet_axes(ax)
    for ax in axes[len(deep):]:
        ax.set_visible(False)
    axes[0].set_ylabel("gradient norm (log)")
    _legend(axes[0], loc="lower left")
    return fig


def e3b_units(table: pd.DataFrame) -> Figure:
    """Dead units and validation loss per activation: depth 2 and 6 with BatchNorm off, and the BN-on references."""
    activations = list(dict.fromkeys(s.split(",")[0] for s in table["setting"]))
    rows = table.set_index("setting")
    groups = {"depth 2, BN off": ", depth 2", "depth 6, BN off": ", depth 6", "depth 6, BN on": ", depth 6, BatchNorm on"}

    def values(column: str, suffix: str, scale: float = 1.0) -> list[float | None]:
        return [scale * rows.loc[f"{a}{suffix}", column] if f"{a}{suffix}" in rows.index else None for a in activations]

    fig = _figure(11, 4.0, "E3b: share of dead units (silent for every validation sample, after 60 epochs) and the\n"
                           "validation loss after 60 epochs; BatchNorm on is a reference for Sigmoid and ReLU at depth 6")
    dead_ax, loss_ax = fig.subplots(1, 2)
    _bars(dead_ax, activations, {g: values("dead_share", s, 100) for g, s in groups.items()}, ".0f")
    dead_ax.set(ylabel="dead units (%)")
    _bars(loss_ax, activations, {g: values("val_loss", s) for g, s in groups.items()}, ".2f")
    loss_ax.set(ylabel="validation loss after 60 epochs")
    _legend_below(fig, dead_ax)
    return fig


# --------------------------------------------------------------------------- E4 to E8

def e4_curves(table: pd.DataFrame, runs: Path, target: float) -> Figure:
    """Validation loss per epoch, one panel per optimizer, a line per learning rate."""
    families = {"Adam": "adam", "SGD": "sgd ", "SGD, momentum 0.9": "sgd momentum", "RMSProp": "rmsprop"}
    fig = _figure(13, 3.6, f"E4: validation loss (at beta_target) per epoch, one seed each; dotted line: the target "
                           f"{target:g}\nthe first 20 epochs anneal beta, so every run starts high")
    axes = fig.subplots(1, len(families), sharey=True)
    for ax, (title, prefix) in zip(axes, families.items()):
        chosen = table[table["setting"].str.startswith(prefix)]
        if prefix == "sgd ":  # keep plain SGD apart from SGD with momentum
            chosen = chosen[~chosen["setting"].str.contains("momentum")]
        for color, (_, row) in zip(SERIES, chosen.iterrows()):
            epochs = pd.read_csv(runs / row["run"] / "log.csv")
            ax.plot(epochs["epoch"], epochs["val_loss"], color=color, linewidth=1.8,
                    label="lr " + row["setting"].split()[-1])
        ax.axhline(target, color=INK_MUTED, linewidth=0.8, linestyle=":")
        ax.set(title=title, xlabel="epoch", ylim=(0.5, 1.5))
        ax.title.set(fontsize=8, color=INK)
        _quiet_axes(ax)
        _legend(ax, loc="upper right")
    axes[0].set_ylabel("validation loss")
    return fig


def e4_epochs(table: pd.DataFrame, target: float) -> Figure:
    """Epochs until the validation loss first reaches the target, and the final selected loss."""
    fig = _figure(11, 3.8, f"E4: epochs until the validation loss reaches {target:g} (missing: never), and the\n"
                           "validation loss at the selected checkpoint")
    epochs_ax, loss_ax = fig.subplots(1, 2)
    reached = [None if pd.isna(v) else float(v) for v in table["epochs_to_target"]]
    _bars(epochs_ax, table["setting"], {"epochs to target": reached}, ".0f")
    epochs_ax.set(ylabel="epochs")
    _bars(loss_ax, table["setting"], {"validation loss": table["val_loss"].tolist()}, ".3f")
    loss_ax.set(ylabel="validation loss", ylim=(0, None))
    for ax in (epochs_ax, loss_ax):
        ax.tick_params(axis="x", labelrotation=30)
    return fig


def e5_tradeoff(table: pd.DataFrame) -> Figure:
    """Reconstruction against diversity of M1's samples, with raw validity and active units."""
    fig = _figure(13, 4.0, "E5: the KL weight beta and the latent size; one seed each\n"
                           "left: reconstruction error (z = mu) against the diversity of M1's valid samples")
    scatter_ax, rvr_ax, units_ax = fig.subplots(1, 3, width_ratios=[1.3, 1, 1])
    scatter_ax.scatter(table["position_error_mean"], table["m1_diversity"], s=30, color=SERIES[0], zorder=3)
    for _, row in table.iterrows():
        scatter_ax.annotate(row["setting"], (row["position_error_mean"], row["m1_diversity"]),
                            textcoords="offset points", xytext=(5, 4), fontsize=6.5, color=INK_SECONDARY)
    scatter_ax.set(xlabel="mean position error, z = mu (m)", ylabel="M1 diversity of valid samples (m)")
    _quiet_axes(scatter_ax)
    _bars(rvr_ax, table["setting"], {"M1 raw valid": (100 * table["m1_rvr"]).tolist()}, ".1f")
    rvr_ax.set(ylabel="M1 raw valid (%)")
    _bars(units_ax, table["setting"], {"active units": table["active_units"].astype(float).tolist()}, ".0f")
    units_ax.set(ylabel="active latent units")
    for ax in (rvr_ax, units_ax):
        ax.tick_params(axis="x", labelrotation=30)
    return fig


def e6_gap(table: pd.DataFrame) -> Figure:
    """Training against validation loss of the selected model, both scored in eval mode (the gap)."""
    fig = _figure(8, 3.9, "E6: regularization; loss of the selected model on the training and validation splits,\n"
                          "both in eval mode with the same noise and beta (the generalization gap), one seed each")
    ax = fig.add_subplot()
    _bars(ax, table["setting"], {"training": table["train_eval_loss"].tolist(),
                                 "validation": table["val_loss"].tolist()}, ".3f")
    ax.set(ylabel="loss", ylim=(0, None))
    _legend_below(fig, ax)
    return fig


def e7_walls(table: pd.DataFrame) -> Figure:
    """Position error near walls (u or v below 0.05 or above 0.95) against the rest."""
    fig = _figure(6.5, 3.9, "E7: does the Sigmoid head saturate near the walls?\n"
                            "mean position error (z = mu) for items within 5% of a wall and for the rest")
    ax = fig.add_subplot()
    _bars(ax, table["setting"], {"near walls": table["position_error_near_walls"].tolist(),
                                 "the rest": table["position_error_rest"].tolist()}, ".3f")
    ax.set(ylabel="mean position error (m)")
    _legend_below(fig, ax)
    return fig


def e8_steps(table: pd.DataFrame) -> Figure:
    """Raw validity, diversity ratio and time against the number of latent-optimization steps."""
    fig = _figure(12, 3.6, "E8: latent-optimization steps on the default CVAE (M2), 100 test rooms x 64 samples\n"
                           "0 steps is M1; the diversity ratio compares the valid samples with the generator (G0)")
    axes = fig.subplots(1, 3)
    panels = (("rvr", 100, "raw valid (%)"), ("diversity_ratio", 1, "diversity ratio to G0"),
              ("seconds_per_room", 1, "seconds per room (CPU)"))
    for ax, (column, scale, label) in zip(axes, panels):
        ax.plot(table["steps"], scale * table[column], color=SERIES[0], linewidth=2, marker="o", markersize=5)
        for x, y in zip(table["steps"], scale * table[column]):
            ax.annotate(f"{y:.2f}" if scale == 1 else f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=6, color=INK_SECONDARY)
        ax.set(xlabel="optimization steps", ylabel=label)
        _quiet_axes(ax)
    return fig


# --------------------------------------------------------------------------- E1 and E10 (first pass)

METHOD_COLORS = {"M1": SERIES[0], "M2": SERIES[1], "B1": SERIES[2], "B2": SERIES[3], "G0": CONTEXT}
METHOD_LABELS = {"M1": "M1 (CVAE samples)", "M2": "M2 (with latent optimization)", "G0": "G0 (generator, reference)"}
SET_LABELS = {"in_distribution": "in distribution", "interpolation": "interpolation",
              "unseen_combination": "unseen combination", "out_of_range": "out of range"}


def e1_results(table: pd.DataFrame) -> Figure:
    """Raw validity, quality and cost per valid layout of every method on the same rooms."""
    rooms, samples = int(table["rooms"].iloc[0]), int(table["samples"].iloc[0] / table["rooms"].iloc[0])
    fig = _figure(12, 3.8, f"E1, first pass on the default configuration: {rooms} test rooms x {samples} raw samples "
                           "per method\nG0 (grey) is the reference: valid by construction, its raw valid rate is its "
                           "acceptance per attempt")
    axes = fig.subplots(1, 3)
    colors = [[METHOD_COLORS[m] for m in table["method"]]]
    panels = (("rvr", 100, "raw valid (%)", ".1f"), ("quality", 1, "quality of the valid layouts", ".2f"),
              ("ms_per_valid", 1, "ms per valid layout (CPU)", ".1f"))
    for ax, (column, scale, label, fmt) in zip(axes, panels):
        _bars(ax, table["method"], {label: (scale * table[column]).tolist()}, fmt, colors)
        ax.set(ylabel=label)
    axes[1].set_ylim(0, 1.05)
    return fig


def e1_ranking(table: pd.DataFrame) -> Figure:
    """Quality of the top 3 the pipeline would show, with the valid layouts in three orders."""
    fig = _figure(8.5, 4.0, "E1: mean rule quality of each room's diverse top 3 (rooms with a valid layout), with the\n"
                            "valid layouts ranked by the CNN evaluator (the pipeline), by the exact rule score, or not at all")
    ax = fig.add_subplot()
    _bars(ax, table["method"], {"random order": table["quality_top3_random"].tolist(),
                                "evaluator ranking (the pipeline)": table["quality_top3"].tolist(),
                                "rule-score ranking (reference)": table["quality_top3_rule"].tolist()}, ".2f",
          colors=[CONTEXT, SERIES[6], SERIES[5]])
    ax.set(ylabel="quality of the top 3", ylim=(0, 1.05))
    _legend_below(fig, ax)
    return fig


def e10_generalization(table: pd.DataFrame) -> Figure:
    """Raw validity and quality of M1, M2 and G0 on the four test sets."""
    sets = list(dict.fromkeys(table["set"]))
    methods = list(dict.fromkeys(table["method"]))
    rows = table.set_index(["set", "method"])
    labels = [f"{SET_LABELS.get(s, s)}\n{rows.loc[(s, methods[0]), 'area_m2']:.0f} m², "
              f"{rows.loc[(s, methods[0]), 'items']:.1f} items" for s in sets]
    fig = _figure(12, 4.3, "E10, first pass on the default configuration: generalization beyond the training rooms\n"
                           "(mean room area and item count under each set); G0 shows how hard each set is by itself")
    rvr_ax, quality_ax = fig.subplots(1, 2)
    colors = [METHOD_COLORS[m] for m in methods]
    _bars(rvr_ax, labels, {METHOD_LABELS.get(m, m): [100 * rows.loc[(s, m), "rvr"] for s in sets] for m in methods},
          ".1f", colors)
    rvr_ax.set(ylabel="raw valid (%)")
    _bars(quality_ax, labels, {METHOD_LABELS.get(m, m): [rows.loc[(s, m), "quality"] for s in sets] for m in methods},
          ".2f", colors)
    quality_ax.set(ylabel="quality of the valid layouts", ylim=(0, 1.05))
    for ax in (rvr_ax, quality_ax):
        ax.tick_params(axis="x", labelsize=7)
    _legend_below(fig, rvr_ax)
    return fig


# --------------------------------------------------------------------------- Gate 2 (T33)

def gate2_candidates(runs: pd.DataFrame) -> Figure:
    """Per candidate: the mean over the seeds with one standard deviation, and every seed."""
    candidates = list(dict.fromkeys(runs["candidate"]))
    fig = _figure(13, 3.9, f"Gate 2: {runs['seed'].nunique()} seeds per candidate; M1 and M2 sample the same Set A "
                           "validation rooms with the same draws\nthe rule decides on M2's raw valid rate and top-3 "
                           "quality (see reports/gate2.md)")
    panels = (("m1_rvr", 100, "M1 raw valid (%)", ".1f"), ("m2_rvr", 100, "M2 raw valid (%)", ".1f"),
              ("m2_top3", 1, "M2 top-3 quality (evaluator ranking)", ".3f"), ("m2_diversity", 1, "M2 diversity (m)", ".2f"))
    axes = fig.subplots(1, len(panels))
    for ax, (column, scale, label, fmt) in zip(axes, panels):
        for i, name in enumerate(candidates):
            values = scale * runs.loc[runs["candidate"] == name, column].to_numpy(dtype=float)
            mean, std = values.mean(), (values.std(ddof=1) if len(values) > 1 else 0.0)
            ax.scatter(np.full(len(values), i + 0.2), values, s=14, color=CONTEXT, zorder=2,
                       label="each seed" if i == 0 else None)
            ax.errorbar(i, mean, yerr=std, fmt="o", color=SERIES[0], markersize=6, elinewidth=1.5, capsize=3,
                        zorder=3, label="mean, one standard deviation" if i == 0 else None)
            ax.annotate(format(mean, fmt), (i, mean + std), textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=6.5, color=INK_SECONDARY)
        ax.set_xticks(range(len(candidates)), candidates)
        ax.tick_params(axis="x", labelrotation=20)
        ax.margins(x=0.2, y=0.15)
        ax.set(ylabel=label)
        _quiet_axes(ax)
    _legend_below(fig, axes[0])
    return fig


FIGURES = {
    "e2":lambda table, runs: {"losses": e2_losses(), "results": e2_results(table), "curves": e2_curves(table, runs)},
    "e3a": lambda table, runs: {"results": e3a_results(table)},
    "e3b": lambda table, runs: {"gradients": e3b_gradients(table, runs), "units": e3b_units(table)},
    "e4": lambda table, runs: {"curves": e4_curves(table, runs, TARGET_LOSS), "epochs": e4_epochs(table, TARGET_LOSS)},
    "e5": lambda table, runs: {"tradeoff": e5_tradeoff(table)},
    "e6": lambda table, runs: {"gap": e6_gap(table)},
    "e7": lambda table, runs: {"walls": e7_walls(table)},
    "e8": lambda table, runs: {"steps": e8_steps(table)},
    "e1": lambda table, runs: {"results": e1_results(table), "ranking": e1_ranking(table)},
    "e10": lambda table, runs: {"generalization": e10_generalization(table)},
    "gate2": lambda table, runs: {"candidates": gate2_candidates(table)},
}
