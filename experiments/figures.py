"""Figures of the screening experiments (T26): E2, E3a and E3b, from their tables and run logs.

Colours follow the reference palette (categorical slots in fixed order); every chart with two or
more series has a legend, values are labelled where few enough to read, and text uses ink colours.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from experiments.screening import HUBER
from spacegen.viz import INK, INK_MUTED, INK_SECONDARY, SURFACE, _quiet_axes

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
LOSSES = ("mse", "mae", *(f"huber {d:g}" for d in HUBER))


def _figure(width: float, height: float, title: str) -> Figure:
    fig = Figure(figsize=(width, height), dpi=150, facecolor=SURFACE, layout="constrained")
    fig.suptitle(title, x=0.01, ha="left", fontsize=9, color=INK, linespacing=1.4)
    return fig


def _legend(ax, **options) -> None:
    ax.legend(frameon=False, fontsize=6.5, labelcolor=INK_SECONDARY, **options)


def _legend_below(fig: Figure, ax) -> None:
    """A legend under the panels, where it cannot cover a bar or its label."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False, fontsize=7,
               labelcolor=INK_SECONDARY)


def _bars(ax, labels, groups: dict[str, list[float]], fmt: str) -> None:
    """Grouped bars: one colour per group, each bar labelled with its value."""
    width = 0.8 / len(groups)
    x = np.arange(len(labels))
    for i, (name, values) in enumerate(groups.items()):
        heights = [np.nan if v is None else v for v in values]
        bars = ax.bar(x + (i - (len(groups) - 1) / 2) * width, heights, width * 0.92, color=SERIES[i], label=name)
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


FIGURES = {
    "e2": lambda table, runs: {"losses": e2_losses(), "results": e2_results(table), "curves": e2_curves(table, runs)},
    "e3a": lambda table, runs: {"results": e3a_results(table)},
    "e3b": lambda table, runs: {"gradients": e3b_gradients(table, runs), "units": e3b_units(table)},
}
