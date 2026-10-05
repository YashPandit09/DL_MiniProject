"""E1 and E10 tests (T28, T30): the harness's rooms and generators, M2 paired with M1, the cache,
the top 3 in three orders, and the figures."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.figures import FIGURES
from experiments.headline import E10_METHODS, METHODS, SETS, Headline, TopThree
from spacegen.dataset import load_layouts
from spacegen.evaluate import EvaluationConfig, evaluate_baselines
from spacegen.latent_opt import load_latent_opt_config
from spacegen.models.cvae import CVAE, CVAEConfig
from spacegen.models.evaluator import Evaluator, EvaluatorConfig
from spacegen.pipeline import PipelineConfig
from spacegen.raster import load_raster_config

SMALL = EvaluationConfig(rooms=4, samples=3)
TIMING = ["ms_per_valid", "seconds_per_room"]


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    raster = load_raster_config()
    evaluator = Evaluator(EvaluatorConfig(channels=(4, 4, 4, 4), hidden=8), pixels=raster.pixels).eval()
    return CVAE(CVAEConfig(hidden=16)).eval(), evaluator, raster


@pytest.fixture
def make(tiny_dataset, tmp_path, catalog, rules, models):
    cvae, evaluator, raster = models
    log = []

    def headline(steps: int = 3, inputs: dict | None = None, fresh: bool = False) -> Headline:
        return Headline(tiny_dataset, tmp_path / "cache", 0, SMALL, cvae, catalog, rules,
                        load_latent_opt_config(steps=steps), PipelineConfig(), evaluator, raster, "cpu", inputs, fresh,
                        log=log.append)

    headline.log = log
    return headline


def test_e1_baselines_reproduce_the_harness(make, tiny_dataset, catalog, rules):
    table = make().e1()
    assert table["method"].tolist() == list(METHODS)
    assert table["rvr"].between(0, 1).all() and (table["samples"] == SMALL.rooms * SMALL.samples).all()
    expected = evaluate_baselines(tiny_dataset, SMALL, 0, catalog, rules, log=lambda message: None)
    shared = [c for c in expected.columns if c not in TIMING]
    pd.testing.assert_frame_equal(table[shared].iloc[:3].reset_index(drop=True), expected[shared])


def test_m2_starts_from_m1s_draws(make):
    """With no optimization steps M2 is M1: both draw the same z for every room."""
    headline = make(steps=0)
    m1, m2 = (headline.per_room("in_distribution", method) for method in ("M1", "M2"))
    columns = ["valid", "overlap", "reachability", "quality", "diversity", "top_rule", "top_evaluator"]
    pd.testing.assert_frame_equal(m1[columns], m2[columns])


def test_e10_takes_the_in_distribution_rows_from_e1(make):
    headline = make()
    e1 = headline.e1().set_index("method")
    computed = len(make.log)
    e10 = headline.e10()
    assert [line.split(",")[0] for line in make.log[computed:]] == [s for s in SETS[1:] for _ in E10_METHODS]
    assert list(zip(e10["set"], e10["method"])) == [(s, m) for s in SETS for m in E10_METHODS]
    assert not any(c.startswith("diversity") for c in e10.columns)
    inside = e10[e10["set"] == "in_distribution"].set_index("method")
    for column in ("rvr", "quality", "quality_top3", "attempts_per_valid"):
        assert inside[column].equals(e1.loc[list(E10_METHODS), column])
    assert (e10.groupby("set", sort=False)["area_m2"].first().loc[["unseen_combination", "out_of_range"]] > 32).all()


def test_cached_rows_are_reused_until_the_inputs_change(make):
    first = make().e1()
    computed = len(make.log)
    assert computed == len(METHODS)
    again = make().e1()
    assert len(make.log) == computed  # nothing recomputed
    pd.testing.assert_frame_equal(again.drop(columns=TIMING), first.drop(columns=TIMING))
    pd.testing.assert_frame_equal(again[TIMING], first[TIMING])  # the timings come from the cache too
    make(inputs={"cvae": "another model"}).e1()
    assert len(make.log) == 2 * computed
    make(inputs={"cvae": "another model"}, fresh=True).e1()
    assert len(make.log) == 3 * computed


def test_top_three_in_three_orders(tiny_dataset, catalog, rules, models):
    _, evaluator, raster = models
    batch = load_layouts(tiny_dataset / "set_a.npz")
    a, b = batch.layout(0, catalog), batch.layout(1, catalog)
    quality = np.array([0.9, 0.8, 0.1])
    config = PipelineConfig(top_k=2)
    top = TopThree(config, np.random.default_rng(0), catalog, rules)
    row = top([a, a, b], quality)  # the copy of a is never diverse enough, so b joins
    assert row["top_rule"] == pytest.approx(0.5) and np.isnan(row["top_evaluator"])
    assert row["top_random"] in (pytest.approx(0.5), pytest.approx(0.45))
    best = TopThree(dataclasses.replace(config, top_k=1), np.random.default_rng(0), catalog, rules, evaluator, raster)
    row = best([a, a, b], quality)
    assert row["top_rule"] == 0.9 and row["top_evaluator"] in (0.9, 0.8, 0.1)
    assert all(np.isnan(v) for v in top([], np.array([])).values())


def test_figures_draw_from_the_tables(make, tmp_path):
    headline = make()
    for name, table in (("e1", headline.e1()), ("e10", headline.e10())):
        figures = FIGURES[name](table, tmp_path)
        assert figures and all(fig.axes for fig in figures.values())
