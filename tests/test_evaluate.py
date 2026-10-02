"""Evaluation harness tests (T16): same rooms and samples for every method, Tech Spec 7 summaries."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from spacegen.baselines import GeneratorBaseline, UniformBaseline
from spacegen.build_dataset import build_dataset, load_dataset_config
from spacegen.evaluate import EvaluationConfig, evaluate, evaluate_baselines, evaluation_rooms, summarize
from spacegen.generator import Condition, load_generator_config, sample_condition
from spacegen.perturb import load_set_b_config
from spacegen.splits import load_split_config

GENERATOR = load_generator_config()


@pytest.fixture(scope="module")
def conditions(catalog, rules):
    rng = np.random.default_rng(0)
    return [sample_condition(rng, catalog, rules, GENERATOR) for _ in range(3)]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory, catalog, rules):
    directory = tmp_path_factory.mktemp("tiny")
    sizes = dataclasses.replace(load_dataset_config(), version="tiny", set_a=40, set_b=10, calibration_rooms=20)
    split = dataclasses.replace(load_split_config(), held_out_rooms=2, diversity_rooms=3, diversity_layouts=3)
    build_dataset(directory, 1, sizes, catalog, rules, GENERATOR, load_set_b_config(), split, log=lambda m: None)
    return directory


def test_every_method_gets_the_same_raw_samples_per_room(catalog, rules, conditions):
    rng = np.random.default_rng(1)
    b1 = evaluate(UniformBaseline(catalog), conditions, 8, rng, catalog, rules)
    g0 = evaluate(GeneratorBaseline(catalog, rules, dataclasses.replace(GENERATOR, attempts=1)), conditions, 8, rng,
                  catalog, rules, prefiltered=True)
    assert b1["attempts"].tolist() == g0["attempts"].tolist() == [8, 8, 8]
    assert (b1["returned"] == 8).all() and (g0["valid"] == g0["returned"]).all()  # G0 keeps only valid layouts
    assert (b1["valid"] <= b1["returned"]).all()


def test_summary_follows_the_metric_definitions():
    rooms = pd.DataFrame({"room": [0, 1], "attempts": [10, 10], "returned": [10, 10], "valid": [4, 0],
                          "overlap": [0.5, 1.5], "reachability": [9.0, 5.0], "quality": [3.2, 0.0],
                          "diversity": [0.6, None], "seconds": [0.2, 0.2]})
    row = summarize("X", rooms, reference=pd.Series({0: 0.8, 1: 0.9}))
    assert row["rvr"] == pytest.approx(0.2) and row["mean_overlap"] == pytest.approx(0.1)
    assert row["reachability"] == pytest.approx(0.7) and row["quality"] == pytest.approx(0.8)
    assert row["diversity_ratio"] == pytest.approx(0.75) and row["diversity_rooms"] == 1  # room 1 has no pairs
    assert row["attempts_per_valid"] == pytest.approx(5.0) and row["ms_per_valid"] == pytest.approx(100.0)
    assert row["rooms_3_valid"] == 0.5
    assert np.isnan(summarize("G0", rooms, pd.Series(dtype=float), prefiltered=True)["mean_overlap"])


def test_evaluation_rooms_start_with_the_diversity_rooms(tiny, catalog):
    rooms, reference = evaluation_rooms(tiny, 6, np.random.default_rng(0), catalog)
    info = pd.read_csv(tiny / "diversity_reference.csv")
    assert len(rooms) == 6 and len(reference) == info["source"].nunique() <= 3
    assert all(isinstance(cond, Condition) for cond in rooms)
    assert len({(c.width, c.depth, c.door_offset) for c in rooms}) == 6  # no room twice
    assert (reference.dropna() >= 0).all()


def test_baselines_table(tiny, catalog, rules):
    table = evaluate_baselines(tiny, EvaluationConfig(rooms=4, samples=5), 0, catalog, rules, log=lambda m: None)
    assert table["method"].tolist() == ["B1", "B2", "G0"]
    assert (table["samples"] == 20).all() and table["rvr"].between(0, 1).all()
    assert np.isnan(table.loc[2, "mean_overlap"]) and not np.isnan(table.loc[0, "mean_overlap"])
