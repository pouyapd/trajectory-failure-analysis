"""
Tests for the reusable analysis code. They use a small synthetic obsmat file
(tests/fixtures/obsmat_small.txt); the reproduction test at the end runs only
when the real ETH files are present (python scripts/download_data.py).
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfa.analysis import (bootstrap_importances, cross_validated, fit_tree, horizon_table, importance_ranks,
                          leaf_rules, risk_map)
from tfa.config import PAPER, PAPER_IMPORTANCES, PAPER_TABLE1, ExperimentConfig
from tfa.data import DataError, load_obsmat, sha256, verify
from tfa.metrics import ade, displacement, fde
from tfa.pipeline import damped_constant_velocity, evaluate_scene, label_failures, scene_summary, split_track

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "obsmat_small.txt"
DATA = ROOT / "data" / "eth"


# ── metrics ─────────────────────────────────────────────────────────────────

def test_ade_fde_known_values():
    gt = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    pred = np.array([[0.0, 3.0], [1.0, 4.0], [2.0, 0.0]])        # errors 3, 4, 0
    assert ade(pred, gt) == pytest.approx(7 / 3)
    assert fde(pred, gt) == pytest.approx(0.0)
    np.testing.assert_allclose(displacement(pred, gt), [3, 4, 0])


def test_ade_truncates_to_common_horizon():
    gt = np.zeros((5, 2))
    pred = np.ones((3, 2))
    assert ade(pred, gt) == pytest.approx(np.sqrt(2))
    assert fde(pred, gt) == pytest.approx(np.sqrt(2))


@pytest.mark.parametrize("pred,gt", [
    (np.zeros((0, 2)), np.zeros((3, 2))),                 # empty
    (np.zeros((3, 3)), np.zeros((3, 2))),                 # wrong width
    (np.array([[0, np.nan]]), np.zeros((1, 2))),          # NaN
])
def test_metrics_reject_invalid_input(pred, gt):
    with pytest.raises(ValueError):
        ade(pred, gt)


# ── predictor and split ──────────────────────────────────────────────────────

def test_damped_constant_velocity():
    out = damped_constant_velocity([0.0, 0.0], [1.0, 0.0], 3, dt=0.4, damping=0.5)
    # velocity is damped before each step: 0.5, 0.25, 0.125 m/s
    np.testing.assert_allclose(out[:, 0], [0.2, 0.3, 0.35])
    np.testing.assert_allclose(out[:, 1], 0.0)
    straight = damped_constant_velocity([1.0, 1.0], [0.0, 1.0], 8, damping=1.0)
    np.testing.assert_allclose(straight[-1], [1.0, 1.0 + 8 * 0.4])


def test_split_follows_notebook_rules():
    track = pd.DataFrame({"x": np.arange(12.0), "y": 0.0, "vx": 1.0, "vy": 0.0, "frame": np.arange(12)})
    s = split_track(track)
    assert len(s["obs_pos"]) == 6 and len(s["gt_pos"]) == 6      # half / half
    s = split_track(pd.concat([track] * 2, ignore_index=True))   # 24 points -> gt capped at 8
    assert len(s["gt_pos"]) == 8
    assert split_track(track.iloc[:3]) is None                    # fewer than min_track_len=4
    assert split_track(track.iloc[:4]) is not None


# ── data loading and validation ──────────────────────────────────────────────

def test_load_fixture_and_skip_short_tracks():
    df = load_obsmat(FIXTURE)
    assert list(df.columns) == ["frame", "ped_id", "x", "y", "vx", "vy"]
    res = evaluate_scene(df, "fixture")
    assert 99 not in res.ped_id.tolist()                          # the 3-point track is skipped
    assert len(res) == df.ped_id.nunique() - 1
    assert res.horizon.between(2, 8).all()


def test_loader_rejects_bad_files(tmp_path):
    with pytest.raises(DataError, match="not found"):
        load_obsmat(tmp_path / "missing.txt")
    bad = tmp_path / "bad.txt"
    bad.write_text("1 2 3\n4 5 6\n")
    with pytest.raises(DataError, match="columns"):
        load_obsmat(bad)
    gaps = tmp_path / "gaps.txt"
    gaps.write_text("\n".join(f"{fr} 1 {fr * 0.1} 0 0 0.25 0 0" for fr in (0, 6, 18)) + "\n")
    with pytest.raises(DataError, match="irregular frame spacing"):
        load_obsmat(gaps)
    dup = tmp_path / "dup.txt"
    dup.write_text("0 1 0 0 0 0 0 0\n0 1 1 0 0 0 0 0\n")
    with pytest.raises(DataError, match="duplicated"):
        load_obsmat(dup)


def test_sha256_and_verify(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"abc")
    assert sha256(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert verify(tmp_path) == {"ETH": False, "Hotel": False}


# ── labels, trees, risk maps ─────────────────────────────────────────────────

def test_failure_label_is_percentile_threshold():
    res = pd.DataFrame({"ade": np.arange(1.0, 101.0)})
    out, thr = label_failures(res, 75)
    assert thr == pytest.approx(np.percentile(np.arange(1.0, 101.0), 75))
    assert out.failure.sum() == 25
    with pytest.raises(ValueError):
        ExperimentConfig(failure_percentile=100)


def test_speed_substitution_switch():
    df = load_obsmat(FIXTURE)
    stopped = pd.DataFrame({"frame": 100 + 6 * np.arange(6), "ped_id": 50, "x": 1.0, "y": 1.0, "vx": 0.0, "vy": 0.0})
    df = pd.concat([df, stopped], ignore_index=True)
    with_sub = evaluate_scene(df, "f").set_index("ped_id")
    no_sub = evaluate_scene(df, "f", replace(PAPER, speed_substitute=None)).set_index("ped_id")
    assert with_sub.loc[50, "init_speed"] == 0.5 and bool(with_sub.loc[50, "speed_substituted"])
    assert no_sub.loc[50, "init_speed"] == 0.0
    assert with_sub.loc[50, "ade"] == no_sub.loc[50, "ade"]       # only the feature changes, not the prediction


def _synthetic_results(n=400, seed=0):
    rng = np.random.default_rng(seed)
    res = pd.DataFrame({"init_speed": rng.uniform(0, 2, n), "init_orientation": rng.uniform(-np.pi, np.pi, n),
                        "init_x": rng.uniform(0, 10, n), "init_y": rng.uniform(0, 10, n)})
    res["ade"] = 0.2 + 0.3 * (np.abs(res.init_orientation) > 2.5) + rng.normal(0, 0.05, n)
    res["horizon"] = rng.integers(2, 9, n)
    return label_failures(res, 75)[0]


def test_tree_cv_bootstrap_and_rules():
    res = _synthetic_results()
    tree = fit_tree(res)
    assert max(tree["importances"], key=tree["importances"].get) == "init_orientation"
    cv = cross_validated(res, n_repeats=2)
    assert cv["cv_accuracy_mean"] > cv["majority_baseline"]       # a real signal is detected on held-out folds
    boot = bootstrap_importances(res, n_boot=20)
    ranks = importance_ranks(boot).set_index("feature")
    assert ranks.loc["init_orientation", "top1_share"] > 0.8
    rules = leaf_rules(tree["model"], PAPER.features)
    assert rules and all("init_orientation" in r["rule"] for r in rules)


def test_risk_map_counts_and_rates():
    res = pd.DataFrame({"a": [0.1, 0.2, 0.9, 0.95], "b": [0.1, 0.1, 0.1, 0.1], "failure": [1, 0, 1, 1]})
    rm = risk_map(res, "a", "b", [0, 0.5, 1.0], [0, 1.0], min_count=1)
    np.testing.assert_allclose(rm["rate"], [[0.5, 1.0]])
    np.testing.assert_allclose(rm["counts"], [[2, 2]])
    sparse = risk_map(res, "a", "b", [0, 0.5, 1.0], [0, 1.0], min_count=3)
    assert np.isnan(sparse["rate"]).all()


def test_horizon_table():
    res = _synthetic_results()
    t = horizon_table(res)
    assert t.trajectories.sum() == len(res) and set(t.horizon) <= set(range(2, 9))


# ── full reproduction (only with the real dataset) ───────────────────────────

@pytest.mark.skipif(not all(verify(DATA).values()), reason="ETH data not downloaded (scripts/download_data.py)")
def test_reproduces_preprint_table1():
    from tfa.data import load_scene
    for scene, expected in PAPER_TABLE1.items():
        res, thr = label_failures(evaluate_scene(load_scene(DATA, scene), scene))
        s = scene_summary(res, thr)
        tree = fit_tree(res)
        assert s["trajectories"] == expected["trajectories"]
        assert round(s["mean_ade"], 3) == expected["mean_ade"]
        assert round(s["mean_fde"], 3) == expected["mean_fde"]
        assert round(s["threshold"], 2) == expected["threshold"]
        assert round(s["failure_rate"], 3) == expected["failure_rate"]
        assert round(tree["train_accuracy"], 3) == expected["tree_accuracy"]
        for feat, value in PAPER_IMPORTANCES[scene].items():
            assert round(tree["importances"][feat], 3) == value
