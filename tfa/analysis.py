"""
analysis.py — interpretable failure analysis on the per-pedestrian results.

* `fit_tree`: the preprint's decision tree (depth 4, >= 10 samples per leaf),
  training accuracy and Gini importances — the numbers reported in the preprint.
* `cross_validated`: held-out accuracy and balanced accuracy with repeated
  stratified k-fold, next to the majority-class baseline (additional analysis).
* `bootstrap_importances`: how stable the importances are under resampling.
* `risk_map`: the binned empirical failure rate R(U_bin) of Eq. 5.
* `leaf_rules`: readable IF-THEN rules for the leaves that predict failure.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from .config import PAPER, ExperimentConfig


def _tree(cfg: ExperimentConfig, seed: Optional[int] = None) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=cfg.max_depth, min_samples_leaf=cfg.min_samples_leaf,
                                  random_state=cfg.random_state if seed is None else seed)


def fit_tree(results: pd.DataFrame, cfg: ExperimentConfig = PAPER) -> Dict[str, Any]:
    """Fit on all trajectories of a scene (preprint setting; accuracy is on the training data)."""
    X, y = results[cfg.features].to_numpy(), results["failure"].to_numpy()
    clf = _tree(cfg).fit(X, y)
    return {
        "model": clf,
        "train_accuracy": float(clf.score(X, y)),
        "importances": {f: float(v) for f, v in zip(cfg.features, clf.feature_importances_)},
        "majority_baseline": float(max(y.mean(), 1 - y.mean())),
        "n_leaves": int(clf.get_n_leaves()),
    }


def cross_validated(results: pd.DataFrame, cfg: ExperimentConfig = PAPER, n_splits: int = 5,
                    n_repeats: int = 20, seed: int = 0) -> Dict[str, float]:
    """Held-out accuracy / balanced accuracy of the same tree settings (repeated stratified k-fold)."""
    X, y = results[cfg.features].to_numpy(), results["failure"].to_numpy()
    acc, bal = [], []
    for train, test in RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                               random_state=seed).split(X, y):
        clf = _tree(cfg).fit(X[train], y[train])
        pred = clf.predict(X[test])
        acc.append(float((pred == y[test]).mean()))
        bal.append(float(balanced_accuracy_score(y[test], pred)))
    return {
        "cv_accuracy_mean": float(np.mean(acc)), "cv_accuracy_std": float(np.std(acc)),
        "cv_balanced_accuracy_mean": float(np.mean(bal)), "cv_balanced_accuracy_std": float(np.std(bal)),
        "majority_baseline": float(max(y.mean(), 1 - y.mean())),
        "n_splits": n_splits, "n_repeats": n_repeats,
    }


def bootstrap_importances(results: pd.DataFrame, cfg: ExperimentConfig = PAPER, n_boot: int = 500,
                          seed: int = 0) -> pd.DataFrame:
    """Gini importances of trees fitted on bootstrap resamples (one row per resample)."""
    rng = np.random.default_rng(seed)
    X, y = results[cfg.features].to_numpy(), results["failure"].to_numpy()
    rows = []
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        clf = _tree(cfg).fit(X[idx], y[idx])
        rows.append(dict(zip(cfg.features, clf.feature_importances_)))
    return pd.DataFrame(rows)


def importance_ranks(boot: pd.DataFrame) -> pd.DataFrame:
    """Share of bootstrap resamples in which each feature is the most important one."""
    top = boot.idxmax(axis=1).value_counts(normalize=True)
    return pd.DataFrame({"feature": boot.columns,
                         "median": boot.median().to_numpy(),
                         "p05": boot.quantile(0.05).to_numpy(),
                         "p95": boot.quantile(0.95).to_numpy(),
                         "top1_share": [float(top.get(c, 0.0)) for c in boot.columns]})


def risk_map(results: pd.DataFrame, x: str, y: str, x_edges: Sequence[float], y_edges: Sequence[float],
             min_count: int = 3) -> Dict[str, np.ndarray]:
    """Empirical failure probability per bin (Eq. 5); bins with < min_count trajectories are NaN."""
    xe, ye = np.asarray(x_edges, dtype=float), np.asarray(y_edges, dtype=float)
    xi = np.clip(np.digitize(results[x], xe) - 1, 0, len(xe) - 2)
    yi = np.clip(np.digitize(results[y], ye) - 1, 0, len(ye) - 2)
    fails = np.zeros((len(ye) - 1, len(xe) - 1))
    counts = np.zeros_like(fails)
    np.add.at(fails, (yi, xi), results["failure"].to_numpy())
    np.add.at(counts, (yi, xi), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(counts >= min_count, fails / counts, np.nan)
    return {"rate": rate, "counts": counts, "x_edges": xe, "y_edges": ye}


def leaf_rules(clf: DecisionTreeClassifier, features: List[str], target: int = 1) -> List[Dict[str, Any]]:
    """One rule per leaf whose majority class is `target`, with merged feature bounds."""
    t = clf.tree_
    out: List[Dict[str, Any]] = []

    def render(bounds):
        parts = []
        for f in features:
            if f in bounds:
                lo, hi = bounds[f]
                if lo > -np.inf and hi < np.inf:
                    parts.append(f"{lo:.2f} < {f} <= {hi:.2f}")
                elif lo > -np.inf:
                    parts.append(f"{f} > {lo:.2f}")
                else:
                    parts.append(f"{f} <= {hi:.2f}")
        return " and ".join(parts) or "(always)"

    def walk(node, bounds):
        if t.children_left[node] == t.children_right[node]:
            share = t.value[node][0] / t.value[node][0].sum()
            if int(np.argmax(share)) == target:
                out.append({"rule": render(bounds), "samples": int(t.n_node_samples[node]),
                            "failure_share": float(share[1])})
            return
        f, thr = features[t.feature[node]], float(t.threshold[node])
        lo, hi = bounds.get(f, (-np.inf, np.inf))
        walk(t.children_left[node], {**bounds, f: (lo, min(hi, thr))})
        walk(t.children_right[node], {**bounds, f: (max(lo, thr), hi)})

    walk(0, {})
    return sorted(out, key=lambda r: -r["samples"])


def horizon_table(results: pd.DataFrame) -> pd.DataFrame:
    """ADE and failure rate by prediction horizon (number of predicted steps)."""
    g = results.groupby("horizon")
    return pd.DataFrame({"trajectories": g.size(), "mean_ade": g["ade"].mean(),
                         "failure_rate": g["failure"].mean()}).reset_index()
