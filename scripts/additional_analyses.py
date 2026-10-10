"""
Additional analyses (NOT part of the preprint). They reuse the preprint's data,
predictor and tree settings and change one thing at a time:

  A1  held-out accuracy: repeated stratified 5-fold CV of the same tree, with
      the majority-class baseline and balanced accuracy (the preprint reports
      training accuracy);
  A2  stability of the Gini importances over 500 bootstrap resamples;
  A3  prediction-horizon confound: ADE and failure rate by number of predicted
      steps, and the analysis repeated on full-horizon (8-step) tracks only;
  A4  speed feature without the 0.5 m/s substitution for near-stationary pedestrians;
  A5  failure threshold at the 60th-90th ADE percentile instead of the 75th.

    python scripts/additional_analyses.py      # -> results/additional/, docs/figures/additional_*.png
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from tfa import plots  # noqa: E402
from tfa.analysis import (bootstrap_importances, cross_validated, fit_tree, horizon_table,  # noqa: E402
                          importance_ranks)
from tfa.config import PAPER  # noqa: E402
from tfa.data import SCENES, load_scene, verify  # noqa: E402
from tfa.pipeline import evaluate_scene, label_failures  # noqa: E402

TAG = "Additional analysis, not in the preprint."


def analyse(results: pd.DataFrame, cfg=PAPER, cv_repeats: int = 20):
    tree = fit_tree(results, cfg)
    cv = cross_validated(results, cfg, n_repeats=cv_repeats)
    return {"n": int(len(results)), "failure_rate": float(results.failure.mean()),
            "train_accuracy": tree["train_accuracy"], **cv,
            "importances": {k: round(v, 4) for k, v in tree["importances"].items()}}


def run(data_dir: Path, out: Path, fig_dir: Path, n_boot: int) -> dict:
    if not all(verify(data_dir).values()):
        raise SystemExit("dataset missing or modified; run scripts/download_data.py first")
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    raw = {s: load_scene(data_dir, s) for s in SCENES}
    base = {}
    for s in SCENES:
        res, thr = label_failures(evaluate_scene(raw[s], s), PAPER.failure_percentile)
        base[s] = res
    summary = {"note": TAG, "config": PAPER.to_dict(), "A1_heldout": {}, "A2_bootstrap": {}, "A3_horizon": {},
               "A4_no_speed_substitution": {}, "A5_threshold": []}

    # A1 ── held-out accuracy
    rows = []
    for s, res in base.items():
        a = analyse(res)
        summary["A1_heldout"][s] = a
        rows += [{"scene": s, "metric": "training accuracy (preprint)", "value": a["train_accuracy"], "err": 0.0},
                 {"scene": s, "metric": "held-out accuracy (5-fold CV x 20)", "value": a["cv_accuracy_mean"],
                  "err": a["cv_accuracy_std"]},
                 {"scene": s, "metric": "majority-class baseline", "value": a["majority_baseline"], "err": 0.0},
                 {"scene": s, "metric": "held-out balanced accuracy", "value": a["cv_balanced_accuracy_mean"],
                  "err": a["cv_balanced_accuracy_std"]}]
    plots.dot_compare_figure(pd.DataFrame(rows), "Decision-tree accuracy: training vs. held-out",
                             TAG + " Whiskers: standard deviation over 100 folds. Balanced accuracy: mean of the "
                             "success and failure recall (0.5 = chance).") \
        .savefig(fig_dir / "additional_tree_accuracy.png")

    # A2 ── bootstrap stability of importances
    boots = {}
    for s, res in base.items():
        boots[s] = bootstrap_importances(res, PAPER, n_boot=n_boot, seed=0)
        boots[s].to_csv(out / f"bootstrap_importances_{s.lower()}.csv", index=False, float_format="%.5f")
        summary["A2_bootstrap"][s] = importance_ranks(boots[s]).round(4).to_dict(orient="records")
    paper_imp = {s: fit_tree(res)["importances"] for s, res in base.items()}
    plots.importance_stability_figure(boots, paper_imp,
                                      TAG + f" Boxes: {n_boot} bootstrap resamples (median, quartiles, 5-95% "
                                            "whiskers); diamonds: the tree fitted on all data (the preprint's values).") \
        .savefig(fig_dir / "additional_importance_stability.png")

    # A3 ── horizon confound
    tables = {}
    for s, res in base.items():
        tables[s] = horizon_table(res)
        full = res[res.horizon == PAPER.max_pred_steps].drop(columns=["failure"])
        full, thr_full = label_failures(full, PAPER.failure_percentile)
        summary["A3_horizon"][s] = {
            "by_horizon": tables[s].round(4).to_dict(orient="records"),
            "full_horizon_share": float((res.horizon == PAPER.max_pred_steps).mean()),
            "full_horizon_only": {"threshold": thr_full, **analyse(full)},
        }
    plots.horizon_figure(tables, TAG + " Each trajectory is predicted for min(8, second-half length) steps; "
                                       "the failure label uses one threshold per scene.") \
        .savefig(fig_dir / "additional_horizon.png")

    # A4 ── no speed substitution
    cfg_nosub = replace(PAPER, speed_substitute=None)
    for s in SCENES:
        res, _ = label_failures(evaluate_scene(raw[s], s, cfg_nosub), PAPER.failure_percentile)
        summary["A4_no_speed_substitution"][s] = {"substituted_in_preprint": int(base[s].speed_substituted.sum()),
                                                  **analyse(res, cfg_nosub)}

    # A5 ── threshold sensitivity
    imp_rows = []
    for pct in (60, 65, 70, 75, 80, 85, 90):
        cfg_p = replace(PAPER, failure_percentile=float(pct))
        for s in SCENES:
            res, thr = label_failures(base[s].drop(columns=["failure"]), pct)
            a = analyse(res, cfg_p, cv_repeats=10)
            summary["A5_threshold"].append({"scene": s, "percentile": pct, "threshold": thr, **a})
            imp_rows += [{"scene": s, "percentile": pct, "feature": f, "importance": v}
                         for f, v in a["importances"].items()]
    plots.threshold_sensitivity_figure(pd.DataFrame(imp_rows),
                                       TAG + " Same data and tree settings; only the failure percentile changes.") \
        .savefig(fig_dir / "additional_threshold_sensitivity.png")

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(ROOT / "data" / "eth"))
    p.add_argument("--out", default=str(ROOT / "results" / "additional"))
    p.add_argument("--figures", default=str(ROOT / "docs" / "figures"))
    p.add_argument("--bootstrap", type=int, default=500)
    args = p.parse_args()
    s = run(Path(args.data_dir), Path(args.out), Path(args.figures), args.bootstrap)
    for scene in SCENES:
        a = s["A1_heldout"][scene]
        print(f"{scene}: train acc {a['train_accuracy']:.3f} | CV acc {a['cv_accuracy_mean']:.3f} ± {a['cv_accuracy_std']:.3f}"
              f" | majority {a['majority_baseline']:.3f} | CV balanced acc {a['cv_balanced_accuracy_mean']:.3f}")
        tops = {r['feature']: r['top1_share'] for r in s["A2_bootstrap"][scene]}
        print(f"   bootstrap: share of resamples where feature is most important: {tops}")
        f = s["A3_horizon"][scene]["full_horizon_only"]
        print(f"   full-horizon only: n={f['n']} importances {f['importances']} CV acc {f['cv_accuracy_mean']:.3f} "
              f"vs majority {f['majority_baseline']:.3f}")
        ns = s["A4_no_speed_substitution"][scene]
        print(f"   no speed substitution ({ns['substituted_in_preprint']} affected): importances {ns['importances']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
