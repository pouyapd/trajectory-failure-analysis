"""
Reproduce the preprint's Table 1, feature importances and figures from the raw
ETH files, and check every number against the values reported in the preprint.

    python scripts/download_data.py
    python scripts/reproduce.py                       # exit code 1 if any number differs

Outputs (default results/reproduction/): per-trajectory results, table1.csv,
reproduction.json, decision-tree rules; figures go to docs/figures/.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tfa.analysis import fit_tree, leaf_rules, risk_map  # noqa: E402
from tfa.config import PAPER, PAPER_IMPORTANCES, PAPER_TABLE1  # noqa: E402
from tfa.data import OPENTRAJ_COMMIT, SCENES, load_scene, verify  # noqa: E402
from tfa.pipeline import evaluate_scene, label_failures, scene_summary  # noqa: E402
from tfa import plots  # noqa: E402

# Precision at which the preprint reports each value.
DECIMALS = {"trajectories": 0, "mean_ade": 3, "mean_fde": 3, "threshold": 2, "failure_rate": 3, "tree_accuracy": 3}


def run(data_dir: Path, out: Path, fig_dir: Path) -> bool:
    status = verify(data_dir)
    if not all(status.values()):
        raise SystemExit(f"dataset missing or modified: {status}. Run scripts/download_data.py first.")
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results, thresholds, trees, rows = {}, {}, {}, []
    all_match = True
    for scene in SCENES:
        res, thr = label_failures(evaluate_scene(load_scene(data_dir, scene), scene), PAPER.failure_percentile)
        tree = fit_tree(res, PAPER)
        results[scene], thresholds[scene], trees[scene] = res, thr, tree
        res.to_csv(out / f"results_{scene.lower()}.csv", index=False, float_format="%.6f")
        summary = scene_summary(res, thr)
        summary["tree_accuracy"] = tree["train_accuracy"]
        for key, paper_value in PAPER_TABLE1[scene].items():
            value = summary[key]
            match = round(value, DECIMALS[key]) == round(paper_value, DECIMALS[key])
            all_match &= match
            rows.append({"scene": scene, "metric": key, "preprint": paper_value, "reproduced": value,
                         "reproduced_rounded": round(value, DECIMALS[key]), "match": match})
        for feat, paper_value in PAPER_IMPORTANCES[scene].items():
            value = tree["importances"][feat]
            match = round(value, 3) == paper_value
            all_match &= match
            rows.append({"scene": scene, "metric": f"importance_{feat}", "preprint": paper_value, "reproduced": value,
                         "reproduced_rounded": round(value, 3), "match": match})
        rules = leaf_rules(tree["model"], PAPER.features)
        (out / f"tree_rules_{scene.lower()}.txt").write_text(
            "\n".join(f"IF {r['rule']} THEN failure ({r['samples']} trajectories, {r['failure_share']:.0%} failures)"
                      for r in rules) + "\n", encoding="utf-8")

    table = pd.DataFrame(rows)
    table.to_csv(out / "table1.csv", index=False)
    (out / "reproduction.json").write_text(json.dumps({
        "opentraj_commit": OPENTRAJ_COMMIT,
        "dataset_sha256": {s: i["sha256"] for s, i in SCENES.items()},
        "config": PAPER.to_dict(),
        "all_values_match_preprint": bool(all_match),
        "checks": rows,
        "importances": {s: t["importances"] for s, t in trees.items()},
        "majority_baseline": {s: t["majority_baseline"] for s, t in trees.items()},
        "extra": {s: scene_summary(results[s], thresholds[s]) for s in SCENES},
    }, indent=2, default=float), encoding="utf-8")

    # ── figures (new files; the preprint's original PNGs in the repository root are unchanged) ──
    src = "Reproduced from the raw ETH files with the preprint's method (scripts/reproduce.py)."
    orient_edges = np.linspace(-np.pi, np.pi, 9)
    speed_edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.6])
    maps = {s: risk_map(results[s], "init_orientation", "init_speed", orient_edges, speed_edges) for s in SCENES}
    plots.risk_maps_figure(maps, "initial orientation φ0 (rad)", "initial speed s0 (m/s)",
                           "Risk maps in orientation-speed space (Eq. 5, binned)",
                           src + " Cell text: failure rate and number of trajectories; bins with < 3 "
                           "trajectories show only n. Speeds of 0.5 m/s include near-stationary pedestrians "
                           "whose speed feature the method sets to 0.5.").savefig(fig_dir / "risk_maps_orientation_speed.png")
    spatial = {}
    for s in SCENES:
        r = results[s]
        xe = np.linspace(np.floor(r.init_x.min()), np.ceil(r.init_x.max()), 7)
        ye = np.linspace(np.floor(r.init_y.min()), np.ceil(r.init_y.max()), 7)
        spatial[s] = risk_map(r, "init_x", "init_y", xe, ye)
    plots.risk_maps_figure(spatial, "initial position x0 (m)", "initial position y0 (m)",
                           "Spatial risk maps (Eq. 5, binned)",
                           src + " Positions in each scene's own world frame (not comparable across scenes).") \
        .savefig(fig_dir / "risk_maps_spatial.png")
    plots.ade_distribution_figure(results, thresholds, src).savefig(fig_dir / "ade_distributions.png")
    plots.tree_figure(trees["ETH"]["model"], PAPER.features,
                      "Decision tree, ETH scene (depth ≤ 4, ≥ 10 trajectories per leaf; training accuracy "
                      f"{trees['ETH']['train_accuracy']:.1%})",
                      src + " Colours: orange = majority Success, blue = majority Failure.") \
        .savefig(fig_dir / "decision_tree_eth_readable.png")
    return all_match


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(ROOT / "data" / "eth"))
    p.add_argument("--out", default=str(ROOT / "results" / "reproduction"))
    p.add_argument("--figures", default=str(ROOT / "docs" / "figures"))
    args = p.parse_args()
    ok = run(Path(args.data_dir), Path(args.out), Path(args.figures))
    table = pd.read_csv(Path(args.out) / "table1.csv")
    print(table[["scene", "metric", "preprint", "reproduced_rounded", "match"]].to_string(index=False))
    print("\nALL VALUES MATCH THE PREPRINT" if ok else "\nSOME VALUES DIFFER FROM THE PREPRINT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
