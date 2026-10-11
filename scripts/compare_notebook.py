"""
Check that the tfa package gives the same per-trajectory results as the original
notebook (SafeTraj_ETH.ipynb).

The notebook's analysis cells (data loading, prediction and evaluation, decision
trees) are executed unchanged except for the data path, which points to the
pinned files of scripts/download_data.py instead of a fresh OpenTraj clone. Their
outputs are compared with tfa at full precision (tolerance 1e-9) and with the
committed CSV files in results/reproduction/, which store six decimals
(rounding error up to 5e-7; tolerance 6e-7).

    python scripts/download_data.py
    python scripts/compare_notebook.py        # exit code 1 if anything differs
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tfa.analysis import fit_tree  # noqa: E402
from tfa.data import load_scene, verify  # noqa: E402
from tfa.pipeline import evaluate_scene, label_failures  # noqa: E402

NOTEBOOK_CELLS = (4, 6, 8)          # load data; predict and evaluate; decision trees
NOTEBOOK_DATA_PATH = "OpenTraj/datasets/ETH/"
COLUMNS = ["ade", "fde", "init_speed", "init_orientation", "init_x", "init_y"]
FEATURES = ["init_speed", "init_orientation", "init_x", "init_y"]


def run_notebook(notebook: Path, data_dir: Path) -> dict:
    """Execute the notebook's analysis cells; return its namespace."""
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    namespace: dict = {}
    for index in NOTEBOOK_CELLS:
        cell = nb["cells"][index]
        assert cell["cell_type"] == "code", f"cell {index} is not a code cell"
        source = "".join(cell["source"]).replace(NOTEBOOK_DATA_PATH, data_dir.as_posix().rstrip("/") + "/")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, f"{notebook.name}:cell{index}", "exec"), namespace)
    return namespace


def compare(notebook: Path, data_dir: Path, committed: Path, tol: float = 1e-9, csv_tol: float = 6e-7) -> dict:
    nb = run_notebook(notebook, data_dir)
    report = {"scenes": {}, "match": True}
    for scene, key in (("ETH", "eth"), ("Hotel", "hotel")):
        theirs = nb[f"results_{key}"].reset_index(drop=True)
        ours, threshold = label_failures(evaluate_scene(load_scene(data_dir, scene), scene))
        ours = ours.reset_index(drop=True)
        csv = pd.read_csv(committed / f"results_{key}.csv")
        tree = fit_tree(ours)
        same_ids = (len(theirs) == len(ours) == len(csv)
                    and (theirs.ped_id.to_numpy() == ours.ped_id.to_numpy()).all()
                    and (theirs.ped_id.to_numpy() == csv.ped_id.to_numpy()).all())
        r = {
            "rows": [len(theirs), len(ours), len(csv)],
            "same_trajectories": bool(same_ids),
            "max_abs_diff_package": float(np.abs(theirs[COLUMNS].to_numpy() - ours[COLUMNS].to_numpy()).max()) if same_ids else None,
            "max_abs_diff_committed_csv": float(np.abs(theirs[COLUMNS].to_numpy() - csv[COLUMNS].to_numpy()).max()) if same_ids else None,
            "same_failure_labels": bool(same_ids and (theirs.failure.to_numpy() == ours.failure.to_numpy()).all()
                                        and (theirs.failure.to_numpy() == csv.failure.to_numpy()).all()),
            "threshold_diff": abs(float(nb[f"threshold_{key}"]) - float(threshold)),
            "importance_max_diff": float(np.abs(nb[f"clf_{key}"].feature_importances_
                                                - tree["model"].feature_importances_).max()),
            "train_accuracy_diff": abs(nb[f"clf_{key}"].score(theirs[FEATURES], theirs["failure"]) - tree["train_accuracy"]),
        }
        r["match"] = bool(r["same_trajectories"] and r["same_failure_labels"]
                          and r["max_abs_diff_package"] <= tol and r["max_abs_diff_committed_csv"] <= csv_tol
                          and r["threshold_diff"] <= tol and r["importance_max_diff"] <= tol
                          and r["train_accuracy_diff"] <= tol)
        report["scenes"][scene] = r
        report["match"] &= r["match"]
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--notebook", default=str(ROOT / "SafeTraj_ETH.ipynb"))
    p.add_argument("--data", default=str(ROOT / "data" / "eth"))
    p.add_argument("--committed", default=str(ROOT / "results" / "reproduction"))
    args = p.parse_args()
    status = verify(args.data)
    if not all(status.values()):
        raise SystemExit(f"dataset missing or modified: {status}. Run scripts/download_data.py first.")
    report = compare(Path(args.notebook), Path(args.data), Path(args.committed))
    for scene, r in report["scenes"].items():
        print(f"{scene}: rows notebook/package/csv {r['rows']}, same trajectories {r['same_trajectories']}, "
              f"same failure labels {r['same_failure_labels']}")
        print(f"  max |difference| vs package {r['max_abs_diff_package']:.1e}, vs committed CSV "
              f"{r['max_abs_diff_committed_csv']:.1e}; threshold {r['threshold_diff']:.1e}; "
              f"tree importances {r['importance_max_diff']:.1e}; training accuracy {r['train_accuracy_diff']:.1e}")
    print("MATCH" if report["match"] else "MISMATCH")
    return 0 if report["match"] else 1


if __name__ == "__main__":
    sys.exit(main())
