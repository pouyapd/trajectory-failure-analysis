"""
plots.py — figures in one visual language (colour-blind-checked palette,
recessive grid, explicit units). Functions return matplotlib Figures.
"""

from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.tree import plot_tree

SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SCENE_COLORS = {"ETH": "#2a78d6", "Hotel": "#eb6834"}
FEATURE_LABELS = {"init_speed": "speed s0", "init_orientation": "orientation φ0", "init_x": "position x0",
                  "init_y": "position y0"}


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"], "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelcolor": INK_2, "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def note(fig, text: str, y: float = -0.02) -> None:
    fig.text(0.01, y, text, fontsize=8, color=MUTED, ha="left", va="top")


def _heatmap(ax, rm: Dict, xlabel: str, ylabel: str, title: str, fmt_count: bool = True):
    xe, ye, rate, counts = rm["x_edges"], rm["y_edges"], rm["rate"], rm["counts"]
    ax.grid(False)
    mesh = ax.pcolormesh(xe, ye, np.ma.masked_invalid(rate), cmap="Oranges", vmin=0, vmax=1, shading="flat")
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            cx, cy = (xe[j] + xe[j + 1]) / 2, (ye[i] + ye[i + 1]) / 2
            if np.isnan(rate[i, j]):
                if counts[i, j] > 0:
                    ax.text(cx, cy, f"n={int(counts[i, j])}", ha="center", va="center", fontsize=6, color=MUTED)
                continue
            ax.text(cx, cy, f"{rate[i, j]:.0%}" + (f"\nn={int(counts[i, j])}" if fmt_count else ""), ha="center",
                    va="center", fontsize=6.5, color="white" if rate[i, j] > 0.6 else INK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return mesh


def risk_maps_figure(maps: Dict[str, Dict], xlabel: str, ylabel: str, title: str, footnote: str,
                     equal_aspect: bool = False):
    apply_style()
    fig, axes = plt.subplots(1, len(maps), figsize=(6.2 * len(maps), 4.6))
    axes = np.atleast_1d(axes)
    mesh = None
    for ax, (scene, rm) in zip(axes, maps.items()):
        mesh = _heatmap(ax, rm, xlabel, ylabel, f"{scene} scene")
        if equal_aspect:
            ax.set_aspect("equal")
    cb = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("failure rate (ADE > scene threshold)")
    cb.outline.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold")
    note(fig, footnote)
    return fig


def ade_distribution_figure(results: Dict[str, pd.DataFrame], thresholds: Dict[str, float], footnote: str):
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for scene, res in results.items():
        x = np.sort(res["ade"].to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        c = SCENE_COLORS[scene]
        ax.step(x, y, where="post", color=c, label=f"{scene} (n={len(x)}, mean ADE {x.mean():.3f} m)")
        thr = thresholds[scene]
        ax.plot([thr, thr], [0, 0.75], color=c, linewidth=1.0, linestyle=(0, (3, 2)))
        ax.plot(thr, 0.75, marker="o", markersize=6, color=c, markeredgecolor=SURFACE)
        ax.annotate(f"{scene} threshold {thr:.2f} m", (thr, 0.75), xytext=(6, -14) if scene == "ETH" else (-122, 6),
                    textcoords="offset points", fontsize=8, color=INK_2)
    ax.axhline(0.75, color=BASELINE, linewidth=0.8)
    ax.set_xlabel("ADE (m)")
    ax.set_ylabel("share of trajectories with ADE ≤ x")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.legend(loc="lower right")
    ax.set_title("ADE distributions and the 75th-percentile failure thresholds")
    note(fig, footnote)
    return fig


def tree_figure(clf, features: List[str], title: str, footnote: str):
    """Decision tree with an explicit colour key (sklearn colours class 0 = Success orange, class 1 = Failure blue)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(18, 7.5))
    ax.grid(False)
    plot_tree(clf, feature_names=features, class_names=["Success", "Failure"], filled=True, rounded=True,
              impurity=False, proportion=False, fontsize=9, ax=ax)
    ax.legend(handles=[Patch(facecolor="#e58139", label="majority Success (class 0)"),
                       Patch(facecolor="#399de5", label="majority Failure (class 1)")],
              loc="upper right", fontsize=10, frameon=True)
    ax.set_title(title, fontsize=13)
    note(fig, footnote, y=0.0)
    return fig


def dot_compare_figure(rows: pd.DataFrame, title: str, footnote: str):
    """rows: scene, metric, value, err (optional). One panel, metrics on the y-axis, scenes as colours."""
    apply_style()
    metrics = list(dict.fromkeys(rows["metric"]))
    fig, ax = plt.subplots(figsize=(7.4, 0.55 * len(metrics) + 1.4))
    offsets = {"ETH": -0.12, "Hotel": 0.12}
    for scene, g in rows.groupby("scene", sort=False):
        ys = [metrics.index(m) + offsets.get(scene, 0) for m in g["metric"]]
        err = g["err"].to_numpy() if "err" in g else None
        ax.errorbar(g["value"], ys, xerr=err, fmt="o", color=SCENE_COLORS[scene], markersize=7, capsize=0,
                    elinewidth=1.2, label=scene)
        for x, y in zip(g["value"], ys):
            ax.annotate(f"{x:.3f}", (x, y), xytext=(7, -3), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy")
    ax.legend(loc="lower right")
    ax.set_title(title)
    note(fig, footnote)
    return fig


def importance_stability_figure(boot: Dict[str, pd.DataFrame], paper: Dict[str, Dict[str, float]], footnote: str):
    apply_style()
    features = list(next(iter(boot.values())).columns)
    fig, axes = plt.subplots(1, len(boot), figsize=(6.2 * len(boot), 3.6), sharey=True)
    for ax, (scene, b) in zip(np.atleast_1d(axes), boot.items()):
        data = [b[f].to_numpy() for f in features]
        bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.5), whiskerprops=dict(color=MUTED),
                        capprops=dict(color=MUTED))
        for patch in bp["boxes"]:
            patch.set(facecolor=SCENE_COLORS[scene], alpha=0.25, edgecolor=SCENE_COLORS[scene])
        for i, f in enumerate(features):
            ax.plot(paper[scene][f], i + 1, marker="D", color=SCENE_COLORS[scene], markersize=7,
                    markeredgecolor=SURFACE, zorder=5)
        ax.set_yticks(range(1, len(features) + 1))
        ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in features])
        ax.set_xlim(-0.02, 1.0)
        ax.set_xlabel("Gini importance")
        ax.set_title(f"{scene}: importances over bootstrap resamples")
        ax.grid(axis="y", visible=False)
    note(fig, footnote)
    return fig


def horizon_figure(tables: Dict[str, pd.DataFrame], footnote: str):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for scene, t in tables.items():
        c = SCENE_COLORS[scene]
        axes[0].plot(t["horizon"], t["mean_ade"], marker="o", color=c, label=scene)
        axes[1].plot(t["horizon"], t["failure_rate"] * 100, marker="o", color=c, label=scene)
        dy = 7 if scene == "Hotel" else -13
        for _, r in t.iterrows():
            axes[1].annotate(f"n={int(r.trajectories)}", (r.horizon, r.failure_rate * 100), xytext=(0, dy),
                             textcoords="offset points", fontsize=7, color=c, ha="center")
    axes[0].set_title("Mean ADE by prediction horizon")
    axes[0].set_ylabel("mean ADE (m)")
    axes[1].set_title("Failure rate by prediction horizon")
    axes[1].set_ylabel("failure rate (%)")
    for ax in axes:
        ax.set_xlabel("predicted steps (0.4 s each)")
        ax.set_xticks(range(2, 9))
    axes[0].legend(loc="upper left")
    note(fig, footnote)
    return fig


def threshold_sensitivity_figure(df: pd.DataFrame, footnote: str):
    """df: scene, percentile, feature, importance."""
    apply_style()
    scenes = list(dict.fromkeys(df["scene"]))
    fig, axes = plt.subplots(1, len(scenes), figsize=(6.2 * len(scenes), 3.8), sharey=True)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for ax, scene in zip(np.atleast_1d(axes), scenes):
        d = df[df.scene == scene]
        for color, (feat, g) in zip(colors, d.groupby("feature", sort=False)):
            ax.plot(g["percentile"], g["importance"], marker="o", markersize=5, color=color,
                    label=FEATURE_LABELS.get(feat, feat))
        ax.axvline(75, color=MUTED, linewidth=1.0, linestyle=(0, (3, 3)))
        ax.text(75.5, 0.95, "preprint (75th)", fontsize=8, color=INK_2, va="top")
        ax.set_xlabel("failure threshold (ADE percentile)")
        ax.set_title(f"{scene}: tree importances vs. threshold")
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("Gini importance")
    np.atleast_1d(axes)[-1].legend(loc="upper right", fontsize=8)
    note(fig, footnote)
    return fig
