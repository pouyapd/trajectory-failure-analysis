"""
pipeline.py — per-pedestrian prediction, errors, initial-state features and
failure labels for one scene.

With the default ExperimentConfig this is the preprint's method, written as
reusable functions (the original notebook is unchanged in the repository):

  * observe the first half of each track, take the last observed position and
    velocity, predict up to 8 future positions with the damped constant-velocity
    model p_{t+1} = p_t + v_t * dt, v_t = 0.98 * v_{t-1} (called a "simplified
    Social Force Model" in the preprint: interaction forces set to zero);
  * ADE / FDE against the second half of the track (truncated to 8 steps);
  * failure = ADE above the scene's 75th percentile (data-driven threshold).

`predict_fn` can be replaced by any predictor with the same signature to apply
the same analysis to another model.
"""

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from .config import PAPER, ExperimentConfig
from .metrics import ade, fde


def damped_constant_velocity(position: np.ndarray, velocity: np.ndarray, n_steps: int,
                             dt: float = 0.4, damping: float = 0.98) -> np.ndarray:
    """Predict n_steps positions; the velocity is damped before every step (as in the notebook)."""
    pos = np.asarray(position, dtype=float).copy()
    vel = np.asarray(velocity, dtype=float).copy()
    out = np.empty((n_steps, 2))
    for k in range(n_steps):
        vel = vel * damping
        pos = pos + vel * dt
        out[k] = pos
    return out


PredictFn = Callable[[np.ndarray, np.ndarray, np.ndarray, int, ExperimentConfig], np.ndarray]


def _default_predict(obs_pos, obs_vel, _gt_unused, n_steps, cfg: ExperimentConfig) -> np.ndarray:
    return damped_constant_velocity(obs_pos[-1], obs_vel[-1], n_steps, cfg.dt, cfg.damping)


def split_track(track: pd.DataFrame, cfg: ExperimentConfig = PAPER) -> Optional[Dict[str, np.ndarray]]:
    """Observation / ground-truth split of one pedestrian's track (None if too short)."""
    positions = track[["x", "y"]].to_numpy(dtype=float)
    velocities = track[["vx", "vy"]].to_numpy(dtype=float)
    if len(positions) < cfg.min_track_len:
        return None
    obs_len = int(len(positions) * cfg.obs_fraction) if cfg.obs_fraction != 0.5 else len(positions) // 2
    if obs_len < 1:
        return None
    gt = positions[obs_len:obs_len + cfg.max_pred_steps]
    if len(gt) < cfg.min_gt_steps:
        return None
    return {"obs_pos": positions[:obs_len], "obs_vel": velocities[:obs_len], "gt_pos": gt}


def evaluate_scene(df: pd.DataFrame, scene: str = "", cfg: ExperimentConfig = PAPER,
                   predict_fn: Optional[PredictFn] = None) -> pd.DataFrame:
    """
    One row per evaluated pedestrian: ped_id, ade, fde, horizon, init_speed,
    init_speed_measured, speed_substituted, init_orientation, init_x, init_y,
    track_len, scene. Failure labels are added by `label_failures`.
    """
    predict_fn = predict_fn or _default_predict
    rows = []
    for ped_id, track in df.groupby("ped_id", sort=True):
        track = track.sort_values("frame")
        split = split_track(track, cfg)
        if split is None:
            continue
        gt = split["gt_pos"]
        pred = predict_fn(split["obs_pos"], split["obs_vel"], gt, len(gt), cfg)
        horizon = min(len(gt), len(pred))
        if horizon < cfg.min_gt_steps:
            continue
        curr_pos, curr_vel = split["obs_pos"][-1], split["obs_vel"][-1]
        speed = float(np.linalg.norm(curr_vel))
        substituted = speed < cfg.min_speed and cfg.speed_substitute is not None
        rows.append({
            "ped_id": ped_id,
            "ade": ade(pred[:horizon], gt[:horizon]),
            "fde": fde(pred[:horizon], gt[:horizon]),
            "horizon": horizon,
            "track_len": len(track),
            "init_speed": float(cfg.speed_substitute) if substituted else speed,
            "init_speed_measured": speed,
            "speed_substituted": bool(substituted),
            "init_orientation": float(np.arctan2(curr_vel[1], curr_vel[0])),
            "init_x": float(curr_pos[0]),
            "init_y": float(curr_pos[1]),
            "scene": scene,
        })
    if not rows:
        raise ValueError(f"no pedestrian in scene '{scene}' has a long enough track")
    return pd.DataFrame(rows)


def label_failures(results: pd.DataFrame, percentile: float = PAPER.failure_percentile):
    """Add `failure` (ADE > percentile threshold); returns (results, threshold)."""
    threshold = float(np.percentile(results["ade"], percentile))
    out = results.copy()
    out["failure"] = (out["ade"] > threshold).astype(int)
    return out, threshold


def scene_summary(results: pd.DataFrame, threshold: float) -> Dict[str, float]:
    return {
        "trajectories": int(len(results)),
        "mean_ade": float(results["ade"].mean()),
        "mean_fde": float(results["fde"].mean()),
        "threshold": float(threshold),
        "failure_rate": float(results["failure"].mean()),
        "full_horizon_share": float((results["horizon"] == results["horizon"].max()).mean()),
        "speed_substituted": int(results["speed_substituted"].sum()),
    }
