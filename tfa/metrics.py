"""
metrics.py — displacement errors with input validation.

ADE = (1 / H) * sum_{t=1..H} || p_hat_t - p_t ||_2      (Eq. 2 of the preprint, H = horizon)
FDE = || p_hat_H - p_H ||_2                              (Eq. 3)

When prediction and ground truth have different lengths, both are truncated
to the shorter one, as in the original notebook.
"""

import numpy as np


def _validate(pred, gt):
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape[1] != 2 or gt.shape[1] != 2:
        raise ValueError(f"expected arrays of shape (T, 2), got {pred.shape} and {gt.shape}")
    n = min(len(pred), len(gt))
    if n == 0:
        raise ValueError("empty trajectory")
    pred, gt = pred[:n], gt[:n]
    if not (np.isfinite(pred).all() and np.isfinite(gt).all()):
        raise ValueError("NaN or infinite coordinates")
    return pred, gt


def displacement(pred, gt) -> np.ndarray:
    """Per-step Euclidean error over the common horizon."""
    pred, gt = _validate(pred, gt)
    return np.sqrt(np.sum((pred - gt) ** 2, axis=1))


def ade(pred, gt) -> float:
    """Average displacement error (metres)."""
    return float(np.mean(displacement(pred, gt)))


def fde(pred, gt) -> float:
    """Final displacement error at the last common step (metres)."""
    return float(displacement(pred, gt)[-1])
