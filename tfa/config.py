"""
config.py — experiment parameters.

`ExperimentConfig()` with default values reproduces the preprint (the original
notebook `SafeTraj_ETH.ipynb`) exactly. The parameters exist so that
sensitivity analyses can change one thing at a time; results produced with
non-default values are additional analyses, not the preprint's results.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ExperimentConfig:
    # trajectory split and prediction (notebook: social_force_predict)
    min_track_len: int = 4            # tracks with fewer points are skipped (the text says 5; the code uses 4)
    obs_fraction: float = 0.5         # first half observed, second half is ground truth
    max_pred_steps: int = 8           # prediction horizon is capped at 8 steps (3.2 s)
    min_gt_steps: int = 2             # at least 2 ground-truth points are needed
    dt: float = 0.4                   # seconds between annotations (2.5 Hz)
    damping: float = 0.98             # velocity multiplied by this factor before every step
    # features
    min_speed: float = 0.01           # below this the notebook replaces the speed *feature* ...
    speed_substitute: Optional[float] = 0.5   # ... by 0.5 m/s (None = keep the measured speed)
    # failure label
    failure_percentile: float = 75.0  # failure if ADE > this percentile of the scene's ADEs
    # decision tree (notebook settings)
    features: List[str] = field(default_factory=lambda: ["init_speed", "init_orientation", "init_x", "init_y"])
    max_depth: int = 4
    min_samples_leaf: int = 10
    random_state: int = 42

    def __post_init__(self):
        if self.min_track_len < 2:
            raise ValueError("min_track_len must be >= 2")
        if not 0 < self.obs_fraction < 1:
            raise ValueError("obs_fraction must be in (0, 1)")
        if self.max_pred_steps < 1 or self.min_gt_steps < 1:
            raise ValueError("max_pred_steps and min_gt_steps must be >= 1")
        if self.dt <= 0 or not 0 < self.damping <= 1:
            raise ValueError("need dt > 0 and 0 < damping <= 1")
        if not 0 < self.failure_percentile < 100:
            raise ValueError("failure_percentile must be in (0, 100)")
        if self.max_depth < 1 or self.min_samples_leaf < 1:
            raise ValueError("max_depth and min_samples_leaf must be >= 1")

    def to_dict(self) -> Dict:
        return asdict(self)


PAPER = ExperimentConfig()

# Values reported in the preprint (Table 1 and Section 4.5), used to check the reproduction.
PAPER_TABLE1 = {
    "ETH": {"trajectories": 353, "mean_ade": 0.326, "mean_fde": 0.653, "threshold": 0.41, "failure_rate": 0.249,
            "tree_accuracy": 0.793},
    "Hotel": {"trajectories": 366, "mean_ade": 0.238, "mean_fde": 0.480, "threshold": 0.33, "failure_rate": 0.251,
              "tree_accuracy": 0.820},
}
PAPER_IMPORTANCES = {
    "ETH": {"init_orientation": 0.400, "init_y": 0.523},
    "Hotel": {"init_orientation": 0.432, "init_speed": 0.218},
}
