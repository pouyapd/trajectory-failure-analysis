"""
tfa — trajectory failure analysis (reusable code for the preprint
"Understanding Failure Modes in Pedestrian Trajectory Prediction via
Interpretable Risk Analysis").

    tfa.data       pinned ETH download, checksums, validated loading
    tfa.pipeline   prediction, ADE/FDE, initial-state features, failure labels
    tfa.metrics    ADE / FDE with input validation
    tfa.analysis   decision trees, held-out evaluation, bootstrap, risk maps, rules
    tfa.plots      figures
    tfa.config     ExperimentConfig (defaults = the preprint's settings)
"""

from .config import PAPER, ExperimentConfig

__version__ = "1.0.0"
__all__ = ["PAPER", "ExperimentConfig"]
