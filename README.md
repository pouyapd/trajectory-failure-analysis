# Trajectory Failure Analysis — ETH Pedestrian Dataset

**Failure Mode Analysis of Pedestrian Trajectory Prediction: A Model-Agnostic Interpretable Risk Assessment Framework**

*Pouya Bathaei Pourmand — University of Genoa, Italy*

---

## Overview

This repository contains the code and figures for an independent research project investigating **when and why trajectory prediction models fail** in pedestrian navigation scenarios.

We apply a model-agnostic evaluation framework to a Social Force Model (constant velocity baseline) evaluated on the **ETH pedestrian dataset**, across two real-world scenes (ETH campus and Hotel). The framework combines:

- **Input-space sensitivity analysis** — risk maps in orientation–speed and spatial space
- **Interpretable decision tree classifiers** — human-readable failure rules
- **Cross-scene evaluation** — comparing failure patterns across environments

---

## Key Results

| Metric | ETH Scene | Hotel Scene |
|---|---|---|
| Trajectories | 353 | 366 |
| Mean ADE | 0.326 m | 0.238 m |
| Failure threshold | 0.41 m | 0.33 m |
| Failure rate | 24.9% | 25.1% |
| Decision tree accuracy | 79.3% | 82.0% |

**Key finding:** Initial orientation is consistently important across both scenes. Positional features are scene-dependent — suggesting that universal failure rules may not generalise across environments.

---

## Repository Contents

```
trajectory-failure-analysis/
├── SafeTraj_ETH.ipynb          # Main notebook — reproduces all experiments
├── paper.pdf                   # Preprint
├── comparison_scenes.png       # Figure 1: Cross-scene risk maps and ADE distributions
├── decision_tree_sfm.png       # Figure 2: Decision tree failure rules (ETH)
└── analysis_sfm.png            # Figure 3: ADE distribution and feature importances
```

---

## How to Run

Open `SafeTraj_ETH.ipynb` in **Google Colab** and run all cells in order. No additional setup required — the dataset is downloaded automatically.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pouyapd/trajectory-failure-analysis/blob/main/SafeTraj_ETH.ipynb)

---

## Dataset

[ETH Pedestrian Dataset](https://icu.ee.ethz.ch/research/datsets.html) — Pellegrini et al., ICCV 2009.  
Downloaded via [OpenTraj](https://github.com/crowdbotp/OpenTraj).

---

## Related Work

This project extends the failure analysis framework developed in my MSc thesis:  
*Analysis of Neural Trajectory Prediction for Collision Avoidance in Smart Wheelchairs*  
University of Genoa & CNR, EU Horizon REXASI-PRO project.

---

## Contact

**Pouya Bathaei Pourmand**  
pouyapd68@gmail.com | [github.com/pouyapd](https://github.com/pouyapd)
