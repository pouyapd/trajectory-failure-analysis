"""
data.py — ETH pedestrian dataset: pinned download, checksum and validated loading.

The preprint's notebook cloned OpenTraj (https://github.com/crowdbotp/OpenTraj)
and read datasets/ETH/seq_eth/obsmat.txt and seq_hotel/obsmat.txt. Both files
are unchanged in OpenTraj since commit 8434e5e (2020-05-22); we pin the commit
that was the repository head when the experiments were run (97ec1d5,
2024-04-04) and verify SHA-256 checksums, so every run reads the same bytes.

obsmat.txt columns (Pellegrini et al., ICCV 2009):
    frame, ped_id, pos_x, pos_z, pos_y, v_x, v_z, v_y   (metres, m/s; z is unused)
"""

import hashlib
import urllib.request
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

OPENTRAJ_COMMIT = "97ec1d5e1b579f26febeaa9083705952deac338a"
URL_TEMPLATE = "https://raw.githubusercontent.com/crowdbotp/OpenTraj/{commit}/datasets/ETH/{scene}/obsmat.txt"

SCENES: Dict[str, Dict[str, str]] = {
    "ETH": {"dir": "seq_eth", "sha256": "d452ae2185ecb1164c2fdf31e75f6236f4c2ffc02c751a6b2ae921740cbc60d1"},
    "Hotel": {"dir": "seq_hotel", "sha256": "2b8577595204a6b780b80258fd50da35adb7293e3e06ad6f8e1b60965e7dde6e"},
}
COLUMNS = ["frame", "ped_id", "x", "z", "y", "vx", "vz", "vy"]
FRAME_STEP = {"ETH": 6, "Hotel": 10}          # annotation every 6th / 10th video frame = 2.5 Hz


class DataError(ValueError):
    """Missing, corrupted or malformed dataset file."""


def sha256(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scene_path(data_dir: Union[str, Path], scene: str) -> Path:
    if scene not in SCENES:
        raise DataError(f"unknown scene '{scene}'. Available: {', '.join(SCENES)}")
    return Path(data_dir) / SCENES[scene]["dir"] / "obsmat.txt"


def download(data_dir: Union[str, Path], force: bool = False, timeout: float = 60.0) -> Dict[str, Path]:
    """Download both scenes from the pinned OpenTraj commit and verify their checksums."""
    out = {}
    for scene, info in SCENES.items():
        path = scene_path(data_dir, scene)
        if path.is_file() and not force and sha256(path) == info["sha256"]:
            out[scene] = path
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = URL_TEMPLATE.format(commit=OPENTRAJ_COMMIT, scene=info["dir"])
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as f:
            f.write(resp.read())
        digest = sha256(tmp)
        if digest != info["sha256"]:
            tmp.unlink(missing_ok=True)
            raise DataError(f"checksum mismatch for {scene}: got {digest}, expected {info['sha256']}")
        tmp.replace(path)
        out[scene] = path
    return out


def verify(data_dir: Union[str, Path]) -> Dict[str, bool]:
    """True per scene if the file exists and matches the pinned checksum."""
    return {s: scene_path(data_dir, s).is_file() and sha256(scene_path(data_dir, s)) == i["sha256"]
            for s, i in SCENES.items()}


def load_obsmat(path: Union[str, Path], check_frames: bool = True) -> pd.DataFrame:
    """
    Read one obsmat.txt into a DataFrame with columns frame, ped_id, x, y, vx, vy,
    sorted by (ped_id, frame). Raises DataError for missing files, wrong column
    counts, non-numeric or non-finite values and duplicated (ped_id, frame) rows.
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"dataset file not found: {path}. Run `python scripts/download_data.py` first.")
    try:
        raw = pd.read_csv(path, sep=r"\s+", header=None)
    except Exception as exc:  # pragma: no cover - pandas parser messages vary
        raise DataError(f"could not parse {path}: {exc}") from exc
    if raw.shape[1] != len(COLUMNS):
        raise DataError(f"{path}: expected {len(COLUMNS)} columns, found {raw.shape[1]}")
    raw.columns = COLUMNS
    raw = raw.apply(pd.to_numeric, errors="coerce")
    if raw.isna().any().any() or not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise DataError(f"{path}: non-numeric or non-finite values")
    df = raw[["frame", "ped_id", "x", "y", "vx", "vy"]].sort_values(["ped_id", "frame"]).reset_index(drop=True)
    if df.duplicated(["ped_id", "frame"]).any():
        raise DataError(f"{path}: duplicated (ped_id, frame) rows")
    if check_frames and len(df):
        steps = df.groupby("ped_id")["frame"].diff().dropna()
        if steps.nunique() > 1:
            raise DataError(f"{path}: irregular frame spacing within tracks {sorted(steps.unique())[:5]}")
    return df


def load_scene(data_dir: Union[str, Path], scene: str) -> pd.DataFrame:
    return load_obsmat(scene_path(data_dir, scene))
