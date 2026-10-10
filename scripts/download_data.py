"""
Download the two ETH scenes from a pinned OpenTraj commit and verify their SHA-256.

    python scripts/download_data.py                 # -> data/eth/seq_eth/obsmat.txt, data/eth/seq_hotel/obsmat.txt
    python scripts/download_data.py --data-dir D    # another location
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tfa.data import OPENTRAJ_COMMIT, SCENES, download, sha256  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/eth")
    p.add_argument("--force", action="store_true", help="download even if a verified copy exists")
    args = p.parse_args()
    paths = download(args.data_dir, force=args.force)
    print(f"OpenTraj commit {OPENTRAJ_COMMIT}")
    for scene, path in paths.items():
        print(f"  {scene:5s} {path}  sha256 {sha256(path)}  (expected {SCENES[scene]['sha256'][:12]}...) OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
