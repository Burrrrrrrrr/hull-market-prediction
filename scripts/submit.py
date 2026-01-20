from __future__ import annotations

import argparse

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hullpred.pipeline import submit_pipeline


def main():
    parser = argparse.ArgumentParser(description="Create submission file")
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml")
    args = parser.parse_args()

    path = submit_pipeline(args.config)
    print("Submission saved to:", path)


if __name__ == "__main__":
    main()
