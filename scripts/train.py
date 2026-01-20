from __future__ import annotations

import argparse

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hullpred.pipeline import train_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train Hull LGBM pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml")
    args = parser.parse_args()

    result = train_pipeline(args.config)
    print("Training complete.")
    print("Overall metrics:", result["overall"])


if __name__ == "__main__":
    main()
