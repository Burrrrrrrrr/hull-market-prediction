from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hullpred.models.patchtst import tune_patchtst


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna tuning for PatchTST")
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml")
    args = parser.parse_args()

    best_params, best_value = tune_patchtst(args.config)
    print("Best params:", best_params)
    print("Best value:", best_value)


if __name__ == "__main__":
    main()
