from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hullpred.models.nbeats import predict_nbeats


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with N-BEATS on test data")
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml")
    args = parser.parse_args()

    out = predict_nbeats(args.config)
    print("N-BEATS test prediction complete.")
    print("Rows:", len(out))


if __name__ == "__main__":
    main()
