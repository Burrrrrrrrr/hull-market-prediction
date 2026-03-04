from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hullpred.core.artifacts import ensure_artifacts_dirs
from hullpred.core.config import load_config
from hullpred.core.paths import resolve_path
from hullpred.evaluation.metrics import compute_fold_metrics


def main() -> None:
    cfg = load_config(None)
    artifacts = ensure_artifacts_dirs(cfg["outputs"]["artifacts_dir"])

    test_path = resolve_path(cfg["data"]["test_path"])
    pred_path = artifacts["preds"] / "nbeats_test_predictions.csv"

    if not pred_path.exists():
        raise FileNotFoundError("nbeats_test_predictions.csv not found. Run scripts/predict/predict_nbeats.py first.")

    test_df = pd.read_csv(test_path)
    pred_df = pd.read_csv(pred_path)

    target_col = cfg["target"]["name"]
    date_col = cfg["target"]["date_col"]

    merged = pd.merge(test_df[[date_col, target_col]], pred_df[[date_col, "nbeats_pred"]], on=date_col, how="inner")
    merged = merged.sort_values(date_col).reset_index(drop=True)

    metrics = compute_fold_metrics(
        y_true=merged[target_col].values,
        pred=merged["nbeats_pred"].values,
        date_id=merged[date_col].values,
        sf=cfg["evaluation"]["sharpe"]["annualization"],
        nw_lag=cfg["evaluation"]["sharpe"]["nw_lag"],
    )

    daily = merged.groupby(date_col, sort=True)[[target_col, "nbeats_pred"]].mean()
    strat = np.sign(daily["nbeats_pred"]) * daily[target_col]
    buy_hold = daily[target_col]

    curve_df = pd.DataFrame(
        {
            date_col: daily.index.values,
            "cum_nbeats": ((1 + strat).cumprod() - 1).values,
            "cum_buy_hold": ((1 + buy_hold).cumprod() - 1).values,
        }
    )

    curve_csv = artifacts["reports"] / "nbeats_test_curve.csv"
    curve_png = artifacts["reports"] / "nbeats_vs_buyhold_test_curve.png"
    metrics_json = artifacts["reports"] / "nbeats_test_metrics.json"

    curve_df.to_csv(curve_csv, index=False)
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plt.figure(figsize=(10, 4))
    plt.plot(curve_df[date_col], curve_df["cum_nbeats"], label="N-BEATS (sign pred)")
    plt.plot(curve_df[date_col], curve_df["cum_buy_hold"], label="Buy & Hold")
    plt.xlabel(date_col)
    plt.ylabel("Cumulative return")
    plt.title("Test Set: N-BEATS vs Buy & Hold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_png, dpi=150)
    plt.close()

    print("Saved:")
    print(f"- {curve_csv}")
    print(f"- {curve_png}")
    print(f"- {metrics_json}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
