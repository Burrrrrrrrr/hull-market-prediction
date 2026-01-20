from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from hullpred.core.artifacts import ensure_artifacts_dirs
from hullpred.core.config import load_config
from hullpred.core.paths import resolve_path
from hullpred.data.loader import load_train_test
from hullpred.evaluation.metrics import compute_fold_metrics
from hullpred.features.engineering import (
    apply_feature_engineering,
    get_feature_cols,
    ic_filter_and_decorrelate,
)
from hullpred.features.regime import make_regime_features_oof
from hullpred.models.lgbm import fit_full_model, train_lgbm_oof
from hullpred.models.optuna_tuner import tune_lgbm_optuna
from hullpred.submission.writer import write_submission


def _prepare_features(df_all: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    target_col = cfg["target"]["name"]
    info_cols = cfg["target"]["info_cols"]
    feat_cfg = cfg["features"]

    return apply_feature_engineering(
        df_all,
        target_col=target_col,
        info_cols=info_cols,
        use_expanding_zscore=feat_cfg["use_expanding_zscore"],
        expanding_min_periods=feat_cfg["expanding_min_periods"],
        lags_by_group=feat_cfg["lags_by_group"],
        rolling_by_group=feat_cfg["rolling_by_group"],
    )


def _select_features(train_df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, list]:
    target_col = cfg["target"]["name"]
    info_cols = cfg["target"]["info_cols"]
    base_features = get_feature_cols(train_df, target_col, info_cols)

    ic_cfg = cfg["features"]["ic_filter"]
    if ic_cfg.get("enabled", False):
        selected = ic_filter_and_decorrelate(
            train_df,
            base_features,
            target_col=target_col,
            min_samples=ic_cfg["min_samples"],
            ic_threshold=ic_cfg["ic_threshold"],
            p_threshold=ic_cfg["p_threshold"],
            corr_threshold=ic_cfg["corr_threshold"],
        )
    else:
        selected = base_features

    return train_df, selected


def train_pipeline(config_path: str | Path | None = None) -> Dict:
    cfg = load_config(config_path)
    artifacts = ensure_artifacts_dirs(cfg["outputs"]["artifacts_dir"])

    train_df_full, test_df_ext = load_train_test(cfg["data"]["train_path"], cfg["data"]["test_path"])

    date_col = cfg["target"]["date_col"]
    test_days = cfg.get("split", {}).get("internal_test_days", 0)
    if test_days and test_days > 0:
        train_df_full = train_df_full.sort_values(date_col).reset_index(drop=True)
        unique_dates = train_df_full[date_col].unique()
        test_dates = unique_dates[-test_days:]
        holdout_mask = train_df_full[date_col].isin(test_dates)
        train_df = train_df_full.loc[~holdout_mask].reset_index(drop=True)
        holdout_df = train_df_full.loc[holdout_mask].reset_index(drop=True)
    else:
        train_df = train_df_full
        holdout_df = None

    df_all = pd.concat([train_df_full, test_df_ext], axis=0).reset_index(drop=True)
    df_all = _prepare_features(df_all, cfg)

    n_train_full = len(train_df_full)
    train_full_fe = df_all.iloc[:n_train_full].reset_index(drop=True)
    test_df = df_all.iloc[n_train_full:].reset_index(drop=True)

    if holdout_df is not None:
        holdout_mask = train_full_fe[date_col].isin(holdout_df[date_col].unique())
        train_df = train_full_fe.loc[~holdout_mask].reset_index(drop=True)
        holdout_df = train_full_fe.loc[holdout_mask].reset_index(drop=True)
    else:
        train_df = train_full_fe

    train_df, selected = _select_features(train_df, cfg)

    if cfg["features"]["regime"]["enabled"]:
        reg_cfg = cfg["features"]["regime"]
        train_df, test_df = make_regime_features_oof(
            train_df=train_df,
            test_df=test_df,
            state_features=reg_cfg["state_features"],
            n_components=reg_cfg["n_components"],
            n_splits=reg_cfg["n_splits"],
            random_state=42,
            min_fit_rows=reg_cfg["min_fit_rows"],
            fill_prob=reg_cfg["fill_prob"],
        )
        prob_cols = [f"regime_prob_{k}" for k in range(reg_cfg["n_components"])]
        selected = [c for c in selected if c in train_df.columns] + prob_cols

    target_col = cfg["target"]["name"]

    df_model = train_df[selected + [target_col, date_col]].dropna().reset_index(drop=True)
    X_train = df_model[selected]
    y_train = df_model[target_col]

    X_test = test_df[selected].copy()

    # drop constant/near-constant columns to reduce LightGBM split warnings
    nunique = X_train.nunique(dropna=True)
    non_constant_cols = nunique[nunique > 1].index.tolist()
    X_train = X_train[non_constant_cols]
    X_test = X_test[non_constant_cols]
    selected = non_constant_cols

    model_cfg = cfg["model"]
    eval_cfg = cfg["evaluation"]["sharpe"]

    # optional Optuna tuning (GPU supported via device_type=gpu in params)
    tuning_cfg = model_cfg.get("tuning", {})
    if tuning_cfg.get("enabled", False):
        best_params, study = tune_lgbm_optuna(
            X_train=X_train,
            y_train=y_train,
            date_id_train=df_model[date_col],
            base_params=model_cfg["lgbm_params"],
            n_trials=tuning_cfg.get("n_trials", 30),
            timeout_seconds=tuning_cfg.get("timeout_seconds", 0),
            direction=tuning_cfg.get("direction", "maximize"),
            n_splits=model_cfg["n_splits"],
            walk_forward=model_cfg.get("walk_forward", {}).get("enabled", False),
            embargo=model_cfg.get("walk_forward", {}).get("embargo", 0),
            val_window=model_cfg.get("walk_forward", {}).get("val_window"),
            step=model_cfg.get("walk_forward", {}).get("step"),
            early_stopping_rounds=model_cfg["early_stopping_rounds"],
            sf=eval_cfg["annualization"],
            nw_lag=eval_cfg["nw_lag"],
        )
        model_cfg["lgbm_params"] = best_params
        (artifacts["reports"] / "optuna_best_params.json").write_text(
            json.dumps(best_params, indent=2), encoding="utf-8"
        )
        (artifacts["reports"] / "optuna_study_best_value.json").write_text(
            json.dumps({"best_value": study.best_value}, indent=2), encoding="utf-8"
        )

    oof_pred, test_pred, fold_metrics, overall, _ = train_lgbm_oof(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        date_id_train=df_model[date_col],
        params=model_cfg["lgbm_params"],
        n_splits=model_cfg["n_splits"],
        early_stopping_rounds=model_cfg["early_stopping_rounds"],
        verbose_eval=model_cfg["verbose_eval"],
        sf=eval_cfg["annualization"],
        nw_lag=eval_cfg["nw_lag"],
        walk_forward=model_cfg.get("walk_forward", {}).get("enabled", False),
        embargo=model_cfg.get("walk_forward", {}).get("embargo", 0),
        val_window=model_cfg.get("walk_forward", {}).get("val_window"),
        step=model_cfg.get("walk_forward", {}).get("step"),
    )

    model_full = fit_full_model(X_train, y_train, model_cfg["lgbm_params"])

    joblib.dump(model_full, artifacts["models"] / cfg["outputs"]["model_name"])

    (artifacts["features"] / "feature_list.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    fold_metrics.to_csv(artifacts["reports"] / "fold_metrics.csv", index=False)
    (artifacts["reports"] / "overall_metrics.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    pd.DataFrame({"oof_pred": oof_pred}).to_csv(artifacts["oof"] / "oof_predictions.csv", index=False)
    pd.DataFrame({"test_pred": test_pred}).to_csv(artifacts["preds"] / "test_predictions.csv", index=False)

    # persist processed datasets for audit/debug
    df_model.to_csv(artifacts["features"] / "train_processed.csv", index=False)
    test_df[selected].to_csv(artifacts["features"] / "test_processed.csv", index=False)

    # holdout evaluation (last N days) if enabled
    if holdout_df is not None:
        if cfg["features"]["regime"]["enabled"]:
            reg_cfg = cfg["features"]["regime"]
            _, holdout_df = make_regime_features_oof(
                train_df=train_df,
                test_df=holdout_df,
                state_features=reg_cfg["state_features"],
                n_components=reg_cfg["n_components"],
                n_splits=reg_cfg["n_splits"],
                random_state=42,
                min_fit_rows=reg_cfg["min_fit_rows"],
                fill_prob=reg_cfg["fill_prob"],
            )

        holdout_model_df = holdout_df[selected + [target_col, date_col]].dropna().reset_index(drop=True)
        holdout_pred = model_full.predict(holdout_model_df[selected])
        holdout_metrics = compute_fold_metrics(
            holdout_model_df[target_col].values,
            holdout_pred,
            date_id=holdout_model_df[date_col].values,
            sf=eval_cfg["annualization"],
            nw_lag=eval_cfg["nw_lag"],
        )
        pd.DataFrame({"holdout_pred": holdout_pred}).to_csv(
            artifacts["preds"] / "holdout_predictions.csv", index=False
        )
        (artifacts["reports"] / "holdout_metrics.json").write_text(
            json.dumps(holdout_metrics, indent=2), encoding="utf-8"
        )
        holdout_model_df.to_csv(artifacts["features"] / "holdout_processed.csv", index=False)

        # holdout curve
        daily = (
            pd.DataFrame(
                {
                    "date_id": holdout_model_df[date_col].values,
                    "y": holdout_model_df[target_col].values,
                    "pred": holdout_pred,
                }
            )
            .groupby("date_id", sort=True)
            .mean()
        )
        strat = np.sign(daily["pred"]) * daily["y"]
        cum_strat = (1 + strat).cumprod() - 1
        cum_bh = (1 + daily["y"]).cumprod() - 1

        curve_df = pd.DataFrame(
            {
                "date_id": daily.index.values,
                "cum_strat": cum_strat.values,
                "cum_buy_hold": cum_bh.values,
            }
        )
        curve_df.to_csv(artifacts["reports"] / "holdout_curve.csv", index=False)

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 4))
            plt.plot(curve_df["date_id"], curve_df["cum_strat"], label="Strategy (sign pred)")
            plt.plot(curve_df["date_id"], curve_df["cum_buy_hold"], label="Buy & Hold")
            plt.xlabel("date_id")
            plt.ylabel("Cumulative return")
            plt.title("Holdout Cumulative Return")
            plt.legend()
            plt.tight_layout()
            plt.savefig(artifacts["reports"] / "holdout_curve.png", dpi=150)
            plt.close()
        except Exception:
            pass

    return {
        "features": selected,
        "overall": overall,
        "fold_metrics": fold_metrics,
        "artifacts": {k: str(v) for k, v in artifacts.items()},
    }


def predict_pipeline(config_path: str | Path | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    artifacts = ensure_artifacts_dirs(cfg["outputs"]["artifacts_dir"])

    train_df, test_df = load_train_test(cfg["data"]["train_path"], cfg["data"]["test_path"])
    df_all = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    df_all = _prepare_features(df_all, cfg)

    n_train = len(train_df)
    train_df = df_all.iloc[:n_train].reset_index(drop=True)
    test_df = df_all.iloc[n_train:].reset_index(drop=True)

    if cfg["features"]["regime"]["enabled"]:
        reg_cfg = cfg["features"]["regime"]
        train_df, test_df = make_regime_features_oof(
            train_df=train_df,
            test_df=test_df,
            state_features=reg_cfg["state_features"],
            n_components=reg_cfg["n_components"],
            n_splits=reg_cfg["n_splits"],
            random_state=42,
            min_fit_rows=reg_cfg["min_fit_rows"],
            fill_prob=reg_cfg["fill_prob"],
        )

    feature_list_path = artifacts["features"] / "feature_list.json"
    selected = json.loads(feature_list_path.read_text(encoding="utf-8"))

    model = joblib.load(artifacts["models"] / cfg["outputs"]["model_name"])

    X_test = test_df[selected].copy()
    preds = model.predict(X_test)
    pred_df = pd.DataFrame({"prediction": preds})
    pred_df.to_csv(artifacts["preds"] / "test_predictions.csv", index=False)
    return pred_df


def submit_pipeline(config_path: str | Path | None = None) -> str:
    cfg = load_config(config_path)
    artifacts = ensure_artifacts_dirs(cfg["outputs"]["artifacts_dir"])

    _, test_df = load_train_test(cfg["data"]["train_path"], cfg["data"]["test_path"])
    pred_path = artifacts["preds"] / "test_predictions.csv"
    preds = pd.read_csv(pred_path)["prediction"].values

    sub_path = artifacts["submission"] / "submission.csv"
    write_submission(preds, test_df, sub_path)
    return str(sub_path)
