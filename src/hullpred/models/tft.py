from __future__ import annotations

import inspect
import json
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd

from hullpred.core.artifacts import ensure_artifacts_dirs
from hullpred.core.config import load_config
from hullpred.core.paths import resolve_path
from hullpred.evaluation.metrics import compute_fold_metrics
from hullpred.models.lgbm import build_walk_forward_splits


def _get_tft_class():
    try:
        from neuralforecast.models import TemporalFusionTransformer

        return TemporalFusionTransformer
    except Exception:
        from neuralforecast.models import TFT

        return TFT


def _supports_param(model_cls, param_name: str) -> bool:
    try:
        sig = inspect.signature(model_cls.__init__)
    except (TypeError, ValueError):
        return False

    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True

    return param_name in sig.parameters


def _build_tft_model(model_cls, model_kwargs: Dict, n_heads: int):
    if _supports_param(model_cls, "n_head"):
        return model_cls(**model_kwargs, n_head=n_heads)
    if _supports_param(model_cls, "n_heads"):
        return model_cls(**model_kwargs, n_heads=n_heads)
    return model_cls(**model_kwargs)


def to_nf_df(df: pd.DataFrame, target_col: str, date_col: str, exog_cols: List[str]) -> pd.DataFrame:
    out = df[[date_col, target_col] + exog_cols].copy()
    out["unique_id"] = 0
    out["ds"] = pd.to_datetime(out[date_col].astype(int), unit="D", origin="2000-01-01")
    out = out.rename(columns={target_col: "y"})
    return out[["unique_id", "ds", "y"] + exog_cols].sort_values("ds").reset_index(drop=True)


def train_and_score_fold(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    target_col: str,
    date_col: str,
    exog_cols: List[str],
    params: Dict,
    input_size: int,
    max_steps: int,
    batch_size: int,
    use_exog: bool,
    device: str,
    embargo: int,
) -> float:
    from neuralforecast import NeuralForecast

    TFTModel = _get_tft_class()

    nf_train = to_nf_df(df_train, target_col, date_col, exog_cols)
    h_valid = len(df_valid)
    h_total = embargo + h_valid
    if h_total <= 0:
        h_total = h_valid

    accelerator = device
    if device == "gpu":
        try:
            import torch

            if not torch.cuda.is_available():
                accelerator = "cpu"
        except Exception:
            accelerator = "cpu"

    trainer_kwargs = {
        "accelerator": accelerator,
        "devices": 1,
        "enable_checkpointing": False,
        "logger": False,
        "enable_progress_bar": True,
        "log_every_n_steps": 1,
    }

    model_kwargs = {
        "h": h_total,
        "input_size": input_size,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": params["learning_rate"],
        "hidden_size": params["hidden_size"],
        "dropout": params["dropout"],
        "start_padding_enabled": True,
        "hist_exog_list": exog_cols if use_exog else None,
        "random_seed": 42,
        **trainer_kwargs,
    }
    model = _build_tft_model(TFTModel, model_kwargs, params["n_heads"])

    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=nf_train, verbose=True)

    pred_df = nf.predict()
    pred_col = [c for c in pred_df.columns if c not in ["unique_id", "ds"]][0]
    y_pred_all = pred_df[pred_col].values.astype(float)
    y_pred = y_pred_all[embargo : embargo + h_valid]

    valid_ds = pd.to_datetime(df_valid[date_col].astype(int), unit="D", origin="2000-01-01").values
    pred_ds = pred_df["ds"].values
    pred_ds_aligned = pred_ds[embargo : embargo + h_valid]
    if len(pred_ds_aligned) != len(valid_ds) or not np.array_equal(pred_ds_aligned, valid_ds):
        raise ValueError("TFT prediction dates do not align with validation dates. Check embargo/frequency.")

    y_true = df_valid[target_col].values
    date_vals = df_valid[date_col].values

    metrics = compute_fold_metrics(y_true, y_pred, date_id=date_vals, sf=252.0, nw_lag=5)
    return float(metrics["adj_sharpe"])


def tune_tft(
    config_path: str | None = None,
) -> Tuple[Dict, float]:
    cfg = load_config(config_path)
    artifacts = ensure_artifacts_dirs(cfg["outputs"]["artifacts_dir"])

    tft_cfg = cfg.get("tft", {})
    train_cfg = tft_cfg.get("training", {})
    tuning_cfg = tft_cfg.get("tuning", {})
    search_cfg = tft_cfg.get("search_space", {})

    train_path = artifacts["features"] / "train_processed.csv"
    if not train_path.exists():
        raise FileNotFoundError("train_processed.csv not found. Run training pipeline first.")

    df = pd.read_csv(train_path)
    target_col = cfg["target"]["name"]
    date_col = cfg["target"]["date_col"]

    exog_cols = [c for c in df.columns if c not in [target_col, date_col]]
    df = df.sort_values(date_col).reset_index(drop=True)

    val_window = train_cfg.get("val_window", 63)
    step = train_cfg.get("step", 21)
    embargo = train_cfg.get("embargo", 20)
    n_splits = cfg["model"]["n_splits"]

    splits = build_walk_forward_splits(
        len(df),
        n_splits=n_splits,
        embargo=embargo,
        val_window=val_window,
        step=step,
    )
    max_folds = tuning_cfg.get("max_folds", 5)
    splits = splits[:max_folds]

    device = "gpu" if cfg["model"]["lgbm_params"].get("device_type", "cpu") == "gpu" else "cpu"
    use_exog = bool(tft_cfg.get("use_exog", True))
    if not use_exog:
        exog_cols = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate", search_cfg["learning_rate"][0], search_cfg["learning_rate"][1], log=True
            ),
            "n_heads": trial.suggest_int("n_heads", search_cfg["n_heads"][0], search_cfg["n_heads"][1]),
            "dropout": trial.suggest_float("dropout", search_cfg["dropout"][0], search_cfg["dropout"][1]),
        }

        h_min, h_max = search_cfg["hidden_size"][0], search_cfg["hidden_size"][1]
        heads = params["n_heads"]
        low = int(np.ceil(h_min / heads))
        high = int(np.floor(h_max / heads))
        if low > high:
            return -np.inf
        multiplier = trial.suggest_int("hidden_mult", low, high)
        params["hidden_size"] = int(multiplier * heads)

        print(
            f"[Trial {trial.number}] lr={params['learning_rate']:.5f} | hidden={params['hidden_size']} | heads={params['n_heads']} | dropout={params['dropout']:.3f}"
        )

        scores = []
        for fold, (tr_idx, va_idx) in enumerate(splits, 1):
            df_train = df.iloc[tr_idx].copy()
            df_valid = df.iloc[va_idx].copy()
            print(
                f"  Fold {fold}/{len(splits)} start | train_rows={len(df_train)} | valid_rows={len(df_valid)}"
            )
            try:
                score = train_and_score_fold(
                    df_train,
                    df_valid,
                    target_col=target_col,
                    date_col=date_col,
                    exog_cols=exog_cols,
                    params=params,
                    input_size=train_cfg.get("input_size", 256),
                    max_steps=train_cfg.get("max_steps", 1000),
                    batch_size=train_cfg.get("batch_size", 32),
                    use_exog=use_exog,
                    device=device,
                    embargo=embargo,
                )
            except Exception as e:
                msg = str(e).lower()
                if "out of memory" in msg:
                    print(f"  Fold {fold} OOM -> trial pruned as -inf")
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    return -np.inf
                raise
            scores.append(score)
            print(f"  Fold {fold}/{len(splits)} adj_sharpe={score:.4f}")

        avg_score = float(np.nanmean(scores))
        print(f"[Trial {trial.number}] mean_adj_sharpe={avg_score:.4f}")
        return avg_score

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=tuning_cfg.get("n_trials", 20),
        timeout=tuning_cfg.get("timeout_seconds", 0) or None,
    )

    best_params = study.best_params
    (artifacts["reports"] / "tft_best_params.json").write_text(
        json.dumps(best_params, indent=2), encoding="utf-8"
    )
    (artifacts["reports"] / "tft_best_value.json").write_text(
        json.dumps({"best_value": study.best_value}, indent=2), encoding="utf-8"
    )

    return best_params, float(study.best_value)


def predict_tft(
    config_path: str | None = None,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    artifacts = ensure_artifacts_dirs(cfg["outputs"]["artifacts_dir"])

    tft_cfg = cfg.get("tft", {})
    train_cfg = tft_cfg.get("training", {})

    train_path = artifacts["features"] / "train_processed.csv"
    test_path = artifacts["features"] / "test_processed.csv"
    if not train_path.exists():
        raise FileNotFoundError("train_processed.csv not found. Run training pipeline first.")
    if not test_path.exists():
        raise FileNotFoundError("test_processed.csv not found. Run training pipeline first.")

    params_path = artifacts["reports"] / "tft_best_params.json"
    if not params_path.exists():
        raise FileNotFoundError("tft_best_params.json not found. Run TFT tuning first.")

    best_params = json.loads(params_path.read_text(encoding="utf-8-sig"))
    if "hidden_size" not in best_params:
        best_params["hidden_size"] = int(best_params["hidden_mult"] * best_params["n_heads"])

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    target_col = cfg["target"]["name"]
    date_col = cfg["target"]["date_col"]

    use_exog = bool(tft_cfg.get("use_exog", True))
    exog_cols = [c for c in df_train.columns if c not in [target_col, date_col]]
    if not use_exog:
        exog_cols = []

    df_train = df_train.sort_values(date_col).reset_index(drop=True)
    if date_col in df_test.columns:
        df_test = df_test.sort_values(date_col).reset_index(drop=True)

    device = "gpu" if cfg["model"]["lgbm_params"].get("device_type", "cpu") == "gpu" else "cpu"
    accelerator = device
    if device == "gpu":
        try:
            import torch

            if not torch.cuda.is_available():
                accelerator = "cpu"
        except Exception:
            accelerator = "cpu"

    from neuralforecast import NeuralForecast

    TFTModel = _get_tft_class()

    nf_train = to_nf_df(df_train, target_col, date_col, exog_cols)
    h = len(df_test)

    max_steps = train_cfg.get("max_steps_predict", train_cfg.get("max_steps", 1000))

    def _fit_predict_with_accel(accel: str) -> pd.DataFrame:
        trainer_kwargs = {
            "accelerator": accel,
            "devices": 1,
            "enable_checkpointing": False,
            "logger": False,
        }
        model_kwargs = {
            "h": h,
            "input_size": train_cfg.get("input_size", 256),
            "max_steps": max_steps,
            "batch_size": train_cfg.get("batch_size", 32),
            "learning_rate": best_params["learning_rate"],
            "hidden_size": best_params["hidden_size"],
            "dropout": best_params["dropout"],
            "start_padding_enabled": True,
            "hist_exog_list": exog_cols if use_exog else None,
            "random_seed": 42,
            **trainer_kwargs,
        }
        model = _build_tft_model(TFTModel, model_kwargs, best_params["n_heads"])
        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(df=nf_train, verbose=False)
        return nf.predict()

    try:
        pred_df = _fit_predict_with_accel(accelerator)
    except Exception as e:
        if accelerator == "gpu" and "out of memory" in str(e).lower():
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            print("GPU OOM during predict_tft, retrying on CPU...")
            pred_df = _fit_predict_with_accel("cpu")
        else:
            raise
    pred_col = [c for c in pred_df.columns if c not in ["unique_id", "ds"]][0]
    y_pred = pred_df[pred_col].values.astype(float)

    if date_col in df_test.columns:
        date_vals = df_test[date_col].values
    else:
        raw_test_path = resolve_path(cfg["data"]["test_path"])
        raw_test = pd.read_csv(raw_test_path)
        date_vals = raw_test[date_col].values

    if len(y_pred) != len(date_vals):
        min_len = min(len(y_pred), len(date_vals))
        y_pred = y_pred[:min_len]
        date_vals = date_vals[:min_len]

    out = pd.DataFrame(
        {
            date_col: date_vals,
            "tft_pred": y_pred,
        }
    )

    out_path = artifacts["preds"] / "tft_test_predictions.csv"
    out.to_csv(out_path, index=False)
    return out
