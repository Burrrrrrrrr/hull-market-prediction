from __future__ import annotations

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


def to_nf_df(df: pd.DataFrame, target_col: str, date_col: str) -> pd.DataFrame:
    out = df[[date_col, target_col]].copy()
    out["unique_id"] = 0
    out["ds"] = pd.to_datetime(out[date_col].astype(int), unit="D", origin="2000-01-01")
    out = out.rename(columns={target_col: "y"})
    return out[["unique_id", "ds", "y"]].sort_values("ds").reset_index(drop=True)


def _make_nbeats_model(
    h: int,
    input_size: int,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    stack_types: list[str],
    n_blocks: list[int],
    mlp_units: list[list[int]],
    n_polynomials: int,
    n_harmonics: int,
    dropout_prob_theta: float = 0.0,
):
    from neuralforecast.models import NBEATS

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

    return NBEATS(
        h=h,
        input_size=input_size,
        stack_types=stack_types,
        n_blocks=n_blocks,
        mlp_units=mlp_units,
        n_polynomials=n_polynomials,
        n_harmonics=n_harmonics,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dropout_prob_theta=dropout_prob_theta,
        start_padding_enabled=True,
        random_seed=42,
        **trainer_kwargs,
    )


def train_and_score_fold(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    target_col: str,
    date_col: str,
    params: Dict,
    device: str,
    embargo: int,
) -> float:
    from neuralforecast import NeuralForecast

    nf_train = to_nf_df(df_train, target_col, date_col)
    h_valid = len(df_valid)
    h_total = embargo + h_valid
    if h_total <= 0:
        h_total = h_valid

    stack_types = params["stack_types"].split("-")
    n_blocks = [params["n_blocks"] for _ in stack_types]
    mlp_units = [[params["mlp_units"] for _ in range(params["mlp_layers"])] for _ in stack_types]

    model = _make_nbeats_model(
        h=h_total,
        input_size=params["input_size"],
        max_steps=params["max_steps"],
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        stack_types=stack_types,
        n_blocks=n_blocks,
        mlp_units=mlp_units,
        n_polynomials=params["n_polynomials"],
        n_harmonics=params["n_harmonics"],
        dropout_prob_theta=params["dropout_prob_theta"],
        device=device,
    )

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
        raise ValueError("N-BEATS prediction dates do not align with validation dates. Check embargo/frequency.")

    y_true = df_valid[target_col].values
    date_vals = df_valid[date_col].values

    metrics = compute_fold_metrics(y_true, y_pred, date_id=date_vals, sf=252.0, nw_lag=5)
    return float(metrics["adj_sharpe"])


def tune_nbeats(config_path: str | None = None) -> Tuple[Dict, float]:
    cfg = load_config(config_path)
    artifacts = ensure_artifacts_dirs(cfg["outputs"]["artifacts_dir"])

    nbeats_cfg = cfg.get("nbeats", {})
    train_cfg = nbeats_cfg.get("training", {})
    tuning_cfg = nbeats_cfg.get("tuning", {})
    search_cfg = nbeats_cfg.get("search_space", {})

    train_path = artifacts["features"] / "train_processed.csv"
    if not train_path.exists():
        raise FileNotFoundError("train_processed.csv not found. Run training pipeline first.")

    df = pd.read_csv(train_path)
    target_col = cfg["target"]["name"]
    date_col = cfg["target"]["date_col"]

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

    def objective(trial: optuna.Trial) -> float:
        lr_rng = search_cfg.get("learning_rate", [0.0005, 0.01])
        input_rng = search_cfg.get("input_size", [32, 96])
        batch_rng = search_cfg.get("batch_size", [4, 16])
        step_rng = search_cfg.get("max_steps", [200, 500])
        mlp_rng = search_cfg.get("mlp_units", [64, 256])
        depth_rng = search_cfg.get("mlp_layers", [1, 3])
        block_rng = search_cfg.get("n_blocks", [1, 3])
        poly_rng = search_cfg.get("n_polynomials", [1, 4])
        harm_rng = search_cfg.get("n_harmonics", [1, 4])
        stacks = search_cfg.get(
            "stack_types",
            [
                "identity-trend-seasonality",
                "identity-trend",
                "identity-seasonality",
            ],
        )

        params = {
            "learning_rate": trial.suggest_float("learning_rate", lr_rng[0], lr_rng[1], log=True),
            "input_size": trial.suggest_int("input_size", input_rng[0], input_rng[1]),
            "batch_size": trial.suggest_int("batch_size", batch_rng[0], batch_rng[1]),
            "max_steps": trial.suggest_int("max_steps", step_rng[0], step_rng[1]),
            "mlp_units": trial.suggest_int("mlp_units", mlp_rng[0], mlp_rng[1], step=32),
            "mlp_layers": trial.suggest_int("mlp_layers", depth_rng[0], depth_rng[1]),
            "n_blocks": trial.suggest_int("n_blocks", block_rng[0], block_rng[1]),
            "stack_types": trial.suggest_categorical("stack_types", stacks),
            "n_polynomials": trial.suggest_int("n_polynomials", poly_rng[0], poly_rng[1]),
            "n_harmonics": trial.suggest_int("n_harmonics", harm_rng[0], harm_rng[1]),
            "dropout_prob_theta": 0.0,
        }

        print(
            f"[Trial {trial.number}] lr={params['learning_rate']:.5f} | input={params['input_size']} | batch={params['batch_size']} | max_steps={params['max_steps']} | mlp={params['mlp_units']}x{params['mlp_layers']} | blocks={params['n_blocks']} | stacks={params['stack_types']} | poly={params['n_polynomials']} | harm={params['n_harmonics']}"
        )

        scores = []
        for fold, (tr_idx, va_idx) in enumerate(splits, 1):
            df_train = df.iloc[tr_idx].copy()
            df_valid = df.iloc[va_idx].copy()
            print(f"  Fold {fold}/{len(splits)} start | train_rows={len(df_train)} | valid_rows={len(df_valid)}")
            try:
                score = train_and_score_fold(
                    df_train,
                    df_valid,
                    target_col=target_col,
                    date_col=date_col,
                    params=params,
                    device=device,
                    embargo=embargo,
                )
            except Exception as e:
                emsg = str(e).lower()
                if "out of memory" in emsg:
                    print(f"  Fold {fold} OOM -> trial pruned as -inf")
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    return -np.inf
                if isinstance(e, IndexError) or "list index out of range" in emsg:
                    print(f"  Fold {fold} invalid stack configuration -> trial pruned as -inf")
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
    (artifacts["reports"] / "nbeats_best_params.json").write_text(
        json.dumps(best_params, indent=2), encoding="utf-8"
    )
    (artifacts["reports"] / "nbeats_best_value.json").write_text(
        json.dumps({"best_value": study.best_value}, indent=2), encoding="utf-8"
    )

    return best_params, float(study.best_value)


def predict_nbeats(config_path: str | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    artifacts = ensure_artifacts_dirs(cfg["outputs"]["artifacts_dir"])

    nbeats_cfg = cfg.get("nbeats", {})
    train_cfg = nbeats_cfg.get("training", {})

    train_path = artifacts["features"] / "train_processed.csv"
    test_path = artifacts["features"] / "test_processed.csv"
    if not train_path.exists():
        raise FileNotFoundError("train_processed.csv not found. Run training pipeline first.")
    if not test_path.exists():
        raise FileNotFoundError("test_processed.csv not found. Run training pipeline first.")

    params_path = artifacts["reports"] / "nbeats_best_params.json"
    if not params_path.exists():
        raise FileNotFoundError("nbeats_best_params.json not found. Run N-BEATS tuning first.")

    best_params = json.loads(params_path.read_text(encoding="utf-8-sig"))

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    target_col = cfg["target"]["name"]
    date_col = cfg["target"]["date_col"]

    df_train = df_train.sort_values(date_col).reset_index(drop=True)
    if date_col in df_test.columns:
        df_test = df_test.sort_values(date_col).reset_index(drop=True)

    device = "gpu" if cfg["model"]["lgbm_params"].get("device_type", "cpu") == "gpu" else "cpu"
    nf_train = to_nf_df(df_train, target_col, date_col)
    h = len(df_test)

    from neuralforecast import NeuralForecast

    def _fit_predict_with_device(dev: str) -> pd.DataFrame:
        stack_types_raw = best_params.get("stack_types", "identity-trend-seasonality")
        stack_types = stack_types_raw.split("-") if isinstance(stack_types_raw, str) else stack_types_raw
        n_blocks = [int(best_params.get("n_blocks", 1)) for _ in stack_types]
        mlp_width = int(best_params.get("mlp_units", 128))
        mlp_layers = int(best_params.get("mlp_layers", 2))
        mlp_units = [[mlp_width for _ in range(mlp_layers)] for _ in stack_types]

        model = _make_nbeats_model(
            h=h,
            input_size=best_params.get("input_size", train_cfg.get("input_size", 64)),
            max_steps=best_params.get("max_steps", train_cfg.get("max_steps_predict", train_cfg.get("max_steps", 400))),
            batch_size=best_params.get("batch_size", train_cfg.get("batch_size", 16)),
            learning_rate=best_params.get("learning_rate", 0.001),
            stack_types=stack_types,
            n_blocks=n_blocks,
            mlp_units=mlp_units,
            n_polynomials=int(best_params.get("n_polynomials", 2)),
            n_harmonics=int(best_params.get("n_harmonics", 2)),
            dropout_prob_theta=best_params.get("dropout_prob_theta", 0.0),
            device=dev,
        )
        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(df=nf_train, verbose=False)
        return nf.predict()

    try:
        pred_df = _fit_predict_with_device(device)
    except Exception as e:
        if device == "gpu" and "out of memory" in str(e).lower():
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            print("GPU OOM during predict_nbeats, retrying on CPU...")
            pred_df = _fit_predict_with_device("cpu")
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

    out = pd.DataFrame({date_col: date_vals, "nbeats_pred": y_pred})
    out_path = artifacts["preds"] / "nbeats_test_predictions.csv"
    out.to_csv(out_path, index=False)
    return out
