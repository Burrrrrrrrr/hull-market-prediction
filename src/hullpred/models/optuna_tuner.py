from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import TimeSeriesSplit

from hullpred.evaluation.metrics import compute_fold_metrics
from hullpred.models.lgbm import build_walk_forward_splits


def _run_oof_cv(
    X: pd.DataFrame,
    y: pd.Series,
    date_id: pd.Series | None,
    params: Dict,
    n_splits: int,
    walk_forward: bool,
    embargo: int,
    val_window: int | None,
    step: int | None,
    early_stopping_rounds: int,
    sf: float,
    nw_lag: int,
) -> float:
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    date_id = None if date_id is None else pd.Series(date_id).reset_index(drop=True)

    if walk_forward:
        splits = build_walk_forward_splits(
            len(X),
            n_splits=n_splits,
            embargo=embargo,
            val_window=val_window,
            step=step,
        )
    else:
        splits = list(TimeSeriesSplit(n_splits=n_splits).split(X))

    oof_pred = np.full(len(X), np.nan, dtype=float)

    for tr_idx, va_idx in splits:
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="l2",
            callbacks=[early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
        )

        pred_va = model.predict(X_va, num_iteration=model.best_iteration_)
        oof_pred[va_idx] = pred_va

    valid_mask = ~np.isnan(oof_pred)
    if not np.any(valid_mask):
        return -np.inf

    did = None if date_id is None else date_id.values[valid_mask]
    metrics = compute_fold_metrics(y.values[valid_mask], oof_pred[valid_mask], date_id=did, sf=sf, nw_lag=nw_lag)
    return float(metrics["adj_sharpe"])


def tune_lgbm_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    date_id_train: pd.Series | None,
    base_params: Dict,
    n_trials: int,
    timeout_seconds: int,
    direction: str,
    n_splits: int,
    walk_forward: bool,
    embargo: int,
    val_window: int | None,
    step: int | None,
    early_stopping_rounds: int,
    sf: float,
    nw_lag: int,
) -> Tuple[Dict, optuna.study.Study]:
    def objective(trial: optuna.Trial) -> float:
        params = dict(base_params)
        params.update(
            {
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
                "max_depth": trial.suggest_int("max_depth", -1, 16),
            }
        )

        score = _run_oof_cv(
            X_train,
            y_train,
            date_id_train,
            params,
            n_splits=n_splits,
            walk_forward=walk_forward,
            embargo=embargo,
            val_window=val_window,
            step=step,
            early_stopping_rounds=early_stopping_rounds,
            sf=sf,
            nw_lag=nw_lag,
        )
        return score

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds if timeout_seconds > 0 else None)

    best_params = dict(base_params)
    best_params.update(study.best_params)
    return best_params, study
