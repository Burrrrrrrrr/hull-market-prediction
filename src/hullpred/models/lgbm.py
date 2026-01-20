from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import TimeSeriesSplit

from hullpred.evaluation.metrics import compute_fold_metrics


def build_walk_forward_splits(
    n_samples: int,
    n_splits: int,
    embargo: int,
    val_window: int | None = None,
    step: int | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    if n_samples <= 2:
        raise ValueError("n_samples too small")

    if val_window is None or val_window <= 0:
        val_window = max(1, n_samples // (n_splits + 1))
    if step is None or step <= 0:
        step = val_window

    splits = []
    start_train_end = val_window
    for i in range(n_splits):
        train_end = start_train_end + i * step
        if train_end >= n_samples:
            break
        valid_start = train_end + max(0, embargo)
        valid_end = valid_start + val_window
        if valid_start >= n_samples:
            break
        if valid_end > n_samples:
            valid_end = n_samples
        train_idx = np.arange(0, train_end)
        valid_idx = np.arange(valid_start, valid_end)
        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue
        splits.append((train_idx, valid_idx))

    if not splits:
        raise ValueError("No valid walk-forward splits; reduce n_splits or embargo")
    return splits


def train_lgbm_oof(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    date_id_train: pd.Series | None,
    params: Dict,
    n_splits: int,
    early_stopping_rounds: int,
    verbose_eval: int,
    sf: float,
    nw_lag: int,
    walk_forward: bool = False,
    embargo: int = 0,
    val_window: int | None = None,
    step: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict, List[LGBMRegressor]]:
    Xtr = X_train.reset_index(drop=True)
    ytr = pd.Series(y_train).reset_index(drop=True)
    Xte = X_test.reset_index(drop=True)
    date_id_train = None if date_id_train is None else pd.Series(date_id_train).reset_index(drop=True)

    if walk_forward:
        splits = build_walk_forward_splits(
            len(Xtr),
            n_splits=n_splits,
            embargo=embargo,
            val_window=val_window,
            step=step,
        )
    else:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(Xtr))
    oof_pred = np.full(len(Xtr), np.nan, dtype=float)
    test_pred_folds = []
    fold_metrics = []
    models = []

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        X_tr, X_va = Xtr.iloc[tr_idx], Xtr.iloc[va_idx]
        y_tr, y_va = ytr.iloc[tr_idx], ytr.iloc[va_idx]

        model = LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="l2",
            callbacks=[
                early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                log_evaluation(period=verbose_eval),
            ],
        )

        pred_va = model.predict(X_va, num_iteration=model.best_iteration_)
        oof_pred[va_idx] = pred_va

        did_va = None if date_id_train is None else date_id_train.iloc[va_idx]
        m = compute_fold_metrics(y_va.values, pred_va, date_id=did_va, sf=sf, nw_lag=nw_lag)
        m["fold"] = fold
        m["best_iter"] = getattr(model, "best_iteration_", None)
        fold_metrics.append(m)

        pred_te = model.predict(Xte, num_iteration=model.best_iteration_)
        test_pred_folds.append(pred_te)
        models.append(model)

    test_pred = np.mean(np.vstack(test_pred_folds), axis=0)

    valid_mask = ~np.isnan(oof_pred)
    overall = compute_fold_metrics(
        ytr.values[valid_mask],
        oof_pred[valid_mask],
        date_id=None if date_id_train is None else date_id_train.values[valid_mask],
        sf=sf,
        nw_lag=nw_lag,
    )

    return oof_pred, test_pred, pd.DataFrame(fold_metrics), overall, models


def fit_full_model(X_train: pd.DataFrame, y_train: pd.Series, params: Dict) -> LGBMRegressor:
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model
