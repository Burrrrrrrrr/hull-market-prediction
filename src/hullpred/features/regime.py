from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import TimeSeriesSplit


def _ensure_state_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def roll_mean(col: str, w: int, mp: int | None = None) -> pd.Series:
        mp = max(2, w // 3) if mp is None else mp
        return out[col].shift(1).rolling(w, min_periods=mp).mean()

    def roll_std(col: str, w: int, mp: int | None = None) -> pd.Series:
        mp = max(2, w // 3) if mp is None else mp
        return out[col].shift(1).rolling(w, min_periods=mp).std()

    for col in ["V10", "V13", "E11", "E19"]:
        if col not in out.columns:
            return out

    if "V10_roll_std_21" not in out.columns:
        out["V10_roll_std_21"] = roll_std("V10", 21, mp=7)
    if "V10_roll_std_63" not in out.columns:
        out["V10_roll_std_63"] = roll_std("V10", 63, mp=21)
    if "V13_roll_mean_63" not in out.columns:
        out["V13_roll_mean_63"] = roll_mean("V13", 63, mp=21)

    def ensure_z(col: str) -> None:
        mean_col = f"{col}_roll_mean_63"
        std_col = f"{col}_roll_std_63"
        if mean_col not in out.columns:
            out[mean_col] = roll_mean(col, 63, mp=21)
        if std_col not in out.columns:
            out[std_col] = roll_std(col, 63, mp=21)
        z_col = f"{col}_z"
        if z_col not in out.columns:
            out[z_col] = (out[col] - out[mean_col]) / (out[std_col] + 1e-12)

    for col in ["V10", "V13", "E11", "E19"]:
        ensure_z(col)

    return out


def make_regime_features_oof(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    state_features: List[str],
    n_components: int = 3,
    n_splits: int = 5,
    random_state: int = 42,
    min_fit_rows: int = 200,
    fill_prob: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = _ensure_state_features(train_df).sort_values("date_id").reset_index(drop=True)
    test_df = _ensure_state_features(test_df).sort_values("date_id").reset_index(drop=True)

    for c in state_features:
        if c not in train_df.columns or c not in test_df.columns:
            raise ValueError(f"Missing state feature: {c}")

    oof_regime = np.full(len(train_df), np.nan)
    oof_probs = np.full((len(train_df), n_components), np.nan)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(train_df), 1):
        X_tr = train_df.loc[tr_idx, state_features]
        X_va = train_df.loc[va_idx, state_features]

        tr_valid = X_tr.dropna().index
        if len(tr_valid) < min_fit_rows:
            continue

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state + fold,
        )
        gmm.fit(X_tr.loc[tr_valid].values)

        va_valid = X_va.dropna().index
        if len(va_valid) == 0:
            continue

        probs = gmm.predict_proba(X_va.loc[va_valid].values)
        labels = probs.argmax(axis=1)
        oof_probs[va_valid, :] = probs
        oof_regime[va_valid] = labels

    train_out = train_df.copy()
    train_out["regime"] = oof_regime
    for k in range(n_components):
        train_out[f"regime_prob_{k}"] = oof_probs[:, k]

    X_full = train_df[state_features]
    full_valid = X_full.dropna().index
    gmm_full = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
    )
    if len(full_valid) >= min_fit_rows:
        gmm_full.fit(X_full.loc[full_valid].values)

    X_test = test_df[state_features]
    test_valid = X_test.dropna().index
    test_probs = np.full((len(test_df), n_components), np.nan)
    test_regime = np.full(len(test_df), np.nan)

    if len(test_valid) > 0 and len(full_valid) >= min_fit_rows:
        probs = gmm_full.predict_proba(X_test.loc[test_valid].values)
        labels = probs.argmax(axis=1)
        test_probs[test_valid, :] = probs
        test_regime[test_valid] = labels

    test_out = test_df.copy()
    test_out["regime"] = test_regime
    for k in range(n_components):
        test_out[f"regime_prob_{k}"] = test_probs[:, k]

    if fill_prob:
        prob_cols = [f"regime_prob_{k}" for k in range(n_components)]
        train_out[prob_cols] = train_out[prob_cols].fillna(1.0 / n_components)
        test_out[prob_cols] = test_out[prob_cols].fillna(1.0 / n_components)

    return train_out, test_out
