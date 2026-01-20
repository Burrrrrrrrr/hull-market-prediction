from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def get_feature_cols(df: pd.DataFrame, target_col: str, info_cols: Iterable[str]) -> List[str]:
    info_set = set(info_cols) | {target_col}
    return [c for c in df.columns if c not in info_set]


def group_by_prefix(feature_cols: Iterable[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for col in feature_cols:
        prefix = col[0] if col else "OTHER"
        groups.setdefault(prefix, []).append(col)
    return groups


def expanding_zscore(df: pd.DataFrame, feature_cols: Iterable[str], min_periods: int = 30) -> pd.DataFrame:
    X = df[list(feature_cols)]
    exp_mean = X.expanding(min_periods=min_periods).mean().shift(1)
    exp_std = X.expanding(min_periods=min_periods).std(ddof=0).shift(1)
    z = (X - exp_mean) / exp_std
    z.columns = [f"{c}_z" for c in X.columns]
    return z


def add_group_lags(df: pd.DataFrame, feature_cols: Iterable[str], lags_by_group: Dict[str, List[int]]) -> pd.DataFrame:
    out = df.copy()
    for feat in feature_cols:
        g = feat[0]
        for lag in lags_by_group.get(g, []):
            out[f"{feat}_lag{lag}"] = out[feat].shift(lag)
    return out


def add_group_rollings(df: pd.DataFrame, feature_cols: Iterable[str], rolling_by_group: Dict[str, List[int]]) -> pd.DataFrame:
    out = df.copy()
    for feat in feature_cols:
        g = feat[0]
        for w in rolling_by_group.get(g, []):
            mp = max(2, w // 3)
            out[f"{feat}_roll_mean_{w}"] = out[feat].shift(1).rolling(w, min_periods=mp).mean()
            if g in {"M", "V"}:
                out[f"{feat}_roll_std_{w}"] = out[feat].shift(1).rolling(w, min_periods=mp).std()
    return out


def ic_filter_and_decorrelate(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
    min_samples: int = 200,
    ic_threshold: float = 0.01,
    p_threshold: float = 0.05,
    corr_threshold: float = 0.9,
) -> List[str]:
    ic_records = []
    for col in feature_cols:
        tmp = df[[col, target_col]].dropna()
        if len(tmp) < min_samples:
            continue
        ic, pval = spearmanr(tmp[col], tmp[target_col])
        ic_records.append({"feature": col, "ic": ic, "abs_ic": abs(ic), "pval": pval, "n": len(tmp)})

    if not ic_records:
        return list(feature_cols)

    ic_df = pd.DataFrame(ic_records).sort_values("abs_ic", ascending=False)
    keep_ic = ic_df[(ic_df["abs_ic"] >= ic_threshold) & (ic_df["pval"] <= p_threshold)]
    selected = keep_ic["feature"].tolist()

    if not selected:
        return list(feature_cols)

    corr = df[selected].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).fillna(0.0)
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    final_features = [c for c in selected if c not in to_drop]
    return final_features


def apply_feature_engineering(
    df: pd.DataFrame,
    target_col: str,
    info_cols: Iterable[str],
    use_expanding_zscore: bool,
    expanding_min_periods: int,
    lags_by_group: Dict[str, List[int]],
    rolling_by_group: Dict[str, List[int]],
) -> pd.DataFrame:
    out = df.sort_values(info_cols[0]).reset_index(drop=True)
    base_features = get_feature_cols(out, target_col, info_cols)

    if use_expanding_zscore:
        z_df = expanding_zscore(out, base_features, min_periods=expanding_min_periods)
        out = pd.concat([out, z_df], axis=1)

    out = add_group_lags(out, base_features, lags_by_group)
    out = add_group_rollings(out, base_features, rolling_by_group)
    return out
