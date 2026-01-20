from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def time_split_by_ratio(df: pd.DataFrame, date_col: str, split_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * split_ratio)
    train = df.iloc[:split_idx].reset_index(drop=True)
    valid = df.iloc[split_idx:].reset_index(drop=True)
    return train, valid


def time_split_by_days(df: pd.DataFrame, date_col: str, n_valid: int, n_test: int):
    df = df.sort_values(date_col).reset_index(drop=True)
    unique_dates = np.sort(df[date_col].unique())
    train_dates = unique_dates[: -(n_valid + n_test)]
    valid_dates = unique_dates[-(n_valid + n_test) : -n_test]
    test_dates = unique_dates[-n_test:]

    train_mask = df[date_col].isin(train_dates)
    valid_mask = df[date_col].isin(valid_dates)
    test_mask = df[date_col].isin(test_dates)
    return train_mask, valid_mask, test_mask
