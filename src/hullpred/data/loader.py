from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from hullpred.core.paths import resolve_path


def load_train_test(train_path: str | Path, test_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = resolve_path(train_path)
    test_path = resolve_path(test_path)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df
