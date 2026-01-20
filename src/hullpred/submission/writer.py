from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def write_submission(
    preds,
    test_df: pd.DataFrame,
    output_path: str | Path,
    row_id_col: Optional[str] = None,
    target_col: str = "prediction",
) -> pd.DataFrame:
    output_path = Path(output_path)

    if row_id_col is None:
        row_id_col = "batch_id" if "batch_id" in test_df.columns else "date_id"

    sub = pd.DataFrame({row_id_col: test_df[row_id_col].values, target_col: preds})
    sub.to_csv(output_path, index=False)
    return sub
