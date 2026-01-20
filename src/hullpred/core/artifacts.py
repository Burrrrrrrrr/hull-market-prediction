from __future__ import annotations

from pathlib import Path
from typing import Dict

from hullpred.core.paths import resolve_path


def ensure_artifacts_dirs(artifacts_dir: str | Path) -> Dict[str, Path]:
    base = resolve_path(artifacts_dir)
    paths = {
        "base": base,
        "features": base / "features",
        "models": base / "models",
        "oof": base / "oof",
        "preds": base / "preds",
        "reports": base / "reports",
        "submission": base / "submission",
    }

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    return paths
