from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from hullpred.core.paths import resolve_path


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = resolve_path("configs/default.yaml")
    else:
        config_path = resolve_path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config format: {config_path}")
    return cfg
