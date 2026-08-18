"""YAML configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict. Dates are normalized to ISO strings."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping: {p}")
    cfg = _stringify_dates(cfg)
    cfg["_config_path"] = str(p.resolve())
    return cfg


def require(cfg: dict[str, Any], dotted_key: str) -> Any:
    """Fetch a nested config value like ``require(cfg, "model.params")``; raise a clear error if missing."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Missing required config key: '{dotted_key}' (failed at '{part}')")
        node = node[part]
    return node


def _stringify_dates(obj: Any) -> Any:
    """PyYAML parses bare dates into datetime.date; normalize to ISO strings for qlib/pandas."""
    import datetime as _dt

    if isinstance(obj, dict):
        return {k: _stringify_dates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_dates(v) for v in obj]
    if isinstance(obj, (_dt.date, _dt.datetime)):
        return obj.isoformat()
    return obj
