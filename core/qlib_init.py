"""Qlib initialization from project config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import REPO_ROOT, require

_INITIALIZED = False


def init_qlib(cfg: dict[str, Any]) -> None:
    """Initialize qlib from the ``qlib:`` section of the config (idempotent per process)."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    import os

    # qlib creates an mlflow file-store experiment manager at init time; mlflow>=3.15
    # requires this explicit opt-in for the filesystem backend (we do our own
    # experiment tracking under results/, mlflow is unused).
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    import qlib
    from qlib.constant import REG_CN, REG_US

    region_key = str(require(cfg, "qlib.region")).lower()
    regions = {"us": REG_US, "cn": REG_CN}
    if region_key not in regions:
        raise ValueError(f"Unsupported qlib region: {region_key!r} (expected one of {sorted(regions)})")

    uri = Path(require(cfg, "qlib.provider_uri"))
    if not uri.is_absolute():
        uri = REPO_ROOT / uri
    if not (uri / "calendars").exists():
        raise FileNotFoundError(
            f"Qlib data not found at {uri}. Run: uv run python scripts/prepare_data.py"
        )
    qlib.init(provider_uri=str(uri), region=regions[region_key])
    _INITIALIZED = True
