"""Local run records, expected-state snapshots, and paper-trading history.

These files are expectations/logs only — the broker remains the source of truth.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import REPO_ROOT

STATE_FILE = "state_latest.json"
HISTORY_FILE = "history.csv"

HISTORY_COLUMNS = [
    "date", "equity", "cash", "buying_power", "n_positions", "gross_exposure",
    "unrealized_pl", "turnover", "orders_submitted", "orders_rejected",
]


def paper_log_dir(cfg: dict[str, Any]) -> Path:
    d = Path(cfg.get("logging", {}).get("dir", "logs/paper_trading"))
    if not d.is_absolute():
        d = REPO_ROOT / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run_record(log_dir: Path, record: dict[str, Any]) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = log_dir / f"run_{ts}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    return path


def load_state(log_dir: Path) -> dict[str, Any] | None:
    path = log_dir / STATE_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(log_dir: Path, state: dict[str, Any]) -> None:
    with (log_dir / STATE_FILE).open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def append_history(log_dir: Path, row: dict[str, Any]) -> Path:
    path = log_dir / HISTORY_FILE
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in HISTORY_COLUMNS})
    return path
