"""Experiment directory management and reproducibility metadata.

Every run writes into its own timestamped directory under results/ —
previous experiments are never overwritten.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import REPO_ROOT, require


def results_root(cfg: dict[str, Any] | None = None) -> Path:
    root = REPO_ROOT / "results"
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_experiment_dir(cfg: dict[str, Any]) -> Path:
    """Create results/<timestamp>_<name>/ and snapshot the config into it."""
    name = require(cfg, "experiment.name")
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_dir = results_root(cfg) / f"{ts}_{name}"
    exp_dir.mkdir(parents=True, exist_ok=False)
    src = cfg.get("_config_path")
    if src and Path(src).exists():
        shutil.copy2(src, exp_dir / "config_snapshot.yaml")
    return exp_dir


def latest_experiment_dir(cfg: dict[str, Any]) -> Path:
    """Return the most recent experiment dir matching the configured experiment name."""
    name = require(cfg, "experiment.name")
    candidates = sorted(d for d in results_root(cfg).iterdir() if d.is_dir() and d.name.endswith(f"_{name}"))
    if not candidates:
        raise FileNotFoundError(f"No experiment dirs for '{name}' under results/. Run scripts/train.py first.")
    return candidates[-1]


def experiment_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    """Collect everything needed to reproduce this experiment."""
    ds = require(cfg, "dataset")
    return {
        "experiment_name": require(cfg, "experiment.name"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_period": ds.get("train"),
        "valid_period": ds.get("valid"),
        "test_period": ds.get("test"),
        "universe": require(cfg, "universe.market"),
        "benchmark": require(cfg, "universe.benchmark"),
        "feature_set": "Alpha158 (no VWAP, 157 features)",
        "prediction_horizon_days": require(cfg, "label.horizon"),
        "label_definition": "close[t+h]/close[t+1] - 1",
        "model_type": require(cfg, "model.type"),
        "model_params": require(cfg, "model.params"),
        "strategy_params": cfg.get("strategy"),
        "backtest_params": cfg.get("backtest"),
        "random_seed": require(cfg, "experiment.seed"),
        "prices": "split+dividend adjusted (qlib US bundle, Yahoo-normalized)",
        "data_provider": require(cfg, "qlib.provider_uri"),
    }


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
