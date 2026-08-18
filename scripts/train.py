"""Train the LightGBM baseline on the Alpha158 dataset.

Usage:
    uv run python scripts/train.py --config configs/lightgbm_alpha158.yaml
"""

from __future__ import annotations

import argparse
import random

import _bootstrap  # noqa: F401

import numpy as np

from core.config import load_config, require
from core.experiment import create_experiment_dir, experiment_metadata, save_json
from core.handlers import build_dataset
from core.log import get_logger
from core.qlib_init import init_qlib
from models import LightGBMModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    log = get_logger("train")
    cfg = load_config(args.config)

    seed = int(require(cfg, "experiment.seed"))
    random.seed(seed)
    np.random.seed(seed)

    init_qlib(cfg)
    log.info("Building Alpha158 dataset (universe=%s)...", require(cfg, "universe.market"))
    dataset = build_dataset(cfg)

    model = LightGBMModel(
        params=require(cfg, "model.params"),
        seed=seed,
        early_stopping_rounds=int(cfg["model"].get("early_stopping_rounds", 50)),
        num_boost_round=int(cfg["model"].get("num_boost_round", 1000)),
    )
    log.info("Training LightGBM (seed=%d)...", seed)
    model.fit(dataset)
    log.info(
        "Training done. best_iteration=%s | valid rank IC @best=%s",
        model.best_iteration,
        model.best_valid_rank_ic,
    )

    exp_dir = create_experiment_dir(cfg)
    model.save(exp_dir / "model.pkl")
    meta = experiment_metadata(cfg)
    meta["best_iteration"] = model.best_iteration
    meta["valid_rank_ic_at_best"] = model.best_valid_rank_ic
    save_json(meta, exp_dir / "experiment.json")
    log.info("Saved model + metadata to %s", exp_dir)


if __name__ == "__main__":
    main()
