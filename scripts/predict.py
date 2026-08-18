"""Generate test-segment predictions and IC / Rank IC analysis.

Usage:
    uv run python scripts/predict.py --config configs/lightgbm_alpha158.yaml [--exp-dir results/...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtests.metrics import ic_analysis
from core.config import load_config
from core.experiment import latest_experiment_dir, save_json
from core.handlers import build_dataset
from core.log import get_logger
from core.qlib_init import init_qlib
from models import LightGBMModel


def plot_ic(daily: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    daily["ic"].rolling(20).mean().plot(ax=ax, label="IC (20d MA)", color="#4269d0")
    daily["rank_ic"].rolling(20).mean().plot(ax=ax, label="Rank IC (20d MA)", color="#efb118")
    ax.axhline(0, color="#9498a0", lw=0.8)
    ax.set_title("Daily IC / Rank IC (20-day moving average)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_pred_analysis(pred: pd.Series, label: pd.Series, out_dist: Path, out_deciles: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pred.dropna(), bins=100, color="#4269d0")
    ax.set_title("Prediction score distribution")
    fig.tight_layout()
    fig.savefig(out_dist, dpi=150)
    plt.close(fig)

    df = pd.concat({"score": pred, "label": label}, axis=1).dropna()
    # decile by prediction within each day, then average realized return per decile
    df["decile"] = df.groupby(level="datetime")["score"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )
    dec = df.groupby("decile")["label"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    dec.plot(kind="bar", ax=ax, color="#4269d0")
    ax.set_title("Realized 5-day return (%) by prediction decile (0=lowest score)")
    ax.set_xlabel("prediction decile")
    fig.tight_layout()
    fig.savefig(out_deciles, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp-dir", default=None, help="experiment dir (default: latest for this experiment name)")
    parser.add_argument("--segment", default="test")
    args = parser.parse_args()

    log = get_logger("predict")
    cfg = load_config(args.config)
    init_qlib(cfg)

    exp_dir = Path(args.exp_dir) if args.exp_dir else latest_experiment_dir(cfg)
    model = LightGBMModel.load(exp_dir / "model.pkl")
    log.info("Loaded model from %s", exp_dir)

    dataset = build_dataset(cfg)
    pred = model.predict(dataset, segment=args.segment)

    from qlib.data.dataset.handler import DataHandlerLP

    label_df = dataset.prepare(args.segment, col_set="label", data_key=DataHandlerLP.DK_I)
    label = label_df.iloc[:, 0]  # raw (unnormalized) forward return

    pred_out = exp_dir / f"pred_{args.segment}.csv"
    pred.to_frame().to_csv(pred_out)
    log.info("Saved %d predictions to %s", len(pred), pred_out)

    daily, summary = ic_analysis(pred, label)
    daily.to_csv(exp_dir / f"ic_daily_{args.segment}.csv")
    save_json(summary, exp_dir / f"ic_summary_{args.segment}.json")
    plot_ic(daily, exp_dir / f"ic_series_{args.segment}.png")
    plot_pred_analysis(pred, label, exp_dir / f"pred_distribution_{args.segment}.png",
                       exp_dir / f"pred_deciles_{args.segment}.png")
    log.info("IC summary: %s", summary)


if __name__ == "__main__":
    main()
