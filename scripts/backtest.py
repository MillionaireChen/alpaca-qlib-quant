"""Run the Top-K historical backtest on saved predictions.

Usage:
    uv run python scripts/backtest.py --config configs/lightgbm_alpha158.yaml [--exp-dir results/...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from core.config import load_config
from core.experiment import latest_experiment_dir
from core.log import get_logger
from core.qlib_init import init_qlib


def load_signal(exp_dir: Path, segment: str = "test") -> pd.Series:
    path = exp_dir / f"pred_{segment}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/predict.py first.")
    df = pd.read_csv(path, parse_dates=["datetime"])
    signal = df.set_index(["datetime", "instrument"])["score"].sort_index()
    return signal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp-dir", default=None)
    args = parser.parse_args()

    log = get_logger("backtest")
    cfg = load_config(args.config)
    init_qlib(cfg)

    exp_dir = Path(args.exp_dir) if args.exp_dir else latest_experiment_dir(cfg)
    signal = load_signal(exp_dir)
    log.info("Loaded signal: %d rows, %s -> %s", len(signal),
             signal.index.get_level_values(0).min().date(),
             signal.index.get_level_values(0).max().date())

    from backtests.engine import run_backtest
    from backtests.report import build_report

    result = run_backtest(cfg, signal)
    out_dir = exp_dir / "backtest"

    ic_path = exp_dir / "ic_summary_test.json"
    ic_summary = json.loads(ic_path.read_text()) if ic_path.exists() else None
    metrics = build_report(result["report"], out_dir, ic_summary=ic_summary)
    result["positions"].to_csv(out_dir / "positions_daily.csv")

    log.info("Backtest metrics (net of costs):")
    for k in ("portfolio_net_annualized_return", "portfolio_net_sharpe_ratio", "portfolio_net_max_drawdown",
              "benchmark_annualized_return", "excess_net_annualized_return", "mean_daily_turnover"):
        log.info("  %s = %.4f", k, metrics[k])
    log.info("Report written to %s", out_dir)


if __name__ == "__main__":
    main()
