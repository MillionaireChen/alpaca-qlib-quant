"""Compare backtest expectations against actual paper-trading results.

Usage:
    uv run python scripts/paper_vs_backtest.py --config configs/paper_trading.yaml [--exp-dir results/...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from core.config import load_config, require
from core.experiment import latest_experiment_dir, save_json
from core.log import get_logger
from trading.state import HISTORY_FILE, paper_log_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper_trading.yaml")
    parser.add_argument("--exp-dir", default=None, help="backtest experiment dir (default: latest)")
    args = parser.parse_args()

    log = get_logger("paper_vs_backtest")
    cfg = load_config(args.config)
    model_cfg = load_config(require(cfg, "signal.model_config"))
    exp_dir = Path(args.exp_dir) if args.exp_dir else latest_experiment_dir(model_cfg)

    report_path = exp_dir / "backtest" / "backtest_report_daily.csv"
    if not report_path.exists():
        raise FileNotFoundError(f"{report_path} not found — run scripts/backtest.py first.")
    backtest_report = pd.read_csv(report_path, index_col=0, parse_dates=True)

    log_dir = paper_log_dir(cfg)
    hist_path = log_dir / HISTORY_FILE
    paper_history = pd.read_csv(hist_path, parse_dates=["date"]) if hist_path.exists() else None

    run_records = []
    for p in sorted(log_dir.glob("run_*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("type") != "reconciliation" and not rec.get("aborted"):
            run_records.append(rec)

    from backtests.comparison import compare_backtest_vs_paper

    result = compare_backtest_vs_paper(backtest_report, paper_history, run_records)
    out = log_dir / "backtest_vs_paper.json"
    save_json(result, out)
    print(json.dumps(result, indent=2, default=str))
    log.info("Comparison written to %s", out)


if __name__ == "__main__":
    main()
