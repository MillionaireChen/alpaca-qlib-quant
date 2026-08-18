"""Performance reporting: metrics JSON + standard plots into the experiment directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtests import metrics as M

_BLUE, _ORANGE, _GRAY, _RED = "#4269d0", "#efb118", "#9498a0", "#ff725c"


def build_report(report_df: pd.DataFrame, out_dir: Path, ic_summary: dict | None = None) -> dict[str, Any]:
    """Compute the full metric block (costs included) and write plots + JSON under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    ret = report_df["return"] - report_df["cost"]        # portfolio daily return, net of costs
    ret_gross = report_df["return"]
    bench = report_df["bench"]
    excess = ret - bench

    result: dict[str, Any] = {
        "period": {"start": str(report_df.index[0].date()), "end": str(report_df.index[-1].date()),
                   "n_days": int(len(report_df))},
        "costs_included": True,
        **M.summarize_returns(ret, prefix="portfolio_net_"),
        **M.summarize_returns(bench, prefix="benchmark_"),
        **M.summarize_returns(excess, prefix="excess_net_"),
        "gross_annualized_return": M.annualized_return(ret_gross),
        "mean_daily_turnover": float(report_df["turnover"].mean()),
        "total_cost_rate_sum": float(report_df["cost"].sum()),
    }
    if ic_summary:
        result["information_coefficient"] = ic_summary

    with (out_dir / "backtest_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    report_df.to_csv(out_dir / "backtest_report_daily.csv")

    _plot_equity(ret, bench, out_dir / "equity_curve.png")
    _plot_excess(excess, out_dir / "excess_return_curve.png")
    _plot_drawdown(ret, bench, out_dir / "drawdown_curve.png")
    _plot_turnover(report_df["turnover"], out_dir / "turnover.png")
    return result


def _plot_equity(ret: pd.Series, bench: pd.Series, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    (1 + ret).cumprod().plot(ax=ax, label="Portfolio (net of costs)", color=_BLUE)
    (1 + bench).cumprod().plot(ax=ax, label="Benchmark", color=_GRAY)
    ax.set_title("Equity curve")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_excess(excess: pd.Series, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    (1 + excess).cumprod().sub(1).plot(ax=ax, color=_ORANGE)
    ax.axhline(0, color=_GRAY, lw=0.8)
    ax.set_title("Cumulative excess return vs benchmark (net of costs)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_drawdown(ret: pd.Series, bench: pd.Series, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    M.drawdown_series(ret).plot(ax=ax, label="Portfolio", color=_RED)
    M.drawdown_series(bench).plot(ax=ax, label="Benchmark", color=_GRAY)
    ax.set_title("Drawdown")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_turnover(turnover: pd.Series, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    turnover.rolling(20).mean().plot(ax=ax, color=_BLUE)
    ax.set_title("Daily turnover (20-day moving average)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
