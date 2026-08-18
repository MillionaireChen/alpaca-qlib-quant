"""Backtest vs paper-trading comparison (Phase 15).

Answers: does the strategy survive realistic execution? Compares expected
(backtest) turnover/returns/holdings against actual paper results. Historical
backtest PnL and paper PnL are reported side by side, never mixed.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def compare_backtest_vs_paper(
    backtest_report: pd.DataFrame,
    paper_history: pd.DataFrame | None,
    run_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the comparison summary. Missing paper data is reported, not fabricated."""
    out: dict[str, Any] = {"expected_from_backtest": {}, "actual_from_paper": {}, "per_run": []}

    net = backtest_report["return"] - backtest_report["cost"]
    out["expected_from_backtest"] = {
        "period": [str(backtest_report.index[0].date()), str(backtest_report.index[-1].date())],
        "mean_daily_return_net": float(net.mean()),
        "daily_return_std": float(net.std()),
        "mean_daily_turnover": float(backtest_report["turnover"].mean()),
    }

    if paper_history is not None and len(paper_history) >= 2:
        hist = paper_history.drop_duplicates("date", keep="last").set_index("date").sort_index()
        paper_ret = hist["equity"].pct_change().dropna()
        out["actual_from_paper"] = {
            "period": [str(hist.index[0].date()), str(hist.index[-1].date())],
            "n_days": int(len(hist)),
            "mean_daily_return": float(paper_ret.mean()) if len(paper_ret) else None,
            "daily_return_std": float(paper_ret.std()) if len(paper_ret) > 1 else None,
            "mean_daily_turnover": float(hist["turnover"].dropna().mean()) if hist["turnover"].notna().any() else None,
            "mean_gross_exposure": float(hist["gross_exposure"].dropna().mean()),
        }
        exp_t = out["expected_from_backtest"]["mean_daily_turnover"]
        act_t = out["actual_from_paper"].get("mean_daily_turnover")
        if act_t is not None:
            out["turnover_gap"] = {"expected": exp_t, "actual": act_t, "actual_minus_expected": act_t - exp_t}
    else:
        out["actual_from_paper"] = {"note": "insufficient paper history (need >= 2 daily snapshots)"}

    for rec in run_records:
        planned = set(rec.get("target_weights", {}))
        actual = {s for s, q in rec.get("positions_after", {}).items() if q and float(q) > 0}
        submitted = rec.get("execution", {}).get("submitted", [])
        n_filled = sum(1 for s in submitted if s.get("status") == "filled")
        out["per_run"].append(
            {
                "run_id": rec.get("run_id"),
                "prediction_date": rec.get("prediction_date"),
                "dry_run": rec.get("dry_run"),
                "planned_holdings": sorted(planned),
                "actual_holdings": sorted(actual),
                "holdings_overlap_ratio": (len(planned & actual) / len(planned)) if planned else None,
                "orders_submitted": len(submitted),
                "orders_filled": n_filled,
                "fill_ratio": (n_filled / len(submitted)) if submitted else None,
                "estimated_turnover": rec.get("estimated_turnover"),
            }
        )
    return out
