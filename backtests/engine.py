"""Historical backtest engine: prediction signal -> Top-K portfolio -> daily PnL.

Transaction costs are always applied (see configs/*.yaml, backtest.exchange).
Zero-cost results are never reported.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.config import require
from strategies.topk import make_topk_strategy


def run_backtest(
    cfg: dict[str, Any],
    signal: pd.Series,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    """Run the daily Top-K backtest over [start_time, end_time] (defaults: test segment).

    Returns dict with:
        report: daily DataFrame (return, bench, turnover, cost, account value, ...)
        positions: long DataFrame (datetime, instrument, amount, price, weight)
    """
    from qlib.backtest import backtest as qlib_backtest

    from qlib.data import D

    ds = require(cfg, "dataset")
    start_time = start_time or str(ds["test"][0])
    end_time = end_time or str(ds["test"][1])
    # qlib's trade calendar needs one day AFTER the backtest end; clamp the end
    # to the second-to-last available calendar day to avoid an IndexError.
    cal = D.calendar(freq="day")
    last_valid = pd.Timestamp(cal[-2])
    if pd.Timestamp(end_time) > last_valid:
        end_time = str(last_valid.date())
    bt = require(cfg, "backtest")
    exch = dict(bt.get("exchange", {}))

    strategy = make_topk_strategy(cfg, signal)
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }
    exchange_kwargs = {
        "freq": "day",
        "limit_threshold": exch.get("limit_threshold"),
        "deal_price": exch.get("deal_price", "close"),
        "open_cost": float(exch.get("open_cost", 0.0005)),
        "close_cost": float(exch.get("close_cost", 0.0005)),
        "min_cost": float(exch.get("min_cost", 1)),
        "trade_unit": int(exch.get("trade_unit", 1)),
    }
    portfolio_metric, indicator = qlib_backtest(
        start_time=start_time,
        end_time=end_time,
        strategy=strategy,
        executor=executor_config,
        benchmark=require(cfg, "universe.benchmark"),
        account=float(bt.get("initial_capital", 1_000_000)),
        exchange_kwargs=exchange_kwargs,
    )
    report_df, positions = portfolio_metric["1day"]
    report_df = report_df.dropna(how="all")
    pos_df = _positions_to_frame(positions)
    return {"report": report_df, "positions": pos_df, "indicator": indicator}


def _positions_to_frame(positions: dict) -> pd.DataFrame:
    """Flatten qlib Position objects into a long DataFrame with daily holdings and weights."""
    rows: list[dict[str, Any]] = []
    for dt, pos in positions.items():
        try:
            stock_amounts = pos.get_stock_amount_dict()
            cash = pos.get_cash()
        except AttributeError:  # defensive: qlib API variations
            continue
        total = pos.calculate_value()
        for code, amount in stock_amounts.items():
            price = pos.get_stock_price(code)
            value = (amount or 0) * (price or 0)
            rows.append(
                {
                    "datetime": pd.Timestamp(dt),
                    "instrument": code,
                    "amount": float(amount),
                    "price": float(price) if price is not None else float("nan"),
                    "value": float(value),
                    "weight": float(value / total) if total else float("nan"),
                }
            )
        rows.append(
            {
                "datetime": pd.Timestamp(dt),
                "instrument": "__CASH__",
                "amount": float(cash),
                "price": 1.0,
                "value": float(cash),
                "weight": float(cash / total) if total else float("nan"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["datetime", "instrument", "amount", "price", "value", "weight"])
    return pd.DataFrame(rows).set_index(["datetime", "instrument"]).sort_index()
