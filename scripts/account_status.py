"""Inspect the Alpaca paper account: equity, cash, positions, PnL, open orders.

Usage:
    uv run python scripts/account_status.py [--config configs/paper_trading.yaml] [--record]

--record appends a daily snapshot row to logs/paper_trading/history.csv and,
when enough history exists, prints paper-trading performance metrics.
"""

from __future__ import annotations

import argparse
import datetime as dt

import _bootstrap  # noqa: F401

import pandas as pd

from core.config import load_config, require
from core.log import get_logger
from trading.state import HISTORY_FILE, append_history, paper_log_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper_trading.yaml")
    parser.add_argument("--record", action="store_true", help="append snapshot to paper history CSV")
    args = parser.parse_args()

    log = get_logger("account_status")
    cfg = load_config(args.config)

    from trading.alpaca_paper import AlpacaBroker

    broker = AlpacaBroker(require(cfg, "broker"))
    account = broker.get_account()
    positions = broker.get_positions()
    open_orders = broker.get_orders(open_only=True)

    print(f"\n=== Alpaca account ({broker.name}) ===")
    print(f"status        : {account.status}")
    print(f"equity        : {account.equity:,.2f} {account.currency}")
    print(f"cash          : {account.cash:,.2f}")
    print(f"buying power  : {account.buying_power:,.2f}")

    print(f"\n--- positions ({len(positions)}) ---")
    if positions:
        rows = [{"symbol": p.symbol, "qty": p.qty, "market_value": p.market_value,
                 "avg_entry": p.avg_entry_price, "price": p.current_price,
                 "unrealized_pl": p.unrealized_pl} for p in positions]
        print(pd.DataFrame(rows).to_string(index=False))
        total_upl = sum(p.unrealized_pl or 0 for p in positions)
        print(f"total market value: {sum(p.market_value for p in positions):,.2f} | "
              f"total unrealized PnL: {total_upl:,.2f}")
    else:
        print("(none)")

    print(f"\n--- open orders ({len(open_orders)}) ---")
    for o in open_orders:
        print(f"  {o.id} | {o.side:4s} {o.symbol:6s} qty={o.qty} filled={o.filled_qty} status={o.status}")
    if not open_orders:
        print("(none)")

    if args.record:
        log_dir = paper_log_dir(cfg)
        append_history(log_dir, {
            "date": str(dt.date.today()), "equity": account.equity, "cash": account.cash,
            "buying_power": account.buying_power, "n_positions": len(positions),
            "gross_exposure": sum(p.market_value for p in positions) / max(account.equity, 1),
            "unrealized_pl": sum(p.unrealized_pl or 0 for p in positions),
            "turnover": None, "orders_submitted": None, "orders_rejected": None,
        })
        log.info("Snapshot appended to %s", log_dir / HISTORY_FILE)

        hist = pd.read_csv(log_dir / HISTORY_FILE, parse_dates=["date"]).drop_duplicates("date", keep="last")
        if len(hist) >= 20:
            from backtests.metrics import summarize_returns

            returns = hist.set_index("date")["equity"].pct_change().dropna()
            print("\n--- paper performance (from history) ---")
            for k, v in summarize_returns(returns, "paper_").items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"\n({len(hist)} history rows — performance metrics start at 20)")


if __name__ == "__main__":
    main()
