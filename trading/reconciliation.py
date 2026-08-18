"""Reconciliation: compare expected (local) state with actual broker state.

The broker is ALWAYS the source of truth; local records are only expectations.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from trading.broker import Broker

QTY_TOL = 1e-6


def reconcile(
    broker: Broker,
    expected_positions: dict[str, float],
    expected_cash: float | None = None,
    cash_tolerance: float = 1.0,
    stale_order_minutes: int = 60,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Return a discrepancy report between expectations and live broker state."""
    now = now or dt.datetime.now(dt.timezone.utc)
    positions = {p.symbol: p.qty for p in broker.get_positions()}
    open_orders = broker.get_orders(open_only=True)
    all_orders = broker.get_orders(open_only=False)

    missing_fills = sorted(s for s, q in expected_positions.items() if q > QTY_TOL and s not in positions)
    unexpected_holdings = sorted(s for s in positions if s not in expected_positions)
    qty_mismatches = {
        s: {"expected": expected_positions[s], "actual": positions[s]}
        for s in set(expected_positions) & set(positions)
        if abs(expected_positions[s] - positions[s]) > QTY_TOL
    }
    partial_fills = [
        {"id": o.id, "symbol": o.symbol, "side": o.side, "qty": o.qty, "filled_qty": o.filled_qty}
        for o in all_orders
        if o.status == "partially_filled" or (o.status == "filled" and abs(o.filled_qty - o.qty) > QTY_TOL)
    ]
    rejected = [
        {"id": o.id, "symbol": o.symbol, "side": o.side, "qty": o.qty, "status": o.status}
        for o in all_orders
        if o.status in ("rejected", "expired", "canceled")
    ]
    stale_open = []
    for o in open_orders:
        age_min = None
        if o.submitted_at is not None:
            sub = o.submitted_at if o.submitted_at.tzinfo else o.submitted_at.replace(tzinfo=dt.timezone.utc)
            age_min = (now - sub).total_seconds() / 60.0
        if age_min is None or age_min > stale_order_minutes:
            stale_open.append({"id": o.id, "symbol": o.symbol, "side": o.side,
                               "status": o.status, "age_minutes": age_min})

    cash_diff = None
    account = broker.get_account()
    if expected_cash is not None:
        cash_diff = float(account.cash) - float(expected_cash)

    ok = not (
        missing_fills
        or unexpected_holdings
        or qty_mismatches
        or stale_open
        or (cash_diff is not None and abs(cash_diff) > cash_tolerance)
    )
    return {
        "ok": ok,
        "checked_at": now.isoformat(),
        "broker": broker.name,
        "account": {"equity": account.equity, "cash": account.cash, "buying_power": account.buying_power},
        "missing_fills": missing_fills,
        "unexpected_holdings": unexpected_holdings,
        "qty_mismatches": qty_mismatches,
        "partial_fills": partial_fills,
        "rejected_or_canceled": rejected,
        "stale_open_orders": stale_open,
        "cash_difference": cash_diff,
        "actual_positions": positions,
        "expected_positions": expected_positions,
    }
