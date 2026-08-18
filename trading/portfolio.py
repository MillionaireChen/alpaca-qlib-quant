"""Target portfolio construction and diff-based order generation.

Orders are generated ONLY from the difference between the target and the
current broker portfolio: unchanged positions are never sold and re-bought.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import pandas as pd

from trading.broker import Account, Order, Position

PriceFn = Callable[[str], float | None]


def compute_target_weights(scores: pd.Series, topk: int, portfolio_cfg: dict[str, Any]) -> dict[str, float]:
    """Equal-weight the top-K symbols, scaled to keep the configured cash reserve.

    ``scores`` is one cross-section (one prediction date), indexed by symbol.
    """
    cash_ratio = float(portfolio_cfg.get("target_cash_ratio", 0.05))
    max_w = float(portfolio_cfg.get("max_position_weight", 0.15))
    ranked = scores.dropna().sort_values(ascending=False)
    if ranked.empty:
        return {}
    selected = list(ranked.head(topk).index)
    weight = min((1.0 - cash_ratio) / max(len(selected), 1), max_w)
    return {str(sym): weight for sym in selected}


def generate_orders(
    target_weights: dict[str, float],
    positions: list[Position],
    account: Account,
    price_of: PriceFn,
    portfolio_cfg: dict[str, Any],
) -> tuple[list[Order], list[str]]:
    """Diff current holdings against target weights; emit minimal integer-share orders.

    Returns (orders, notes). Sells come before buys so proceeds fund purchases.
    Symbols whose price is unavailable are skipped with a note (never trade blind).
    """
    equity = float(account.equity)
    min_trade_value = float(portfolio_cfg.get("min_trade_value", 0.0))
    tolerance = float(portfolio_cfg.get("rebalance_tolerance", 0.0))

    held = {p.symbol: p for p in positions}
    notes: list[str] = []
    sells: list[Order] = []
    buys: list[Order] = []

    # 1) full exits: held but not in target
    for sym, pos in held.items():
        if sym in target_weights:
            continue
        if pos.qty <= 0:
            notes.append(f"skip {sym}: non-positive qty {pos.qty}")
            continue
        sells.append(Order(symbol=sym, qty=math.floor(pos.qty), side="sell"))

    # 2) adjustments and entries
    for sym, tw in target_weights.items():
        pos = held.get(sym)
        cur_value = pos.market_value if pos else 0.0
        cur_w = cur_value / equity if equity > 0 else 0.0
        diff_w = tw - cur_w
        if abs(diff_w) <= tolerance:
            continue  # keep position untouched
        price = pos.current_price if (pos and pos.current_price) else price_of(sym)
        if price is None or price <= 0:
            notes.append(f"skip {sym}: no valid price")
            continue
        diff_value = diff_w * equity
        if abs(diff_value) < min_trade_value:
            continue
        qty = math.floor(abs(diff_value) / price)
        if qty < 1:
            continue
        if diff_value > 0:
            buys.append(Order(symbol=sym, qty=qty, side="buy"))
        else:
            qty = min(qty, math.floor(pos.qty)) if pos else 0
            if qty >= 1:
                sells.append(Order(symbol=sym, qty=qty, side="sell"))

    return sells + buys, notes
