"""Pre-trade risk controls. A validation failure means DO NOT TRADE (with logged reasons).

Trading safety is higher priority than strategy performance.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from trading.broker import Account, BrokerOrder, Order, Position
from trading.portfolio import PriceFn


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)

    @classmethod
    def failure(cls, reasons: list[str]) -> "ValidationResult":
        return cls(ok=False, reasons=reasons)

    @classmethod
    def success(cls) -> "ValidationResult":
        return cls(ok=True)


def validate_predictions(
    scores: pd.Series,
    prediction_date: dt.date,
    signal_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    today: dt.date | None = None,
    ignore_staleness: bool = False,
) -> ValidationResult:
    """Reject empty / NaN-heavy / degenerate / stale prediction cross-sections."""
    reasons: list[str] = []
    today = today or dt.date.today()

    if scores is None or len(scores) == 0:
        return ValidationResult.failure(["predictions are empty"])
    nan_ratio = float(scores.isna().mean())
    max_nan = float(risk_cfg.get("max_nan_ratio", 0.2))
    if nan_ratio > max_nan:
        reasons.append(f"NaN ratio {nan_ratio:.2%} exceeds limit {max_nan:.2%}")
    clean = scores.dropna()
    if len(clean) > 0 and float(clean.std()) == 0.0:
        reasons.append("predictions have zero cross-sectional variance (degenerate model output)")

    max_age = int(signal_cfg.get("max_prediction_age_days", 5))
    age = (today - prediction_date).days
    if age > max_age and not ignore_staleness:
        reasons.append(
            f"prediction date {prediction_date} is {age} days old (limit {max_age}); "
            "refresh data/model or pass --ignore-staleness for offline testing"
        )
    return ValidationResult(ok=not reasons, reasons=reasons)


def validate_orders(
    orders: list[Order],
    account: Account,
    positions: list[Position],
    open_orders: list[BrokerOrder],
    risk_cfg: dict[str, Any],
    price_of: PriceFn,
) -> ValidationResult:
    """Portfolio-level pre-trade checks. Rejects the whole batch on any violation."""
    reasons: list[str] = []
    equity = float(account.equity)
    if equity <= 0:
        return ValidationResult.failure(["account equity is non-positive"])

    def order_value(o: Order) -> float:
        px = price_of(o.symbol)
        return float(o.qty) * float(px) if px else float("nan")

    values = {id(o): order_value(o) for o in orders}
    if any(pd.isna(v) for v in values.values()):
        bad = [o.symbol for o in orders if pd.isna(values[id(o)])]
        reasons.append(f"no price available to value orders: {bad}")
        return ValidationResult.failure(reasons)

    max_single = float(risk_cfg.get("max_single_order_weight", 0.15))
    for o in orders:
        w = values[id(o)] / equity
        if w > max_single:
            reasons.append(f"order {o.side} {o.symbol} weight {w:.2%} exceeds max_single_order_weight {max_single:.2%}")

    turnover = sum(values.values()) / equity
    max_turnover = float(risk_cfg.get("max_daily_turnover", 0.5))
    is_initial_build = not positions  # bootstrapping an empty portfolio is not churn
    if turnover > max_turnover and not is_initial_build:
        reasons.append(f"turnover {turnover:.2%} exceeds max_daily_turnover {max_turnover:.2%}")

    buy_cost = sum(values[id(o)] for o in orders if o.side == "buy")
    sell_proceeds = sum(values[id(o)] for o in orders if o.side == "sell")
    exposure_now = sum(p.market_value for p in positions)
    exposure_post = (exposure_now + buy_cost - sell_proceeds) / equity
    max_expo = float(risk_cfg.get("max_total_exposure", 0.95))
    if exposure_post > max_expo:
        reasons.append(f"post-trade exposure {exposure_post:.2%} exceeds max_total_exposure {max_expo:.2%}")

    cash_post = float(account.cash) + sell_proceeds - buy_cost
    min_cash = float(risk_cfg.get("min_cash_ratio", 0.0)) * equity
    if cash_post < min_cash:
        reasons.append(f"post-trade cash {cash_post:.0f} below minimum reserve {min_cash:.0f}")
    if buy_cost > float(account.buying_power) + sell_proceeds:
        reasons.append("insufficient buying power for buy orders (even after sells)")

    open_keys = {(bo.symbol, bo.side) for bo in open_orders}
    for o in orders:
        if (o.symbol, o.side) in open_keys:
            reasons.append(f"duplicate open order exists for {o.side} {o.symbol}")

    return ValidationResult(ok=not reasons, reasons=reasons)
