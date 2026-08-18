"""Unit tests for the paper-trading layer: order generation, risk, execution, reconciliation, safety."""

import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.alpaca_paper import LiveTradingBlockedError, resolve_paper_mode  # noqa: E402
from trading.broker import Account, Order, Position  # noqa: E402
from trading.executor import Executor  # noqa: E402
from trading.mock_broker import MockBroker  # noqa: E402
from trading.portfolio import compute_target_weights, generate_orders  # noqa: E402
from trading.reconciliation import reconcile  # noqa: E402
from trading.risk import validate_orders, validate_predictions  # noqa: E402

LOG = logging.getLogger("test")

PORTFOLIO_CFG = {"target_cash_ratio": 0.05, "max_position_weight": 0.15,
                 "min_trade_value": 100, "rebalance_tolerance": 0.02}
RISK_CFG = {"max_daily_turnover": 0.50, "max_single_order_weight": 0.15,
            "max_total_exposure": 0.95, "min_cash_ratio": 0.03, "max_nan_ratio": 0.2}
SIGNAL_CFG = {"max_prediction_age_days": 5}


def acct(equity=100_000.0, cash=100_000.0):
    return Account(equity=equity, cash=cash, buying_power=cash)


def pos(symbol, value, price=100.0):
    return Position(symbol=symbol, qty=value / price, market_value=value, current_price=price)


# --------------------------- portfolio construction ---------------------------

def test_target_weights_equal_weight_with_cash_reserve():
    scores = pd.Series({f"S{i}": 10 - i for i in range(20)})
    target = compute_target_weights(scores, topk=10, portfolio_cfg=PORTFOLIO_CFG)
    assert len(target) == 10
    assert set(target) == {f"S{i}" for i in range(10)}  # highest scores win
    assert all(w == pytest.approx(0.095) for w in target.values())


def test_target_weights_capped():
    scores = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0})
    target = compute_target_weights(scores, topk=3, portfolio_cfg=PORTFOLIO_CFG)
    assert all(w == pytest.approx(0.15) for w in target.values())  # (1-5%)/3≈0.317 capped at 0.15


def test_generate_orders_keeps_unchanged_positions():
    """Spec example: keep AAPL+MSFT, sell NVDA, buy AMZN — no churn."""
    account = acct()
    positions = [pos("AAPL", 9_500), pos("MSFT", 9_500), pos("NVDA", 9_500)]
    target = {"AAPL": 0.095, "MSFT": 0.095, "AMZN": 0.095}
    prices = {"AAPL": 100.0, "MSFT": 100.0, "NVDA": 100.0, "AMZN": 100.0}
    orders, _ = generate_orders(target, positions, account, prices.get, PORTFOLIO_CFG)
    acts = {(o.side, o.symbol) for o in orders}
    assert acts == {("sell", "NVDA"), ("buy", "AMZN")}
    assert orders[0].side == "sell"  # sells first


def test_generate_orders_skips_symbol_without_price():
    orders, notes = generate_orders({"XXXX": 0.095}, [], acct(), lambda s: None, PORTFOLIO_CFG)
    assert orders == []
    assert any("no valid price" in n for n in notes)


def test_generate_orders_respects_min_trade_and_tolerance():
    account = acct()
    positions = [pos("AAPL", 9_400)]  # current weight 9.4% vs target 9.5% -> within tolerance
    orders, _ = generate_orders({"AAPL": 0.095}, positions, account, {"AAPL": 100.0}.get, PORTFOLIO_CFG)
    assert orders == []


# --------------------------------- risk ---------------------------------------

def today():
    return dt.date(2026, 8, 18)


def test_predictions_empty_blocked():
    res = validate_predictions(pd.Series(dtype=float), today(), SIGNAL_CFG, RISK_CFG, today=today())
    assert not res.ok


def test_predictions_nan_heavy_blocked():
    scores = pd.Series([1.0, float("nan"), float("nan"), float("nan")], index=list("ABCD"))
    res = validate_predictions(scores, today(), SIGNAL_CFG, RISK_CFG, today=today())
    assert not res.ok and any("NaN" in r for r in res.reasons)


def test_predictions_stale_blocked_and_override():
    scores = pd.Series({"A": 1.0, "B": 2.0})
    old = today() - dt.timedelta(days=30)
    assert not validate_predictions(scores, old, SIGNAL_CFG, RISK_CFG, today=today()).ok
    assert validate_predictions(scores, old, SIGNAL_CFG, RISK_CFG, today=today(), ignore_staleness=True).ok


def test_predictions_zero_variance_blocked():
    scores = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0})
    res = validate_predictions(scores, today(), SIGNAL_CFG, RISK_CFG, today=today())
    assert not res.ok and any("variance" in r for r in res.reasons)


def test_orders_oversize_blocked():
    orders = [Order("AAPL", qty=200, side="buy")]  # 20% of equity
    res = validate_orders(orders, acct(), [], [], RISK_CFG, {"AAPL": 100.0}.get)
    assert not res.ok and any("max_single_order_weight" in r for r in res.reasons)


def test_orders_turnover_blocked_when_rebalancing():
    orders = [Order(f"S{i}", qty=100, side="buy") for i in range(6)]  # 6 x 10% = 60% > 50%
    positions = [pos("MSFT", 10_000)]  # existing portfolio -> turnover cap applies
    res = validate_orders(orders, acct(), positions, [], {**RISK_CFG, "max_total_exposure": 1.0},
                          (lambda s: 100.0))
    assert not res.ok and any("max_daily_turnover" in r for r in res.reasons)


def test_initial_build_exempt_from_turnover_cap():
    orders = [Order(f"S{i}", qty=95, side="buy") for i in range(10)]  # 95% deployment from empty
    res = validate_orders(orders, acct(), [], [], {**RISK_CFG, "max_single_order_weight": 0.10},
                          (lambda s: 100.0))
    assert res.ok, res.reasons


def test_orders_duplicate_open_order_blocked():
    from trading.broker import BrokerOrder

    open_orders = [BrokerOrder("1", "AAPL", "buy", 10, 0, "new")]
    orders = [Order("AAPL", qty=10, side="buy")]
    res = validate_orders(orders, acct(), [], open_orders, RISK_CFG, {"AAPL": 100.0}.get)
    assert not res.ok and any("duplicate" in r for r in res.reasons)


def test_orders_cash_reserve_blocked():
    orders = [Order("AAPL", qty=140, side="buy")]  # uses 14k of 14k cash -> below 3% reserve
    account = Account(equity=100_000, cash=14_000, buying_power=14_000)
    positions = [pos("MSFT", 86_000)]
    res = validate_orders(orders, account, positions, [], {**RISK_CFG, "max_total_exposure": 1.0,
                                                           "max_single_order_weight": 0.2}, {"AAPL": 100.0}.get)
    assert not res.ok and any("reserve" in r for r in res.reasons)


# ------------------------------- execution -------------------------------------

def test_dry_run_submits_nothing():
    broker = MockBroker(price_map={"AAPL": 100.0})
    ex = Executor(broker, LOG, poll_interval_seconds=0.01, poll_timeout_seconds=1)
    report = ex.execute([Order("AAPL", 10, "buy")], dry_run=True, run_id="t1")
    assert report.dry_run and len(report.proposed) == 1
    assert broker.get_orders(open_only=False) == []
    assert broker.cash == 100_000.0


def test_execution_fills_and_updates_positions():
    broker = MockBroker(price_map={"AAPL": 100.0, "NVDA": 50.0}, positions={"NVDA": 20})
    ex = Executor(broker, LOG, poll_interval_seconds=0.01, poll_timeout_seconds=2)
    report = ex.execute([Order("NVDA", 20, "sell"), Order("AAPL", 5, "buy")], dry_run=False, run_id="t2")
    statuses = {s["symbol"]: s["status"] for s in report.submitted}
    assert statuses == {"NVDA": "filled", "AAPL": "filled"}
    qty = {p.symbol: p.qty for p in broker.get_positions()}
    assert qty == {"AAPL": 5}
    assert broker.cash == pytest.approx(100_000 + 20 * 50 - 5 * 100)


def test_execution_records_rejection_and_continues():
    broker = MockBroker(price_map={"AAPL": 100.0, "BAD": 10.0}, reject_symbols={"BAD"})
    ex = Executor(broker, LOG, poll_interval_seconds=0.01, poll_timeout_seconds=2)
    report = ex.execute([Order("BAD", 1, "buy"), Order("AAPL", 1, "buy")], dry_run=False, run_id="t3")
    statuses = {s["symbol"]: s["status"] for s in report.submitted}
    assert statuses["BAD"] == "rejected" and statuses["AAPL"] == "filled"


def test_duplicate_client_order_id_surfaces_as_error():
    broker = MockBroker(price_map={"AAPL": 100.0})
    ex = Executor(broker, LOG, poll_interval_seconds=0.01, poll_timeout_seconds=1)
    ex.execute([Order("AAPL", 1, "buy")], dry_run=False, run_id="same-run")
    report2 = ex.execute([Order("AAPL", 1, "buy")], dry_run=False, run_id="same-run")
    assert report2.errors and "duplicate" in report2.errors[0]


# ------------------------------ safety: live blocked ----------------------------

def test_paper_mode_default_ok():
    assert resolve_paper_mode({"mode": "paper", "allow_live_trading": False}) is True


def test_live_mode_blocked_without_override():
    with pytest.raises(LiveTradingBlockedError):
        resolve_paper_mode({"mode": "live", "allow_live_trading": False})


def test_live_mode_requires_explicit_override():
    assert resolve_paper_mode({"mode": "live", "allow_live_trading": True}) is False


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        resolve_paper_mode({"mode": "yolo"})


# ------------------------------- reconciliation --------------------------------

def test_reconcile_ok():
    broker = MockBroker(price_map={"AAPL": 100.0}, positions={"AAPL": 10})
    report = reconcile(broker, {"AAPL": 10.0}, expected_cash=broker.cash)
    assert report["ok"]


def test_reconcile_detects_missing_and_unexpected():
    broker = MockBroker(price_map={"AAPL": 100.0, "TSLA": 200.0}, positions={"TSLA": 3})
    report = reconcile(broker, {"AAPL": 10.0})
    assert not report["ok"]
    assert report["missing_fills"] == ["AAPL"]
    assert report["unexpected_holdings"] == ["TSLA"]


def test_reconcile_detects_partial_fill_qty_mismatch():
    broker = MockBroker(price_map={"AAPL": 100.0}, partial_fills={"AAPL": 0.5})
    ex = Executor(broker, LOG, poll_interval_seconds=0.01, poll_timeout_seconds=1)
    ex.execute([Order("AAPL", 10, "buy")], dry_run=False, run_id="t4")
    report = reconcile(broker, {"AAPL": 10.0})
    assert not report["ok"]
    assert report["qty_mismatches"]["AAPL"]["actual"] == pytest.approx(5.0)
    assert report["partial_fills"]


def test_reconcile_detects_cash_difference():
    broker = MockBroker(price_map={"AAPL": 100.0})
    report = reconcile(broker, {}, expected_cash=broker.cash - 500)
    assert not report["ok"] and report["cash_difference"] == pytest.approx(500)
