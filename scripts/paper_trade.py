"""Daily paper-trading run: predictions -> target Top-K portfolio -> diff orders -> Alpaca paper.

Usage (ALWAYS start with dry-run):
    uv run python scripts/paper_trade.py --config configs/paper_trading.yaml --dry-run
    uv run python scripts/paper_trade.py --config configs/paper_trading.yaml

Offline verification without broker access:
    uv run python scripts/paper_trade.py --config configs/paper_trading.yaml --dry-run --mock --ignore-staleness

Safety: paper endpoint only (live blocked in config); invalid/stale predictions
=> DO NOT TRADE; every submitted order is polled, never assumed filled.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from core.config import load_config, require
from core.experiment import latest_experiment_dir
from core.log import get_logger
from trading.executor import Executor
from trading.portfolio import compute_target_weights, generate_orders
from trading.risk import validate_orders, validate_predictions
from trading.state import append_history, paper_log_dir, save_run_record, save_state


def load_latest_predictions(model_cfg: dict) -> tuple[pd.Series, dt.date]:
    """Latest cross-section of scores from the most recent experiment's predictions."""
    exp_dir = latest_experiment_dir(model_cfg)
    path = exp_dir / "pred_test.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/predict.py first.")
    df = pd.read_csv(path, parse_dates=["datetime"])
    last_date = df["datetime"].max()
    cross = df[df["datetime"] == last_date].set_index("instrument")["score"]
    return cross, last_date.date()


def make_mock_broker(scores: pd.Series, model_cfg: dict):
    """MockBroker priced from the last available closes in the qlib data (offline testing)."""
    from core.qlib_init import init_qlib
    from trading.mock_broker import MockBroker

    init_qlib(model_cfg)
    from qlib.data import D

    cal = D.calendar(freq="day")
    closes = D.features(list(scores.index), ["$close"],
                        start_time=str(cal[-5].date()), end_time=str(cal[-1].date()))
    last_close = closes.groupby(level="instrument")["$close"].last()
    return MockBroker(cash=100_000.0, price_map={str(k): float(v) for k, v in last_close.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper_trading.yaml")
    parser.add_argument("--dry-run", action="store_true", help="propose orders but submit NOTHING")
    parser.add_argument("--mock", action="store_true", help="use in-memory MockBroker (no network)")
    parser.add_argument("--ignore-staleness", action="store_true",
                        help="allow trading on old predictions (offline testing only)")
    parser.add_argument("--force", action="store_true",
                        help="allow re-running a rebalance already executed for this prediction date")
    args = parser.parse_args()

    log = get_logger("paper_trade")
    cfg = load_config(args.config)
    model_cfg = load_config(require(cfg, "signal.model_config"))
    log_dir = paper_log_dir(cfg)

    scores, prediction_date = load_latest_predictions(model_cfg)
    log.info("Loaded %d scores for prediction date %s", len(scores), prediction_date)

    # --- kill-switch: invalid predictions => DO NOT TRADE -------------------
    check = validate_predictions(scores, prediction_date, require(cfg, "signal"), require(cfg, "risk"),
                                 ignore_staleness=args.ignore_staleness)
    if not check.ok:
        for r in check.reasons:
            log.error("PREDICTION CHECK FAILED: %s", r)
        log.error("DO NOT TRADE — aborting run.")
        save_run_record(log_dir, {"aborted": True, "stage": "prediction_validation",
                                  "reasons": check.reasons, "prediction_date": prediction_date})
        raise SystemExit(2)

    # --- duplicate-rebalance protection -------------------------------------
    from trading.state import load_state

    state = load_state(log_dir)
    if (not args.dry_run and not args.force and state
            and state.get("prediction_date") == str(prediction_date)):
        log.error("Rebalance for prediction date %s was already executed (see %s). "
                  "Use --force to override.", prediction_date, log_dir / "state_latest.json")
        raise SystemExit(3)

    # --- broker --------------------------------------------------------------
    if args.mock:
        broker = make_mock_broker(scores, model_cfg)
        price_of = lambda s: broker.price_map.get(s)  # noqa: E731
    else:
        from trading.alpaca_paper import AlpacaBroker, AlpacaPriceProvider

        broker = AlpacaBroker(require(cfg, "broker"))
        price_of = AlpacaPriceProvider()
    log.info("Broker: %s", broker.name)

    account = broker.get_account()
    positions = broker.get_positions()
    open_orders = broker.get_orders(open_only=True)
    log.info("Account: equity=%.2f cash=%.2f buying_power=%.2f | %d positions | %d open orders",
             account.equity, account.cash, account.buying_power, len(positions), len(open_orders))

    # --- target portfolio and diff orders ------------------------------------
    target = compute_target_weights(scores, int(require(cfg, "strategy.topk")), require(cfg, "portfolio"))
    log.info("Target portfolio (%d names): %s", len(target),
             {k: round(v, 4) for k, v in sorted(target.items())})
    orders, notes = generate_orders(target, positions, account, price_of, require(cfg, "portfolio"))
    for n in notes:
        log.warning("order-gen: %s", n)
    if not orders:
        log.info("Portfolio already on target — no orders needed.")
        save_run_record(log_dir, {"prediction_date": prediction_date, "target": target,
                                  "orders": [], "note": "already on target"})
        return

    # --- pre-trade risk validation -------------------------------------------
    check = validate_orders(orders, account, positions, open_orders, require(cfg, "risk"), price_of)
    if not check.ok:
        for r in check.reasons:
            log.error("PRE-TRADE CHECK FAILED: %s", r)
        log.error("DO NOT TRADE — aborting run.")
        save_run_record(log_dir, {"aborted": True, "stage": "order_validation", "reasons": check.reasons,
                                  "prediction_date": prediction_date, "target": target,
                                  "proposed_orders": [vars(o) for o in orders]})
        raise SystemExit(2)

    # --- execute --------------------------------------------------------------
    exec_cfg = cfg.get("execution", {})
    executor = Executor(broker, log,
                        poll_interval_seconds=float(exec_cfg.get("poll_interval_seconds", 2)),
                        poll_timeout_seconds=float(exec_cfg.get("poll_timeout_seconds", 120)))
    run_id = f"pt-{prediction_date:%Y%m%d}"
    report = executor.execute(orders, dry_run=args.dry_run, run_id=run_id)

    # --- record ----------------------------------------------------------------
    account_after = broker.get_account()
    positions_after = {p.symbol: p.qty for p in broker.get_positions()}
    turnover = sum(float(o["qty"]) * (price_of(o["symbol"]) or 0.0) for o in report.proposed) / max(account.equity, 1)
    record = {
        "run_id": run_id,
        "dry_run": args.dry_run,
        "broker": broker.name,
        "prediction_date": prediction_date,
        "model_experiment": str(latest_experiment_dir(model_cfg)),
        "strategy": require(cfg, "strategy"),
        "selected": sorted(target),
        "target_weights": target,
        "positions_before": {p.symbol: p.qty for p in positions},
        "execution": report.to_dict(),
        "account_before": vars(account) | {"raw": None},
        "account_after": vars(account_after) | {"raw": None},
        "positions_after": positions_after,
        "estimated_turnover": turnover,
    }
    path = save_run_record(log_dir, record)
    log.info("Run record written to %s", path)

    if not args.dry_run:
        save_state(log_dir, {"prediction_date": str(prediction_date), "run_id": run_id,
                             "expected_positions": positions_after,
                             "expected_cash": account_after.cash,
                             "recorded_at": dt.datetime.now().isoformat(timespec="seconds")})
        n_rejected = sum(1 for s in report.submitted if s.get("status") in ("rejected", "expired", "canceled"))
        append_history(log_dir, {
            "date": str(dt.date.today()), "equity": account_after.equity, "cash": account_after.cash,
            "buying_power": account_after.buying_power, "n_positions": len(positions_after),
            "gross_exposure": (account_after.equity - account_after.cash) / max(account_after.equity, 1),
            "unrealized_pl": None, "turnover": turnover,
            "orders_submitted": len(report.submitted), "orders_rejected": n_rejected,
        })
        log.info("State + history updated. Run scripts/reconcile.py to verify against the broker.")


if __name__ == "__main__":
    main()
