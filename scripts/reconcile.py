"""Reconcile expected local state against actual broker state (broker wins).

Usage:
    uv run python scripts/reconcile.py [--config configs/paper_trading.yaml]

Exit codes: 0 = consistent, 2 = discrepancies found (see printed report).
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from core.config import load_config, require
from core.log import get_logger
from trading.reconciliation import reconcile
from trading.state import load_state, paper_log_dir, save_run_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper_trading.yaml")
    parser.add_argument("--stale-minutes", type=int, default=60)
    args = parser.parse_args()

    log = get_logger("reconcile")
    cfg = load_config(args.config)
    log_dir = paper_log_dir(cfg)

    state = load_state(log_dir)
    if state is None:
        log.warning("No local state (%s) — reconciling against an empty expectation.", log_dir / "state_latest.json")
        expected_positions, expected_cash = {}, None
    else:
        expected_positions = {k: float(v) for k, v in state.get("expected_positions", {}).items()}
        expected_cash = state.get("expected_cash")
        log.info("Expected state from run %s (prediction date %s)", state.get("run_id"), state.get("prediction_date"))

    from trading.alpaca_paper import AlpacaBroker

    broker = AlpacaBroker(require(cfg, "broker"))
    report = reconcile(broker, expected_positions, expected_cash, stale_order_minutes=args.stale_minutes)

    print(json.dumps(report, indent=2, default=str))
    save_run_record(log_dir, {"type": "reconciliation", **report})

    if report["ok"]:
        log.info("Reconciliation OK — broker state matches expectations.")
        raise SystemExit(0)
    log.error("Reconciliation found discrepancies (broker state is authoritative).")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
