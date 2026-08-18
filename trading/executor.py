"""Order execution: dry-run printing or real submission with status polling.

Never assumes an order fills — every submitted order is polled until it
reaches a terminal status or the poll times out (leftover state is then
reported for reconciliation).
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from trading.broker import Broker, BrokerOrder, Order, TERMINAL_STATUSES


@dataclass
class ExecutionReport:
    dry_run: bool
    proposed: list[dict] = field(default_factory=list)
    submitted: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timed_out: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Executor:
    def __init__(
        self,
        broker: Broker,
        logger: logging.Logger,
        poll_interval_seconds: float = 2.0,
        poll_timeout_seconds: float = 120.0,
    ) -> None:
        self.broker = broker
        self.log = logger
        self.poll_interval = poll_interval_seconds
        self.poll_timeout = poll_timeout_seconds

    def execute(self, orders: list[Order], dry_run: bool, run_id: str) -> ExecutionReport:
        """Execute (or just print, when dry_run) the order batch. Sells are first by construction."""
        report = ExecutionReport(dry_run=dry_run)
        for o in orders:
            report.proposed.append(
                {"symbol": o.symbol, "side": o.side, "qty": o.qty,
                 "type": o.order_type, "tif": o.time_in_force}
            )
            self.log.info("%s order: %-4s %-6s qty=%s", "PROPOSED (dry-run)" if dry_run else "SUBMITTING",
                          o.side, o.symbol, o.qty)
        if dry_run:
            self.log.info("Dry-run: %d orders proposed, NONE submitted.", len(orders))
            return report

        submitted: list[BrokerOrder] = []
        for i, o in enumerate(orders):
            client_id = f"{run_id}-{i:03d}-{o.side}-{o.symbol}"
            try:
                bo = self.broker.submit_order(
                    Order(symbol=o.symbol, qty=o.qty, side=o.side, order_type=o.order_type,
                          time_in_force=o.time_in_force, client_order_id=client_id)
                )
                submitted.append(bo)
                self.log.info("submitted %s %s qty=%s -> id=%s status=%s", o.side, o.symbol, o.qty, bo.id, bo.status)
            except Exception as exc:  # noqa: BLE001 - broker errors must be surfaced, not crash the batch
                msg = f"submit failed for {o.side} {o.symbol} qty={o.qty}: {type(exc).__name__}: {exc}"
                self.log.error(msg)
                report.errors.append(msg)

        final = self._poll_until_terminal(submitted, report)
        report.submitted = [vars(bo) | {"submitted_at": str(bo.submitted_at)} for bo in final]
        return report

    def _poll_until_terminal(self, submitted: list[BrokerOrder], report: ExecutionReport) -> list[BrokerOrder]:
        deadline = time.monotonic() + self.poll_timeout
        pending = {bo.id for bo in submitted}
        latest = {bo.id: bo for bo in submitted}
        while pending and time.monotonic() < deadline:
            for oid in list(pending):
                try:
                    bo = self.broker.get_order(oid)
                except Exception as exc:  # noqa: BLE001
                    report.errors.append(f"poll failed for {oid}: {type(exc).__name__}: {exc}")
                    continue
                latest[oid] = bo
                if bo.status in TERMINAL_STATUSES:
                    self.log.info("order %s (%s %s) final status: %s filled=%s @ %s",
                                  oid, bo.side, bo.symbol, bo.status, bo.filled_qty, bo.filled_avg_price)
                    pending.discard(oid)
            if pending:
                time.sleep(self.poll_interval)
        for oid in pending:
            bo = latest[oid]
            self.log.warning("order %s (%s %s) NOT terminal after %.0fs: status=%s filled=%s — check reconcile.py",
                             oid, bo.side, bo.symbol, self.poll_timeout, bo.status, bo.filled_qty)
            report.timed_out.append(oid)
        return list(latest.values())
