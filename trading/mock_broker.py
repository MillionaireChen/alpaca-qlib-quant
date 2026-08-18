"""In-memory mock broker for unit tests and offline dry-runs.

Fills market orders immediately at prices supplied via ``price_map`` (unless a
symbol is listed in ``reject_symbols`` or ``partial_fills``). No network access.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timezone

from trading.broker import Account, Broker, BrokerOrder, Order, Position


class MockBroker(Broker):
    name = "mock"

    def __init__(
        self,
        cash: float = 100_000.0,
        price_map: dict[str, float] | None = None,
        positions: dict[str, float] | None = None,
        reject_symbols: set[str] | None = None,
        partial_fills: dict[str, float] | None = None,
    ) -> None:
        self.cash = float(cash)
        self.price_map = dict(price_map or {})
        self._qty: dict[str, float] = dict(positions or {})
        self.reject_symbols = set(reject_symbols or ())
        self.partial_fills = dict(partial_fills or {})  # symbol -> fill fraction
        self._orders: dict[str, BrokerOrder] = {}
        self._seq = itertools.count(1)
        self._client_ids: set[str] = set()

    # --- helpers ---------------------------------------------------------
    def price(self, symbol: str) -> float:
        if symbol not in self.price_map:
            raise KeyError(f"MockBroker has no price for {symbol}")
        return self.price_map[symbol]

    def equity(self) -> float:
        return self.cash + sum(q * self.price(s) for s, q in self._qty.items() if q)

    # --- Broker interface --------------------------------------------------
    def get_account(self) -> Account:
        eq = self.equity()
        return Account(equity=eq, cash=self.cash, buying_power=self.cash, status="ACTIVE")

    def get_positions(self) -> list[Position]:
        out = []
        for s, q in self._qty.items():
            if abs(q) < 1e-9:
                continue
            px = self.price(s)
            out.append(Position(symbol=s, qty=q, market_value=q * px, current_price=px))
        return out

    def get_orders(self, open_only: bool = True) -> list[BrokerOrder]:
        orders = list(self._orders.values())
        if open_only:
            orders = [o for o in orders if not o.is_terminal]
        return orders

    def submit_order(self, order: Order) -> BrokerOrder:
        if order.client_order_id:
            if order.client_order_id in self._client_ids:
                raise ValueError(f"duplicate client_order_id: {order.client_order_id}")
            self._client_ids.add(order.client_order_id)

        oid = f"mock-{next(self._seq)}"
        now = datetime.now(timezone.utc)
        if order.symbol in self.reject_symbols:
            bo = BrokerOrder(oid, order.symbol, order.side, order.qty, 0.0, "rejected",
                             order.client_order_id, None, now)
            self._orders[oid] = bo
            return bo

        fill_fraction = self.partial_fills.get(order.symbol, 1.0)
        filled_qty = order.qty * fill_fraction
        px = self.price(order.symbol)
        signed = filled_qty if order.side == "buy" else -filled_qty
        self._qty[order.symbol] = self._qty.get(order.symbol, 0.0) + signed
        self.cash -= signed * px
        status = "filled" if fill_fraction >= 1.0 else "partially_filled"
        bo = BrokerOrder(oid, order.symbol, order.side, order.qty, filled_qty, status,
                         order.client_order_id, px, now)
        self._orders[oid] = bo
        return bo

    def cancel_order(self, order_id: str) -> None:
        o = self._orders[order_id]
        if not o.is_terminal:
            o.status = "canceled"

    def get_order(self, order_id: str) -> BrokerOrder:
        return self._orders[order_id]
