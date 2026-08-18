"""Alpaca broker adapter (paper trading by default, live hard-blocked).

Credentials come ONLY from environment variables (or a local .env file):
    ALPACA_API_KEY / ALPACA_SECRET_KEY
Secret values are never logged and never included in error messages.
"""

from __future__ import annotations

import os
from typing import Any

from core.config import REPO_ROOT
from trading.broker import Account, Broker, BrokerOrder, Order, Position


class LiveTradingBlockedError(RuntimeError):
    """Raised when a live-trading configuration is requested without the explicit override."""


def _load_credentials() -> tuple[str, str]:
    """Fetch API credentials from env, loading .env if present. Values are never logged."""
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file)
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError(
            "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY. "
            "Copy .env.example to .env and fill in your Alpaca *paper* keys."
        )
    return key, secret


def resolve_paper_mode(broker_cfg: dict[str, Any]) -> bool:
    """Return True for paper mode; refuse live mode unless explicitly overridden.

    There is NO automatic fallback between paper and live in either direction.
    """
    mode = str(broker_cfg.get("mode", "paper")).lower()
    allow_live = bool(broker_cfg.get("allow_live_trading", False))
    if mode == "paper":
        return True
    if mode == "live":
        if not allow_live:
            raise LiveTradingBlockedError(
                "broker.mode is 'live' but allow_live_trading is false — refusing to start. "
                "This project defaults to PAPER trading; enable the override only if you "
                "fully understand the consequences."
            )
        return False
    raise ValueError(f"Unknown broker.mode: {mode!r} (expected 'paper' or 'live')")


class AlpacaBroker(Broker):
    """Alpaca implementation of the Broker interface via alpaca-py."""

    def __init__(self, broker_cfg: dict[str, Any]) -> None:
        if str(broker_cfg.get("provider", "alpaca")).lower() != "alpaca":
            raise ValueError(f"AlpacaBroker got provider={broker_cfg.get('provider')!r}")
        paper = resolve_paper_mode(broker_cfg)
        key, secret = _load_credentials()

        from alpaca.trading.client import TradingClient

        self._client = TradingClient(key, secret, paper=paper)
        self.paper = paper
        self.name = f"alpaca-{'paper' if paper else 'LIVE'}"

    # --- mapping helpers -------------------------------------------------
    @staticmethod
    def _to_broker_order(o: Any) -> BrokerOrder:
        return BrokerOrder(
            id=str(o.id),
            symbol=str(o.symbol),
            side=str(getattr(o.side, "value", o.side)).lower(),
            qty=float(o.qty) if o.qty is not None else 0.0,
            filled_qty=float(o.filled_qty or 0),
            status=str(getattr(o.status, "value", o.status)).lower(),
            client_order_id=getattr(o, "client_order_id", None),
            filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
            submitted_at=getattr(o, "submitted_at", None),
        )

    # --- Broker interface -------------------------------------------------
    def get_account(self) -> Account:
        a = self._client.get_account()
        return Account(
            equity=float(a.equity),
            cash=float(a.cash),
            buying_power=float(a.buying_power),
            currency=str(a.currency or "USD"),
            status=str(getattr(a.status, "value", a.status)),
        )

    def get_positions(self) -> list[Position]:
        out = []
        for p in self._client.get_all_positions():
            out.append(
                Position(
                    symbol=str(p.symbol),
                    qty=float(p.qty),
                    market_value=float(p.market_value or 0),
                    avg_entry_price=float(p.avg_entry_price) if p.avg_entry_price else None,
                    current_price=float(p.current_price) if p.current_price else None,
                    unrealized_pl=float(p.unrealized_pl) if p.unrealized_pl else None,
                )
            )
        return out

    def get_orders(self, open_only: bool = True) -> list[BrokerOrder]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        status = QueryOrderStatus.OPEN if open_only else QueryOrderStatus.ALL
        orders = self._client.get_orders(GetOrdersRequest(status=status, limit=500))
        return [self._to_broker_order(o) for o in orders]

    def submit_order(self, order: Order) -> BrokerOrder:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        if order.order_type != "market":
            raise NotImplementedError("Only market orders are supported in the baseline.")
        req = MarketOrderRequest(
            symbol=order.symbol,
            qty=order.qty,
            side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY if order.time_in_force == "day" else TimeInForce(order.time_in_force),
            client_order_id=order.client_order_id,
        )
        return self._to_broker_order(self._client.submit_order(req))

    def cancel_order(self, order_id: str) -> None:
        self._client.cancel_order_by_id(order_id)

    def get_order(self, order_id: str) -> BrokerOrder:
        return self._to_broker_order(self._client.get_order_by_id(order_id))


class AlpacaPriceProvider:
    """Latest-trade price lookup via the Alpaca market data API.

    Returns None when a price is unavailable — callers must then SKIP the
    symbol rather than trade blind.
    """

    def __init__(self) -> None:
        key, secret = _load_credentials()
        from alpaca.data.historical import StockHistoricalDataClient

        self._client = StockHistoricalDataClient(key, secret)

    def __call__(self, symbol: str) -> float | None:
        try:
            from alpaca.data.requests import StockLatestTradeRequest

            trades = self._client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
            return float(trades[symbol].price)
        except Exception:  # noqa: BLE001 - unavailable price => skip symbol upstream
            return None
