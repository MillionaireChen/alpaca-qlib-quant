"""Live paper-trading layer: broker abstraction, portfolio construction, risk, execution, reconciliation.

Safety invariants (see project spec):
  * paper trading is the ONLY default mode; live endpoints are blocked unless
    `allow_live_trading: true` is explicitly set in the config;
  * invalid/stale/NaN-heavy predictions => DO NOT TRADE;
  * broker state is the source of truth; local state is never authoritative.
"""
