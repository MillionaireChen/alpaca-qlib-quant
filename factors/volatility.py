"""Volatility factors."""

from __future__ import annotations

import pandas as pd


def realized_vol(ohlcv: pd.DataFrame, window: int) -> pd.Series:
    """Rolling standard deviation of daily simple returns."""
    returns = ohlcv["close"].pct_change()
    return returns.rolling(window).std()


def realized_vol_20(ohlcv: pd.DataFrame) -> pd.Series:
    return realized_vol(ohlcv, 20)
