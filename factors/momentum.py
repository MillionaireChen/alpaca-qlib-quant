"""Momentum and short-term reversal factors."""

from __future__ import annotations

import pandas as pd


def momentum(ohlcv: pd.DataFrame, window: int) -> pd.Series:
    """N-day price momentum: close[t] / close[t-window] - 1."""
    close = ohlcv["close"]
    return close / close.shift(window) - 1


def momentum_5(ohlcv: pd.DataFrame) -> pd.Series:
    return momentum(ohlcv, 5)


def momentum_20(ohlcv: pd.DataFrame) -> pd.Series:
    return momentum(ohlcv, 20)


def momentum_60(ohlcv: pd.DataFrame) -> pd.Series:
    return momentum(ohlcv, 60)


def reversal_1(ohlcv: pd.DataFrame) -> pd.Series:
    """Short-term (1-day) reversal: negative of yesterday-to-today return."""
    return -momentum(ohlcv, 1)
