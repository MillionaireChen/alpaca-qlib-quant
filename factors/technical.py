"""Technical indicator factors: RSI, MACD, moving-average deviation."""

from __future__ import annotations

import pandas as pd

_EPS = 1e-12


def rsi(ohlcv: pd.DataFrame, window: int) -> pd.Series:
    """RSI with simple (equal-weight) rolling means of up/down moves.

    Matches the qlib expression in factors.QLIB_FEATURES (Cutler's RSI),
    not Wilder's exponentially smoothed variant.
    """
    delta = ohlcv["close"].diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta).clip(lower=0).rolling(window).mean()
    return 100 - 100 / (1 + up / (down + _EPS))


def rsi_14(ohlcv: pd.DataFrame) -> pd.Series:
    return rsi(ohlcv, 14)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd_line(ohlcv: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """MACD line normalized by price (comparable across stocks)."""
    close = ohlcv["close"]
    return (_ema(close, fast) - _ema(close, slow)) / close


def macd_signal(ohlcv: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Signal line: EMA of the normalized MACD line."""
    return _ema(macd_line(ohlcv, fast, slow), signal)


def ma_deviation(ohlcv: pd.DataFrame, window: int) -> pd.Series:
    """Deviation of price from its rolling moving average."""
    close = ohlcv["close"]
    return close / close.rolling(window).mean() - 1


def ma_deviation_20(ohlcv: pd.DataFrame) -> pd.Series:
    return ma_deviation(ohlcv, 20)
