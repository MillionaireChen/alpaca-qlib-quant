"""Volume factors."""

from __future__ import annotations

import pandas as pd

_EPS = 1e-12


def volume_ratio(ohlcv: pd.DataFrame, window: int) -> pd.Series:
    """Today's volume relative to its rolling mean."""
    volume = ohlcv["volume"]
    return volume / (volume.rolling(window).mean() + _EPS)


def volume_ratio_20(ohlcv: pd.DataFrame) -> pd.Series:
    return volume_ratio(ohlcv, 20)
