"""Unit tests for custom factors: hand-computed values and no-look-ahead guarantees."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factors import PANDAS_FACTORS  # noqa: E402


def make_ohlcv(close: np.ndarray, volume: np.ndarray | None = None) -> pd.DataFrame:
    n = len(close)
    if volume is None:
        volume = np.full(n, 1_000_000.0)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": volume},
        index=idx,
    )


def random_ohlcv(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.02, n))
    volume = rng.uniform(0.5e6, 5e6, n)
    return make_ohlcv(close, volume)


def test_momentum_hand_computed():
    df = make_ohlcv(np.arange(1.0, 11.0))  # 1..10
    mom5 = PANDAS_FACTORS["MOM5"](df)
    assert mom5.iloc[-1] == pytest.approx(10.0 / 5.0 - 1.0)
    assert np.isnan(mom5.iloc[4])  # needs 5 past days


def test_reversal_is_negative_one_day_return():
    df = make_ohlcv(np.array([100.0, 110.0, 99.0]))
    rev = PANDAS_FACTORS["REV1"](df)
    assert rev.iloc[1] == pytest.approx(-0.10)
    assert rev.iloc[2] == pytest.approx(-(99.0 / 110.0 - 1.0))


def test_rsi_extremes():
    up = make_ohlcv(np.linspace(100, 200, 40))
    down = make_ohlcv(np.linspace(200, 100, 40))
    assert PANDAS_FACTORS["RSI14"](up).iloc[-1] > 99.0
    assert PANDAS_FACTORS["RSI14"](down).iloc[-1] < 1.0


def test_realized_vol_zero_for_constant_growth():
    df = make_ohlcv(100.0 * 1.01 ** np.arange(60))
    vol = PANDAS_FACTORS["VOL20"](df)
    assert vol.iloc[-1] == pytest.approx(0.0, abs=1e-12)


def test_volume_ratio_constant_volume_is_one():
    df = random_ohlcv()
    df["volume"] = 2e6
    vr = PANDAS_FACTORS["VR20"](df)
    assert vr.iloc[-1] == pytest.approx(1.0, rel=1e-6)


def test_ma_deviation_constant_price_is_zero():
    df = make_ohlcv(np.full(60, 42.0))
    md = PANDAS_FACTORS["MADEV20"](df)
    assert md.iloc[-1] == pytest.approx(0.0, abs=1e-12)


def test_macd_constant_price_is_zero():
    df = make_ohlcv(np.full(80, 55.0))
    assert PANDAS_FACTORS["MACD"](df).iloc[-1] == pytest.approx(0.0, abs=1e-12)
    assert PANDAS_FACTORS["MACDSIG"](df).iloc[-1] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("name", list(PANDAS_FACTORS))
def test_no_lookahead(name):
    """Factor value at date t must not change when future data changes."""
    df = random_ohlcv(n=120)
    t = 70
    full = PANDAS_FACTORS[name](df).iloc[: t + 1]
    truncated = PANDAS_FACTORS[name](df.iloc[: t + 1])
    pd.testing.assert_series_equal(full, truncated, check_names=False)


@pytest.mark.parametrize("name", list(PANDAS_FACTORS))
def test_output_index_matches_input(name):
    df = random_ohlcv()
    out = PANDAS_FACTORS[name](df)
    assert out.index.equals(df.index)
