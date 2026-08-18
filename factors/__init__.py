"""Custom factors, independent of qlib internals.

Each factor exists in two equivalent forms:
  * a pure-pandas implementation ``f(ohlcv: pd.DataFrame) -> pd.Series`` where
    ``ohlcv`` is indexed by date with columns open/high/low/close/volume, and
    the value at date t uses ONLY data up to and including t (no look-ahead);
  * a qlib expression (QLIB_FEATURES) consumed by ``Alpha158PlusCustom``.
"""

from collections import OrderedDict

from factors.momentum import momentum_5, momentum_20, momentum_60, reversal_1
from factors.technical import ma_deviation_20, macd_line, macd_signal, rsi_14
from factors.volatility import realized_vol_20
from factors.volume import volume_ratio_20

PANDAS_FACTORS = OrderedDict(
    [
        ("MOM5", momentum_5),
        ("MOM20", momentum_20),
        ("MOM60", momentum_60),
        ("REV1", reversal_1),
        ("RSI14", rsi_14),
        ("MACD", macd_line),
        ("MACDSIG", macd_signal),
        ("VOL20", realized_vol_20),
        ("VR20", volume_ratio_20),
        ("MADEV20", ma_deviation_20),
    ]
)

QLIB_FEATURES = OrderedDict(
    [
        ("MOM5", "$close/Ref($close, 5) - 1"),
        ("MOM20", "$close/Ref($close, 20) - 1"),
        ("MOM60", "$close/Ref($close, 60) - 1"),
        ("REV1", "1 - $close/Ref($close, 1)"),  # == -(1-day return); qlib parser has no unary minus
        (
            "RSI14",
            "100 - 100/(1 + Mean(Greater($close-Ref($close, 1), 0), 14)"
            "/(Mean(Greater(Ref($close, 1)-$close, 0), 14) + 1e-12))",
        ),
        ("MACD", "(EMA($close, 12) - EMA($close, 26))/$close"),
        ("MACDSIG", "EMA((EMA($close, 12) - EMA($close, 26))/$close, 9)"),
        ("VOL20", "Std($close/Ref($close, 1) - 1, 20)"),
        ("VR20", "$volume/(Mean($volume, 20) + 1e-12)"),
        ("MADEV20", "$close/Mean($close, 20) - 1"),
    ]
)


def get_qlib_features() -> tuple[list[str], list[str]]:
    """(fields, names) for a qlib data loader."""
    return list(QLIB_FEATURES.values()), list(QLIB_FEATURES.keys())
