"""Transparent Top-K strategy construction (wraps qlib's TopkDropoutStrategy)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

from core.config import require


def make_topk_strategy(cfg: dict[str, Any], signal: pd.Series) -> TopkDropoutStrategy:
    """Build a TopkDropoutStrategy from config.

    Logic per rebalance day: rank stocks by score, hold the top ``topk``,
    replacing at most ``n_drop`` per day (limits turnover).
    """
    strat = require(cfg, "strategy")
    return TopkDropoutStrategy(
        signal=signal,
        topk=int(strat["topk"]),
        n_drop=int(strat["n_drop"]),
        hold_thresh=int(strat.get("hold_thresh", 1)),
        only_tradable=False,
    )
