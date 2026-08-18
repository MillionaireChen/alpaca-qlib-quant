"""Pure performance-metric functions (unit-testable, no qlib dependency).

All functions take daily return series (in decimal, e.g. 0.01 = 1%).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def cumulative_return(returns: pd.Series) -> float:
    """Total compounded return over the period."""
    return float((1.0 + returns).prod() - 1.0)


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Geometric annualized return."""
    n = len(returns)
    if n == 0:
        return float("nan")
    total = (1.0 + returns).prod()
    return float(total ** (periods / n) - 1.0)


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Sharpe ratio on daily returns (rf given as annual rate)."""
    excess = returns - risk_free / periods
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (negative number, e.g. -0.25 = -25%)."""
    curve = (1.0 + returns).cumprod()
    peak = curve.cummax()
    return float((curve / peak - 1.0).min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    curve = (1.0 + returns).cumprod()
    return curve / curve.cummax() - 1.0


def win_rate(returns: pd.Series) -> float:
    """Fraction of days with positive return."""
    if len(returns) == 0:
        return float("nan")
    return float((returns > 0).mean())


def ic_analysis(pred: pd.Series, label: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Daily cross-sectional IC (Pearson) and Rank IC (Spearman) between score and realized label."""
    df = pd.concat({"score": pred, "label": label}, axis=1).dropna()
    by_day = df.groupby(level="datetime")
    ic = by_day.apply(lambda x: x["score"].corr(x["label"]))
    rank_ic = by_day.apply(lambda x: x["score"].corr(x["label"], method="spearman"))
    daily = pd.DataFrame({"ic": ic, "rank_ic": rank_ic})
    summary = {
        "ic_mean": float(daily["ic"].mean()),
        "ic_std": float(daily["ic"].std()),
        "icir": float(daily["ic"].mean() / daily["ic"].std()) if daily["ic"].std() else None,
        "rank_ic_mean": float(daily["rank_ic"].mean()),
        "rank_ic_std": float(daily["rank_ic"].std()),
        "rank_icir": float(daily["rank_ic"].mean() / daily["rank_ic"].std()) if daily["rank_ic"].std() else None,
        "ic_positive_ratio": float((daily["ic"] > 0).mean()),
        "n_days": int(len(daily)),
        "n_obs": int(len(df)),
    }
    return daily, summary


def summarize_returns(returns: pd.Series, prefix: str = "") -> dict[str, float]:
    """Standard metric block for a daily return series."""
    return {
        f"{prefix}cumulative_return": cumulative_return(returns),
        f"{prefix}annualized_return": annualized_return(returns),
        f"{prefix}annualized_volatility": annualized_volatility(returns),
        f"{prefix}sharpe_ratio": sharpe_ratio(returns),
        f"{prefix}max_drawdown": max_drawdown(returns),
        f"{prefix}win_rate": win_rate(returns),
    }
