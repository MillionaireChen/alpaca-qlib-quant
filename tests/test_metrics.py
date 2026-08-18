"""Unit tests for performance metrics, IC analysis, label config, and embargo logic."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtests import metrics as M  # noqa: E402
from core.handlers import apply_embargo, label_config  # noqa: E402


def test_cumulative_and_annualized_return():
    r = pd.Series([0.01] * 252)
    assert M.cumulative_return(r) == pytest.approx(1.01**252 - 1)
    assert M.annualized_return(r) == pytest.approx(1.01**252 - 1)


def test_max_drawdown_hand_computed():
    r = pd.Series([0.10, -0.50, 0.20])
    # curve: 1.10, 0.55, 0.66 ; peak 1.10 -> mdd = 0.55/1.10 - 1 = -0.5
    assert M.max_drawdown(r) == pytest.approx(-0.5)


def test_sharpe_zero_vol_is_nan():
    assert np.isnan(M.sharpe_ratio(pd.Series([0.01] * 50)))


def test_win_rate():
    assert M.win_rate(pd.Series([0.01, -0.01, 0.02, 0.0])) == pytest.approx(0.5)


def test_ic_analysis_perfect_correlation():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-01", "2020-01-02"]), list("ABCDE")], names=["datetime", "instrument"]
    )
    rng = np.random.default_rng(0)
    label = pd.Series(rng.normal(size=10), index=idx)
    daily, summary = M.ic_analysis(label.copy(), label)  # score == label
    assert summary["ic_mean"] == pytest.approx(1.0)
    assert summary["rank_ic_mean"] == pytest.approx(1.0)
    assert summary["n_days"] == 2


def test_ic_analysis_anti_correlation():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-01"]), list("ABCD")], names=["datetime", "instrument"]
    )
    label = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    _, summary = M.ic_analysis(-label, label)
    assert summary["rank_ic_mean"] == pytest.approx(-1.0)


def test_label_config_expression():
    fields, names = label_config(5)
    assert fields == ["Ref($close, -5)/Ref($close, -1) - 1"]
    assert names == ["LABEL0"]
    with pytest.raises(ValueError):
        label_config(0)


def test_apply_embargo_shifts_train_and_valid_ends_only():
    segments = {"train": ("2008-01-01", "2016-12-31"),
                "valid": ("2017-01-01", "2018-12-31"),
                "test": ("2019-01-01", "2020-11-10")}
    out = apply_embargo(segments, 10)
    assert out["train"] == ("2008-01-01", "2016-12-21")
    assert out["valid"] == ("2017-01-01", "2018-12-21")
    assert out["test"] == segments["test"]  # test end untouched
    assert apply_embargo(segments, 0) == segments
