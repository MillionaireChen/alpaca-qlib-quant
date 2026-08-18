"""LightGBM baseline model with Rank-IC-based early stopping.

Cross-sectional stock selection cares about daily ranking quality, not MSE, so
early stopping maximizes the mean daily Rank IC on the validation segment.

Interface contract (future models must match):
    fit(dataset) -> None
    predict(dataset, segment) -> pd.Series indexed by (datetime, instrument), name="score"
    save(path) / load(path)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

_LOSS_TO_OBJECTIVE = {"mse": "regression", "binary": "binary"}


def make_rank_ic_feval(index: pd.MultiIndex, y: np.ndarray) -> Callable:
    """Build a LightGBM feval computing mean daily Rank IC (Spearman) against labels ``y``.

    Ranks are standardized per day, so the mean per-day product of the two
    standardized rank vectors equals the per-day Spearman correlation.
    """
    days = index.get_level_values("datetime")
    lab_rank = pd.Series(np.asarray(y, dtype=float), index=days).groupby(level=0).rank()
    grp = lab_rank.groupby(level=0)
    lab_z = (lab_rank - grp.transform("mean")) / grp.transform("std")

    def feval(preds: np.ndarray, _: lgb.Dataset) -> tuple[str, float, bool]:
        pr = pd.Series(preds, index=days).groupby(level=0).rank()
        g = pr.groupby(level=0)
        pz = (pr - g.transform("mean")) / g.transform("std")
        daily_ic = (pz * lab_z).groupby(level=0).mean()
        return "rank_ic", float(daily_ic.mean()), True  # higher is better

    return feval


class LightGBMModel:
    """Cross-sectional return prediction with LightGBM (higher score = stronger expected return)."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        early_stopping_rounds: int = 100,
        num_boost_round: int = 800,
    ) -> None:
        params = dict(params or {})
        loss = str(params.pop("loss", "mse"))
        if loss not in _LOSS_TO_OBJECTIVE:
            raise ValueError(f"Unsupported loss: {loss!r}")
        params.setdefault("objective", _LOSS_TO_OBJECTIVE[loss])
        params.setdefault("seed", seed)
        params.setdefault("verbosity", -1)
        self.params: dict[str, Any] = params
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.booster: lgb.Booster | None = None
        self.feature_names: list[str] | None = None
        self.evals_result: dict[str, Any] = {}

    @staticmethod
    def _prepare(dataset: DatasetH, segment: str) -> tuple[pd.DataFrame, pd.Series]:
        data = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        return data["feature"], data["label"].iloc[:, 0]

    def fit(self, dataset: DatasetH) -> None:
        """Train on 'train'; early-stop on mean daily Rank IC over 'valid' (if present)."""
        x_train, y_train = self._prepare(dataset, "train")
        self.feature_names = list(x_train.columns)
        dtrain = lgb.Dataset(x_train.values, label=y_train.values)

        valid_sets, callbacks, feval = [], [lgb.record_evaluation(self.evals_result)], None
        if "valid" in dataset.segments:
            x_valid, y_valid = self._prepare(dataset, "valid")
            valid_sets = [lgb.Dataset(x_valid.values, label=y_valid.values, reference=dtrain)]
            feval = make_rank_ic_feval(x_valid.index, y_valid.values)
            callbacks.append(
                lgb.early_stopping(self.early_stopping_rounds, first_metric_only=False, verbose=False)
            )

        self.booster = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=["valid"] if valid_sets else None,
            feval=feval,
            callbacks=callbacks,
        )

    @property
    def best_iteration(self) -> int | None:
        return self.booster.best_iteration if self.booster is not None else None

    @property
    def best_valid_rank_ic(self) -> float | None:
        curve = self.evals_result.get("valid", {}).get("rank_ic")
        if not curve or not self.best_iteration:
            return None
        return float(curve[self.best_iteration - 1])

    def predict(self, dataset: DatasetH, segment: str = "test") -> pd.Series:
        if self.booster is None:
            raise RuntimeError("Model not fitted/loaded.")
        x = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        scores = self.booster.predict(x.values, num_iteration=self.best_iteration or None)
        return pd.Series(scores, index=x.index, name="score")

    def save(self, path: str | Path) -> None:
        if self.booster is None:
            raise RuntimeError("Nothing to save: model not fitted.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": self.params,
            "booster": self.booster,
            "feature_names": self.feature_names,
            "early_stopping_rounds": self.early_stopping_rounds,
            "num_boost_round": self.num_boost_round,
            "evals_result": self.evals_result,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMModel":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        obj = cls.__new__(cls)
        obj.params = payload["params"]
        obj.booster = payload["booster"]
        obj.feature_names = payload["feature_names"]
        obj.early_stopping_rounds = payload["early_stopping_rounds"]
        obj.num_boost_round = payload["num_boost_round"]
        obj.evals_result = payload.get("evals_result", {})
        return obj
