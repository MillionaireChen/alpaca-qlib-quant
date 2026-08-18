"""Dataset/handler construction for the Alpha158 + LightGBM baseline.

Label convention (see project spec):
    return[t] = close[t+horizon] / close[t+1] - 1
i.e. the position is entered at the close of t+1, so a prediction made with
information up to t is never executed at prices already known at t.

Leakage controls:
  * processor statistics (if any) are fitted on the train segment only;
  * an embargo shifts train/valid segment ends earlier so forward-looking
    labels cannot overlap the following segment's period.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL
from qlib.data.dataset import DatasetH

from core.config import require

SEGMENT_NAMES = ("train", "valid", "test")


class Alpha158NoVWAP(Alpha158):
    """Alpha158 without VWAP-based features.

    The qlib US daily bundle does not ship a ``$vwap`` field, so the single
    VWAP price feature is removed (157 features remain). Everything else is
    identical to the stock Alpha158 handler; qlib source is not modified.
    """

    def get_feature_config(self):  # type: ignore[override]
        conf = {
            "kbar": {},
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW"]},
            "rolling": {},
        }
        return Alpha158DL.get_feature_config(conf)


class Alpha158PlusCustom(Alpha158NoVWAP):
    """Alpha158 (no VWAP) plus the project's custom factors from factors/."""

    def get_feature_config(self):  # type: ignore[override]
        fields, names = super().get_feature_config()
        from factors import get_qlib_features

        cfields, cnames = get_qlib_features()
        return list(fields) + cfields, list(names) + cnames


def label_config(horizon: int) -> tuple[list[str], list[str]]:
    """Qlib label expression for an h-day forward return entered at t+1 close."""
    if horizon < 1:
        raise ValueError(f"label horizon must be >= 1, got {horizon}")
    return [f"Ref($close, -{horizon})/Ref($close, -1) - 1"], ["LABEL0"]


def apply_embargo(segments: dict[str, tuple[str, str]], embargo_days: int) -> dict[str, tuple[str, str]]:
    """Shift train/valid segment ends earlier by ``embargo_days`` calendar days.

    The label at date t looks horizon+1 days into the future; without an
    embargo, labels near a segment boundary overlap the next segment's period.
    """
    if embargo_days <= 0:
        return dict(segments)
    out = dict(segments)
    for seg in ("train", "valid"):
        if seg in out:
            start, end = out[seg]
            shifted = dt.date.fromisoformat(str(end)[:10]) - dt.timedelta(days=embargo_days)
            out[seg] = (start, shifted.isoformat())
    return out


def build_handler(
    cfg: dict[str, Any],
    data_range: tuple[str, str] | None = None,
    fit_range: tuple[str, str] | None = None,
) -> Alpha158NoVWAP:
    """Build the feature handler (Alpha158, optionally + custom factors)."""
    start_time, end_time = data_range or (
        str(require(cfg, "data.start_time")),
        str(require(cfg, "data.end_time")),
    )
    ds = require(cfg, "dataset")
    fit_start, fit_end = fit_range or (str(ds["train"][0]), str(ds["train"][1]))
    use_custom = bool(cfg.get("features", {}).get("custom_factors", False))
    handler_cls = Alpha158PlusCustom if use_custom else Alpha158NoVWAP
    return handler_cls(
        instruments=require(cfg, "universe.market"),
        start_time=start_time,
        end_time=end_time,
        fit_start_time=fit_start,
        fit_end_time=fit_end,
        label=label_config(int(require(cfg, "label.horizon"))),
    )


def build_dataset(
    cfg: dict[str, Any],
    segments: dict[str, tuple[str, str]] | None = None,
    handler: Alpha158NoVWAP | None = None,
) -> DatasetH:
    """Build the DatasetH from config, with optional segment/handler overrides.

    Passing a pre-built ``handler`` (e.g. in walk-forward) is safe here because
    no processor in this pipeline fits statistics over time (DropnaLabel and
    CSZScoreNorm are per-date cross-sectional operations).
    """
    if segments is None:
        ds = require(cfg, "dataset")
        segments = {k: (str(ds[k][0]), str(ds[k][1])) for k in SEGMENT_NAMES if k in ds}
    if "train" not in segments:
        raise ValueError("dataset segments must include 'train'")
    embargo = int(cfg.get("dataset", {}).get("embargo_days", 0))
    segments = apply_embargo(segments, embargo)
    if handler is None:
        handler = build_handler(cfg, fit_range=segments["train"])
    return DatasetH(handler, dict(segments))
