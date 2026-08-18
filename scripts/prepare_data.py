"""Download and verify the qlib US daily data bundle.

Usage:
    uv run python scripts/prepare_data.py --config configs/lightgbm_alpha158.yaml

Idempotent: skips the download when data already exists (use --force to redo).

Data facts (documented per project spec):
  * daily OHLCV, split+dividend adjusted prices (Yahoo-normalized), $factor included;
  * `nasdaq100`/`sp500` instrument files carry point-in-time membership dates,
    which limits (but does not fully remove) survivorship bias;
  * bundle coverage ends 2020-11-10; refreshing to current data requires an
    external market-data source (e.g. Alpaca) and network access.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from core.config import REPO_ROOT, load_config, require
from core.log import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lightgbm_alpha158.yaml")
    parser.add_argument("--force", action="store_true", help="re-download even if data exists")
    args = parser.parse_args()

    log = get_logger("prepare_data")
    cfg = load_config(args.config)
    target = Path(require(cfg, "qlib.provider_uri"))
    if not target.is_absolute():
        target = REPO_ROOT / target
    region = str(require(cfg, "qlib.region")).lower()

    if (target / "calendars").exists() and not args.force:
        log.info("Qlib data already present at %s (use --force to re-download).", target)
    else:
        from qlib.tests.data import GetData

        log.info("Downloading qlib %s daily bundle to %s ...", region, target)
        GetData().qlib_data(target_dir=str(target), region=region, interval="1d", exists_skip=not args.force)
        for leftover in target.glob("*.zip"):
            leftover.unlink()

    # --- verification ---
    from core.qlib_init import init_qlib

    init_qlib(cfg)
    from qlib.data import D

    cal = D.calendar(freq="day")
    market = require(cfg, "universe.market")
    check_day = str(cal[-250].date())
    members = D.list_instruments(D.instruments(market), start_time=check_day, end_time=check_day, as_list=True)
    sample = D.features(members[:2], ["$open", "$high", "$low", "$close", "$volume"],
                        start_time=str(cal[-30].date()), end_time=str(cal[-1].date()))
    if sample.empty or sample.isna().all().any():
        raise RuntimeError("Data verification failed: sample OHLCV is empty or has all-NaN fields.")
    log.info("VERIFIED | calendar %s -> %s (%d days) | %s members on %s: %d | sample rows: %d",
             cal[0].date(), cal[-1].date(), len(cal), market, check_day, len(members), len(sample))


if __name__ == "__main__":
    main()
