"""Walk-forward evaluation: retrain per rolling window, test strictly out-of-sample.

Usage:
    uv run python scripts/walk_forward.py --config configs/lightgbm_alpha158.yaml

Each window trains a fresh model; no future samples ever influence earlier
models. The feature handler is shared across windows, which is safe because
no processor in this pipeline fits statistics over time (per-date
cross-sectional operations only); the embargo from the config is applied to
every window's train/valid ends.
"""

from __future__ import annotations

import argparse
import copy

import _bootstrap  # noqa: F401

import pandas as pd

from core.config import load_config, require
from core.experiment import create_experiment_dir, experiment_metadata, save_json
from core.handlers import build_dataset, build_handler
from core.log import get_logger
from core.qlib_init import init_qlib
from models import LightGBMModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    log = get_logger("walk_forward")
    cfg = load_config(args.config)
    init_qlib(cfg)

    from backtests.engine import run_backtest
    from backtests.metrics import ic_analysis, summarize_returns
    from backtests.report import build_report
    from qlib.data.dataset.handler import DataHandlerLP

    windows = require(cfg, "walk_forward.windows")
    wf_cfg = copy.deepcopy(cfg)
    wf_cfg["experiment"]["name"] = str(require(cfg, "experiment.name")) + "_walkforward"
    exp_dir = create_experiment_dir(wf_cfg)

    log.info("Building shared feature handler ...")
    handler = build_handler(cfg)

    rows: list[dict] = []
    reports: list[pd.DataFrame] = []
    for i, w in enumerate(windows, start=1):
        segments = {k: (str(w[k][0]), str(w[k][1])) for k in ("train", "valid", "test")}
        log.info("Window %d | train %s | valid %s | test %s", i, segments["train"], segments["valid"], segments["test"])
        ds = build_dataset(cfg, segments=segments, handler=handler)

        model = LightGBMModel(
            params=require(cfg, "model.params"),
            seed=int(require(cfg, "experiment.seed")),
            early_stopping_rounds=int(cfg["model"].get("early_stopping_rounds", 100)),
            num_boost_round=int(cfg["model"].get("num_boost_round", 800)),
        )
        model.fit(ds)
        pred = model.predict(ds, "test")
        label = ds.prepare("test", col_set="label", data_key=DataHandlerLP.DK_I).iloc[:, 0]
        ic_daily, ic = ic_analysis(pred, label)

        bt = run_backtest(cfg, pred, start_time=segments["test"][0], end_time=segments["test"][1])
        rep = bt["report"]
        net = rep["return"] - rep["cost"]

        wdir = exp_dir / f"window_{i}"
        wdir.mkdir(parents=True, exist_ok=True)
        pred.to_frame().to_csv(wdir / "pred_test.csv")
        ic_daily.to_csv(wdir / "ic_daily.csv")
        rep.to_csv(wdir / "report_daily.csv")

        row = {
            "window": i,
            "test_start": segments["test"][0],
            "test_end": segments["test"][1],
            "best_iteration": model.best_iteration,
            "rank_ic_mean": ic["rank_ic_mean"],
            "rank_icir": ic["rank_icir"],
            **summarize_returns(net, "net_"),
            **summarize_returns(rep["bench"], "bench_"),
            "mean_turnover": float(rep["turnover"].mean()),
        }
        rows.append(row)
        reports.append(rep)
        log.info("Window %d done | RankIC %.4f | net ann %.3f | bench ann %.3f | sharpe %.2f",
                 i, row["rank_ic_mean"], row["net_annualized_return"], row["bench_annualized_return"],
                 row["net_sharpe_ratio"])

    table = pd.DataFrame(rows)
    table.to_csv(exp_dir / "windows_summary.csv", index=False)

    stitched = pd.concat(reports).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    overall = build_report(stitched, exp_dir / "stitched")
    save_json({"windows": rows, "stitched_overall": overall}, exp_dir / "walkforward_summary.json")

    meta = experiment_metadata(cfg)
    meta["mode"] = "walk_forward"
    meta["windows"] = windows
    save_json(meta, exp_dir / "experiment.json")

    with pd.option_context("display.width", 200):
        log.info("Walk-forward summary:\n%s", table[[
            "window", "test_start", "test_end", "rank_ic_mean", "net_annualized_return",
            "bench_annualized_return", "net_sharpe_ratio", "net_max_drawdown", "mean_turnover",
        ]].round(4).to_string(index=False))
    log.info("Stitched OOS (net): ann=%.4f sharpe=%.3f maxdd=%.4f | results in %s",
             overall["portfolio_net_annualized_return"], overall["portfolio_net_sharpe_ratio"],
             overall["portfolio_net_max_drawdown"], exp_dir)


if __name__ == "__main__":
    main()
