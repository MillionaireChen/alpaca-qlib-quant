**English** | [中文](README.zh-CN.md)

# quant_project — NASDAQ 100 Quantitative Research & Paper-Trading System

An end-to-end, reproducible quantitative equity system: it learns a
cross-sectional return-ranking model on NASDAQ-100 stocks, turns the ranking
into a Top-K portfolio, measures it under realistic transaction costs, and then
trades that same portfolio live against a simulated brokerage account through
the Alpaca **Paper Trading** API.

Paper trading is the default and only enabled execution mode — live endpoints
are hard-blocked in code, not merely discouraged by convention.

## What this project is

Most "quant strategy" code stops at a backtest curve. The problem is that a
backtest is the easiest thing in finance to accidentally fake: one misaligned
label, one normalization fitted on future data, one zero-cost assumption, and
an unprofitable idea looks excellent. This repository is built the other way
round — **correctness and honest measurement come first, returns second**:

* every result is produced by a committed script on real historical data, with
  transaction costs, and is reproducible from a config snapshot;
* the leakage traps are handled explicitly and documented (label alignment,
  execution shift, chronological splits with an embargo, no future information
  in normalization, point-in-time index membership);
* the same strategy logic that is backtested is what the paper-trading layer
  executes, so the two can be compared directly;
* the trading layer treats operational safety as a higher priority than
  performance — invalid signals refuse to trade, orders are validated before
  submission, fills are never assumed, and the broker is always the source of
  truth.

It is a research and paper-trading platform, not a money-making promise. The
measured edge (see [Verified results](#6-verified-results)) is real but thin and
regime-dependent, and the README says so plainly.

## What it does

The full pipeline, all of it implemented and runnable:

```
Market data (daily OHLCV, adjusted, point-in-time universe)
   ↓  scripts/prepare_data.py       — download + verify the qlib US bundle
Feature engineering (Alpha158, 157 features, + 10 custom factors)
   ↓  core/handlers.py, factors/    — qlib expressions & pandas twins, cross-verified
Dataset construction (chronological splits, embargo, forward-return label)
   ↓  core/handlers.py
LightGBM training (early stopping on validation Rank IC)
   ↓  scripts/train.py              — model + full reproducibility metadata
Return prediction & signal quality analysis (daily IC / Rank IC / deciles)
   ↓  scripts/predict.py
Cross-sectional ranking → Top-K portfolio (equal weight, capped turnover)
   ↓  strategies/topk.py
Cost-aware backtest + performance report (9 metrics, 4 charts)
   ↓  scripts/backtest.py
Walk-forward evaluation (5 rolling windows, model retrained per window)
   ↓  scripts/walk_forward.py
Paper trading: target portfolio → diff orders → risk checks → Alpaca paper
   ↓  scripts/paper_trade.py        — dry-run mode first, always
Order / position / PnL monitoring, reconciliation, backtest-vs-paper comparison
      scripts/account_status.py, reconcile.py, paper_vs_backtest.py
```

Concretely, the repository contains: 9 CLI entry points, a broker-agnostic
trading layer with an Alpaca adapter and an in-memory mock broker, 10 custom
factors with pandas/qlib twin implementations, a pure-function metrics library,
timestamped experiment tracking, and 61 unit tests covering factor correctness,
no-look-ahead guarantees, metric math, and every trading-safety path.

## How it works

The design decisions that matter, and why:

| Decision | Reason |
|---|---|
| Label `close[t+5]/close[t+1] − 1`, executed at t+1 close | The signal is formed after the close of t, so it can never be executed at a price already known when the features were computed |
| Early stopping on validation **Rank IC**, not MSE | Stock selection is a ranking problem. With MSE the model stops at round 1 on this ~100-name cross-section (the signal sits below the L2 noise floor); Rank IC selects models with actual ordering power |
| `embargo_days: 10` on train/valid segment ends | A 5-day forward label needs up to 6 future trading days; without the embargo, training labels would overlap the validation period |
| Point-in-time `nasdaq100` membership | Using today's constituents for a 2016 backtest would bake in survivorship bias |
| Costs always on (5 bps/side + $1 minimum) | A zero-cost backtest of a 0.40-turnover strategy is not a result, it is an artifact |
| Walk-forward as the headline number | A single split invites period-specific overfitting; 5 retrained windows show how the edge behaves across regimes |
| Orders from the target-vs-current **difference** | Rebuilding the portfolio daily would churn 100% of it; diffing touches only what changed |
| Broker state authoritative, fills polled | Local expectations drift from reality (partial fills, rejections, stale orders); reconciliation surfaces the gap instead of hiding it |
| Live trading blocked in code | A config typo should not be able to send real money anywhere |

Everything a researcher would want to change — universe, benchmark, horizon,
splits, model hyperparameters, K, turnover caps, costs, risk limits — lives in
YAML under `configs/`, not in Python.

## Project status

Verified end-to-end on real data in this repository: data preparation, feature
engineering, training, prediction/IC analysis, Top-K backtesting, performance
reporting, custom factors, and walk-forward evaluation. The paper-trading layer
(broker abstraction, order generation, risk controls, dry-run, execution,
reconciliation, comparison) is implemented and verified against the mock
broker; the one remaining step is the first **live paper session** against
Alpaca, which must be run from your own machine — see
[First milestone](#73-first-milestone-run-on-the-mac-in-order).

Known limitation: the bundled dataset ends 2020-11, so live paper trading needs
a data refresh first (top item on the [Roadmap](#12-roadmap)); until then the
staleness guard deliberately refuses to trade.

---

## Contents

1. [Requirements & installation](#1-requirements--installation)
2. [Quick start](#2-quick-start)
3. [Data](#3-data)
4. [Research design & leakage controls](#4-research-design--leakage-controls)
5. [Configuration reference](#5-configuration-reference)
6. [Verified results](#6-verified-results)
7. [Paper trading](#7-paper-trading)
8. [Repository layout](#8-repository-layout)
9. [Experiment tracking & reproducibility](#9-experiment-tracking--reproducibility)
10. [Tests](#10-tests)
11. [FAQ](#11-faq)
12. [Roadmap](#12-roadmap)
13. [Disclaimer](#13-disclaimer)

---

## 1. Requirements & installation

Target machine: Apple Silicon Mac mini (macOS, ARM64). Linux x86_64 works
equally well (all results in this repository were produced on Linux).

Dependency management uses `uv` only — no Conda, no system Python.

```bash
# Install uv (if missing)
brew install uv

# Reproduce the environment exactly from uv.lock
# (Python 3.11; pyqlib 0.9.7 ships native macOS universal2 ARM64 wheels)
uv sync
```

Key pinned versions (locked in `uv.lock`): `pyqlib 0.9.7`, `lightgbm 4.7`,
`pandas 2.3 (<3)`, `numpy 2.4`, `alpaca-py 0.43`, `mlflow ≥3.15` (required by
qlib initialization; this project does its own experiment tracking).

> Note: `pandas<3` and `mlflow>=2.9` are deliberate constraints — pyqlib's
> dependency metadata is under-specified and would otherwise resolve to an
> incompatible mlflow 1.27 + protobuf 7 combination.

## 2. Quick start

```bash
# ① Data: download the qlib official US daily bundle (~450MB, 1.3GB unpacked), with verification
uv run python scripts/prepare_data.py

# ② Training (Alpha158 → LightGBM, early stopping on validation Rank IC)
uv run python scripts/train.py --config configs/lightgbm_alpha158.yaml

# ③ Prediction + IC analysis (daily IC / Rank IC / decile-return charts on the test segment)
uv run python scripts/predict.py --config configs/lightgbm_alpha158.yaml

# ④ Top-K backtest (transaction costs included) + performance report
uv run python scripts/backtest.py --config configs/lightgbm_alpha158.yaml

# ⑤ Walk-forward evaluation (5 rolling windows, fresh model per window)
uv run python scripts/walk_forward.py --config configs/lightgbm_alpha158.yaml

# ⑥ Paper trading (configure keys first — see section 7; ALWAYS dry-run first)
uv run python scripts/paper_trade.py --config configs/paper_trading.yaml --dry-run
uv run python scripts/paper_trade.py --config configs/paper_trading.yaml
uv run python scripts/account_status.py --record
uv run python scripts/reconcile.py
uv run python scripts/paper_vs_backtest.py

# ⑦ Unit tests (61 tests)
uv run pytest tests/ -q
```

Every experiment writes into `results/<timestamp>_<name>/` — previous
experiments are never overwritten. Run logs go to `logs/`.

## 3. Data

| Item | Details |
|---|---|
| Source | qlib official US daily bundle (auto-downloaded and verified by `scripts/prepare_data.py`) |
| Coverage | **1999-12-31 → 2020-11-10**, 5,250 trading days |
| Fields | open / high / low / close / volume / factor |
| Adjustment | **Prices are split+dividend adjusted** (Yahoo-normalized); verified against the AAPL 2020-08-31 split (no jump); `$factor` preserved |
| Membership | `nasdaq100` is **point-in-time** (constituents enter/leave on their historical dates), which substantially reduces — but does not fully remove — survivorship bias |
| Benchmark | `^ndx` (NASDAQ-100 index; stored as a normalized level, returns are unaffected) |
| VWAP | The bundle has no `$vwap` field, so Alpha158's single VWAP feature is removed (157 features remain) — see `core/handlers.py` |

**Refreshing data**: the bundle ends 2020-11. Refreshing to the present
requires an external EOD source (the Alpaca market-data API is the natural
choice — your paper-trading keys already include it); writing daily bars into
qlib bin format plugs straight into the existing pipeline. This is the top
roadmap item. Until refreshed, the staleness guard refuses to trade on old
signals unless `--ignore-staleness` is passed explicitly for offline drills.

## 4. Research design & leakage controls

**Label**: `return[t] = close[t+5] / close[t+1] − 1` (signal produced after the
close of t, position entered at the close of t+1, held to the close of t+5).
The horizon is configurable via `label.horizon`.

**Execution alignment (no look-ahead)**: in backtests, TopkDropoutStrategy
trades day T using the signal from T−1 (`shift=1`) — empirically verified: the
first backtest day holds 100% cash (no prior-day signal exists). Features are
computed on data through the close of t and executed at t+1, exactly matching
the label definition.

**Chronological splits**: strictly time-ordered, never shuffled. An
`embargo_days: 10` shift pulls the train/valid segment ends back so
forward-looking labels (which need up to 6 future trading days of closes) can
never overlap the next segment — training labels contain no validation-period
information.

**Normalization**: labels use CSZScoreNorm (per-date cross-sectional z-score,
never across time — no future information); features go into LightGBM raw
(tree models handle NaN and scale natively).

**Early-stopping metric**: mean daily **Rank IC** on the validation segment,
not MSE. Cross-sectional stock selection is a ranking problem; empirically,
MSE-based early stopping halts at round 1 on this ~100-name, highly efficient
cross-section (the signal is below the L2 noise floor), while Rank-IC early
stopping selects models with actual ranking power.

**Transaction costs**: every backtest includes costs (default 5 bps per side +
$1 minimum per trade, configurable under `backtest.exchange`). Zero-cost
results are never reported.

**Walk-forward**: 5 rolling yearly windows (2016→2020), a fresh model trained
from scratch per window, strictly out-of-sample test segments, no information
flowing between windows. Sharing the feature handler across windows is safe
here because no processor in the pipeline fits statistics over time (all
operations are per-date cross-sectional).

## 5. Configuration reference

### configs/lightgbm_alpha158.yaml (research)

| Section | Keys | Notes |
|---|---|---|
| `experiment` | `name` / `seed` | Experiment name (suffix of results dirs) and global random seed (42) |
| `qlib` | `provider_uri` / `region` | Data directory and market region (us) |
| `universe` | `market: nasdaq100` / `benchmark: ^ndx` | Universe and benchmark, both swappable (e.g. sp500) |
| `data` | `start_time / end_time` | Handler data window (starts ~1y before training for 60-day rolling-feature warmup) |
| `label` | `horizon: 5` | Prediction horizon in trading days |
| `dataset` | `train / valid / test` + `embargo_days` | Chronological splits and the anti-overlap embargo |
| `features` | `custom_factors: false` | true enables Alpha158 + custom factors from `factors/` (167 features) |
| `model` | `params` etc. | LightGBM hyperparameters (tuned down for the ~100-name cross-section: lr 0.05 / 64 leaves / strong regularization) |
| `strategy` | `topk: 10 / n_drop: 2` | Hold the top 10 daily, replace at most 2 per day (turnover control) |
| `backtest` | `initial_capital` / `exchange.*` | Capital, deal price, two-sided costs, minimum fee, trade unit |
| `walk_forward` | `windows` | Rolling window definitions (train/valid/test per window) |

### configs/paper_trading.yaml (paper trading)

| Section | Keys | Notes |
|---|---|---|
| `signal` | `model_config` / `max_prediction_age_days` | Signal source experiment + staleness guard (default 5 days) |
| `broker` | `provider: alpaca / mode: paper / allow_live_trading: false` | **Live trading hard-blocked**: mode=live without the explicit override raises immediately; no automatic paper↔live fallback |
| `strategy` | `topk / n_drop / rebalance_frequency` | Top-K parameters mirroring the backtest |
| `portfolio` | `target_cash_ratio: 0.05` etc. | Cash reserve, 0.15 per-position cap, $200 minimum trade, 2% rebalance tolerance |
| `execution` | `order_type: market / time_in_force: day` + polling | Order type and fill-status polling (interval/timeout) |
| `risk` | see section 7 | All risk thresholds |
| `logging` | `dir` | Trading log directory |

## 6. Verified results

Everything below was produced by this repository's code on real historical
data (seed 42, costs included), stored under `results/`, and reproducible with
the commands in section 2.

**Baseline single split** (train 2008-2016, valid 2017-2018, test 2019-01-02 →
2020-11-09, 465 trading days):

| Metric | Value |
|---|---|
| Test daily Rank IC / ICIR | 0.021 / 0.17 (58% positive days) |
| Net annualized return (costs incl.) | 62.4% |
| Benchmark (NDX) annualized | 39.9% |
| Net excess annualized | +15.4% |
| Net Sharpe | 1.72 |
| Max drawdown | −29.5% (2020-03 COVID crash) |
| Mean daily turnover | 0.40 |

**Walk-forward (5 windows, strictly out-of-sample):**

| Window | Test period | Rank IC | Net ann. | Bench ann. | Sharpe | Max DD |
|---|---|---|---|---|---|---|
| 1 | 2016 | 0.023 | 15.7% | 5.9% | 0.78 | −15.3% |
| 2 | 2017 | 0.007 | 24.1% | 31.7% | 1.72 | −6.4% |
| 3 | 2018 | −0.003 | −7.4% | −1.0% | −0.23 | −18.2% |
| 4 | 2019 | −0.002 | 27.1% | 38.0% | 1.34 | −15.2% |
| 5 | 2020→11-10 | 0.059 | 66.0% | 42.3% | 1.35 | −35.3% |
| **Stitched** | 2016→2020-11 | — | **21.9%** | — | **0.89** | **−35.3%** |

**Honest read**: the edge is real but thin and regime-dependent — strongest in
high-volatility periods (2020), near zero or negative in the quiet 2018-2019
tape. The single-split baseline overstates what walk-forward supports; **treat
the stitched numbers as the reference**. This is precisely why the system
insists on walk-forward plus paper-trading validation before anything else.

## 7. Paper trading

### 7.1 Safety model (higher priority than performance)

* **Live trading hard-blocked**: `mode: paper` is the default; `mode: live`
  with `allow_live_trading: false` raises `LiveTradingBlockedError`. There is
  no automatic paper↔live switching in either direction.
* **Credentials**: read only from `.env` (`ALPACA_API_KEY` /
  `ALPACA_SECRET_KEY`), gitignored, never logged, never included in error
  messages.
* **Signal kill-switch (DO NOT TRADE)**: empty / NaN-heavy / zero-variance /
  stale predictions ⇒ refuse to trade, log the reason, exit non-zero.
* **Pre-trade validation**: per-order weight cap, daily turnover cap (initial
  build from an empty portfolio is exempt), post-trade exposure cap, minimum
  cash reserve, buying-power check, duplicate-open-order check — any violation
  rejects the whole batch locally.
* **Turnover minimization**: orders come only from the **difference** between
  target and current holdings; unchanged positions are never sold and
  re-bought. Sells execute before buys.
* **No fill assumptions**: every submitted order is polled to a terminal
  status (filled / canceled / rejected / expired); timeouts are logged loudly
  and left for reconciliation.
* **Duplicate-run protection**: `client_order_id` embeds the prediction date,
  so re-running the same rebalance is rejected both locally (exit code 3) and
  broker-side; only `--force` overrides.
* **Broker is the source of truth**: local state is only an expectation;
  `reconcile.py` detects missing fills, partial fills, rejections, unexpected
  holdings, cash drift, and stale open orders.

### 7.2 Credentials

```bash
cp .env.example .env
# Edit .env and fill in the paper keys generated at
# https://app.alpaca.markets/paper/dashboard/overview
```

### 7.3 First milestone (run on the Mac, in order)

```text
① uv run python scripts/account_status.py        # verify the paper endpoint (~$100k simulated equity)
② uv run python scripts/paper_trade.py --config configs/paper_trading.yaml --dry-run
   → manually inspect the proposed orders (symbol / side / qty / value)
③ Same command without --dry-run during US market hours → watch submissions and fills
④ uv run python scripts/account_status.py --record   # snapshot into history.csv
⑤ uv run python scripts/reconcile.py                 # expect "Reconciliation OK"
```

Milestone achieved when: model produces signals → dry-run produces valid
orders → paper API accepts them → broker confirms fills → local system reads
back positions → reconciliation passes.

### 7.4 Daily operation & monitoring

Each trading day: `paper_trade.py` (dry-run first, then live-paper) →
`account_status.py --record` → `reconcile.py`. After ~20 snapshots,
`account_status.py --record` starts printing paper Sharpe / drawdown /
volatility / win rate, and `paper_vs_backtest.py` compares expected vs actual
turnover, holdings overlap, and fill ratios — the test of whether the strategy
survives realistic execution. Every run record (prediction date, target
weights, current positions, orders, broker order IDs, fill statuses, account
equity, errors) is written to `logs/paper_trading/run_*.json`. Paper
performance and historical backtest performance are stored separately and
never mixed.

## 8. Repository layout

```
quant_project/
├── configs/                  # All research/trading parameters (zero hard-coding)
│   ├── lightgbm_alpha158.yaml
│   └── paper_trading.yaml
├── core/
│   ├── config.py             # YAML loading, date normalization, nested-key validation
│   ├── log.py                # Console + logs/ file logging
│   ├── qlib_init.py          # qlib initialization (region/data checks)
│   ├── handlers.py           # Alpha158NoVWAP / +custom handler, label expression, embargo, dataset builder
│   └── experiment.py         # Timestamped experiment dirs, config snapshots, reproducibility metadata
├── factors/                  # Custom factors: pandas + qlib expression twins, cross-verified
│   ├── momentum.py           # 5/20/60-day momentum, 1-day reversal
│   ├── technical.py          # RSI, MACD (price-normalized), MA deviation
│   ├── volatility.py         # 20-day realized volatility
│   └── volume.py             # Volume ratio
├── models/
│   └── lightgbm_model.py     # LightGBM wrapper: Rank-IC early stopping, fit/predict/save/load interface
├── strategies/
│   └── topk.py               # Top-K (TopkDropout) strategy construction
├── backtests/
│   ├── engine.py             # Backtest engine (costs, calendar edge handling, position flattening)
│   ├── metrics.py            # Pure-function metric library + IC analysis (independently unit-tested)
│   ├── report.py             # Metrics JSON + equity/excess/drawdown/turnover charts
│   └── comparison.py         # Backtest vs paper comparison
├── trading/
│   ├── broker.py             # Broker-agnostic interface and order/position/account types
│   ├── alpaca_paper.py       # Alpaca adapter (paper default, live blocked, latest-price provider)
│   ├── mock_broker.py        # In-memory mock broker (tests/offline drills; rejections & partial fills)
│   ├── portfolio.py          # Target portfolio (equal weight + cash reserve + caps) and diff-based orders
│   ├── risk.py               # Signal kill-switch + full pre-trade validation
│   ├── executor.py           # Dry-run / submission + status polling
│   ├── reconciliation.py     # Reconciliation (broker authoritative)
│   └── state.py              # Run records, expected state, performance history
├── scripts/                  # 9 CLI entry points (see section 2)
├── tests/                    # 61 unit tests: factors, metrics, labels, embargo, trading safety
├── results/                  # Experiment outputs (timestamped, never overwritten)
├── logs/                     # Run and trading logs
├── .env.example              # Credential template (.env is gitignored)
├── pyproject.toml / uv.lock  # Fully locked environment
└── README.md / README.zh-CN.md / CLAUDE.md   # This document / Chinese version / engineering spec
```

## 9. Experiment tracking & reproducibility

Each `train.py` run creates `results/<timestamp>_<name>/` containing:

* `config_snapshot.yaml` — the full config used for that run
* `experiment.json` — train/valid/test periods, universe, benchmark, feature
  set, horizon, model & strategy parameters, costs, random seed, best
  iteration, validation Rank IC
* `model.pkl` — the model (booster + feature names)
* `pred_test.csv`, `ic_daily_test.csv`, `ic_summary_test.json`, and the
  IC/distribution/decile charts
* `backtest/` — daily report, daily positions, metrics JSON, four curve charts

Walk-forward additionally produces a `_walkforward` directory with per-window
outputs, `windows_summary.csv`, and the full stitched report. Given the same
data and config, anyone can reproduce any result.

## 10. Tests

```bash
uv run pytest tests/ -q      # 61 passed
```

Coverage: hand-computed factor values and per-factor no-look-ahead checks
(truncation comparison), hand-computed metrics, IC extreme-correlation checks,
label expression, embargo boundaries, target portfolio construction,
diff-based orders (including the canonical "swap NVDA→AMZN, don't touch
AAPL/MSFT" case), every risk-rejection path, dry-run submits nothing,
fill/rejection/partial-fill handling, double duplicate-order protection, live
mode blocking, and all reconciliation discrepancy types.

## 11. FAQ

**Q: Backtest raises `IndexError: index ... out of bounds`?**
The backtest end date cannot equal the last calendar day (qlib needs the "next
day" to compute step boundaries). The engine clamps the end automatically; no
handling needed when calling `run_backtest`.

**Q: mlflow "file-store maintenance mode" error?**
mlflow ≥3.15 disables the filesystem backend by default; `core/qlib_init.py`
sets `MLFLOW_ALLOW_FILE_STORE=true` automatically (this project does not use
mlflow for tracking).

**Q: Paper trading says "prediction date ... is N days old"?**
The staleness guard is working. Refresh data and retrain for real operation;
add `--ignore-staleness` only for offline drills.

**Q: What are exit codes 2 / 3 from `paper_trade.py`?**
2 = risk validation refused (reasons in logs and the run record);
3 = this prediction date's rebalance was already executed (duplicate-run guard).

**Q: Installation fails on Apple Silicon?**
Use `uv sync` (not manual pip); pyqlib 0.9.7 ships universal2 wheels, no
compilation needed. If a package fails, read the actual error first — do not
switch package managers.

**Q: How do I change the universe?**
Set `universe.market` (the bundle ships `sp500` and other instrument files)
and `universe.benchmark`. Nothing else changes.

## 12. Roadmap

In priority order: ① an Alpaca→qlib data refresh script (closing the loop:
fresh data → same-day signals → same-day paper orders); ② a full controlled
experiment for the `features.custom_factors: true` branch; ③ additional models
(XGBoost / MLP / LSTM / Transformer) behind the `models/lightgbm_model.py`
interface; ④ portfolio weight optimization (the baseline is deliberately
equal-weight); ⑤ reinforcement learning considered only for sizing/execution,
and only after the supervised baseline survives paper trading.

## 13. Disclaimer

This repository is for quantitative research and simulated (paper) trading
only. Historical backtests and simulated fills promise nothing about future
returns and constitute no investment advice. Never connect this system to a
real-money account without thorough validation.
