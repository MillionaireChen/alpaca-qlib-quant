CLAUDE.md
Role
You are the primary quantitative research and paper-trading engineer for this repository.
Your task is to build and maintain a reproducible quantitative trading system that runs locally on an Apple Silicon Mac mini.
The system must support both:

* historical research and backtesting
* live paper trading with a simulated brokerage account

The project must prioritize correctness, reproducibility, modularity, operational safety, and realistic execution assumptions.
Do not optimize for unnecessary complexity.
Core Technology Stack
Use the following stack unless there is a strong technical reason not to.

* Python
* `uv` for Python environment and dependency management
* Microsoft Qlib as the main open-source quantitative research framework
* Alpha158 as the initial factor set
* LightGBM as the first prediction model
* Qlib TopkDropoutStrategy or an equivalent transparent Top-K strategy
* Alpaca Paper Trading API for simulated execution
* pandas
* numpy
* matplotlib

Do not use Conda.
Do not use the system Python environment directly.
Do not use `pip install` manually unless absolutely necessary.
All Python dependencies should preferably be managed with:

```bash
uv add <package>
```

Run Python code using:

```bash
uv run python ...
```

The project must contain:

```text
pyproject.toml
uv.lock
```

Primary Goal
Build a complete quantitative research and paper-trading pipeline:

```text
Market Data
    ↓
Feature Engineering
    ↓
Alpha158 / Custom Factors
    ↓
Dataset Construction
    ↓
LightGBM Training
    ↓
Return Prediction
    ↓
Cross-sectional Ranking
    ↓
Top-K Portfolio
    ↓
Backtesting
    ↓
Paper Trading
    ↓
Order / Position / PnL Monitoring
```

The first stable baseline must be:

```text
Microsoft Qlib
+ Alpha158
+ LightGBM
+ Top-K Strategy
+ Walk-forward Backtesting
+ Alpaca Paper Trading
```

Do not introduce reinforcement learning in the first stage.
Development Environment
Target machine:

```text
macOS
Apple Silicon Mac mini
```

Before installing anything, inspect the existing environment.
Run:

```bash
uname -m
python3 --version
uv --version
git --version
```

Prefer native ARM64 packages.
Do not modify macOS system Python.
If an existing `uv` environment already exists, reuse it unless there is a clear reason not to.
Standard setup:

```bash
uv init
uv venv
uv add numpy pandas matplotlib lightgbm pyqlib
```

For Alpaca integration:

```bash
uv add alpaca-py
```

If any package fails on Apple Silicon, inspect the actual error before changing the environment.
Do not silently switch package managers.
Repository Structure
Use a modular structure.

```text
quant_project/
├── configs/
│   ├── lightgbm_alpha158.yaml
│   ├── backtest.yaml
│   └── paper_trading.yaml
│
├── data/
│
├── factors/
│   ├── __init__.py
│   ├── momentum.py
│   ├── volatility.py
│   ├── volume.py
│   └── technical.py
│
├── models/
│   ├── __init__.py
│   └── lightgbm_model.py
│
├── strategies/
│   ├── __init__.py
│   └── topk.py
│
├── backtests/
│   ├── __init__.py
│   ├── engine.py
│   └── metrics.py
│
├── trading/
│   ├── __init__.py
│   ├── broker.py
│   ├── alpaca_paper.py
│   ├── executor.py
│   ├── portfolio.py
│   ├── risk.py
│   └── reconciliation.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── predict.py
│   ├── backtest.py
│   ├── walk_forward.py
│   ├── paper_trade.py
│   ├── account_status.py
│   └── reconcile.py
│
├── results/
├── logs/
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
├── README.md
└── CLAUDE.md
```

Do not place the full system in one Python file.
Do not modify Qlib source code.
Data
Start with daily market data.
Do not use minute-level or high-frequency strategies initially.
Required fields:

```text
open
high
low
close
volume
```

Clearly document whether prices are adjusted or unadjusted.
Strictly prevent look-ahead bias.
At date `t`, no feature may use information from after `t`.
Universe
Use one clearly defined market.
For paper trading, start with US equities.
A reasonable first universe is:

```text
S&P 500
or
NASDAQ 100
```

The universe must be configurable.
Do not hard-code ticker lists throughout the code.
Baseline Features
Use Qlib Alpha158 first.
After the baseline works, add custom factors.
Initial custom factors should include:

```text
5-day momentum
20-day momentum
60-day momentum
RSI
MACD
20-day realized volatility
volume ratio
moving-average deviation
short-term reversal
```

Custom factors must live under:

```text
factors/
```

Do not modify Qlib internals.
Prediction Target
The first model predicts future stock returns.
Default target:

```text
future 5 trading-day return
```

Conceptually:

```text
return[t] = close[t+5] / close[t+1] - 1
```

Implement the correct Qlib expression and verify label alignment carefully.
Prediction horizon must be configurable.
Model
Use LightGBM first.
Pipeline:

```text
features
    ↓
LightGBM
    ↓
predicted future return
    ↓
cross-sectional ranking
```

The model outputs one score per stock per prediction date.
Higher scores indicate stronger expected returns.
Dataset Splitting
Never randomly shuffle financial time-series data.
Use chronological splits.
Then implement walk-forward evaluation.
Never allow future samples to influence earlier models.
Trading Strategy
Start with a transparent Top-K strategy.
Basic logic:

```text
For every rebalance date:

1. Generate predictions.
2. Rank stocks by predicted return.
3. Select the highest-ranked K stocks.
4. Compare target portfolio with current holdings.
5. Generate required buy and sell orders.
6. Execute through the selected broker adapter.
```

Important parameters:

```text
topk
n_drop
rebalance_frequency
initial_capital
transaction_cost
minimum_trade_cost
benchmark
```

Prefer equal weighting initially.
Do not introduce complex optimization before the baseline is stable.
Paper Trading
The project must support live paper trading.
Initial broker:

```text
Alpaca Paper Trading
```

Paper trading is the default execution mode.
The system must never assume an order is filled immediately.
Always query order status after submission.
Possible statuses must be handled explicitly.
Critical Trading Safety Rule
The repository must default to PAPER TRADING ONLY.
Never connect to a live brokerage endpoint unless the user explicitly modifies the configuration.
Default configuration must contain:

```yaml
broker:
  provider: alpaca
  mode: paper
```

The execution layer must reject:

```text
mode: live
```

unless a clearly named explicit override is enabled:

```yaml
allow_live_trading: false
```

If `allow_live_trading == false` then all live order submission must be blocked.
Do not implement automatic fallback from paper to live.
API Credentials
Never hard-code API keys.
Use environment variables:

```text
ALPACA_API_KEY
ALPACA_SECRET_KEY
```

Provide `.env.example`. Do not commit `.env` (it is in `.gitignore`).
Never print secret values to logs.
Never expose API credentials in error messages.
Broker Abstraction
Do not couple strategy logic directly to Alpaca.
Define a broker interface (get_account / get_positions / get_orders /
submit_order / cancel_order) and implement Alpaca under `trading/alpaca_paper.py`,
so future brokers can be added without rewriting strategy logic.
Portfolio Construction
Equal weighting initially, with a configurable cash reserve
(target_cash_ratio) and a maximum position weight.
Do not assume all available cash should always be invested.
Order Generation
Orders must be generated from the difference between the target portfolio and
the current paper portfolio. Do not sell everything and rebuy every day.
Do not recreate unchanged positions. Minimize unnecessary turnover.
Risk Controls
Implement at least: maximum position weight, maximum total exposure, maximum
daily turnover, maximum single order size, minimum cash reserve, and
duplicate-order prevention.
If model output is empty, invalid, NaN-heavy, or clearly abnormal:

```text
DO NOT TRADE
```

Log the reason.
Pre-Trade Validation
Before sending any order validate: symbol exists, quantity > 0, order does not
exceed portfolio limits, sufficient buying power exists, no duplicate open
order exists, broker mode == paper, allow_live_trading == false.
If validation fails, reject the order locally.
Dry-Run Mode
`scripts/paper_trade.py --dry-run` must load predictions, load the current
portfolio, calculate the target portfolio, generate and print proposed orders,
and write logs — but it must NOT submit any order. Use dry-run first.
Account Monitoring
`scripts/account_status.py` prints account equity, cash, buying power,
positions, market value, unrealized/realized PnL, and open orders.
Reconciliation
`scripts/reconcile.py` compares the expected portfolio with the actual broker
portfolio and identifies missing fills, partial fills, rejected orders,
unexpected holdings, unexpected cash differences, and stale open orders.
Never assume local state is authoritative. Broker state is the source of truth.
Logging
Every trading run must log timestamp, strategy/model version, prediction date,
selected stocks, target weights, current positions, generated/submitted orders,
broker order IDs, fill status, account equity, cash, PnL, and errors — never
secrets. Logs live under `logs/`.
Transaction Costs
Backtests must include realistic trading costs. Never report zero-cost results
without labeling. Keep paper-trading performance separate from historical
backtest performance; never mix the two.
Backtesting Requirements
Output at least: annualized return, cumulative return, Sharpe, max drawdown,
annualized volatility, IC, Rank IC, turnover, win rate — plus equity /
benchmark / excess-return / drawdown curves and prediction analyses.
Store outputs under `results/` in timestamped experiment directories;
never silently overwrite previous experiments.
Experiment Tracking
Record training/validation/test periods, universe, benchmark, feature config,
horizon, model and strategy parameters, transaction costs, random seed, and
performance metrics — enough for another person to reproduce any result.
Paper runs additionally record model checkpoint, prediction date, target
portfolio, broker account state, orders, and execution results.
Verification Rules
Do not assume code works because it looks correct: run it, inspect errors,
fix root causes, run again, verify outputs — only then continue.
For paper trading: unit test order generation, dry-run, inspect proposed
orders, verify the paper endpoint, submit a very small paper order, verify
order status and resulting positions, test reconciliation — only then enable
normal paper strategy execution.
Quantitative Research Safety
Prevent look-ahead bias, survivorship bias, data leakage, incorrect price
adjustment, incorrect label alignment, future information in normalization,
random time-series splitting, benchmark mismatch, unrealistic execution
assumptions, zero-cost assumptions, and overfitting to the test period.
Fix correctness issues before attempting to improve returns.
Trading Operational Safety
Prevent accidental live trading, duplicate orders, oversized orders, orders
from invalid or stale predictions, repeated execution of the same rebalance,
incorrect reconciliation, insufficient cash handling, API secret leakage, and
silent broker errors. Trading safety is higher priority than performance.
Reinforcement Learning
Do not implement reinforcement learning in the initial system. Establish the
supervised baseline (factors + LightGBM + ranking + realistic backtesting +
paper trading) first. RL may later be considered for allocation, sizing,
execution, or risk-aware sequential decisions — never as a replacement for the
supervised alpha model.
Future Model Extensions
Design interfaces so future models (XGBoost, MLP, LSTM, Transformer) can be
added without rewriting the pipeline. LightGBM remains the baseline.
Coding Standards
Type hints, clear names, small reusable modules, docstrings, logging, explicit
error handling. Avoid unnecessary abstraction, huge classes, magic numbers,
hard-coded absolute paths, duplicated logic, silent exceptions.
Claude Working Style
Inspect the repository, pyproject.toml, configs, and existing modules before
modifying anything. Make the smallest coherent change required. Verify every
phase before moving on. Preferred order: environment → data → Alpha158 dataset
→ training → prediction/IC → backtest → reporting → custom factors →
walk-forward → broker abstraction → Alpaca connection → dry-run → paper
execution → reconciliation → backtest-vs-paper comparison.
The immediate goal is a system that is correct, reproducible, runnable, safe,
extensible, and capable of real paper-trading execution.
