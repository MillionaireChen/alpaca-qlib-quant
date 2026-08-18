# quant_project — NASDAQ 100 量化研究与纸面交易系统

基于 Microsoft Qlib + Alpha158 + LightGBM + Top-K 策略的可复现量化研究系统,
并通过 Alpaca **Paper Trading**(模拟账户)API 支持真实的纸面交易执行。
默认且唯一开放的执行模式是纸面交易,实盘端点被硬性拦截。

```
市场数据 → 特征工程(Alpha158 + 自定义因子)→ 数据集构建 → LightGBM 训练
→ 收益预测 → 截面排序 → Top-K 组合 → 含成本回测 → 纸面交易 → 订单/持仓/PnL 监控与对账
```

---

## 目录

1. [系统要求与安装](#1-系统要求与安装)
2. [快速开始](#2-快速开始)
3. [数据说明](#3-数据说明)
4. [研究方法与防泄漏设计](#4-研究方法与防泄漏设计)
5. [配置详解](#5-配置详解)
6. [已验证结果](#6-已验证结果)
7. [纸面交易](#7-纸面交易)
8. [目录结构与模块说明](#8-目录结构与模块说明)
9. [实验追踪与复现](#9-实验追踪与复现)
10. [测试](#10-测试)
11. [常见问题](#11-常见问题)
12. [未来扩展](#12-未来扩展)
13. [免责声明](#13-免责声明)

---

## 1. 系统要求与安装

目标机器:Apple Silicon Mac mini(macOS,ARM64)。Linux x86_64 同样可运行(本仓库全部结果即在 Linux 上验证生成)。

依赖管理只用 `uv`,不用 Conda,不直接使用系统 Python。

```bash
# 安装 uv(如未安装)
brew install uv

# 在仓库根目录按 uv.lock 精确复现环境(Python 3.11,pyqlib 0.9.7 提供 macOS universal2 原生 ARM64 轮子)
uv sync
```

关键依赖版本(由 `uv.lock` 锁定):`pyqlib 0.9.7`、`lightgbm 4.7`、`pandas 2.3(<3)`、
`numpy 2.4`、`alpaca-py 0.43`、`mlflow ≥3.15`(qlib 初始化需要,本项目自身不用它做追踪)。

> 注:`pandas<3` 与 `mlflow>=2.9` 是刻意约束——pyqlib 对依赖上限声明不完整,
> 放开会解析出 mlflow 1.27 + protobuf 7 的不兼容组合。

## 2. 快速开始

```bash
# ① 数据:下载 qlib 官方美股日线包(约 450MB,解压后 1.3GB),含验证步骤
uv run python scripts/prepare_data.py

# ② 训练(Alpha158 → LightGBM,验证集 Rank-IC 早停)
uv run python scripts/train.py --config configs/lightgbm_alpha158.yaml

# ③ 预测 + IC 分析(测试段逐日 IC / Rank IC / 分层收益图)
uv run python scripts/predict.py --config configs/lightgbm_alpha158.yaml

# ④ Top-K 回测(含交易成本)+ 绩效报告
uv run python scripts/backtest.py --config configs/lightgbm_alpha158.yaml

# ⑤ Walk-forward 滚动窗口评估(5 窗口,逐窗重训)
uv run python scripts/walk_forward.py --config configs/lightgbm_alpha158.yaml

# ⑥ 纸面交易(先配密钥,见第 7 节;永远先 dry-run)
uv run python scripts/paper_trade.py --config configs/paper_trading.yaml --dry-run
uv run python scripts/paper_trade.py --config configs/paper_trading.yaml
uv run python scripts/account_status.py --record
uv run python scripts/reconcile.py
uv run python scripts/paper_vs_backtest.py

# ⑦ 单元测试(61 项)
uv run pytest tests/ -q
```

所有实验输出进入 `results/<时间戳>_<实验名>/`,绝不覆盖历史实验;运行日志进入 `logs/`。

## 3. 数据说明

| 项目 | 说明 |
|---|---|
| 来源 | qlib 官方美股日线数据包(`scripts/prepare_data.py` 自动下载并验证) |
| 覆盖 | **1999-12-31 → 2020-11-10**,日历 5250 个交易日 |
| 字段 | open / high / low / close / volume / factor |
| 复权 | **价格为拆股+分红复权价**(Yahoo 规范化),已用 AAPL 2020-08-31 拆股日验证无跳变;`$factor` 保留 |
| 成分股 | `nasdaq100` 为**逐日时点成员**(个股按历史日期进出指数),显著降低——但不能完全消除——幸存者偏差 |
| 基准 | `^ndx`(NASDAQ-100 指数,存储为归一化点位,收益率不受影响) |
| VWAP | 数据包无 `$vwap` 字段,Alpha158 中唯一的 VWAP 特征被移除(剩 157 特征),见 `core/handlers.py` |

**数据刷新**:数据止于 2020-11。刷新到当前需要外部行情源(推荐 Alpaca 行情 API,
纸面交易密钥自带权限),把日线写入 qlib bin 格式即可无缝接入现有管线;这是第一优先的后续扩展。
在刷新之前,纸面交易的信号过期保护(`signal.max_prediction_age_days`)会拒绝交易,
只有显式传 `--ignore-staleness` 才能用旧信号做离线演练。

## 4. 研究方法与防泄漏设计

**标签**:`return[t] = close[t+5] / close[t+1] − 1`(t 日收盘后出信号,t+1 收盘价建仓,
持有到 t+5 收盘)。horizon 通过 `label.horizon` 配置。

**执行对齐(无未来函数)**:回测中 TopkDropoutStrategy 以 `shift=1` 取**前一交易日**的信号
决定当日交易——已实证验证:回测首日持仓为 100% 现金(首日没有前一日信号)。
特征在 t 日收盘数据上计算、t+1 日成交,与标签定义严格一致。

**时间切分**:严格按时间顺序切分,绝不随机打乱。并加 `embargo_days: 10`:
train/valid 段末端整体回退 10 天,使前视标签(需要未来 6 个交易日的收盘价)
不会与下一段的时间区间重叠——训练标签不含验证期信息。

**归一化**:标签归一化用 CSZScoreNorm(逐日截面 z-score,不跨时间,无未来信息);
特征原样进 LightGBM(树模型天然处理 NaN 与量纲)。

**早停指标**:验证集**日均 Rank IC**(而非 MSE)。截面选股关心的是排序质量;
实测 MSE 早停在 ~100 只股票的高效市场截面上会在第 1 轮就停(信号弱于噪声的 L2 下界),
Rank-IC 早停才能选出有排序能力的模型。

**交易成本**:所有回测强制含成本(默认买卖各 5bp + 每笔最低 $1,`backtest.exchange` 可配),
不报告任何零成本结果。

**Walk-forward**:5 个滚动年度窗口(2016→2020),每窗**从零重训**,
测试段严格样本外,窗口间无任何信息回流;共享特征 handler 是安全的
(管线中不存在需要拟合统计量的 processor,全部为逐日截面运算)。

## 5. 配置详解

### configs/lightgbm_alpha158.yaml(研究)

| 段 | 关键项 | 说明 |
|---|---|---|
| `experiment` | `name` / `seed` | 实验名(决定 results 目录后缀)与全局随机种子(42) |
| `qlib` | `provider_uri` / `region` | 数据目录与市场区域(us) |
| `universe` | `market: nasdaq100` / `benchmark: ^ndx` | 股票池与基准,均可替换(如 sp500) |
| `data` | `start_time / end_time` | handler 数据窗(起点早于训练起点 1 年,预热 60 日滚动特征) |
| `label` | `horizon: 5` | 预测期(交易日) |
| `dataset` | `train / valid / test` + `embargo_days` | 时间切分与防重叠回退 |
| `features` | `custom_factors: false` | true 时启用 Alpha158 + `factors/` 自定义因子(167 特征) |
| `model` | `params` 等 | LightGBM 超参(为 ~100 只股票的窄截面调低:lr 0.05 / 64 叶 / 强正则) |
| `strategy` | `topk: 10 / n_drop: 2` | 每日持有前 10 名,单日最多换 2 只(控制换手) |
| `backtest` | `initial_capital` / `exchange.*` | 初始资金与成交价、双边成本、最低费用、交易单位 |
| `walk_forward` | `windows` | 滚动窗口定义,逐窗训练/验证/测试区间 |

### configs/paper_trading.yaml(纸面交易)

| 段 | 关键项 | 说明 |
|---|---|---|
| `signal` | `model_config` / `max_prediction_age_days` | 信号来源实验 + 过期保护(默认 5 天) |
| `broker` | `provider: alpaca / mode: paper / allow_live_trading: false` | **实盘硬拦截**:mode=live 且未显式开启覆盖时直接抛错,无自动降级/升级 |
| `strategy` | `topk / n_drop / rebalance_frequency` | 与回测同构的 Top-K 参数 |
| `portfolio` | `target_cash_ratio: 0.05` 等 | 现金保留、单仓上限 0.15、最小交易额 $200、再平衡容差 2% |
| `execution` | `order_type: market / time_in_force: day` + 轮询参数 | 订单类型与成交状态轮询(间隔/超时) |
| `risk` | 见第 7 节 | 全部风控阈值 |
| `logging` | `dir` | 交易日志目录 |

## 6. 已验证结果

以下全部为本仓库代码在真实历史数据上运行的产物(seed 42,含成本),
对应 `results/` 内的实验目录,任何人可按第 2 节命令复现。

**基线单段**(训练 2008-2016,验证 2017-2018,测试 2019-01-02 → 2020-11-09,465 交易日):

| 指标 | 数值 |
|---|---|
| 测试段日均 Rank IC / ICIR | 0.021 / 0.17(58% 天数为正) |
| 组合净年化(含成本) | 62.4% |
| 基准(NDX)年化 | 39.9% |
| 净超额年化 | +15.4% |
| 净 Sharpe | 1.72 |
| 最大回撤 | −29.5%(2020-03 疫情崩盘) |
| 日均换手 | 0.40 |

**Walk-forward(5 窗口,严格样本外)**:

| 窗口 | 测试期 | Rank IC | 净年化 | 基准年化 | Sharpe | 最大回撤 |
|---|---|---|---|---|---|---|
| 1 | 2016 | 0.023 | 15.7% | 5.9% | 0.78 | −15.3% |
| 2 | 2017 | 0.007 | 24.1% | 31.7% | 1.72 | −6.4% |
| 3 | 2018 | −0.003 | −7.4% | −1.0% | −0.23 | −18.2% |
| 4 | 2019 | −0.002 | 27.1% | 38.0% | 1.34 | −15.2% |
| 5 | 2020→11-10 | 0.059 | 66.0% | 42.3% | 1.35 | −35.3% |
| **拼接** | 2016→2020-11 | — | **21.9%** | — | **0.89** | **−35.3%** |

**诚实解读**:信号真实但薄、且依赖市场状态——高波动期(2020)最强,
平静期(2018-2019)接近于零甚至为负。单段基线高估了 walk-forward 能支撑的水平,
**以拼接结果为准**。这正是本系统坚持 walk-forward + 纸面交易验证、而不急于上实盘的原因。

## 7. 纸面交易

### 7.1 安全模型(优先级高于收益)

* **实盘硬拦截**:`mode: paper` 为默认;`mode: live` 且 `allow_live_trading: false` 时
  抛 `LiveTradingBlockedError`,不存在 paper↔live 自动切换。
* **密钥**:仅从 `.env` 读取(`ALPACA_API_KEY` / `ALPACA_SECRET_KEY`),
  已在 `.gitignore`,永不入日志、永不出现在报错信息里。
* **信号熔断(DO NOT TRADE)**:预测为空 / NaN 超限 / 截面零方差 / 过期 → 拒绝交易,
  记录原因,非零退出码。
* **交易前校验**:单笔权重上限、日换手上限(空仓首次建仓豁免)、事后总敞口上限、
  最低现金保留、购买力校验、重复挂单校验——任一违反,整批本地拒绝。
* **最小化换手**:订单只来自目标组合与当前持仓的**差分**,不动的仓位绝不先卖后买;
  卖单先于买单执行。
* **成交不做假设**:每笔提交后轮询至终态(filled / canceled / rejected / expired),
  超时未终态会在日志中警告并留给对账处理。
* **防重复执行**:`client_order_id` 内嵌预测日期,同一 rebalance 重跑会被本地状态检查
  (退出码 3)与券商端重复 ID 双重拒绝;`--force` 才能覆盖。
* **对账以券商为准**:本地状态只是"期望",`reconcile.py` 检出缺失成交、部分成交、
  拒单、意外持仓、现金偏差、过期挂单。

### 7.2 密钥配置

```bash
cp .env.example .env
# 编辑 .env,填入 https://app.alpaca.markets/paper/dashboard/overview 生成的 paper 密钥
```

### 7.3 首次里程碑(在 Mac 上按序执行)

```text
① uv run python scripts/account_status.py        # 验证 paper 端点(约 $100k 模拟资金)
② uv run python scripts/paper_trade.py --config configs/paper_trading.yaml --dry-run
   → 人工检查拟下订单(标的/方向/数量/金额)
③ 同命令去掉 --dry-run,在美股盘中运行 → 观察提交与成交日志
④ uv run python scripts/account_status.py --record   # 记录快照到 history.csv
⑤ uv run python scripts/reconcile.py                 # 期望 "Reconciliation OK"
```

达成:模型出信号 → dry-run 出有效订单 → paper API 提交 → 券商确认成交
→ 本地读回持仓 → 对账通过。

### 7.4 日常运行与监控

每个交易日:`paper_trade.py`(先 dry-run 后正式)→ `account_status.py --record` →
`reconcile.py`。累计 ≥20 个快照后,`account_status.py --record` 自动输出纸面
Sharpe / 回撤 / 波动 / 胜率;`paper_vs_backtest.py` 输出回测预期 vs 实际执行的对比
(换手差、持仓重合率、成交率),用于判断策略是否经得起真实执行。
所有运行记录(预测日期、目标权重、当前持仓、订单、券商订单 ID、成交状态、
账户权益、错误)写入 `logs/paper_trading/run_*.json`,纸面绩效与历史回测绩效
**分开存放,绝不混算**。

## 8. 目录结构与模块说明

```
quant_project/
├── configs/                  # 全部研究/交易参数(代码零硬编码)
│   ├── lightgbm_alpha158.yaml
│   └── paper_trading.yaml
├── core/
│   ├── config.py             # YAML 加载、日期规范化、嵌套键校验
│   ├── log.py                # 控制台 + logs/ 文件双路日志
│   ├── qlib_init.py          # qlib 初始化(区域/数据目录校验)
│   ├── handlers.py           # Alpha158NoVWAP / +自定义因子 handler、标签表达式、embargo、数据集构建
│   └── experiment.py         # 时间戳实验目录、配置快照、复现元数据
├── factors/                  # 自定义因子:pandas 实现 + qlib 表达式双版本,已交叉验证一致
│   ├── momentum.py           # 5/20/60 日动量、1 日反转
│   ├── technical.py          # RSI、MACD(价格归一)、均线偏离
│   ├── volatility.py         # 20 日已实现波动
│   └── volume.py             # 量比
├── models/
│   └── lightgbm_model.py     # LightGBM 封装:Rank-IC 早停、fit/predict/save/load 统一接口
├── strategies/
│   └── topk.py               # Top-K(TopkDropout)策略构建
├── backtests/
│   ├── engine.py             # 回测引擎(成本、日历边界、持仓展平)
│   ├── metrics.py            # 纯函数指标库 + IC 分析(可独立单测)
│   ├── report.py             # 指标 JSON + 净值/超额/回撤/换手四图
│   └── comparison.py         # 回测 vs 纸面对比
├── trading/
│   ├── broker.py             # 券商抽象接口与订单/持仓/账户数据类型
│   ├── alpaca_paper.py       # Alpaca 适配器(paper 默认、live 拦截、最新价查询)
│   ├── mock_broker.py        # 内存模拟券商(测试/离线演练;可配拒单、部分成交)
│   ├── portfolio.py          # 目标组合(等权+现金保留+上限)与差分订单生成
│   ├── risk.py               # 信号熔断 + 交易前全套风控校验
│   ├── executor.py           # dry-run / 提交 + 状态轮询
│   ├── reconciliation.py     # 对账(券商为准)
│   └── state.py              # 运行记录、期望状态、绩效历史
├── scripts/                  # 9 个 CLI 入口(见第 2 节)
├── tests/                    # 61 项单测:因子、指标、标签、embargo、交易安全全覆盖
├── results/                  # 实验产物(时间戳目录,永不覆盖)
├── logs/                     # 运行与交易日志
├── .env.example              # 密钥模板(.env 已被 git 忽略)
├── pyproject.toml / uv.lock  # 环境完全锁定
└── README.md / CLAUDE.md     # 本文档 / 项目工程规范
```

## 9. 实验追踪与复现

每次 `train.py` 生成 `results/<时间戳>_<实验名>/`,内含:

* `config_snapshot.yaml` — 当次配置完整快照
* `experiment.json` — 训练/验证/测试区间、股票池、基准、特征集、预测期、
  模型与策略参数、成本、随机种子、最优迭代、验证 Rank IC
* `model.pkl` — 模型(含 booster 与特征名)
* `pred_test.csv`、`ic_daily_test.csv`、`ic_summary_test.json`、IC/分布/分层三张图
* `backtest/` — 日度报表、逐日持仓、指标 JSON、四张曲线图

walk-forward 另生成 `_walkforward` 目录:逐窗产物 + `windows_summary.csv` +
拼接段完整报告。给定相同数据与配置,任何人可完整复现任一结果。

## 10. 测试

```bash
uv run pytest tests/ -q      # 61 passed
```

覆盖:因子手算值与无未来函数验证(逐因子截断对比)、指标手算值、IC 极端相关校验、
标签表达式、embargo 边界、目标组合构建、差分订单(含"只换 NVDA→AMZN 不动 AAPL/MSFT"规范用例)、
全部风控拒绝路径、dry-run 零提交、成交/拒单/部分成交处理、重复订单双重防护、
实盘模式拦截、对账各类偏差检出。

## 11. 常见问题

**Q:回测报 `IndexError: index ... out of bounds`?**
回测终点不能等于日历最后一天(qlib 需要"下一日"计算步长)。引擎已自动收缩终点,
自定义调用 `run_backtest` 时无需处理。

**Q:mlflow 报 file-store maintenance 错误?**
mlflow ≥3.15 默认禁用文件后端;`core/qlib_init.py` 已自动设置
`MLFLOW_ALLOW_FILE_STORE=true`(本项目不用 mlflow 追踪)。

**Q:纸面交易报 "prediction date ... is N days old"?**
信号过期保护在起作用。正式运行请先刷新数据重训;离线演练加 `--ignore-staleness`。

**Q:`paper_trade.py` 退出码 2 / 3 是什么?**
2 = 风控拒绝(原因已写日志与运行记录);3 = 该预测日期的 rebalance 已执行过(防重复)。

**Q:Apple Silicon 安装失败?**
确认 `uv sync`(而非手动 pip);pyqlib 0.9.7 有 universal2 轮子,无需编译。
若个别包报错,先看真实报错信息,不要更换包管理器。

**Q:想换股票池?**
改 `universe.market`(数据包内置 `sp500` 等 instruments 文件)与 `universe.benchmark`,
其余零改动。

## 12. 未来扩展

按优先级:① Alpaca→qlib 数据刷新脚本(打通"最新数据→当日信号→当日纸面下单");
② `features.custom_factors: true` 分支的完整对照实验;③ 更多模型
(XGBoost/MLP/LSTM/Transformer)——接口已按 `models/lightgbm_model.py` 约定预留;
④ 组合权重优化(基线刻意保持等权);⑤ 强化学习仅考虑用于仓位/执行层,
且必须在监督基线经纸面验证之后。

## 13. 免责声明

本仓库仅用于量化研究与模拟(纸面)交易。历史回测与模拟成交均不构成对未来收益的
承诺,不构成任何投资建议。切勿在未充分验证前将本系统连接真实资金账户。
