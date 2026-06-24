# 美股量化交易策略回测平台

一个**自用**的美股量化交易策略回测工具。输入美股代码、选择策略与参数，即可一键回测并查看完整绩效报告。全中文界面。

> ⚠️ 本工具仅用于学习与个人研究，所有回测结果基于历史数据，**不构成任何投资建议**。

---

## 功能特性

界面分为十个标签页：

- 📋 **一键体检**：输入一只股票，自动跑完整条决策链——判市 → 推荐策略 → 自动参数寻优 → 样本外验证 → 赚钱概率 → **综合评分（0~100）与可执行结论**，并列出风险提示。一个按钮拿到完整决策报告。
- 🧭 **策略推荐**：自动诊断标的当前是**趋势市还是震荡市**（含方向与波动水平，基于 ADX + 效率比 + 均线/波动率分位），再结合策略适用条件与近一年实测表现，给出最适配的策略排名（高度契合 / 中性 / 不契合，绿/红高亮）。

- 🎯 **单策略回测**：选定策略与参数，查看 K 线、净值曲线、绩效指标与交易明细。
- 🔍 **参数寻优**：对策略参数做网格搜索，按目标指标（夏普/收益/卡尔玛等）自动找出最优组合，并以热力图展示参数敏感性。
- 📊 **策略对比**：用默认参数把多个策略放在同一标的与区间上横向对比净值与绩效。
- 🧺 **组合回测（含风险控制）**：对一篮子标的应用同一策略并每日再平衡。支持多种权重方案（等权 / 自定义 / 逆波动率 / 风险平价）、**单标的权重上限**（控制集中度）、以及**目标波动率**（动态调节整体仓位，把组合年化波动率控制在设定值附近）。输出组合净值、回撤、权重饼图、动态仓位系数曲线与各标的表现。
- 🔔 **信号扫描**：对自选股列表应用策略，列出每只股票最新交易日应执行的买/卖/持有动作，并高亮今日触发信号变动的标的，可导出 CSV。
- 🧪 **样本外验证**：用「单次训练/测试划分」或「滚动前向（Walk-Forward）」检验参数是否过拟合——先在历史前段寻优，再用最优参数在没见过的后段数据上交易，对比样本内/样本外表现差距。
- 💼 **模拟交易（Paper Trading）**：本地虚拟账户、零资金风险。按策略信号在多头标的间等权调仓，持久化记录持仓、现金、成交流水与权益曲线，长期跟踪模拟盘表现。可被定时脚本自动驱动。
- 💰 **赚钱概率**：查看每个策略的**适用条件**（类别、最适用/应避免的市场环境），并用历史数据测算「赚钱概率」——随机进场持有 1 月/3 月/6 月/1 年为正的概率、单笔胜率与盈亏比、跑赢基准概率，以及在一篮子标的上的盈利占比。

通用能力：

- 📈 **实时行情**：通过 Yahoo Finance 拉取美股历史日线数据（自带缓存）。
- 🧠 **内置 11 种策略**（每个都标注了适用市场与使用建议）：双均线交叉、RSI 均值回归、MACD 趋势、布林带回归、动量策略、唐奇安通道突破（海龟）、肯特纳通道突破、ATR 跟踪止损趋势、趋势+动量双确认、Z-Score 均值回归、买入持有基准。
- ⚙️ **参数可调**：每个策略的参数都可在界面上用滑块实时调整。
- 🔁 **向量化回测引擎**：自动顺延信号一日成交，避免使用未来信息；支持手续费与滑点。
- 📋 **完整绩效指标**：累计/年化收益、年化波动率、夏普、索提诺、卡尔玛、最大回撤、交易次数、胜率。
- 📥 **交易明细导出**：每一笔开平仓记录可下载为 CSV。
- 🟢🔴 **做多/做空**：可选是否允许做空。

---

## 快速开始

### 方式一：一键脚本（推荐）

```bash
cd /Users/Admin/Desktop/666
chmod +x run.sh
./run.sh
```

脚本会自动创建虚拟环境、安装依赖并启动应用。

### 方式二：手动运行

```bash
cd /Users/Admin/Desktop/666
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

启动后浏览器会自动打开 `http://localhost:8501`。

**部署到云端 / 装进 iPhone**：见 [`DEPLOY.md`](DEPLOY.md)（GitHub 推送 → Streamlit Cloud → iOS 壳）。

---

## 使用说明

1. 在左侧 **全局配置** 中填写美股代码（如 `AAPL`、`NVDA`、`SPY`）、起止日期、初始资金、手续费/滑点，以及是否允许做空。这些设置对三个标签页通用。
2. 切换到对应标签页：
   - **一键体检**：点击「开始一键体检」，几秒内得到该股票的市场状态、推荐策略、最优参数、样本外验证、赚钱概率与综合评分结论（最省事的入口，建议从这里开始）。
   - **策略推荐**：点击「诊断市场并推荐策略」，即可看到当前市场状态判断与最适配的策略排名。
   - **单策略回测**：选策略 → 调参数 → 点击「运行回测」。
   - **参数寻优**：选策略 → 设置每个参数的最小/最大/步长 → 选优化目标 → 点击「开始寻优」，得到最优参数、热力图与完整结果表。
   - **策略对比**：勾选要对比的策略 → 点击「开始对比」，得到净值曲线叠加图与绩效对比表。
   - **组合回测**：输入多个标的（逗号分隔）→ 选权重方案（等权/自定义/逆波动率/风险平价）→ 选策略与参数 → 按需设置单标的权重上限与目标波动率 → 点击「运行组合回测」。
   - **信号扫描**：输入自选股列表 → 选策略与参数 → 点击「扫描今日信号」，查看今天该买/卖/持有哪些标的。
   - **样本外验证**：选策略 → 设参数搜索范围 → 选验证方式（单次划分 / 滚动前向）→ 点击「开始验证」，对比样本内外表现，识别过拟合。
   - **模拟交易**：新建虚拟账户 → 输入自选股、选策略 → 点击「按今日信号调仓」，账户会持久化保存，可每天反复执行、长期跟踪。
   - **赚钱概率**：选策略查看其适用条件 → 选「单标的」测滚动持有期赚钱概率，或选「多标的」测一篮子标的的盈利占比。

---

## 策略适用条件一览

| 策略 | 类别 | 最适用市场 | 应避免 |
|---|---|---|---|
| 双均线交叉 | 趋势跟踪 | 单边趋势（牛/熊市） | 横盘震荡 |
| MACD 趋势 | 趋势跟踪 | 中期趋势明确 | 高频窄幅震荡 |
| 趋势+动量双确认 | 趋势跟踪 | 中长期趋势 | 方向不明的震荡 |
| ATR 跟踪止损趋势 | 趋势 + 风控 | 单边上涨且需控回撤 | 剧烈震荡 |
| 动量策略 | 动量 | 趋势惯性强的标的/指数 | 频繁反转 |
| 唐奇安通道突破（海龟） | 突破 | 趋势启动、波动放大 | 窄幅横盘（假突破多） |
| 肯特纳通道突破 | 突破 | 趋势 + 波动率放大 | 低波动横盘 |
| RSI 均值回归 | 均值回归 | 区间震荡、超跌反弹 | 单边强趋势 |
| 布林带回归 | 均值回归 | 震荡市、绕均值波动 | 趋势性下跌 |
| Z-Score 均值回归 | 均值回归 | 平稳、强均值回复标的 | 趋势性突破 |
| 买入持有（基准） | 基准 | 长期向上的优质资产/指数 | 长期下跌/横盘个股 |

> 经验法则：**趋势/突破/动量类**靠少数大行情赚钱（胜率不一定高，但盈亏比大），适合趋势市；**均值回归类**靠高频小赢积累（胜率高、盈亏比小），适合震荡市。先用「💰 赚钱概率」和「🧪 样本外验证」两个页签验证，再决定是否实盘。

### 「赚钱概率」怎么算的？

- **滚动持有期赚钱概率**：把历史切成大量重叠窗口（如所有可能的「持有 3 个月」区间），统计其中账户收益为正的比例 —— 回答「我随便挑一天按这个策略进场、持有 N 天，赚钱的可能性多大」。
- **单笔胜率 / 盈亏比**：每笔开平仓盈利的比例，以及平均盈利 ÷ 平均亏损。
- **跑赢基准概率**：同样按滚动窗口，策略收益高于「买入持有」的比例。
- **多标的盈利占比**：把策略用到一篮子股票上，盈利标的数 / 总标的数。

> ⚠️ 这些概率全部基于**历史回测**，是「过去发生的频率」，**不是对未来的承诺**。务必结合样本外验证一起看。

---

## 定时自动扫描信号（无需打开网页）

除了网页里的「信号扫描」标签页，还提供一个独立命令行脚本 `scan_daily.py`，可由系统定时执行，在触发买卖信号时弹**桌面通知**（macOS）并可选**发邮件**。

### 1. 配置

编辑 `scan_config.json`：

```json
{
  "watchlist": ["AAPL", "MSFT", "NVDA", "SPY"],
  "strategy": "双均线交叉",
  "params": {"fast": 20, "slow": 60},
  "allow_short": false,
  "lookback_days": 400,
  "notify": { "desktop": true, "email": { "enabled": false } }
}
```

如需邮件，把 `email.enabled` 设为 `true`，填好 SMTP 信息，并把邮箱授权码放进环境变量（默认变量名 `SCAN_EMAIL_PASSWORD`，可在配置里改 `password_env`）。

如需让定时扫描**同时自动驱动模拟账户调仓**，把 `paper.enabled` 设为 `true`——每次扫描都会按当日信号对 `paper_account.json` 调仓并打印权益，相当于全自动模拟盘。

### 2. 手动运行

```bash
source .venv/bin/activate
python scan_daily.py            # 正常扫描并通知
python scan_daily.py --dry-run  # 只打印结果，不发通知
```

每次扫描结果都会追加到 `scan_history.csv`。

### 3. 定时执行（macOS launchd）

仓库内 `com.quant.scan.plist` 是现成模板（默认每天 16:30 运行）：

```bash
cp com.quant.scan.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.quant.scan.plist
launchctl start com.quant.scan   # 立即测试一次
```

> 也可以用 cron：`crontab -e` 后加入
> `30 16 * * 1-5 cd /Users/Admin/Desktop/666 && .venv/bin/python scan_daily.py`

---

## 方向性事件策略：恐慌反弹做多（全市场寻优）

除网页平台外，仓库还含一套「单日异动（涨跌幅榜）后续规律」的研究与实盘脚本。基于真实 OHLCV、最近 5 年、全市场 400 只、含成本与次日开盘真实入场，结论：

- **单纯「涨/跌 >阈值」本身几乎无方向边缘**（胜率 47–53%）；**追涨必亏**（暴涨延续/过热做空全市场全亏，禁用）。
- **全市场唯一稳健正期望 = 恐慌反弹做多**：已深跌≥30% 的票当日再暴跌≥10% 放恐慌盘 → 次日开盘做多。样本外年化 +54~77%、胜率 54~63%、回撤 -18%。
- 「年化>100% + 回撤<10% + 胜率>90%」三目标**不可同时达成**（768 组寻优全部 0/3）；要「高胜率+低回撤」请用期权卖方（Tier A CSP 96% 胜率 / -5% 回撤）。

完整结论与参数前沿见 [`research/extreme_move_strategy_notes.md`](research/extreme_move_strategy_notes.md)。

```bash
python3 panic_rebound_daily.py                 # 今日恐慌反弹候选 + 止盈止损价
python3 strategy_daily.py                       # 全系统每日 Top3（已含恐慌反弹）
python3 research/extreme15_pattern.py --pool broad --threshold 10 --optimize  # 重跑寻优
```

定时：双击 `恐慌反弹做多_开启定时.command`（macOS launchd，收盘后自动跑）。

---

## 运行测试

核心引擎（指标、回测、绩效、策略、寻优、组合风控、信号、模拟账户、样本外验证、赚钱概率、市场状态、一键体检）均有单元测试覆盖，验证了「无未来函数」「成本影响」「权重归一/上限」「目标波动率」「账户调仓幂等」等关键不变量。

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt   # 安装 pytest
pytest                                 # 运行全部测试
```

---

## 装进 iPhone / iPad（原生 App）

`ios/` 目录内含一个 SwiftUI + WKWebView 的原生壳工程，可把本平台打包成 iOS App。

思路：Streamlit 需要服务器进程，所以先把本应用部署到云端（或自托管）拿到网址，再用 iOS 壳加载它。

简要步骤（完整版见 [`ios/README.md`](ios/README.md)）：

1. **部署后端**：把项目推到 GitHub，用 [Streamlit Community Cloud](https://share.streamlit.io) 一键部署 `app.py`，得到 `https://xxx.streamlit.app`（也可在本机/局域网自托管）。
2. **生成工程**：`brew install xcodegen && cd ios && xcodegen generate`。
3. **填地址**：把上面网址写进 `ios/Sources/Config.swift` 的 `serverURLString`。
4. **装机**：Xcode 打开工程 → 选你的签名 Team 与唯一 Bundle ID → 连接设备 → Run。

> 免费 Apple ID 即可装到自己真机（证书 7 天过期，重 Run 即可）；付费开发者账号（$99/年）证书 1 年且支持 TestFlight 分发。

---

## 项目结构

```
666/
├── app.py                 # Streamlit 应用入口（UI + 图表）
├── scan_daily.py          # 定时自动信号扫描脚本（CLI + 桌面通知/邮件）
├── scan_config.json       # 定时扫描配置（自选股/策略/通知方式）
├── com.quant.scan.plist   # macOS launchd 定时任务模板
├── requirements.txt       # Python 依赖
├── requirements-dev.txt   # 开发/测试依赖（pytest）
├── tests/                 # 单元测试（pytest）
├── pytest.ini             # pytest 配置
├── run.sh                 # 一键启动脚本
├── .streamlit/config.toml # Streamlit 主题与服务配置（用于云端部署）
├── assets/icon.png        # 网页页面图标（与 iOS App 图标同款）
├── ios/                   # iOS 原生壳工程（SwiftUI + WKWebView + XcodeGen，含 App 图标）
├── README.md
└── quant/                 # 核心逻辑包
    ├── data.py            # 行情数据获取与规范化
    ├── indicators.py      # 技术指标（SMA/EMA/RSI/MACD/布林带/动量/ATR/唐奇安/肯特纳/ADX/效率比）
    ├── strategies.py      # 策略定义与注册表
    ├── backtest.py        # 向量化回测引擎
    ├── metrics.py         # 绩效指标计算
    ├── optimize.py        # 参数网格寻优 + 多策略对比
    ├── portfolio.py       # 多标的组合回测
    ├── signals.py         # 自选股当日信号扫描
    ├── validation.py      # 样本外验证（训练/测试划分 + Walk-Forward）
    ├── paper.py           # 本地模拟交易账户引擎
    ├── probability.py     # 赚钱概率分析（滚动持有期/胜率/盈利占比）
    ├── regime.py          # 市场状态诊断 + 智能策略推荐
    └── report.py          # 一键体检：全流程编排 + 综合评分与结论
```

---

## 如何新增自定义策略

在 `quant/strategies.py` 中：

1. 编写一个函数，接收 `df`（含 `Close` 列）与参数，返回一个**目标仓位序列**（取值 `1` 做多 / `0` 空仓 / `-1` 做空）。
2. 在 `REGISTRY` 字典中注册它，并用 `ParamSpec` 描述其可调参数。

示例：

```python
def _my_strategy(df, threshold=0.0):
    ret = df["Close"].pct_change(5)
    pos = pd.Series(0.0, index=df.index)
    pos[ret > threshold] = 1.0
    return pos

REGISTRY["我的策略"] = Strategy(
    name="我的策略",
    description="过去 5 日涨幅超过阈值则买入。",
    func=_my_strategy,
    params=[ParamSpec("threshold", "涨幅阈值", 0.0, -0.1, 0.1, 0.01, is_int=False)],
)
```

保存后刷新页面即可在策略下拉框中看到。

---

## 行情数据源（可切换）

默认用 **Yahoo Finance**（免费、零配置，但数据质量一般），也支持切换到**专业数据源**：

| 数据源 | 说明 | 配置 |
|---|---|---|
| **Polygon.io** | 交易所级聚合数据，质量高（推荐） | 注册免费 Key：[polygon.io](https://polygon.io) |
| **Alpaca Markets** | 券商级行情，免费开户即用 | API Key/Secret：[alpaca.markets](https://alpaca.markets) |
| **Yahoo Finance** | 免费备用，无需配置 | 无 |

### 配置方式

本地复制 `.streamlit/secrets.toml.example` 为 `.streamlit/secrets.toml`（Streamlit Cloud 在控制台 Secrets 填写）：

```toml
[data]
provider = "polygon"          # 可选 polygon / alpaca / yahoo
polygon_api_key = "你的Key"
```

或用环境变量（适合定时脚本 / CI）：

```bash
export DATA_PROVIDER=polygon
export POLYGON_API_KEY=xxx
```

> 未配置密钥时会**自动回退到 Yahoo**，不会报错。当前数据源会显示在侧边栏。

---

## 常见问题

- **数据拉取失败 / 超时**：Yahoo 偶有不稳定，可配置 Polygon/Alpaca 专业源（见上）；或检查网络/代理。
- **代码无效**：请确认填写的是有效的美股代码。
- **首次启动较慢**：因为需要安装依赖；之后启动会很快。

---

## 技术栈

- [Streamlit](https://streamlit.io/) — 交互式 Web 界面
- 行情数据 — [Polygon.io](https://polygon.io) / [Alpaca](https://alpaca.markets) / [yfinance](https://github.com/ranaroussi/yfinance)（可切换）
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) — 数据处理与指标计算
- [Plotly](https://plotly.com/python/) — 交互式图表
