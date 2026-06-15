"""交易策略模块。

每个策略接收行情 DataFrame 与参数，返回一个"目标仓位"序列 position：
    1  = 满仓做多
    0  = 空仓
   -1  = 满仓做空（仅在允许做空时使用）

position 表示当日收盘后应持有的仓位；回测引擎会自动将其顺延一日，
以收盘价成交，避免使用未来信息（look-ahead bias）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from . import indicators as ind


@dataclass
class ParamSpec:
    """单个策略参数的描述，用于在 UI 中自动生成控件。"""

    key: str
    label: str
    default: float
    min_value: float
    max_value: float
    step: float = 1.0
    is_int: bool = True


@dataclass
class Strategy:
    name: str
    description: str
    func: Callable[..., pd.Series]
    params: list[ParamSpec] = field(default_factory=list)
    category: str = ""          # 策略类别：趋势跟踪 / 均值回归 / 突破 / 动量 / 基准
    best_market: str = ""       # 最适用的市场环境
    avoid_market: str = ""      # 应避免的市场环境
    applicability: str = ""     # 适用条件与使用建议

    def generate(self, df: pd.DataFrame, allow_short: bool = False, **kwargs) -> pd.Series:
        pos = self.func(df, **kwargs)
        if not allow_short:
            pos = pos.clip(lower=0)
        return pos.reindex(df.index).fillna(0.0)


# ---------------------------------------------------------------------------
# 具体策略实现
# ---------------------------------------------------------------------------

def _ma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 60) -> pd.Series:
    """双均线策略：短均线上穿长均线做多，下穿则做空/空仓。"""
    fast, slow = int(fast), int(slow)
    close = df["Close"]
    fast_ma = ind.sma(close, fast)
    slow_ma = ind.sma(close, slow)
    pos = pd.Series(0.0, index=df.index)
    pos[fast_ma > slow_ma] = 1.0
    pos[fast_ma < slow_ma] = -1.0
    return pos


def _rsi_reversion(
    df: pd.DataFrame, window: int = 14, lower: float = 30, upper: float = 70
) -> pd.Series:
    """RSI 均值回归：低于下轨买入持有，高于上轨离场。"""
    window = int(window)
    r = ind.rsi(df["Close"], window)
    pos = pd.Series(float("nan"), index=df.index)
    pos[r < lower] = 1.0
    pos[r > upper] = 0.0
    # 在信号之间维持上一次仓位。
    return pos.ffill().fillna(0.0)


def _macd_trend(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """MACD 趋势：MACD 线上穿信号线做多，下穿做空/离场。"""
    m = ind.macd(df["Close"], int(fast), int(slow), int(signal))
    pos = pd.Series(0.0, index=df.index)
    pos[m["macd"] > m["signal"]] = 1.0
    pos[m["macd"] < m["signal"]] = -1.0
    return pos


def _bollinger_reversion(
    df: pd.DataFrame, window: int = 20, num_std: float = 2.0
) -> pd.Series:
    """布林带均值回归：跌破下轨买入，回到中轨离场。"""
    bands = ind.bollinger_bands(df["Close"], int(window), float(num_std))
    close = df["Close"]
    pos = pd.Series(float("nan"), index=df.index)
    pos[close < bands["lower"]] = 1.0
    pos[close >= bands["mid"]] = 0.0
    return pos.ffill().fillna(0.0)


def _momentum(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """动量策略：过去 window 日收益为正则做多，否则做空/离场。"""
    mom = ind.momentum(df["Close"], int(window))
    pos = pd.Series(0.0, index=df.index)
    pos[mom > 0] = 1.0
    pos[mom < 0] = -1.0
    return pos


def _donchian_breakout(
    df: pd.DataFrame, entry: int = 20, exit: int = 10
) -> pd.Series:
    """唐奇安通道突破（海龟入场）：突破 entry 日新高做多，跌破 exit 日新低离场/做空。"""
    up = df["High"].rolling(int(entry)).max().shift(1)
    dn = df["Low"].rolling(int(exit)).min().shift(1)
    close = df["Close"]
    pos = pd.Series(float("nan"), index=df.index)
    pos[close > up] = 1.0
    pos[close < dn] = -1.0
    return pos.ffill().fillna(0.0)


def _keltner_breakout(
    df: pd.DataFrame, window: int = 20, atr_window: int = 10, mult: float = 2.0
) -> pd.Series:
    """肯特纳通道突破：收盘突破上轨做多，跌破下轨离场/做空。"""
    k = ind.keltner(df, int(window), int(atr_window), float(mult))
    close = df["Close"]
    pos = pd.Series(float("nan"), index=df.index)
    pos[close > k["upper"]] = 1.0
    pos[close < k["lower"]] = -1.0
    return pos.ffill().fillna(0.0)


def _atr_trailing(
    df: pd.DataFrame, ma_window: int = 50, atr_window: int = 14, mult: float = 3.0
) -> pd.Series:
    """ATR 跟踪止损趋势（吊灯出场）：价格上穿均线做多，并以最高价回撤 mult 倍 ATR 跟踪止损。

    仅做多/空仓（long-flat）。
    """
    close = df["Close"]
    ma = ind.sma(close, int(ma_window))
    a = ind.atr(df, int(atr_window))

    pos = pd.Series(0.0, index=df.index)
    in_pos = False
    stop = 0.0
    peak = 0.0
    c = close.to_numpy()
    m = ma.to_numpy()
    av = a.to_numpy()

    for i in range(len(c)):
        if pd.isna(m[i]) or pd.isna(av[i]):
            continue
        if not in_pos:
            if c[i] > m[i]:
                in_pos = True
                peak = c[i]
                stop = c[i] - mult * av[i]
        else:
            peak = max(peak, c[i])
            stop = max(stop, peak - mult * av[i])
            if c[i] < stop:
                in_pos = False
        pos.iloc[i] = 1.0 if in_pos else 0.0
    return pos


def _trend_momentum(
    df: pd.DataFrame, ma_window: int = 100, mom_window: int = 60
) -> pd.Series:
    """趋势+动量双确认：价在长均线上方且动量为正才做多；双双转负则离场/做空。"""
    close = df["Close"]
    ma = ind.sma(close, int(ma_window))
    mom = ind.momentum(close, int(mom_window))
    long_cond = (close > ma) & (mom > 0)
    short_cond = (close < ma) & (mom < 0)
    pos = pd.Series(0.0, index=df.index)
    pos[long_cond] = 1.0
    pos[short_cond] = -1.0
    return pos


def _zscore_reversion(
    df: pd.DataFrame, window: int = 20, entry: float = 2.0, exit: float = 0.5
) -> pd.Series:
    """Z-Score 均值回归：价格偏离均值 entry 倍标准差时反向开仓，回到 exit 倍内离场。"""
    close = df["Close"]
    ma = ind.sma(close, int(window))
    std = close.rolling(int(window), min_periods=int(window)).std()
    z = (close - ma) / std.replace(0.0, pd.NA)
    pos = pd.Series(float("nan"), index=df.index)
    pos[z < -entry] = 1.0          # 超跌，做多
    pos[z > entry] = -1.0          # 超涨，做空
    pos[z.abs() < exit] = 0.0      # 回归均值，离场
    return pos.ffill().fillna(0.0)


def _buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """买入持有基准。"""
    return pd.Series(1.0, index=df.index)


# ---------------------------------------------------------------------------
# 策略注册表
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Strategy] = {
    "双均线交叉": Strategy(
        name="双均线交叉",
        description="短期均线上穿长期均线时做多，下穿时离场/做空。经典趋势跟踪策略。",
        func=_ma_cross,
        category="趋势跟踪",
        best_market="单边趋势行情（明显的牛市或熊市）",
        avoid_market="横盘震荡市（均线频繁交叉，反复止损）",
        applicability="适合波动有方向、趋势能持续的标的（如宽基指数、龙头股）。建议短/长周期拉开差距（如 20/60、50/200），减少假信号。",
        params=[
            ParamSpec("fast", "短期均线周期", 20, 2, 120, 1),
            ParamSpec("slow", "长期均线周期", 60, 5, 250, 1),
        ],
    ),
    "RSI 均值回归": Strategy(
        name="RSI 均值回归",
        description="RSI 跌破下轨时买入，升破上轨时离场。适合震荡行情的逆势策略。",
        func=_rsi_reversion,
        category="均值回归",
        best_market="区间震荡市、超跌反弹",
        avoid_market="单边强趋势（会过早卖出或在下跌中接刀）",
        applicability="适合估值稳定、价格围绕中枢上下波动的标的。强趋势中应配合趋势过滤使用，避免逆势硬扛。",
        params=[
            ParamSpec("window", "RSI 周期", 14, 2, 60, 1),
            ParamSpec("lower", "超卖阈值", 30, 5, 50, 1),
            ParamSpec("upper", "超买阈值", 70, 50, 95, 1),
        ],
    ),
    "MACD 趋势": Strategy(
        name="MACD 趋势",
        description="MACD 线上穿信号线做多，下穿离场/做空。趋势确认策略。",
        func=_macd_trend,
        category="趋势跟踪",
        best_market="中期趋势明确、波段清晰的行情",
        avoid_market="高频窄幅震荡（金叉死叉过于频繁）",
        applicability="适合有中等波段节奏的标的；信号比双均线更平滑但有滞后。震荡剧烈时交易成本会侵蚀收益。",
        params=[
            ParamSpec("fast", "快线周期", 12, 2, 50, 1),
            ParamSpec("slow", "慢线周期", 26, 5, 100, 1),
            ParamSpec("signal", "信号线周期", 9, 2, 50, 1),
        ],
    ),
    "布林带回归": Strategy(
        name="布林带回归",
        description="价格跌破下轨买入，回升至中轨离场。均值回归策略。",
        func=_bollinger_reversion,
        category="均值回归",
        best_market="震荡市、价格围绕均值波动",
        avoid_market="趋势性下跌（不断跌破下轨，越买越亏）",
        applicability="适合波动率相对稳定、有均值回复特性的标的。趋势行情中建议关闭或加趋势过滤。",
        params=[
            ParamSpec("window", "均线周期", 20, 5, 100, 1),
            ParamSpec("num_std", "标准差倍数", 2.0, 0.5, 4.0, 0.5, is_int=False),
        ],
    ),
    "动量策略": Strategy(
        name="动量策略",
        description="过去 N 日收益为正则做多，为负则离场/做空。跨周期动量策略。",
        func=_momentum,
        category="动量",
        best_market="趋势延续性强、有惯性的标的或指数",
        avoid_market="频繁反转的震荡市",
        applicability="适合成长股、强势指数等动量效应显著的标的。回看周期越长越稳健但越滞后（常用 60~120 日）。",
        params=[
            ParamSpec("window", "回看周期", 60, 5, 250, 1),
        ],
    ),
    "唐奇安通道突破（海龟）": Strategy(
        name="唐奇安通道突破（海龟）",
        description="突破过去 N 日最高价做多，跌破过去 M 日最低价离场/做空。经典海龟入场。",
        func=_donchian_breakout,
        category="突破",
        best_market="趋势启动、波动率放大、有大行情的标的",
        avoid_market="窄幅横盘（假突破多，来回打脸）",
        applicability="适合容易走出大趋势的品种（指数、商品类 ETF）。胜率通常不高但盈亏比大，靠少数大趋势赚钱，需严格执行。",
        params=[
            ParamSpec("entry", "入场突破周期", 20, 5, 120, 1),
            ParamSpec("exit", "离场突破周期", 10, 3, 60, 1),
        ],
    ),
    "肯特纳通道突破": Strategy(
        name="肯特纳通道突破",
        description="以 EMA 为中轨、上下各若干倍 ATR；突破上轨做多，跌破下轨离场/做空。",
        func=_keltner_breakout,
        category="突破",
        best_market="趋势行情，且波动率有规律地放大",
        avoid_market="无方向的低波动横盘",
        applicability="带波动率（ATR）过滤的突破，比纯价格突破更抗噪。ATR 倍数越大越不易触发，信号更可靠但更滞后。",
        params=[
            ParamSpec("window", "中轨 EMA 周期", 20, 5, 100, 1),
            ParamSpec("atr_window", "ATR 周期", 10, 3, 50, 1),
            ParamSpec("mult", "ATR 倍数", 2.0, 0.5, 5.0, 0.5, is_int=False),
        ],
    ),
    "ATR 跟踪止损趋势": Strategy(
        name="ATR 跟踪止损趋势",
        description="价格上穿均线做多，并以最高价回撤若干倍 ATR 作为吊灯跟踪止损（仅做多）。",
        func=_atr_trailing,
        category="趋势跟踪 + 风控",
        best_market="单边上涨趋势，且希望严格控制回撤",
        avoid_market="剧烈震荡（容易被止损扫出后又拉升）",
        applicability="唯一自带动态止损的趋势策略，能让利润奔跑、亏损有限。适合追求'拿得住趋势又控回撤'的场景。止损倍数越大越不易被甩下车。",
        params=[
            ParamSpec("ma_window", "趋势均线周期", 50, 5, 200, 1),
            ParamSpec("atr_window", "ATR 周期", 14, 3, 50, 1),
            ParamSpec("mult", "止损 ATR 倍数", 3.0, 1.0, 6.0, 0.5, is_int=False),
        ],
    ),
    "趋势+动量双确认": Strategy(
        name="趋势+动量双确认",
        description="价在长期均线上方且动量为正才做多，双双转负才做空。过滤震荡的稳健趋势策略。",
        func=_trend_momentum,
        category="趋势跟踪",
        best_market="中长期趋势行情",
        avoid_market="方向不明的反复震荡",
        applicability="双重条件（趋势+动量）同时满足才进场，信号少而精，能有效过滤假突破。适合稳健型、不想频繁交易的使用者。",
        params=[
            ParamSpec("ma_window", "趋势均线周期", 100, 20, 250, 5),
            ParamSpec("mom_window", "动量回看周期", 60, 10, 200, 5),
        ],
    ),
    "Z-Score 均值回归": Strategy(
        name="Z-Score 均值回归",
        description="价格偏离均值达 N 倍标准差时反向开仓（超跌做多/超涨做空），回归均值附近离场。",
        func=_zscore_reversion,
        category="均值回归",
        best_market="平稳、有强均值回复特性的标的（如配对价差、低波蓝筹）",
        avoid_market="趋势性突破行情（偏离会持续扩大）",
        applicability="统计套利思路，用标准差量化'偏离程度'。开仓阈值越大越保守、交易越少。最适合本身就在区间内波动的标的。",
        params=[
            ParamSpec("window", "均值窗口", 20, 5, 120, 1),
            ParamSpec("entry", "开仓阈值(σ)", 2.0, 0.5, 4.0, 0.5, is_int=False),
            ParamSpec("exit", "离场阈值(σ)", 0.5, 0.0, 2.0, 0.25, is_int=False),
        ],
    ),
    "买入持有（基准）": Strategy(
        name="买入持有（基准）",
        description="始终满仓持有，作为对比基准。",
        func=_buy_and_hold,
        category="基准",
        best_market="长期向上的优质资产、宽基指数",
        avoid_market="长期下跌或长期横盘的个股",
        applicability="不择时、不止损，吃满 beta。对长期上行的优质资产/指数往往是最难被战胜的基准——任何策略都应先跑赢它。",
        params=[],
    ),
}


def list_strategies() -> list[str]:
    return list(REGISTRY.keys())


def get_strategy(name: str) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"未知策略: {name}")
    return REGISTRY[name]
