"""市场状态（Regime）诊断与策略推荐引擎。

先判断标的当前处于「趋势市 / 震荡市 / 过渡」以及方向（上行/下行）与波动水平，
再结合各策略的适用类别 + 近期实测表现，给出最适配的策略推荐。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import backtest, indicators as ind
from .strategies import get_strategy, list_strategies

# 各市场状态下"契合"的策略类别。
REGIME_FIT: dict[str, set[str]] = {
    "趋势-上行": {"趋势跟踪", "趋势跟踪 + 风控", "突破", "动量", "基准"},
    "趋势-下行": {"趋势跟踪", "趋势跟踪 + 风控", "动量"},
    "震荡": {"均值回归"},
    "过渡": set(),  # 过渡期无强烈偏好
}

# 与市场状态明显冲突的类别（标记为"不契合"）。
REGIME_AVOID: dict[str, set[str]] = {
    "趋势-上行": {"均值回归"},
    "趋势-下行": {"均值回归", "基准"},
    "震荡": {"趋势跟踪", "趋势跟踪 + 风控", "突破", "动量"},
    "过渡": set(),
}


@dataclass
class Regime:
    trend_label: str       # 趋势市 / 震荡市 / 过渡
    direction: str         # 上行 / 下行 / 中性
    vol_label: str         # 低波动 / 中等波动 / 高波动
    key: str               # 内部键：趋势-上行 / 趋势-下行 / 震荡 / 过渡
    adx: float
    er: float
    annual_vol: float
    vol_pct: float         # 当前波动在历史中的分位
    summary: str           # 一句话总结


def detect_regime(df: pd.DataFrame, window: int = 20, trend_ma: int = 120) -> Regime:
    """诊断当前市场状态（基于最新一段数据）。"""
    close = df["Close"]
    adx_df = ind.adx(df, 14)
    adx_now = float(adx_df["adx"].iloc[-1]) if not np.isnan(adx_df["adx"].iloc[-1]) else 0.0
    er_now = float(ind.efficiency_ratio(close, window).iloc[-1])

    # 趋势 / 震荡判定：ADX 为主，ER 为辅。
    if adx_now >= 25 or er_now >= 0.45:
        trend_label, key_trend = "趋势市", "趋势"
    elif adx_now < 18 and er_now < 0.30:
        trend_label, key_trend = "震荡市", "震荡"
    else:
        trend_label, key_trend = "过渡", "过渡"

    # 方向：价格相对长期均线 + 近期斜率。
    ma = ind.sma(close, trend_ma)
    above = close.iloc[-1] > ma.iloc[-1] if not np.isnan(ma.iloc[-1]) else close.iloc[-1] > close.mean()
    slope = close.iloc[-1] / close.iloc[-min(trend_ma, len(close))] - 1.0
    if above and slope > 0:
        direction = "上行"
    elif not above and slope < 0:
        direction = "下行"
    else:
        direction = "中性"

    # 波动水平：近 window 日年化波动率，及其历史分位。
    daily_ret = close.pct_change()
    realized = daily_ret.rolling(window).std() * np.sqrt(252)
    annual_vol = float(realized.iloc[-1]) if not np.isnan(realized.iloc[-1]) else 0.0
    valid = realized.dropna()
    vol_pct = float((valid < annual_vol).mean()) if len(valid) else 0.5
    if vol_pct >= 0.7:
        vol_label = "高波动"
    elif vol_pct <= 0.3:
        vol_label = "低波动"
    else:
        vol_label = "中等波动"

    # 组合内部键。
    if key_trend == "趋势":
        key = "趋势-上行" if direction != "下行" else "趋势-下行"
    elif key_trend == "震荡":
        key = "震荡"
    else:
        key = "过渡"

    summary = f"{direction}{trend_label} · {vol_label}"
    return Regime(
        trend_label=trend_label, direction=direction, vol_label=vol_label, key=key,
        adx=adx_now, er=er_now, annual_vol=annual_vol, vol_pct=vol_pct, summary=summary,
    )


def _fit_label(category: str, regime_key: str) -> tuple[str, int]:
    """返回 (契合度文字, 契合分数 2/1/0)。"""
    if category in REGIME_FIT.get(regime_key, set()):
        return "高度契合", 2
    if category in REGIME_AVOID.get(regime_key, set()):
        return "不契合", 0
    return "中性", 1


def recommend(
    df: pd.DataFrame,
    allow_short: bool = False,
    cost: dict | None = None,
    perf_window: int = 252,
) -> tuple[Regime, pd.DataFrame]:
    """诊断市场状态并对所有策略评分排序。

    评分 = 契合度(0/1/2) 为主，近一年夏普为辅。
    返回 (Regime, 排序后的推荐表)。
    """
    regime = detect_regime(df)
    cost = cost or {}
    bt_kwargs = {
        "initial_capital": cost.get("capital", 100_000.0),
        "fee_bps": cost.get("fee_bps", 5.0),
        "slippage_bps": cost.get("slippage_bps", 2.0),
    }

    recent = df.iloc[-perf_window:] if len(df) > perf_window else df

    rows = []
    for name in list_strategies():
        strat = get_strategy(name)
        fit_text, fit_score = _fit_label(strat.category, regime.key)
        params = {p.key: p.default for p in strat.params}
        try:
            pos = strat.generate(recent, allow_short=allow_short, **params)
            res = backtest.run_backtest(recent, pos, **bt_kwargs)
            ret = res.stats["累计收益率"]
            sharpe = res.stats["夏普比率"]
            mdd = res.stats["最大回撤"]
        except Exception:  # noqa: BLE001
            ret, sharpe, mdd = 0.0, 0.0, 0.0
        rows.append(
            {
                "策略": name,
                "类别": strat.category,
                "契合度": fit_text,
                "_fit": fit_score,
                "近一年收益": ret,
                "近一年夏普": sharpe,
                "近一年最大回撤": mdd,
                "适用市场": strat.best_market,
            }
        )

    table = pd.DataFrame(rows)
    # 综合评分：契合度优先，其次近一年夏普。
    table["推荐分"] = table["_fit"] * 10 + table["近一年夏普"].clip(-3, 5)
    table = table.sort_values(["推荐分"], ascending=False).drop(columns=["_fit", "推荐分"]).reset_index(drop=True)
    return regime, table
