"""当日交易信号扫描模块。

对一组自选股应用同一策略，比较最近两日的目标仓位，
判断每只股票"今天"应执行的动作（买入 / 卖出 / 加空 / 平仓 / 维持）。
"""

from __future__ import annotations

import pandas as pd

from .strategies import get_strategy


def _action(prev: float, curr: float) -> str:
    """根据仓位变化生成动作描述。"""
    if curr == prev:
        if curr > 0:
            return "持有多头"
        if curr < 0:
            return "持有空头"
        return "空仓观望"
    # 发生变化。
    if prev <= 0 and curr > 0:
        return "🟢 买入开多"
    if prev >= 0 and curr < 0:
        return "🔴 卖出开空"
    if prev > 0 and curr == 0:
        return "🟡 平多离场"
    if prev < 0 and curr == 0:
        return "🟡 平空离场"
    return "调整仓位"


def scan(
    data: dict[str, pd.DataFrame],
    strategy_name: str,
    params: dict | None = None,
    allow_short: bool = False,
) -> pd.DataFrame:
    """扫描自选股，返回每只股票的最新信号表。

    data: {代码: 行情DataFrame}
    """
    params = params or {}
    strat = get_strategy(strategy_name)

    rows = []
    for ticker, df in data.items():
        if df is None or df.empty or len(df) < 2:
            rows.append({"代码": ticker, "最新日期": "-", "最新价": None,
                         "今日动作": "数据不足", "目标仓位": "-", "昨日仓位": "-"})
            continue

        pos = strat.generate(df, allow_short=allow_short, **params)
        curr = float(pos.iloc[-1])
        prev = float(pos.iloc[-2])
        last_dt = df.index[-1]
        last_price = float(df["Close"].iloc[-1])

        rows.append(
            {
                "代码": ticker,
                "最新日期": pd.Timestamp(last_dt).strftime("%Y-%m-%d"),
                "最新价": round(last_price, 2),
                "今日动作": _action(prev, curr),
                "目标仓位": _pos_label(curr),
                "昨日仓位": _pos_label(prev),
            }
        )

    df_out = pd.DataFrame(rows)
    # 把有动作变化的排在前面。
    df_out["_changed"] = df_out["今日动作"].str.contains("🟢|🔴|🟡", regex=True)
    df_out = df_out.sort_values("_changed", ascending=False).drop(columns="_changed").reset_index(drop=True)
    return df_out


def _pos_label(pos: float) -> str:
    if pos > 0:
        return "多头"
    if pos < 0:
        return "空头"
    return "空仓"
