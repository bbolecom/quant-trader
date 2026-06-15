"""赚钱概率分析模块。

基于历史回测，从多个角度量化一个策略"赚钱"的概率：

1. 滚动持有期为正的概率：把历史切成大量重叠的持有窗口（如持有 1 个月 / 3 个月 /
   半年 / 1 年），统计其中收益为正的比例 —— 回答"我随便挑一天进场、持有 H 天，
   赚钱的概率有多大"。
2. 跑赢买入持有的概率：同样按滚动窗口，统计策略收益高于买入持有的比例。
3. 单笔交易胜率：每一笔开平仓盈利的比例。
4. 多标的盈利占比：把策略用到一篮子标的上，统计盈利标的、跑赢基准标的的比例。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest
from .strategies import get_strategy

# 常用持有期（交易日）。
HORIZONS = {
    "持有 1 个月": 21,
    "持有 3 个月": 63,
    "持有 6 个月": 126,
    "持有 1 年": 252,
}


def _bt_kwargs(cost: dict | None) -> dict:
    cost = cost or {}
    return {
        "initial_capital": cost.get("capital", 100_000.0),
        "fee_bps": cost.get("fee_bps", 5.0),
        "slippage_bps": cost.get("slippage_bps", 2.0),
    }


def rolling_positive_prob(returns: pd.Series, horizon: int) -> tuple[float, int]:
    """滚动持有 horizon 日、累计收益为正的概率。返回 (概率, 样本数)。"""
    if returns.empty or len(returns) <= horizon:
        return 0.0, 0
    logr = np.log1p(returns.fillna(0.0))
    rolled = logr.rolling(horizon).sum().dropna()
    if rolled.empty:
        return 0.0, 0
    return float((rolled > 0).mean()), int(len(rolled))


def rolling_beat_prob(strat_ret: pd.Series, bench_ret: pd.Series, horizon: int) -> float:
    """滚动持有 horizon 日、策略累计收益跑赢基准的概率。"""
    if strat_ret.empty or len(strat_ret) <= horizon:
        return 0.0
    s = np.log1p(strat_ret.fillna(0.0)).rolling(horizon).sum()
    b = np.log1p(bench_ret.fillna(0.0)).rolling(horizon).sum()
    diff = (s - b).dropna()
    if diff.empty:
        return 0.0
    return float((diff > 0).mean())


def analyze_single(
    df: pd.DataFrame,
    strategy_name: str,
    params: dict | None = None,
    allow_short: bool = False,
    cost: dict | None = None,
) -> dict:
    """单标的赚钱概率分析。"""
    params = params or {}
    strat = get_strategy(strategy_name)
    pos = strat.generate(df, allow_short=allow_short, **params)
    res = backtest.run_backtest(df, pos, **_bt_kwargs(cost))

    bench_ret = df["Close"].pct_change().fillna(0.0)

    horizon_rows = []
    for label, h in HORIZONS.items():
        p_pos, n = rolling_positive_prob(res.returns, h)
        p_beat = rolling_beat_prob(res.returns, bench_ret, h)
        if n > 0:
            horizon_rows.append(
                {"持有期": label, "赚钱概率": p_pos, "跑赢基准概率": p_beat, "样本数": n}
            )

    trade_ret = res.trades["收益率"] if not res.trades.empty else pd.Series(dtype=float)
    win_rate = float((trade_ret > 0).mean()) if len(trade_ret) else 0.0
    avg_win = float(trade_ret[trade_ret > 0].mean()) if (trade_ret > 0).any() else 0.0
    avg_loss = float(trade_ret[trade_ret < 0].mean()) if (trade_ret < 0).any() else 0.0
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
    # 凯利公式估算的最优下注比例（仅作参考）。
    kelly = (win_rate - (1 - win_rate) / payoff) if payoff > 0 else 0.0

    return {
        "horizons": pd.DataFrame(horizon_rows),
        "win_rate": win_rate,
        "num_trades": int(len(trade_ret)),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff": payoff,
        "kelly": max(0.0, kelly),
        "daily_positive": float((res.returns > 0).mean()),
        "total_return": res.stats["累计收益率"],
        "benchmark_return": res.stats["基准收益率"],
        "sharpe": res.stats["夏普比率"],
        "max_drawdown": res.stats["最大回撤"],
        "result": res,
    }


def analyze_basket(
    data: dict[str, pd.DataFrame],
    strategy_name: str,
    params: dict | None = None,
    allow_short: bool = False,
    cost: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    """多标的赚钱概率分析：在一篮子标的上跑同一策略，统计盈利/跑赢比例。

    返回 (汇总字典, 各标的明细表)。
    """
    params = params or {}
    strat = get_strategy(strategy_name)
    bt_kwargs = _bt_kwargs(cost)

    rows = []
    for ticker, df in data.items():
        pos = strat.generate(df, allow_short=allow_short, **params)
        res = backtest.run_backtest(df, pos, **bt_kwargs)
        ret = res.stats["累计收益率"]
        bench = res.stats["基准收益率"]
        rows.append(
            {
                "标的": ticker,
                "策略收益": ret,
                "买入持有": bench,
                "超额收益": ret - bench,
                "夏普比率": res.stats["夏普比率"],
                "最大回撤": res.stats["最大回撤"],
                "是否盈利": ret > 0,
                "是否跑赢": ret > bench,
            }
        )

    table = pd.DataFrame(rows).sort_values("策略收益", ascending=False).reset_index(drop=True)
    n = len(table)
    summary = {
        "标的数": n,
        "盈利概率": float(table["是否盈利"].mean()) if n else 0.0,
        "跑赢基准概率": float(table["是否跑赢"].mean()) if n else 0.0,
        "平均策略收益": float(table["策略收益"].mean()) if n else 0.0,
        "平均超额收益": float(table["超额收益"].mean()) if n else 0.0,
        "中位策略收益": float(table["策略收益"].median()) if n else 0.0,
    }
    return summary, table
