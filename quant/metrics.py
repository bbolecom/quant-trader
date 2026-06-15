"""绩效指标计算模块。"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(equity: pd.Series) -> float:
    """累计收益率。"""
    if equity.empty:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series) -> float:
    """年化复合增长率。"""
    if len(equity) < 2:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)


def annual_volatility(returns: pd.Series) -> float:
    """年化波动率。"""
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """年化夏普比率。"""
    if returns.empty or returns.std(ddof=0) == 0:
        return 0.0
    excess = returns - risk_free / TRADING_DAYS
    return float(excess.mean() / returns.std(ddof=0) * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """年化索提诺比率（仅惩罚下行波动）。"""
    if returns.empty:
        return 0.0
    excess = returns - risk_free / TRADING_DAYS
    downside = excess[excess < 0]
    dd = downside.std(ddof=0)
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤（负值）。"""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(equity: pd.Series) -> float:
    """卡尔玛比率 = 年化收益 / |最大回撤|。"""
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(cagr(equity) / abs(mdd))


def win_rate(trade_returns: pd.Series) -> float:
    """单笔交易胜率。"""
    if trade_returns is None or len(trade_returns) == 0:
        return 0.0
    return float((trade_returns > 0).mean())


def summary(
    equity: pd.Series,
    returns: pd.Series,
    trade_returns: pd.Series | None = None,
    num_trades: int = 0,
) -> dict[str, float]:
    """汇总常用绩效指标。"""
    return {
        "累计收益率": total_return(equity),
        "年化收益率": cagr(equity),
        "年化波动率": annual_volatility(returns),
        "夏普比率": sharpe_ratio(returns),
        "索提诺比率": sortino_ratio(returns),
        "最大回撤": max_drawdown(equity),
        "卡尔玛比率": calmar_ratio(equity),
        "交易次数": float(num_trades),
        "胜率": win_rate(trade_returns) if trade_returns is not None else 0.0,
    }
