"""多标的组合回测模块。

对组合中的每个标的应用同一策略，按目标权重每日再平衡，
合成组合层面的净值曲线与绩效指标。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import backtest, metrics as M
from .strategies import get_strategy


TRADING_DAYS = 252


@dataclass
class PortfolioResult:
    equity: pd.Series                  # 组合净值曲线（起始 = 1）
    returns: pd.Series                 # 组合日收益
    drawdown: pd.Series                # 组合回撤曲线
    weights: dict[str, float]          # 归一化后的目标权重
    asset_equity: dict[str, pd.Series] # 各标的策略净值曲线
    asset_stats: pd.DataFrame          # 各标的绩效
    stats: dict[str, float]            # 组合绩效
    leverage: pd.Series | None = None  # 目标波动率下的动态仓位系数（如启用）


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, w) for w in weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: max(0.0, w) / total for k, w in weights.items()}


def apply_weight_cap(weights: dict[str, float], cap: float) -> dict[str, float]:
    """对单个标的权重设上限并重新归一化（迭代直至全部不超过上限）。"""
    w = normalize_weights(weights)
    n = len(w)
    if cap <= 0 or cap >= 1.0 or cap * n < 1.0 - 1e-9:
        # cap 太小导致无法分配满仓时，退化为等权。
        if cap * n < 1.0 - 1e-9:
            return {k: 1.0 / n for k in w}
        return w
    for _ in range(100):
        over = {k: v for k, v in w.items() if v > cap + 1e-9}
        if not over:
            break
        capped_total = cap * len(over)
        free_total = sum(v for k, v in w.items() if k not in over)
        remain = 1.0 - capped_total
        new_w = {}
        for k, v in w.items():
            if k in over:
                new_w[k] = cap
            else:
                new_w[k] = (v / free_total * remain) if free_total > 0 else cap
        w = new_w
    return w


def compute_weights(
    data: dict[str, pd.DataFrame],
    mode: str = "等权",
    custom: dict[str, float] | None = None,
    vol_window: int = 60,
    cap: float = 1.0,
) -> dict[str, float]:
    """按不同方案计算目标权重。

    mode 可选："等权" / "自定义" / "逆波动率" / "风险平价"。
    （逆波动率与简化风险平价在此等价：权重与各标的波动率成反比。）
    """
    tickers = list(data.keys())
    if mode == "自定义" and custom:
        weights = {t: float(custom.get(t, 0.0)) for t in tickers}
    elif mode in ("逆波动率", "风险平价"):
        weights = {}
        for t in tickers:
            ret = data[t]["Close"].pct_change().dropna()
            vol = float(ret.tail(max(vol_window, 20)).std(ddof=0))
            weights[t] = (1.0 / vol) if vol > 1e-9 else 0.0
        if sum(weights.values()) <= 0:
            weights = {t: 1.0 for t in tickers}
    else:  # 等权
        weights = {t: 1.0 for t in tickers}

    return apply_weight_cap(weights, cap)


def run_portfolio(
    data: dict[str, pd.DataFrame],
    weights: dict[str, float],
    strategy_name: str,
    params: dict | None = None,
    allow_short: bool = False,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
    initial_capital: float = 100_000.0,
    vol_target: float | None = None,
    max_leverage: float = 1.5,
    vol_window: int = 20,
) -> PortfolioResult:
    """执行组合回测。

    data: {代码: 行情DataFrame}
    weights: {代码: 权重}（会自动归一化）
    vol_target: 目标年化波动率（如 0.15 表示 15%）；为 None 时不做仓位调节。
    max_leverage: 目标波动率模式下允许的最大仓位系数。
    vol_window: 估计已实现波动率的滚动窗口（日）。
    其余参数同单标的回测。
    """
    if not data:
        raise ValueError("组合中没有任何标的。")

    params = params or {}
    strat = get_strategy(strategy_name)
    weights = normalize_weights({k: weights.get(k, 0.0) for k in data})

    asset_ret: dict[str, pd.Series] = {}
    asset_equity: dict[str, pd.Series] = {}
    stat_rows = []

    for ticker, df in data.items():
        pos = strat.generate(df, allow_short=allow_short, **params)
        res = backtest.run_backtest(
            df, pos, initial_capital=initial_capital,
            fee_bps=fee_bps, slippage_bps=slippage_bps,
        )
        asset_ret[ticker] = res.returns
        asset_equity[ticker] = res.equity
        stat_rows.append(
            {
                "标的": ticker,
                "权重": weights[ticker],
                "累计收益率": res.stats["累计收益率"],
                "年化收益率": res.stats["年化收益率"],
                "夏普比率": res.stats["夏普比率"],
                "最大回撤": res.stats["最大回撤"],
            }
        )

    # 对齐日期后按权重每日再平衡合成组合收益。
    ret_df = pd.DataFrame(asset_ret).fillna(0.0)
    w_vec = pd.Series(weights).reindex(ret_df.columns).fillna(0.0)
    base_ret = ret_df.mul(w_vec, axis=1).sum(axis=1)

    # 目标波动率：用滚动已实现波动率反推仓位系数（顺延一日避免使用未来信息）。
    leverage = None
    port_ret = base_ret
    if vol_target and vol_target > 0:
        realized = base_ret.rolling(vol_window, min_periods=max(5, vol_window // 2)).std(ddof=0) * np.sqrt(TRADING_DAYS)
        lev = (vol_target / realized).clip(lower=0.0, upper=max_leverage)
        lev = lev.shift(1).fillna(1.0).clip(lower=0.0, upper=max_leverage)
        leverage = lev
        port_ret = base_ret * lev

    equity = (1.0 + port_ret).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0

    stats = M.summary(equity=equity, returns=port_ret, trade_returns=None, num_trades=0)
    stats.pop("胜率", None)
    stats.pop("交易次数", None)
    stats["初始资金"] = initial_capital
    stats["期末资金"] = float(initial_capital * equity.iloc[-1]) if not equity.empty else initial_capital
    if leverage is not None:
        stats["平均仓位系数"] = float(leverage.mean())

    asset_stats = pd.DataFrame(stat_rows).sort_values("权重", ascending=False).reset_index(drop=True)

    return PortfolioResult(
        equity=equity,
        returns=port_ret,
        drawdown=drawdown,
        weights=weights,
        asset_equity=asset_equity,
        asset_stats=asset_stats,
        stats=stats,
        leverage=leverage,
    )
