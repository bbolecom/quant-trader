"""向量化回测引擎。

核心思路：
1. 策略产生"目标仓位" position（当日收盘后的目标）。
2. 为避免使用未来信息，实际持仓 = position.shift(1)，即次日才按收盘价建仓。
3. 策略收益 = 实际持仓 * 标的日收益。
4. 换仓时按换手量扣除交易成本（双边手续费 + 滑点）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics as M


@dataclass
class BacktestResult:
    equity: pd.Series          # 策略净值曲线（起始 = 1）
    benchmark: pd.Series       # 买入持有净值曲线
    returns: pd.Series         # 策略日收益
    position: pd.Series        # 实际持仓（已顺延一日）
    drawdown: pd.Series        # 回撤曲线
    trades: pd.DataFrame       # 交易明细
    stats: dict[str, float]    # 绩效指标
    df: pd.DataFrame           # 原始行情


def run_backtest(
    df: pd.DataFrame,
    position: pd.Series,
    initial_capital: float = 100_000.0,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> BacktestResult:
    """执行回测。

    参数
    ----
    df: 含 Close 列的行情数据。
    position: 策略目标仓位序列（-1 ~ 1）。
    initial_capital: 初始资金，仅用于展示金额。
    fee_bps: 单边手续费（基点，1 bp = 0.01%）。
    slippage_bps: 单边滑点（基点）。
    """
    close = df["Close"].astype(float)
    asset_ret = close.pct_change().fillna(0.0)

    # 顺延一日，避免使用当日收盘后才知道的信号当日成交。
    actual_pos = position.shift(1).fillna(0.0)

    # 换手成本：仓位变化幅度 * 单边成本。
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    turnover = actual_pos.diff().abs().fillna(actual_pos.abs())
    cost = turnover * cost_rate

    strat_ret = actual_pos * asset_ret - cost

    equity = (1.0 + strat_ret).cumprod()
    benchmark = (1.0 + asset_ret).cumprod()

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0

    trades = _extract_trades(close, actual_pos)
    trade_returns = trades["收益率"] if not trades.empty else pd.Series(dtype=float)

    stats = M.summary(
        equity=equity,
        returns=strat_ret,
        trade_returns=trade_returns,
        num_trades=len(trades),
    )
    stats["初始资金"] = initial_capital
    stats["期末资金"] = float(initial_capital * equity.iloc[-1]) if not equity.empty else initial_capital
    stats["基准收益率"] = M.total_return(benchmark)

    return BacktestResult(
        equity=equity,
        benchmark=benchmark,
        returns=strat_ret,
        position=actual_pos,
        drawdown=drawdown,
        trades=trades,
        stats=stats,
        df=df,
    )


def _extract_trades(close: pd.Series, position: pd.Series) -> pd.DataFrame:
    """从持仓序列中还原出每一笔完整交易（开仓 -> 平仓）。"""
    records = []
    entry_date = None
    entry_price = None
    entry_side = 0.0

    prev = 0.0
    for dt, pos in position.items():
        if pos != prev:
            # 先平掉旧仓位。
            if prev != 0.0 and entry_date is not None:
                exit_price = close.loc[dt]
                ret = (exit_price / entry_price - 1.0) * np.sign(entry_side)
                records.append(
                    {
                        "开仓日期": entry_date,
                        "平仓日期": dt,
                        "方向": "做多" if entry_side > 0 else "做空",
                        "开仓价": round(float(entry_price), 4),
                        "平仓价": round(float(exit_price), 4),
                        "收益率": float(ret),
                    }
                )
                entry_date = None
            # 再开新仓位。
            if pos != 0.0:
                entry_date = dt
                entry_price = close.loc[dt]
                entry_side = pos
            prev = pos

    # 回测结束时仍持仓 -> 以最后价格平仓。
    if prev != 0.0 and entry_date is not None:
        last_dt = close.index[-1]
        exit_price = close.iloc[-1]
        ret = (exit_price / entry_price - 1.0) * np.sign(entry_side)
        records.append(
            {
                "开仓日期": entry_date,
                "平仓日期": last_dt,
                "方向": "做多" if entry_side > 0 else "做空",
                "开仓价": round(float(entry_price), 4),
                "平仓价": round(float(exit_price), 4),
                "收益率": float(ret),
            }
        )

    return pd.DataFrame(records)
