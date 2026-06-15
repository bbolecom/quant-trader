"""样本外验证模块（防止参数过拟合）。

提供两种方法：

1. holdout_validate —— 单次"训练/测试"划分：
   在前段（训练集）上做网格寻优，再把最优参数原封不动地用到后段（测试集），
   对比样本内（IS）与样本外（OOS）的表现差异。

2. walk_forward —— 滚动前向验证：
   把时间轴切成若干段，每次用"过去一段"寻优、在"紧接着的下一段"上交易，
   把各段样本外收益拼接成一条连续的样本外净值曲线，最贴近真实交易场景。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import backtest, metrics as M, optimize
from .strategies import get_strategy


@dataclass
class HoldoutResult:
    best_params: dict
    is_stats: dict          # 训练集（样本内）绩效
    oos_stats: dict         # 测试集（样本外）绩效
    is_equity: pd.Series
    oos_equity: pd.Series
    split_date: pd.Timestamp


@dataclass
class WalkForwardResult:
    oos_equity: pd.Series           # 拼接后的连续样本外净值
    oos_benchmark: pd.Series        # 对应区间买入持有净值
    oos_stats: dict                 # 样本外整体绩效
    windows: pd.DataFrame           # 每个窗口的参数与样本外表现
    drawdown: pd.Series


def _bt_kwargs(cost: dict | None) -> dict:
    cost = cost or {}
    return {
        "initial_capital": cost.get("capital", 100_000.0),
        "fee_bps": cost.get("fee_bps", 5.0),
        "slippage_bps": cost.get("slippage_bps", 2.0),
    }


def holdout_validate(
    df: pd.DataFrame,
    strategy_name: str,
    param_grid: dict[str, list],
    sort_by: str = "夏普比率",
    train_ratio: float = 0.7,
    allow_short: bool = False,
    cost: dict | None = None,
) -> HoldoutResult:
    """单次训练/测试划分验证。"""
    n = len(df)
    if n < 60:
        raise ValueError("数据量过少，无法进行样本外验证。")

    split = int(n * train_ratio)
    train = df.iloc[:split]
    test = df.iloc[split:]
    if len(test) < 20:
        raise ValueError("测试集样本过少，请减小训练集比例或扩大日期范围。")

    opt = optimize.grid_search(
        train, strategy_name, param_grid, sort_by=sort_by,
        allow_short=allow_short, cost=cost,
    )
    best = opt.best_params

    strat = get_strategy(strategy_name)
    bt_kwargs = _bt_kwargs(cost)

    is_pos = strat.generate(train, allow_short=allow_short, **best)
    is_res = backtest.run_backtest(train, is_pos, **bt_kwargs)

    oos_pos = strat.generate(test, allow_short=allow_short, **best)
    oos_res = backtest.run_backtest(test, oos_pos, **bt_kwargs)

    return HoldoutResult(
        best_params=best,
        is_stats=is_res.stats,
        oos_stats=oos_res.stats,
        is_equity=is_res.equity,
        oos_equity=oos_res.equity,
        split_date=pd.Timestamp(test.index[0]),
    )


def walk_forward(
    df: pd.DataFrame,
    strategy_name: str,
    param_grid: dict[str, list],
    sort_by: str = "夏普比率",
    n_splits: int = 4,
    train_ratio: float = 0.6,
    allow_short: bool = False,
    cost: dict | None = None,
) -> WalkForwardResult:
    """滚动前向验证。

    把数据切成 n_splits 个测试窗口；每个测试窗口之前的一段数据用于寻优。
    采用"锚定式"训练（训练集起点固定为数据开头，随窗口推进而扩展）。
    """
    n = len(df)
    if n < 120:
        raise ValueError("数据量过少，无法进行滚动验证（建议至少半年以上日线）。")

    strat = get_strategy(strategy_name)
    bt_kwargs = _bt_kwargs(cost)

    # 测试段从 train_ratio 之后开始，均分为 n_splits 段。
    test_start = int(n * train_ratio)
    boundaries = np.linspace(test_start, n, n_splits + 1).astype(int)

    oos_returns_parts = []
    asset_returns_parts = []
    rows = []

    for i in range(n_splits):
        te_lo, te_hi = boundaries[i], boundaries[i + 1]
        if te_hi - te_lo < 10:
            continue
        train = df.iloc[:te_lo]          # 锚定式：从头到当前测试段之前
        test = df.iloc[te_lo:te_hi]

        try:
            opt = optimize.grid_search(
                train, strategy_name, param_grid, sort_by=sort_by,
                allow_short=allow_short, cost=cost,
            )
            best = opt.best_params
        except ValueError:
            continue

        oos_pos = strat.generate(test, allow_short=allow_short, **best)
        oos_res = backtest.run_backtest(test, oos_pos, **bt_kwargs)

        oos_returns_parts.append(oos_res.returns)
        asset_returns_parts.append(test["Close"].pct_change().fillna(0.0))

        rows.append(
            {
                "窗口": f"#{i + 1}",
                "测试起": pd.Timestamp(test.index[0]).strftime("%Y-%m-%d"),
                "测试止": pd.Timestamp(test.index[-1]).strftime("%Y-%m-%d"),
                "最优参数": str(best),
                "样本外收益": oos_res.stats["累计收益率"],
                "夏普比率": oos_res.stats["夏普比率"],
                "最大回撤": oos_res.stats["最大回撤"],
            }
        )

    if not oos_returns_parts:
        raise ValueError("未能生成任何有效的样本外窗口。")

    oos_ret = pd.concat(oos_returns_parts)
    bench_ret = pd.concat(asset_returns_parts)
    oos_equity = (1.0 + oos_ret).cumprod()
    oos_benchmark = (1.0 + bench_ret).cumprod()

    running_max = oos_equity.cummax()
    drawdown = oos_equity / running_max - 1.0

    oos_stats = M.summary(equity=oos_equity, returns=oos_ret, trade_returns=None, num_trades=0)
    oos_stats.pop("胜率", None)
    oos_stats.pop("交易次数", None)
    oos_stats["基准收益率"] = M.total_return(oos_benchmark)

    return WalkForwardResult(
        oos_equity=oos_equity,
        oos_benchmark=oos_benchmark,
        oos_stats=oos_stats,
        windows=pd.DataFrame(rows),
        drawdown=drawdown,
    )
