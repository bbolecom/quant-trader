"""参数寻优与多策略对比模块。

- grid_search: 对单个策略的参数做网格搜索，按指定指标排序。
- compare_strategies: 用默认参数横向对比所有（或指定）策略。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from . import backtest
from .strategies import Strategy, get_strategy, list_strategies

# 可作为寻优/排序目标的指标。
SORT_METRICS = [
    "夏普比率",
    "累计收益率",
    "年化收益率",
    "卡尔玛比率",
    "索提诺比率",
    "最大回撤",
]


@dataclass
class OptimizeResult:
    table: pd.DataFrame          # 每组参数及其绩效
    best_params: dict            # 最优参数
    best_stats: dict             # 最优参数对应的绩效
    sort_by: str                 # 排序指标


def _bt_kwargs(cost: dict | None) -> dict:
    cost = cost or {}
    return {
        "initial_capital": cost.get("capital", 100_000.0),
        "fee_bps": cost.get("fee_bps", 5.0),
        "slippage_bps": cost.get("slippage_bps", 2.0),
    }


def grid_search(
    df: pd.DataFrame,
    strategy_name: str,
    param_grid: dict[str, list],
    sort_by: str = "夏普比率",
    allow_short: bool = False,
    cost: dict | None = None,
    max_combos: int = 2000,
) -> OptimizeResult:
    """对策略参数做网格搜索。

    param_grid: {参数名: [候选值, ...]}
    sort_by: 排序目标指标，最大回撤越大（越接近 0）越好，其余越大越好。
    """
    strat: Strategy = get_strategy(strategy_name)
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]

    combos = list(itertools.product(*value_lists)) if keys else [()]
    if len(combos) > max_combos:
        raise ValueError(
            f"参数组合数 {len(combos)} 超过上限 {max_combos}，请缩小搜索范围或步长。"
        )

    bt_kwargs = _bt_kwargs(cost)
    rows = []
    for combo in combos:
        params = dict(zip(keys, combo))
        # 跳过不合理组合：如双均线中短期 >= 长期。
        if "fast" in params and "slow" in params and params["fast"] >= params["slow"]:
            continue
        try:
            pos = strat.generate(df, allow_short=allow_short, **params)
            res = backtest.run_backtest(df, pos, **bt_kwargs)
        except Exception:  # noqa: BLE001 单组失败不影响整体
            continue
        row = dict(params)
        row.update(
            {
                "累计收益率": res.stats["累计收益率"],
                "年化收益率": res.stats["年化收益率"],
                "夏普比率": res.stats["夏普比率"],
                "索提诺比率": res.stats["索提诺比率"],
                "卡尔玛比率": res.stats["卡尔玛比率"],
                "最大回撤": res.stats["最大回撤"],
                "交易次数": int(res.stats["交易次数"]),
            }
        )
        rows.append(row)

    if not rows:
        raise ValueError("没有产生任何有效的参数组合结果。")

    table = pd.DataFrame(rows)
    ascending = sort_by == "最大回撤"  # 回撤是负值，越大越好 -> 降序；这里用 ascending=False
    table = table.sort_values(sort_by, ascending=False).reset_index(drop=True)

    best = table.iloc[0].to_dict()
    best_params = {k: _cast(best[k]) for k in keys}
    best_stats = {k: best[k] for k in table.columns if k not in keys}

    return OptimizeResult(
        table=table, best_params=best_params, best_stats=best_stats, sort_by=sort_by
    )


def _cast(v):
    """把 numpy 标量转回 python 原生类型，整数保持整数。"""
    f = float(v)
    return int(f) if f.is_integer() else f


def compare_strategies(
    df: pd.DataFrame,
    names: list[str] | None = None,
    allow_short: bool = False,
    cost: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """用默认参数横向对比多个策略。

    返回 (绩效汇总表, {策略名: 净值曲线})。
    """
    names = names or list_strategies()
    bt_kwargs = _bt_kwargs(cost)

    rows = []
    curves: dict[str, pd.Series] = {}
    for name in names:
        strat = get_strategy(name)
        params = {p.key: p.default for p in strat.params}
        pos = strat.generate(df, allow_short=allow_short, **params)
        res = backtest.run_backtest(df, pos, **bt_kwargs)
        curves[name] = res.equity
        rows.append(
            {
                "策略": name,
                "累计收益率": res.stats["累计收益率"],
                "年化收益率": res.stats["年化收益率"],
                "年化波动率": res.stats["年化波动率"],
                "夏普比率": res.stats["夏普比率"],
                "卡尔玛比率": res.stats["卡尔玛比率"],
                "最大回撤": res.stats["最大回撤"],
                "交易次数": int(res.stats["交易次数"]),
                "胜率": res.stats["胜率"],
            }
        )

    table = pd.DataFrame(rows).sort_values("夏普比率", ascending=False).reset_index(drop=True)
    return table, curves
