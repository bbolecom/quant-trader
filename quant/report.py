"""一键全流程体检编排引擎。

把整条决策链串成一次调用：
    判市（regime）→ 推荐策略 → 自动参数寻优 → 样本外验证 → 赚钱概率 → 综合评分与结论。

最终产出一份结构化报告（FullReport），供 UI 渲染或脚本调用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import backtest, probability, regime as regime_mod, validation
from .regime import Regime
from .strategies import get_strategy


@dataclass
class FullReport:
    ticker: str
    regime: Regime
    recommend_table: pd.DataFrame
    strategy: str
    best_params: dict
    is_stats: dict | None
    oos_stats: dict | None
    overfit_gap: float
    final_result: backtest.BacktestResult
    prob: dict
    score: float
    grade: str
    verdict: str
    flags: list[str] = field(default_factory=list)


def auto_param_grid(strat, max_combos: int = 240) -> dict[str, list]:
    """围绕参数取值范围自动生成网格（控制总组合数）。"""
    params = strat.params
    if not params:
        return {}
    k = len(params)
    per = max(3, int(round(max_combos ** (1.0 / k))))
    grid: dict[str, list] = {}
    for p in params:
        if p.is_int:
            vals = np.unique(np.linspace(p.min_value, p.max_value, per).round().astype(int))
            grid[p.key] = [int(v) for v in vals]
        else:
            vals = np.unique(np.round(np.linspace(p.min_value, p.max_value, per), 2))
            grid[p.key] = [float(v) for v in vals]
    return grid


def run_full_report(
    df: pd.DataFrame,
    ticker: str = "",
    allow_short: bool = False,
    cost: dict | None = None,
    sort_by: str = "夏普比率",
    train_ratio: float = 0.7,
) -> FullReport:
    """执行一键体检，返回完整报告。"""
    # 1) 判市 + 推荐。
    regime, rec_table = regime_mod.recommend(df, allow_short=allow_short, cost=cost)
    strategy = str(rec_table.iloc[0]["策略"])
    strat = get_strategy(strategy)

    # 2) 自动寻优 + 样本外验证（无参数策略如买入持有则跳过）。
    best_params: dict = {}
    is_stats = oos_stats = None
    overfit_gap = 0.0
    grid = auto_param_grid(strat)
    if grid and len(df) >= 80:
        try:
            hv = validation.holdout_validate(
                df, strategy, grid, sort_by=sort_by, train_ratio=train_ratio,
                allow_short=allow_short, cost=cost,
            )
            best_params = hv.best_params
            is_stats, oos_stats = hv.is_stats, hv.oos_stats
            overfit_gap = hv.is_stats["夏普比率"] - hv.oos_stats["夏普比率"]
        except ValueError:
            best_params = {p.key: p.default for p in strat.params}
    else:
        best_params = {p.key: p.default for p in strat.params}

    # 3) 用最优参数在全量数据上做最终回测。
    bt_kwargs = {
        "initial_capital": (cost or {}).get("capital", 100_000.0),
        "fee_bps": (cost or {}).get("fee_bps", 5.0),
        "slippage_bps": (cost or {}).get("slippage_bps", 2.0),
    }
    pos = strat.generate(df, allow_short=allow_short, **best_params)
    final_result = backtest.run_backtest(df, pos, **bt_kwargs)

    # 4) 赚钱概率。
    prob = probability.analyze_single(df, strategy, params=best_params,
                                      allow_short=allow_short, cost=cost)

    # 5) 综合评分与结论。
    score, flags = _score(regime, rec_table, final_result, oos_stats, overfit_gap, prob)
    grade, verdict = _verdict(score, strategy, regime)

    return FullReport(
        ticker=ticker, regime=regime, recommend_table=rec_table, strategy=strategy,
        best_params=best_params, is_stats=is_stats, oos_stats=oos_stats,
        overfit_gap=overfit_gap, final_result=final_result, prob=prob,
        score=score, grade=grade, verdict=verdict, flags=flags,
    )


def _score(regime, rec_table, final_result, oos_stats, overfit_gap, prob) -> tuple[float, list[str]]:
    """0~100 综合评分 + 风险提示。"""
    s = 50.0
    flags: list[str] = []
    stats = final_result.stats

    # 样本外夏普（最看重）。
    oos_sharpe = oos_stats["夏普比率"] if oos_stats else stats["夏普比率"]
    s += float(np.clip(oos_sharpe, -2, 3)) * 12

    # 跑赢基准。
    if stats["累计收益率"] > stats["基准收益率"]:
        s += 8
    else:
        s -= 6
        flags.append("策略未跑赢买入持有，需谨慎——也许直接持有更省心。")

    # 过拟合惩罚。
    if oos_stats is not None and overfit_gap > 1.0:
        s -= 15
        flags.append(f"样本外夏普比样本内低 {overfit_gap:.2f}，过拟合风险较高。")

    # 回撤。
    if stats["最大回撤"] < -0.35:
        s -= 8
        flags.append(f"历史最大回撤达 {stats['最大回撤']:.1%}，波动剧烈，注意仓位与止损。")

    # 交易样本。
    if prob["num_trades"] < 5:
        flags.append(f"历史仅 {prob['num_trades']} 笔交易，样本过少，统计结论不稳健。")

    # 市场状态契合。
    top_fit = str(rec_table.iloc[0]["契合度"])
    if top_fit == "高度契合":
        s += 8
    if regime.trend_label == "过渡":
        flags.append("当前处于趋势过渡期，方向不明，建议降低仓位或观望。")
    if regime.vol_label == "高波动":
        flags.append("当前波动偏高，建议配合 ATR 止损并减小仓位。")

    return float(np.clip(s, 0, 100)), flags


def _verdict(score: float, strategy: str, regime) -> tuple[str, str]:
    if score >= 72:
        return "优秀", (
            f"在「{regime.summary}」下，**{strategy}** 历史表现稳健且样本外验证良好，"
            f"建议先用「模拟交易」跑一段确认，再考虑小仓位实盘。"
        )
    if score >= 55:
        return "中等", (
            f"**{strategy}** 与当前市场较契合，但优势不够突出。"
            f"建议进一步调参、扩大数据或结合其它确认信号，并务必先模拟盘验证。"
        )
    if score >= 40:
        return "偏弱", (
            f"**{strategy}** 在当前环境下表现一般，证据不足。"
            f"建议观望，或对比「策略对比」中其它策略，不宜贸然实盘。"
        )
    return "不建议", (
        f"当前数据下没有可靠的盈利证据，**不建议**用 {strategy} 实盘。"
        f"可考虑空仓等待更清晰的行情，或仅做买入持有。"
    )
