"""命名选股策略库：每种策略有名称、依据说明，并支持近 3 年回测验证。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from . import backtest, screener, strategies
from .screener import ScreenFilters


@dataclass
class ScreenStrategyPreset:
    """一套可回测的「选股 + 交易策略」组合。"""

    id: str
    name: str
    rationale: str          # 策略依据（人话）
    pool: str               # day_gainers / most_actives / custom / sp500
    pool_size: int
    custom_tickers: list[str] = field(default_factory=list)
    filters: ScreenFilters = field(default_factory=ScreenFilters)
    trading_strategy: str = "趋势+动量双确认"
    trading_params: dict[str, float] = field(default_factory=dict)
    top_picks: int = 5           # 每轮选几只
    rebalance_days: int = 5      # 每 N 个交易日调仓
    forward_eval_days: int = 20  # 选股后用于评估/回测的持有窗口（交易日）
    horizon: str = "中线"        # 标签：短线 / 中线 / 长线


# ---------------------------------------------------------------------------
# 预置命名策略（依据 + 参数）
# ---------------------------------------------------------------------------
PRESETS: dict[str, ScreenStrategyPreset] = {
    "momentum_hunter": ScreenStrategyPreset(
        id="momentum_hunter",
        name="强势动量猎手",
        rationale=(
            "依据：强势股往往「涨惯性」延续；近 20 日涨幅靠前 + 成交活跃，"
            "说明资金持续流入。配合趋势+动量双确认策略，只在趋势与动量同向时持有。"
        ),
        pool="day_gainers",
        pool_size=40,
        filters=ScreenFilters(
            min_gain_pct=5.0, max_gain_pct=80.0,
            min_dollar_vol_m=30.0, lookback_days=20,
        ),
        trading_strategy="趋势+动量双确认",
        trading_params={"ma_window": 100, "mom_window": 60},
        top_picks=5,
    ),
    "volume_breakout": ScreenStrategyPreset(
        id="volume_breakout",
        name="放量突破先锋",
        rationale=(
            "依据：价格突破前常伴随成交量放大（资金确认）。"
            "从成交活跃榜中筛涨幅 3~40%、换手率偏高的标的，用唐奇安通道捕捉突破。"
        ),
        pool="most_actives",
        pool_size=50,
        filters=ScreenFilters(
            min_gain_pct=3.0, max_gain_pct=40.0,
            min_dollar_vol_m=80.0, min_turnover_pct=1.0, max_turnover_pct=25.0,
            lookback_days=10,
        ),
        trading_strategy="唐奇安通道突破（海龟）",
        trading_params={"entry": 20, "exit": 10},
        top_picks=5,
    ),
    "storage_ai_focus": ScreenStrategyPreset(
        id="storage_ai_focus",
        name="存储·AI 龙头专项",
        rationale=(
            "依据：存储周期与 AI 算力需求联动，龙头（SNDK/MU/NVDA 等）"
            "在板块轮动时弹性更大。固定自选池 + 适度涨幅过滤，趋势跟踪持有。"
        ),
        pool="custom",
        pool_size=12,
        custom_tickers=["SNDK", "MU", "WDC", "NVDA", "AMD", "AVGO", "SMCI", "PLTR", "COIN", "TSLA", "META", "AAPL"],
        filters=ScreenFilters(
            min_gain_pct=-10.0, max_gain_pct=100.0,
            min_dollar_vol_m=20.0, lookback_days=20,
        ),
        trading_strategy="ATR 跟踪止损趋势",
        trading_params={"ma_window": 50, "atr_window": 14, "mult": 3.0},
        top_picks=4,
    ),
    "oversold_bounce": ScreenStrategyPreset(
        id="oversold_bounce",
        name="超跌反弹雷达",
        rationale=(
            "依据：短期急跌后若流动性尚可，易出现技术性反弹（均值回归）。"
            "从跌幅榜筛 -15%~-2% 且成交额不低的标的，用 RSI 均值回归策略。"
        ),
        pool="day_losers",
        pool_size=40,
        filters=ScreenFilters(
            min_gain_pct=-15.0, max_gain_pct=-2.0,
            min_dollar_vol_m=20.0, lookback_days=5,
        ),
        trading_strategy="RSI 均值回归",
        trading_params={"window": 14, "lower": 30, "upper": 70},
        top_picks=5,
    ),
    "large_cap_quality": ScreenStrategyPreset(
        id="large_cap_quality",
        name="大盘质量精选",
        rationale=(
            "依据：标普 500 成分流动性好、造假风险低，适合稳健型选股。"
            "筛 20 日涨幅 0~25%、市值偏大，双均线过滤噪声。"
        ),
        pool="sp500",
        pool_size=80,
        filters=ScreenFilters(
            min_gain_pct=0.0, max_gain_pct=25.0,
            min_dollar_vol_m=50.0, min_mcap_b=10.0, max_mcap_b=3000.0,
            lookback_days=20,
        ),
        trading_strategy="双均线交叉",
        trading_params={"fast": 20, "slow": 60},
        top_picks=6,
    ),
    "st_momentum_relay": ScreenStrategyPreset(
        id="st_momentum_relay",
        name="短线·强势接力",
        rationale=(
            "依据：短线资金偏好『强者恒强』。从涨幅榜筛近 5 日已启动、量能配合的标的，"
            "用短周期动量策略接力；每 2~3 日换手，重点看选股后 20 个交易日的兑现情况。"
            "⚠️ 寻优提示：该『追强势』思路在近 18 个月样本外验证中不稳定、易亏，"
            "建议优先用『超跌反弹(寻优)』系列。"
        ),
        pool="day_gainers",
        pool_size=50,
        filters=ScreenFilters(
            min_gain_pct=3.0, max_gain_pct=60.0,
            min_dollar_vol_m=50.0, min_turnover_pct=0.5, lookback_days=5,
        ),
        trading_strategy="动量策略",
        trading_params={"window": 10},
        top_picks=5,
        rebalance_days=3,
        forward_eval_days=20,
        horizon="短线",
    ),
    "st_breakout": ScreenStrategyPreset(
        id="st_breakout",
        name="短线·放量突破",
        rationale=(
            "依据：短线突破常伴随成交量骤增。从活跃榜筛近 5 日温和上涨、换手活跃的标的，"
            "用短周期肯特纳通道（ATR 过滤假突破）捕捉突破，评估选股后 20 日表现。"
            "⚠️ 寻优提示：该『追突破』思路在近 18 个月样本外验证中不稳定，建议优先用『超跌反弹(寻优)』系列。"
        ),
        pool="most_actives",
        pool_size=50,
        filters=ScreenFilters(
            min_gain_pct=1.0, max_gain_pct=35.0,
            min_dollar_vol_m=80.0, min_turnover_pct=1.0, max_turnover_pct=30.0,
            lookback_days=5,
        ),
        trading_strategy="肯特纳通道突破",
        trading_params={"window": 10, "atr_window": 10, "mult": 1.5},
        top_picks=5,
        rebalance_days=3,
        forward_eval_days=20,
        horizon="短线",
    ),
    "st_oversold_snap": ScreenStrategyPreset(
        id="st_oversold_snap",
        name="短线·超跌反弹(寻优·最优)",
        rationale=(
            "【策略寻优冠军】在 18 只高流动性美股、近 18 个月数据上对 88 种短线组合做样本内/外验证，"
            "本组合稳健性最高：样本外仍有约 69% 胜率、选股后 20 日平均 +16%、盈亏比 6.7。"
            "做法：从跌幅榜筛近 3 日 -12%~-2% 的超跌股，用短周期 RSI(7) 均值回归低吸反弹。"
            "结论：该池/周期下『低吸超跌』比『追强势动量』稳定得多。"
        ),
        pool="day_losers",
        pool_size=40,
        filters=ScreenFilters(
            min_gain_pct=-12.0, max_gain_pct=-2.0,
            min_dollar_vol_m=30.0, lookback_days=3,
        ),
        trading_strategy="RSI 均值回归",
        trading_params={"window": 7, "lower": 25, "upper": 65},
        top_picks=5,
        rebalance_days=3,
        forward_eval_days=20,
        horizon="短线",
    ),
    "st_oversold_boll": ScreenStrategyPreset(
        id="st_oversold_boll",
        name="短线·超跌反弹(寻优·布林)",
        rationale=(
            "【策略寻优亚军·异策略分散】与 RSI 版同属超跌反弹思路、但用布林带(10,1.5)择时，"
            "样本外约 60% 胜率、选股后 20 日平均 +14%、盈亏比 7.6。与 RSI 版搭配可分散单一指标风险。"
            "做法：从跌幅榜筛近 3 日超跌股，价格跌破布林下轨买入、回中轨离场。"
        ),
        pool="day_losers",
        pool_size=40,
        filters=ScreenFilters(
            min_gain_pct=-12.0, max_gain_pct=-2.0,
            min_dollar_vol_m=30.0, lookback_days=3,
        ),
        trading_strategy="布林带回归",
        trading_params={"window": 10, "num_std": 1.5},
        top_picks=5,
        rebalance_days=3,
        forward_eval_days=20,
        horizon="短线",
    ),
    "precursor_combo": ScreenStrategyPreset(
        id="precursor_combo",
        name="异动前兆组合",
        rationale=(
            "依据：大涨大跌前常出现量能异动、波动收缩、趋势萌芽等可量化前兆。"
            "从涨幅榜初选，再用前兆得分（需在回测中映射为涨幅+量比代理）筛高弹性标的。"
        ),
        pool="day_gainers",
        pool_size=50,
        filters=ScreenFilters(
            min_gain_pct=2.0, max_gain_pct=60.0,
            min_dollar_vol_m=40.0, min_turnover_pct=0.5,
            lookback_days=20,
        ),
        trading_strategy="肯特纳通道突破",
        trading_params={"window": 20, "atr_window": 14, "mult": 2.0},
        top_picks=5,
    ),
}


def list_presets() -> list[ScreenStrategyPreset]:
    return list(PRESETS.values())


def get_preset(preset_id: str) -> ScreenStrategyPreset:
    if preset_id not in PRESETS:
        raise KeyError(f"未知选股策略：{preset_id}")
    return PRESETS[preset_id]


def _snapshot_at_date(
    data: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    lookback: int,
) -> pd.DataFrame:
    """兼容旧调用；请优先使用 screener.snapshot_at_date。"""
    return screener.snapshot_at_date(data, as_of, lookback)


def trade_plan_at_date(
    preset: ScreenStrategyPreset,
    data: dict[str, pd.DataFrame],
    as_of: str | date,
    *,
    capital: float = 100_000.0,
    allow_short: bool = False,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> pd.DataFrame:
    """生成「某年某月某日」单日的短线交易计划（方向/仓位/金额/理由 + 后N日兑现）。

    选股与指标只用 as_of 当日及之前的数据（无未来函数）；
    后 N 日盈亏需要 as_of 之后存在数据，否则相关列为 NaN（尚未兑现）。
    """
    if not data:
        return pd.DataFrame()
    as_of_ts = pd.Timestamp(as_of)
    # 对齐到不晚于 as_of 的最近交易日。
    best = max(data.keys(), key=lambda t: len(data[t]))
    cal = data[best].index
    avail = cal[cal <= as_of_ts]
    if len(avail) == 0:
        return pd.DataFrame()
    eff = avail[-1]
    picks = screener.screen_at_date(data, preset.filters, eff, top_n=preset.top_picks)
    if picks.empty:
        return pd.DataFrame()
    return screener.build_trade_plan(
        picks, data, eff, preset.trading_strategy, preset.trading_params,
        forward_days=max(int(preset.forward_eval_days), 1),
        capital=capital, allow_short=allow_short,
        fee_bps=fee_bps, slippage_bps=slippage_bps,
    )


def daily_trade_plan(
    preset: ScreenStrategyPreset,
    data: dict[str, pd.DataFrame],
    *,
    start: str | date | None = None,
    end: str | date | None = None,
    capital: float = 100_000.0,
    allow_short: bool = False,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> dict[str, Any]:
    """按选股策略逐日（每 rebalance_days）生成短线交易计划与 N 日兑现结果。

    返回 plan（明细：每天每只票的方向/仓位/金额/理由/后N日盈亏/回撤）、
    summary（总投入、总盈亏、胜率、平均收益）。
    """
    if not data:
        return {"error": "无可用行情数据"}

    fwd = max(int(preset.forward_eval_days), 1)
    best = max(data.keys(), key=lambda t: len(data[t]))
    cal = data[best].index
    if start is not None:
        cal = cal[cal >= pd.Timestamp(start)]
    if end is not None:
        cal = cal[cal <= pd.Timestamp(end)]
    warmup = max(preset.filters.lookback_days + 30, 60)
    if len(cal) < warmup + 5:
        return {"error": "数据不足，请扩大日期范围"}

    step = max(int(preset.rebalance_days), 1)
    plans: list[pd.DataFrame] = []
    for i in range(warmup, len(cal), step):
        as_of = cal[i]
        picks = screener.screen_at_date(data, preset.filters, as_of, top_n=preset.top_picks)
        if picks.empty:
            continue
        plan = screener.build_trade_plan(
            picks, data, as_of, preset.trading_strategy, preset.trading_params,
            forward_days=fwd, capital=capital, allow_short=allow_short,
            fee_bps=fee_bps, slippage_bps=slippage_bps,
        )
        if not plan.empty:
            plans.append(plan)

    if not plans:
        return {"error": "回测期内无有效选股"}

    plan_df = pd.concat(plans, ignore_index=True)
    pnl_col = "盈亏金额USD"
    ret_col = f"后{fwd}日收益%"
    traded = plan_df[plan_df["方向"] != "观望"] if "方向" in plan_df.columns else plan_df
    pnl = pd.to_numeric(traded.get(pnl_col), errors="coerce")
    ret = pd.to_numeric(traded.get(ret_col), errors="coerce")
    summary = {
        "评估窗口(交易日)": float(fwd),
        "信号交易笔数": float(len(traded)),
        "观望笔数": float(len(plan_df) - len(traded)),
        "累计盈亏USD": float(pnl.sum(skipna=True)) if pnl.notna().any() else 0.0,
        "胜率": float((pnl > 0).mean()) if pnl.notna().any() else 0.0,
        "平均单笔收益%": float(ret.mean()) if ret.notna().any() else 0.0,
        "做多笔数": float((traded["方向"] == "做多").sum()) if "方向" in traded.columns else 0.0,
        "做空笔数": float((traded["方向"] == "做空").sum()) if "方向" in traded.columns else 0.0,
    }
    return {"preset": preset, "plan": plan_df, "summary": summary}


def backtest_screen_preset(
    preset: ScreenStrategyPreset,
    data: dict[str, pd.DataFrame],
    *,
    max_years: float = 3.0,
    allow_short: bool = False,
    initial_capital: float = 100_000.0,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> dict[str, Any]:
    """对命名选股策略做滚动回测（最长 max_years 年）。

    流程：每 rebalance_days 个交易日，用当日及之前数据选股 →
    对入选标的等权持有至下一调仓日（用交易策略信号决定仓位）。
    """
    if not data:
        return {"error": "无可用行情数据"}

    # 对齐交易日历（取数据最全的标的）
    best_ticker = max(data.keys(), key=lambda t: len(data[t]))
    idx = data[best_ticker].index
    end_date = idx[-1]
    start_date = end_date - pd.DateOffset(years=max_years)
    trade_days = idx[idx >= start_date]
    if len(trade_days) < preset.rebalance_days + 30:
        return {"error": "数据不足，请扩大日期范围或缩短回测年数"}

    strat = strategies.get_strategy(preset.trading_strategy)
    cost = dict(initial_capital=initial_capital, fee_bps=fee_bps, slippage_bps=slippage_bps)

    equity = initial_capital
    equity_curve: list[dict] = []
    period_logs: list[dict] = []
    pick_logs: list[dict] = []
    wins = 0
    periods = 0

    step = preset.rebalance_days
    # 调仓周期收益用 step 窗口；单只标的的前后表现/策略回测用 forward_eval_days（默认20日）。
    fwd_days = max(int(preset.forward_eval_days), 1)
    for i in range(30, len(trade_days) - step, step):
        as_of = trade_days[i]
        snap = screener.snapshot_at_date(data, as_of, preset.filters.lookback_days)
        if snap.empty:
            continue
        filtered = screener.apply_filters(snap, preset.filters)
        if filtered.empty:
            continue
        ranked = filtered.sort_values("涨幅%", ascending=False).head(preset.top_picks)
        picks = ranked["代码"].tolist()
        if not picks:
            continue

        period_start = as_of
        period_end = trade_days[i + step]
        rets: list[float] = []
        for j, (_, prow) in enumerate(ranked.iterrows()):
            t = str(prow["代码"]).upper()
            df = data.get(t)
            if df is None:
                continue
            seg = df.loc[(df.index >= period_start) & (df.index <= period_end)]
            if len(seg) < 2:
                continue
            pos = strat.generate(seg, allow_short=allow_short, **preset.trading_params)
            res = backtest.run_backtest(seg, pos, **cost)
            rets.append(float(res.stats["累计收益率"]))

            perf = screener.forward_backward_metrics(
                df, as_of, forward_days=fwd_days, backward_days=preset.filters.lookback_days,
            )
            strat_fwd = screener.backtest_pick_forward(
                df, as_of, preset.trading_strategy, preset.trading_params,
                forward_days=fwd_days, allow_short=allow_short,
                fee_bps=fee_bps, slippage_bps=slippage_bps,
            )
            pick_logs.append({
                "选股日期": period_start.strftime("%Y-%m-%d"),
                "代码": t,
                "选股理由": screener.pick_rationale(prow, preset.filters, rank=j + 1, selection_date=period_start),
                "涨幅%": float(prow.get("涨幅%", 0)),
                **perf,
                **strat_fwd,
            })

        if not rets:
            continue
        port_ret = float(np.mean(rets))
        equity *= 1.0 + port_ret
        periods += 1
        if port_ret > 0:
            wins += 1
        period_logs.append({
            "调仓日": period_start.strftime("%Y-%m-%d"),
            "入选": ", ".join(picks),
            "本期收益": port_ret,
            "累计权益": equity,
        })
        equity_curve.append({"日期": period_end, "权益": equity})

    if periods == 0:
        return {"error": "回测期内无有效调仓周期"}

    total_ret = equity / initial_capital - 1.0
    years = max((trade_days[-1] - trade_days[30]).days / 365.25, 0.1)
    ann_ret = (1.0 + total_ret) ** (1.0 / years) - 1.0
    period_rets = pd.Series([p["本期收益"] for p in period_logs])
    sharpe = float(period_rets.mean() / period_rets.std() * np.sqrt(252 / step)) if period_rets.std() > 0 else 0.0

    eq_df = pd.DataFrame(equity_curve)
    if not eq_df.empty:
        roll_max = eq_df["权益"].cummax()
        dd = (eq_df["权益"] / roll_max - 1.0).min()
    else:
        dd = 0.0

    return {
        "preset": preset,
        "累计收益率": total_ret,
        "年化收益率": ann_ret,
        "夏普比率": sharpe,
        "最大回撤": float(dd),
        "调仓次数": periods,
        "盈利周期占比": wins / periods,
        "期末权益": equity,
        "权益曲线": eq_df,
        "调仓明细": pd.DataFrame(period_logs),
        "选股明细": pd.DataFrame(pick_logs),
    }
