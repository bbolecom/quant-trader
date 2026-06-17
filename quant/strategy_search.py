"""短线选股策略搜索器。

在同一份历史数据上枚举「选股过滤 × 交易策略 × 参数 × 调仓周期 × 方向」组合，
按*稳健性*而非单纯收益排序，并切分样本内(train)/样本外(test)以抑制过拟合。

稳健性核心指标（均基于每笔『选股后 N 日方向调整收益』，方向无关）：
    - 胜率：盈利笔数占比
    - 信息比：平均收益 / 收益标准差（越高越稳）
    - 盈亏比：平均盈利 / 平均亏损绝对值
    - 最差单笔：尾部风险参考
评分 = 0.6 × 信息比 + 0.4 × (胜率-0.5)×2，要求样本内笔数达标；
样本外需平均收益>0 且 信息比>0 才记为『稳健通过』。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .screener import (
    ScreenFilters,
    forward_backward_metrics,
    screen_at_date,
    signal_direction_at,
)

# 高流动性、波动充足、适合短线的美股候选池（搜索默认池）
DEFAULT_SHORT_TERM_POOL = [
    "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN", "GOOGL", "NFLX",
    "AVGO", "MU", "SMCI", "PLTR", "COIN", "MARA", "QQQ", "SOXL", "TQQQ",
]


@dataclass
class CandidateCombo:
    """一个待评估的短线选股+交易组合。"""

    id: str
    idea: str                 # 思路标签：强势动量 / 突破 / 超跌反弹
    trading_strategy: str
    params: dict[str, float]
    filters: ScreenFilters
    rebalance_days: int = 3
    top_picks: int = 5
    allow_short: bool = False
    forward_days: int = 20

    @property
    def label(self) -> str:
        ps = ",".join(f"{k}={v}" for k, v in self.params.items())
        short = "可空" if self.allow_short else "仅多"
        return (
            f"{self.idea}｜{self.trading_strategy}({ps})｜"
            f"看{self.filters.lookback_days}日·每{self.rebalance_days}日·{short}"
        )


# ---------------------------------------------------------------------------
# 搜索空间
# ---------------------------------------------------------------------------
def _strong_filter(lookback: int) -> ScreenFilters:
    return ScreenFilters(
        min_gain_pct=2.0, max_gain_pct=80.0,
        min_dollar_vol_m=50.0, min_turnover_pct=0.0, lookback_days=lookback,
    )


def _breakout_filter(lookback: int) -> ScreenFilters:
    return ScreenFilters(
        min_gain_pct=1.0, max_gain_pct=40.0,
        min_dollar_vol_m=80.0, min_turnover_pct=1.0, max_turnover_pct=30.0,
        lookback_days=lookback,
    )


def _oversold_filter(lookback: int, floor_pct: float) -> ScreenFilters:
    return ScreenFilters(
        min_gain_pct=floor_pct, max_gain_pct=-2.0,
        min_dollar_vol_m=30.0, lookback_days=lookback,
    )


def build_search_space(
    *,
    rebalance_options: tuple[int, ...] = (2, 3),
    top_picks: int = 5,
    forward_days: int = 20,
    include_short: bool = True,
) -> list[CandidateCombo]:
    """构造短线策略搜索空间（按思路配对合理的过滤与交易策略）。"""
    combos: list[CandidateCombo] = []

    def add(idea, strat, params, filt, short_opts):
        for rb in rebalance_options:
            for short in short_opts:
                cid = (
                    f"{idea}-{strat}-"
                    f"{'-'.join(f'{k}{v}' for k, v in params.items())}-rb{rb}-"
                    f"{'S' if short else 'L'}"
                )
                combos.append(CandidateCombo(
                    id=cid, idea=idea, trading_strategy=strat, params=dict(params),
                    filters=filt, rebalance_days=rb, top_picks=top_picks,
                    allow_short=short, forward_days=forward_days,
                ))

    short_opts_dir = (False, True) if include_short else (False,)
    long_only = (False,)

    # 强势动量思路
    for lb in (5, 10):
        sf = _strong_filter(lb)
        for w in (5, 10, 20):
            add("强势动量", "动量策略", {"window": w}, sf, short_opts_dir)
        for ma, mom in ((20, 10), (50, 20)):
            add("强势动量", "趋势+动量双确认", {"ma_window": ma, "mom_window": mom}, sf, short_opts_dir)

    # 突破思路
    bf = _breakout_filter(5)
    for w, aw, mult in ((10, 10, 1.5), (20, 10, 2.0)):
        add("突破", "肯特纳通道突破", {"window": w, "atr_window": aw, "mult": mult}, bf, short_opts_dir)
    for entry, exit_ in ((10, 5), (20, 10)):
        add("突破", "唐奇安通道突破（海龟）", {"entry": entry, "exit": exit_}, bf, short_opts_dir)

    # 超跌反弹思路（均值回归一般仅做多）
    for lb, floor in ((3, -12.0), (5, -18.0)):
        of = _oversold_filter(lb, floor)
        for w, lo, up in ((7, 25, 65), (14, 30, 70)):
            add("超跌反弹", "RSI 均值回归", {"window": w, "lower": lo, "upper": up}, of, long_only)
        for w, ns in ((10, 1.5), (20, 2.0)):
            add("超跌反弹", "布林带回归", {"window": w, "num_std": ns}, of, long_only)
        for w, en, ex in ((10, 1.5, 0.5), (20, 2.0, 0.5)):
            add("超跌反弹", "Z-Score 均值回归", {"window": w, "entry": en, "exit": ex}, of, short_opts_dir)

    return combos


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def _collect_returns(
    data: dict[str, pd.DataFrame],
    combo: CandidateCombo,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> tuple[list[float], int, int]:
    """逐日(每 rebalance_days)选股，收集每笔『后 N 日方向调整收益』(小数)。

    返回 (收益列表, 做多笔数, 做空笔数)。
    """
    if not data:
        return [], 0, 0
    fwd = max(int(combo.forward_days), 1)
    best = max(data.keys(), key=lambda t: len(data[t]))
    cal = data[best].index
    cal = cal[(cal >= start_ts) & (cal <= end_ts)]
    warmup = max(combo.filters.lookback_days + 30, 60)
    if len(cal) < warmup + fwd + 3:
        return [], 0, 0

    rets: list[float] = []
    longs = shorts = 0
    step = max(int(combo.rebalance_days), 1)
    for i in range(warmup, len(cal) - fwd, step):
        as_of = cal[i]
        picks = screen_at_date(data, combo.filters, as_of, top_n=combo.top_picks)
        if picks.empty:
            continue
        for code in picks["代码"]:
            df = data.get(str(code).upper())
            if df is None or df.empty:
                continue
            sig = signal_direction_at(
                df, as_of, combo.trading_strategy, combo.params,
                allow_short=combo.allow_short,
            )
            dir_sign = 1.0 if sig > 0 else (-1.0 if sig < 0 else 0.0)
            if dir_sign == 0:
                continue
            fb = forward_backward_metrics(df, as_of, forward_days=fwd, backward_days=1)
            raw = fb.get(f"后{fwd}日收益", np.nan)
            if pd.isna(raw):
                continue
            rets.append(float(dir_sign * raw))
            if dir_sign > 0:
                longs += 1
            else:
                shorts += 1
    return rets, longs, shorts


def _metrics(rets: list[float]) -> dict[str, float]:
    """基于单笔收益序列(小数)计算稳健性指标。"""
    n = len(rets)
    if n == 0:
        return {"笔数": 0}
    arr = np.asarray(rets, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0  # 负值
    info = mean / std if std > 0 else 0.0
    if avg_loss < 0:
        pl = avg_win / abs(avg_loss)
    elif avg_win > 0:
        pl = float("inf")
    else:
        pl = 0.0
    return {
        "笔数": float(n),
        "胜率": float((arr > 0).mean()),
        "平均收益%": mean * 100.0,
        "收益中位数%": float(np.median(arr)) * 100.0,
        "收益波动%": std * 100.0,
        "信息比": info,
        "盈亏比": pl,
        "平均盈利%": avg_win * 100.0,
        "平均亏损%": avg_loss * 100.0,
        "最差单笔%": float(arr.min()) * 100.0,
        "最佳单笔%": float(arr.max()) * 100.0,
        "累计方向收益%": float(arr.sum()) * 100.0,
    }


def _score(train_m: dict[str, float]) -> float:
    """样本内稳健评分：信息比为主，胜率为辅。"""
    if train_m.get("笔数", 0) <= 0:
        return float("-inf")
    info = train_m.get("信息比", 0.0)
    win = train_m.get("胜率", 0.0)
    return 0.6 * info + 0.4 * (win - 0.5) * 2.0


def evaluate_combo(
    data: dict[str, pd.DataFrame],
    combo: CandidateCombo,
    *,
    split_ratio: float = 0.6,
    min_trades: int = 30,
) -> dict[str, Any]:
    """在样本内/外分别评估一个组合，返回指标与评分。"""
    if not data:
        return {}
    best = max(data.keys(), key=lambda t: len(data[t]))
    cal = data[best].index
    if len(cal) < 80:
        return {}
    split_i = int(len(cal) * split_ratio)
    train_start, train_end = cal[0], cal[split_i]
    test_start, test_end = cal[split_i], cal[-1]

    tr_rets, tr_long, tr_short = _collect_returns(data, combo, train_start, train_end)
    te_rets, te_long, te_short = _collect_returns(data, combo, test_start, test_end)
    tr_m = _metrics(tr_rets)
    te_m = _metrics(te_rets)

    train_score = _score(tr_m) if tr_m.get("笔数", 0) >= min_trades else float("-inf")
    robust = bool(
        te_m.get("笔数", 0) >= max(min_trades * 0.4, 10)
        and te_m.get("平均收益%", 0.0) > 0.0
        and te_m.get("信息比", 0.0) > 0.0
    )
    return {
        "combo": combo,
        "train": tr_m,
        "test": te_m,
        "train_score": train_score,
        "robust": robust,
        "做多笔数": float(tr_long + te_long),
        "做空笔数": float(tr_short + te_short),
    }


def search_short_term(
    data: dict[str, pd.DataFrame],
    *,
    rebalance_options: tuple[int, ...] = (2, 3),
    top_picks: int = 5,
    forward_days: int = 20,
    include_short: bool = True,
    split_ratio: float = 0.6,
    min_trades: int = 30,
    progress: Any = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """对整个搜索空间评估并按稳健性排序。

    返回 (排行榜 DataFrame, 原始评估结果列表)。排序优先稳健通过、再按样本内评分。
    """
    space = build_search_space(
        rebalance_options=rebalance_options, top_picks=top_picks,
        forward_days=forward_days, include_short=include_short,
    )
    results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    total = len(space)
    for k, combo in enumerate(space):
        ev = evaluate_combo(data, combo, split_ratio=split_ratio, min_trades=min_trades)
        if progress is not None:
            try:
                progress(k + 1, total, combo)
            except Exception:  # noqa: BLE001
                pass
        if not ev or not ev.get("train"):
            continue
        results.append(ev)
        tr, te = ev["train"], ev["test"]
        rows.append({
            "组合": combo.label,
            "思路": combo.idea,
            "交易策略": combo.trading_strategy,
            "参数": ",".join(f"{x}={y}" for x, y in combo.params.items()),
            "看盘日": combo.filters.lookback_days,
            "调仓日": combo.rebalance_days,
            "方向": "可空" if combo.allow_short else "仅多",
            "稳健通过": "✅" if ev["robust"] else "",
            "样本内评分": round(ev["train_score"], 3) if np.isfinite(ev["train_score"]) else np.nan,
            "内-笔数": int(tr.get("笔数", 0)),
            "内-胜率": round(tr.get("胜率", 0) * 100, 1),
            "内-平均收益%": round(tr.get("平均收益%", 0), 2),
            "内-信息比": round(tr.get("信息比", 0), 3),
            "内-盈亏比": round(tr.get("盈亏比", 0), 2),
            "外-笔数": int(te.get("笔数", 0)),
            "外-胜率": round(te.get("胜率", 0) * 100, 1),
            "外-平均收益%": round(te.get("平均收益%", 0), 2),
            "外-信息比": round(te.get("信息比", 0), 3),
            "外-盈亏比": round(te.get("盈亏比", 0), 2),
            "外-最差单笔%": round(te.get("最差单笔%", 0), 2),
            "_id": combo.id,
        })

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            ["稳健通过", "样本内评分"], ascending=[False, False],
        ).reset_index(drop=True)
    return table, results


def combo_to_preset(combo: CandidateCombo, name: str, rationale: str) -> Any:
    """把搜索胜出的组合固化为 ScreenStrategyPreset（延迟导入避免循环依赖）。"""
    from .screen_strategies import ScreenStrategyPreset

    pool = "day_gainers"
    if combo.idea == "超跌反弹":
        pool = "day_losers"
    elif combo.idea == "突破":
        pool = "most_actives"
    return ScreenStrategyPreset(
        id=f"opt_{combo.id}".lower()[:48],
        name=name,
        rationale=rationale,
        pool=pool,
        pool_size=50,
        filters=combo.filters,
        trading_strategy=combo.trading_strategy,
        trading_params=dict(combo.params),
        top_picks=combo.top_picks,
        rebalance_days=combo.rebalance_days,
        forward_eval_days=combo.forward_days,
        horizon="短线",
    )
