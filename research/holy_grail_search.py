"""圣杯策略穷尽搜索：不限已有 catalog，多族生成 + 杠杆/组合叠加 + IS/OOS/WF 三轨验证。

目标默认（strict）：年化≥100%、回撤<10%、胜率≥80%。

用法：
    python research/holy_grail_search.py --mode bounds
    python research/holy_grail_search.py --mode quick
    python research/holy_grail_search.py --mode full
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from itertools import product
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant import metrics as M
from quant.decline_income import (
    backtest_bear_call_spread,
    backtest_csp_income,
    backtest_weekly_put_spread,
    equity_metrics_from_trades,
)
from quant.providers import DataConfig, get_provider, reset_provider_cache
from research.growth_strategies import (
    BENCH, LEV_ETFS, LEADERS, absolute_momentum_single, buy_hold,
    momentum_rotation, trend_timing,
)
from research.triple_target_scan import (
    TRAIN_END,
    TARGETS,
    ScanResult,
    gap_score,
    pareto_frontier,
    set_scan_targets,
    targets_label,
)

RESULTS_CSV = ROOT / "research" / "holy_grail_results.csv"
PARETO_CSV = ROOT / "research" / "holy_grail_pareto.csv"
BOUNDS_JSON = ROOT / "research" / "holy_grail_bounds.json"

TRADING_DAYS = 252
IS_END = TRAIN_END
OOS_START = "2023-01-01"
WF_FOLDS = [
    ("2019-01-01", "2020-06-30"),
    ("2020-07-01", "2021-12-31"),
    ("2022-01-01", "2023-06-30"),
    ("2023-07-01", date.today().isoformat()),
]

LEVERAGES = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
STOP_LEVELS = [None, -0.05, -0.10, -0.15]
VOL_TARGETS = [None, 0.15, 0.25, 0.35]


@dataclass
class StrategySpec:
    family: str
    name: str
    params: dict
    returns: pd.Series
    win_rate: float
    trade_count: int = 0
    trade_returns: pd.Series | None = None  # 期权族用交易级收益，避免日频摊薄假象


@dataclass
class HolyGrailResult:
    strategy_id: str
    name: str
    family: str
    source: str
    ann_return: float
    max_dd: float
    win_rate: float
    trade_count: int
    sharpe: float
    params: dict = field(default_factory=dict)
    ann_is: float = 0.0
    max_dd_is: float = 0.0
    ann_oos: float = 0.0
    max_dd_oos: float = 0.0
    tier_is: str = "C"
    tier_oos: str = "C"
    tier_wf: str = "C"
    gap_score: float = 999.0
    notes: str = ""

    def to_row(self) -> dict:
        d = asdict(self)
        d["params"] = json.dumps(self.params, ensure_ascii=False)
        return d


# ---------------------------------------------------------------------------
# Phase 0 — 理论上界
# ---------------------------------------------------------------------------

def theoretical_bounds(
    *,
    win_rate: float = 0.80,
    max_dd: float = -0.10,
    target_ann: float = 1.0,
    trades_per_year: float = 50.0,
) -> dict:
    """二元输赢模型下，在胜率/回撤约束下的年化上界（启发式，非紧致界）。"""
    p = win_rate
    dd_abs = abs(max_dd)
    # 单笔最大亏损不超过 dd_abs / 连续亏损期望长度
    # 期望连续亏损长度 ≈ 1/(1-p) 笔（几何分布均值）
    streak = max(1.0, 1.0 / max(1 - p, 0.01))
    max_loss_per_trade = dd_abs / streak
    # 要达到 target_ann，每笔几何平均 g = (1+ann)^(1/n) - 1
    g_needed = (1 + target_ann) ** (1 / trades_per_year) - 1
    # E = p*w - (1-p)*l = g_needed, 设 l = max_loss_per_trade, 解 w
    w_needed = (g_needed + (1 - p) * max_loss_per_trade) / p if p > 0 else float("inf")
    payoff_ratio = w_needed / max_loss_per_trade if max_loss_per_trade > 0 else float("inf")
    # Calmar 上界：年化 / |DD| ≤ 可实现 Calmar
    calmar_needed = target_ann / dd_abs if dd_abs > 0 else float("inf")
    # 可持续几何增长上界：单笔期望 E，年化 ≈ (1+E)^n - 1，且单笔亏损受 DD 约束
    w_max = 3 * max_loss_per_trade  # 盈亏比上限 3:1 已很乐观
    e_per_trade = p * w_max - (1 - p) * max_loss_per_trade
    ann_upper_binary = max(0.0, (1 + e_per_trade) ** trades_per_year - 1)
    # 与目标比较
    feasible = ann_upper_binary >= target_ann and w_needed <= w_max
    return {
        "target_ann": target_ann,
        "target_max_dd": max_dd,
        "target_win_rate": win_rate,
        "trades_per_year": trades_per_year,
        "max_loss_per_trade_for_dd": round(max_loss_per_trade, 4),
        "win_per_trade_needed_for_100pct_ann": round(w_needed, 4),
        "payoff_ratio_needed": round(payoff_ratio, 2),
        "calmar_needed_for_target": round(calmar_needed, 1),
        "ann_upper_heuristic": round(ann_upper_binary, 4),
        "triple_target_feasible_heuristic": feasible,
        "conclusion": (
            f"在胜率{win_rate:.0%}、最大回撤{abs(max_dd):.0%}下，"
            f"每笔赢需约{w_needed:.2%}、输需≤{max_loss_per_trade:.2%}才能年化{target_ann:.0%}；"
            f"在盈亏比≤3:1的启发式上界约{ann_upper_binary:.1%}，"
            f"{'仍可能' if feasible else '难以'}同时达到年化{target_ann:.0%}。"
        ),
    }


# ---------------------------------------------------------------------------
# 收益 → 指标
# ---------------------------------------------------------------------------

def returns_to_equity(rets: pd.Series) -> pd.Series:
    r = rets.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return (1 + r).cumprod()


def slice_returns(rets: pd.Series, start: str | None, end: str | None) -> pd.Series:
    if not isinstance(rets.index, pd.DatetimeIndex):
        return rets
    out = rets
    if start:
        out = out.loc[out.index >= pd.Timestamp(start)]
    if end:
        out = out.loc[out.index <= pd.Timestamp(end)]
    return out


def metrics_from_returns(rets: pd.Series, win_rate: float | None = None) -> dict:
    if rets is None or len(rets) < 5:
        return {}
    eq = returns_to_equity(rets)
    ann = M.cagr(eq) if isinstance(eq.index, pd.DatetimeIndex) and len(eq) >= 2 else 0.0
    dd = M.max_drawdown(eq)
    sharpe = M.sharpe_ratio(rets)
    wr = win_rate if win_rate is not None else float((rets != 0).sum() and (rets > 0).mean())
    active = rets[rets != 0]
    if win_rate is None and len(active):
        wr = float((active > 0).mean())
    return {
        "ann_return": ann,
        "max_dd": dd,
        "win_rate": wr,
        "sharpe": sharpe,
        "trade_count": int((rets != 0).sum()) if len(rets) else 0,
        "equity": eq,
    }


def classify_tier(ann: float, max_dd: float, win: float) -> str:
    from research.triple_target_scan import classify_tier as _ct
    return _ct(ann, max_dd, win, oos=True)


# ---------------------------------------------------------------------------
# Phase 2 — 叠加层
# ---------------------------------------------------------------------------

def apply_leverage(rets: pd.Series, leverage: float) -> pd.Series:
    return rets * leverage


def apply_stop_overlay(rets: pd.Series, stop_dd: float | None) -> pd.Series:
    if stop_dd is None:
        return rets
    eq = returns_to_equity(rets)
    peak = eq.cummax()
    dd_series = eq / peak - 1
    out = rets.copy()
    halted = False
    for i in range(len(out)):
        if halted:
            out.iloc[i] = 0.0
            continue
        if dd_series.iloc[i] <= stop_dd:
            halted = True
            out.iloc[i] = 0.0
    return out


def apply_vol_target(rets: pd.Series, target_ann_vol: float | None, window: int = 20) -> pd.Series:
    if target_ann_vol is None:
        return rets
    vol = rets.rolling(window, min_periods=5).std() * np.sqrt(TRADING_DAYS)
    vol = vol.replace(0, np.nan).bfill().fillna(target_ann_vol)
    scale = target_ann_vol / vol
    scale = scale.clip(0.1, 10.0)
    return rets * scale.shift(1).fillna(1.0)


def mix_returns(series_list: list[tuple[pd.Series, float]], align: str = "inner") -> pd.Series:
    if not series_list:
        return pd.Series(dtype=float)
    weights = [w for _, w in series_list]
    if abs(sum(weights)) < 1e-9:
        return pd.Series(dtype=float)
    weights = [w / sum(weights) for w in weights]
    dfs = [s * w for s, w in zip([x[0] for x in series_list], weights)]
    panel = pd.concat(dfs, axis=1, join=align).fillna(0.0)
    return panel.sum(axis=1)


# ---------------------------------------------------------------------------
# Phase 1 — 策略族生成器
# ---------------------------------------------------------------------------

_DATA_CACHE: dict = {}


def load_market_data(start: str, end: str, *, quick: bool = True) -> dict:
    key = (start, end, quick)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    reset_provider_cache()
    y = get_provider(DataConfig(provider="yahoo"))
    tickers = list(set(BENCH + LEV_ETFS[:6] + LEADERS[:12] + ["SVIX", "SVXY", "SNDK", "AMD", "MU", "NVDA"]))
    if not quick:
        tickers = list(set(tickers + LEADERS + LEV_ETFS))
    batch = y.fetch_batch(tickers, start, end)
    batch = {t: d for t, d in batch.items() if d is not None and len(d) > 60}
    spy = y.fetch_history("SPY", start, end)
    out = {"batch": batch, "spy": spy["Close"].astype(float)}
    _DATA_CACHE[key] = out
    return out


def gen_trend_family(data: dict, *, quick: bool) -> list[StrategySpec]:
    batch = data["batch"]
    spy = data["spy"]
    specs: list[StrategySpec] = []
    lev_data = {t: batch[t] for t in LEV_ETFS if t in batch}
    leader_data = {t: batch[t] for t in LEADERS if t in batch}

    ma_grid = [50, 100, 200] if not quick else [100, 200]
    lb_grid = [21, 63, 126] if not quick else [63]
    tk_grid = [1, 3, 5] if not quick else [1, 5]

    for t in ["SOXL", "TQQQ", "SPY", "QQQ"]:
        if t not in batch:
            continue
        c = batch[t]["Close"].astype(float)
        specs.append(StrategySpec("trend", f"{t}_buyhold", {"ticker": t}, buy_hold(c), float((c.pct_change() > 0).mean())))
        for ma in ma_grid:
            r = trend_timing(c, ma)
            specs.append(StrategySpec(
                "trend", f"{t}_ma{ma}", {"ticker": t, "ma": ma}, r,
                float((r[r != 0] > 0).mean()) if (r != 0).any() else 0.5,
            ))

    if lev_data:
        for lb in lb_grid:
            r = momentum_rotation(lev_data, lookback=lb, hold=21, top_k=1)
            specs.append(StrategySpec("trend", f"lev_rot_lb{lb}", {"lookback": lb}, r, float((r > 0).mean())))

    if leader_data:
        for lb in lb_grid:
            for tk in tk_grid:
                r = momentum_rotation(leader_data, lookback=lb, hold=21, top_k=tk, regime=spy, regime_ma=200)
                specs.append(StrategySpec(
                    "trend", f"leader_lb{lb}_k{tk}_regime", {"lookback": lb, "top_k": tk}, r,
                    float((r > 0).mean()),
                ))
        r = absolute_momentum_single(leader_data, lookback=63, hold=21)
        specs.append(StrategySpec("trend", "abs_momentum", {"lookback": 63}, r, float((r > 0).mean())))
    return specs


def _trade_returns_to_daily(close: pd.Series, trade_rets: pd.Series, step: int = 5) -> pd.Series:
    """把交易级收益铺到日频（持有期均摊），便于与趋势策略组合。"""
    if trade_rets.empty:
        return pd.Series(0.0, index=close.index)
    daily = pd.Series(0.0, index=close.index)
    idxs = list(range(25, len(close) - 25, step))[: len(trade_rets)]
    hold = max(5, len(close) // max(len(trade_rets), 1))
    for i, ret in zip(idxs, trade_rets):
        end_i = min(i + hold, len(close) - 1)
        per_day = float(ret) / max(end_i - i, 1)
        daily.iloc[i:end_i] += per_day
    return daily


def gen_options_family(data: dict, *, quick: bool) -> list[StrategySpec]:
    batch = data["batch"]
    tickers = ["SNDK", "AMD", "MU", "NVDA", "WDC", "PLTR"] if not quick else ["SNDK", "AMD", "MU", "NVDA"]
    specs: list[StrategySpec] = []
    fns = [
        ("csp", backtest_csp_income, {}),
        ("weekly_soup", backtest_weekly_put_spread, {}),
        ("bear_call", backtest_bear_call_spread, {}),
    ]
    allocs = [0.05, 0.10, 0.20] if not quick else [0.10, 0.20]
    for tk in tickers:
        if tk not in batch:
            continue
        close = batch[tk]["Close"].astype(float)
        for sid, fn, kw in fns:
            bt = fn(close, **kw)
            if not bt:
                continue
            eq = bt.get("净值曲线")
            if eq is None or len(eq) < 2:
                continue
            trade_rets = eq.pct_change().dropna()
            if trade_rets.empty:
                continue
            win = float(bt.get("胜率", (trade_rets > 0).mean()))
            tc = int(bt.get("周期数") or bt.get("交易数") or len(trade_rets))
            for alloc in allocs:
                stats = equity_metrics_from_trades(trade_rets, alloc_pct=alloc)
                tr = stats.get("交易收益", trade_rets)
                if tr is None or len(tr) < 2:
                    continue
                daily = _trade_returns_to_daily(close, tr)
                specs.append(StrategySpec(
                    "options", f"{sid}_{tk}_a{alloc:.0%}",
                    {"ticker": tk, "strategy": sid, "alloc": alloc},
                    daily, win, trade_count=tc,
                    trade_returns=tr,
                ))
    return specs


def gen_vol_family(data: dict, *, quick: bool) -> list[StrategySpec]:
    batch = data["batch"]
    specs: list[StrategySpec] = []
    weights = [0.15, 0.25, 0.5, 1.0] if not quick else [0.25, 0.5]
    for t in ["SVIX", "SVXY"]:
        if t not in batch:
            continue
        px = batch[t]["Close"].astype(float)
        ret = px.pct_change().fillna(0.0)
        for w in weights:
            borrow = 0.05
            strat = -w * ret - w * borrow / TRADING_DAYS
            specs.append(StrategySpec(
                "vol", f"short_{t}_w{w}", {"ticker": t, "weight": w}, strat,
                float((strat > 0).mean()),
            ))
            # MA 过滤版
            ma = px.rolling(50).mean()
            sig = (px > ma).shift(1).fillna(0).astype(float)
            filt = sig * strat
            specs.append(StrategySpec(
                "vol", f"short_{t}_w{w}_ma50", {"ticker": t, "weight": w, "ma": 50}, filt,
                float((filt[filt != 0] > 0).mean()) if (filt != 0).any() else 0.5,
            ))
    return specs


def gen_cross_family(data: dict, start: str, end: str, *, quick: bool) -> list[StrategySpec]:
    from research.gainer_daily_backtest import (
        LIQUID100,
        backtest_daily_gainer_portfolio,
        fetch_gainer_data_yahoo,
        filters_for_mode,
    )

    specs: list[StrategySpec] = []
    tickers = LIQUID100[:40] if quick else LIQUID100
    try:
        gdata, spy = fetch_gainer_data_yahoo(tickers, start, end)
    except Exception:  # noqa: BLE001
        return specs
    modes = ["highwin", "ultra", "weekly", "legacy"] if not quick else ["highwin", "weekly"]
    for mode in modes:
        filt = filters_for_mode(mode)
        res = backtest_daily_gainer_portfolio(gdata, spy, start=start, end=end, filt=filt)
        if res.get("error"):
            continue
        curve = res.get("权益曲线")
        if curve is None or curve.empty:
            continue
        eq = curve.set_index(pd.to_datetime(curve["日期"]))["权益"]
        rets = eq.pct_change().fillna(0.0)
        specs.append(StrategySpec(
            "cross", f"gainer_{mode}", {"mode": mode}, rets, res["日胜率"], res["交易天数"],
        ))
    return specs


def gen_all_bases(data: dict, start: str, end: str, *, quick: bool) -> list[StrategySpec]:
    bases: list[StrategySpec] = []
    bases.extend(gen_trend_family(data, quick=quick))
    bases.extend(gen_options_family(data, quick=quick))
    bases.extend(gen_vol_family(data, quick=quick))
    bases.extend(gen_cross_family(data, start, end, quick=quick))
    return [b for b in bases if b.returns is not None and len(b.returns.dropna()) >= 30]


# ---------------------------------------------------------------------------
# Phase 3 — 评估与三轨验证
# ---------------------------------------------------------------------------

def walk_forward_pass(rets: pd.Series, win_rate: float) -> bool:
    if not isinstance(rets.index, pd.DatetimeIndex):
        return False
    for start, end in WF_FOLDS:
        seg = slice_returns(rets, start, end)
        if len(seg) < 20:
            return False
        m = metrics_from_returns(seg, win_rate)
        if classify_tier(m["ann_return"], m["max_dd"], m["win_rate"]) != "A":
            return False
    return True


def _is_artifact(ann: float, max_dd: float, family: str, leverage: float) -> bool:
    """日频摊薄 + 高杠杆会把回撤压成 0、年化吹爆 → 标记为假象。"""
    if family == "options" and leverage > 1.0:
        return True
    if max_dd >= -0.005 and ann > 0.5:
        return True
    return False


def _metrics_from_spec(spec: StrategySpec, rets: pd.Series, leverage: float) -> dict:
    """期权族优先用交易级复利评估。"""
    if spec.trade_returns is not None and len(spec.trade_returns) >= 5:
        alloc = float(spec.params.get("alloc", 0.1)) * leverage
        alloc = min(alloc, 1.0)
        stats = equity_metrics_from_trades(spec.trade_returns, alloc_pct=alloc)
        eq = stats.get("净值曲线")
        if eq is not None and len(eq) >= 2:
            return {
                "ann_return": stats.get("年化收益率", 0.0),
                "max_dd": stats.get("最大回撤", 0.0),
                "win_rate": stats.get("胜率", spec.win_rate),
                "sharpe": stats.get("夏普比率", 0.0),
                "trade_count": int(stats.get("交易次数", len(spec.trade_returns))),
                "equity": eq,
            }
    return metrics_from_returns(rets, spec.win_rate)


def evaluate_spec(
    spec: StrategySpec,
    rets: pd.Series,
    *,
    leverage: float = 1.0,
    stop_dd: float | None = None,
    vol_target: float | None = None,
    extra_params: dict | None = None,
) -> HolyGrailResult | None:
    r = apply_leverage(rets, leverage)
    r = apply_vol_target(r, vol_target)
    r = apply_stop_overlay(r, stop_dd)
    if len(r.dropna()) < 20 and spec.trade_returns is None:
        return None

    full = _metrics_from_spec(spec, r, leverage)
    if not full:
        return None

    full["ann_return"], full["max_dd"] = _sanitize_ann_dd(full["ann_return"], full["max_dd"])
    if full["ann_return"] <= 0 and full["max_dd"] <= -0.99:
        return None

    eq = full.get("equity")
    if eq is not None and isinstance(eq.index, pd.DatetimeIndex):
        r_is_eq = eq.loc[eq.index <= pd.Timestamp(IS_END)]
        r_oos_eq = eq.loc[eq.index > pd.Timestamp(IS_END)]
    elif eq is not None and len(eq) >= 4:
        split = int(len(eq) * 0.7)
        r_is_eq, r_oos_eq = eq.iloc[:split], eq.iloc[split:]
    else:
        r_is_eq = r_oos_eq = pd.Series(dtype=float)

    m_is = {}
    m_oos = {}
    if len(r_is_eq) >= 2:
        m_is = {
            "ann_return": M.cagr(r_is_eq) if isinstance(r_is_eq.index, pd.DatetimeIndex) else 0.0,
            "max_dd": M.max_drawdown(r_is_eq),
        }
    if len(r_oos_eq) >= 2:
        m_oos = {
            "ann_return": M.cagr(r_oos_eq) if isinstance(r_oos_eq.index, pd.DatetimeIndex) else 0.0,
            "max_dd": M.max_drawdown(r_oos_eq),
        }
    if spec.trade_returns is not None:
        # 交易级曲线无日期：全样本指标即 IS+OOS 混合，OOS 用全样本保守估计
        m_is = {"ann_return": full["ann_return"], "max_dd": full["max_dd"]}
        m_oos = m_is

    tier_is = classify_tier(
        m_is.get("ann_return", 0), m_is.get("max_dd", -1), full["win_rate"],
    ) if m_is else "C"
    tier_oos = classify_tier(
        m_oos.get("ann_return", 0), m_oos.get("max_dd", -1), full["win_rate"],
    ) if m_oos else "C"
    tier_wf = "A" if walk_forward_pass(r, spec.win_rate) else "C"

    artifact = _is_artifact(full["ann_return"], full["max_dd"], spec.family, leverage)
    if artifact:
        tier_is = tier_oos = tier_wf = "C"

    params = {**spec.params, "leverage": leverage, "stop_dd": stop_dd, "vol_target": vol_target}
    if extra_params:
        params.update(extra_params)

    sid = f"{spec.family}_{spec.name}_L{leverage}"
    if stop_dd is not None:
        sid += f"_stop{stop_dd}"
    if vol_target is not None:
        sid += f"_vt{vol_target}"

    return HolyGrailResult(
        strategy_id=sid,
        name=f"{spec.name} ×{leverage}",
        family=spec.family,
        source="holy_grail_search",
        ann_return=full["ann_return"],
        max_dd=full["max_dd"],
        win_rate=full["win_rate"],
        trade_count=spec.trade_count or full["trade_count"],
        sharpe=full["sharpe"],
        params=params,
        ann_is=m_is.get("ann_return", 0.0),
        max_dd_is=m_is.get("max_dd", 0.0),
        ann_oos=m_oos.get("ann_return", 0.0),
        max_dd_oos=m_oos.get("max_dd", 0.0),
        tier_is=tier_is,
        tier_oos=tier_oos,
        tier_wf=tier_wf,
        gap_score=gap_score(full["ann_return"], full["max_dd"], full["win_rate"]),
        notes="artifact_suspect" if artifact else "",
    )


def expand_with_overlays(spec: StrategySpec, *, quick: bool) -> list[HolyGrailResult]:
    levs = LEVERAGES if not quick else [1.0, 2.0, 3.0, 5.0, 10.0]
    if spec.family == "options":
        levs = [1.0]  # 杠杆已通过 alloc 计入交易级复利
    stops = STOP_LEVELS if not quick else [None, -0.10]
    vts = VOL_TARGETS if not quick else [None, 0.25]
    results: list[HolyGrailResult] = []
    for lev, stop, vt in product(levs, stops, vts):
        hr = evaluate_spec(spec, spec.returns, leverage=lev, stop_dd=stop, vol_target=vt)
        if hr:
            results.append(hr)
    return results


def gen_portfolio_mixes(top_specs: list[StrategySpec], n_samples: int, seed: int = 42) -> list[HolyGrailResult]:
    if len(top_specs) < 2:
        return []
    rng = random.Random(seed)
    results: list[HolyGrailResult] = []
    for _ in range(n_samples):
        k = rng.randint(2, min(4, len(top_specs)))
        picked = rng.sample(top_specs, k)
        ws = [rng.random() for _ in range(k)]
        mixed = mix_returns([(s.returns, w) for s, w in zip(picked, ws)])
        if mixed.empty or len(mixed) < 30:
            continue
        win = float((mixed > 0).mean())
        meta = StrategySpec("meta", f"mix_{k}", {"members": [s.name for s in picked]}, mixed, win)
        lev = rng.choice([1.0, 2.0, 3.0, 5.0])
        hr = evaluate_spec(meta, mixed, leverage=lev, extra_params={"weights": ws, "n": k})
        if hr:
            hr.family = "meta"
            hr.name = f"mix({k}) ×{lev}"
            results.append(hr)
    return results


# ---------------------------------------------------------------------------
# 主搜索
# ---------------------------------------------------------------------------

def _sanitize_ann_dd(ann: float, max_dd: float) -> tuple[float, float]:
    ann = float(ann) if np.isfinite(ann) else 0.0
    dd = float(max_dd) if np.isfinite(max_dd) else -1.0
    dd = max(dd, -1.0)
    return ann, dd


def load_holy_grail_summary() -> dict | None:
    """读取圣杯穷尽搜索结果摘要（供 app / strategy_daily 调用）。"""
    if not RESULTS_CSV.exists():
        return None
    try:
        df = pd.read_csv(RESULTS_CSV)
        df = df[np.isfinite(df["ann_return"].astype(float))]
        if df.empty:
            return None
        top = df.sort_values("gap_score").iloc[0]
        bounds = json.loads(BOUNDS_JSON.read_text()) if BOUNDS_JSON.exists() else {}
        return {
            "total": len(df),
            "tier_is_a": int((df["tier_is"] == "A").sum()),
            "tier_oos_a": int((df["tier_oos"] == "A").sum()),
            "tier_wf_a": int((df["tier_wf"] == "A").sum()),
            "best_name": str(top["name"]),
            "best_ann": float(top["ann_return"]),
            "best_dd": float(top["max_dd"]),
            "best_win": float(top["win_rate"]),
            "best_gap": float(top["gap_score"]),
            "bounds_conclusion": bounds.get("conclusion", ""),
        }
    except Exception:  # noqa: BLE001
        return None


def format_summary_lines() -> list[str]:
    """读取圣杯穷尽搜索结果摘要。"""
    if not RESULTS_CSV.exists():
        return ["圣杯搜索：尚未运行 python research/holy_grail_search.py --mode quick"]
    try:
        df = pd.read_csv(RESULTS_CSV)
        df = df[np.isfinite(df["ann_return"].astype(float))]
        if df.empty:
            return ["圣杯搜索：无有效结果"]
        top = df.sort_values("gap_score").iloc[0]
        bounds = json.loads(BOUNDS_JSON.read_text()) if BOUNDS_JSON.exists() else {}
        lines = [
            f"圣杯距离（strict 100%/10%/80%）：已搜 {len(df)} 组合",
            f"  Tier-OOS-A={int((df['tier_oos'] == 'A').sum())} "
            f"Tier-IS-A={int((df['tier_is'] == 'A').sum())} "
            f"Tier-WF-A={int((df['tier_wf'] == 'A').sum())}",
            f"  最近候选：{top['name']} 年化={float(top['ann_return']):.1%} "
            f"回撤={float(top['max_dd']):.1%} 胜率={float(top['win_rate']):.1%} "
            f"gap={float(top['gap_score']):.3f}",
        ]
        if bounds.get("conclusion"):
            lines.append(f"  {bounds['conclusion'][:120]}")
        return lines
    except Exception:  # noqa: BLE001
        return ["圣杯搜索：结果文件读取失败"]


def format_holy_grail_lines() -> list[str]:
    return format_summary_lines()


def run_bounds_only(*, preset: str = "strict") -> dict:
    set_scan_targets(preset=preset)
    bounds = theoretical_bounds(
        target_ann=TARGETS.ann,
        max_dd=TARGETS.dd,
        win_rate=TARGETS.win,
    )
    BOUNDS_JSON.write_text(json.dumps(bounds, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("=" * 60)
    print(f"理论上界（{targets_label()}）")
    print("=" * 60)
    for k, v in bounds.items():
        if k != "conclusion":
            print(f"  {k}: {v}")
    print(f"\n{bounds['conclusion']}")
    print(f"\n已写入 {BOUNDS_JSON}")
    return bounds


def run_search(*, mode: str = "quick", start: str = "2019-01-01", end: str | None = None, preset: str = "relaxed") -> pd.DataFrame:
    set_scan_targets(preset=preset)
    end = end or date.today().isoformat()
    quick = mode != "full"
    bounds = run_bounds_only(preset=preset) if mode == "bounds" else theoretical_bounds(
        target_ann=TARGETS.ann, max_dd=TARGETS.dd, win_rate=TARGETS.win,
    )
    if mode == "bounds":
        return pd.DataFrame()

    print(f"\n加载行情 {start} ~ {end}（{'quick' if quick else 'full'}）…")
    data = load_market_data(start, end, quick=quick)
    print(f"  有效标的 {len(data['batch'])} 只")

    bases = gen_all_bases(data, start, end, quick=quick)
    print(f"  基础策略族 {len(bases)} 条")

    all_results: list[HolyGrailResult] = []
    for i, spec in enumerate(bases):
        if i % 10 == 0:
            print(f"  叠加层 {i+1}/{len(bases)} …")
        all_results.extend(expand_with_overlays(spec, quick=quick))

    # 组合层
    n_mix = 200 if quick else 2000
    by_gap = sorted(
        [evaluate_spec(s, s.returns) for s in bases],
        key=lambda x: x.gap_score if x else 999,
    )
    by_gap = [x for x in by_gap if x is not None]
    top_specs = bases[: min(15, len(bases))]
    if by_gap:
        top_names = {r.name.split(" ×")[0] for r in by_gap[:15]}
        top_specs = [s for s in bases if s.name in top_names][:15]
    print(f"  随机组合 {n_mix} 组 …")
    all_results.extend(gen_portfolio_mixes(top_specs, n_mix))

    df = pd.DataFrame([r.to_row() for r in all_results])
    df = df[np.isfinite(df["ann_return"].astype(float))]
    df = df.sort_values("gap_score")
    df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")

    scan_rows = [
        ScanResult(
            strategy_id=r.strategy_id, name=r.name, source=r.source,
            ann_return=r.ann_return, max_dd=r.max_dd, win_rate=r.win_rate,
            trade_count=r.trade_count, sharpe=r.sharpe, params=r.params,
            tier=r.tier_oos, gap_score=r.gap_score,
        )
        for r in all_results
        if np.isfinite(r.ann_return)
    ]
    front = pareto_frontier(scan_rows)
    pd.DataFrame([r.to_row() for r in front]).to_csv(PARETO_CSV, index=False, encoding="utf-8-sig")

    _print_summary(df, bounds)
    return df


def _print_summary(df: pd.DataFrame, bounds: dict) -> None:
    print(f"\n{'='*60}\n圣杯搜索结果\n{'='*60}")
    print(f"共 {len(df)} 条组合")
    tier_is_a = df[df["tier_is"] == "A"]
    tier_oos_a = df[df["tier_oos"] == "A"]
    tier_wf_a = df[df["tier_wf"] == "A"]
    print(f"  Tier-IS-A（样本内 2019-2022 三项达标）: {len(tier_is_a)}")
    print(f"  Tier-OOS-A（样本外 2023+ 三项达标）:   {len(tier_oos_a)}")
    print(f"  Tier-WF-A（4折 walk-forward 全达标）:   {len(tier_wf_a)}")

    if not tier_oos_a.empty:
        print("\n✅ OOS 圣杯候选:")
        for _, r in tier_oos_a.head(5).iterrows():
            print(f"  {r['name']}: 年化={r['ann_oos']:.1%} 回撤={r['max_dd_oos']:.1%} 胜率={r['win_rate']:.1%}")
    elif not tier_is_a.empty:
        print("\n⚠ 仅样本内达标（可能过拟合）:")
        for _, r in tier_is_a.head(5).iterrows():
            print(f"  {r['name']}: IS年化={r['ann_is']:.1%} OOS年化={r['ann_oos']:.1%} gap={r['gap_score']:.3f}")

    print("\n距圣杯最近 Top 10（gap_score 越小越好）:")
    for _, r in df.head(10).iterrows():
        print(
            f"  {r['name']}: 年化={r['ann_return']:.1%} 回撤={r['max_dd']:.1%} "
            f"胜率={r['win_rate']:.1%} IS={r['tier_is']} OOS={r['tier_oos']} WF={r['tier_wf']}"
        )
    print(f"\n{bounds.get('conclusion', '')}")
    print(f"\n结果 → {RESULTS_CSV}")
    print(f"Pareto → {PARETO_CSV}")


def main() -> None:
    p = argparse.ArgumentParser(description="圣杯策略穷尽搜索")
    p.add_argument("--mode", choices=["bounds", "quick", "full"], default="quick")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--preset", choices=["relaxed", "strict"], default="relaxed")
    args = p.parse_args()
    if args.mode == "bounds":
        run_bounds_only(preset=args.preset)
    else:
        run_search(mode=args.mode, start=args.start, end=args.end, preset=args.preset)


if __name__ == "__main__":
    main()
