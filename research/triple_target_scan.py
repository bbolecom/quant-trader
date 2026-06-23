"""三重目标策略扫描器：默认年化≥50%、最大回撤<15%、胜率≥85%。

Phase 1：汇总现有策略/CSV 锚点，统一净值口径初筛。
Phase 2：参数网格 + 收入引擎组合 + 杠杆维度搜索，输出 Tier A/B/C 与 Pareto 前沿。

用法：
    python research/triple_target_scan.py --mode quick
    python research/triple_target_scan.py --mode full --ann 0.5 --max-dd -0.15 --win 0.85
    python research/triple_target_scan.py --preset strict   # 100% / 10% / 80%
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant import metrics as M
from quant.decline_income import (
    DECLINE_INCOME_UNIVERSE,
    backtest_bear_call_spread,
    backtest_csp_income,
    backtest_weekly_put_spread,
    equity_metrics_from_trades,
)
from quant.providers import DataConfig, get_provider, reset_provider_cache


@dataclass
class ScanTargets:
    ann: float = 0.50
    dd: float = -0.15
    win: float = 0.85


TARGETS = ScanTargets()
TRAIN_END = "2022-12-31"
BASELINE_CSV = ROOT / "research" / "triple_scan_baseline.csv"
RESULTS_CSV = ROOT / "research" / "triple_target_results.csv"
PARETO_CSV = ROOT / "research" / "triple_target_pareto.csv"

PRESETS = {
    "relaxed": ScanTargets(0.50, -0.15, 0.85),
    "strict": ScanTargets(1.00, -0.10, 0.80),
}


def set_scan_targets(
    ann: float | None = None,
    max_dd: float | None = None,
    win: float | None = None,
    *,
    preset: str | None = "relaxed",
) -> ScanTargets:
    global TARGETS
    if preset and preset in PRESETS:
        TARGETS = ScanTargets(**asdict(PRESETS[preset]))
    else:
        TARGETS = ScanTargets()
    if ann is not None:
        TARGETS.ann = ann
    if max_dd is not None:
        TARGETS.dd = max_dd
    if win is not None:
        TARGETS.win = win
    return TARGETS


def targets_label() -> str:
    t = TARGETS
    return f"年化≥{t.ann:.0%}、回撤<{abs(t.dd):.0%}、胜率≥{t.win:.0%}"


@dataclass
class ScanResult:
    strategy_id: str
    name: str
    source: str
    ann_return: float
    max_dd: float
    win_rate: float
    trade_count: int
    sharpe: float
    params: dict = field(default_factory=dict)
    oos_pass: bool = False
    tier: str = "C"
    gap_score: float = 0.0
    sample_ok: bool = True
    notes: str = ""

    def to_row(self) -> dict:
        d = asdict(self)
        d["params"] = json.dumps(self.params, ensure_ascii=False)
        return d


def gap_score(ann: float, max_dd: float, win: float) -> float:
    """距三重目标的加权差距（越小越好）。"""
    t = TARGETS
    ann_gap = max(0.0, t.ann - ann)
    dd_gap = max(0.0, max_dd - t.dd)
    win_gap = max(0.0, t.win - win)
    return ann_gap * 2.0 + dd_gap * 3.0 + win_gap * 1.5


def classify_tier(ann: float, max_dd: float, win: float, *, oos: bool = True) -> str:
    t = TARGETS
    ann_ok = ann >= t.ann
    dd_ok = max_dd > t.dd
    win_ok = win >= t.win
    if ann_ok and dd_ok and win_ok and oos:
        return "A"
    misses = sum([not ann_ok, not dd_ok, not win_ok])
    if misses == 1:
        return "B"
    return "C"


def metrics_from_equity_curve(
    equity: pd.Series,
    *,
    trade_returns: pd.Series | None = None,
    win_rate_override: float | None = None,
    trade_count: int = 0,
) -> dict:
    if equity is None or len(equity) < 1:
        return {}
    if isinstance(equity.index, pd.DatetimeIndex) and len(equity) >= 2:
        rets = equity.pct_change().fillna(0.0)
        stats = M.summary(equity, rets, trade_returns=trade_returns, num_trades=trade_count)
    else:
        tr = trade_returns if trade_returns is not None else equity.pct_change().fillna(0.0)
        n_trades = trade_count or max(len(equity) - 1, 1)
        years = max(n_trades / 12.0, 0.1)
        ann = float(equity.iloc[-1] ** (1 / years) - 1) if len(equity) >= 1 else 0.0
        rets = equity.pct_change().fillna(0.0)
        stats = {
            "累计收益率": float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) >= 1 else 0.0,
            "年化收益率": ann,
            "最大回撤": M.max_drawdown(equity) if len(equity) >= 2 else 0.0,
            "夏普比率": M.sharpe_ratio(rets) if len(rets) > 1 else 0.0,
            "胜率": M.win_rate(tr) if trade_returns is not None else float((tr > 0).mean()),
            "交易次数": float(trade_count),
        }
    if win_rate_override is not None:
        stats["胜率"] = win_rate_override
    return stats


def split_equity_oos(equity: pd.Series, train_end: str = TRAIN_END) -> tuple[pd.Series, pd.Series]:
    if isinstance(equity.index, pd.DatetimeIndex) and len(equity) > 1:
        cut = pd.Timestamp(train_end)
        train = equity.loc[equity.index <= cut]
        test = equity.loc[equity.index > cut]
        if len(test) >= 2:
            return train, test
    # 交易级净值无日期：按 70/30 切分
    n = len(equity)
    if n < 4:
        return equity, equity.iloc[0:0]
    split = int(n * 0.7)
    return equity.iloc[:split], equity.iloc[split:]


def oos_meets_target(equity: pd.Series, win_rate: float) -> bool:
    _, test = split_equity_oos(equity)
    if len(test) < 2:
        return False
    if isinstance(test.index, pd.DatetimeIndex):
        ann = M.cagr(test)
    else:
        years = max(len(test) / 12.0, 0.1)
        ann = float(test.iloc[-1] ** (1 / years) - 1)
    dd = M.max_drawdown(test)
    return ann >= TARGETS.ann and dd > TARGETS.dd and win_rate >= TARGETS.win


def pareto_frontier(rows: list[ScanResult]) -> list[ScanResult]:
    """三维 (年化, -回撤, 胜率) 非支配前沿。"""
    if not rows:
        return []
    front: list[ScanResult] = []
    for a in rows:
        dominated = False
        for b in rows:
            if a is b:
                continue
            if (
                b.ann_return >= a.ann_return
                and b.max_dd >= a.max_dd
                and b.win_rate >= a.win_rate
                and (
                    b.ann_return > a.ann_return
                    or b.max_dd > a.max_dd
                    or b.win_rate > a.win_rate
                )
            ):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda x: (-x.ann_return, -x.max_dd, -x.win_rate))


def _make_result(
    strategy_id: str,
    name: str,
    source: str,
    equity: pd.Series,
    *,
    win_rate: float,
    trade_count: int,
    params: dict | None = None,
    notes: str = "",
    min_trades: int = 0,
) -> ScanResult | None:
    stats = metrics_from_equity_curve(equity, trade_count=trade_count, win_rate_override=win_rate)
    if not stats:
        return None
    ann = stats["年化收益率"]
    dd = stats["最大回撤"]
    sample_ok = trade_count >= min_trades if min_trades > 0 else True
    oos = oos_meets_target(equity, win_rate)
    tier = classify_tier(ann, dd, win_rate, oos=oos and sample_ok)
    return ScanResult(
        strategy_id=strategy_id,
        name=name,
        source=source,
        ann_return=ann,
        max_dd=dd,
        win_rate=win_rate,
        trade_count=trade_count,
        sharpe=stats.get("夏普比率", 0.0),
        params=params or {},
        oos_pass=oos,
        tier=tier,
        gap_score=gap_score(ann, dd, win_rate),
        sample_ok=sample_ok,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Phase 1 — 静态锚点 + 复跑
# ---------------------------------------------------------------------------

def load_ranker_catalog() -> list[ScanResult]:
    from research.strategy_ranker import CATALOG

    rows: list[ScanResult] = []
    for m in CATALOG:
        tier = classify_tier(m.ann_return, -abs(m.max_dd), m.win_rate, oos=False)
        rows.append(ScanResult(
            strategy_id=m.id,
            name=m.name,
            source="strategy_ranker.CATALOG",
            ann_return=m.ann_return,
            max_dd=-abs(m.max_dd),
            win_rate=m.win_rate,
            trade_count=0,
            sharpe=m.sharpe,
            params={"category": m.category},
            oos_pass=False,
            tier=tier,
            gap_score=gap_score(m.ann_return, -abs(m.max_dd), m.win_rate),
            sample_ok=False,
            notes=m.note,
        ))
    return rows


def load_csv_baselines() -> list[ScanResult]:
    rows: list[ScanResult] = []
    growth = ROOT / "research" / "growth_strategies_results.csv"
    if growth.exists():
        df = pd.read_csv(growth)
        for _, r in df.iterrows():
            ann = float(r["CAGR"])
            dd = float(r["最大回撤"])
            tier = classify_tier(ann, dd, 0.5, oos=False)
            rows.append(ScanResult(
                strategy_id=f"growth_{r['方案']}",
                name=str(r["方案"]),
                source="growth_strategies_results.csv",
                ann_return=ann,
                max_dd=dd,
                win_rate=0.5,
                trade_count=0,
                sharpe=float(r.get("夏普", 0)),
                oos_pass=False,
                tier=tier,
                gap_score=gap_score(ann, dd, 0.5),
                sample_ok=False,
                notes="无胜率列，假设50%",
            ))
    vol = ROOT / "research" / "vol_decay_putspread_results.csv"
    if vol.exists():
        df = pd.read_csv(vol)
        for _, r in df.iterrows():
            ann = float(r["CAGR"])
            dd = float(r["最大回撤"])
            win = float(r.get("胜率", 0.5))
            name = f"{r['标的']}|{r['结构']}"
            tier = classify_tier(ann, dd, win, oos=False)
            rows.append(ScanResult(
                strategy_id=f"vol_{name}",
                name=name,
                source="vol_decay_putspread_results.csv",
                ann_return=ann,
                max_dd=dd,
                win_rate=win,
                trade_count=0,
                sharpe=float(r.get("夏普", 0)),
                oos_pass=False,
                tier=tier,
                gap_score=gap_score(ann, dd, win),
                sample_ok=False,
            ))
    return rows


def rerun_gainer_modes(start: str, end: str) -> list[ScanResult]:
    from research.gainer_daily_backtest import (
        LIQUID100,
        compare_gainer_modes,
        fetch_gainer_data_yahoo,
        filters_for_mode,
        backtest_daily_gainer_portfolio,
    )

    data, spy = fetch_gainer_data_yahoo(LIQUID100, start, end)
    years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 0.1)
    rows: list[ScanResult] = []
    for mode in ["ultra", "highwin", "weekly", "legacy"]:
        filt = filters_for_mode(mode)
        res = backtest_daily_gainer_portfolio(data, spy, start=start, end=end, filt=filt)
        if res.get("error"):
            continue
        curve = res.get("权益曲线")
        if curve is None or curve.empty:
            continue
        eq = curve.set_index(pd.to_datetime(curve["日期"]))["权益"]
        r = _make_result(
            f"gainer_{mode}",
            f"gainer_{mode}",
            "gainer_daily_backtest",
            eq,
            win_rate=res["日胜率"],
            trade_count=res["交易天数"],
            params={"mode": mode, "top_n": filt.top_n},
        )
        if r:
            rows.append(r)
    return rows


def rerun_decline_income(start: str, end: str) -> list[ScanResult]:
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    rows: list[ScanResult] = []
    batch = yahoo.fetch_batch(DECLINE_INCOME_UNIVERSE[:8], start, end)
    for ticker, df in batch.items():
        if df is None or df.empty:
            continue
        close = df["Close"].astype(float)
        tests = [
            ("bear_call", backtest_bear_call_spread, {}),
            ("csp", backtest_csp_income, {}),
            ("weekly_soup", backtest_weekly_put_spread, {}),
        ]
        for sid, fn, kwargs in tests:
            bt = fn(close, **kwargs)
            if not bt:
                continue
            eq = bt.get("净值曲线")
            if eq is None or len(eq) < 2:
                continue
            tc = int(bt.get("周期数") or bt.get("交易数") or len(eq))
            wr = float(bt.get("胜率", 0.0))
            r = _make_result(
                f"{sid}_{ticker}",
                f"{sid}·{ticker}",
                "decline_income",
                eq,
                win_rate=wr,
                trade_count=tc,
                params={"ticker": ticker},
            )
            if r:
                rows.append(r)
    return rows


def rerun_growth_equity(start: str, end: str) -> list[ScanResult]:
    from research.growth_strategies import (
        LEV_ETFS, LEADERS, LOTTERY, BENCH,
        buy_hold, trend_timing, momentum_rotation,
        absolute_momentum_single, basket_buy_hold, _ret_to_equity,
    )

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    cache = ROOT / "research" / "gainer_universe_cache.json"
    uni = json.loads(cache.read_text()) if cache.exists() else []
    need = sorted(set(uni + LEV_ETFS + LEADERS + LOTTERY + BENCH))
    data = yahoo.fetch_batch(need, start, end)

    def closeof(t: str) -> pd.Series | None:
        df = data.get(t)
        return df["Close"].astype(float) if df is not None and not df.empty else None

    spy = closeof("SPY")
    schemes: dict[str, pd.Series] = {}
    for t in ["SPY", "QQQ", "SOXL"]:
        c = closeof(t)
        if c is not None:
            schemes[f"{t}_bh"] = buy_hold(c)
    for t, ma in [("SOXL", 100), ("TQQQ", 100)]:
        c = closeof(t)
        if c is not None:
            schemes[f"{t}_ma{ma}"] = trend_timing(c, ma)
    lev_data = {t: data[t] for t in LEV_ETFS if t in data}
    if lev_data:
        schemes["lev_rotation"] = momentum_rotation(lev_data, lookback=63, hold=21, top_k=1)
    leader_data = {t: data[t] for t in LEADERS if t in data}
    if leader_data:
        schemes["leader_top5"] = momentum_rotation(leader_data, lookback=63, hold=21, top_k=5)
        if spy is not None:
            schemes["leader_top5_regime"] = momentum_rotation(
                leader_data, lookback=63, hold=21, top_k=5, regime=spy, regime_ma=200,
            )
        schemes["abs_momentum"] = absolute_momentum_single(leader_data)
    lot_data = {t: data[t] for t in LOTTERY if t in data}
    if lot_data:
        schemes["lottery_basket"] = basket_buy_hold(lot_data)

    rows: list[ScanResult] = []
    for sid, rets in schemes.items():
        rets = rets.dropna()
        if len(rets) < 30:
            continue
        eq = _ret_to_equity(rets)
        eq.index = rets.index
        win = float((rets > 0).mean())
        r = _make_result(
            f"growth_{sid}",
            sid,
            "growth_strategies",
            eq,
            win_rate=win,
            trade_count=int((rets != 0).sum()),
        )
        if r:
            rows.append(r)
    return rows


def rerun_vol_ma(start: str, end: str) -> list[ScanResult]:
    from quant.vol_decay import ma_timing_backtest
    from quant.data import fetch_history

    rows: list[ScanResult] = []
    for ticker in ["SVIX", "SVXY"]:
        try:
            df = fetch_history(ticker, start=start, end=end)
        except Exception:
            continue
        close = df["Close"].astype(float)
        bt = ma_timing_backtest(close, ma_window=50)
        st = bt.get("均线择时", {})
        if not st:
            continue
        ret = close.pct_change().fillna(0.0)
        ma = close.rolling(50).mean()
        sig = (close > ma).shift(1).fillna(False).astype(float)
        strat_ret = sig * ret
        eq = (1 + strat_ret).cumprod()
        eq.index = close.index
        win = float((strat_ret[sig > 0] > 0).mean()) if (sig > 0).any() else 0.0
        r = _make_result(
            f"vol_ma_{ticker}",
            f"{ticker} MA50",
            "vol_decay",
            eq,
            win_rate=win,
            trade_count=int((sig.diff().abs() > 0).sum()),
            params={"ma": 50},
        )
        if r:
            rows.append(r)
    return rows


def run_baseline_scan(start: str, end: str, min_trades: int = 0) -> pd.DataFrame:
    rows: list[ScanResult] = []
    rows.extend(load_ranker_catalog())
    rows.extend(load_csv_baselines())
    print("复跑 gainer 模式…")
    rows.extend(rerun_gainer_modes(start, end))
    print("复跑 decline_income…")
    rows.extend(rerun_decline_income(start, end))
    print("复跑 growth 净值…")
    rows.extend(rerun_growth_equity(start, end))
    print("复跑 vol MA…")
    rows.extend(rerun_vol_ma(start, end))

    df = pd.DataFrame([r.to_row() for r in rows])
    df = df.sort_values("gap_score")
    df.to_csv(BASELINE_CSV, index=False)
    _print_tier_summary(df, "Phase 1 Baseline")
    return df


# ---------------------------------------------------------------------------
# Phase 2 — 参数网格 + 组合
# ---------------------------------------------------------------------------

def grid_gainer_leverage(
    start: str, end: str, leverages: list[float] | None = None,
) -> list[ScanResult]:
    from research.gainer_daily_backtest import (
        LIQUID100, fetch_gainer_data_yahoo, filters_for_mode,
        backtest_daily_gainer_portfolio,
    )

    leverages = leverages or [0.5, 1.0, 1.5, 2.0, 3.0]
    data, spy = fetch_gainer_data_yahoo(LIQUID100, start, end)
    rows: list[ScanResult] = []
    for mode in ["highwin", "ultra"]:
        filt = filters_for_mode(mode)
        res = backtest_daily_gainer_portfolio(data, spy, start=start, end=end, filt=filt)
        if res.get("error"):
            continue
        curve = res["权益曲线"]
        base_eq = curve.set_index(pd.to_datetime(curve["日期"]))["权益"]
        base_rets = base_eq.pct_change().fillna(0.0)
        for lev in leverages:
            lev_rets = base_rets * lev
            eq = (1 + lev_rets).cumprod()
            eq.index = base_eq.index
            r = _make_result(
                f"gainer_{mode}_lev{lev}",
                f"gainer {mode} ×{lev}",
                "grid_gainer",
                eq,
                win_rate=res["日胜率"],
                trade_count=res["交易天数"],
                params={"mode": mode, "leverage": lev},
            )
            if r:
                rows.append(r)
    return rows


def grid_csp_params(start: str, end: str) -> list[ScanResult]:
    from research.liquid_tier_a_scan import build_candidate_pool, _avg_dollar_vol

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    pool = build_candidate_pool(use_broad=False, max_names=60)
    batch = yahoo.fetch_batch(pool, start, end)
    liquid = []
    for tk, df in batch.items():
        if df is None or df.empty:
            continue
        dvol_m = _avg_dollar_vol(df["Close"], df["Volume"]) / 1e6
        if dvol_m >= 50:
            liquid.append((tk, df, dvol_m))
    liquid.sort(key=lambda x: -x[2])
    tickers = [x[0] for x in liquid[:25]]
    rows: list[ScanResult] = []
    for ticker in tickers:
        try:
            df = yahoo.fetch_history(ticker, start, end)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        close = df["Close"].astype(float)
        for delta, ma, tp, alloc in product(
            [0.15, 0.20, 0.25],
            [0, 50],
            [0.0, 0.5],
            [0.10, 0.20, 0.35, 0.50, 0.75],
        ):
            rors: list[float] = []
            from quant.vol_decay import (
                TRADING_DAYS, realized_vol, bs_put_price, strike_for_put_delta,
            )
            from quant.decline_income import CSP_HOLD_TD, CSP_STEP_TD, CSP_DTE_CAL, DEFAULT_VRP

            rv = realized_vol(close)
            ma_s = close.rolling(ma).mean() if ma else None
            T = CSP_DTE_CAL / TRADING_DAYS
            i = max(25, ma)
            while i + CSP_HOLD_TD < len(close):
                S = float(close.iloc[i])
                sigma = float(rv.iloc[i])
                if not np.isfinite(sigma) or sigma <= 0:
                    i += CSP_STEP_TD
                    continue
                if ma_s is not None and not (S > float(ma_s.iloc[i])):
                    i += CSP_STEP_TD
                    continue
                iv = sigma * (1 + DEFAULT_VRP)
                K = strike_for_put_delta(S, T, iv, target_delta=delta)
                credit = bs_put_price(S, K, T, iv)
                if K <= 0:
                    i += CSP_STEP_TD
                    continue
                ST = float(close.iloc[i + CSP_HOLD_TD])
                rors.append((credit - max(0.0, K - ST)) / K)
                i += CSP_STEP_TD
            if not rors:
                continue
            stats = equity_metrics_from_trades(rors, alloc_pct=alloc)
            eq = stats.get("净值曲线")
            if eq is None or len(eq) < 2:
                continue
            r = _make_result(
                f"csp_{ticker}_d{delta}_ma{ma}_a{alloc}",
                f"CSP {ticker} δ={delta} alloc={alloc:.0%}",
                "grid_csp",
                eq,
                win_rate=stats["胜率"],
                trade_count=len(rors),
                params={"ticker": ticker, "delta": delta, "ma": ma, "alloc": alloc, "tp": tp},
            )
            if r:
                rows.append(r)
    return rows


def grid_portfolio_combos(start: str, end: str) -> list[ScanResult]:
    """收入引擎三件套权重扫描（call_spread + gainer + csp 日收益合成）。"""
    from research.gainer_daily_backtest import (
        LIQUID100, fetch_gainer_data_yahoo, high_win_filters,
        backtest_daily_gainer_portfolio,
    )

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    data, spy = fetch_gainer_data_yahoo(LIQUID100, start, end)

    # gainer 日收益
    g_res = backtest_daily_gainer_portfolio(
        data, spy, start=start, end=end, filt=high_win_filters(),
    )
    if g_res.get("error"):
        return []
    g_curve = g_res["权益曲线"].set_index(pd.to_datetime(g_res["权益曲线"]["日期"]))
    g_rets = g_curve["权益"].pct_change().fillna(0.0)

    # bear call 代表 income（SNDK）
    df = yahoo.fetch_history("SNDK", start, end)
    bc = backtest_bear_call_spread(df["Close"].astype(float)) if df is not None and not df.empty else {}
    bc_eq = bc.get("净值曲线")
    if bc_eq is None or len(bc_eq) < 2:
        return []
    bc_rets = bc_eq.pct_change().fillna(0.0)
    bc_rets.index = pd.RangeIndex(len(bc_rets))

    # csp
    csp = backtest_csp_income(df["Close"].astype(float)) if df is not None and not df.empty else {}
    csp_eq = csp.get("净值曲线")
    if csp_eq is None or len(csp_eq) < 2:
        return []

    rows: list[ScanResult] = []
    weight_grid = [0.0, 0.2, 0.4, 0.6]
    for w_g, w_bc, w_csp in product(weight_grid, weight_grid, weight_grid):
        if abs(w_g + w_bc + w_csp - 1.0) > 1e-6 or w_g + w_bc + w_csp == 0:
            continue
        # 对齐到 gainer 交易日（最短公共索引）
        n = min(len(g_rets), len(bc_rets), len(csp_eq) - 1)
        if n < 10:
            continue
        csp_rets = csp_eq.pct_change().fillna(0.0).iloc[-n:]
        port_rets = (
            w_g * g_rets.iloc[-n:].values
            + w_bc * bc_rets.iloc[-n:].values
            + w_csp * csp_rets.values
        )
        port_rets = pd.Series(port_rets, index=g_rets.iloc[-n:].index)
        eq = (1 + port_rets).cumprod()
        win = float((port_rets > 0).mean())
        r = _make_result(
            f"combo_{w_g:.1f}_{w_bc:.1f}_{w_csp:.1f}",
            f"组合 g={w_g:.0%} bc={w_bc:.0%} csp={w_csp:.0%}",
            "portfolio_combo",
            eq,
            win_rate=win,
            trade_count=n,
            params={"w_gainer": w_g, "w_bear_call": w_bc, "w_csp": w_csp},
        )
        if r:
            rows.append(r)
    return rows


def run_full_scan(
    start: str,
    end: str,
    *,
    mode: str = "quick",
    min_trades: int = 30,
) -> pd.DataFrame:
    rows: list[ScanResult] = []

    # 先跑 baseline
    baseline = run_baseline_scan(start, end, min_trades=min_trades)
    for _, br in baseline.iterrows():
        rows.append(ScanResult(
            strategy_id=br["strategy_id"],
            name=br["name"],
            source=br["source"],
            ann_return=float(br["ann_return"]),
            max_dd=float(br["max_dd"]),
            win_rate=float(br["win_rate"]),
            trade_count=int(br["trade_count"]),
            sharpe=float(br.get("sharpe", 0)),
            params=json.loads(br["params"]) if isinstance(br["params"], str) and br["params"].startswith("{") else {},
            oos_pass=bool(br.get("oos_pass", False)),
            tier=str(br.get("tier", "C")),
            gap_score=float(br.get("gap_score", 999)),
            sample_ok=bool(br.get("sample_ok", True)),
            notes=str(br.get("notes", "")),
        ))

    print("网格：gainer 杠杆…")
    rows.extend(grid_gainer_leverage(start, end))

    if mode == "full":
        print("网格：CSP 参数…")
        rows.extend(grid_csp_params(start, end))
        print("网格：组合权重…")
        rows.extend(grid_portfolio_combos(start, end))

    # 重新分级（含 min_trades）
    for r in rows:
        if r.trade_count < min_trades and r.trade_count > 0:
            r.sample_ok = False
            if r.tier == "A":
                r.tier = "B"
        r.gap_score = gap_score(r.ann_return, r.max_dd, r.win_rate)

    df = pd.DataFrame([r.to_row() for r in rows])
    df = df.sort_values("gap_score")
    df.to_csv(RESULTS_CSV, index=False)

    front = pareto_frontier(rows)
    pd.DataFrame([r.to_row() for r in front]).to_csv(PARETO_CSV, index=False)

    _print_tier_summary(df, f"Phase 2 ({mode})")
    _print_conclusion(df, front)
    if mode == "full":
        print("\n--- 圣杯穷尽搜索（holy_grail_search --mode full）---")
        try:
            from research.holy_grail_search import run_search, format_summary_lines
            run_search(mode="full", start=start, end=end)
            for line in format_summary_lines():
                print(f"  {line}")
        except Exception as e:  # noqa: BLE001
            print(f"  圣杯搜索失败: {e}")
    return df


def _print_tier_summary(df: pd.DataFrame, title: str) -> None:
    print(f"\n{'='*60}\n{title}  [{targets_label()}]\n{'='*60}")
    for tier in ["A", "B", "C"]:
        sub = df[df["tier"] == tier]
        print(f"Tier {tier}: {len(sub)} 条")
    tier_a = df[df["tier"] == "A"]
    if not tier_a.empty:
        print("\n✅ Tier A（全达标 + OOS）:")
        for _, r in tier_a.head(10).iterrows():
            print(
                f"  {r['name']}: 年化={r['ann_return']:.1%} "
                f"回撤={r['max_dd']:.1%} 胜率={r['win_rate']:.1%}"
            )
    else:
        print("\n❌ 无 Tier A 达标方案")
    tier_b = df[df["tier"] == "B"].head(5)
    if not tier_b.empty:
        print("\n接近目标（差一项）Top 5:")
        for _, r in tier_b.iterrows():
            print(
                f"  {r['name']}: 年化={r['ann_return']:.1%} "
                f"回撤={r['max_dd']:.1%} 胜率={r['win_rate']:.1%} "
                f"gap={r['gap_score']:.3f}"
            )


def _print_conclusion(df: pd.DataFrame, front: list[ScanResult]) -> None:
    print(f"\n{'='*60}\n结论\n{'='*60}")
    tier_a = df[df["tier"] == "A"]
    if tier_a.empty:
        print(f"在 BS 近似回测下，无方案同时满足：{targets_label()}（含 OOS）。")
        best_ann = df.loc[df["ann_return"].idxmax()]
        best_dd = df.loc[df["max_dd"].idxmax()]
        best_win = df.loc[df["win_rate"].idxmax()]
        print(f"  最高年化: {best_ann['name']} → {best_ann['ann_return']:.1%} "
              f"(回撤 {best_ann['max_dd']:.1%}, 胜率 {best_ann['win_rate']:.1%})")
        print(f"  最小回撤: {best_dd['name']} → {best_dd['max_dd']:.1%} "
              f"(年化 {best_dd['ann_return']:.1%}, 胜率 {best_dd['win_rate']:.1%})")
        print(f"  最高胜率: {best_win['name']} → {best_win['win_rate']:.1%} "
              f"(年化 {best_win['ann_return']:.1%}, 回撤 {best_win['max_dd']:.1%})")
    else:
        print(f"发现 {len(tier_a)} 个 Tier A 方案，详见 {RESULTS_CSV}")
    print(f"Pareto 前沿 {len(front)} 条 → {PARETO_CSV}")


def main() -> None:
    p = argparse.ArgumentParser(description="三重目标策略扫描")
    p.add_argument("--mode", choices=["baseline", "quick", "full"], default="quick")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--min-trades", type=int, default=30)
    p.add_argument("--preset", choices=list(PRESETS), default="relaxed",
                   help="relaxed=50%%/15%%/85%%, strict=100%%/10%%/80%%")
    p.add_argument("--ann", type=float, default=None, help="目标年化（如 0.5）")
    p.add_argument("--max-dd", type=float, default=None, help="最大回撤下限（如 -0.15）")
    p.add_argument("--win", type=float, default=None, help="目标胜率（如 0.85）")
    args = p.parse_args()

    set_scan_targets(args.ann, args.max_dd, args.win, preset=args.preset)
    print(f"扫描目标：{targets_label()}")

    if args.mode == "baseline":
        run_baseline_scan(args.start, args.end, min_trades=args.min_trades)
    else:
        run_full_scan(args.start, args.end, mode=args.mode, min_trades=args.min_trades)


if __name__ == "__main__":
    main()
