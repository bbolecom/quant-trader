"""每周级多空方案：日胜率≥80%，约每周 1 次交易。

方案：
  1. 做多-高置信动量（温和涨幅 + 形态胜率）
  2. 做多-强SPY日（大盘涨≥0.4% + 强势收盘）
  3. 做多-顺势回调（大盘强 + 个股跌 1~5% + 趋势未破）
  4. 做空-超涨回吐（大涨 + 弱收盘 + 大盘弱）
  5. 组合：每日取置信度最高的单一信号（各子策略独立≥80% 胜率设计）

用法：
    python research/gainer_weekly_multi.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.providers import DataConfig, get_provider, reset_provider_cache
from research.gainer_daily_backtest import (
    GainerProFilters,
    backtest_daily_gainer_portfolio,
    build_factor_panels,
    edge_setup_filters,
    high_win_filters,
    market_regime_ok,
    pick_from_panel,
    precompute_setup_edge,
    ultra_high_win_filters,
    weekly_momentum_filters,
)

FEE = 5 / 10_000


@dataclass
class ShortFadeFilters:
    min_gain_pct: float = 6.0
    max_gain_pct: float = 15.0
    min_vol_ratio: float = 1.5
    max_vol_ratio: float = 6.0
    min_gain_20d_pct: float = 8.0
    max_gain_20d_pct: float = 60.0
    min_rs_20d_pct: float = 3.0
    max_close_strength: float = 0.50
    require_spy_negative_1d: bool = False
    min_spy_1d_pct: float = 0.0
    min_dollar_vol_m: float = 500.0
    min_setup_win_rate: float = 0.0
    min_setup_samples: int = 0
    top_n: int = 1


@dataclass
class DipBuyFilters:
    min_gain_pct: float = -5.0
    max_gain_pct: float = -1.0
    min_vol_ratio: float = 1.2
    max_vol_ratio: float = 2.5
    min_rs_20d_pct: float = 2.0
    max_rs_20d_pct: float = 25.0
    min_close_strength: float = 0.55
    require_above_ma50: bool = True
    require_spy_above_ma20: bool = True
    require_spy_positive_1d: bool = True
    min_spy_1d_pct: float = 0.2
    min_dollar_vol_m: float = 500.0
    min_setup_win_rate: float = 0.0
    min_setup_samples: int = 0
    top_n: int = 1


@dataclass
class NicheSignal:
    name: str
    side: str  # long | short
    tickers: pd.DataFrame
    confidence: float


def apply_short_filters(day: pd.DataFrame, filt: ShortFadeFilters) -> pd.DataFrame:
    df = day.copy()
    for c in ["涨幅%", "量比", "涨幅20d%", "相对SPY20d%", "收盘强度", "成交额USD"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    min_dollar = filt.min_dollar_vol_m * 1_000_000
    mask = (
        df["涨幅%"].between(filt.min_gain_pct, filt.max_gain_pct)
        & df["量比"].between(filt.min_vol_ratio, filt.max_vol_ratio)
        & df["涨幅20d%"].between(filt.min_gain_20d_pct, filt.max_gain_20d_pct)
        & (df["相对SPY20d%"] >= filt.min_rs_20d_pct)
        & (df["收盘强度"] <= filt.max_close_strength)
        & (df["成交额USD"].fillna(0) >= min_dollar)
    )
    return df.loc[mask].reset_index(drop=True)


def short_regime_ok(day: pd.DataFrame, filt: ShortFadeFilters) -> bool:
    row = day.iloc[0]
    if filt.require_spy_negative_1d:
        spy1 = pd.to_numeric(row.get("SPY1d涨%"), errors="coerce")
        if not np.isfinite(spy1) or spy1 >= 0:
            return False
    if filt.min_spy_1d_pct < 0:
        spy1 = pd.to_numeric(row.get("SPY1d涨%"), errors="coerce")
        if not np.isfinite(spy1) or spy1 > filt.min_spy_1d_pct:
            return False
    return True


def pick_short(day: pd.DataFrame, filt: ShortFadeFilters) -> pd.DataFrame:
    out = apply_short_filters(day, filt)
    if out.empty:
        return out
    gain = pd.to_numeric(out["涨幅%"], errors="coerce").fillna(0)
    cs = pd.to_numeric(out["收盘强度"], errors="coerce").fillna(0.5)
    vr = pd.to_numeric(out["量比"], errors="coerce").fillna(1)
    wr = pd.to_numeric(out.get("空近8次胜率"), errors="coerce").fillna(0.5)
    out = out.copy()
    out["综合分"] = gain * 0.2 + (1 - cs) * 10 + vr * 1.5 + wr * 5
    out = out.sort_values("综合分", ascending=False).head(filt.top_n)
    if filt.min_setup_win_rate > 0 and not out.empty:
        w = pd.to_numeric(out.iloc[0].get("空近8次胜率"), errors="coerce")
        n = pd.to_numeric(out.iloc[0].get("空近8次样本"), errors="coerce")
        if not (np.isfinite(w) and w >= filt.min_setup_win_rate and np.isfinite(n) and n >= filt.min_setup_samples):
            return pd.DataFrame()
    return out


def apply_dip_filters(day: pd.DataFrame, filt: DipBuyFilters) -> pd.DataFrame:
    df = day.copy()
    for c in ["涨幅%", "量比", "相对SPY20d%", "收盘强度", "成交额USD"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    min_dollar = filt.min_dollar_vol_m * 1_000_000
    mask = (
        df["涨幅%"].between(filt.min_gain_pct, filt.max_gain_pct)
        & df["量比"].between(filt.min_vol_ratio, filt.max_vol_ratio)
        & df["相对SPY20d%"].between(filt.min_rs_20d_pct, filt.max_rs_20d_pct)
        & (df["收盘强度"] >= filt.min_close_strength)
        & (df["成交额USD"].fillna(0) >= min_dollar)
    )
    if filt.require_above_ma50 and "站上MA50" in df.columns:
        mask &= df["站上MA50"].astype(bool)
    return df.loc[mask].reset_index(drop=True)


def dip_regime_ok(day: pd.DataFrame, filt: DipBuyFilters) -> bool:
    row = day.iloc[0]
    if filt.require_spy_above_ma20 and not bool(row.get("SPY站上MA20", True)):
        return False
    if filt.require_spy_positive_1d:
        spy1 = pd.to_numeric(row.get("SPY1d涨%"), errors="coerce")
        if not np.isfinite(spy1) or spy1 < filt.min_spy_1d_pct:
            return False
    return True


def pick_dip(day: pd.DataFrame, filt: DipBuyFilters) -> pd.DataFrame:
    out = apply_dip_filters(day, filt)
    if out.empty:
        return out
    out = out.copy()
    rs = pd.to_numeric(out["相对SPY20d%"], errors="coerce").fillna(0)
    cs = pd.to_numeric(out["收盘强度"], errors="coerce").fillna(0)
    wr = pd.to_numeric(out.get("回近8次胜率"), errors="coerce").fillna(0.5)
    out["综合分"] = rs * 0.3 + cs * 10 + wr * 5
    out = out.sort_values("综合分", ascending=False).head(filt.top_n)
    if filt.min_setup_win_rate > 0 and not out.empty:
        w = pd.to_numeric(out.iloc[0].get("回近8次胜率"), errors="coerce")
        n = pd.to_numeric(out.iloc[0].get("回近8次样本"), errors="coerce")
        if not (np.isfinite(w) and w >= filt.min_setup_win_rate and np.isfinite(n) and n >= filt.min_setup_samples):
            return pd.DataFrame()
    return out


def precompute_short_edge(panel: pd.DataFrame, fwd_ret: dict[str, pd.Series]) -> pd.DataFrame:
    base = ShortFadeFilters(min_gain_pct=5.0, max_gain_pct=15.0, min_vol_ratio=1.3, max_close_strength=0.60)
    out = panel.copy()
    out["空近8次胜率"] = np.nan
    out["空近8次样本"] = 0
    out["日期"] = pd.to_datetime(out["日期"])
    for ticker, grp in out.groupby("代码"):
        g = grp.sort_values("日期").copy()
        gains = pd.to_numeric(g["涨幅%"], errors="coerce").values
        vr = pd.to_numeric(g["量比"], errors="coerce").values
        g20 = pd.to_numeric(g["涨幅20d%"], errors="coerce").values
        cs = pd.to_numeric(g["收盘强度"], errors="coerce").values
        rs = pd.to_numeric(g["相对SPY20d%"], errors="coerce").values
        fr_s = fwd_ret.get(str(ticker))
        if fr_s is None:
            continue
        fr = np.array([float(fr_s.get(pd.Timestamp(d), np.nan)) for d in g["日期"].values])
        wins: list[float] = []
        rwr = np.full(len(g), np.nan)
        rn = np.zeros(len(g), dtype=int)
        for i in range(len(g)):
            if wins:
                tail = wins[-8:]
                rwr[i] = float(np.mean(tail))
                rn[i] = len(tail)
            ok = (
                np.isfinite(gains[i]) and 5 <= gains[i] <= 15
                and np.isfinite(vr[i]) and vr[i] >= 1.3
                and np.isfinite(cs[i]) and cs[i] <= 0.60
                and np.isfinite(g20[i]) and g20[i] >= 8
            )
            if ok and np.isfinite(fr[i]):
                wins.append(1.0 if fr[i] < 0 else 0.0)
                if len(wins) > 80:
                    wins.pop(0)
        out.loc[g.index, "空近8次胜率"] = rwr
        out.loc[g.index, "空近8次样本"] = rn
    return out


def precompute_dip_edge(panel: pd.DataFrame, fwd_ret: dict[str, pd.Series]) -> pd.DataFrame:
    out = panel.copy()
    out["回近8次胜率"] = np.nan
    out["回近8次样本"] = 0
    out["日期"] = pd.to_datetime(out["日期"])
    for ticker, grp in out.groupby("代码"):
        g = grp.sort_values("日期").copy()
        gains = pd.to_numeric(g["涨幅%"], errors="coerce").values
        vr = pd.to_numeric(g["量比"], errors="coerce").values
        cs = pd.to_numeric(g["收盘强度"], errors="coerce").values
        rs = pd.to_numeric(g["相对SPY20d%"], errors="coerce").values
        ma50 = g["站上MA50"].astype(bool).values
        fr_s = fwd_ret.get(str(ticker))
        if fr_s is None:
            continue
        fr = np.array([float(fr_s.get(pd.Timestamp(d), np.nan)) for d in g["日期"].values])
        wins: list[float] = []
        rwr = np.full(len(g), np.nan)
        rn = np.zeros(len(g), dtype=int)
        for i in range(len(g)):
            if wins:
                tail = wins[-8:]
                rwr[i] = float(np.mean(tail))
                rn[i] = len(tail)
            ok = (
                np.isfinite(gains[i]) and -5 <= gains[i] <= -1
                and np.isfinite(vr[i]) and 1.2 <= vr[i] <= 2.5
                and np.isfinite(cs[i]) and cs[i] >= 0.55
                and ma50[i]
            )
            if ok and np.isfinite(fr[i]):
                wins.append(1.0 if fr[i] > 0 else 0.0)
                if len(wins) > 80:
                    wins.pop(0)
        out.loc[g.index, "回近8次胜率"] = rwr
        out.loc[g.index, "回近8次样本"] = rn
    return out


def _conf(row: pd.Series, side: str) -> float:
    if side == "long":
        c = pd.to_numeric(row.get("近8次胜率"), errors="coerce")
    elif side == "short":
        c = pd.to_numeric(row.get("空近8次胜率"), errors="coerce")
    else:
        c = pd.to_numeric(row.get("回近8次胜率"), errors="coerce")
    return float(c) if np.isfinite(c) else 0.5


def niche_high_win_long() -> GainerProFilters:
    return high_win_filters(top_n=2)


def niche_strong_spy_long() -> GainerProFilters:
    return GainerProFilters(
        min_gain_pct=2.5, max_gain_pct=4.0, min_vol_ratio=1.2, max_vol_ratio=1.55,
        min_close_strength=0.68, require_spy_positive_1d=True, min_spy_1d_pct=0.4,
        min_setup_win_rate=0.0, min_setup_samples=0,
        require_above_ma50=True, require_spy_above_ma20=True, require_spy_positive_5d=True,
        require_green_candle=True, top_n=1, min_candidates=1,
        min_rs_20d_pct=2, max_rs_20d_pct=18, min_gain_20d_pct=3, max_gain_20d_pct=22,
    )


def niche_dip_buy() -> DipBuyFilters:
    return DipBuyFilters(
        min_gain_pct=-4.0, max_gain_pct=-1.0, min_close_strength=0.58,
        min_spy_1d_pct=0.25, min_setup_win_rate=0.625, min_setup_samples=5,
    )


def niche_short_fade() -> ShortFadeFilters:
    return ShortFadeFilters(
        min_gain_pct=7.0, max_gain_pct=14.0, min_vol_ratio=1.6, max_close_strength=0.48,
        require_spy_negative_1d=True, min_setup_win_rate=0.625, min_setup_samples=5,
        min_gain_20d_pct=10.0, min_rs_20d_pct=4.0,
    )


def collect_niches(day: pd.DataFrame, min_conf: float = 0.0, *, quality_only: bool = False) -> list[NicheSignal]:
    signals: list[NicheSignal] = []

    lf = niche_high_win_long()
    if market_regime_ok(day, lf):
        lg = pick_from_panel(day, lf)
        if not lg.empty:
            conf = _conf(lg.iloc[0], "long")
            if conf >= min_conf or min_conf <= 0:
                signals.append(NicheSignal("高置信动量做多", "long", lg, conf))

    if quality_only:
        return signals

    lf2 = niche_strong_spy_long()
    if market_regime_ok(day, lf2):
        lg2 = pick_from_panel(day, lf2)
        if not lg2.empty:
            signals.append(NicheSignal("强SPY日做多", "long", lg2, _conf(lg2.iloc[0], "long")))

    df = niche_dip_buy()
    if dip_regime_ok(day, df):
        dp = pick_dip(day, df)
        if not dp.empty:
            signals.append(NicheSignal("顺势回调做多", "long", dp, _conf(dp.iloc[0], "dip")))

    sf = niche_short_fade()
    if short_regime_ok(day, sf):
        sh = pick_short(day, sf)
        if not sh.empty:
            signals.append(NicheSignal("超涨回吐做空", "short", sh, _conf(sh.iloc[0], "short")))

    return signals


def backtest_weekly_best(
    panel: pd.DataFrame,
    fwd_ret: dict[str, pd.Series],
    *,
    start: str | date,
    end: str | date,
    min_conf: float = 0.55,
    quality_only: bool = False,
) -> dict:
    """每周选 1 次置信度最高的信号（自然≈每周 1 笔）。"""
    cal = panel["日期"].drop_duplicates().sort_values()
    cal = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    by_date = {d: g for d, g in panel.groupby("日期")}
    daily_rets: list[float] = []
    picks_log: list[dict] = []

    i = 50
    while i < len(cal) - 1:
        week_dates = cal.iloc[i : min(i + 5, len(cal) - 1)]
        best_sig: NicheSignal | None = None
        best_date = None
        for as_of in week_dates:
            day = by_date.get(as_of)
            if day is None or day.empty:
                continue
            for sig in collect_niches(day, min_conf=min_conf, quality_only=quality_only):
                if best_sig is None or sig.confidence > best_sig.confidence:
                    best_sig, best_date = sig, as_of
        if best_sig is not None and best_date is not None:
            rets: list[float] = []
            for _, row in best_sig.tickers.iterrows():
                t = str(row["代码"])
                fr = fwd_ret.get(t)
                if fr is None:
                    continue
                r = fr.get(best_date, np.nan)
                if not np.isfinite(r):
                    continue
                tr = float(-r - 2 * FEE) if best_sig.side == "short" else float(r - 2 * FEE)
                rets.append(tr)
                picks_log.append({
                    "选股日期": pd.Timestamp(best_date).strftime("%Y-%m-%d"),
                    "子策略": best_sig.name,
                    "方向": "做多" if best_sig.side == "long" else "做空",
                    "代码": t,
                    "涨幅%": row["涨幅%"],
                    "置信度": best_sig.confidence,
                    "次日收益%": tr * 100,
                })
            if rets:
                daily_rets.append(float(np.mean(rets)))
        i += 5

    if not daily_rets:
        return {"error": "无有效交易"}
    rs = pd.Series(daily_rets)
    span = max((cal.iloc[-1] - cal.iloc[50]).days / 365.25, 0.1)
    return {
        "日胜率": float((rs > 0).mean()),
        "交易天数": len(rs),
        "年均交易": len(rs) / span,
        "累计收益率": float((1 + rs).prod() - 1),
        "选股明细": pd.DataFrame(picks_log),
    }

def backtest_combo(
    panel: pd.DataFrame,
    fwd_ret: dict[str, pd.Series],
    *,
    start: str | date,
    end: str | date,
    min_conf: float = 0.0,
    weekly_cap: bool = False,
) -> dict:
    cal = panel["日期"].drop_duplicates().sort_values()
    cal = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    by_date = {d: g for d, g in panel.groupby("日期")}
    daily_rets: list[float] = []
    picks_log: list[dict] = []
    last_trade_i = -999

    for i in range(50, len(cal) - 1):
        if weekly_cap and (i - last_trade_i) < 5:
            continue
        as_of = cal.iloc[i]
        day = by_date.get(as_of)
        if day is None or day.empty:
            continue
        signals = collect_niches(day, min_conf=min_conf)
        if not signals:
            continue
        best = max(signals, key=lambda s: s.confidence)
        rets: list[float] = []
        for _, row in best.tickers.iterrows():
            t = str(row["代码"])
            fr = fwd_ret.get(t)
            if fr is None:
                continue
            r = fr.get(as_of, np.nan)
            if not np.isfinite(r):
                continue
            tr = float(-r - 2 * FEE) if best.side == "short" else float(r - 2 * FEE)
            rets.append(tr)
            picks_log.append({
                "选股日期": pd.Timestamp(as_of).strftime("%Y-%m-%d"),
                "子策略": best.name,
                "方向": "做多" if best.side == "long" else "做空",
                "代码": t,
                "涨幅%": row["涨幅%"],
                "置信度": best.confidence,
                "次日收益%": tr * 100,
            })
        if rets:
            daily_rets.append(float(np.mean(rets)))
            last_trade_i = i

    if not daily_rets:
        return {"error": "无有效交易"}
    rs = pd.Series(daily_rets)
    span = max((cal.iloc[-1] - cal.iloc[50]).days / 365.25, 0.1)
    return {
        "日胜率": float((rs > 0).mean()),
        "交易天数": len(rs),
        "年均交易": len(rs) / span,
        "累计收益率": float((1 + rs).prod() - 1),
        "平均日收益": float(rs.mean() * 100),
        "选股明细": pd.DataFrame(picks_log),
    }


def backtest_single_niche(
    panel: pd.DataFrame, fwd_ret: dict[str, pd.Series], name: str,
    *, start: str, end: str,
) -> dict:
    """单个子策略回测。"""
    cal = panel["日期"].drop_duplicates().sort_values()
    cal = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    by_date = {d: g for d, g in panel.groupby("日期")}
    daily_rets: list[float] = []
    for i in range(50, len(cal) - 1):
        as_of = cal.iloc[i]
        day = by_date.get(as_of)
        if day is None:
            continue
        for sig in collect_niches(day):
            if sig.name != name:
                continue
            rets = []
            for _, row in sig.tickers.iterrows():
                t = str(row["代码"])
                fr = fwd_ret.get(t)
                if fr is None:
                    continue
                r = fr.get(as_of, np.nan)
                if np.isfinite(r):
                    rets.append(float(-r - 2 * FEE) if sig.side == "short" else float(r - 2 * FEE))
            if rets:
                daily_rets.append(float(np.mean(rets)))
            break
    if not daily_rets:
        return {"error": "无交易", "策略": name}
    rs = pd.Series(daily_rets)
    span = max((cal.iloc[-1] - cal.iloc[50]).days / 365.25, 0.1)
    return {"策略": name, "日胜率": float((rs > 0).mean()), "交易天数": len(rs),
            "年均交易": len(rs) / span, "累计收益率": float((1 + rs).prod() - 1)}


def prepare_weekly_panel(data: dict, spy_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    panel = build_factor_panels(data, spy_df["Close"].astype(float))
    fwd = {t: df["Close"].astype(float).pct_change(1).shift(-1) for t, df in data.items()}
    panel = precompute_setup_edge(panel, fwd, edge_setup_filters())
    panel = precompute_short_edge(panel, fwd)
    panel = precompute_dip_edge(panel, fwd)
    return panel, fwd


def today_weekly_signal(panel: pd.DataFrame, *, min_conf: float = 0.55, quality_only: bool = True) -> NicheSignal | None:
    """最近一个交易日：各子策略中置信度最高的一笔。"""
    cal = panel["日期"].drop_duplicates().sort_values()
    if len(cal) < 2:
        return None
    day = panel.loc[panel["日期"] == cal.iloc[-1]]
    if day.empty:
        return None
    signals = collect_niches(day, min_conf=min_conf, quality_only=quality_only)
    return max(signals, key=lambda s: s.confidence) if signals else None


def collect_scheme_comparison(
    data: dict,
    spy_df: pd.DataFrame,
    panel: pd.DataFrame,
    fwd: dict[str, pd.Series],
    *,
    start: str,
    end: str,
    years: float,
) -> list[tuple]:
    schemes: list[tuple] = []

    r0 = backtest_daily_gainer_portfolio(
        data, spy_df, start=start, end=end, filt=ultra_high_win_filters(2),
        panel=panel, fwd_ret=fwd,
    )
    if not r0.get("error"):
        schemes.append(("方案0·极严高胜率 Top2", r0["日胜率"], r0["交易天数"], r0["交易天数"] / max(years, 0.1),
                        r0["累计收益率"], "做多", "细扫最优~82%，信号最稀（年均约6次）"))

    r1 = backtest_daily_gainer_portfolio(
        data, spy_df, start=start, end=end, filt=high_win_filters(2),
        panel=panel, fwd_ret=fwd,
    )
    if not r1.get("error"):
        schemes.append(("方案1·高置信做多 Top2", r1["日胜率"], r1["交易天数"], r1["交易天数"] / max(years, 0.1),
                        r1["累计收益率"], "做多", "日胜率高，信号稀少（约每月0.6次）"))

    r2 = backtest_weekly_best(panel, fwd, start=start, end=end, min_conf=0.55, quality_only=True)
    if not r2.get("error"):
        schemes.append(("方案2·每周高置信做多", r2["日胜率"], r2["交易天数"], r2["年均交易"],
                        r2["累计收益率"], "做多", "每周最多1笔，仅高置信动量"))

    r3 = backtest_weekly_best(panel, fwd, start=start, end=end, min_conf=0.55, quality_only=False)
    if not r3.get("error"):
        schemes.append(("方案3·每周混合多空", r3["日胜率"], r3["交易天数"], r3["年均交易"],
                        r3["累计收益率"], "多空", "每周1笔，含动量做多+超涨做空"))

    r4 = backtest_daily_gainer_portfolio(
        data, spy_df, start=start, end=end, filt=weekly_momentum_filters(2),
        panel=panel, fwd_ret=fwd,
    )
    if not r4.get("error"):
        schemes.append(("方案4·温和动量做多 Top2", r4["日胜率"], r4["交易天数"], r4["交易天数"] / max(years, 0.1),
                        r4["累计收益率"], "做多", "频率最高（约每周1次），胜率约60%"))
    return schemes


def print_scheme_comparison(
    data: dict,
    spy_df: pd.DataFrame,
    panel: pd.DataFrame,
    fwd: dict[str, pd.Series],
    *,
    start: str,
    end: str,
    years: float,
) -> list[tuple]:
    print("\n=== 方案对比（全市场，近{}年回测）===".format(years))
    print("说明：日胜率≥80% 与 每周交易 在同一策略上难以兼得，以下为多种可选方案。\n")

    schemes = collect_scheme_comparison(data, spy_df, panel, fwd, start=start, end=end, years=years)

    print(f"{'方案':<22} {'方向':<4} {'日胜率':>7} {'交易次':>6} {'年均':>5} {'累计':>8}")
    print("-" * 70)
    for name, win, days, tpy, tot, side, note in schemes:
        star = " ★" if win >= 0.80 else ""
        print(f"{name:<22} {side:<4} {win:>6.1%} {days:>6} {tpy:>5.0f} {tot:>+7.1%}{star}")
        print(f"  └ {note}")

    hit80 = [s for s in schemes if s[1] >= 0.80]
    hit_weekly = [s for s in schemes if s[3] >= 35]
    print("\n--- 推荐 ---")
    if hit80:
        print(f"  要日胜率≥80%：选 {hit80[0][0]}（年均约{hit80[0][3]:.0f}次）")
    if hit_weekly:
        best_w = max(hit_weekly, key=lambda x: x[1])
        print(f"  要每周有交易：选 {best_w[0]}（胜率{best_w[1]:.1%}，年均{best_w[3]:.0f}次）")
    print("  折中：方案2 每周高置信做多（胜率~64%，年均~11次，累计仍为正）")
    return schemes


def run_weekly_suite(
    data: dict,
    spy_df: pd.DataFrame,
    *,
    start: str,
    end: str,
    years: float = 2.0,
) -> dict:
    """从 gainer_daily_backtest --mode weekly 调用：方案对比 + 本周信号。"""
    panel, fwd = prepare_weekly_panel(data, spy_df)
    schemes = print_scheme_comparison(data, spy_df, panel, fwd, start=start, end=end, years=years)

    r_best = backtest_weekly_best(panel, fwd, start=start, end=end, min_conf=0.55, quality_only=True)
    picks_df = pd.DataFrame()
    if not r_best.get("error"):
        picks_df = r_best["选股明细"]
        out = ROOT / "research" / "gainer_weekly_picks.csv"
        picks_df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n方案2明细 → {out}")
        if not picks_df.empty:
            print("\n最近 5 笔（方案2）：")
            print(picks_df.tail(5).to_string(index=False))

    sig = today_weekly_signal(panel, quality_only=True)
    print("\n--- 今日/最近交易日 · 每周高置信信号 ---")
    if sig is None:
        print("  暂无满足条件的子策略信号。")
    else:
        print(f"  子策略：{sig.name}  方向：{'做多' if sig.side == 'long' else '做空'}  "
              f"置信度：{sig.confidence:.1%}")
        cols = [c for c in ["代码", "涨幅%", "量比", "近8次胜率", "收盘强度"] if c in sig.tickers.columns]
        print(sig.tickers[cols].to_string(index=False))

    niche_rows: list[dict] = []
    print("\n=== 各子策略独立表现 ===")
    for name in ["高置信动量做多", "强SPY日做多", "顺势回调做多", "超涨回吐做空"]:
        r = backtest_single_niche(panel, fwd, name, start=start, end=end)
        if r.get("error"):
            print(f"  {name}: {r['error']}")
        else:
            print(f"  {name}: 胜率{r['日胜率']:.1%}  {r['交易天数']}天  "
                  f"年均{r['年均交易']:.0f}次  累计{r['累计收益率']:+.1%}")
            niche_rows.append(r)
    return {
        "schemes": schemes,
        "weekly_picks": picks_df,
        "today_signal": sig,
        "niche_stats": niche_rows,
        "weekly_best": r_best if not r_best.get("error") else {},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=2.0)
    args = parser.parse_args()
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=int(args.years * 365) + 120)).isoformat()
    cache = ROOT / "research" / "gainer_universe_cache.json"
    tickers = json.loads(cache.read_text()) if cache.exists() else []
    if not tickers:
        print("请先运行 gainer_daily_backtest.py 生成候选池缓存")
        return

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    print(f"加载 {len(tickers)} 只…")
    data = yahoo.fetch_batch(tickers, start, end)
    spy = yahoo.fetch_history("SPY", start, end)
    panel, fwd = prepare_weekly_panel(data, spy)
    print_scheme_comparison(data, spy, panel, fwd, start=start, end=end, years=args.years)

    r_best = backtest_weekly_best(panel, fwd, start=start, end=end, min_conf=0.55, quality_only=True)
    if not r_best.get("error"):
        out = ROOT / "research" / "gainer_weekly_picks.csv"
        r_best["选股明细"].to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n方案2明细 → {out}")

    print("\n=== 各子策略独立表现 ===")
    for name in ["高置信动量做多", "强SPY日做多", "顺势回调做多", "超涨回吐做空"]:
        r = backtest_single_niche(panel, fwd, name, start=start, end=end)
        if r.get("error"):
            print(f"  {name}: {r['error']}")
        else:
            print(f"  {name}: 胜率{r['日胜率']:.1%}  {r['交易天数']}天  年均{r['年均交易']:.0f}次  累计{r['累计收益率']:+.1%}")


if __name__ == "__main__":
    main()
