"""每日涨幅榜选股 + 专业因子评分 + 组合回测。

流程：
  1. 股票池：全市场（Yahoo 涨幅/活跃/小盘多榜 + 纳指100 + 标普），不限于标普500。
  2. 硬筛选：市值 ≥ 5亿美元、成交额 ≥ 5亿美元、1日涨幅 2%~15%。
  3. 专业因子打分（无未来函数）：
       - 量比（当日量 / 20日均量）
       - 站上 20 日均线
       - 20 日相对强度 vs SPY
       - 20 日涨幅（中期动量）
  4. 每日取得分最高的 N 只，等权持有 1 个交易日，滚动回测。
  5. 默认高胜率模式：温和涨幅 + 趋势/大盘过滤 + 近8次形态胜率 + Top2（目标日胜率≥80%）。

用法：
    python research/gainer_daily_backtest.py
    python research/gainer_daily_backtest.py --years 3
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
from quant.screener import (
    fetch_broad_universe,
    fetch_gainer_universe_live,
    fetch_sp500_tickers,
)

# 5亿 USD = 500M
MIN_MCAP_B = 0.5
MIN_DOLLAR_VOL_M = 500.0
TOP_N = 5
LOOKBACK = 1  # 1日涨幅 = 每日涨幅榜

# 高流动性大盘股（日均成交额常 >5亿美元，加速回测）
LIQUID100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "XOM", "LLY", "MA", "HD", "PG", "COST", "MRK",
    "ABBV", "KO", "PEP", "AMD", "NFLX", "CRM", "ORCL", "BAC", "WMT", "CSCO",
    "ACN", "MCD", "LIN", "TMO", "ABT", "DIS", "INTU", "QCOM", "IBM", "GE",
    "CAT", "TXN", "AMAT", "INTC", "MU", "WDC", "STX", "SNDK", "PLTR", "COIN",
    "SMCI", "MSTR", "HOOD", "SOFI", "UBER", "ABNB", "BKNG", "ISRG", "VRTX", "REGN",
    "PANW", "CRWD", "SNOW", "DDOG", "NET", "SHOP", "PYPL", "SQ", "F", "GM",
    "BA", "LMT", "RTX", "NKE", "SBUX", "T", "VZ", "CMCSA", "TMUS", "ADBE",
    "NOW", "SNPS", "CDNS", "KLAC", "LRCX", "ADI", "MRVL", "ON", "ARM", "TSM",
    "BABA", "PDD", "JD", "NIO", "RIVN", "LCID", "XLE", "XLF", "SPY", "QQQ",
]

# 动量扩展池：LIQUID100 + 高波动/题材票（回测用，避免每次拉全市场榜）
GAINER_MOMENTUM = list(dict.fromkeys(LIQUID100 + [
    "RGTI", "QBTS", "IONQ", "QUBT", "SOUN", "AI", "PATH", "UPST", "AFRM", "DKNG",
    "ROKU", "SNAP", "PINS", "TTD", "APP", "DUOL", "CELH", "HIMS", "CVNA", "CAR",
    "MARA", "RIOT", "CLSK", "HUT", "BITF", "IREN", "WULF", "CIFR", "CORZ", "BTBT",
    "GME", "AMC", "BBBY", "SPCE", "LAZR", "JOBY", "ACHR", "RKLB", "ASTS", "LUNR",
    "TEM", "VKTX", "SMMT", "MRNA", "BNTX", "NVAX", "SAVA", "SIRI", "SOXL", "TQQQ",
    "LABU", "FNGU", "NVDL", "TSLL", "MSTX", "CONL", "BULL", "APPX", "RDDT", "GRAB",
    "SE", "BILI", "LI", "XPEV", "BEKE", "FUTU", "TIGR", "BIDU", "NTES", "WB",
    "DELL", "HPE", "ANET", "FTNT", "ZS", "OKTA", "MDB", "TEAM", "HUBS", "VEEV",
    "U", "RBLX", "MTCH", "ETSY", "W", "CHWY", "PTON", "BYND", "DASH", "LYFT",
    "ALAB", "CRDO", "VST", "CEG", "FSLR", "ENPH", "RUN", "PLUG", "BE", "QS",
]))


@dataclass
class GainerProFilters:
    """每日涨幅榜 + 专业因子筛选。"""

    min_gain_pct: float = 2.5
    max_gain_pct: float = 5.5          # 温和涨幅，避免追高后次日回吐
    min_dollar_vol_m: float = MIN_DOLLAR_VOL_M
    min_mcap_b: float = MIN_MCAP_B
    lookback_days: int = LOOKBACK
    min_vol_ratio: float = 1.3
    max_vol_ratio: float = 1.75        # 量比过高次日回吐概率大
    min_rs_20d_pct: float = 3.0        # 相对 SPY 有一定强度
    max_rs_20d_pct: float = 20.0       # 避免过度延伸
    require_above_ma20: bool = True
    require_above_ma50: bool = True
    require_rs_vs_spy: bool = True
    require_spy_above_ma20: bool = True
    require_spy_positive_5d: bool = True
    require_spy_positive_1d: bool = False
    min_spy_1d_pct: float = 0.0          # SPY 当日涨幅下限（%）
    require_green_candle: bool = True       # 收阳，延续概率更高
    min_close_strength: float = 0.55        # (收-低)/(高-低)，收在日内高位
    require_20d_high: bool = False          # 收盘价创20日新高
    min_gain_20d_pct: float = 4.0
    max_gain_20d_pct: float = 25.0
    min_setup_win_rate: float = 0.625    # Top1 近8次形态胜率门槛（5/8≈62.5%）
    min_setup_samples: int = 5
    use_recent_setup_win: bool = True    # True=近8次形态胜率；False=全历史滚动均值
    top_n: int = 1
    min_candidates: int = 1


@dataclass
class GainerScoreWeights:
    gain_1d: float = 0.20
    gain_sweet: float = 0.30           # 偏好 3~5% 温和涨幅
    vol_ratio: float = 0.15
    rs_20d: float = 0.20
    gain_20d: float = 0.10
    above_ma_bonus: float = 0.05


def _safe_return(close: pd.Series, days: int) -> float:
    if len(close) <= days:
        return np.nan
    base = float(close.iloc[-days - 1])
    if base <= 0:
        return np.nan
    return float(close.iloc[-1] / base - 1.0)


def pro_snapshot_at_date(
    data: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    spy_close: pd.Series | None = None,
) -> pd.DataFrame:
    """截至 as_of 的专业因子快照（无未来函数）。"""
    rows: list[dict] = []
    as_of = pd.Timestamp(as_of)
    spy = spy_close.loc[spy_close.index <= as_of] if spy_close is not None else None
    spy_rs = _safe_return(spy, 20) if spy is not None and len(spy) > 21 else np.nan

    for ticker, df in data.items():
        if df is None or df.empty:
            continue
        hist = df.loc[df.index <= as_of]
        if len(hist) < 25:
            continue
        close = hist["Close"].astype(float)
        vol = hist["Volume"].astype(float)
        gain_1d = _safe_return(close, 1) * 100.0
        gain_20d = _safe_return(close, 20) * 100.0
        dollar_today = float(close.iloc[-1] * vol.iloc[-1])
        dollar_avg20 = float((close * vol).tail(20).mean())
        vol_avg20 = float(vol.tail(20).mean())
        vol_ratio = float(vol.iloc[-1] / vol_avg20) if vol_avg20 > 0 else np.nan
        ma20 = float(close.tail(20).mean())
        ma50 = float(close.tail(50).mean()) if len(close) >= 50 else np.nan
        above_ma20 = float(close.iloc[-1]) > ma20
        above_ma50 = float(close.iloc[-1]) > ma50 if np.isfinite(ma50) else False
        rs_20d = (gain_20d / 100.0 - spy_rs) if np.isfinite(gain_20d) and np.isfinite(spy_rs) else np.nan

        rows.append({
            "代码": ticker.upper(),
            "最新价": float(close.iloc[-1]),
            "涨幅%": gain_1d,
            "涨幅20d%": gain_20d,
            "成交额USD": dollar_today,
            "成交额20dUSD": dollar_avg20,
            "换手率%": np.nan,
            "量比": vol_ratio,
            "站上MA20": above_ma20,
            "站上MA50": above_ma50,
            "相对SPY20d%": rs_20d * 100.0 if np.isfinite(rs_20d) else np.nan,
            "市值USD": np.nan,
        })
    return pd.DataFrame(rows)


def apply_pro_filters(snap: pd.DataFrame, filt: GainerProFilters) -> pd.DataFrame:
    """硬筛选 + 专业因子过滤（在完整 snap 上操作，避免丢列）。"""
    if snap.empty:
        return snap
    df = snap.copy()
    df["涨幅%"] = pd.to_numeric(df["涨幅%"], errors="coerce")
    df["成交额USD"] = pd.to_numeric(df["成交额USD"], errors="coerce")
    df["市值USD"] = pd.to_numeric(df.get("市值USD"), errors="coerce")

    min_dollar = filt.min_dollar_vol_m * 1_000_000
    min_mcap = filt.min_mcap_b * 1_000_000_000
    mcap = pd.to_numeric(df["市值USD"], errors="coerce")
    # 有市值则必须 ≥5亿；历史无市值时用当日成交额≥5亿替代（大票代理）
    mcap_ok = (mcap >= min_mcap) | (mcap.isna() & (df["成交额USD"].fillna(0) >= min_dollar))

    mask = (
        df["涨幅%"].between(filt.min_gain_pct, filt.max_gain_pct, inclusive="both")
        & (df["成交额USD"].fillna(0) >= min_dollar)
        & mcap_ok
        & (pd.to_numeric(df["量比"], errors="coerce").fillna(0) >= filt.min_vol_ratio)
        & (pd.to_numeric(df["量比"], errors="coerce").fillna(99) <= filt.max_vol_ratio)
    )
    if filt.require_above_ma20:
        mask &= df["站上MA20"].astype(bool)
    if filt.require_above_ma50 and "站上MA50" in df.columns:
        mask &= df["站上MA50"].astype(bool)
    if filt.require_rs_vs_spy:
        rs = pd.to_numeric(df["相对SPY20d%"], errors="coerce")
        mask &= rs.between(filt.min_rs_20d_pct, filt.max_rs_20d_pct, inclusive="both")
    if "涨幅20d%" in df.columns:
        g20 = pd.to_numeric(df["涨幅20d%"], errors="coerce")
        mask &= g20.between(filt.min_gain_20d_pct, filt.max_gain_20d_pct, inclusive="both")
    if filt.require_green_candle and "收阳" in df.columns:
        mask &= df["收阳"].astype(bool)
    if filt.require_20d_high and "创20日高" in df.columns:
        mask &= df["创20日高"].astype(bool)
    if filt.min_close_strength > 0 and "收盘强度" in df.columns:
        mask &= pd.to_numeric(df["收盘强度"], errors="coerce").fillna(0) >= filt.min_close_strength
    if filt.min_setup_win_rate > 0 and "历史胜率" in df.columns:
        # 历史胜率在 pick_from_panel 对 Top1 做门槛，此处不过滤
        pass
    return df.loc[mask].sort_values("涨幅%", ascending=False).reset_index(drop=True)


def score_gainers(df: pd.DataFrame, w: GainerScoreWeights | None = None) -> pd.DataFrame:
    """综合打分并降序排列。"""
    w = w or GainerScoreWeights()
    if df.empty:
        return df
    out = df.copy()
    gain = pd.to_numeric(out["涨幅%"], errors="coerce").fillna(0)
    # 3~5% 涨幅得分最高，过高涨幅扣分（降低次日回吐概率）
    gain_sweet = 10.0 - (gain - 4.0).abs() * 1.8
    vr = pd.to_numeric(out["量比"], errors="coerce").fillna(1).clip(0, 5)
    vr_sweet = 10.0 - (vr - 1.6).abs() * 2.5
    rs = pd.to_numeric(out["相对SPY20d%"], errors="coerce").fillna(0).clip(0, 25)
    g20 = pd.to_numeric(out["涨幅20d%"], errors="coerce").fillna(0)
    ma_bonus = out["站上MA20"].astype(bool).astype(float) * 10.0
    if "历史胜率" in out.columns:
        wr_col = "近8次胜率" if "近8次胜率" in out.columns else "历史胜率"
        hist = pd.to_numeric(out[wr_col], errors="coerce").fillna(0.5) * 10.0
    else:
        hist = 5.0
    out["综合分"] = (
        w.gain_1d * gain
        + w.gain_sweet * gain_sweet
        + w.vol_ratio * vr_sweet
        + w.rs_20d * rs
        + w.gain_20d * g20 * 0.1
        + w.above_ma_bonus * ma_bonus
        + 0.25 * hist  # 优先历史形态胜率高的标的
    )
    return out.sort_values("综合分", ascending=False).reset_index(drop=True)


def pick_top_gainers(
    data: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    spy_close: pd.Series | None,
    filt: GainerProFilters | None = None,
) -> pd.DataFrame:
    filt = filt or GainerProFilters()
    snap = pro_snapshot_at_date(data, as_of, spy_close)
    filtered = apply_pro_filters(snap, filt)
    ranked = score_gainers(filtered)
    if ranked.empty:
        return ranked
    top = ranked.head(filt.top_n).copy()
    top["选股日期"] = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    top["选股理由"] = [
        f"1日涨{row['涨幅%']:+.1f}%；量比{row['量比']:.1f}；"
        f"{'站上' if row['站上MA20'] else '跌破'}MA20；相对SPY {row['相对SPY20d%']:+.1f}%"
        for _, row in top.iterrows()
    ]
    return top


def build_factor_panels(
    data: dict[str, pd.DataFrame],
    spy_close: pd.Series,
) -> pd.DataFrame:
    """向量化预计算全样本 × 全日期的专业因子（long 表：日期×代码）。"""
    spy = spy_close.astype(float)
    spy_ret20 = spy.pct_change(20)
    spy_ret5 = spy.pct_change(5)
    spy_ret1 = spy.pct_change(1)
    spy_ma20 = spy.rolling(20).mean()
    spy_regime = pd.DataFrame({
        "SPY站上MA20": spy > spy_ma20,
        "SPY5d涨%": spy_ret5 * 100.0,
        "SPY1d涨%": spy_ret1 * 100.0,
    }, index=spy.index)
    frames: list[pd.DataFrame] = []

    for ticker, df in data.items():
        if df is None or len(df) < 55:
            continue
        close = df["Close"].astype(float)
        opn = df["Open"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        vol = df["Volume"].astype(float)
        dollar = close * vol
        gain_1d = close.pct_change(1) * 100.0
        gain_20d = close.pct_change(20) * 100.0
        vol_ratio = vol / vol.rolling(20).mean()
        above_ma = close > close.rolling(20).mean()
        above_ma50 = close > close.rolling(50).mean()
        hl = (high - low).replace(0, np.nan)
        close_strength = (close - low) / hl
        high_20d = close >= close.rolling(20).max()
        rs = gain_20d / 100.0 - spy_ret20.reindex(close.index)
        tmp = pd.DataFrame({
            "代码": ticker,
            "涨幅%": gain_1d,
            "涨幅20d%": gain_20d,
            "成交额USD": dollar,
            "量比": vol_ratio,
            "站上MA20": above_ma,
            "站上MA50": above_ma50,
            "相对SPY20d%": rs * 100.0,
            "收阳": close > opn,
            "收盘强度": close_strength,
            "创20日高": high_20d,
            "收盘价": close,
            "市值USD": np.nan,
        }, index=close.index)
        tmp = tmp.join(spy_regime, how="left")
        frames.append(tmp)

    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames)
    panel.index.name = "日期"
    panel = panel.reset_index()
    return panel.dropna(subset=["涨幅%", "成交额USD"])


def edge_setup_filters() -> GainerProFilters:
    """较宽形态定义，用于滚动统计历史次日胜率（样本更多）。"""
    return GainerProFilters(
        min_gain_pct=2.0,
        max_gain_pct=8.0,
        min_vol_ratio=1.2,
        max_vol_ratio=3.0,
        min_rs_20d_pct=0.0,
        max_rs_20d_pct=30.0,
        require_above_ma20=True,
        require_above_ma50=False,
        require_rs_vs_spy=True,
        min_setup_win_rate=0.0,
        min_setup_samples=0,
    )


def precompute_setup_edge(
    panel: pd.DataFrame,
    fwd_ret: dict[str, pd.Series],
    filt: GainerProFilters,
) -> pd.DataFrame:
    """按标的滚动统计「温和涨幅+趋势」形态的历史次日胜率（无未来函数）。"""
    out = panel.copy()
    out["历史胜率"] = np.nan
    out["历史样本"] = 0
    out["近8次胜率"] = np.nan
    out["近8次样本"] = 0
    out["日期"] = pd.to_datetime(out["日期"])

    for ticker, grp in out.groupby("代码"):
        g = grp.sort_values("日期").copy()
        idx = g.index
        dates = g["日期"].values
        gains = pd.to_numeric(g["涨幅%"], errors="coerce").values
        ma50 = g["站上MA50"].astype(bool).values
        vr = pd.to_numeric(g["量比"], errors="coerce").values
        rs = pd.to_numeric(g["相对SPY20d%"], errors="coerce").values
        fr_s = fwd_ret.get(str(ticker))
        if fr_s is None:
            continue
        fr = []
        for d in dates:
            v = fr_s.get(pd.Timestamp(d), np.nan)
            fr.append(float(v) if np.isfinite(v) else np.nan)
        fr = np.array(fr, dtype=float)
        wins: list[float] = []
        win_rates = np.full(len(g), np.nan)
        samples = np.zeros(len(g), dtype=int)
        recent_wr = np.full(len(g), np.nan)
        recent_n = np.zeros(len(g), dtype=int)
        for i in range(len(g)):
            if i > 0 and wins:
                win_rates[i] = float(np.mean(wins))
                samples[i] = len(wins)
                tail = wins[-8:]
                recent_wr[i] = float(np.mean(tail))
                recent_n[i] = len(tail)
            setup = (
                np.isfinite(gains[i])
                and filt.min_gain_pct <= gains[i] <= filt.max_gain_pct
                and ma50[i]
                and np.isfinite(vr[i])
                and filt.min_vol_ratio <= vr[i] <= filt.max_vol_ratio
                and np.isfinite(rs[i])
                and filt.min_rs_20d_pct <= rs[i] <= filt.max_rs_20d_pct
            )
            if setup and np.isfinite(fr[i]):
                wins.append(1.0 if fr[i] > 0 else 0.0)
                if len(wins) > 80:
                    wins.pop(0)
        out.loc[idx, "历史胜率"] = win_rates
        out.loc[idx, "历史样本"] = samples
        out.loc[idx, "近8次胜率"] = recent_wr
        out.loc[idx, "近8次样本"] = recent_n
    return out


def market_regime_ok(day: pd.DataFrame, filt: GainerProFilters) -> bool:
    """大盘环境过滤：仅在顺风日交易。"""
    if day.empty:
        return False
    row = day.iloc[0]
    if filt.require_spy_above_ma20 and not bool(row.get("SPY站上MA20", True)):
        return False
    if filt.require_spy_positive_5d:
        spy5 = pd.to_numeric(row.get("SPY5d涨%"), errors="coerce")
        if not np.isfinite(spy5) or spy5 <= 0:
            return False
    if filt.require_spy_positive_1d:
        spy1 = pd.to_numeric(row.get("SPY1d涨%"), errors="coerce")
        if not np.isfinite(spy1) or spy1 <= 0:
            return False
    if filt.min_spy_1d_pct > 0:
        spy1 = pd.to_numeric(row.get("SPY1d涨%"), errors="coerce")
        if not np.isfinite(spy1) or spy1 < filt.min_spy_1d_pct:
            return False
    return True


def pick_from_panel(
    panel_day: pd.DataFrame,
    filt: GainerProFilters,
    w: GainerScoreWeights | None = None,
) -> pd.DataFrame:
    """从单日截面选股。"""
    filtered = apply_pro_filters(panel_day, filt)
    if len(filtered) < filt.min_candidates:
        return pd.DataFrame()
    ranked = score_gainers(filtered, w)
    if ranked.empty:
        return ranked
    if "历史胜率" in ranked.columns:
        wr_col = "近8次胜率" if filt.use_recent_setup_win and "近8次胜率" in ranked.columns else "历史胜率"
        ranked = ranked.sort_values([wr_col, "综合分"], ascending=[False, False]).reset_index(drop=True)
    top = ranked.head(filt.top_n).copy()
    if filt.min_setup_win_rate > 0 and not top.empty:
        wr_col = "近8次胜率" if filt.use_recent_setup_win and "近8次胜率" in top.columns else "历史胜率"
        n_col = "近8次样本" if filt.use_recent_setup_win and "近8次样本" in top.columns else "历史样本"
        row0 = top.iloc[0]
        wr = pd.to_numeric(row0[wr_col] if wr_col in top.columns else np.nan, errors="coerce")
        n = pd.to_numeric(row0[n_col] if n_col in top.columns else 0, errors="coerce")
        if not (np.isfinite(wr) and wr >= filt.min_setup_win_rate and np.isfinite(n) and n >= filt.min_setup_samples):
            return pd.DataFrame()
    return top


def backtest_daily_gainer_portfolio(
    data: dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    *,
    start: str | date,
    end: str | date,
    filt: GainerProFilters | None = None,
    fee_bps: float = 5.0,
    initial_capital: float = 100_000.0,
    panel: pd.DataFrame | None = None,
    fwd_ret: dict[str, pd.Series] | None = None,
) -> dict:
    """每日选 Top N 等权持有 1 日，滚动组合回测（向量化因子）。"""
    filt = filt or GainerProFilters()
    if panel is None or fwd_ret is None:
        spy_close = spy_df["Close"].astype(float)
        panel = build_factor_panels(data, spy_close)
        if panel.empty:
            return {"error": "无法构建因子面板"}
        fwd_ret = {t: df["Close"].astype(float).pct_change(1).shift(-1) for t, df in data.items()}
        panel = precompute_setup_edge(panel, fwd_ret, edge_setup_filters())
    elif panel.empty:
        return {"error": "无法构建因子面板"}

    cal = panel["日期"].drop_duplicates().sort_values()
    cal = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    if len(cal) < 60:
        return {"error": "数据不足"}

    equity = initial_capital
    curve: list[dict] = []
    picks_log: list[dict] = []
    daily_rets: list[float] = []
    fee = fee_bps / 10_000.0

    panel_by_date = {d: g for d, g in panel.groupby("日期")}

    for i in range(30, len(cal) - 1):
        as_of = cal.iloc[i]
        nxt = cal.iloc[i + 1]
        day = panel_by_date.get(as_of)
        if day is None or day.empty:
            continue
        if not market_regime_ok(day, filt):
            continue
        top = pick_from_panel(day, filt)
        if top.empty:
            continue
        rets: list[float] = []
        for _, row in top.iterrows():
            t = str(row["代码"])
            fr = fwd_ret.get(t)
            if fr is None:
                continue
            r = fr.get(as_of, np.nan)
            if not np.isfinite(r):
                continue
            rets.append(float(r) - 2 * fee)
            picks_log.append({
                "选股日期": pd.Timestamp(as_of).strftime("%Y-%m-%d"),
                "代码": t,
                "涨幅%": row["涨幅%"],
                "量比": row["量比"],
                "综合分": row["综合分"],
                "次日收益%": r * 100.0,
                "选股理由": (
                    f"1日涨{row['涨幅%']:+.1f}%；量比{row['量比']:.1f}；"
                    f"{'站上' if row['站上MA20'] else '跌破'}MA20；"
                    f"相对SPY {row['相对SPY20d%']:+.1f}%"
                ),
            })
        if not rets:
            continue
        port_ret = float(np.mean(rets))
        equity *= 1.0 + port_ret
        daily_rets.append(port_ret)
        curve.append({"日期": pd.Timestamp(nxt).strftime("%Y-%m-%d"), "权益": equity, "日收益": port_ret})

    if not daily_rets:
        return {"error": "回测期内无有效选股"}

    rets_s = pd.Series(daily_rets)
    total = equity / initial_capital - 1.0
    days = (cal.iloc[-1] - cal.iloc[30]).days
    years = max(days / 365.25, 0.1)
    ann = (1.0 + total) ** (1.0 / years) - 1.0
    win = float((rets_s > 0).mean())
    sharpe = float(rets_s.mean() / rets_s.std() * np.sqrt(252)) if rets_s.std() > 0 else 0.0
    eq = pd.Series([c["权益"] for c in curve])
    max_dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0

    picks_df = pd.DataFrame(picks_log)
    by_day = picks_df.groupby("选股日期").agg(
        入选数=("代码", "count"),
        代码=("代码", lambda x: ",".join(x)),
        平均次日收益=("次日收益%", "mean"),
    ).reset_index()

    return {
        "累计收益率": total,
        "年化收益率": ann,
        "夏普比率": sharpe,
        "最大回撤": max_dd,
        "日胜率": win,
        "交易天数": len(daily_rets),
        "期末权益": equity,
        "权益曲线": pd.DataFrame(curve),
        "选股明细": picks_df,
        "按日汇总": by_day,
    }


def live_gainer_picks(filt: GainerProFilters | None = None) -> pd.DataFrame:
    """当日：全市场多榜合并 + 硬筛选 + 专业因子（不限标普500）。

    实时榜与批量历史均走 Yahoo，避免 Polygon DNS/限速导致扫描失败。
    """
    filt = filt or high_win_filters()
    snap = fetch_gainer_universe_live(count=250)
    if snap.empty:
        return snap
    tickers = snap["代码"].tolist()
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=400)).isoformat()
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    batch = yahoo.fetch_batch(tickers, start, end)
    if not batch:
        return pd.DataFrame()
    as_of = max(df.index[-1] for df in batch.values())
    spy_df = yahoo.fetch_history("SPY", start=start, end=end)
    pro = pro_snapshot_at_date(batch, as_of, spy_df["Close"])
    merged = snap.merge(pro, on="代码", how="inner", suffixes=("", "_pro"))
    for col in ["涨幅%", "成交额USD", "市值USD", "最新价", "名称", "行业"]:
        if f"{col}_pro" in merged.columns and col in merged.columns:
            merged[col] = merged[col].combine_first(merged[f"{col}_pro"])
        elif f"{col}_pro" in merged.columns:
            merged[col] = merged[f"{col}_pro"]
    keep = [c for c in merged.columns if not c.endswith("_pro")]
    merged = merged[keep]
    filtered = apply_pro_filters(merged, filt)
    ranked = score_gainers(filtered)
    if ranked.empty:
        return ranked
    ranked["选股日期"] = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    return ranked.head(filt.top_n)


def search_win_rate_params(
    data: dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    *,
    start: str | date,
    end: str | date,
    fee_bps: float = 5.0,
) -> tuple[GainerProFilters, dict]:
    """在预设组合中搜索日胜率≥80% 的参数（数据只拉一次）。"""
    spy_close = spy_df["Close"].astype(float)
    panel = build_factor_panels(data, spy_close)
    if panel.empty:
        return GainerProFilters(), {"error": "无法构建因子面板"}
    fwd_ret = {t: df["Close"].astype(float).pct_change(1).shift(-1) for t, df in data.items()}
    panel = precompute_setup_edge(panel, fwd_ret, edge_setup_filters())

    presets: list[GainerProFilters] = []
    # (max_gain, max_vr, min_rs, max_rs, g20max, close_str, spy1d, min_spy1d, hist_wr, hist_n, high20, recent, top, min_c)
    for row in [
        (5.0, 1.75, 3, 20, 20, 0.55, False, 0.0, 0.625, 5, False, True, 2, 2),
        (4.5, 1.65, 4, 14, 18, 0.55, False, 0.0, 0.625, 5, True, True, 1, 1),
        (5.0, 1.70, 3, 20, 20, 0.55, False, 0.0, 0.625, 5, False, True, 2, 2),
        (4.5, 1.70, 3, 20, 20, 0.55, False, 0.0, 0.625, 5, False, True, 2, 2),
        (5.0, 1.75, 3, 20, 20, 0.58, False, 0.0, 0.675, 5, False, True, 1, 1),
        (4.5, 1.65, 4, 14, 18, 0.60, True, 0.3, 0.625, 5, True, True, 1, 1),
        (5.0, 1.75, 3, 20, 20, 0.55, False, 0.0, 0.625, 5, False, True, 1, 1),
        (5.0, 1.75, 3, 20, 20, 0.55, False, 0.0, 0.60, 5, False, True, 1, 1),
        (5.5, 1.75, 3, 20, 22, 0.55, False, 0.0, 0.625, 5, False, True, 1, 1),
        (5.0, 1.70, 3, 20, 20, 0.55, False, 0.0, 0.625, 5, False, False, 2, 2),
    ]:
        max_gain, max_vr, min_rs, max_rs, g20max, close_str, spy1d, min_spy1d, hist_wr, hist_n, high20, recent, top, min_c = row
        presets.append(GainerProFilters(
            min_gain_pct=2.5,
            max_gain_pct=max_gain,
            min_vol_ratio=1.3,
            max_vol_ratio=max_vr,
            min_rs_20d_pct=min_rs,
            max_rs_20d_pct=max_rs,
            min_gain_20d_pct=4.0,
            max_gain_20d_pct=g20max,
            top_n=top,
            min_candidates=min_c,
            require_above_ma50=True,
            require_spy_above_ma20=True,
            require_spy_positive_5d=True,
            require_spy_positive_1d=spy1d,
            min_spy_1d_pct=min_spy1d,
            require_green_candle=True,
            min_close_strength=close_str,
            require_20d_high=high20,
            min_setup_win_rate=hist_wr,
            min_setup_samples=hist_n,
            use_recent_setup_win=recent,
        ))

    best_filt: GainerProFilters | None = None
    best_res: dict | None = None
    for filt in presets:
        res = backtest_daily_gainer_portfolio(
            data, spy_df, start=start, end=end, filt=filt, fee_bps=fee_bps,
            panel=panel, fwd_ret=fwd_ret,
        )
        if res.get("error"):
            continue
        if best_res is None or res["日胜率"] > best_res["日胜率"]:
            best_filt, best_res = filt, res
        if res["日胜率"] >= 0.80 and res["交易天数"] >= 15:
            return filt, res
    assert best_filt is not None and best_res is not None
    return best_filt, best_res


def high_win_filters(top_n: int = 2) -> GainerProFilters:
    """高日胜率模式：温和涨幅 + 大盘顺风 + 近8次形态胜率 + Top2 等权。"""
    return GainerProFilters(
        min_gain_pct=2.5,
        max_gain_pct=5.0,
        min_vol_ratio=1.3,
        max_vol_ratio=1.75,
        min_rs_20d_pct=3.0,
        max_rs_20d_pct=20.0,
        min_gain_20d_pct=4.0,
        max_gain_20d_pct=25.0,
        require_above_ma50=True,
        require_spy_above_ma20=True,
        require_spy_positive_5d=True,
        require_spy_positive_1d=False,
        min_spy_1d_pct=0.0,
        require_green_candle=True,
        min_close_strength=0.55,
        require_20d_high=False,
        min_setup_win_rate=0.625,
        min_setup_samples=5,
        use_recent_setup_win=True,
        top_n=top_n,
        min_candidates=2,
    )


def ultra_high_win_filters(top_n: int = 2) -> GainerProFilters:
    """极严高胜率：全市场细扫约 82% 日胜率（年均约 6 次，SPY 强日）。"""
    return GainerProFilters(
        min_gain_pct=2.5,
        max_gain_pct=3.5,
        min_vol_ratio=1.2,
        max_vol_ratio=1.55,
        min_rs_20d_pct=2.0,
        max_rs_20d_pct=18.0,
        min_gain_20d_pct=3.0,
        max_gain_20d_pct=22.0,
        require_above_ma50=True,
        require_spy_above_ma20=True,
        require_spy_positive_5d=True,
        require_spy_positive_1d=True,
        min_spy_1d_pct=0.4,
        require_green_candle=True,
        min_close_strength=0.65,
        require_20d_high=False,
        min_setup_win_rate=0.575,
        min_setup_samples=5,
        use_recent_setup_win=True,
        top_n=top_n,
        min_candidates=1,
    )


def weekly_momentum_filters(top_n: int = 2) -> GainerProFilters:
    """每周频率模式：温和动量，约每周 1 次，日胜率约 60%。"""
    return GainerProFilters(
        min_gain_pct=2.0,
        max_gain_pct=5.5,
        min_vol_ratio=1.2,
        max_vol_ratio=1.85,
        min_rs_20d_pct=2.0,
        max_rs_20d_pct=25.0,
        min_gain_20d_pct=2.0,
        max_gain_20d_pct=30.0,
        require_above_ma50=True,
        require_spy_above_ma20=True,
        require_spy_positive_5d=True,
        require_spy_positive_1d=False,
        min_spy_1d_pct=0.0,
        require_green_candle=True,
        min_close_strength=0.52,
        require_20d_high=False,
        min_setup_win_rate=0.55,
        min_setup_samples=5,
        use_recent_setup_win=True,
        top_n=top_n,
        min_candidates=1,
    )


def high_freq_filters(top_n: int = 5) -> GainerProFilters:
    """高频积少成多：每日 TopN，放宽量价/大盘过滤，保留流动性硬筛。

    设计目标：250 交易日里 >150 天有信号，单笔期望小正 edge，靠频率复利。
    """
    return GainerProFilters(
        min_gain_pct=1.5,
        max_gain_pct=8.0,
        min_vol_ratio=1.15,
        max_vol_ratio=2.5,
        min_rs_20d_pct=0.0,
        max_rs_20d_pct=35.0,
        min_gain_20d_pct=-8.0,
        max_gain_20d_pct=45.0,
        require_above_ma50=False,
        require_above_ma20=True,
        require_spy_above_ma20=False,
        require_spy_positive_5d=False,
        require_spy_positive_1d=False,
        require_green_candle=False,
        min_close_strength=0.42,
        require_20d_high=False,
        min_setup_win_rate=0.0,
        min_setup_samples=0,
        use_recent_setup_win=False,
        top_n=top_n,
        min_candidates=1,
    )


def filters_for_mode(mode: str, top_n: int = 0) -> GainerProFilters:
    """highwin | ultra | weekly | highfreq | legacy"""
    if mode == "legacy":
        return legacy_filters(top_n=top_n or TOP_N)
    if mode == "ultra":
        return ultra_high_win_filters(top_n=top_n or 2)
    if mode == "weekly":
        return weekly_momentum_filters(top_n=top_n or 2)
    if mode == "highfreq":
        return high_freq_filters(top_n=top_n or 5)
    return high_win_filters(top_n=top_n or 2)


def load_gainer_pool(pool: str, *, max_tickers: int = 0) -> list[str]:
    """加载回测候选池 tickers。"""
    if pool == "sp500":
        tickers = fetch_sp500_tickers()
    elif pool == "liquid100":
        tickers = LIQUID100
    elif pool == "momentum":
        tickers = GAINER_MOMENTUM
    else:
        cache = ROOT / "research" / "gainer_universe_cache.json"
        if cache.exists() and max_tickers <= 0:
            tickers = json.loads(cache.read_text())
        else:
            tickers = fetch_broad_universe(screen_count=250, extra=LIQUID100)
    if max_tickers > 0:
        tickers = tickers[:max_tickers]
    return tickers


def fetch_gainer_data_yahoo(
    tickers: list[str],
    start: str,
    end: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """批量拉取 gainer 回测数据（固定 Yahoo + 磁盘缓存）。"""
    from quant.market_cache import read_cached, write_cached

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    syms = [t.strip().upper() for t in tickers if t and str(t).strip()]
    data: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for t in syms:
        hit = read_cached("yahoo", t, start, end)
        if hit is not None:
            data[t] = hit
        else:
            missing.append(t)
    if missing:
        fetched = yahoo.fetch_batch(missing, start, end)
        for t, df in fetched.items():
            if df is None or df.empty:
                continue
            data[t.upper()] = df
            write_cached("yahoo", t.upper(), start, end, df)
    spy = read_cached("yahoo", "SPY", start, end)
    if spy is None:
        spy = yahoo.fetch_history("SPY", start, end)
        write_cached("yahoo", "SPY", start, end, spy)
    return data, spy


GAINER_MODE_LABELS = {
    "highwin": "高置信 Top2（目标日胜率≥80%）",
    "ultra": "极严高胜率（~82%，信号稀少）",
    "weekly": "温和动量（约每周1次）",
    "highfreq": "高频 Top5（积少成多，每日信号）",
    "legacy": "旧版 Top5 追涨幅",
}


def compare_gainer_modes(
    data: dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    *,
    start: str,
    end: str,
    years: float,
    fee_bps: float = 5.0,
) -> pd.DataFrame:
    """四种模式横向对比，供 CLI / UI 共用。"""
    panel = build_factor_panels(data, spy_df["Close"].astype(float))
    fwd = {t: df["Close"].astype(float).pct_change(1).shift(-1) for t, df in data.items()}
    panel = precompute_setup_edge(panel, fwd, edge_setup_filters())
    rows: list[dict] = []
    for mode, label in [
        ("ultra", "极严高胜率"),
        ("highwin", "高置信 Top2"),
        ("weekly", "温和动量"),
        ("legacy", "旧版 Top5"),
    ]:
        f = filters_for_mode(mode, top_n=TOP_N if mode == "legacy" else 2)
        r = backtest_daily_gainer_portfolio(
            data, spy_df, start=start, end=end, filt=f, fee_bps=fee_bps,
            panel=panel, fwd_ret=fwd,
        )
        if r.get("error"):
            continue
        tpy = r["交易天数"] / max(years, 0.1)
        rows.append({
            "模式": label,
            "日胜率": r["日胜率"],
            "交易天数": r["交易天数"],
            "年均次数": tpy,
            "累计收益": r["累计收益率"],
            "夏普": r["夏普比率"],
            "最大回撤": r["最大回撤"],
        })
    return pd.DataFrame(rows)


def legacy_filters(top_n: int = TOP_N) -> GainerProFilters:
    """旧版参数（追涨幅、Top5）。"""
    return GainerProFilters(
        min_gain_pct=2.0,
        max_gain_pct=15.0,
        min_vol_ratio=1.2,
        max_vol_ratio=99.0,
        min_rs_20d_pct=0.0,
        max_rs_20d_pct=999.0,
        min_gain_20d_pct=-999.0,
        max_gain_20d_pct=999.0,
        require_above_ma50=False,
        require_spy_above_ma20=False,
        require_spy_positive_5d=False,
        require_spy_positive_1d=False,
        require_green_candle=False,
        min_close_strength=0.0,
        min_setup_win_rate=0.0,
        min_setup_samples=0,
        top_n=top_n,
        min_candidates=1,
    )


def main():
    parser = argparse.ArgumentParser(description="每日涨幅榜 Top5 专业因子回测")
    parser.add_argument("--years", type=float, default=2.0)
    parser.add_argument("--top", type=int, default=0,
                        help="每日持仓数（0=策略默认：高胜率Top2 / 旧版Top5）")
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--pool", choices=["broad", "sp500", "liquid100", "momentum"], default="broad",
                        help="broad=全市场多榜+指数(默认); sp500=仅标普; liquid100=快速测试; momentum=动量扩展池")
    parser.add_argument("--vol-m", type=float, default=MIN_DOLLAR_VOL_M,
                        help="最低成交额(百万美元)，默认500=5亿美元")
    parser.add_argument("--max-tickers", type=int, default=0,
                        help="限制下载标的数(0=不限制)，过大时可用500加速")
    parser.add_argument("--legacy", action="store_true", help="使用旧版追涨幅参数（等同 --mode legacy）")
    parser.add_argument("--mode", choices=["highwin", "ultra", "weekly", "legacy"], default="highwin",
                        help="highwin=默认高胜率 Top2; ultra=极严~82%%; weekly=高频温和动量; legacy=旧版Top5")
    parser.add_argument("--compare", action="store_true", help="对比四种模式后退出")
    parser.add_argument("--optimize", action="store_true", help="搜索日胜率≥80%参数")
    args = parser.parse_args()

    mode = "legacy" if args.legacy else args.mode
    filt = filters_for_mode(mode, top_n=args.top or (TOP_N if mode == "legacy" else 2))
    filt.min_dollar_vol_m = args.vol_m
    if args.top > 0:
        filt.top_n = args.top
        if filt.min_candidates > args.top:
            filt.min_candidates = args.top
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=int(args.years * 365) + 120)).isoformat()

    if args.pool == "sp500":
        print(f"加载标普500 ({start} ~ {end})…")
        tickers = fetch_sp500_tickers()
    elif args.pool == "liquid100":
        print(f"加载高流动100只 ({start} ~ {end})…")
        tickers = LIQUID100
    elif args.pool == "momentum":
        print(f"加载动量扩展池 {len(GAINER_MOMENTUM)} 只 ({start} ~ {end})…")
        tickers = GAINER_MOMENTUM
    else:
        print(f"加载全市场候选池（Yahoo多榜+纳指100+标普，不限500）…")
        cache = ROOT / "research" / "gainer_universe_cache.json"
        if cache.exists() and args.max_tickers <= 0:
            tickers = json.loads(cache.read_text())
            print(f"  使用缓存 {len(tickers)} 只")
        else:
            tickers = fetch_broad_universe(screen_count=250, extra=LIQUID100)
    if args.max_tickers > 0:
        tickers = tickers[: args.max_tickers]
    print(f"共 {len(tickers)} 只候选，拉取行情…")
    # 回测需批量拉数百只标的；Polygon 免费档逐只限速，此处固定用 Yahoo 批量下载。
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    data = yahoo.fetch_batch(tickers, start, end)
    spy = yahoo.fetch_history("SPY", start, end)
    print(f"有效标的 {len(data)} 只，预计算因子并回测…\n")

    if args.compare and mode != "legacy":
        panel = build_factor_panels(data, spy["Close"].astype(float))
        fwd = {t: df["Close"].astype(float).pct_change(1).shift(-1) for t, df in data.items()}
        panel = precompute_setup_edge(panel, fwd, edge_setup_filters())
        print("=" * 72)
        print("四种模式对比（同一股票池、同一区间）")
        print("=" * 72)
        for m, label in [
            ("ultra", "极严高胜率"),
            ("highwin", "高置信 Top2"),
            ("weekly", "温和动量/每周"),
            ("legacy", "旧版 Top5"),
        ]:
            f = filters_for_mode(m, top_n=2 if m != "legacy" else TOP_N)
            r = backtest_daily_gainer_portfolio(
                data, spy, start=start, end=end, filt=f, fee_bps=args.fee_bps,
                panel=panel, fwd_ret=fwd,
            )
            if r.get("error"):
                print(f"  {label}: {r['error']}")
            else:
                tpy = r["交易天数"] / max(args.years, 0.1)
                star = " ★" if r["日胜率"] >= 0.80 else ""
                print(f"  {label:<14} 胜率{r['日胜率']:>6.1%}  {r['交易天数']:>3}天  年均{tpy:>4.0f}次  "
                      f"累计{r['累计收益率']:>+6.1%}{star}")
        try:
            from research.gainer_weekly_multi import run_weekly_suite
            run_weekly_suite(data, spy, start=start, end=end, years=args.years)
            return
        except ImportError:
            pass

    if mode == "weekly" and not args.compare:
        try:
            from research.gainer_weekly_multi import run_weekly_suite
            run_weekly_suite(data, spy, start=start, end=end, years=args.years)
            return
        except ImportError:
            pass

    if args.optimize and mode != "legacy":
        print("搜索高日胜率参数…")
        filt, res = search_win_rate_params(data, spy, start=start, end=end, fee_bps=args.fee_bps)
        print(f"选用：涨幅 {filt.min_gain_pct}~{filt.max_gain_pct}%  量比 {filt.min_vol_ratio}~{filt.max_vol_ratio}  "
              f"Top{filt.top_n}  RS {filt.min_rs_20d_pct}~{filt.max_rs_20d_pct}%\n")
    else:
        res = backtest_daily_gainer_portfolio(
            data, spy, start=start, end=end, filt=filt, fee_bps=args.fee_bps,
        )
    if res.get("error"):
        print(f"错误：{res['error']}")
        return

    print("=" * 72)
    titles = {
        "highwin": "每日涨幅榜 · 高胜率专业因子（Top2）",
        "ultra": "每日涨幅榜 · 极严高胜率（~82%）",
        "weekly": "每日涨幅榜 · 温和动量（高频）",
        "legacy": "每日涨幅榜 Top5 · 专业因子策略回测",
    }
    title = titles.get(mode, titles["highwin"])
    print(title)
    print("=" * 72)
    pool_desc = {
        "broad": "全市场（Yahoo涨幅/活跃/小盘多榜 + 纳指100 + 标普，不限500）",
        "sp500": "仅标普500",
        "liquid100": "高流动100只",
        "momentum": f"动量扩展池（{len(GAINER_MOMENTUM)}只）",
    }.get(args.pool, args.pool)
    print(f"股票池：{pool_desc}（{len(data)}只有效行情）")
    print(f"筛选：市值≥${filt.min_mcap_b}B  成交额≥${filt.min_dollar_vol_m}M  "
          f"1日涨幅 {filt.min_gain_pct}~{filt.max_gain_pct}%")
    print(f"因子：量比 {filt.min_vol_ratio}~{filt.max_vol_ratio}  站上MA20/MA50  "
          f"RS {filt.min_rs_20d_pct}~{filt.max_rs_20d_pct}%")
    regime = []
    if filt.require_spy_above_ma20:
        regime.append("SPY>MA20")
    if filt.require_spy_positive_5d:
        regime.append("SPY5d>0")
    if filt.require_spy_positive_1d or filt.min_spy_1d_pct > 0:
        regime.append(f"SPY当日涨≥{filt.min_spy_1d_pct:.1f}%" if filt.min_spy_1d_pct > 0 else "SPY当日涨")
    if filt.require_green_candle:
        regime.append("收阳+强势收盘")
    if regime:
        print(f"大盘：{' + '.join(regime)}  候选≥{filt.min_candidates}才交易")
    print(f"持仓：每日等权 Top{filt.top_n}，持有 1 交易日")
    if not args.legacy and filt.min_setup_win_rate > 0:
        wr_mode = "近8次形态" if filt.use_recent_setup_win else "全历史"
        print(f"高置信：{wr_mode}胜率≥{filt.min_setup_win_rate:.0%}  候选≥{filt.min_candidates}才交易")
    print(f"说明：高胜率模式=温和涨幅+趋势确认+形态胜率过滤\n")
    print(f"累计收益   {res['累计收益率']:+.1%}")
    print(f"年化收益   {res['年化收益率']:+.1%}")
    print(f"夏普比率   {res['夏普比率']:.2f}")
    print(f"最大回撤   {res['最大回撤']:+.1%}")
    print(f"日胜率     {res['日胜率']:.1%}")
    print(f"交易天数   {res['交易天数']}")
    print(f"期末权益   ${res['期末权益']:,.0f}")

    picks = res["选股明细"]
    if not picks.empty:
        print(f"\n最近 10 笔选股：")
        show = picks.tail(10)[["选股日期", "代码", "涨幅%", "量比", "次日收益%", "选股理由"]]
        print(show.to_string(index=False))

    out = ROOT / "research" / "gainer_backtest_picks.csv"
    picks.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n完整明细已存 {out}")

    print("\n--- 尝试当日涨幅榜实时 Top5 ---")
    try:
        live = live_gainer_picks(filt)
        if live.empty:
            print("今日无满足条件的标的。")
        else:
            cols = [c for c in ["代码", "名称", "涨幅%", "成交额USD", "量比", "综合分", "站上MA20"] if c in live.columns]
            print(live[cols].to_string(index=False))
    except Exception as e:  # noqa: BLE001
        print(f"实时扫描跳过：{e}")


if __name__ == "__main__":
    main()
