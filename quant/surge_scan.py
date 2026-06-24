"""突破/延续暴涨扫描：区分 A 类突破型与 B 类延续/高潮型。

A 类（突破型，如 SMCI 5/20）：横盘收口 → 放量突破 20 日高，日涨 7%~15%。
B 类（延续型，如 SMCI 6/17）：趋势已强 → 沿 BOLL 上轨加速，WR 超买。
C 类（前兆蓄势）：尚未暴涨，但 BOLL 收口 + 贴近 20 日高，适合提前盯盘。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from quant import indicators as ind
from quant.providers import DataConfig, get_provider, reset_provider_cache
from quant.screener import fetch_gainer_universe_live
from research.gainer_daily_backtest import GAINER_MOMENTUM, LIQUID100, build_factor_panels
from quant.speculative_pool import load_pool_tickers
from research.liquid_tier_a_scan import build_candidate_pool

ROOT = Path(__file__).resolve().parents[1]

SurgeKind = Literal["breakout", "continuation", "precursor"]

SURGE_LABELS: dict[SurgeKind, str] = {
    "breakout": "突破型",
    "continuation": "延续/高潮型",
    "precursor": "前兆蓄势",
}


@dataclass
class SurgeScanConfig:
    """暴涨扫描参数。"""

    breakout_min_gain_pct: float = 7.0
    breakout_max_gain_pct: float = 20.0
    continuation_min_gain_pct: float = 5.0
    continuation_min_gain_20d_pct: float = 30.0
    min_dvol_m: float = 50.0
    min_vol_ratio_breakout: float = 1.5
    min_vol_ratio_5d_breakout: float = 1.28
    min_vol_ratio_continuation: float = 1.1
    early_breakout_max_gain20_pct: float = 25.0
    early_breakout_min_close_strength: float = 0.75
    boll_squeeze_pctile: float = 0.20
    boll_window: int = 20
    wr_overbought: float = -20.0
    gainer_count: int = 250
    use_broad_pool: bool = True
    quick: bool = False
    include_precursors: bool = True


@dataclass
class SurgeHit:
    代码: str
    日期: str
    类型: SurgeKind
    类型名: str
    涨幅_pct: float
    涨幅20d_pct: float
    成交额M: float
    量比: float
    收盘强度: float
    创20日高: bool
    WR: float
    BOLL带宽分位: float
    综合分: float
    说明: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _close_strength(row: pd.Series) -> float:
    hi = float(row["High"])
    lo = float(row["Low"])
    cl = float(row["Close"])
    if hi <= lo:
        return 0.5
    return float((cl - lo) / (hi - lo))


def compute_surge_features(df: pd.DataFrame, *, boll_window: int = 20) -> pd.DataFrame:
    """为每根 K 线计算暴涨识别因子。"""
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["涨幅_pct"] = close.pct_change() * 100.0
    out["涨幅20d_pct"] = close.pct_change(20) * 100.0
    out["成交额USD"] = close * vol
    vol_ma = vol.rolling(20).mean()
    out["量比"] = vol / vol_ma.replace(0, np.nan)
    vol_ma5 = vol.rolling(5).mean()
    out["量比5d"] = vol / vol_ma5.replace(0, np.nan)
    out["high_20d"] = close.rolling(20).max().shift(1)
    out["high_10d"] = close.rolling(10).max().shift(1)
    out["创20日高"] = close >= out["high_20d"].fillna(close)
    out["创10日高"] = close >= out["high_10d"].fillna(close)
    out["收盘强度"] = df.apply(_close_strength, axis=1)

    bb = ind.bollinger_bands(close, boll_window, 2.0)
    out["boll_mid"] = bb["mid"]
    out["boll_upper"] = bb["upper"]
    out["boll_lower"] = bb["lower"]
    width = (bb["upper"] - bb["lower"]) / bb["mid"].replace(0, np.nan)
    out["boll_width"] = width
    out["boll_width_pctile"] = width.rolling(60, min_periods=20).apply(
        lambda s: float((s.iloc[-1] <= s).mean()) if len(s.dropna()) else 0.5,
        raw=False,
    )
    out["boll_squeeze_recent"] = width.rolling(10, min_periods=5).min().rolling(60, min_periods=20).apply(
        lambda s: float((s.iloc[-1] <= s).mean()) if len(s.dropna()) else 0.5,
        raw=False,
    )
    out["WR"] = ind.williams_r(df, 6)
    out["站上MA20"] = close > close.rolling(20).mean()
    out["站上MA50"] = close > close.rolling(50).mean()
    return out


def classify_surge_row(row: pd.Series, cfg: SurgeScanConfig) -> tuple[SurgeKind | None, float, str]:
    """判定单日暴涨类型，返回 (类型, 综合分, 说明)。"""
    gain = float(row.get("涨幅_pct") or 0)
    gain20 = float(row.get("涨幅20d_pct") or 0)
    vol_ratio = float(row.get("量比") or 0)
    vol_ratio_5d = float(row.get("量比5d") or 0)
    dvol_m = float(row.get("成交额USD") or 0) / 1e6
    wr = float(row.get("WR") or -50)
    bw_pct = float(row.get("boll_width_pctile") or 0.5)
    squeeze = float(row.get("boll_squeeze_recent") or bw_pct)
    close = float(row.get("Close") or row.get("close") or 0)
    boll_mid = float(row.get("boll_mid") or 0)
    boll_upper = float(row.get("boll_upper") or 0)
    hi20 = bool(row.get("创20日高", False))
    hi10 = bool(row.get("创10日高", False))
    cs = float(row.get("收盘强度") or 0.5)
    vol_ok = vol_ratio >= cfg.min_vol_ratio_breakout or vol_ratio_5d >= cfg.min_vol_ratio_5d_breakout

    if dvol_m < cfg.min_dvol_m:
        return None, 0.0, ""

    in_breakout_gain = cfg.breakout_min_gain_pct <= gain <= cfg.breakout_max_gain_pct

    # A1 经典突破：创高 + 收口 + 放量
    if (
        in_breakout_gain
        and (hi20 or hi10)
        and vol_ok
        and close > boll_mid > 0
        and squeeze <= cfg.boll_squeeze_pctile
    ):
        score = min(1.0, gain / 15.0 * 0.4 + max(vol_ratio, vol_ratio_5d) / 4.0 * 0.3 + cs * 0.3)
        tag = "创20日高" if hi20 else "创10日高"
        note = f"日涨{gain:.1f}% · 量比{vol_ratio:.1f}/5d{vol_ratio_5d:.1f} · BOLL收口 · {tag}"
        return "breakout", score, note

    # A2 早期突破：尚未创 20 日高，但强势阳线 + 5 日放量（如 SMCI 5/20）
    if (
        in_breakout_gain
        and gain20 < cfg.early_breakout_max_gain20_pct
        and vol_ratio_5d >= cfg.min_vol_ratio_5d_breakout
        and close > boll_mid > 0
        and cs >= cfg.early_breakout_min_close_strength
        and squeeze <= cfg.boll_squeeze_pctile + 0.25
    ):
        score = min(1.0, gain / 12.0 * 0.35 + vol_ratio_5d / 3.0 * 0.35 + cs * 0.3)
        note = f"日涨{gain:.1f}% · 5日量比{vol_ratio_5d:.1f} · 强势阳线 · 20日涨{gain20:.0f}%"
        return "breakout", score, note

    # B 类：延续/高潮型
    if (
        gain >= cfg.continuation_min_gain_pct
        and gain20 >= cfg.continuation_min_gain_20d_pct
        and (vol_ratio >= cfg.min_vol_ratio_continuation or vol_ratio_5d >= 1.2)
        and boll_upper > 0
        and close >= boll_upper * 0.98
        and wr >= cfg.wr_overbought
    ):
        score = min(1.0, gain20 / 60.0 * 0.35 + max(vol_ratio, vol_ratio_5d) / 5.0 * 0.25 + (wr + 100) / 100.0 * 0.2 + cs * 0.2)
        note = f"日涨{gain:.1f}% · 20日涨{gain20:.0f}% · 沿BOLL上轨 · WR={wr:.0f}（超买）"
        return "continuation", score, note

    # C 类：前兆蓄势（当日尚未大涨）
    if cfg.include_precursors and gain < cfg.breakout_min_gain_pct:
        near_hi = hi20 or (
            float(row.get("high_20d") or 0) > 0
            and close >= float(row.get("high_20d")) * 0.97
        )
        if (
            bw_pct <= cfg.boll_squeeze_pctile
            and near_hi
            and vol_ratio <= 1.3
            and gain >= -1.0
        ):
            score = min(1.0, (cfg.boll_squeeze_pctile - bw_pct) / cfg.boll_squeeze_pctile * 0.5 + 0.3)
            note = f"BOLL收口 · 贴近20日高 · 量比{vol_ratio:.1f}（尚未放量）"
            return "precursor", score, note

    return None, 0.0, ""


def scan_ticker_history(
    ticker: str,
    df: pd.DataFrame,
    cfg: SurgeScanConfig | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
) -> list[SurgeHit]:
    """扫描单票历史暴涨点。"""
    cfg = cfg or SurgeScanConfig()
    if df is None or len(df) < 80:
        return []

    feats = compute_surge_features(df, boll_window=cfg.boll_window)
    merged = df.join(feats)
    if start:
        merged = merged.loc[merged.index >= pd.Timestamp(start)]
    if end:
        merged = merged.loc[merged.index <= pd.Timestamp(end)]

    hits: list[SurgeHit] = []
    for ts, row in merged.iterrows():
        kind, score, note = classify_surge_row(row, cfg)
        if kind is None:
            continue
        hits.append(
            SurgeHit(
                代码=ticker.upper(),
                日期=ts.strftime("%Y-%m-%d"),
                类型=kind,
                类型名=SURGE_LABELS[kind],
                涨幅_pct=round(float(row["涨幅_pct"]), 2),
                涨幅20d_pct=round(float(row["涨幅20d_pct"]), 2),
                成交额M=round(float(row["成交额USD"]) / 1e6, 1),
                量比=round(float(row["量比"]), 2),
                收盘强度=round(float(row["收盘强度"]), 2),
                创20日高=bool(row["创20日高"]),
                WR=round(float(row["WR"]), 1),
                BOLL带宽分位=round(float(row["boll_width_pctile"]), 2),
                综合分=round(score, 2),
                说明=note,
            )
        )
    return hits


def scan_ticker_latest(
    ticker: str,
    df: pd.DataFrame,
    cfg: SurgeScanConfig | None = None,
    *,
    as_of: str | None = None,
) -> SurgeHit | None:
    """扫描单票指定日（默认最新）是否命中暴涨。"""
    cfg = cfg or SurgeScanConfig()
    if df is None or len(df) < 80:
        return None
    as_of = as_of or df.index[-1].strftime("%Y-%m-%d")
    hits = scan_ticker_history(ticker, df, cfg, start=as_of, end=as_of)
    return hits[0] if hits else None


def scan_universe(
    data: dict[str, pd.DataFrame],
    cfg: SurgeScanConfig | None = None,
    *,
    as_of: str | None = None,
) -> pd.DataFrame:
    """批量扫描最新一日暴涨/前兆。"""
    cfg = cfg or SurgeScanConfig()
    as_of = as_of or max(
        (df.index[-1].strftime("%Y-%m-%d") for df in data.values() if df is not None and not df.empty),
        default=date.today().isoformat(),
    )
    rows: list[dict[str, Any]] = []
    for ticker, df in data.items():
        if df is None or df.empty:
            continue
        hit = scan_ticker_latest(ticker, df, cfg, as_of=as_of)
        if hit is None:
            continue
        rows.append(hit.to_dict())
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    kind_order = {"突破型": 0, "延续/高潮型": 1, "前兆蓄势": 2}
    out["_sort"] = out["类型名"].map(kind_order).fillna(9)
    return out.sort_values(["_sort", "综合分", "涨幅_pct"], ascending=[True, False, False]).drop(
        columns="_sort"
    ).reset_index(drop=True)


def run_surge_scan(cfg_dict: dict | None = None, *, as_of: str | None = None) -> dict[str, Any]:
    """执行全市场暴涨扫描（当日 + 可选历史）。"""
    cfg_raw = cfg_dict or {}
    cfg = SurgeScanConfig(
        breakout_min_gain_pct=float(cfg_raw.get("breakout_min_gain_pct", 7.0)),
        breakout_max_gain_pct=float(cfg_raw.get("breakout_max_gain_pct", 20.0)),
        continuation_min_gain_pct=float(cfg_raw.get("continuation_min_gain_pct", 5.0)),
        continuation_min_gain_20d_pct=float(cfg_raw.get("continuation_min_gain_20d_pct", 30.0)),
        min_dvol_m=float(cfg_raw.get("min_dvol_m", 50.0)),
        min_vol_ratio_breakout=float(cfg_raw.get("min_vol_ratio_breakout", 1.5)),
        min_vol_ratio_5d_breakout=float(cfg_raw.get("min_vol_ratio_5d_breakout", 1.28)),
        min_vol_ratio_continuation=float(cfg_raw.get("min_vol_ratio_continuation", 1.1)),
        boll_squeeze_pctile=float(cfg_raw.get("boll_squeeze_pctile", 0.20)),
        gainer_count=int(cfg_raw.get("gainer_count", 250)),
        use_broad_pool=bool(cfg_raw.get("use_broad_pool", True)),
        quick=bool(cfg_raw.get("quick", False)),
        include_precursors=bool(cfg_raw.get("include_precursors", True)),
    )
    as_of = as_of or date.today().isoformat()
    history_ticker = cfg_raw.get("history_ticker")
    history_days = int(cfg_raw.get("history_days", 0))

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    start = (date.fromisoformat(as_of) - timedelta(days=max(120, history_days + 30))).isoformat()

    tickers: set[str] = set()
    pool_name = cfg_raw.get("pool", "")
    if pool_name == "surge_drop":
        from quant.surge_drop_pool import load_pool as load_surge_drop_pool
        tickers.update(load_surge_drop_pool())
    elif history_ticker:
        tickers.add(str(history_ticker).upper())
    else:
        snap_live = fetch_gainer_universe_live(count=cfg.gainer_count)
        if not snap_live.empty:
            tickers.update(snap_live["代码"].astype(str).tolist())
        if cfg.use_broad_pool:
            pool = build_candidate_pool(use_broad=not cfg.quick, max_names=80 if cfg.quick else 0)
            tickers.update(pool)
        tickers.update(GAINER_MOMENTUM)
        tickers.update(LIQUID100)
        if cfg_raw.get("use_speculative_pool", True):
            pool_file = cfg_raw.get("speculative_pool") or "research/speculative_pool.json"
            tickers.update(load_pool_tickers(ROOT / pool_file))

    batch = yahoo.fetch_batch(sorted(tickers), start, as_of)
    spy_df = batch.pop("SPY", None)
    if spy_df is None or spy_df.empty:
        spy_df = yahoo.fetch_history("SPY", start, as_of)
    spy_close = spy_df["Close"].astype(float)
    spy_close.index = pd.to_datetime(spy_df.index)

    # 对齐 as_of 到最近交易日
    all_dates = sorted(
        {pd.Timestamp(d) for df in batch.values() if df is not None and not df.empty for d in df.index}
    )
    if all_dates:
        target = pd.Timestamp(as_of)
        valid = [d for d in all_dates if d <= target]
        as_of = (valid[-1] if valid else all_dates[-1]).strftime("%Y-%m-%d")

    history_hits: list[dict[str, Any]] = []
    if history_ticker and history_days > 0:
        tk = str(history_ticker).upper()
        df = batch.get(tk)
        if df is not None and not df.empty:
            hist_start = (date.fromisoformat(as_of) - timedelta(days=history_days)).isoformat()
            for h in scan_ticker_history(tk, df, cfg, start=hist_start, end=as_of):
                history_hits.append(h.to_dict())

    today_df = scan_universe(batch, cfg, as_of=as_of)
    breakout = today_df[today_df["类型"] == "breakout"].to_dict("records") if not today_df.empty else []
    continuation = today_df[today_df["类型"] == "continuation"].to_dict("records") if not today_df.empty else []
    precursor = today_df[today_df["类型"] == "precursor"].to_dict("records") if not today_df.empty else []

    spy_hist = spy_close.loc[spy_close.index <= pd.Timestamp(as_of)]
    spy_ma20 = float(spy_hist.tail(20).mean()) if len(spy_hist) >= 20 else None
    spy_px = float(spy_hist.iloc[-1]) if len(spy_hist) else None

    return {
        "date": as_of,
        "config": asdict(cfg),
        "market": {
            "SPY": spy_px,
            "MA20": spy_ma20,
            "站上MA20": spy_px > spy_ma20 if spy_ma20 else None,
        },
        "scan_stats": {
            "universe": len(batch),
            "breakout": len(breakout),
            "continuation": len(continuation),
            "precursor": len(precursor),
            "total": len(today_df),
        },
        "breakout": breakout,
        "continuation": continuation,
        "precursor": precursor,
        "history": history_hits,
        "all_hits": today_df.to_dict("records") if not today_df.empty else [],
    }
