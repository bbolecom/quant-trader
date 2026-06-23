"""Meme / 极端涨幅路由：卖 Call 池过滤 + 超涨回吐看跌价差。

规则（每只候选独立判定）：
  · 满足超涨回吐 + 弱市(SPY<MA50) → 买 Put 价差（不进卖 Call）
  · 黑名单 / 涨幅·振幅超阈 → 观望
  · 其余 → 卖 Call 价差
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from quant.vol_decay import TRADING_DAYS, bs_put_price, strike_for_put_delta

RouteAction = Literal["bear_call", "short_fade", "skip"]
DEFAULT_VRP = 0.15

DEFAULT_BLOCKLIST = frozenset({
    "GME", "AMC", "BB", "BBBY", "PLTR", "COIN", "MSTR", "NIO", "BILI",
    "HOOD", "RIVN", "LCID", "SOFI", "SMCI", "RDDT", "DJT", "MARA", "RIOT",
    "TSLA", "CLSK", "HUT", "SPCE", "NKLA", "CVNA", "UPST", "AFRM",
    "DKNG", "RBLX", "IONQ", "QUBT", "BITF", "SOUN", "OPEN",
})


@dataclass
class ShortFadeParams:
    enabled: bool = True
    min_gain_pct: float = 7.0
    max_gain_pct: float = 14.0
    min_vol_ratio: float = 1.5
    max_vol_ratio: float = 6.0
    max_close_strength: float = 0.50
    min_gain_20d_pct: float = 8.0
    min_rs_20d_pct: float = 4.0
    regime_filter: str = "SPY_MA50"  # none | SPY_MA50 | SPY_negative_1d
    structure: str = "put_spread"  # put_spread | stock_short
    put_delta: float = 0.40
    put_width_pct: float = 0.08
    dte_days: int = 5
    hold_days: int = 1
    top_n: int = 1


@dataclass
class MemeRouteConfig:
    enabled: bool = True
    blocklist: frozenset[str] = DEFAULT_BLOCKLIST
    exclude_gain_pct: float = 12.0
    exclude_amp_pct: float = 25.0
    short_fade: ShortFadeParams = ShortFadeParams()


def parse_meme_route(cfg: dict | None) -> MemeRouteConfig:
    raw = (cfg or {}).get("meme_route") or {}
    sf = raw.get("short_fade") or {}
    block = raw.get("blocklist")
    bl = frozenset(str(t).upper() for t in block) if block else DEFAULT_BLOCKLIST
    regime = sf.get("regime_filter")
    if regime is None:
        regime = "SPY_negative_1d" if sf.get("require_spy_negative_1d") else "SPY_MA50"
    return MemeRouteConfig(
        enabled=bool(raw.get("enabled", True)),
        blocklist=bl,
        exclude_gain_pct=float(raw.get("exclude_gain_pct", 12.0)),
        exclude_amp_pct=float(raw.get("exclude_amp_pct", 25.0)),
        short_fade=ShortFadeParams(
            enabled=bool(sf.get("enabled", True)),
            min_gain_pct=float(sf.get("min_gain_pct", 7.0)),
            max_gain_pct=float(sf.get("max_gain_pct", 14.0)),
            min_vol_ratio=float(sf.get("min_vol_ratio", 1.5)),
            max_vol_ratio=float(sf.get("max_vol_ratio", 6.0)),
            max_close_strength=float(sf.get("max_close_strength", 0.50)),
            min_gain_20d_pct=float(sf.get("min_gain_20d_pct", 8.0)),
            min_rs_20d_pct=float(sf.get("min_rs_20d_pct", 4.0)),
            regime_filter=str(regime),
            structure=str(sf.get("structure", "put_spread")),
            put_delta=float(sf.get("put_delta", 0.40)),
            put_width_pct=float(sf.get("put_width_pct", 0.08)),
            dte_days=int(sf.get("dte_days", 5)),
            hold_days=int(sf.get("hold_days", 1)),
            top_n=int(sf.get("top_n", 1)),
        ),
    )


def _num(row: pd.Series, key: str, default: float = np.nan) -> float:
    v = pd.to_numeric(row.get(key), errors="coerce")
    return float(v) if np.isfinite(v) else default


def is_meme_excluded(row: pd.Series, mrc: MemeRouteConfig) -> bool:
    ticker = str(row.get("代码", "")).upper()
    gain = _num(row, "涨幅%", 0.0)
    amp = _num(row, "振幅%", abs(gain))
    if ticker in mrc.blocklist:
        return True
    if gain >= mrc.exclude_gain_pct:
        return True
    if amp >= mrc.exclude_amp_pct:
        return True
    return False


def _regime_ok(
    sf: ShortFadeParams,
    *,
    spy_bear: bool | None,
    spy_1d_pct: float | None,
    row: pd.Series,
) -> bool:
    filt = sf.regime_filter.upper()
    if filt in ("NONE", ""):
        return True
    if filt == "SPY_MA50":
        return spy_bear is True
    if filt == "SPY_NEGATIVE_1D":
        spy1 = spy_1d_pct if spy_1d_pct is not None else _num(row, "SPY1d涨%")
        return np.isfinite(spy1) and spy1 < 0
    return True


def qualifies_short_fade(
    row: pd.Series,
    mrc: MemeRouteConfig,
    *,
    spy_1d_pct: float | None = None,
    spy_bear: bool | None = None,
) -> bool:
    sf = mrc.short_fade
    if not mrc.enabled or not sf.enabled:
        return False
    gain = _num(row, "涨幅%")
    vr = _num(row, "量比")
    cs = _num(row, "收盘强度")
    g20 = _num(row, "涨幅20d%")
    rs = _num(row, "相对SPY20d%")
    if not np.isfinite(gain) or not sf.min_gain_pct <= gain <= sf.max_gain_pct:
        return False
    if not np.isfinite(vr) or not sf.min_vol_ratio <= vr <= sf.max_vol_ratio:
        return False
    if not np.isfinite(cs) or cs > sf.max_close_strength:
        return False
    if np.isfinite(g20) and g20 < sf.min_gain_20d_pct:
        return False
    if np.isfinite(rs) and rs < sf.min_rs_20d_pct:
        return False
    if not _regime_ok(sf, spy_bear=spy_bear, spy_1d_pct=spy_1d_pct, row=row):
        return False
    return True


def route_action(
    row: pd.Series,
    mrc: MemeRouteConfig,
    *,
    spy_1d_pct: float | None = None,
    spy_bear: bool | None = None,
) -> RouteAction:
    if not mrc.enabled:
        return "bear_call"
    if qualifies_short_fade(row, mrc, spy_1d_pct=spy_1d_pct, spy_bear=spy_bear):
        return "short_fade"
    if is_meme_excluded(row, mrc):
        return "skip"
    return "bear_call"


def skip_reason(row: pd.Series, mrc: MemeRouteConfig) -> str:
    ticker = str(row.get("代码", "")).upper()
    gain = _num(row, "涨幅%", 0.0)
    amp = _num(row, "振幅%", abs(gain))
    if ticker in mrc.blocklist:
        return f"meme黑名单 {ticker}"
    if gain >= mrc.exclude_gain_pct:
        return f"涨幅{gain:+.1f}%≥{mrc.exclude_gain_pct}%"
    if amp >= mrc.exclude_amp_pct:
        return f"振幅{amp:.1f}%≥{mrc.exclude_amp_pct}%"
    return "极端波动观望"


def estimate_bear_put_debit_spread(
    spot: float,
    rv_annual: float,
    *,
    delta: float = 0.40,
    width_pct: float = 0.08,
    dte_days: int = 5,
    vrp: float = DEFAULT_VRP,
) -> tuple[float, float, float, float, float]:
    """买 Put 价差：返回 (买Put K高, 卖Put K低, 净成本/股, 最大亏损/股, 最大盈利/股)。"""
    if spot <= 0 or rv_annual <= 0:
        return spot * 0.95, spot * 0.88, 0.0, 0.0, 0.0
    iv = rv_annual * (1 + vrp)
    T = dte_days / TRADING_DAYS
    k_long = strike_for_put_delta(spot, T, iv, target_delta=delta)
    k_short = max(0.01, k_long * (1 - width_pct))
    if k_short >= k_long:
        k_short = k_long * 0.92
    debit = bs_put_price(spot, k_long, T, iv) - bs_put_price(spot, k_short, T, iv)
    width = k_long - k_short
    max_loss = max(debit, 0.01)
    max_profit = max(width - debit, 0.0)
    return round(k_long, 2), round(k_short, 2), round(debit, 2), round(max_loss, 2), round(max_profit, 2)


def put_spread_value(spot: float, k_long: float, k_short: float, T: float, rv_annual: float, *, vrp: float = DEFAULT_VRP) -> float:
    if spot <= 0 or T <= 0 or rv_annual <= 0:
        intrinsic = max(0.0, k_long - spot) - max(0.0, k_short - spot)
        return intrinsic
    iv = rv_annual * (1 + vrp)
    return bs_put_price(spot, k_long, T, iv) - bs_put_price(spot, k_short, T, iv)


def pnl_put_spread_hold(
    spot_entry: float,
    spot_exit: float,
    rv_annual: float,
    *,
    delta: float = 0.40,
    width_pct: float = 0.08,
    dte_days: int = 5,
    hold_days: int = 1,
    vrp: float = DEFAULT_VRP,
) -> tuple[float, float, float, float, float]:
    """持有 hold_days 后 Put 价差盈亏（$/股）及行权价。返回 (pnl, k_long, k_short, debit, pnl_pct_on_risk)。"""
    k_long, k_short, debit, max_loss, _ = estimate_bear_put_debit_spread(
        spot_entry, rv_annual, delta=delta, width_pct=width_pct, dte_days=dte_days, vrp=vrp,
    )
    T0 = dte_days / TRADING_DAYS
    T1 = max((dte_days - hold_days) / TRADING_DAYS, 1 / TRADING_DAYS / 2)
    entry_val = put_spread_value(spot_entry, k_long, k_short, T0, rv_annual, vrp=vrp)
    exit_val = put_spread_value(spot_exit, k_long, k_short, T1, rv_annual, vrp=vrp)
    pnl = exit_val - entry_val
    risk = max(max_loss, debit, 0.01)
    pnl_pct = float(pnl / risk * 100)
    return pnl, k_long, k_short, debit, pnl_pct


def pnl_stock_short_1d(r1: float, fee: float = 5 / 10_000) -> float:
    return float(-r1 - 2 * fee) * 100


def short_fade_module_label(structure: str) -> str:
    return "超涨回吐·Put价差" if structure == "put_spread" else "超涨回吐·做空"


def short_fade_direction(structure: str) -> str:
    return "买Put价差" if structure == "put_spread" else "做空"


def enrich_movers_panel(panel: pd.DataFrame, batch: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """为 live movers 补 量比 / 收盘强度 / 20d 涨幅（供 meme 路由）。"""
    if panel.empty:
        return panel
    out = panel.copy()
    extras: list[dict] = []
    for _, r in out.iterrows():
        t = str(r["代码"])
        df = batch.get(t)
        row = {"代码": t}
        if df is None or len(df) < 25:
            extras.append(row)
            continue
        c = df["Close"].astype(float)
        h = df["High"].astype(float)
        lo = df["Low"].astype(float)
        v = df["Volume"].astype(float)
        hi, lo1 = float(h.iloc[-1]), float(lo.iloc[-1])
        row["收盘强度"] = (float(c.iloc[-1]) - lo1) / (hi - lo1) if hi > lo1 else 0.5
        vma = float(v.iloc[-21:-1].mean())
        row["量比"] = float(v.iloc[-1] / vma) if vma > 0 else 1.0
        row["涨幅20d%"] = (float(c.iloc[-1]) / float(c.iloc[-21]) - 1) * 100 if len(c) >= 21 else np.nan
        extras.append(row)
    ext = pd.DataFrame(extras)
    return out.merge(ext, on="代码", how="left")
