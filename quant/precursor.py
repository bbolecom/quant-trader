"""异动前兆扫描：在大涨/大跌前捕捉可量化迹象。

思路：价格很少「无缘无故」波动；量能、波动收缩、趋势萌芽、相对强弱等
往往在方向性行情之前出现。本模块对单只或多只标的扫描这些前兆并打分排序。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from . import indicators as ind

Direction = Literal["bull", "bear", "neutral"]


@dataclass(frozen=True)
class PrecursorDef:
    """前兆信号定义。"""

    id: str
    name: str
    direction: Direction
    description: str
    weight: float = 1.0


PRECURSOR_CATALOG: dict[str, PrecursorDef] = {
    "volume_surge": PrecursorDef(
        "volume_surge", "量能异动", "bull",
        "成交量显著高于近期均量，可能有资金提前介入。", 1.2,
    ),
    "volume_dryup": PrecursorDef(
        "volume_dryup", "缩量整理", "neutral",
        "波动收窄且量能萎缩，常见于突破前的「蓄势」。", 0.8,
    ),
    "vol_squeeze": PrecursorDef(
        "vol_squeeze", "波动收缩", "neutral",
        "布林带宽度处于近期低位，方向选择临近。", 1.0,
    ),
    "breakout_setup": PrecursorDef(
        "breakout_setup", "突破蓄势", "bull",
        "价格贴近近期高点且量能配合，具备向上突破形态。", 1.3,
    ),
    "breakdown_setup": PrecursorDef(
        "breakdown_setup", "破位预警", "bear",
        "价格贴近近期低点且放量，具备向下破位风险。", 1.3,
    ),
    "adx_awakening": PrecursorDef(
        "adx_awakening", "趋势萌芽", "bull",
        "ADX 从低位回升，趋势强度正在建立。", 1.0,
    ),
    "adx_fading": PrecursorDef(
        "adx_fading", "趋势衰竭", "bear",
        "ADX 从高位回落，原有趋势可能接近尾声。", 0.9,
    ),
    "rsi_divergence_bull": PrecursorDef(
        "rsi_divergence_bull", "RSI 底背离", "bull",
        "价格创新低但 RSI 抬高，下跌动能减弱。", 1.1,
    ),
    "rsi_divergence_bear": PrecursorDef(
        "rsi_divergence_bear", "RSI 顶背离", "bear",
        "价格创新高但 RSI 走低，上涨动能减弱。", 1.1,
    ),
    "ma_golden_near": PrecursorDef(
        "ma_golden_near", "均线金叉临近", "bull",
        "短期均线逼近长期均线且向上，经典趋势转强信号。", 1.0,
    ),
    "ma_death_near": PrecursorDef(
        "ma_death_near", "均线死叉临近", "bear",
        "短期均线逼近长期均线且向下，趋势转弱信号。", 1.0,
    ),
    "rel_strength": PrecursorDef(
        "rel_strength", "相对强势", "bull",
        "近阶段跑赢基准（如 SPY），资金偏好明显。", 0.9,
    ),
    "rel_weakness": PrecursorDef(
        "rel_weakness", "相对弱势", "bear",
        "近阶段跑输基准，资金流出迹象。", 0.9,
    ),
    "macd_turn_up": PrecursorDef(
        "macd_turn_up", "MACD 转强", "bull",
        "MACD 柱体由负转正或加速上行。", 0.8,
    ),
    "macd_turn_down": PrecursorDef(
        "macd_turn_down", "MACD 转弱", "bear",
        "MACD 柱体由正转负或加速下行。", 0.8,
    ),
}


@dataclass
class PrecursorHit:
    """单条触发的信号。"""

    signal_id: str
    name: str
    direction: Direction
    description: str
    strength: float  # 0~1


@dataclass
class ScanResult:
    ticker: str
    score: float
    bull_score: float
    bear_score: float
    bias: Direction
    hits: list[PrecursorHit] = field(default_factory=list)
    latest_close: float = 0.0
    gain_5d: float = 0.0
    gain_20d: float = 0.0
    volume_ratio: float = 0.0


def _safe_last(series: pd.Series, default=np.nan):
    if series is None or series.empty:
        return default
    v = series.iloc[-1]
    return float(v) if pd.notna(v) else default


def _pct_change(close: pd.Series, days: int) -> float:
    if len(close) <= days:
        return 0.0
    a, b = float(close.iloc[-1]), float(close.iloc[-days - 1])
    if b == 0:
        return 0.0
    return (a / b - 1.0) * 100.0


def scan_precursors(
    df: pd.DataFrame,
    benchmark: pd.Series | None = None,
    *,
    min_bars: int = 80,
) -> list[PrecursorHit]:
    """扫描单只标的的前兆信号。"""
    if df is None or len(df) < min_bars:
        return []

    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    hits: list[PrecursorHit] = []

    vol_ma = vol.rolling(20).mean()
    vol_ratio = float(vol.iloc[-1] / vol_ma.iloc[-1]) if vol_ma.iloc[-1] > 0 else 1.0
    if vol_ratio >= 2.0:
        hits.append(_hit("volume_surge", min(1.0, (vol_ratio - 2.0) / 2.0 + 0.5)))
    elif vol_ratio <= 0.6 and close.pct_change().tail(10).std() < close.pct_change().tail(60).std() * 0.7:
        hits.append(_hit("volume_dryup", 0.6))

    bb = ind.bollinger_bands(close, 20, 2.0)
    width = (bb["upper"] - bb["lower"]) / bb["mid"].replace(0, np.nan)
    w = width.dropna()
    if len(w) >= 60:
        pct_rank = float((w.iloc[-1] <= w.tail(60)).mean())
        if pct_rank <= 0.15:
            hits.append(_hit("vol_squeeze", 1.0 - pct_rank))

    hi20 = close.tail(21).iloc[:-1].max() if len(close) > 21 else close.max()
    lo20 = close.tail(21).iloc[:-1].min() if len(close) > 21 else close.min()
    last = float(close.iloc[-1])
    if hi20 > 0 and last >= hi20 * 0.97 and vol_ratio >= 1.2:
        hits.append(_hit("breakout_setup", min(1.0, (last / hi20 - 0.97) / 0.03 + 0.5)))
    if lo20 > 0 and last <= lo20 * 1.03 and vol_ratio >= 1.2:
        hits.append(_hit("breakdown_setup", min(1.0, (1.03 - last / lo20) / 0.03 + 0.5)))

    adx_s = ind.adx(df, 14)
    if len(adx_s.dropna()) >= 10:
        adx_now = float(adx_s.iloc[-1])
        adx_5 = float(adx_s.iloc[-6])
        if adx_5 < 20 and adx_now >= 22 and adx_now > adx_5:
            hits.append(_hit("adx_awakening", min(1.0, (adx_now - 20) / 15)))
        if adx_5 > 35 and adx_now < adx_5 - 5:
            hits.append(_hit("adx_fading", min(1.0, (adx_5 - adx_now) / 20)))

    rsi_s = ind.rsi(close, 14)
    if len(close) >= 30:
        c10 = close.tail(10)
        r10 = rsi_s.tail(10)
        if c10.iloc[-1] < c10.iloc[0] and r10.iloc[-1] > r10.iloc[0] + 3:
            hits.append(_hit("rsi_divergence_bull", 0.7))
        if c10.iloc[-1] > c10.iloc[0] and r10.iloc[-1] < r10.iloc[0] - 3:
            hits.append(_hit("rsi_divergence_bear", 0.7))

    fast = ind.sma(close, 20)
    slow = ind.sma(close, 60)
    if pd.notna(fast.iloc[-1]) and pd.notna(slow.iloc[-1]) and slow.iloc[-1] > 0:
        gap = (fast.iloc[-1] - slow.iloc[-1]) / slow.iloc[-1]
        if -0.02 <= gap <= 0.005 and fast.iloc[-1] > fast.iloc[-5]:
            hits.append(_hit("ma_golden_near", 0.75))
        if -0.005 <= gap <= 0.02 and fast.iloc[-1] < fast.iloc[-5]:
            hits.append(_hit("ma_death_near", 0.75))

    if benchmark is not None and len(benchmark) >= 25:
        b = benchmark.reindex(close.index).ffill()
        rs5 = _pct_change(close, 5) - _pct_change(b, 5)
        rs20 = _pct_change(close, 20) - _pct_change(b, 20)
        if rs20 > 3:
            hits.append(_hit("rel_strength", min(1.0, rs20 / 15)))
        elif rs20 < -3:
            hits.append(_hit("rel_weakness", min(1.0, abs(rs20) / 15)))

    macd_df = ind.macd(close)
    hist = macd_df["hist"]
    if len(hist.dropna()) >= 5:
        h0, h1 = float(hist.iloc[-2]), float(hist.iloc[-1])
        if h0 <= 0 < h1:
            hits.append(_hit("macd_turn_up", 0.7))
        if h0 >= 0 > h1:
            hits.append(_hit("macd_turn_down", 0.7))

    return hits


def _hit(signal_id: str, strength: float) -> PrecursorHit:
    d = PRECURSOR_CATALOG[signal_id]
    return PrecursorHit(
        signal_id=signal_id,
        name=d.name,
        direction=d.direction,
        description=d.description,
        strength=float(np.clip(strength, 0.0, 1.0)),
    )


def score_hits(hits: list[PrecursorHit]) -> tuple[float, float, float, Direction]:
    """返回 (总分, 多头分, 空头分, 偏向)。"""
    bull = bear = 0.0
    for h in hits:
        w = PRECURSOR_CATALOG[h.signal_id].weight * h.strength
        if h.direction == "bull":
            bull += w
        elif h.direction == "bear":
            bear += w
    total = bull + bear * 0.5  # 空头信号也计入总分，但权重略低
    if bull > bear * 1.2:
        bias: Direction = "bull"
    elif bear > bull * 1.2:
        bias = "bear"
    else:
        bias = "neutral"
    return total, bull, bear, bias


def scan_ticker(
    ticker: str,
    df: pd.DataFrame,
    benchmark: pd.Series | None = None,
) -> ScanResult:
    """扫描单只并汇总。"""
    hits = scan_precursors(df, benchmark)
    total, bull, bear, bias = score_hits(hits)
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    vol_ma = vol.rolling(20).mean()
    vol_ratio = float(vol.iloc[-1] / vol_ma.iloc[-1]) if len(vol_ma.dropna()) and vol_ma.iloc[-1] > 0 else 1.0
    return ScanResult(
        ticker=ticker.upper(),
        score=total,
        bull_score=bull,
        bear_score=bear,
        bias=bias,
        hits=hits,
        latest_close=_safe_last(close, 0.0),
        gain_5d=_pct_change(close, 5),
        gain_20d=_pct_change(close, 20),
        volume_ratio=vol_ratio,
    )


def scan_universe(
    data: dict[str, pd.DataFrame],
    benchmark: pd.Series | None = None,
    *,
    min_score: float = 0.5,
) -> pd.DataFrame:
    """批量扫描，返回按总分排序的 DataFrame。"""
    rows: list[dict] = []
    for ticker, df in data.items():
        if df is None or df.empty:
            continue
        r = scan_ticker(ticker, df, benchmark)
        if r.score < min_score and not r.hits:
            continue
        sig_names = "、".join(h.name for h in r.hits[:5])
        if len(r.hits) > 5:
            sig_names += f" 等{len(r.hits)}项"
        bias_cn = {"bull": "偏多", "bear": "偏空", "neutral": "中性"}[r.bias]
        rows.append({
            "代码": r.ticker,
            "前兆得分": round(r.score, 2),
            "多头分": round(r.bull_score, 2),
            "空头分": round(r.bear_score, 2),
            "偏向": bias_cn,
            "触发信号": sig_names,
            "最新价": r.latest_close,
            "近5日%": round(r.gain_5d, 2),
            "近20日%": round(r.gain_20d, 2),
            "量比": round(r.volume_ratio, 2),
            "_hits": r.hits,
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["前兆得分", "多头分"], ascending=False).reset_index(drop=True)


def list_catalog() -> pd.DataFrame:
    """返回全部前兆信号说明表（供 UI 展示）。"""
    rows = [
        {
            "信号": d.name,
            "方向": {"bull": "看涨前兆", "bear": "看跌前兆", "neutral": "蓄势/待变"}[d.direction],
            "说明": d.description,
        }
        for d in PRECURSOR_CATALOG.values()
    ]
    return pd.DataFrame(rows)
