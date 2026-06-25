"""全市场快速扫描 · 5 分钟内完成机会信号探测。

Phase-1（~10–20s）：并行拉 Yahoo 多榜快照（价/涨跌幅/成交额/市值），无需逐只历史。
Phase-2（可选，~30–90s）：对 Top 候选并行拉 30 日 K 线算 RV/振幅，仅 enrich 信号行。

设计目标：典型 < 120s，硬上限 budget_sec=300（5 分钟）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.screener import UNIVERSE_PRESETS, fetch_yahoo_screens_parallel

ROOT = Path(__file__).resolve().parents[1]

# 全市场扫描默认覆盖的 Yahoo 榜（每榜最多 screen_count 只）
DEFAULT_SCREEN_PRESETS: tuple[str, ...] = (
    "day_gainers",
    "day_losers",
    "most_actives",
    "small_cap_gainers",
    "aggressive_small_caps",
    "growth_technology_stocks",
)


@dataclass
class ScanConfig:
    screen_count: int = 250
    screen_presets: tuple[str, ...] = DEFAULT_SCREEN_PRESETS
    min_price: float = 2.0
    min_dvol_m: float = 5.0
    budget_sec: float = 300.0
    enrich_rv: bool = True
    enrich_top_n: int = 120
    rv_lookback_days: int = 35
    rules: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "gainer10": {"min_gain_pct": 10.0, "min_dvol_m": 100.0},
        "gainer5": {"min_gain_pct": 5.0, "min_dvol_m": 50.0},
        "extreme_up": {"min_gain_pct": 15.0, "min_dvol_m": 30.0},
        "extreme_down": {"max_gain_pct": -15.0, "min_dvol_m": 30.0},
        "high_active": {"min_dvol_m": 200.0},
    })


def _dvol_m(row: pd.Series) -> float:
    raw = row.get("成交额USD")
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return 0.0
    return float(raw) / 1e6


def _gain_pct(row: pd.Series) -> float:
    try:
        return float(row.get("涨幅%") or 0)
    except (TypeError, ValueError):
        return 0.0


def _price(row: pd.Series) -> float:
    try:
        return float(row.get("最新价") or 0)
    except (TypeError, ValueError):
        return 0.0


def _base_filter(df: pd.DataFrame, cfg: ScanConfig) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_dvol_m"] = out.apply(_dvol_m, axis=1)
    out["_gain"] = out.apply(_gain_pct, axis=1)
    out["_price"] = out.apply(_price, axis=1)
    mask = (
        (out["_price"] >= cfg.min_price)
        & (out["_dvol_m"] >= cfg.min_dvol_m)
        & out["_gain"].notna()
    )
    return out[mask].reset_index(drop=True)


def _match_rules(row: pd.Series, cfg: ScanConfig) -> list[str]:
    hits: list[str] = []
    g = float(row.get("_gain", _gain_pct(row)))
    dv = float(row.get("_dvol_m", _dvol_m(row)))
    rules = cfg.rules or {}
    g10 = rules.get("gainer10") or {}
    if g >= float(g10.get("min_gain_pct", 10)) and dv >= float(g10.get("min_dvol_m", 100)):
        hits.append("Gainer10+")
    g5 = rules.get("gainer5") or {}
    if g >= float(g5.get("min_gain_pct", 5)) and dv >= float(g5.get("min_dvol_m", 50)):
        hits.append("涨幅5%+")
    eu = rules.get("extreme_up") or {}
    if g >= float(eu.get("min_gain_pct", 15)) and dv >= float(eu.get("min_dvol_m", 30)):
        hits.append("极端上涨")
    ed = rules.get("extreme_down") or {}
    if g <= float(ed.get("max_gain_pct", -15)) and dv >= float(ed.get("min_dvol_m", 30)):
        hits.append("极端下跌")
    ha = rules.get("high_active") or {}
    src = str(row.get("_来源") or "")
    if dv >= float(ha.get("min_dvol_m", 200)) and "活跃" in src:
        hits.append("成交活跃")
    if not hits and g >= 3.0 and dv >= 20.0:
        hits.append("动量观察")
    return hits


def _enrich_rv_batch(
    tickers: list[str],
    *,
    lookback_days: int = 35,
    max_workers: int = 8,
) -> dict[str, dict[str, float]]:
    """并行拉短历史，补 RV%/振幅%。"""
    if not tickers:
        return {}
    from quant.providers import DataConfig, get_provider

    yahoo = get_provider(DataConfig(provider="yahoo"))
    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    end = date.today().isoformat()
    batch = yahoo.fetch_batch(tickers, start, end, max_workers=max_workers)
    out: dict[str, dict[str, float]] = {}
    for t, df in batch.items():
        if df is None or df.empty or len(df) < 5:
            continue
        c = df["Close"].astype(float)
        h = df["High"].astype(float)
        lo = df["Low"].astype(float)
        prev = float(c.iloc[-2]) if len(c) >= 2 else float(c.iloc[-1])
        spot = float(c.iloc[-1])
        amp = float((h.iloc[-1] - lo.iloc[-1]) / prev * 100) if prev > 0 else 0.0
        rv = float(c.pct_change().rolling(min(20, len(c) - 1)).std().iloc[-1] * sqrt(252) * 100)
        if not np.isfinite(rv):
            rv = 0.0
        out[t.upper()] = {"现价": spot, "振幅%": amp, "RV%": rv}
    return out


def _row_to_signal(row: pd.Series, tags: list[str], rv_extra: dict[str, float] | None) -> dict[str, Any]:
    sym = str(row.get("代码", "")).upper()
    gain = float(row.get("_gain", _gain_pct(row)))
    dvol = float(row.get("_dvol_m", _dvol_m(row)))
    price = float((rv_extra or {}).get("现价") or row.get("_price") or _price(row))
    amp = (rv_extra or {}).get("振幅%")
    rv = (rv_extra or {}).get("RV%")
    src = str(row.get("_来源") or "")
    sector = str(row.get("行业") or row.get("_行业EN") or "")
    name = str(row.get("名称") or sym)
    primary = tags[0] if tags else "动量观察"
    reason = f"涨幅{gain:+.1f}% · 成交额${dvol:.0f}M"
    if amp is not None and np.isfinite(amp):
        reason += f" · 振幅{amp:.1f}%"
    if rv is not None and np.isfinite(rv):
        reason += f" · RV{rv:.0f}%"
    if src:
        reason += f" · {src}"
    score = gain + min(dvol / 10.0, 50.0) + (10.0 if "Gainer10+" in tags else 0.0)
    return {
        "代码": sym,
        "名称": name,
        "模块": "全市场快扫",
        "信号": primary,
        "标签": tags,
        "状态": "机会",
        "方向": "偏多" if gain >= 0 else "偏空",
        "现价": round(price, 4) if price else None,
        "涨幅%": round(gain, 2),
        "成交额M": round(dvol, 1),
        "振幅%": round(float(amp), 2) if amp is not None and np.isfinite(amp) else None,
        "RV%": round(float(rv), 1) if rv is not None and np.isfinite(rv) else None,
        "行业": sector,
        "来源榜": src,
        "选股理由": reason,
        "机会评分": round(score, 1),
        "数据源": "真实行情",
        "数据有效": True,
        "可交易": True,
    }


def run_market_scan(cfg: ScanConfig | None = None) -> dict[str, Any]:
    """执行一次全市场快扫，返回 JSON 文档。"""
    cfg = cfg or ScanConfig()
    t0 = time.time()
    deadline = t0 + float(cfg.budget_sec)

    snap = fetch_yahoo_screens_parallel(cfg.screen_presets, count=cfg.screen_count)
    phase1_sec = time.time() - t0
    universe_n = len(snap)

    filtered = _base_filter(snap, cfg)
    signals: list[dict[str, Any]] = []
    enrich_map: dict[str, dict[str, float]] = {}
    phase2_sec = 0.0

    candidates: list[tuple[pd.Series, list[str], float]] = []
    for _, row in filtered.iterrows():
        tags = _match_rules(row, cfg)
        if not tags:
            continue
        if tags == ["动量观察"] and float(row.get("_gain", 0)) < 5.0:
            continue
        score = float(row.get("_gain", 0)) + float(row.get("_dvol_m", 0)) / 10.0
        candidates.append((row, tags, score))
    candidates.sort(key=lambda x: -x[2])

    if cfg.enrich_rv and candidates and time.time() < deadline - 5:
        top_syms = [str(r["代码"]).upper() for r, _, _ in candidates[: cfg.enrich_top_n]]
        t2 = time.time()
        enrich_map = _enrich_rv_batch(
            top_syms,
            lookback_days=cfg.rv_lookback_days,
        )
        phase2_sec = time.time() - t2

    for row, tags, _ in candidates:
        sym = str(row["代码"]).upper()
        sig = _row_to_signal(row, tags, enrich_map.get(sym))
        signals.append(sig)

    signals.sort(key=lambda s: (-float(s.get("机会评分") or 0), str(s.get("代码") or "")))
    elapsed = time.time() - t0
    tag_counts: dict[str, int] = {}
    for s in signals:
        for t in s.get("标签") or []:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    return {
        "扫描时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "扫描日期": date.today().isoformat(),
        "scan_mode": "fast_market",
        "budget_sec": cfg.budget_sec,
        "elapsed_sec": round(elapsed, 2),
        "within_budget": elapsed <= cfg.budget_sec,
        "scan_stats": {
            "universe": universe_n,
            "after_liquidity_filter": len(filtered),
            "signals": len(signals),
            "phase1_sec": round(phase1_sec, 2),
            "phase2_sec": round(phase2_sec, 2),
            "presets": [UNIVERSE_PRESETS.get(p, p) for p in cfg.screen_presets],
            "tag_counts": tag_counts,
        },
        "summary": {
            "总信号": len(signals),
            "Gainer10+": tag_counts.get("Gainer10+", 0),
            "极端波动": tag_counts.get("极端上涨", 0) + tag_counts.get("极端下跌", 0),
            "说明": f"全市场 {universe_n} 只 · {elapsed:.0f}s 内完成",
        },
        "signals": signals,
        "picks": signals,
    }


def config_from_dict(raw: dict | None) -> ScanConfig:
    raw = raw or {}
    presets = raw.get("screen_presets") or list(DEFAULT_SCREEN_PRESETS)
    return ScanConfig(
        screen_count=int(raw.get("screen_count", 250)),
        screen_presets=tuple(str(p) for p in presets),
        min_price=float(raw.get("min_price", 2.0)),
        min_dvol_m=float(raw.get("min_dvol_m", 5.0)),
        budget_sec=float(raw.get("budget_sec", 300.0)),
        enrich_rv=bool(raw.get("enrich_rv", True)),
        enrich_top_n=int(raw.get("enrich_top_n", 120)),
        rv_lookback_days=int(raw.get("rv_lookback_days", 35)),
        rules=raw.get("rules") or ScanConfig().rules,
    )


def save_scan(doc: dict[str, Any], path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p
