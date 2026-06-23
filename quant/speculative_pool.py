"""SPCE 类投机票池：从历史暴涨事件 + 可选实时形态，筛出同类标的。

画像（以 SPCE 为原型）：
  · 小中盘（市值约 $0.05B~$5B）
  · 多次日涨 ≥15%（暴涨基因）
  · 单日最大涨幅较高（垂直拉升能力）
  · 暴涨日成交额中等（$15M~$800M），非 mega 流动性
  · 排除 ETF / 杠杆 / 超大盘
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = ROOT / "research" / "gainer_top100_events.csv"
DEFAULT_SHARES = ROOT / "research" / "shares_cache.json"
DEFAULT_OUT = ROOT / "research" / "speculative_pool.json"

# 杠杆 / 指数 ETF，非单票投机
EXCLUDE_ETFS = frozenset({
    "SOXL", "TQQQ", "SQQQ", "LABU", "LABD", "FNGU", "FNGD", "NVDL", "TSLL",
    "MSTX", "MSTU", "CONL", "BULL", "APPX", "SPY", "QQQ", "IWM", "SPXU",
    "UVXY", "VXX", "TNA", "TZA", "UPRO", "SPXU", "TECL", "TECS",
})

# 与 SPCE 同类的种子（meme / 太空 / 小盘高波）
DEFAULT_SEEDS = sorted(set([
    "SPCE", "RKLB", "LUNR", "ACHR", "JOBY", "ASTS", "LAZR", "ASTR", "RDW",
    "OPEN", "SOUN", "AI", "IONQ", "QUBT", "RGTI", "QBTS", "BTBT", "NVTS",
    "AEHR", "SMR", "OKLO", "LUNR", "PLUG", "BE", "QS", "RUN", "NKLA",
    "UPST", "AFRM", "CVNA", "HOOD", "SOFI", "RIVN", "LCID", "GME", "AMC",
    "MARA", "RIOT", "CLSK", "HUT", "BITF", "SOUN", "BBAI", "DNA", "GRRR",
    "MSTR", "COIN", "SMCI", "DJT", "RDDT", "DKNG", "RBLX", "WULF", "CIFR",
]))


@dataclass
class SpeculativePoolConfig:
    archetype: str = "SPCE"
    min_spikes15: int = 5
    min_max_gain_pct: float = 25.0
    min_mcap_b: float = 0.05
    max_mcap_b: float = 5.0
    min_med_dvol_m: float = 15.0
    max_med_dvol_m: float = 800.0
    max_med_dvol_m_hard: float = 1500.0
    pool_size: int = 50
    core_size: int = 25
    min_price: float = 0.5
    max_price: float = 100.0
    events_csv: str = "research/gainer_top100_events.csv"
    shares_cache: str = "research/shares_cache.json"
    seed_tickers: list[str] = field(default_factory=lambda: list(DEFAULT_SEEDS))
    exclude_tickers: list[str] = field(default_factory=lambda: sorted(EXCLUDE_ETFS))
    enrich_live: bool = True
    live_lookback_days: int = 400


@dataclass
class PoolMember:
    代码: str
    相似分: float
    暴涨15次数: int
    暴涨30次数: int
    最大单日涨_pct: float
    事件总数: int
    暴涨日成交额M中位: float
    市值B: float | None
    现价: float | None
    阶段: str | None = None
    说明: str = ""
    tier: str = "extended"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_shares(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k).upper(): float(v) for k, v in raw.items() if v is not None}
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return {}


def _archetype_stats(events: pd.DataFrame, ticker: str) -> dict[str, float]:
    sub = events[events["代码"] == ticker.upper()]
    if sub.empty:
        return {
            "spikes15": 10.0,
            "spikes30": 2.0,
            "max_gain": 40.0,
            "med_dvol_m": 300.0,
            "n": 100.0,
        }
    gain = pd.to_numeric(sub["涨幅%"], errors="coerce")
    dvol = pd.to_numeric(sub["dvol_m"], errors="coerce")
    return {
        "spikes15": float((gain >= 15).sum()),
        "spikes30": float((gain >= 30).sum()),
        "max_gain": float(gain.max()),
        "med_dvol_m": float(dvol.median()) if dvol.notna().any() else 300.0,
        "n": float(len(sub)),
    }


def _ratio_score(value: float, target: float, *, cap: float = 2.0) -> float:
    if target <= 0 or not np.isfinite(value):
        return 0.0
    r = value / target
    if r > cap:
        r = cap
    if r < 1 / cap:
        r = 1 / cap
    return float(1.0 - abs(np.log(r)) / np.log(cap))


def similarity_score(row: pd.Series, arch: dict[str, float]) -> float:
    """与原型票（默认 SPCE）的多维相似度，0~1。"""
    parts = [
        _ratio_score(float(row["spikes15"]), arch["spikes15"]) * 0.30,
        _ratio_score(float(row["max_gain"]), arch["max_gain"]) * 0.20,
        _ratio_score(float(row["med_dvol_m"]), arch["med_dvol_m"]) * 0.20,
        _ratio_score(float(row["n"]), arch["n"]) * 0.15,
        min(float(row["spikes30"]) / max(arch["spikes30"], 1), 1.5) / 1.5 * 0.15,
    ]
    return round(float(sum(parts)), 4)


def aggregate_ticker_stats(events: pd.DataFrame) -> pd.DataFrame:
    """按代码汇总历史暴涨事件。"""
    df = events.copy()
    df["涨幅%"] = pd.to_numeric(df["涨幅%"], errors="coerce")
    df["dvol_m"] = pd.to_numeric(df["dvol_m"], errors="coerce")
    g = df.groupby("代码").agg(
        n=("涨幅%", "count"),
        spikes15=("涨幅%", lambda s: int((s >= 15).sum())),
        spikes30=("涨幅%", lambda s: int((s >= 30).sum())),
        max_gain=("涨幅%", "max"),
        med_gain=("涨幅%", "median"),
        med_dvol_m=("dvol_m", "median"),
    ).reset_index()
    return g


def build_pool_from_events(
    cfg: SpeculativePoolConfig | None = None,
    *,
    events_path: Path | None = None,
    shares_path: Path | None = None,
) -> tuple[list[PoolMember], dict[str, Any]]:
    """从历史 Top100 暴涨事件构建 SPCE 类票池（离线可用）。"""
    cfg = cfg or SpeculativePoolConfig()
    events_path = events_path or (ROOT / cfg.events_csv)
    shares_path = shares_path or (ROOT / cfg.shares_cache)

    if not events_path.exists():
        raise FileNotFoundError(f"事件文件不存在: {events_path}")

    events = pd.read_csv(events_path, encoding="utf-8-sig")
    stats = aggregate_ticker_stats(events)
    arch = _archetype_stats(events, cfg.archetype.upper())
    exclude = {str(t).upper() for t in cfg.exclude_tickers} | EXCLUDE_ETFS

    # 最新收盘价估算市值（事件 CSV 中市值常为空）
    last_close: dict[str, float] = {}
    if "收盘价" in events.columns:
        ev = events.copy()
        ev["日期"] = pd.to_datetime(ev["日期"], errors="coerce")
        ev["收盘价"] = pd.to_numeric(ev["收盘价"], errors="coerce")
        ev = ev.dropna(subset=["日期", "收盘价"]).sort_values("日期")
        last_close = ev.groupby("代码")["收盘价"].last().to_dict()

    shares = _load_shares(shares_path)

    filt = stats[
        (stats["spikes15"] >= cfg.min_spikes15)
        & (stats["max_gain"] >= cfg.min_max_gain_pct)
        & (stats["med_dvol_m"] >= cfg.min_med_dvol_m)
        & (stats["med_dvol_m"] <= cfg.max_med_dvol_m_hard)
    ].copy()
    filt = filt[~filt["代码"].isin(exclude)]

    # 市值过滤
    mcap_b: dict[str, float] = {}
    for tk in filt["代码"]:
        sh = shares.get(str(tk).upper())
        px = last_close.get(tk)
        if sh and px and px > 0:
            mcap_b[tk] = sh * px / 1e9
    filt["mcap_b"] = filt["代码"].map(mcap_b)

    has_mcap = filt["mcap_b"].notna()
    in_range = (filt["mcap_b"] >= cfg.min_mcap_b) & (filt["mcap_b"] <= cfg.max_mcap_b)
    filt = filt[~has_mcap | in_range]

    # med_dvol 软上限（SPCE 类非 mega 流）
    filt = filt[filt["med_dvol_m"] <= cfg.max_med_dvol_m]

    filt["相似分"] = filt.apply(lambda r: similarity_score(r, arch), axis=1)

    # 种子票：补充漏网同类，但仍遵守市值/流动性硬约束
    arch_tk = cfg.archetype.upper()
    seed_set = {str(t).upper() for t in cfg.seed_tickers}
    seed_rows = stats[stats["代码"].isin(seed_set) & ~stats["代码"].isin(filt["代码"])].copy()
    if not seed_rows.empty:
        seed_rows["mcap_b"] = seed_rows["代码"].map(mcap_b)
        seed_rows["相似分"] = seed_rows.apply(lambda r: similarity_score(r, arch) * 0.85, axis=1)
        sm = seed_rows[
            (seed_rows["代码"] == arch_tk)
            | (
                (seed_rows["spikes15"] >= max(3, cfg.min_spikes15 - 2))
                & (seed_rows["max_gain"] >= cfg.min_max_gain_pct * 0.8)
                & (seed_rows["med_dvol_m"] <= cfg.max_med_dvol_m_hard)
            )
        ]
        has_mcap_s = sm["mcap_b"].notna()
        in_range_s = (sm["mcap_b"] >= cfg.min_mcap_b) & (sm["mcap_b"] <= cfg.max_mcap_b)
        sm = sm[(sm["代码"] == arch_tk) | ~has_mcap_s | in_range_s]
        filt = pd.concat([filt, sm], ignore_index=True)

    filt = filt.sort_values(["相似分", "spikes15", "max_gain"], ascending=False)
    filt = filt.drop_duplicates(subset=["代码"], keep="first")
    filt = filt.head(cfg.pool_size)

    members: list[PoolMember] = []
    for i, row in filt.iterrows():
        tk = str(row["代码"])
        tier = "core" if len(members) < cfg.core_size else "extended"
        px = last_close.get(tk)
        members.append(
            PoolMember(
                代码=tk,
                相似分=float(row["相似分"]),
                暴涨15次数=int(row["spikes15"]),
                暴涨30次数=int(row["spikes30"]),
                最大单日涨_pct=round(float(row["max_gain"]), 2),
                事件总数=int(row["n"]),
                暴涨日成交额M中位=round(float(row["med_dvol_m"]), 1),
                市值B=round(float(row["mcap_b"]), 3) if pd.notna(row.get("mcap_b")) else None,
                现价=round(float(px), 4) if px and np.isfinite(px) else None,
                tier=tier,
                说明=(
                    f"15%+×{int(row['spikes15'])} · 最大{float(row['max_gain']):.0f}% · "
                    f"暴涨额${float(row['med_dvol_m']):.0f}M"
                ),
            )
        )

    meta = {
        "archetype": cfg.archetype.upper(),
        "archetype_stats": arch,
        "filters": {
            "min_spikes15": cfg.min_spikes15,
            "min_max_gain_pct": cfg.min_max_gain_pct,
            "mcap_b": [cfg.min_mcap_b, cfg.max_mcap_b],
            "med_dvol_m": [cfg.min_med_dvol_m, cfg.max_med_dvol_m],
        },
        "pool_size": len(members),
        "core_size": min(cfg.core_size, len(members)),
    }
    return members, meta


def enrich_pool_live(
    members: list[PoolMember],
    cfg: SpeculativePoolConfig | None = None,
    *,
    as_of: str | None = None,
) -> list[PoolMember]:
    """拉最新行情，标注 C/A/B 阶段（surge_scan）。"""
    if not members:
        return members
    cfg = cfg or SpeculativePoolConfig()
    as_of = as_of or date.today().isoformat()

    from quant.providers import DataConfig, get_provider, reset_provider_cache
    from quant.surge_scan import SurgeScanConfig, scan_ticker_latest

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    start = (date.fromisoformat(as_of) - timedelta(days=cfg.live_lookback_days)).isoformat()
    tickers = [m.代码 for m in members]
    batch = yahoo.fetch_batch(tickers, start, as_of)

    surge_cfg = SurgeScanConfig(min_dvol_m=5.0)
    out: list[PoolMember] = []
    for m in members:
        df = batch.get(m.代码)
        stage = "观望"
        note = m.说明
        px = m.现价
        if df is not None and not df.empty:
            px = round(float(df["Close"].iloc[-1]), 4)
            hit = scan_ticker_latest(m.代码, df, surge_cfg, as_of=as_of)
            if hit is not None:
                stage = hit.类型名
                note = f"{stage} · {hit.说明}"
            else:
                close = df["Close"].astype(float)
                ret_20d = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 21 else 0
                if ret_20d <= -0.15:
                    stage = "出清/深跌"
                elif ret_20d >= 0.30:
                    stage = "趋势中"
                else:
                    stage = "盘整"
        out.append(
            PoolMember(
                代码=m.代码,
                相似分=m.相似分,
                暴涨15次数=m.暴涨15次数,
                暴涨30次数=m.暴涨30次数,
                最大单日涨_pct=m.最大单日涨_pct,
                事件总数=m.事件总数,
                暴涨日成交额M中位=m.暴涨日成交额M中位,
                市值B=m.市值B,
                现价=px,
                阶段=stage,
                说明=note,
                tier=m.tier,
            )
        )
    return out


def run_speculative_pool(
    cfg_dict: dict | None = None,
    *,
    as_of: str | None = None,
) -> dict[str, Any]:
    """构建并可选 enrich 票池，返回完整文档。"""
    raw = cfg_dict or {}
    cfg = SpeculativePoolConfig(
        archetype=str(raw.get("archetype", "SPCE")),
        min_spikes15=int(raw.get("min_spikes15", 5)),
        min_max_gain_pct=float(raw.get("min_max_gain_pct", 25.0)),
        min_mcap_b=float(raw.get("min_mcap_b", 0.05)),
        max_mcap_b=float(raw.get("max_mcap_b", 5.0)),
        min_med_dvol_m=float(raw.get("min_med_dvol_m", 15.0)),
        max_med_dvol_m=float(raw.get("max_med_dvol_m", 800.0)),
        pool_size=int(raw.get("pool_size", 50)),
        core_size=int(raw.get("core_size", 25)),
        events_csv=str(raw.get("events_csv", "research/gainer_top100_events.csv")),
        shares_cache=str(raw.get("shares_cache", "research/shares_cache.json")),
        seed_tickers=list(raw.get("seed_tickers") or DEFAULT_SEEDS),
        exclude_tickers=list(raw.get("exclude_tickers") or sorted(EXCLUDE_ETFS)),
        enrich_live=bool(raw.get("enrich_live", True)),
    )
    as_of = as_of or date.today().isoformat()

    members, meta = build_pool_from_events(cfg)
    if cfg.enrich_live:
        try:
            members = enrich_pool_live(members, cfg, as_of=as_of)
        except Exception as exc:  # noqa: BLE001
            meta["live_enrich_error"] = str(exc)

    core = [m.to_dict() for m in members if m.tier == "core"]
    extended = [m.to_dict() for m in members if m.tier == "extended"]
    precursors = [m.to_dict() for m in members if m.阶段 == "前兆蓄势"]
    breakouts = [m.to_dict() for m in members if m.阶段 == "突破型"]

    return {
        "updated": as_of,
        "name": "SPCE类投机票池",
        "description": "历史暴涨基因 + 小中盘 + 与 SPCE 多维相似度排序",
        "meta": meta,
        "tickers": [m.代码 for m in members],
        "core_tickers": [m.代码 for m in members if m.tier == "core"],
        "core": core,
        "extended": extended,
        "today_precursors": precursors,
        "today_breakouts": breakouts,
        "members": [m.to_dict() for m in members],
    }


def load_pool_tickers(path: Path | None = None, *, tier: str = "all") -> list[str]:
    """从 JSON 读取票池代码列表。tier: all | core | extended。"""
    p = path or DEFAULT_OUT
    if not p.exists():
        return []
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
        if tier == "core":
            raw = doc.get("core_tickers") or [m.get("代码") for m in doc.get("core") or []]
        elif tier == "extended":
            raw = [m.get("代码") for m in doc.get("extended") or []]
        else:
            raw = doc.get("tickers") or []
        out: list[str] = []
        seen: set[str] = set()
        for t in raw:
            u = str(t).upper()
            if u and u not in seen:
                seen.add(u)
                out.append(u)
        return out
    except (json.JSONDecodeError, OSError):
        return []
