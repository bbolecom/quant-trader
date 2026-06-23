"""暴涨>15% + 成交额门槛 → 80%规则每日扫描。

观察池 → 次日/3日确认 → 追多 / 回避 信号。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.providers import DataConfig, get_provider, reset_provider_cache
from quant.screener import fetch_gainer_universe_live
from research.gainer_daily_backtest import GAINER_MOMENTUM, LIQUID100, build_factor_panels
from research.liquid_tier_a_scan import build_candidate_pool


@dataclass
class Gain15ScanConfig:
    min_gain_pct: float = 15.0
    min_dvol_m: float = 50.0
    gainer_count: int = 250
    watch_max_days: int = 5
    use_broad_pool: bool = True
    quick: bool = False


@dataclass
class RuleHit:
    rule_id: str
    rule_name: str
    action: str
    hit_rate: float
    avg_fwd_5d_pct: float
    n_backtest: int


@dataclass
class WatchEvent:
    代码: str
    暴涨日: str
    涨幅_pct: float
    成交额M: float
    gain_rank: int
    站上MA20: bool
    站上MA50: bool
    创20日高: bool
    涨幅20d_pct: float | None
    相对SPY20d_pct: float | None
    量比: float | None
    收盘强度: float | None
    SPY站上MA20: bool
    暴涨收盘价: float
    status: str = "watching"  # watching | buy | avoid | expired

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> WatchEvent:
        return cls(
            代码=str(d["代码"]),
            暴涨日=str(d["暴涨日"]),
            涨幅_pct=float(d["涨幅_pct"]),
            成交额M=float(d["成交额M"]),
            gain_rank=int(d["gain_rank"]),
            站上MA20=bool(d.get("站上MA20", False)),
            站上MA50=bool(d.get("站上MA50", False)),
            创20日高=bool(d.get("创20日高", False)),
            涨幅20d_pct=_opt_float(d.get("涨幅20d_pct")),
            相对SPY20d_pct=_opt_float(d.get("相对SPY20d_pct")),
            量比=_opt_float(d.get("量比")),
            收盘强度=_opt_float(d.get("收盘强度")),
            SPY站上MA20=bool(d.get("SPY站上MA20", False)),
            暴涨收盘价=float(d["暴涨收盘价"]),
            status=str(d.get("status", "watching")),
        )


def _opt_float(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _close_strength(df: pd.DataFrame, idx: int) -> float:
    row = df.iloc[idx]
    hi = float(row["High"]) if "High" in df.columns else float(row["Close"])
    lo = float(row["Low"]) if "Low" in df.columns else float(row["Close"])
    cl = float(row["Close"])
    if hi <= lo:
        return 0.5
    return (cl - lo) / (hi - lo)


def _safe_return(close: pd.Series, n: int) -> float:
    if len(close) <= n:
        return np.nan
    base = float(close.iloc[-1 - n])
    if base <= 0:
        return np.nan
    return float(close.iloc[-1] / base - 1.0)


def returns_from_spike(df: pd.DataFrame, spike_date: str) -> dict[str, Any]:
    """从暴涨日收盘起算，T+1/T+3 累计收益（交易日）。"""
    out: dict[str, Any] = {
        "ret_1d": None,
        "ret_3d": None,
        "ret_td": None,
        "tdays_since": 0,
        "last_close": None,
    }
    if df is None or df.empty:
        return out
    d = df.copy()
    d.index = pd.to_datetime(d.index)
    spike_ts = pd.Timestamp(spike_date)
    if spike_ts not in d.index:
        prior = d.index[d.index <= spike_ts]
        if prior.empty:
            return out
        spike_ts = prior[-1]
    idx = int(d.index.get_loc(spike_ts))
    close = d["Close"].astype(float)
    base = float(close.iloc[idx])
    out["spike_close"] = base
    out["last_close"] = float(close.iloc[-1])
    out["tdays_since"] = len(close) - idx - 1
    out["ret_td"] = float(close.iloc[-1] / base - 1.0) if base > 0 else None
    for n, key in [(1, "ret_1d"), (3, "ret_3d")]:
        if idx + n < len(close) and base > 0:
            out[key] = float(close.iloc[idx + n] / base - 1.0)
    return out


def eval_surge_rules(ev: WatchEvent, rets: dict[str, Any]) -> list[RuleHit]:
    """80%+ 继续暴涨规则。"""
    hits: list[RuleHit] = []
    d1 = rets.get("ret_1d")
    d3 = rets.get("ret_3d")
    top3 = ev.gain_rank <= 3

    def add(rid: str, name: str, rate: float, avg5: float, n: int) -> None:
        hits.append(RuleHit(rid, name, "追多", rate, avg5, n))

    if d1 is not None and d3 is not None:
        if d1 > 0.10 and d3 > 0.20:
            add("S1", "次日涨>10%+3日累计涨>20%", 0.901, 51.3, 152)
        if top3 and d1 > 0.10 and d3 > 0.20:
            add("S2", "Top3+次日涨>10%+3日累计涨>20%", 0.886, 53.5, 105)
        if top3 and d1 > 0.10 and d3 > 0.15:
            add("S3", "Top3+次日涨>10%+3日累计涨>15%", 0.858, 45.2, 134)
        if d1 > 0.10 and d3 > 0.15:
            add("S4", "次日涨>10%+3日累计涨>15%", 0.843, 42.5, 198)
        if top3 and d1 > 0.07 and d3 > 0.12:
            add("S5", "Top3+次日涨>7%+3日累计涨>12%", 0.815, 39.1, 173)

    if d3 is not None:
        if top3 and d3 > 0.20:
            add("S6", "Top3+3日累计涨>20%", 0.873, 45.2, 165)
        if top3 and d3 > 0.15:
            add("S7", "Top3+3日累计涨>15%", 0.828, 37.4, 232)

    if d1 is not None and top3 and ev.涨幅20d_pct is not None:
        if 20 <= ev.涨幅20d_pct <= 50 and d1 > 0.15:
            add("S8", "Top3+前期涨20~50%+次日涨>15%", 0.846, 42.8, 26)

    return hits


def eval_drop_rules(ev: WatchEvent, rets: dict[str, Any]) -> list[RuleHit]:
    """80%+ 大幅回调规则。"""
    hits: list[RuleHit] = []
    d1 = rets.get("ret_1d")
    d3 = rets.get("ret_3d")
    top3 = ev.gain_rank <= 3
    g20 = ev.涨幅20d_pct
    spy_ok = ev.SPY站上MA20

    def add(rid: str, name: str, rate: float, avg5: float, n: int) -> None:
        hits.append(RuleHit(rid, name, "回避/做空", rate, avg5, n))

    if d3 is not None and top3:
        if d3 < -0.20:
            add("D1", "Top3+3日累计跌>20%", 0.938, -30.9, 130)
        elif d3 < -0.15:
            add("D2", "Top3+3日累计跌>15%", 0.913, -25.2, 231)
        elif d3 < -0.10:
            add("D3", "Top3+3日累计跌>10%", 0.821, -20.3, 369)

    if d1 is not None:
        if d1 < -0.20:
            add("D4", "次日跌>20%", 0.878, -33.2, 74)
        elif d1 < -0.15:
            if top3:
                add("D5", "Top3+次日跌>15%", 0.823, -25.4, 113)
            else:
                add("D6", "次日跌>15%", 0.840, -25.6, 150)
            if g20 is not None and g20 > 50 and top3:
                add("D7", "前期涨>50%+Top3+次日跌>15%", 0.853, -28.5, 75)
            if g20 is not None and g20 > 50 and ev.涨幅_pct > 30:
                add("D8", "前期涨>50%+当日涨>30%+次日跌>15%", 0.842, -29.5, 57)
            if not spy_ok and g20 is not None and g20 > 50:
                add("D9", "大盘弱+前期涨>50%+次日跌>15%", 0.958, -34.9, 24)
        elif d1 < -0.12 and not spy_ok and g20 is not None and g20 > 50:
            add("D10", "大盘弱+前期涨>50%+次日跌>12%", 0.893, -31.6, 28)

    if d1 is not None and not ev.站上MA20 and d1 < -0.15:
        add("D11", "MA20下+次日跌>15%", 0.846, -26.3, 26)

    return hits


def eval_early_hints(ev: WatchEvent, rets: dict[str, Any]) -> list[str]:
    """未达80%但值得关注的早期提示。"""
    hints: list[str] = []
    d1 = rets.get("ret_1d")
    tdays = int(rets.get("tdays_since") or 0)
    if d1 is None or tdays < 1:
        return hints
    if d1 > 0.05:
        hints.append(f"次日涨{d1*100:.1f}%，偏强，等3日确认（目标累计涨>15%）")
    elif d1 < -0.05:
        hints.append(f"次日跌{abs(d1)*100:.1f}%，偏弱，警惕深调")
    if tdays >= 3 and rets.get("ret_3d") is not None:
        d3 = rets["ret_3d"]
        if 0.10 < d3 < 0.15 and ev.gain_rank <= 3:
            hints.append(f"3日累计涨{d3*100:.1f}%，接近追多确认线(15%)")
        if -0.15 < d3 < -0.10 and ev.gain_rank <= 3:
            hints.append(f"3日累计跌{abs(d3)*100:.1f}%，接近回避确认线(15%)")
    return hints


def build_spike_snapshot(
    panel: pd.DataFrame,
    spy_close: pd.Series,
    cfg: Gain15ScanConfig,
    as_of: str,
) -> pd.DataFrame:
    """当日暴涨候选（涨幅>门槛，成交额>门槛）。"""
    d = panel.copy()
    d["dvol_m"] = pd.to_numeric(d["成交额USD"], errors="coerce") / 1e6
    d = d.dropna(subset=["涨幅%"])
    d = d[(d["涨幅%"] > cfg.min_gain_pct) & (d["dvol_m"] >= cfg.min_dvol_m)]
    if d.empty:
        return d
    d["gain_rank"] = (
        d["涨幅%"].rank(method="first", ascending=False).astype(int)
    )
    spy_hist = spy_close.loc[spy_close.index <= pd.Timestamp(as_of)]
    spy_ma20 = float(spy_hist.tail(20).mean()) if len(spy_hist) >= 20 else np.nan
    spy_px = float(spy_hist.iloc[-1]) if len(spy_hist) else np.nan
    d["SPY站上MA20"] = spy_px > spy_ma20 if np.isfinite(spy_ma20) else False
    return d.sort_values("gain_rank")


def enrich_spike_row(row: pd.Series, data: dict[str, pd.DataFrame], as_of: str) -> dict[str, Any]:
    tk = str(row["代码"])
    px = row.get("收盘价", row.get("最新价"))
    if px is None or (isinstance(px, float) and np.isnan(px)):
        df = data.get(tk)
        if df is not None and not df.empty:
            hist = df.loc[df.index <= pd.Timestamp(as_of)]
            px = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        else:
            px = 0.0
    cs = _round_or_none(row.get("收盘强度"))
    hi20 = bool(row.get("创20日高", False))
    if cs is None or not hi20:
        df = data.get(tk)
        if df is not None and not df.empty:
            hist = df.loc[df.index <= pd.Timestamp(as_of)]
            if len(hist) >= 20:
                close = hist["Close"].astype(float)
                if not hi20:
                    hi20 = float(close.iloc[-1]) >= float(close.tail(20).max()) - 1e-9
                if cs is None:
                    cs = round(_close_strength(hist, -1), 3)
    return {
        "代码": tk,
        "暴涨日": as_of,
        "涨幅_pct": round(float(row["涨幅%"]), 2),
        "成交额M": round(float(row["dvol_m"]), 1),
        "gain_rank": int(row["gain_rank"]),
        "站上MA20": bool(row.get("站上MA20", False)),
        "站上MA50": bool(row.get("站上MA50", False)),
        "创20日高": hi20,
        "涨幅20d_pct": _round_or_none(row.get("涨幅20d%")),
        "相对SPY20d_pct": _round_or_none(row.get("相对SPY20d%")),
        "量比": _round_or_none(row.get("量比")),
        "收盘强度": cs,
        "SPY站上MA20": bool(row.get("SPY站上MA20", False)),
        "暴涨收盘价": round(float(px), 4),
        "status": "watching",
    }


def _round_or_none(v: Any, nd: int = 2) -> float | None:
    x = _opt_float(v)
    return round(x, nd) if x is not None else None


def load_watch_pool(path: Path) -> list[WatchEvent]:
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    return [WatchEvent.from_dict(e) for e in doc.get("events", [])]


def save_watch_pool(path: Path, events: list[WatchEvent], as_of: str) -> None:
    doc = {"updated": as_of, "events": [e.to_dict() for e in events]}
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def prune_watch_pool(events: list[WatchEvent], as_of: str, max_days: int) -> list[WatchEvent]:
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=max_days + 2)
    out: list[WatchEvent] = []
    for ev in events:
        if pd.Timestamp(ev.暴涨日) < cutoff:
            if ev.status == "watching":
                ev.status = "expired"
            continue
        if ev.status in ("buy", "avoid"):
            # 已触发信号保留 2 天供查阅
            if pd.Timestamp(ev.暴涨日) < pd.Timestamp(as_of) - pd.Timedelta(days=2):
                continue
        out.append(ev)
    return out


def merge_new_spikes(
    pool: list[WatchEvent],
    spikes: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    as_of: str,
) -> list[WatchEvent]:
    existing = {(e.代码, e.暴涨日) for e in pool}
    for _, row in spikes.iterrows():
        key = (str(row["代码"]), as_of)
        if key in existing:
            continue
        pool.append(WatchEvent.from_dict(enrich_spike_row(row, data, as_of)))
    return pool


def confirm_watch_events(
    pool: list[WatchEvent],
    batch: dict[str, pd.DataFrame],
    as_of: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """返回 (追多确认, 回避确认, 待观察)。"""
    buys: list[dict] = []
    avoids: list[dict] = []
    watching: list[dict] = []

    for ev in pool:
        if ev.status in ("buy", "avoid", "expired"):
            continue
        df = batch.get(ev.代码)
        rets = returns_from_spike(df, ev.暴涨日)
        surge = eval_surge_rules(ev, rets)
        drop = eval_drop_rules(ev, rets)
        hints = eval_early_hints(ev, rets)

        base = {
            "代码": ev.代码,
            "暴涨日": ev.暴涨日,
            "暴涨日涨幅%": ev.涨幅_pct,
            "涨幅榜排名": ev.gain_rank,
            "成交额M": ev.成交额M,
            "已过交易日": rets.get("tdays_since", 0),
            "累计涨跌%": round(rets["ret_td"] * 100, 2) if rets.get("ret_td") is not None else None,
            "次日涨跌%": round(rets["ret_1d"] * 100, 2) if rets.get("ret_1d") is not None else None,
            "3日累计%": round(rets["ret_3d"] * 100, 2) if rets.get("ret_3d") is not None else None,
            "现价": rets.get("last_close"),
            "早期提示": hints,
        }

        if drop:
            best = max(drop, key=lambda h: h.hit_rate)
            ev.status = "avoid"
            avoids.append({
                **base,
                "信号": "回避/做空",
                "规则ID": best.rule_id,
                "规则": best.rule_name,
                "历史命中率": f"{best.hit_rate:.0%}",
                "历史5日均": f"{best.avg_fwd_5d_pct:+.1f}%",
                "全部命中": [h.rule_name for h in drop],
            })
        elif surge:
            best = max(surge, key=lambda h: h.hit_rate)
            ev.status = "buy"
            buys.append({
                **base,
                "信号": "追多",
                "规则ID": best.rule_id,
                "规则": best.rule_name,
                "历史命中率": f"{best.hit_rate:.0%}",
                "历史5日均": f"{best.avg_fwd_5d_pct:+.1f}%",
                "全部命中": [h.rule_name for h in surge],
            })
        else:
            watching.append({**base, "信号": "观察中"})

    return buys, avoids, watching


def run_gain15_scan(cfg_dict: dict | None = None, *, as_of: str | None = None) -> dict[str, Any]:
    """执行完整每日扫描。"""
    cfg_raw = cfg_dict or {}
    cfg = Gain15ScanConfig(
        min_gain_pct=float(cfg_raw.get("min_gain_pct", 15.0)),
        min_dvol_m=float(cfg_raw.get("min_dvol_m", 50.0)),
        gainer_count=int(cfg_raw.get("gainer_count", 250)),
        watch_max_days=int(cfg_raw.get("watch_max_days", 5)),
        use_broad_pool=bool(cfg_raw.get("use_broad_pool", True)),
        quick=bool(cfg_raw.get("quick", False)),
    )
    as_of = as_of or date.today().isoformat()
    paths = cfg_raw.get("outputs") or {}
    watch_path = Path(cfg_raw.get("watch_pool") or paths.get("watch_pool") or "research/gain15_watch_pool.json")
    if not watch_path.is_absolute():
        root = Path(__file__).resolve().parents[1]
        watch_path = root / watch_path

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    start = (date.fromisoformat(as_of) - timedelta(days=120)).isoformat()

    snap_live = fetch_gainer_universe_live(count=cfg.gainer_count)
    tickers: set[str] = set(snap_live["代码"].astype(str).tolist()) if not snap_live.empty else set()
    if cfg.use_broad_pool:
        pool = build_candidate_pool(use_broad=not cfg.quick, max_names=80 if cfg.quick else 0)
        tickers.update(pool)
    tickers.update(GAINER_MOMENTUM)
    tickers.update(LIQUID100)
    tickers = sorted(tickers)

    batch = yahoo.fetch_batch(tickers, start, as_of)
    spy_df = batch.pop("SPY", None)
    if spy_df is None or spy_df.empty:
        spy_df = yahoo.fetch_history("SPY", start, as_of)
    spy_close = spy_df["Close"].astype(float)
    spy_close.index = pd.to_datetime(spy_df.index)

    panel = build_factor_panels(batch, spy_close)
    panel["日期"] = pd.to_datetime(panel["日期"])
    panel_today = panel[panel["日期"] == pd.Timestamp(as_of)].copy()
    if panel_today.empty and not panel.empty:
        last_day = panel["日期"].max()
        panel_today = panel[panel["日期"] == last_day].copy()
        as_of = last_day.strftime("%Y-%m-%d")

    spikes = build_spike_snapshot(panel_today, spy_close, cfg, as_of)

    new_rows: list[dict] = []
    for _, row in spikes.iterrows():
        new_rows.append(enrich_spike_row(row, batch, as_of))

    pool = load_watch_pool(watch_path)
    pool = prune_watch_pool(pool, as_of, cfg.watch_max_days)
    pool = merge_new_spikes(pool, spikes, batch, as_of)

    # 确认扫描需要观察池内所有 ticker 行情
    watch_tickers = sorted({e.代码 for e in pool})
    confirm_batch = {k: batch[k] for k in watch_tickers if k in batch}
    missing = [t for t in watch_tickers if t not in confirm_batch]
    if missing:
        extra = yahoo.fetch_batch(missing, start, as_of)
        confirm_batch.update(extra)

    buys, avoids, watching = confirm_watch_events(pool, confirm_batch, as_of)
    save_watch_pool(watch_path, pool, as_of)

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
            "universe": len(tickers),
            "new_spikes": len(new_rows),
            "watch_pool": len(pool),
        },
        "new_spikes": new_rows,
        "buy_confirmed": buys,
        "avoid_confirmed": avoids,
        "watching": watching,
        "recent_triggered": [
            e.to_dict() for e in pool if e.status in ("buy", "avoid")
        ],
    }
