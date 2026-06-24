#!/usr/bin/env python3
"""5 年暴涨/暴跌偏好池 · 挖掘 + 规律研究。

流程：
  1. 从全市场候选中，按 5 年「单日涨≥7% / 跌≤-7%」频率筛出 Top 池
  2. 对池内标的扫描 ±15% 极端事件，统计暴涨后/暴跌后的 1/3/5/10 日多空收益
  3. 输出池名单、个股画像、池级规律 JSON

用法：
    python research/surge_drop_pool_research.py
    python research/surge_drop_pool_research.py --top 60 --threshold 15
    python research/surge_drop_pool_research.py --build-only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.providers import DataConfig, get_provider, reset_provider_cache
from quant.surge_drop_pool import (
    POOL_CSV,
    POOL_JSON,
    SurgeDropFilter,
    build_surge_drop_pool,
    load_pool,
    save_pool,
)
from research.extreme15_pattern import TRAIN_END, _stat, build_event_panel

RESULT_JSON = ROOT / "research" / "surge_drop_pool_research.json"
EVENTS_CSV = ROOT / "research" / "surge_drop_pool_events.csv"


def discover_pool_events(ev: pd.DataFrame) -> dict:
    """池内 ±阈值 极端事件 · 暴涨后/暴跌后规律。"""
    out: dict = {}
    for d, label in [("surge", "暴涨后"), ("drop", "暴跌后")]:
        sub = ev[ev["direction"] == d]
        if sub.empty:
            continue
        block: dict = {"事件数": int(len(sub)), "涉及代码": int(sub["代码"].nunique())}
        for k in (1, 3, 5, 10):
            lo = sub[f"long_open_{k}d"]
            block[f"做多{k}日"] = _stat(lo)
            block[f"做空{k}日"] = _stat(-lo)
        buckets: list[dict] = []
        col = "long_open_3d"
        for name, mask in [
            ("收盘强度≥0.8", sub["close_strength"] >= 0.8),
            ("收盘强度≤0.3", sub["close_strength"] <= 0.3),
            ("量比≥3", sub["vol_ratio"] >= 3),
            ("前20日已涨>30%", sub["pre20"] > 0.30),
            ("前20日已跌>30%", sub["pre20"] < -0.30),
            ("创20日高", sub["high20"] == True),  # noqa: E712
            ("创20日低", sub["low20"] == True),  # noqa: E712
            ("大盘多头", sub["spy_bull"] == True),  # noqa: E712
        ]:
            seg = sub[mask]
            if len(seg) < 15:
                continue
            buckets.append({
                "条件": name,
                "做多3日": _stat(seg[col]),
                "做空3日": _stat(-seg[col]),
            })
        block["条件分档"] = buckets
        out[label] = block
    return out


def ticker_event_stats(ev: pd.DataFrame) -> pd.DataFrame:
    """按代码汇总极端事件频率与后续收益。"""
    rows: list[dict] = []
    for tk, sub in ev.groupby("代码"):
        surge = sub[sub["direction"] == "surge"]
        drop = sub[sub["direction"] == "drop"]
        rows.append({
            "代码": tk,
            "极端事件": len(sub),
            "暴涨事件": len(surge),
            "暴跌事件": len(drop),
            "暴涨后3日均": round(float(surge["long_open_3d"].mean()) * 100, 2) if len(surge) else None,
            "暴跌后3日均": round(float(drop["long_open_3d"].mean()) * 100, 2) if len(drop) else None,
            "暴涨后3日胜率": round(float((surge["long_open_3d"] > 0).mean()), 3) if len(surge) else None,
            "暴跌后3日胜率": round(float((drop["long_open_3d"] > 0).mean()), 3) if len(drop) else None,
        })
    return pd.DataFrame(rows).sort_values("极端事件", ascending=False).reset_index(drop=True)


def run_research(
    *,
    years: int = 5,
    top_n: int = 80,
    threshold: float = 15.0,
    min_dvol_m: float = 50.0,
    build_only: bool = False,
) -> dict:
    filt = SurgeDropFilter(years=years, top_n=top_n, min_dvol_m=min_dvol_m)
    print(f"① 构建暴涨/暴跌池 · {years} 年 · Top {top_n}…")
    profiles = build_surge_drop_pool(filt=filt, include_seed=True)
    if profiles.empty:
        return {"error": "无候选画像"}

    selected = profiles[profiles["Top池"]] if "Top池" in profiles.columns else profiles[profiles["入选"]]
    tickers = selected["代码"].tolist()
    print(f"   入选 {len(tickers)} 只（自 {len(profiles)} 只候选）")

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=years * 365 + 120)).isoformat()

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    data = yahoo.fetch_batch(tickers + ["SPY"], start, end)
    spy = data.pop("SPY", None)
    if spy is None or spy.empty:
        spy = yahoo.fetch_history("SPY", start, end)
    spy_close = spy["Close"].astype(float)
    spy_close.index = pd.to_datetime(spy.index)

    pool_doc = save_pool(profiles, filt)
    if build_only:
        return pool_doc

    print(f"② 池内 ±{threshold}% 极端事件研究 · {start}~{end}…")
    ev = build_event_panel(
        data,
        spy_close,
        threshold_pct=threshold,
        min_price=3.0,
        min_dvol_m=min_dvol_m,
    )
    if not ev.empty:
        ev.to_csv(EVENTS_CSV, index=False, encoding="utf-8-sig")

    discover = discover_pool_events(ev) if not ev.empty else {}
    by_ticker = ticker_event_stats(ev) if not ev.empty else pd.DataFrame()

    is_ev = ev[pd.to_datetime(ev["日期"]) <= pd.Timestamp(TRAIN_END)] if not ev.empty else ev
    oos_ev = ev[pd.to_datetime(ev["日期"]) > pd.Timestamp(TRAIN_END)] if not ev.empty else ev

    doc = {
        **pool_doc,
        "research": {
            "period": {"start": start, "end": end, "train_end": TRAIN_END},
            "event_threshold_pct": threshold,
            "events": {
                "total": int(len(ev)),
                "surge": int((ev["direction"] == "surge").sum()) if not ev.empty else 0,
                "drop": int((ev["direction"] == "drop").sum()) if not ev.empty else 0,
                "is": int(len(is_ev)),
                "oos": int(len(oos_ev)),
            },
            "discover": discover,
            "discover_is": discover_pool_events(is_ev) if not is_ev.empty else {},
            "discover_oos": discover_pool_events(oos_ev) if not oos_ev.empty else {},
            "by_ticker": by_ticker.head(30).to_dict(orient="records") if not by_ticker.empty else [],
        },
    }
    RESULT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def print_report(doc: dict) -> None:
    if doc.get("error"):
        print(doc["error"])
        return

    s = doc.get("summary") or {}
    r = doc.get("research") or {}
    ev = r.get("events") or {}
    print("\n" + "=" * 78)
    print(f"暴涨/暴跌池研究 · {doc.get('count', 0)} 只 · 扫描 {s.get('candidates_scanned', '?')} 候选")
    print("=" * 78)
    print(
        f"池均：年均极端 {s.get('avg_extreme_days_yr', 0):.1f} 天 · "
        f"暴涨 {s.get('avg_surge_days_yr', 0):.1f} · 暴跌 {s.get('avg_drop_days_yr', 0):.1f} · "
        f"波动 {s.get('avg_realized_vol', 0):.2f}"
    )
    print(f"±{r.get('event_threshold_pct', 15)}% 事件 {ev.get('total', 0)} 条 "
          f"（暴涨 {ev.get('surge', 0)} · 暴跌 {ev.get('drop', 0)}）")

    for tag, blk in [("全样本", r.get("discover")), ("IS", r.get("discover_is")), ("OOS", r.get("discover_oos"))]:
        if not blk:
            continue
        print(f"\n【{tag} · 极端事件后续 · 次日开盘入场】")
        for label, b in blk.items():
            print(f"  {label} · {b['事件数']} 事件 / {b.get('涉及代码', '?')} 只")
            for k in (1, 3, 5, 10):
                lo = b.get(f"做多{k}日") or {}
                sh = b.get(f"做空{k}日") or {}
                if not lo.get("n"):
                    continue
                print(
                    f"    {k:>2}日 做多 均{lo['mean%']:+.2f}% 胜{lo['win%']:.0f}%  | "
                    f"做空 均{sh['mean%']:+.2f}% 胜{sh['win%']:.0f}%"
                )

    top = r.get("by_ticker") or []
    if top:
        print("\n【池内极端事件 Top10】")
        print(f"{'代码':<7}{'事件':>6}{'暴涨':>6}{'暴跌':>6}{'涨后3d':>9}{'跌后3d':>9}{'涨后胜率':>9}{'跌后胜率':>9}")
        for row in top[:10]:
            print(
                f"{row['代码']:<7}{row['极端事件']:>6}{row['暴涨事件']:>6}{row['暴跌事件']:>6}"
                f"{(row['暴涨后3日均'] or 0):>+9.1f}"
                f"{(row['暴跌后3日均'] or 0):>+9.1f}"
                f"{(row['暴涨后3日胜率'] or 0):>9.0%}"
                f"{(row['暴跌后3日胜率'] or 0):>9.0%}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="5年暴涨/暴跌池 · 挖掘与研究")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--top", type=int, default=80)
    ap.add_argument("--threshold", type=float, default=15.0, help="极端事件研究阈值(%)")
    ap.add_argument("--min-dvol-m", type=float, default=50.0)
    ap.add_argument("--build-only", action="store_true", help="只建池，不做事件研究")
    args = ap.parse_args()

    doc = run_research(
        years=args.years,
        top_n=args.top,
        threshold=args.threshold,
        min_dvol_m=args.min_dvol_m,
        build_only=args.build_only,
    )
    print_report(doc)
    print(f"\n→ {POOL_JSON.name}")
    if not args.build_only:
        print(f"→ {RESULT_JSON.name} · {EVENTS_CSV.name}")


if __name__ == "__main__":
    main()
