#!/usr/bin/env python3
"""涨幅>15% + 成交额≥5000万美元 → 后续5日走势专项分析。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS_CSV = ROOT / "research" / "gainer_top100_events.csv"
OUT_JSON = ROOT / "research" / "gain15_dvol50_analysis.json"

MIN_GAIN = 15.0
MIN_DVOL_M = 50.0
TRAIN_END = "2023-12-31"


def stats(s: pd.DataFrame, col: str) -> dict | None:
    x = pd.to_numeric(s[col], errors="coerce").dropna()
    if len(x) == 0:
        return None
    return {
        "n": int(len(x)),
        "mean_pct": round(float(x.mean()) * 100, 2),
        "median_pct": round(float(x.median()) * 100, 2),
        "win_rate": round(float((x > 0).mean()), 3),
        "p25_pct": round(float(x.quantile(0.25)) * 100, 2),
        "p75_pct": round(float(x.quantile(0.75)) * 100, 2),
    }


def bucket_report(sub: pd.DataFrame, col: str, buckets: list, fwd: str = "fwd_5d") -> list:
    rows = []
    for lo, hi, label in buckets:
        if col == "gain_rank":
            b = sub[(sub[col] >= lo) & (sub[col] <= hi)]
        else:
            b = sub[(sub[col] >= lo) & (sub[col] < hi)]
        if len(b) < 10:
            continue
        rows.append({
            "bucket": label,
            "n": int(len(b)),
            "fwd_1d": stats(b, "fwd_1d"),
            "fwd_5d": stats(b, fwd),
        })
    return rows


def combo_report(sub: pd.DataFrame) -> list:
    combos = [
        (
            "易继续上涨: MA20上+相对SPY强+量比适中+涨幅<30%",
            (sub["站上MA20"] == True)
            & (sub["相对SPY20d%"].fillna(0) > 10)
            & (sub["量比"].fillna(0) < 5)
            & (sub["涨幅%"] < 30),
        ),
        (
            "易回调: MA20下+暴涨≥30%+量比≥3",
            (sub["站上MA20"] == False)
            & (sub["涨幅%"] >= 30)
            & (sub["量比"].fillna(0) >= 3),
        ),
        (
            "趋势延续: 站上MA20+创20日高+收阳",
            (sub["站上MA20"] == True)
            & (sub["创20日高"] == True)
            & (sub["收阳"] == True),
        ),
        (
            "超跌反弹: MA20下+前期20日跌",
            (sub["站上MA20"] == False)
            & (pd.to_numeric(sub["涨幅20d%"], errors="coerce").fillna(0) < 0),
        ),
        (
            "高位放量: 涨幅≥30%+量比≥5",
            (sub["涨幅%"] >= 30) & (sub["量比"].fillna(0) >= 5),
        ),
        (
            "大盘股暴涨: 成交额>$500M+涨幅15~25%",
            (sub["dvol_m"] >= 500)
            & (sub["涨幅%"] >= 15)
            & (sub["涨幅%"] < 25),
        ),
    ]
    rows = []
    for label, mask in combos:
        b = sub[mask]
        if len(b) < 10:
            continue
        rows.append({
            "combo": label,
            "n": int(len(b)),
            "fwd_1d": stats(b, "fwd_1d"),
            "fwd_5d": stats(b, "fwd_5d"),
            "fwd_10d": stats(b, "fwd_10d"),
        })
    return rows


def run() -> dict:
    events = pd.read_csv(EVENTS_CSV, encoding="utf-8-sig")
    events["dt"] = pd.to_datetime(events["日期"])
    sub = events[
        (events["涨幅%"] > MIN_GAIN) & (events["dvol_m"] >= MIN_DVOL_M)
    ].copy()

    is_df = sub[sub["dt"] <= pd.Timestamp(TRAIN_END)]
    oos_df = sub[sub["dt"] > pd.Timestamp(TRAIN_END)]

    pu = pd.to_numeric(sub["path_up_5d"], errors="coerce").dropna()
    pd_ = pd.to_numeric(sub["path_down_5d"], errors="coerce").dropna()

    sub2 = sub.copy()
    sub2["rs20"] = pd.to_numeric(sub2["相对SPY20d%"], errors="coerce")
    sub2["g20"] = pd.to_numeric(sub2["涨幅20d%"], errors="coerce")
    sub2["cs"] = pd.to_numeric(sub2["收盘强度"], errors="coerce")

    doc = {
        "filters": {"min_gain_pct": MIN_GAIN, "min_dvol_m_usd": MIN_DVOL_M},
        "summary": {
            "events": int(len(sub)),
            "unique_days": int(sub["日期"].nunique()),
            "unique_tickers": int(sub["代码"].nunique()),
            "date_range": [str(sub["日期"].min()), str(sub["日期"].max())],
            "avg_gain_pct": round(float(sub["涨幅%"].mean()), 2),
            "avg_dvol_m": round(float(sub["dvol_m"].mean()), 1),
        },
        "fwd_all": {
            "fwd_1d": stats(sub, "fwd_1d"),
            "fwd_3d": stats(sub, "fwd_3d"),
            "fwd_5d": stats(sub, "fwd_5d"),
            "fwd_10d": stats(sub, "fwd_10d"),
            "fwd_20d": stats(sub, "fwd_20d"),
            "path_up_5d_ge2pct": round(float((pu >= 0.02).mean()), 3),
            "path_down_5d_ge2pct": round(float((pd_ <= -0.02).mean()), 3),
        },
        "is_2019_2023": {"fwd_5d": stats(is_df, "fwd_5d"), "events": int(len(is_df))},
        "oos_2024_plus": {"fwd_5d": stats(oos_df, "fwd_5d"), "events": int(len(oos_df))},
        "by_gain": bucket_report(
            sub,
            "涨幅%",
            [(15, 20, "15~20%"), (20, 30, "20~30%"), (30, 50, "30~50%"), (50, 999, ">50%")],
        ),
        "by_vol_ratio": bucket_report(
            sub2,
            "量比",
            [(0, 1.5, "量比<1.5"), (1.5, 2.5, "1.5~2.5"), (2.5, 5, "2.5~5"), (5, 999, ">5")],
        ),
        "by_dvol": bucket_report(
            sub,
            "dvol_m",
            [(50, 100, "$50~100M"), (100, 500, "$100~500M"), (500, 2000, "$500M~2B"), (2000, 99999, ">$2B")],
        ),
        "by_rs20": bucket_report(
            sub2,
            "rs20",
            [(-999, 0, "弱于大盘"), (0, 10, "略强0~10%"), (10, 30, "强10~30%"), (30, 999, "超强>30%")],
        ),
        "by_gain_rank": bucket_report(
            sub,
            "gain_rank",
            [(1, 3, "Top1~3"), (4, 10, "Top4~10"), (11, 30, "Top11~30"), (31, 100, "Top31~100")],
        ),
        "by_prior_20d": bucket_report(
            sub2,
            "g20",
            [(-999, 0, "前期跌"), (0, 20, "前期涨0~20%"), (20, 50, "前期涨20~50%"), (50, 999, "前期涨>50%")],
        ),
        "by_close_strength": bucket_report(
            sub2,
            "cs",
            [(0, 0.3, "收盘弱<0.3"), (0.3, 0.7, "收盘中0.3~0.7"), (0.7, 1.01, "收盘强>0.7")],
        ),
        "by_ma20": [
            {"bucket": f"MA20={'上' if v else '下'}", "n": int(len(b)), "fwd_5d": stats(b, "fwd_5d")}
            for v in [True, False]
            for b in [sub[sub["站上MA20"] == v]]
            if len(b) >= 10
        ],
        "combos": combo_report(sub2),
        "next_day_effect": [
            {
                "bucket": label,
                "n": int(len(b)),
                "fwd_5d": stats(b, "fwd_5d"),
            }
            for label, b in [
                ("次日涨>3%", sub.dropna(subset=["fwd_1d", "fwd_5d"]).query("fwd_1d > 0.03")),
                ("次日跌>3%", sub.dropna(subset=["fwd_1d", "fwd_5d"]).query("fwd_1d < -0.03")),
                ("次日平±3%", sub.dropna(subset=["fwd_1d", "fwd_5d"]).query("fwd_1d >= -0.03 and fwd_1d <= 0.03")),
            ]
            if len(b) >= 10
        ],
    }
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def print_report(doc: dict) -> None:
    s = doc["summary"]
    print("=" * 72)
    print(f"涨幅>{doc['filters']['min_gain_pct']}% · 成交额≥${doc['filters']['min_dvol_m_usd']:.0f}M → 后续走势")
    print("=" * 72)
    print(f"事件 {s['events']} · {s['unique_days']} 交易日 · {s['unique_tickers']} 标的")
    print(f"范围 {s['date_range'][0]} ~ {s['date_range'][1]} · 均涨幅 {s['avg_gain_pct']}% · 均成交额 ${s['avg_dvol_m']}M")

    print("\n【后续走势】")
    for hz, label in [("fwd_1d", "次日"), ("fwd_3d", "3日"), ("fwd_5d", "5日"), ("fwd_10d", "10日"), ("fwd_20d", "20日")]:
        x = doc["fwd_all"].get(hz)
        if x:
            print(f"  {label}: 均{x['mean_pct']:+.2f}%  中位{x['median_pct']:+.2f}%  胜率{x['win_rate']:.1%}  (n={x['n']})")
    fa = doc["fwd_all"]
    print(f"  5日路径涨≥2%: {fa['path_up_5d_ge2pct']:.1%}  5日路径跌≥2%: {fa['path_down_5d_ge2pct']:.1%}")

    is5 = doc["is_2019_2023"]["fwd_5d"]
    oos5 = doc["oos_2024_plus"]["fwd_5d"]
    print(f"\n样本内5日: 均{is5['mean_pct']:+.2f}% 胜率{is5['win_rate']:.0%} (n={is5['n']})")
    print(f"样本外5日: 均{oos5['mean_pct']:+.2f}% 胜率{oos5['win_rate']:.0%} (n={oos5['n']})")

    for title, key in [
        ("按当日涨幅", "by_gain"),
        ("按量比", "by_vol_ratio"),
        ("按成交额", "by_dvol"),
        ("按相对SPY强度", "by_rs20"),
        ("按涨幅榜排名", "by_gain_rank"),
        ("按前期20日涨幅", "by_prior_20d"),
        ("按收盘强度", "by_close_strength"),
    ]:
        rows = doc.get(key) or []
        if not rows:
            continue
        print(f"\n【{title} · 5日后续】")
        for r in rows:
            f5 = r.get("fwd_5d") or {}
            f1 = r.get("fwd_1d") or {}
            print(
                f"  {r['bucket']:16s} n={r['n']:4d}  "
                f"次日{f1.get('mean_pct', 0):+.2f}%/{f1.get('win_rate', 0):.0%}  "
                f"5日{f5.get('mean_pct', 0):+.2f}%/{f5.get('win_rate', 0):.0%}"
            )

    print("\n【MA20位置 · 5日后续】")
    for r in doc.get("by_ma20") or []:
        f5 = r["fwd_5d"]
        print(f"  {r['bucket']:8s} n={r['n']:4d}  5日均{f5['mean_pct']:+.2f}%  胜率{f5['win_rate']:.0%}")

    print("\n【组合筛选】")
    for r in doc.get("combos") or []:
        f5 = r["fwd_5d"]
        f1 = r["fwd_1d"]
        print(f"  {r['combo']}")
        print(f"    n={r['n']}  次日{f1['mean_pct']:+.2f}%/{f1['win_rate']:.0%}  5日{f5['mean_pct']:+.2f}%/{f5['win_rate']:.0%}")

    print("\n【次日走势对5日影响】")
    for r in doc.get("next_day_effect") or []:
        f5 = r["fwd_5d"]
        print(f"  {r['bucket']:12s} n={r['n']:4d}  5日均{f5['mean_pct']:+.2f}%  胜率{f5['win_rate']:.0%}")

    print("=" * 72)
    print(f"→ {OUT_JSON}")


if __name__ == "__main__":
    doc = run()
    print_report(doc)
