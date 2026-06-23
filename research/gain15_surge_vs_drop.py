#!/usr/bin/env python3
"""继续暴涨 vs 大幅回调 — 深度规律挖掘。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "research" / "gainer_top100_events.csv"
OUT = ROOT / "research" / "gain15_surge_vs_drop.json"

MIN_GAIN = 15.0
MIN_DVOL_M = 50.0
# 继续暴涨：5日收盘涨≥10%；大幅回调：5日收盘跌≤-10%
SURGE_TH = 0.10
DROP_TH = -0.10


def load() -> pd.DataFrame:
    df = pd.read_csv(EVENTS, encoding="utf-8-sig")
    df["dt"] = pd.to_datetime(df["日期"])
    df = df[(df["涨幅%"] > MIN_GAIN) & (df["dvol_m"] >= MIN_DVOL_M)].copy()
    df["fwd_5d"] = pd.to_numeric(df["fwd_5d"], errors="coerce")
    df["fwd_1d"] = pd.to_numeric(df["fwd_1d"], errors="coerce")
    df["path_up_5d"] = pd.to_numeric(df["path_up_5d"], errors="coerce")
    df["path_down_5d"] = pd.to_numeric(df["path_down_5d"], errors="coerce")
    df["rs20"] = pd.to_numeric(df["相对SPY20d%"], errors="coerce")
    df["g20"] = pd.to_numeric(df["涨幅20d%"], errors="coerce")
    df["g5_prior"] = pd.to_numeric(df.get("SPY5d涨%"), errors="coerce")  # market, not stock
    df["cs"] = pd.to_numeric(df["收盘强度"], errors="coerce")
    df["vol_ratio"] = pd.to_numeric(df["量比"], errors="coerce")
    df["mcap_b"] = pd.to_numeric(df["市值USD"], errors="coerce") / 1e9
    return df.dropna(subset=["fwd_5d"])


def pct(x: float) -> str:
    return f"{x*100:+.1f}%" if abs(x) < 10 else f"{x*100:+.0f}%"


def compare_numeric(surge: pd.DataFrame, drop: pd.DataFrame, col: str, label: str) -> dict:
    a = pd.to_numeric(surge[col], errors="coerce").dropna()
    b = pd.to_numeric(drop[col], errors="coerce").dropna()
    if a.empty or b.empty:
        return {}
    return {
        "factor": label,
        "surge_mean": round(float(a.mean()), 2),
        "drop_mean": round(float(b.mean()), 2),
        "surge_median": round(float(a.median()), 2),
        "drop_median": round(float(b.median()), 2),
        "delta_mean": round(float(a.mean() - b.mean()), 2),
    }


def compare_bool(surge: pd.DataFrame, drop: pd.DataFrame, col: str, label: str) -> dict:
    sa = surge[col].astype(bool)
    da = drop[col].astype(bool)
    sr = float(sa.mean())
    dr = float(da.mean())
    return {
        "factor": label,
        "surge_rate": round(sr, 3),
        "drop_rate": round(dr, 3),
        "delta": round(sr - dr, 3),
    }


def rule_lift(df: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    sub = df[mask]
    if len(sub) < 15:
        return {}
    n = len(sub)
    surge_rate = float((sub["fwd_5d"] >= SURGE_TH).mean())
    drop_rate = float((sub["fwd_5d"] <= DROP_TH).mean())
    avg5 = float(sub["fwd_5d"].mean()) * 100
    base_surge = float((df["fwd_5d"] >= SURGE_TH).mean())
    base_drop = float((df["fwd_5d"] <= DROP_TH).mean())
    return {
        "rule": label,
        "n": n,
        "pct_of_total": round(n / len(df), 3),
        "avg_fwd_5d_pct": round(avg5, 2),
        "surge_rate": round(surge_rate, 3),
        "drop_rate": round(drop_rate, 3),
        "surge_lift": round(surge_rate / base_surge, 2) if base_surge else None,
        "drop_lift": round(drop_rate / base_drop, 2) if base_drop else None,
    }


def top_rules(df: pd.DataFrame, rules: list[tuple[str, pd.Series]], top_n: int = 12) -> tuple[list, list]:
    scored_s = []
    scored_d = []
    for label, mask in rules:
        r = rule_lift(df, mask, label)
        if not r:
            continue
        scored_s.append(r)
        scored_d.append(r)
    by_surge = sorted(scored_s, key=lambda x: (x["surge_rate"], x["avg_fwd_5d_pct"]), reverse=True)[:top_n]
    by_drop = sorted(scored_d, key=lambda x: (x["drop_rate"], -x["avg_fwd_5d_pct"]), reverse=True)[:top_n]
    return by_surge, by_drop


def run() -> dict:
    df = load()
    surge = df[df["fwd_5d"] >= SURGE_TH]
    drop = df[df["fwd_5d"] <= DROP_TH]
    mid = df[(df["fwd_5d"] > DROP_TH) & (df["fwd_5d"] < SURGE_TH)]

    numeric_cols = [
        ("涨幅%", "当日涨幅%"),
        ("gain_rank", "涨幅榜排名(越小越好)"),
        ("dvol_m", "成交额(百万USD)"),
        ("vol_ratio", "量比"),
        ("g20", "前期20日涨幅%"),
        ("rs20", "相对SPY20d%"),
        ("cs", "收盘强度"),
        ("fwd_1d", "次日收益(小数)"),
        ("path_up_5d", "5日路径最高涨幅"),
        ("path_down_5d", "5日路径最大回撤"),
        ("mcap_b", "市值(十亿美元)"),
    ]
    bool_cols = [
        ("站上MA20", "站上MA20"),
        ("站上MA50", "站上MA50"),
        ("创20日高", "创20日高"),
        ("收阳", "收阳"),
        ("SPY站上MA20", "大盘站上MA20"),
    ]

    num_cmp = [compare_numeric(surge, drop, c, l) for c, l in numeric_cols]
    num_cmp = [x for x in num_cmp if x]
    bool_cmp = [compare_bool(surge, drop, c, l) for c, l in bool_cols]
    bool_cmp = [x for x in bool_cmp if x]

    # 单因子分档：每档的暴涨/回调率
    def bucket_table(col, buckets, label):
        rows = []
        for lo, hi, name in buckets:
            if col == "gain_rank":
                b = df[(df[col] >= lo) & (df[col] <= hi)]
            else:
                b = df[(df[col] >= lo) & (df[col] < hi)]
            if len(b) < 20:
                continue
            rows.append({
                "bucket": name,
                "n": len(b),
                "surge_rate": round(float((b["fwd_5d"] >= SURGE_TH).mean()), 3),
                "drop_rate": round(float((b["fwd_5d"] <= DROP_TH).mean()), 3),
                "avg_5d_pct": round(float(b["fwd_5d"].mean()) * 100, 2),
            })
        return {"factor": label, "buckets": rows}

    bucket_tables = [
        bucket_table("fwd_1d", [(-1, -0.05, "次日跌>5%"), (-0.05, -0.03, "次日跌3~5%"),
                                (-0.03, 0.03, "次日平±3%"), (0.03, 0.05, "次日涨3~5%"),
                                (0.05, 1, "次日涨>5%")], "次日走势"),
        bucket_table("gain_rank", [(1, 3, "Top1~3"), (4, 10, "Top4~10"),
                                   (11, 30, "Top11~30"), (31, 100, "Top31+")], "涨幅榜排名"),
        bucket_table("涨幅%", [(15, 20, "15~20%"), (20, 30, "20~30%"),
                               (30, 50, "30~50%"), (50, 999, ">50%")], "当日涨幅"),
        bucket_table("vol_ratio", [(0, 2, "量比<2"), (2, 4, "量比2~4"),
                                   (4, 8, "量比4~8"), (8, 999, "量比>8")], "量比"),
        bucket_table("g20", [(-999, 0, "前期跌"), (0, 20, "前期涨0~20%"),
                             (20, 50, "前期涨20~50%"), (50, 999, "前期涨>50%")], "前期20日涨幅"),
        bucket_table("rs20", [(-999, 0, "弱于SPY"), (0, 20, "强于SPY0~20%"),
                              (20, 50, "强于SPY20~50%"), (50, 999, "强于SPY>50%")], "相对SPY强度"),
        bucket_table("cs", [(0, 0.4, "收盘弱"), (0.4, 0.7, "收盘中"), (0.7, 1.01, "收盘强")], "收盘强度"),
    ]

    rules = [
        ("Top3 + MA20上 + 相对SPY>20%", (df["gain_rank"] <= 3) & (df["站上MA20"]) & (df["rs20"] > 20)),
        ("Top3 + 次日涨>3%", (df["gain_rank"] <= 3) & (df["fwd_1d"] > 0.03)),
        ("Top3 + 次日涨>5%", (df["gain_rank"] <= 3) & (df["fwd_1d"] > 0.05)),
        ("MA20上 + 创20日高 + 涨幅15~25%", (df["站上MA20"]) & (df["创20日高"]) & df["涨幅%"].between(15, 25)),
        ("MA20上 + 前期涨20~50% + 涨幅<30%", (df["站上MA20"]) & df["g20"].between(20, 50) & (df["涨幅%"] < 30)),
        ("MA20上 + 量比2~5 + 相对SPY>30%", (df["站上MA20"]) & df["vol_ratio"].between(2, 5) & (df["rs20"] > 30)),
        ("次日涨>5% + MA20上", (df["fwd_1d"] > 0.05) & (df["站上MA20"])),
        ("次日涨>5% + Top10", (df["fwd_1d"] > 0.05) & (df["gain_rank"] <= 10)),
        ("次日涨>5% + 相对SPY>30%", (df["fwd_1d"] > 0.05) & (df["rs20"] > 30)),
        ("MA20下 + 涨幅>30%", (~df["站上MA20"]) & (df["涨幅%"] > 30)),
        ("MA20下 + 前期跌", (~df["站上MA20"]) & (df["g20"] < 0)),
        ("MA20下 + 次日跌>3%", (~df["站上MA20"]) & (df["fwd_1d"] < -0.03)),
        ("涨幅>40% + 量比>5", (df["涨幅%"] > 40) & (df["vol_ratio"] > 5)),
        ("涨幅>40% + 收盘弱<0.4", (df["涨幅%"] > 40) & (df["cs"] < 0.4)),
        ("Top30+ + 涨幅>25%", (df["gain_rank"] > 30) & (df["涨幅%"] > 25)),
        ("次日跌>5%", df["fwd_1d"] < -0.05),
        ("次日跌>5% + 涨幅>30%", (df["fwd_1d"] < -0.05) & (df["涨幅%"] > 30)),
        ("次日跌>5% + MA20下", (df["fwd_1d"] < -0.05) & (~df["站上MA20"])),
        ("次日跌>5% + 前期涨>50%", (df["fwd_1d"] < -0.05) & (df["g20"] > 50)),
        ("前期涨>50% + 当日涨>30%", (df["g20"] > 50) & (df["涨幅%"] > 30)),
        ("前期涨>50% + 次日跌>3%", (df["g20"] > 50) & (df["fwd_1d"] < -0.03)),
        ("收盘弱 + 涨幅>30%", (df["cs"] < 0.4) & (df["涨幅%"] > 30)),
        ("收盘强>0.85 + 涨幅>35%", (df["cs"] > 0.85) & (df["涨幅%"] > 35)),
        ("大盘弱(SPY未上MA20) + 暴涨>25%", (~df["SPY站上MA20"]) & (df["涨幅%"] > 25)),
        ("大盘强 + Top3 + 次日涨", (df["SPY站上MA20"]) & (df["gain_rank"] <= 3) & (df["fwd_1d"] > 0.03)),
        ("小盘(市值<2B) + 涨>35%", (df["mcap_b"] < 2) & (df["涨幅%"] > 35)),
        ("大盘(市值>10B) + 涨15~22%", (df["mcap_b"] > 10) & df["涨幅%"].between(15, 22)),
    ]

    best_surge, best_drop = top_rules(df, rules)

    # 典型样本
    surge_ex = surge.nlargest(8, "fwd_5d")[["日期", "代码", "涨幅%", "gain_rank", "dvol_m",
                                             "vol_ratio", "g20", "rs20", "fwd_1d", "fwd_5d",
                                             "站上MA20", "创20日高"]]
    drop_ex = drop.nsmallest(8, "fwd_5d")[["日期", "代码", "涨幅%", "gain_rank", "dvol_m",
                                            "vol_ratio", "g20", "rs20", "fwd_1d", "fwd_5d",
                                            "站上MA20", "创20日高"]]

    doc = {
        "definition": {
            "filter": f"涨幅>{MIN_GAIN}% 成交额>=${MIN_DVOL_M}M",
            "surge": f"5日收盘涨>={SURGE_TH*100:.0f}%",
            "drop": f"5日收盘跌<={DROP_TH*100:.0f}%",
        },
        "counts": {
            "total": len(df),
            "surge": len(surge),
            "drop": len(drop),
            "mid": len(mid),
            "surge_pct": round(len(surge) / len(df), 3),
            "drop_pct": round(len(drop) / len(df), 3),
        },
        "group_avg_fwd_5d_pct": {
            "surge": round(float(surge["fwd_5d"].mean()) * 100, 2),
            "drop": round(float(drop["fwd_5d"].mean()) * 100, 2),
            "mid": round(float(mid["fwd_5d"].mean()) * 100, 2),
        },
        "numeric_compare": sorted(num_cmp, key=lambda x: abs(x["delta_mean"]), reverse=True),
        "bool_compare": sorted(bool_cmp, key=lambda x: abs(x["delta"]), reverse=True),
        "bucket_tables": bucket_tables,
        "best_surge_rules": best_surge,
        "best_drop_rules": best_drop,
        "surge_examples": surge_ex.to_dict("records"),
        "drop_examples": drop_ex.to_dict("records"),
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def print_report(doc: dict) -> None:
    c = doc["counts"]
    d = doc["definition"]
    print("=" * 76)
    print(f"深度对比：{d['surge']} vs {d['drop']}  （筛选：{d['filter']}）")
    print("=" * 76)
    print(f"总样本 {c['total']}  |  继续暴涨 {c['surge']} ({c['surge_pct']:.1%})  "
          f"|  大幅回调 {c['drop']} ({c['drop_pct']:.1%})  |  中间 {c['mid']}")
    g = doc["group_avg_fwd_5d_pct"]
    print(f"三组5日均收益：暴涨组 {g['surge']:+.1f}%  |  中间 {g['mid']:+.1f}%  |  回调组 {g['drop']:+.1f}%")

    print("\n【数值因子：暴涨组 vs 回调组 均值对比】")
    for r in doc["numeric_compare"]:
        print(f"  {r['factor']:18s}  暴涨组 {r['surge_mean']:>8}  回调组 {r['drop_mean']:>8}  "
              f"差值 {r['delta_mean']:+.2f}")

    print("\n【布尔因子：暴涨组 vs 回调组 出现率】")
    for r in doc["bool_compare"]:
        print(f"  {r['factor']:14s}  暴涨组 {r['surge_rate']:.0%}  回调组 {r['drop_rate']:.0%}  "
              f"差 {r['delta']:+.0%}")

    print("\n【单因子分档 · 继续暴涨率 / 大幅回调率】")
    for tbl in doc["bucket_tables"]:
        print(f"  ▶ {tbl['factor']}")
        for b in tbl["buckets"]:
            print(f"    {b['bucket']:14s} n={b['n']:4d}  暴涨率{b['surge_rate']:.0%}  "
                  f"回调率{b['drop_rate']:.0%}  5日均{b['avg_5d_pct']:+.1f}%")

    print("\n【最易继续暴涨的组合规则 TOP】")
    for r in doc["best_surge_rules"]:
        print(f"  {r['rule']}")
        print(f"    n={r['n']} ({r['pct_of_total']:.0%})  5日均{r['avg_fwd_5d_pct']:+.1f}%  "
              f"暴涨率{r['surge_rate']:.0%}(×{r['surge_lift']})  回调率{r['drop_rate']:.0%}")

    print("\n【最易大幅回调的组合规则 TOP】")
    for r in doc["best_drop_rules"]:
        print(f"  {r['rule']}")
        print(f"    n={r['n']} ({r['pct_of_total']:.0%})  5日均{r['avg_fwd_5d_pct']:+.1f}%  "
              f"回调率{r['drop_rate']:.0%}(×{r['drop_lift']})  暴涨率{r['surge_rate']:.0%}")

    print("=" * 76)
    print(f"→ {OUT}")


if __name__ == "__main__":
    doc = run()
    print_report(doc)
