#!/usr/bin/env python3
"""搜索继续暴涨/大幅回调概率≥80%的条件组合。"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "research" / "gainer_top100_events.csv"
OUT = ROOT / "research" / "gain15_rules_80pct.json"

MIN_GAIN = 15.0
MIN_DVOL_M = 50.0
SURGE_TH = 0.10   # 5日涨≥10%
DROP_TH = -0.10   # 5日跌≤-10%
MIN_N = 20        # 最少样本
TARGET = 0.80


def load() -> pd.DataFrame:
    df = pd.read_csv(EVENTS, encoding="utf-8-sig")
    df = df[(df["涨幅%"] > MIN_GAIN) & (df["dvol_m"] >= MIN_DVOL_M)].copy()
    for c in ["fwd_5d", "fwd_1d", "fwd_3d", "path_up_5d", "path_down_5d",
              "rs20", "g20", "cs", "量比", "涨幅%", "gain_rank", "dvol_m", "市值USD"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["mcap_b"] = df["市值USD"] / 1e9
    df["vol_ratio"] = df["量比"]
    df["g20"] = df["涨幅20d%"]
    df["rs20"] = df["相对SPY20d%"]
    df["cs"] = df["收盘强度"]
    return df.dropna(subset=["fwd_5d"])


def eval_rule(df: pd.DataFrame, mask: pd.Series, label: str) -> dict | None:
    sub = df[mask]
    n = len(sub)
    if n < MIN_N:
        return None
    surge = float((sub["fwd_5d"] >= SURGE_TH).mean())
    drop = float((sub["fwd_5d"] <= DROP_TH).mean())
    avg5 = float(sub["fwd_5d"].mean()) * 100
    return {
        "rule": label,
        "n": n,
        "surge_rate": round(surge, 3),
        "drop_rate": round(drop, 3),
        "avg_fwd_5d_pct": round(avg5, 2),
        "median_fwd_5d_pct": round(float(sub["fwd_5d"].median()) * 100, 2),
    }


def build_rules(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    rules: list[tuple[str, pd.Series]] = []
    d1 = df["fwd_1d"]
    g = df["涨幅%"]
    gr = df["gain_rank"]
    g20 = df["g20"]
    rs = df["rs20"]
    vr = df["vol_ratio"]
    cs = df["cs"]
    ma20 = df["站上MA20"].astype(bool)
    ma50 = df["站上MA50"].astype(bool)
    hi20 = df["创20日高"].astype(bool)
    spy_ma = df["SPY站上MA20"].astype(bool)

    # 次日阈值网格
    for th in [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]:
        rules.append((f"次日涨>{th*100:.0f}%", d1 > th))
        rules.append((f"次日跌<-{th*100:.0f}%", d1 < -th))

    # 次日 + 排名
    for th in [0.05, 0.07, 0.10, 0.12, 0.15]:
        for rk, rlabel in [(3, "Top3"), (5, "Top5"), (10, "Top10")]:
            rules.append((f"{rlabel}+次日涨>{th*100:.0f}%", (gr <= rk) & (d1 > th)))
            rules.append((f"{rlabel}+次日跌>{th*100:.0f}%", (gr <= rk) & (d1 < -th)))

    # 次日 + MA20
    for th in [0.05, 0.07, 0.10, 0.12, 0.15]:
        rules.append((f"MA20上+次日涨>{th*100:.0f}%", ma20 & (d1 > th)))
        rules.append((f"MA20下+次日跌>{th*100:.0f}%", (~ma20) & (d1 < -th)))

    # 次日 + 相对SPY
    for th in [0.05, 0.07, 0.10]:
        for rs_th in [20, 30, 50]:
            rules.append((f"次日涨>{th*100:.0f}%+相对SPY>{rs_th}%", (d1 > th) & (rs > rs_th)))
            rules.append((f"次日跌>{th*100:.0f}%+相对SPY>{rs_th}%", (d1 < -th) & (rs > rs_th)))

    # 次日 + 前期涨幅
    for th in [0.03, 0.05, 0.07, 0.10]:
        rules.append((f"前期涨>50%+次日跌>{th*100:.0f}%", (g20 > 50) & (d1 < -th)))
        rules.append((f"前期涨>50%+次日涨>{th*100:.0f}%", (g20 > 50) & (d1 > th)))
        rules.append((f"前期涨20~50%+次日涨>{th*100:.0f}%", g20.between(20, 50) & (d1 > th)))
        rules.append((f"前期跌+次日跌>{th*100:.0f}%", (g20 < 0) & (d1 < -th)))

    # 当日涨幅 + 次日
    for gth in [20, 25, 30, 40]:
        for dth in [0.05, 0.07, 0.10]:
            rules.append((f"当日涨>{gth}%+次日涨>{dth*100:.0f}%", (g > gth) & (d1 > dth)))
            rules.append((f"当日涨>{gth}%+次日跌>{dth*100:.0f}%", (g > gth) & (d1 < -dth)))

    # 三重组合
    for th in [0.07, 0.10, 0.12, 0.15]:
        rules.append((f"Top3+MA20上+次日涨>{th*100:.0f}%", (gr <= 3) & ma20 & (d1 > th)))
        rules.append((f"Top3+MA20上+次日跌>{th*100:.0f}%", (gr <= 3) & ma20 & (d1 < -th)))
        rules.append((f"Top5+MA20上+相对SPY>20%+次日涨>{th*100:.0f}%",
                      (gr <= 5) & ma20 & (rs > 20) & (d1 > th)))
        rules.append((f"Top3+前期涨20~50%+次日涨>{th*100:.0f}%",
                      (gr <= 3) & g20.between(20, 50) & (d1 > th)))
        rules.append((f"Top3+涨幅15~25%+次日涨>{th*100:.0f}%",
                      (gr <= 3) & g.between(15, 25) & (d1 > th)))
        rules.append((f"前期涨>50%+当日涨>30%+次日跌>{th*100:.0f}%",
                      (g20 > 50) & (g > 30) & (d1 < -th)))
        rules.append((f"前期涨>50%+Top3+次日跌>{th*100:.0f}%",
                      (g20 > 50) & (gr <= 3) & (d1 < -th)))
        rules.append((f"大盘强+Top3+次日涨>{th*100:.0f}%",
                      spy_ma & (gr <= 3) & (d1 > th)))
        rules.append((f"大盘弱+前期涨>50%+次日跌>{th*100:.0f}%",
                      (~spy_ma) & (g20 > 50) & (d1 < -th)))

    # 3日确认（T+1~T+3累计）
    df["fwd_3d_cum"] = df["fwd_3d"]
    for th in [0.10, 0.15, 0.20]:
        rules.append((f"Top3+3日累计涨>{th*100:.0f}%", (gr <= 3) & (df["fwd_3d"] > th)))
        rules.append((f"Top3+3日累计跌<-{th*100:.0f}%", (gr <= 3) & (df["fwd_3d"] < -th)))

    # 路径标签：5日内已涨/跌多少（用 path_up/down 作信号日已知 proxy 不行，这些是 forward）
    # 用 fwd_1d + fwd_3d 组合
    for d1_th, d3_th in [(0.05, 0.10), (0.07, 0.12), (0.10, 0.15), (0.10, 0.20)]:
        rules.append((f"次日涨>{d1_th*100:.0f}%+3日累计涨>{d3_th*100:.0f}%",
                      (d1 > d1_th) & (df["fwd_3d"] > d3_th)))
        rules.append((f"Top3+次日涨>{d1_th*100:.0f}%+3日累计涨>{d3_th*100:.0f}%",
                      (gr <= 3) & (d1 > d1_th) & (df["fwd_3d"] > d3_th)))

    # 去重
    seen = set()
    unique = []
    for label, mask in rules:
        key = tuple(mask.fillna(False).values)
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, mask.fillna(False)))
    return unique


def run() -> dict:
    df = load()
    base_surge = float((df["fwd_5d"] >= SURGE_TH).mean())
    base_drop = float((df["fwd_5d"] <= DROP_TH).mean())

    rules = build_rules(df)
    surge_hits = []
    drop_hits = []
    for label, mask in rules:
        r = eval_rule(df, mask, label)
        if not r:
            continue
        if r["surge_rate"] >= TARGET:
            surge_hits.append(r)
        if r["drop_rate"] >= TARGET:
            drop_hits.append(r)

    surge_hits.sort(key=lambda x: (x["surge_rate"], x["n"], x["avg_fwd_5d_pct"]), reverse=True)
    drop_hits.sort(key=lambda x: (x["drop_rate"], x["n"], -x["avg_fwd_5d_pct"]), reverse=True)

    # 若80%不够，找最接近的 top rules 并标注
    all_rules = [eval_rule(df, m, l) for l, m in rules]
    all_rules = [r for r in all_rules if r]
    near_surge = sorted(all_rules, key=lambda x: x["surge_rate"], reverse=True)[:15]
    near_drop = sorted(all_rules, key=lambda x: x["drop_rate"], reverse=True)[:15]

    doc = {
        "target_rate": TARGET,
        "min_sample_n": MIN_N,
        "base_rates": {"surge": round(base_surge, 3), "drop": round(base_drop, 3)},
        "surge_rules_80plus": surge_hits,
        "drop_rules_80plus": drop_hits,
        "best_near_80_surge": near_surge,
        "best_near_80_drop": near_drop,
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def print_report(doc: dict) -> None:
    t = doc["target_rate"]
    print("=" * 76)
    print(f"目标概率 ≥{t:.0%}  |  最少样本 n≥{doc['min_sample_n']}")
    print(f"基准：继续暴涨(5日≥10%) {doc['base_rates']['surge']:.1%}  |  大幅回调(5日≤-10%) {doc['base_rates']['drop']:.1%}")
    print("=" * 76)

    sr = doc["surge_rules_80plus"]
    dr = doc["drop_rules_80plus"]

    print(f"\n【继续暴涨 ≥{t:.0%}】共 {len(sr)} 条规则")
    if sr:
        for r in sr:
            print(f"  {r['rule']}")
            print(f"    n={r['n']}  暴涨率={r['surge_rate']:.1%}  回调率={r['drop_rate']:.1%}  "
                  f"5日均{r['avg_fwd_5d_pct']:+.1f}%  中位{r['median_fwd_5d_pct']:+.1f}%")
    else:
        print("  未找到≥80%规则，最接近的：")
        for r in doc["best_near_80_surge"][:8]:
            print(f"  {r['rule']}: n={r['n']}  暴涨率={r['surge_rate']:.1%}  5日均{r['avg_fwd_5d_pct']:+.1f}%")

    print(f"\n【大幅回调 ≥{t:.0%}】共 {len(dr)} 条规则")
    if dr:
        for r in dr:
            print(f"  {r['rule']}")
            print(f"    n={r['n']}  回调率={r['drop_rate']:.1%}  暴涨率={r['surge_rate']:.1%}  "
                  f"5日均{r['avg_fwd_5d_pct']:+.1f}%  中位{r['median_fwd_5d_pct']:+.1f}%")
    else:
        print("  未找到≥80%规则，最接近的：")
        for r in doc["best_near_80_drop"][:8]:
            print(f"  {r['rule']}: n={r['n']}  回调率={r['drop_rate']:.1%}  5日均{r['avg_fwd_5d_pct']:+.1f}%")

    print("=" * 76)
    print(f"→ {OUT}")


if __name__ == "__main__":
    print_report(run())
