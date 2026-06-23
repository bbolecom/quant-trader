#!/usr/bin/env python3
"""暴涨扫描信号 · 历史回测胜率。

对 A 突破 / B 延续 / C 前兆 三类信号，统计信号日收盘买入后的
1/3/5/10/20 日收益与胜率（IS/OOS 分段）。

用法：
    python research/surge_scan_backtest.py
    python research/surge_scan_backtest.py --years 5
    python research/surge_scan_backtest.py --quick
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

from quant.surge_scan import (
    SURGE_LABELS,
    SurgeScanConfig,
    classify_surge_row,
    compute_surge_features,
)
from research.gainer_daily_backtest import GAINER_MOMENTUM, LIQUID100, fetch_gainer_data_yahoo

TRAIN_END = "2023-12-31"
EVENTS_CSV = ROOT / "research" / "surge_scan_events.csv"
RESULT_JSON = ROOT / "research" / "surge_scan_backtest.json"

FWD_COLS = ["fwd_1d", "fwd_3d", "fwd_5d", "fwd_10d", "fwd_20d"]
FWD_SHIFTS = [1, 3, 5, 10, 20]


def collect_events(
    data: dict[str, pd.DataFrame],
    cfg: SurgeScanConfig,
    *,
    event_start: str,
    event_end: str,
) -> pd.DataFrame:
    """扫描全样本历史暴涨事件。"""
    rows: list[dict] = []
    t0 = pd.Timestamp(event_start)
    t1 = pd.Timestamp(event_end)

    for ticker, df in data.items():
        if df is None or len(df) < 80:
            continue
        feats = compute_surge_features(df, boll_window=cfg.boll_window)
        merged = df.join(feats)
        merged = merged.loc[(merged.index >= t0) & (merged.index <= t1)]
        close = df["Close"].astype(float)

        for ts, row in merged.iterrows():
            kind, score, note = classify_surge_row(row, cfg)
            if kind is None:
                continue
            idx = close.index.get_loc(ts)
            rec: dict = {
                "代码": ticker.upper(),
                "日期": ts.strftime("%Y-%m-%d"),
                "类型": kind,
                "类型名": SURGE_LABELS[kind],
                "涨幅_pct": round(float(row["涨幅_pct"]), 2),
                "涨幅20d_pct": round(float(row["涨幅20d_pct"]), 2),
                "成交额M": round(float(row["成交额USD"]) / 1e6, 1),
                "量比": round(float(row["量比"]), 2),
                "收盘强度": round(float(row["收盘强度"]), 2),
                "创20日高": bool(row["创20日高"]),
                "WR": round(float(row["WR"]), 1),
                "综合分": round(score, 2),
                "说明": note,
            }
            px = float(close.iloc[idx])
            for shift, col in zip(FWD_SHIFTS, FWD_COLS):
                if idx + shift < len(close):
                    rec[col] = float(close.iloc[idx + shift] / px - 1.0)
                else:
                    rec[col] = np.nan
            # 5 日内最高涨幅（前兆类尤其有用）
            if idx + 5 < len(close):
                rec["max_up_5d"] = float(close.iloc[idx + 1 : idx + 6].max() / px - 1.0)
            else:
                rec["max_up_5d"] = np.nan
            rows.append(rec)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["dt"] = pd.to_datetime(out["日期"])
    return out.sort_values(["日期", "代码"]).reset_index(drop=True)


def _ret_stats(sub: pd.DataFrame, col: str) -> dict:
    s = pd.to_numeric(sub[col], errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean_pct": round(float(s.mean()) * 100, 2),
        "median_pct": round(float(s.median()) * 100, 2),
        "win_rate": round(float((s > 0).mean()), 4),
        "p25_pct": round(float(s.quantile(0.25)) * 100, 2),
        "p75_pct": round(float(s.quantile(0.75)) * 100, 2),
    }


def _rate_stats(sub: pd.DataFrame, col: str, th: float, *, ge: bool = True) -> dict:
    s = pd.to_numeric(sub[col], errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    hit = s >= th if ge else s <= th
    return {
        "n": int(len(s)),
        "rate": round(float(hit.mean()), 4),
        "threshold_pct": round(th * 100, 1),
    }


def summarize_events(events: pd.DataFrame) -> dict:
    """按类型与 IS/OOS 汇总胜率。"""
    if events.empty:
        return {"total_events": 0}

    def block(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {"events": 0}
        return {
            "events": int(len(sub)),
            "unique_tickers": int(sub["代码"].nunique()),
            "unique_days": int(sub["日期"].nunique()),
            "avg_signal_gain_pct": round(float(sub["涨幅_pct"].mean()), 2),
            "fwd_1d": _ret_stats(sub, "fwd_1d"),
            "fwd_3d": _ret_stats(sub, "fwd_3d"),
            "fwd_5d": _ret_stats(sub, "fwd_5d"),
            "fwd_10d": _ret_stats(sub, "fwd_10d"),
            "fwd_20d": _ret_stats(sub, "fwd_20d"),
            "fwd_5d_ge_10pct": _rate_stats(sub, "fwd_5d", 0.10),
            "fwd_5d_le_neg10pct": _rate_stats(sub, "fwd_5d", -0.10, ge=False),
            "max_up_5d_ge_7pct": _rate_stats(sub, "max_up_5d", 0.07),
        }

    by_type: dict[str, dict] = {}
    for kind, label in SURGE_LABELS.items():
        sub = events[events["类型"] == kind]
        by_type[kind] = {
            "label": label,
            "all": block(sub),
            "is": block(sub[sub["dt"] <= pd.Timestamp(TRAIN_END)]),
            "oos": block(sub[sub["dt"] > pd.Timestamp(TRAIN_END)]),
        }

    is_all = events[events["dt"] <= pd.Timestamp(TRAIN_END)]
    oos_all = events[events["dt"] > pd.Timestamp(TRAIN_END)]

    return {
        "total_events": int(len(events)),
        "train_end": TRAIN_END,
        "all": block(events),
        "is": block(is_all),
        "oos": block(oos_all),
        "by_type": by_type,
        "sub_rules_oos": _sub_rules(events[events["dt"] > pd.Timestamp(TRAIN_END)]),
    }


def _sub_rules(oos: pd.DataFrame) -> list[dict]:
    """OOS 子规则胜率（5 日持有）。"""
    if oos.empty:
        return []
    specs = [
        ("A突破·全样本", oos["类型"] == "breakout"),
        ("A突破·创20日高", (oos["类型"] == "breakout") & oos["创20日高"].astype(bool)),
        ("A突破·早期(20d<25%)", (oos["类型"] == "breakout") & (oos["涨幅20d_pct"] < 25)),
        ("B延续·全样本", oos["类型"] == "continuation"),
        ("B延续·20d涨≥50%", (oos["类型"] == "continuation") & (oos["涨幅20d_pct"] >= 50)),
        ("C前兆·全样本", oos["类型"] == "precursor"),
    ]
    out: list[dict] = []
    for name, mask in specs:
        sub = oos.loc[mask]
        f5 = _ret_stats(sub, "fwd_5d")
        ge10 = _rate_stats(sub, "fwd_5d", 0.10)
        le10 = _rate_stats(sub, "fwd_5d", -0.10, ge=False)
        if f5.get("n", 0) == 0:
            continue
        out.append({
            "rule": name,
            "n": f5["n"],
            "fwd_1d_win": _ret_stats(sub, "fwd_1d").get("win_rate"),
            "fwd_5d_win": f5.get("win_rate"),
            "fwd_5d_mean_pct": f5.get("mean_pct"),
            "fwd_5d_ge_10pct": ge10.get("rate"),
            "fwd_5d_le_neg10pct": le10.get("rate"),
        })
    return out


def run_backtest(
    *,
    years: int = 3,
    quick: bool = False,
    min_dvol_m: float = 50.0,
    end: str | None = None,
) -> dict:
    end_d = end or date.today().isoformat()
    start_d = (date.fromisoformat(end_d) - timedelta(days=years * 365 + 120)).isoformat()
    # 事件统计区间（留 20 日 forward）
    event_end = (date.fromisoformat(end_d) - timedelta(days=25)).isoformat()

    tickers = sorted(set(LIQUID100 if quick else list(dict.fromkeys(LIQUID100 + GAINER_MOMENTUM))))
    cfg = SurgeScanConfig(min_dvol_m=min_dvol_m)

    data, spy = fetch_gainer_data_yahoo(tickers, start_d, end_d)
    events = collect_events(data, cfg, event_start=start_d, event_end=event_end)
    summary = summarize_events(events)

    spy_close = spy["Close"].astype(float)
    spy_close.index = pd.to_datetime(spy.index)
    spy_last = float(spy_close.iloc[-1]) if len(spy_close) else None
    spy_ma20 = float(spy_close.tail(20).mean()) if len(spy_close) >= 20 else None

    doc = {
        "generated": date.today().isoformat(),
        "period": {"start": start_d, "end": end_d, "event_end": event_end, "years": years},
        "universe": {"tickers": len(tickers), "quick": quick},
        "config": {
            "min_dvol_m": min_dvol_m,
            "breakout_gain": f"{cfg.breakout_min_gain_pct}~{cfg.breakout_max_gain_pct}%",
            "continuation_min_gain_20d": f"{cfg.continuation_min_gain_20d_pct}%",
        },
        "market": {
            "SPY": spy_last,
            "MA20": spy_ma20,
            "站上MA20": spy_last > spy_ma20 if spy_ma20 and spy_last else None,
        },
        "summary": summary,
    }
    return doc, events


def print_report(doc: dict) -> None:
    s = doc["summary"]
    if s.get("total_events", 0) == 0:
        print("无事件")
        return

    period = doc["period"]
    print(f"暴涨扫描回测 · {period['start']} ~ {period['event_end']} · {s['total_events']} 事件")
    print(f"股票池 {doc['universe']['tickers']} 只 · IS 截止 {s['train_end']}")
    print()

    def line(tag: str, blk: dict) -> None:
        if blk.get("events", 0) == 0:
            print(f"  {tag}: 无样本")
            return
        f1 = blk["fwd_1d"]
        f5 = blk["fwd_5d"]
        s10 = blk["fwd_5d_ge_10pct"]
        d10 = blk["fwd_5d_le_neg10pct"]
        print(
            f"  {tag}: n={blk['events']} · "
            f"1日胜率{f1['win_rate']:.1%} 均{f1['mean_pct']:+.2f}% · "
            f"5日胜率{f5['win_rate']:.1%} 均{f5['mean_pct']:+.2f}% · "
            f"5日涨≥10% {s10['rate']:.1%} · 5日跌≤10% {d10['rate']:.1%}"
        )

    print("【全样本】")
    line("合计", s["all"])
    line(f"IS ≤{s['train_end']}", s["is"])
    line(f"OOS >{s['train_end']}", s["oos"])
    print()

    for kind, info in s["by_type"].items():
        blk = info["all"]
        if blk.get("events", 0) == 0:
            continue
        print(f"【{info['label']}】")
        line("全样本", blk)
        line("IS", info["is"])
        line("OOS", info["oos"])
        if kind == "precursor":
            m7 = blk.get("max_up_5d_ge_7pct", {})
            if m7.get("n", 0):
                print(f"  5日内冲高≥7%: {m7['rate']:.1%} (n={m7['n']})")
        print()

    subs = s.get("sub_rules_oos") or []
    if subs:
        print("【OOS 子规则 · 5日持有】")
        for r in subs:
            print(
                f"  {r['rule']:20s} n={r['n']:4d} · "
                f"1d={r['fwd_1d_win']:.1%} 5d={r['fwd_5d_win']:.1%} "
                f"均5d={r['fwd_5d_mean_pct']:+.1f}% · "
                f"涨≥10%={r['fwd_5d_ge_10pct']:.1%} 跌≤10%={r['fwd_5d_le_neg10pct']:.1%}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="暴涨扫描回测胜率")
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--min-dvol-m", type=float, default=50.0)
    parser.add_argument("--end", type=str, default="")
    args = parser.parse_args()

    doc, events = run_backtest(
        years=args.years,
        quick=args.quick,
        min_dvol_m=args.min_dvol_m,
        end=args.end or None,
    )

    EVENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not events.empty:
        events.drop(columns=["dt"], errors="ignore").to_csv(EVENTS_CSV, index=False, encoding="utf-8-sig")
    RESULT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print_report(doc)
    print(f"\n已写入 {RESULT_JSON.name} · {EVENTS_CSV.name}")


if __name__ == "__main__":
    main()
