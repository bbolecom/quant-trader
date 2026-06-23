#!/usr/bin/env python3
"""涨幅榜 Top100 + 成交额门槛 → 后续走势统计。

条件（每个交易日）：
  1. 当日成交额 ≥ min_dvol_m 百万美元（默认 100M = 1亿美元）
  2. 在候选池内按 1 日涨幅排名，取前 top_n（默认 100）

输出：次日/5日/20日收益分布、路径涨跌、分档规律、IS/OOS。

用法：
    python research/gainer_top100_followthrough.py
    python research/gainer_top100_followthrough.py --from-panel
    python research/gainer_top100_followthrough.py --min-dvol-m 100 --top-n 100
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.move_pattern import compute_forward_path_labels
from research.gainer_daily_backtest import (
    GAINER_MOMENTUM,
    LIQUID100,
    build_factor_panels,
    fetch_gainer_data_yahoo,
    load_gainer_pool,
)

TRAIN_END = "2023-12-31"
EVENTS_CSV = ROOT / "research" / "gainer_top100_events.csv"
RESULT_JSON = ROOT / "research" / "gainer_top100_followthrough.json"
HIGHWIN_PANEL = ROOT / "research" / "move_pattern_highwin_panel.csv"


def _spy_series(spy: pd.DataFrame) -> pd.Series:
    if spy is None or spy.empty:
        return pd.Series(dtype=float)
    c = spy["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.astype(float)


def _attach_forward(data: dict[str, pd.DataFrame], panel: pd.DataFrame) -> pd.DataFrame:
    """为面板附加 fwd 收盘收益与 5 日路径标签（按 ticker 向量化）。"""
    frames: list[pd.DataFrame] = []
    for tk, df in data.items():
        if df is None or df.empty:
            continue
        close = df["Close"].astype(float)
        close.index = pd.to_datetime(df.index)
        paths = compute_forward_path_labels(
            df, horizon=5, up_threshold=0.02, down_threshold=0.02,
        )
        paths.index = pd.to_datetime(paths.index)
        tmp = pd.DataFrame({"代码": tk}, index=paths.index)
        for shift, col in [(1, "fwd_1d"), (3, "fwd_3d"), (5, "fwd_5d"), (10, "fwd_10d"), (20, "fwd_20d")]:
            tmp[col] = close.shift(-shift) / close - 1.0
        tmp["path_up_5d"] = paths["path_up_5d"]
        tmp["path_down_5d"] = paths["path_down_5d"]
        tmp = tmp.reset_index()
        if "index" in tmp.columns:
            tmp = tmp.rename(columns={"index": "日期"})
        elif tmp.columns[0] != "日期":
            tmp = tmp.rename(columns={tmp.columns[0]: "日期"})
        frames.append(tmp)

    if not frames:
        return panel
    fwd = pd.concat(frames, ignore_index=True)
    fwd["日期"] = pd.to_datetime(fwd["日期"])
    out = panel.copy()
    out["日期"] = pd.to_datetime(out["日期"])
    drop_cols = [c for c in out.columns if c.startswith("fwd_") or c.startswith("path_")]
    if drop_cols:
        out = out.drop(columns=drop_cols, errors="ignore")
    out = out.merge(fwd, on=["日期", "代码"], how="left")
    out["gain_rank"] = (
        out.groupby("日期")["涨幅%"]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )
    return out


def select_top100_events(
    panel: pd.DataFrame,
    *,
    min_dvol_m: float = 100.0,
    top_n: int = 100,
    min_gain_pct: float = 0.0,
) -> pd.DataFrame:
    """筛选每日涨幅榜 TopN（成交额门槛内）。"""
    d = panel.copy()
    d["dvol_m"] = pd.to_numeric(d["成交额USD"], errors="coerce") / 1e6
    d = d.dropna(subset=["涨幅%", "dvol_m", "gain_rank"])
    d = d[(d["dvol_m"] >= min_dvol_m) & (d["涨幅%"] >= min_gain_pct)]
    d = d[d["gain_rank"] <= top_n].copy()
    d["日期"] = pd.to_datetime(d["日期"]).dt.strftime("%Y-%m-%d")
    return d


def _stats(sub: pd.DataFrame, col: str) -> dict:
    s = pd.to_numeric(sub[col], errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean_pct": round(float(s.mean()) * 100, 3),
        "median_pct": round(float(s.median()) * 100, 3),
        "win_rate": round(float((s > 0).mean()), 4),
        "p25_pct": round(float(s.quantile(0.25)) * 100, 3),
        "p75_pct": round(float(s.quantile(0.75)) * 100, 3),
    }


def _path_stats(sub: pd.DataFrame, col: str, th: float) -> dict:
    s = pd.to_numeric(sub[col], errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    if col == "path_up_5d":
        hit = s >= th
    else:
        hit = s <= -th
    return {"n": int(len(s)), "hit_rate": round(float(hit.mean()), 4), "threshold_pct": th * 100}


def analyze_events(events: pd.DataFrame) -> dict:
    events = events.copy()
    events["dt"] = pd.to_datetime(events["日期"])
    is_df = events[events["dt"] <= pd.Timestamp(TRAIN_END)]
    oos_df = events[events["dt"] > pd.Timestamp(TRAIN_END)]

    def block(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {}
        return {
            "events": int(len(sub)),
            "unique_days": int(sub["日期"].nunique()),
            "unique_tickers": int(sub["代码"].nunique()),
            "avg_gain_pct": round(float(sub["涨幅%"].mean()), 2),
            "median_gain_pct": round(float(sub["涨幅%"].median()), 2),
            "avg_dvol_m": round(float(sub["dvol_m"].mean()), 1),
            "avg_vol_ratio": round(float(sub["量比"].mean()), 2) if sub["量比"].notna().any() else None,
            "fwd_1d": _stats(sub, "fwd_1d"),
            "fwd_3d": _stats(sub, "fwd_3d"),
            "fwd_5d": _stats(sub, "fwd_5d"),
            "fwd_10d": _stats(sub, "fwd_10d"),
            "fwd_20d": _stats(sub, "fwd_20d"),
            "path_up_5d_ge2pct": _path_stats(sub, "path_up_5d", 0.02),
            "path_down_5d_ge2pct": _path_stats(sub, "path_down_5d", 0.02),
        }

    # 涨幅分档
    buckets = []
    for lo, hi, label in [
        (2, 5, "2~5% 温和涨"),
        (5, 10, "5~10% 中涨"),
        (10, 15, "10~15% 大涨"),
        (15, 999, ">15% 暴涨"),
    ]:
        sub = events[(events["涨幅%"] >= lo) & (events["涨幅%"] < hi)]
        if len(sub) < 30:
            continue
        buckets.append({
            "bucket": label,
            "n": int(len(sub)),
            "fwd_1d": _stats(sub, "fwd_1d"),
            "fwd_5d": _stats(sub, "fwd_5d"),
            "fwd_20d": _stats(sub, "fwd_20d"),
        })

    # 量比分档
    vol_buckets = []
    for lo, hi, label in [
        (0, 1.5, "量比<1.5"),
        (1.5, 2.5, "量比1.5~2.5"),
        (2.5, 999, "量比>2.5"),
    ]:
        sub = events[(events["量比"] >= lo) & (events["量比"] < hi)]
        if len(sub) < 30:
            continue
        vol_buckets.append({
            "bucket": label,
            "n": int(len(sub)),
            "fwd_1d": _stats(sub, "fwd_1d"),
            "fwd_5d": _stats(sub, "fwd_5d"),
        })

    return {
        "all": block(events),
        "is_2019_2023": block(is_df),
        "oos_2024_plus": block(oos_df),
        "by_gain_bucket": buckets,
        "by_vol_ratio_bucket": vol_buckets,
    }


def build_panel_from_cache() -> pd.DataFrame:
    if not HIGHWIN_PANEL.exists():
        return pd.DataFrame()
    panel = pd.read_csv(HIGHWIN_PANEL, encoding="utf-8-sig")
    panel["日期"] = pd.to_datetime(panel["日期"])
    panel["gain_rank"] = (
        panel.groupby("日期")["涨幅%"]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )
    return panel


def run(
    *,
    start: str = "2019-01-01",
    end: str | None = None,
    min_dvol_m: float = 100.0,
    top_n: int = 100,
    from_panel: bool = False,
    pool: str = "broad",
) -> dict:
    end = end or date.today().isoformat()

    if from_panel and HIGHWIN_PANEL.exists():
        print(f"从缓存面板加载 {HIGHWIN_PANEL.name} …")
        panel = build_panel_from_cache()
        # 缓存面板仅有 fwd_1d，需补全路径/远期 — 仍拉行情
        tickers = sorted(panel["代码"].astype(str).unique())
        data, spy = fetch_gainer_data_yahoo(tickers, start, end)
    else:
        tickers = load_gainer_pool(pool)
        tickers = list(dict.fromkeys(tickers + GAINER_MOMENTUM + LIQUID100))
        print(f"候选池 {len(tickers)} 只 · {start} ~ {end}")
        data, spy = fetch_gainer_data_yahoo(tickers, start, end)
        spy_close = _spy_series(spy)
        panel = build_factor_panels(data, spy_close)
        if panel.empty:
            return {"error": "面板为空"}

    print(f"附加远期标签 …")
    panel = _attach_forward(data, panel)
    events = select_top100_events(panel, min_dvol_m=min_dvol_m, top_n=top_n)
    events.to_csv(EVENTS_CSV, index=False, encoding="utf-8-sig")
    print(f"事件 {len(events)} 条 → {EVENTS_CSV}")

    stats = analyze_events(events)
    doc = {
        "updated": date.today().isoformat(),
        "method": "真实OHLCV · 日涨幅排名 · 非BS",
        "universe_note": (
            f"候选池≈{len(tickers)}只（Yahoo多榜+指数成分+动量池），"
            f"非NYSE/NASDAQ全量；池内按日涨幅取Top{top_n}"
        ),
        "filters": {
            "min_dvol_m_usd": min_dvol_m,
            "top_n_by_gain": top_n,
            "min_gain_pct": 0.0,
        },
        "train_end": TRAIN_END,
        "stats": stats,
    }
    RESULT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def _print_report(doc: dict) -> None:
    f = doc.get("filters", {})
    print("\n" + "=" * 72)
    print(f"涨幅榜 Top{f.get('top_n_by_gain', 100)} · 成交额≥${f.get('min_dvol_m_usd', 100):.0f}M → 后续走势")
    print("=" * 72)
    print(doc.get("universe_note", ""))

    for key, label in [
        ("all", "全样本"),
        ("is_2019_2023", "样本内 2019–2023"),
        ("oos_2024_plus", "样本外 2024+"),
    ]:
        b = doc.get("stats", {}).get(key, {})
        if not b:
            continue
        print(f"\n【{label}】事件 {b.get('events', 0)} · {b.get('unique_days', 0)} 交易日 · "
              f"均涨幅 {b.get('avg_gain_pct')}% · 均成交额 ${b.get('avg_dvol_m')}M")
        for hz in ("fwd_1d", "fwd_3d", "fwd_5d", "fwd_10d", "fwd_20d"):
            s = b.get(hz, {})
            if not s or not s.get("n"):
                continue
            tag = {"fwd_1d": "次日", "fwd_3d": "3日", "fwd_5d": "5日", "fwd_10d": "10日", "fwd_20d": "20日"}[hz]
            print(f"  {tag}: 均{s['mean_pct']:+.2f}%  中位{s['median_pct']:+.2f}%  "
                  f"胜率{s['win_rate']:.1%}  (n={s['n']})")
        pu = b.get("path_up_5d_ge2pct", {})
        pd_ = b.get("path_down_5d_ge2pct", {})
        if pu.get("n"):
            print(f"  5日路径涨≥2%: {pu['hit_rate']:.1%} (n={pu['n']})")
        if pd_.get("n"):
            print(f"  5日路径跌≥2%: {pd_['hit_rate']:.1%} (n={pd_['n']})")

    buckets = doc.get("stats", {}).get("by_gain_bucket") or []
    if buckets:
        print("\n【按当日涨幅分档 · 5日后续】")
        for row in buckets:
            f5 = row.get("fwd_5d", {})
            f1 = row.get("fwd_1d", {})
            print(f"  {row['bucket']:12s} n={row['n']:5d}  "
                  f"次日均{f1.get('mean_pct', 0):+.2f}%/{f1.get('win_rate', 0):.0%}  "
                  f"5日均{f5.get('mean_pct', 0):+.2f}%/{f5.get('win_rate', 0):.0%}")

    vb = doc.get("stats", {}).get("by_vol_ratio_bucket") or []
    if vb:
        print("\n【按量比分档 · 次日】")
        for row in vb:
            f1 = row.get("fwd_1d", {})
            print(f"  {row['bucket']:14s} n={row['n']:5d}  "
                  f"次日均{f1.get('mean_pct', 0):+.2f}%  胜率{f1.get('win_rate', 0):.0%}")
    print("=" * 72 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="涨幅榜Top100+成交额→后续走势")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--min-dvol-m", type=float, default=100.0, help="成交额下限(百万USD)")
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--from-panel", action="store_true", help="用已缓存 highwin 面板加速")
    ap.add_argument("--pool", default="broad", choices=["broad", "momentum", "liquid100"])
    args = ap.parse_args()
    doc = run(
        start=args.start,
        end=args.end,
        min_dvol_m=args.min_dvol_m,
        top_n=args.top_n,
        from_panel=args.from_panel,
        pool=args.pool,
    )
    if doc.get("error"):
        print(doc["error"])
        sys.exit(1)
    _print_report(doc)
    print(f"→ {RESULT_JSON}")


if __name__ == "__main__":
    main()
