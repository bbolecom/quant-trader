#!/usr/bin/env python3
"""放宽版「安飞士做空案例」扫描——永远给排序名单。

严格版(blowoff_top_scan / short_candidates_scan)只在出现教科书级
「暴涨+天量+收弱见顶」时才报票，常常空仓。本脚本对全市场涨幅+活跃榜
候选统一算「见顶过热分」，强制按分数降序输出 Top N，方便人工盯盘。

过热分 = 前期涨幅(runup) + 贴顶程度 + 量能放大 + 收盘转弱 + 破位扣分项
分越高 = 越像抛物线顶 / 越值得作为做空候选观察。
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.providers import DataConfig, get_provider, reset_provider_cache
from research.short_candidates_scan import fetch_short_universe

MIN_DVOL_M = 20.0
MIN_PRICE = 3.0
MIN_RUNUP = 0.20  # 近 5/10/20 日至少涨过 20% 才算「涨过」


def _clv(o, h, l, c):
    rng = h - l
    return 0.5 if rng <= 0 else ((c - l) - (h - c)) / rng


def analyze(t: str, df: pd.DataFrame) -> dict | None:
    if df is None or df.empty or len(df) < 25:
        return None
    o, h, l, c, v = (df[k].astype(float) for k in ["Open", "High", "Low", "Close", "Volume"])
    last = c.iloc[-1]
    if last < MIN_PRICE:
        return None
    vma = v.rolling(20).mean().iloc[-1]
    if not np.isfinite(vma) or vma <= 0:
        return None
    dvol_m = last * v.iloc[-1] / 1e6
    if dvol_m < MIN_DVOL_M:
        return None

    vol_x = v.iloc[-1] / vma
    chg = last / c.iloc[-2] - 1
    clv_now = _clv(o.iloc[-1], h.iloc[-1], l.iloc[-1], c.iloc[-1])
    ret_5d = last / c.iloc[-6] - 1 if len(c) > 6 else np.nan
    ret_10d = last / c.iloc[-11] - 1 if len(c) > 11 else np.nan
    ret_20d = last / c.iloc[-21] - 1 if len(c) > 21 else np.nan
    runup = np.nanmax([ret_5d, ret_10d, ret_20d])
    if not np.isfinite(runup) or runup < MIN_RUNUP:
        return None

    hi20 = h.iloc[-21:].max()
    off_hi20 = last / hi20 - 1
    ma10 = c.rolling(10).mean().iloc[-1]
    ma20 = c.rolling(20).mean().iloc[-1]
    below_ma10 = last < ma10
    # 离布林上轨距离（20,2）
    bb_mid = c.rolling(20).mean().iloc[-1]
    bb_std = c.rolling(20).std().iloc[-1]
    bb_up = bb_mid + 2 * bb_std
    above_bb = last / bb_up - 1 if np.isfinite(bb_up) and bb_up > 0 else 0.0

    # 过热分
    score = 0.0
    score += min(runup, 2.0) * 50           # 涨幅越大越过热
    score += max(0.0, above_bb) * 200        # 站上布林上轨越多越极端
    score += max(0.0, vol_x - 1.0) * 8       # 放量
    score += (0.5 - clv_now) * 40            # 收盘转弱加分
    if off_hi20 <= -0.05:                     # 已从高点回撤 = 右侧破位，更可空
        score += (-off_hi20) * 80
    if below_ma10:
        score += 15

    if off_hi20 >= -0.03 and clv_now <= 0.5 and vol_x >= 1.5:
        kind = "A贴顶天量收弱"
    elif below_ma10 and off_hi20 <= -0.08:
        kind = "B已破位下行"
    elif off_hi20 >= -0.05 and clv_now <= 0.4:
        kind = "C冲高回落派发"
    elif off_hi20 >= -0.03:
        kind = "D贴顶过热(强)"
    else:
        kind = "E回撤中"

    return {
        "代码": t,
        "现价": round(last, 2),
        "当日%": round(chg * 100, 1),
        "5日%": round(ret_5d * 100, 1) if np.isfinite(ret_5d) else None,
        "20日%": round(ret_20d * 100, 1) if np.isfinite(ret_20d) else None,
        "量倍": round(vol_x, 2),
        "收强": round(clv_now, 2),
        "距20高%": round(off_hi20 * 100, 1),
        "超布林%": round(above_bb * 100, 1),
        "破MA10": "是" if below_ma10 else "否",
        "额M": round(dvol_m, 0),
        "类型": kind,
        "过热分": round(score, 0),
    }


def run(count: int = 250, top: int = 30) -> pd.DataFrame:
    print(f"① 拉涨幅+跌幅+活跃榜 (count={count}) …")
    cand = fetch_short_universe(count=count)
    if not cand:
        return pd.DataFrame()
    print(f"② 候选 {len(cand)} 只，抓日线算过热分 …")
    reset_provider_cache()
    y = get_provider(DataConfig(provider="yahoo"))
    start = (date.today() - timedelta(days=70)).isoformat()
    end = date.today().isoformat()
    batch = y.fetch_batch(cand, start, end)
    rows = []
    for t in cand:
        d = batch.get(t)
        try:
            r = analyze(t, d) if d is not None else None
        except Exception:
            r = None
        if r:
            rows.append(r)
    if not rows:
        print("\n无任何近期涨过 20% 的候选")
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("过热分", ascending=False)
    return df.head(top)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=250)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    res = run(count=args.count, top=args.top)
    if not res.empty:
        pd.set_option("display.width", 240)
        pd.set_option("display.max_columns", 30)
        print("\n=== 安飞士式做空候选（过热分降序，放宽版）===")
        print(res.to_string(index=False))
        print("\n类型: A贴顶天量收弱(最像安飞士) | B已破位(最稳可空) | C冲高回落派发 | D贴顶过热 | E回撤中")
        print("纪律: 优先 A/B；A 等次日破位再空；止损放新高上方；裸空风险高，优先卖Call价差。")
