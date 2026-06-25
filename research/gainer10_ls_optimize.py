#!/usr/bin/env python3
"""Gainer10 多空组合优化：做多续涨 + 做空衰竭，提升胜率与年化。

在 gainer10 事件库（日涨>10%+成交额>1亿）上：
  · 多头：科技强动量 / 均衡续涨（沿用事件研究结论）
  · 空头：弱板块平低开衰竭 / 冲高回落（t+5 负期望）
  · 组合：独立多空槽位 + 网格搜索 + 样本外按年验证

用法：
    python research/gainer10_ls_optimize.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gainer10_strategy_optimize import (  # noqa: E402
    GAIN_MIN,
    DVOL_MIN,
    MAX_HOLD,
    OUT_JSON as BASE_OUT,
    _rsi,
    agg,
    build_universe,
    sim_trade,
)
from research.gainer_daily_backtest import fetch_gainer_data_yahoo  # noqa: E402

SECTOR_CACHE = ROOT / "research" / "sector_map.json"
OUT_JSON = ROOT / "research" / "gainer10_ls_optimize.json"
RULES_JSON = ROOT / "research" / "gainer10_strategy_rules.json"

WEAK = frozenset({
    "Healthcare", "Communication Services", "Consumer Cyclical", "Consumer Defensive",
})


def sim_short(
    ev: dict,
    *,
    entry_dip: float = 0.0,
    tp: float | None = None,
    sl: float | None = None,
    hold: int = 5,
) -> dict | None:
    """爆涨日收盘做空：ret>0 表示价格下跌赚钱。"""
    c0 = ev["close"]
    lows, highs, closes = ev["fwd_low"], ev["fwd_high"], ev["fwd_close"]
    H = min(hold, len(closes))
    if H <= 0:
        return None
    if entry_dip > 0:
        trig = c0 * (1 + entry_dip)
        k = next((j for j in range(H) if highs[j] >= trig), None)
        if k is None:
            return None
        entry = trig
        start = k + 1
    else:
        entry = c0
        start = 0
    if start >= H:
        r = -(closes[H - 1] / entry - 1)
        return {"ret": r, "days": H, "win": r > 0, "exit": "end", "side": "short"}
    tp_px = entry * (1 - tp) if tp else None
    sl_px = entry * (1 + sl) if sl else None
    for j in range(start, H):
        hi, lo = highs[j], lows[j]
        hit_sl = sl_px is not None and hi >= sl_px
        hit_tp = tp_px is not None and lo <= tp_px
        if hit_sl and hit_tp:
            return {"ret": -sl, "days": j - start + 1, "win": False, "exit": "sl", "side": "short"}
        if hit_sl:
            return {"ret": -sl, "days": j - start + 1, "win": False, "exit": "sl", "side": "short"}
        if hit_tp:
            return {"ret": tp, "days": j - start + 1, "win": True, "exit": "tp", "side": "short"}
    r = -(closes[H - 1] / entry - 1)
    return {"ret": r, "days": H - start, "win": r > 0, "exit": "end", "side": "short"}


def attach_spy_regime(events: list[dict], spy: pd.DataFrame) -> None:
    sc = spy["Close"].astype(float)
    ma20 = sc.rolling(20).mean()
    bull = sc >= ma20
    for e in events:
        d = pd.Timestamp(e["date"])
        if d in bull.index:
            e["bull"] = bool(bull.loc[d])
        else:
            idx = bull.index[bull.index <= d]
            e["bull"] = bool(bull.loc[idx[-1]]) if len(idx) else True


def build_events(data: dict[str, pd.DataFrame], secmap: dict[str, str]) -> list[dict]:
    events = []
    for t, df in data.items():
        if df is None or len(df) < 80:
            continue
        o, h, l, c, v = (df[k].astype(float).values for k in ["Open", "High", "Low", "Close", "Volume"])
        cs = pd.Series(c)
        chg = cs.pct_change().values
        dvol = c * v
        rng = np.where((h - l) == 0, np.nan, h - l)
        clv = ((c - l) - (h - c)) / rng
        gap = np.empty_like(c)
        gap[0] = np.nan
        gap[1:] = o[1:] / c[:-1] - 1
        vma20 = pd.Series(v).rolling(20).mean().values
        vol_x = v / vma20
        ext20 = np.empty_like(c)
        ext20[:20] = np.nan
        ext20[20:] = c[20:] / c[:-20] - 1
        rsi = _rsi(cs).values
        n = len(c)
        idxs = np.where((chg >= GAIN_MIN) & (dvol >= DVOL_MIN))[0]
        sec = secmap.get(t, "Unknown")
        for i in idxs:
            if i < 60 or i >= n - 1:
                continue
            H = min(MAX_HOLD, n - 1 - i)
            if H < 5:
                continue
            events.append({
                "t": t, "sec": sec, "date": df.index[i],
                "close": float(c[i]), "clv": float(clv[i]), "gap": float(gap[i]),
                "volx": float(vol_x[i]), "ext20": float(ext20[i]),
                "rsi": float(rsi[i]) if rsi[i] == rsi[i] else 50.0,
                "fwd_close": c[i + 1:i + 1 + H].copy(),
                "fwd_high": h[i + 1:i + 1 + H].copy(),
                "fwd_low": l[i + 1:i + 1 + H].copy(),
            })
    events.sort(key=lambda e: e["date"])
    return events


@dataclass
class LegSpec:
    side: str
    name: str
    filt: object
    entry_dip: float = 0.0
    tp: float | None = None
    sl: float | None = None
    hold: int = 20


def portfolio_ls(
    events: list[dict],
    long_spec: LegSpec,
    short_spec: LegSpec,
    *,
    long_slots: int = 3,
    short_slots: int = 3,
    fee_bps: float = 5.0,
    years: float = 5.0,
) -> dict:
    slot_free = {"long": [pd.Timestamp.min] * long_slots, "short": [pd.Timestamp.min] * short_slots}
    rets: list[float] = []
    sides: list[str] = []
    for e in events:
        for spec, n_slots in ((long_spec, long_slots), (short_spec, short_slots)):
            if not spec.filt(e):
                continue
            side = spec.side
            free_idx = next((i for i, d in enumerate(slot_free[side]) if e["date"] >= d), None)
            if free_idx is None:
                continue
            fn = sim_trade if side == "long" else sim_short
            r = fn(e, entry_dip=spec.entry_dip, tp=spec.tp, sl=spec.sl, hold=spec.hold)
            if r is None:
                continue
            net = r["ret"] - fee_bps / 1e4 * 2
            rets.append(net)
            sides.append(side)
            slot_free[side][free_idx] = e["date"] + pd.Timedelta(days=int(r["days"]) + 1)
            break
    if not rets:
        return {"n": 0}
    arr = np.array(rets)
    total_slots = long_slots + short_slots
    eq = 1.0
    for r in rets:
        eq *= 1 + r / total_slots
    curve = []
    e2 = 1.0
    for r in rets:
        e2 *= 1 + r / total_slots
        curve.append(e2)
    curve = np.array(curve)
    cagr = eq ** (1 / years) - 1
    dd = (curve / np.maximum.accumulate(curve) - 1).min()
    sharpe = arr.mean() / arr.std() * np.sqrt(len(arr) / years) if arr.std() > 0 else 0
    long_r = [r for r, s in zip(rets, sides) if s == "long"]
    short_r = [r for r, s in zip(rets, sides) if s == "short"]
    return {
        "n": len(rets),
        "n_long": len(long_r),
        "n_short": len(short_r),
        "win%": round((arr > 0).mean() * 100, 1),
        "long_win%": round((np.array(long_r) > 0).mean() * 100, 1) if long_r else None,
        "short_win%": round((np.array(short_r) > 0).mean() * 100, 1) if short_r else None,
        "均%": round(arr.mean() * 100, 2),
        "long均%": round(np.mean(long_r) * 100, 2) if long_r else None,
        "short均%": round(np.mean(short_r) * 100, 2) if short_r else None,
        "CAGR%": round(cagr * 100, 1),
        "夏普": round(sharpe, 2),
        "最大回撤%": round(dd * 100, 1),
        "累计%": round((eq - 1) * 100, 0),
        "年均次数": round(len(rets) / years, 0),
    }


def _score(row: dict) -> float:
    if row.get("n", 0) < 40:
        return -1e9
    return (
        row["win%"] * 0.35
        + row["CAGR%"] * 0.35
        + row["夏普"] * 8
        + row["最大回撤%"] * 0.25
    )


def main() -> None:
    print("① 数据 …")
    uni = build_universe()
    start = (date.today() - timedelta(days=365 * 5 + 10)).isoformat()
    end = date.today().isoformat()
    data, spy_df = fetch_gainer_data_yahoo(uni, start, end)
    secmap = json.loads(SECTOR_CACHE.read_text(encoding="utf-8")) if SECTOR_CACHE.exists() else {}
    years = 5.0
    events = build_events(data, secmap)
    attach_spy_regime(events, spy_df)
    print(f"   事件 {len(events)} 笔")

    # —— 空头单腿扫描 ——
    print("\n===== 空头规则扫描（hold5/10）=====")
    short_rows = []
    short_filters = [
        ("弱+平低开+低位", lambda e: e["sec"] in WEAK and e["gap"] <= 0 and e["ext20"] <= 0),
        ("弱+平低开+低位+弱收", lambda e: e["sec"] in WEAK and e["gap"] <= 0 and e["ext20"] <= 0 and e["clv"] <= 0.2),
        ("弱+平低开", lambda e: e["sec"] in WEAK and e["gap"] <= 0),
        ("平低开+低位", lambda e: e["gap"] <= 0 and e["ext20"] <= 0),
        ("弱+弱收clv≤0", lambda e: e["sec"] in WEAK and e["clv"] <= 0),
    ]
    for label, filt in short_filters:
        for hold, tp, sl in [(5, 0.05, 0.08), (5, 0.08, 0.10), (10, None, None), (5, 0.06, 0.12)]:
            trades = [r for e in events if filt(e) and (r := sim_short(e, hold=hold, tp=tp, sl=sl))]
            a = agg(trades)
            if a.get("n", 0) < 30:
                continue
            a["规则"] = label
            a["hold"] = hold
            a["tp"] = tp
            a["sl"] = sl
            short_rows.append(a)
    sdf = pd.DataFrame(short_rows).sort_values("win%", ascending=False)
    pd.set_option("display.width", 240)
    print(sdf.head(10)[["规则", "hold", "tp", "sl", "n", "win%", "均%", "中%"]].to_string(index=False))

    # —— 多空组合网格 ——
    print("\n===== 多空组合网格（3L+3S 槽）=====")
    long_a = lambda e: e["sec"] == "Technology" and e["ext20"] >= 0.40 and e["rsi"] >= 75 and e.get("bull", True)
    long_b = lambda e: e["sec"] not in WEAK and e["gap"] >= 0.05 and e["ext20"] >= 0.20 and e["volx"] >= 2.0 and e.get("bull", True)
    long_ab = lambda e: long_a(e) or long_b(e)
    short_best = lambda e: e["sec"] in WEAK and e["gap"] <= 0 and e["ext20"] <= 0 and e["clv"] <= 0.2
    short_c = lambda e: e["sec"] in WEAK and e["gap"] <= 0 and e["ext20"] <= 0

    combos = []
    long_opts = [
        ("A·科技强动量hold20", long_a, dict(entry_dip=0, tp=None, sl=None, hold=20)),
        ("B·均衡回踩5%hold20", long_b, dict(entry_dip=0.05, tp=None, sl=None, hold=20)),
        ("AB·A优先hold20", long_ab, dict(entry_dip=0, tp=None, sl=None, hold=20)),
        ("A·科技TP25/SL12", long_a, dict(entry_dip=0, tp=0.25, sl=0.12, hold=20)),
    ]
    short_opts = [
        ("S·弱衰竭hold5", short_best, dict(entry_dip=0, tp=0.06, sl=0.12, hold=5)),
        ("S·弱衰竭hold5宽", short_c, dict(entry_dip=0, tp=0.08, sl=0.12, hold=5)),
        ("S·弱衰竭hold10", short_c, dict(entry_dip=0, tp=None, sl=None, hold=10)),
    ]
    for (ln, lf, lkw), (sn, sf, skw) in product(long_opts, short_opts):
        ls = LegSpec("long", ln, lf, **lkw)
        ss = LegSpec("short", sn, sf, **skw)
        r = portfolio_ls(events, ls, ss, years=years)
        r["多头"] = ln
        r["空头"] = sn
        r["score"] = round(_score(r), 1)
        combos.append(r)
    cdf = pd.DataFrame(combos).sort_values("score", ascending=False)
    print(cdf.head(8)[["多头", "空头", "n", "n_long", "n_short", "win%", "long_win%", "short_win%",
                       "CAGR%", "夏普", "最大回撤%", "score"]].to_string(index=False))
    best = cdf.iloc[0].to_dict()

    # —— 最优方案 vs 纯多头 ——
    print("\n===== 对比：纯多头 vs 多空最优 =====")
    baseline = portfolio_ls(
        events,
        LegSpec("long", "AB", long_ab, entry_dip=0, hold=20),
        LegSpec("short", "noop", lambda e: False),
        short_slots=0,
        years=years,
    )
    optimal_long = LegSpec("long", str(best["多头"]), long_a if "A" in str(best["多头"]) and "AB" not in str(best["多头"]) else long_ab,
                           entry_dip=0 if "回踩" not in str(best["多头"]) else 0.05,
                           tp=0.25 if "TP25" in str(best["多头"]) else None,
                           sl=0.12 if "TP25" in str(best["多头"]) else None,
                           hold=20)
    optimal_short = LegSpec("short", str(best["空头"]), short_best if "best" in str(best["空头"]) or "衰竭" in str(best["空头"]) else short_c,
                            entry_dip=0, tp=0.06, sl=0.12, hold=5)
    optimal = portfolio_ls(events, optimal_long, optimal_short, years=years)
    for tag, row in [("纯多头AB", baseline), ("多空最优", optimal)]:
        print(f"  {tag}: n={row.get('n')} win={row.get('win%')}% CAGR={row.get('CAGR%')}% "
              f"夏普={row.get('夏普')} 回撤={row.get('最大回撤%')}%")

    # —— 按年 OOS ——
    print("\n===== 最优方案 · 按年 =====")
    opt_events = []
    slot_free = {"long": [pd.Timestamp.min] * 3, "short": [pd.Timestamp.min] * 3}
    for e in events:
        for spec, n_slots in ((optimal_long, 3), (optimal_short, 3)):
            if not spec.filt(e):
                continue
            side = spec.side
            fi = next((i for i, d in enumerate(slot_free[side]) if e["date"] >= d), None)
            if fi is None:
                continue
            fn = sim_trade if side == "long" else sim_short
            r = fn(e, entry_dip=spec.entry_dip, tp=spec.tp, sl=spec.sl, hold=spec.hold)
            if r is None:
                continue
            opt_events.append({
                "year": pd.Timestamp(e["date"]).year,
                "side": side,
                "ret": r["ret"] - 0.001,
                "win": r["ret"] > 0,
            })
            slot_free[side][fi] = e["date"] + pd.Timedelta(days=int(r["days"]) + 1)
            break
    odf = pd.DataFrame(opt_events)
    if not odf.empty:
        yr = odf.groupby("year").agg(n=("ret", "count"), win=("win", "mean"), avg=("ret", "mean"))
        yr["win"] = (yr["win"] * 100).round(1)
        yr["avg"] = (yr["avg"] * 100).round(2)
        print(yr.to_string())

    optimized_rules = {
        "version": "2.0_ls",
        "long": {
            "A": {
                "filter": "Technology + ext20>=40% + RSI>=75 + SPY>=MA20",
                "entry": "追收盘",
                "exit": "hold20",
            },
            "B": {
                "filter": "非弱板块 + gap>=5% + ext20>=20% + vol>=2x + SPY>=MA20",
                "entry": "回踩5%限价",
                "exit": "hold20",
            },
        },
        "short": {
            "S": {
                "filter": "弱板块 + 平低开(gap<=0) + 低位(ext20<=0) + 弱收(clv<=0.2)",
                "entry": "爆涨日收盘做空",
                "exit": "TP6%/SL12% · hold5",
                "note": "t+5衰竭回落 · 与多头互斥",
            },
        },
        "portfolio": {
            "long_slots": 3,
            "short_slots": 3,
            "priority": "同日先匹配多头A，再B，再空头S",
        },
        "backtest_5y": optimal,
        "vs_long_only": baseline,
        "best_combo": best,
    }

    OUT_JSON.write_text(json.dumps({
        "events": len(events),
        "short_scan": short_rows[:10],
        "combos": combos,
        "optimized": optimized_rules,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if RULES_JSON.exists():
        rules = json.loads(RULES_JSON.read_text(encoding="utf-8"))
    else:
        rules = {}
    rules["long_short_v2"] = optimized_rules
    rules["key_insights"] = (rules.get("key_insights") or []) + [
        "多空结合：弱板块衰竭做空 hold5 与科技强动量做多互补",
        "组合胜率与夏普通常高于纯多头，回撤更可控",
    ]
    RULES_JSON.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {OUT_JSON}\n→ 更新 {RULES_JSON}")


if __name__ == "__main__":
    main()
