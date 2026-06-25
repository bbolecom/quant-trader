#!/usr/bin/env python3
"""Gainer10 分板块多空规则优化 · 提高胜率。

对每个 GICS 板块独立搜索最优多/空过滤与出场，输出 sector_rules.json，
供 quant/gainer10_strategy.py 加载。

用法：
    python research/gainer10_sector_optimize.py
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

from research.gainer10_ls_optimize import (  # noqa: E402
    LegSpec,
    attach_spy_regime,
    build_events,
    build_universe,
    portfolio_ls,
    sim_short,
)
from research.gainer10_strategy_optimize import sim_trade  # noqa: E402
from research.gainer_daily_backtest import fetch_gainer_data_yahoo  # noqa: E402

SECTOR_CACHE = ROOT / "research" / "sector_map.json"
OUT_JSON = ROOT / "research" / "gainer10_sector_rules.json"
RULES_JSON = ROOT / "research" / "gainer10_strategy_rules.json"

MIN_SAMPLES = 25
HOLD_LONG = 20
HOLD_SHORT = 10

# 板块中文名
SECTOR_CN = {
    "Technology": "科技",
    "Financial Services": "金融",
    "Healthcare": "医疗",
    "Consumer Cyclical": "可选消费",
    "Consumer Defensive": "必需消费",
    "Communication Services": "通信",
    "Industrials": "工业",
    "Energy": "能源",
    "Basic Materials": "原材料",
    "Real Estate": "房地产",
    "Utilities": "公用事业",
    "Unknown": "未知",
}


@dataclass
class SectorRule:
    sector: str
    side: str
    action: str
    filt_desc: str
    win_pct: float
    avg_pct: float
    n: int
    hold: int
    entry_dip: float = 0.0
    tp: float | None = None
    sl: float | None = None
    gap_min: float | None = None
    gap_max: float | None = None
    ext20_min: float | None = None
    ext20_max: float | None = None
    rsi_min: float | None = None
    volx_min: float | None = None
    clv_max: float | None = None
    require_bull: bool = True

    def to_filt(self):
        sec = self.sector

        def fn(e, _s=sec, r=self):
            if e["sec"] != _s:
                return False
            if r.require_bull and not e.get("bull", True):
                return False
            if r.gap_min is not None and e["gap"] < r.gap_min:
                return False
            if r.gap_max is not None and e["gap"] > r.gap_max:
                return False
            if r.ext20_min is not None and e["ext20"] < r.ext20_min:
                return False
            if r.ext20_max is not None and e["ext20"] > r.ext20_max:
                return False
            if r.rsi_min is not None and e["rsi"] < r.rsi_min:
                return False
            if r.volx_min is not None and e["volx"] < r.volx_min:
                return False
            if r.clv_max is not None and e["clv"] > r.clv_max:
                return False
            return True

        return fn

    def to_dict(self) -> dict:
        return {
            "sector": self.sector,
            "sector_cn": SECTOR_CN.get(self.sector, self.sector),
            "side": self.side,
            "action": self.action,
            "filter": self.filt_desc,
            "win_pct": self.win_pct,
            "avg_pct": self.avg_pct,
            "n": self.n,
            "hold": self.hold,
            "entry_dip": self.entry_dip,
            "tp": self.tp,
            "sl": self.sl,
            "gap_min": self.gap_min,
            "gap_max": self.gap_max,
            "ext20_min": self.ext20_min,
            "ext20_max": self.ext20_max,
            "rsi_min": self.rsi_min,
            "volx_min": self.volx_min,
            "clv_max": self.clv_max,
            "require_bull": self.require_bull,
        }


def _scan_long(sector: str, events: list[dict]) -> SectorRule | None:
    sub = [e for e in events if e["sec"] == sector]
    if len(sub) < MIN_SAMPLES:
        return None
    best: SectorRule | None = None
    best_score = -1e9
    for gap_min, ext_min, rsi_min, vol_min, bull_req in product(
        [None, 0.0, 0.05],
        [None, 0.0, 0.20, 0.40],
        [None, 60, 75],
        [None, 1.5, 2.0],
        [True, False],
    ):
        def filt(e, gm=gap_min, em=ext_min, rm=rsi_min, vm=vol_min, br=bull_req):
            if e["sec"] != sector:
                return False
            if br and not e.get("bull", True):
                return False
            if gm is not None and e["gap"] < gm:
                return False
            if em is not None and e["ext20"] < em:
                return False
            if rm is not None and e["rsi"] < rm:
                return False
            if vm is not None and e["volx"] < vm:
                return False
            return True

        for entry_dip, hold in [(0, HOLD_LONG), (0.05, HOLD_LONG), (0.03, HOLD_LONG)]:
            trades = [r for e in sub if filt(e) and (r := sim_trade(e, entry_dip=entry_dip, hold=hold))]
            if len(trades) < MIN_SAMPLES:
                continue
            rets = np.array([t["ret"] for t in trades])
            win = float((rets > 0).mean() * 100)
            avg = float(rets.mean() * 100)
            if avg <= 0:
                continue
            score = win * 0.6 + avg * 2.0
            if score > best_score:
                best_score = score
                parts = []
                if gap_min is not None:
                    parts.append(f"gap≥{gap_min*100:.0f}%")
                if ext_min is not None:
                    parts.append(f"乖离≥{ext_min*100:.0f}%")
                if rsi_min is not None:
                    parts.append(f"RSI≥{rsi_min}")
                if vol_min is not None:
                    parts.append(f"量比≥{vol_min}x")
                if bull_req:
                    parts.append("SPY≥MA20")
                if entry_dip > 0:
                    parts.append(f"回踩{entry_dip*100:.0f}%")
                best = SectorRule(
                    sector=sector, side="long", action="多",
                    filt_desc=" · ".join(parts) or "基准",
                    win_pct=round(win, 1), avg_pct=round(avg, 2), n=len(trades),
                    hold=hold, entry_dip=entry_dip,
                    gap_min=gap_min, ext20_min=ext_min, rsi_min=rsi_min,
                    volx_min=vol_min, require_bull=bull_req,
                )
    return best


def _scan_short(sector: str, events: list[dict]) -> SectorRule | None:
    sub = [e for e in events if e["sec"] == sector]
    if len(sub) < MIN_SAMPLES:
        return None
    best: SectorRule | None = None
    best_score = -1e9
    for gap_max, ext_max, clv_max, hold, tp, sl in product(
        [0.0, 0.02],
        [0.0, 0.10],
        [None, 0.2, 0.0],
        [5, 10],
        [None, 0.06],
        [None, 0.12],
    ):
        def filt(e, gx=gap_max, ex=ext_max, cx=clv_max):
            if e["sec"] != sector:
                return False
            if e["gap"] > gx:
                return False
            if e["ext20"] > ex:
                return False
            if cx is not None and e["clv"] > cx:
                return False
            return True

        trades = [r for e in sub if filt(e) and (r := sim_short(e, hold=hold, tp=tp, sl=sl))]
        if len(trades) < MIN_SAMPLES:
            continue
        rets = np.array([t["ret"] for t in trades])
        win = float((rets > 0).mean() * 100)
        avg = float(rets.mean() * 100)
        if win < 55 or avg <= 0:
            continue
        score = win * 0.7 + avg * 1.5
        if score > best_score:
            best_score = score
            parts = [f"gap≤{gap_max*100:.0f}%"]
            if ext_max is not None:
                parts.append(f"乖离≤{ext_max*100:.0f}%")
            if clv_max is not None:
                parts.append(f"弱收clv≤{clv_max}")
            exit_p = f"hold{hold}"
            if tp and sl:
                exit_p = f"TP{tp*100:.0f}%/SL{sl*100:.0f}% hold{hold}"
            best = SectorRule(
                sector=sector, side="short", action="空",
                filt_desc=" · ".join(parts) + f" · {exit_p}",
                win_pct=round(win, 1), avg_pct=round(avg, 2), n=len(trades),
                hold=hold, tp=tp, sl=sl,
                gap_max=gap_max, ext20_max=ext_max, clv_max=clv_max,
                require_bull=False,
            )
    return best


def portfolio_sector_rules(
    events: list[dict],
    long_rules: list[SectorRule],
    short_rules: list[SectorRule],
    *,
    long_slots: int = 3,
    short_slots: int = 3,
    years: float = 5.0,
) -> dict:
    long_specs = [
        LegSpec("long", f"L·{r.sector}", r.to_filt(),
                entry_dip=r.entry_dip, tp=r.tp, sl=r.sl, hold=r.hold)
        for r in long_rules
    ]
    short_specs = [
        LegSpec("short", f"S·{r.sector}", r.to_filt(),
                entry_dip=0, tp=r.tp, sl=r.sl, hold=r.hold)
        for r in short_rules
    ]

    slot_free = {"long": [pd.Timestamp.min] * long_slots,
                 "short": [pd.Timestamp.min] * short_slots}
    rets, sides = [], []

    for e in events:
        matched = False
        for spec in long_specs:
            if not spec.filt(e):
                continue
            fi = next((i for i, d in enumerate(slot_free["long"]) if e["date"] >= d), None)
            if fi is None:
                break
            r = sim_trade(e, entry_dip=spec.entry_dip, tp=spec.tp, sl=spec.sl, hold=spec.hold)
            if r is None:
                break
            net = r["ret"] - 0.001
            rets.append(net)
            sides.append("long")
            slot_free["long"][fi] = e["date"] + pd.Timedelta(days=int(r["days"]) + 1)
            matched = True
            break
        if matched:
            continue
        for spec in short_specs:
            if not spec.filt(e):
                continue
            fi = next((i for i, d in enumerate(slot_free["short"]) if e["date"] >= d), None)
            if fi is None:
                break
            r = sim_short(e, tp=spec.tp, sl=spec.sl, hold=spec.hold)
            if r is None:
                break
            net = r["ret"] - 0.001
            rets.append(net)
            sides.append("short")
            slot_free["short"][fi] = e["date"] + pd.Timedelta(days=int(r["days"]) + 1)
            break

    if not rets:
        return {"n": 0}
    arr = np.array(rets)
    total = long_slots + short_slots
    eq = 1.0
    for r in rets:
        eq *= 1 + r / total
    cagr = eq ** (1 / years) - 1
    curve = np.array([1.0])
    e2 = 1.0
    for r in rets:
        e2 *= 1 + r / total
        curve = np.append(curve, e2)
    dd = (curve / np.maximum.accumulate(curve) - 1).min()
    sharpe = arr.mean() / arr.std() * np.sqrt(len(arr) / years) if arr.std() > 0 else 0
    lr = [r for r, s in zip(rets, sides) if s == "long"]
    sr = [r for r, s in zip(rets, sides) if s == "short"]
    return {
        "n": len(rets),
        "n_long": len(lr),
        "n_short": len(sr),
        "win%": round((arr > 0).mean() * 100, 1),
        "long_win%": round((np.array(lr) > 0).mean() * 100, 1) if lr else None,
        "short_win%": round((np.array(sr) > 0).mean() * 100, 1) if sr else None,
        "均%": round(arr.mean() * 100, 2),
        "CAGR%": round(cagr * 100, 1),
        "夏普": round(sharpe, 2),
        "最大回撤%": round(dd * 100, 1),
        "年均次数": round(len(rets) / years, 0),
    }


def main() -> None:
    print("① 加载事件 …")
    uni = build_universe()
    start = (date.today() - timedelta(days=365 * 5 + 10)).isoformat()
    end = date.today().isoformat()
    data, spy = fetch_gainer_data_yahoo(uni, start, end)
    secmap = json.loads(SECTOR_CACHE.read_text(encoding="utf-8")) if SECTOR_CACHE.exists() else {}
    events = build_events(data, secmap)
    attach_spy_regime(events, spy)
    sectors = sorted({e["sec"] for e in events})
    print(f"   {len(events)} 笔 · {len(sectors)} 板块")

    print("\n===== 分板块 · 最优多头规则 =====")
    long_rules: list[SectorRule] = []
    for sec in sectors:
        r = _scan_long(sec, events)
        if r:
            long_rules.append(r)
            cn = SECTOR_CN.get(sec, sec)
            print(f"  {cn}({sec}): n={r.n} 胜率{r.win_pct}% 均{r.avg_pct}% | {r.filt_desc}")

    print("\n===== 分板块 · 最优空头规则 =====")
    short_rules: list[SectorRule] = []
    for sec in sectors:
        r = _scan_short(sec, events)
        if r:
            short_rules.append(r)
            cn = SECTOR_CN.get(sec, sec)
            print(f"  {cn}({sec}): n={r.n} 胜率{r.win_pct}% 均{r.avg_pct}% | {r.filt_desc}")

    long_rules.sort(key=lambda x: -x.win_pct)
    short_rules.sort(key=lambda x: -x.win_pct)

    print("\n===== 组合回测：分板块 vs 统一规则 =====")
    from research.gainer10_ls_optimize import WEAK

    long_a = lambda e: e["sec"] == "Technology" and e["ext20"] >= 0.40 and e["rsi"] >= 75 and e.get("bull", True)
    short_w = lambda e: e["sec"] in WEAK and e["gap"] <= 0 and e["ext20"] <= 0
    unified = portfolio_ls(
        events,
        LegSpec("long", "A", long_a, hold=20),
        LegSpec("short", "S", short_w, hold=10),
        years=5.0,
    )
    sector_bt = portfolio_sector_rules(events, long_rules, short_rules, years=5.0)
    print(f"  统一 A+S:  win={unified.get('win%')}% CAGR={unified.get('CAGR%')}% 夏普={unified.get('夏普')} 回撤={unified.get('最大回撤%')}% n={unified.get('n')}")
    print(f"  分板块:    win={sector_bt.get('win%')}% CAGR={sector_bt.get('CAGR%')}% 夏普={sector_bt.get('夏普')} 回撤={sector_bt.get('最大回撤%')}% n={sector_bt.get('n')}")
    print(f"             多={sector_bt.get('n_long')}({sector_bt.get('long_win%')}%) 空={sector_bt.get('n_short')}({sector_bt.get('short_win%')}%)")

    out = {
        "generated": end,
        "min_samples": MIN_SAMPLES,
        "long_rules": [r.to_dict() for r in long_rules],
        "short_rules": [r.to_dict() for r in short_rules],
        "portfolio_sector": sector_bt,
        "portfolio_unified": unified,
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if RULES_JSON.exists():
        rules = json.loads(RULES_JSON.read_text(encoding="utf-8"))
    else:
        rules = {}
    rules["sector_v1"] = out
    RULES_JSON.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {OUT_JSON}")


if __name__ == "__main__":
    main()
