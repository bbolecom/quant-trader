#!/usr/bin/env python3
"""在 gainer10 事件库上优化策略：提高胜率与年化。

三步：
  A 入场时机 × 出场：追收盘 vs 等回踩(-x%) × 固定持有 vs 止盈止损
  B 过滤组合网格搜索：板块/跳空/乖离/RSI/量比/位置 → 找最高边际(min样本)
  C 最优规则组合回测：等权 N 槽组合，量出 胜率/CAGR/夏普/最大回撤/年均次数

数据复用 gainer_daily_backtest 的磁盘缓存（已 warm）。
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gainer_daily_backtest import GAINER_MOMENTUM, fetch_gainer_data_yahoo
from quant.screener import fetch_nasdaq100_tickers, fetch_sp500_tickers

GAIN_MIN, DVOL_MIN = 0.10, 1e8
MAX_HOLD = 20
SECTOR_CACHE = ROOT / "research" / "sector_map.json"
OUT_JSON = ROOT / "research" / "gainer10_strategy_optimize.json"


def build_universe() -> list[str]:
    u: set[str] = set(GAINER_MOMENTUM)
    for fn in (fetch_sp500_tickers, fetch_nasdaq100_tickers):
        try:
            u.update(fn())
        except Exception:  # noqa: BLE001
            pass
    return sorted({t.strip().upper() for t in u if t and str(t).strip()})


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


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
        gap = np.empty_like(c); gap[0] = np.nan; gap[1:] = o[1:] / c[:-1] - 1
        vma20 = pd.Series(v).rolling(20).mean().values
        vol_x = v / vma20
        ext20 = np.empty_like(c); ext20[:20] = np.nan; ext20[20:] = c[20:] / c[:-20] - 1
        rsi = _rsi(cs).values
        hi60 = pd.Series(h).rolling(60).max().values
        lo60 = pd.Series(l).rolling(60).min().values
        pos60 = (c - lo60) / (hi60 - lo60)
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
                "pos60": float(pos60[i]) if pos60[i] == pos60[i] else 0.5,
                "fwd_close": c[i + 1:i + 1 + H].copy(),
                "fwd_high": h[i + 1:i + 1 + H].copy(),
                "fwd_low": l[i + 1:i + 1 + H].copy(),
            })
    events.sort(key=lambda e: e["date"])
    return events


def sim_trade(ev: dict, *, entry_dip: float = 0.0, tp: float | None = None,
              sl: float | None = None, hold: int = MAX_HOLD) -> dict | None:
    """模拟单笔。entry_dip>0 表示等回踩 close*(1-dip) 才入场(用 fwd_low 触发)。
    返回 None = 未触发(回踩没等到)。否则返回 {ret, days, win, exit}。"""
    c0 = ev["close"]
    lows, highs, closes = ev["fwd_low"], ev["fwd_high"], ev["fwd_close"]
    H = min(hold, len(closes))
    if H <= 0:
        return None
    # 入场
    if entry_dip > 0:
        trig = c0 * (1 - entry_dip)
        k = None
        for j in range(H):
            if lows[j] <= trig:
                k = j
                break
        if k is None:
            return None
        entry = trig
        start = k + 1  # 次日起持有
    else:
        entry = c0
        start = 0
    if start >= H:
        # 回踩发生在窗口最后一天，按收盘出
        return {"ret": closes[H - 1] / entry - 1, "days": H, "win": closes[H - 1] > entry, "exit": "end"}
    tp_px = entry * (1 + tp) if tp else None
    sl_px = entry * (1 - sl) if sl else None
    for j in range(start, H):
        hi, lo = highs[j], lows[j]
        hit_sl = sl_px is not None and lo <= sl_px
        hit_tp = tp_px is not None and hi >= tp_px
        if hit_sl and hit_tp:  # 同根K：保守按止损先
            return {"ret": -sl, "days": j - start + 1, "win": False, "exit": "sl"}
        if hit_sl:
            return {"ret": -sl, "days": j - start + 1, "win": False, "exit": "sl"}
        if hit_tp:
            return {"ret": tp, "days": j - start + 1, "win": True, "exit": "tp"}
    r = closes[H - 1] / entry - 1
    return {"ret": r, "days": H - start, "win": r > 0, "exit": "end"}


def agg(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0}
    rets = np.array([t["ret"] for t in trades])
    return {"n": len(trades), "win%": round((rets > 0).mean() * 100, 1),
            "均%": round(rets.mean() * 100, 2), "中%": round(np.median(rets) * 100, 2),
            "盈亏比": round(abs(rets[rets > 0].mean() / rets[rets < 0].mean()), 2) if (rets < 0).any() and (rets > 0).any() else None}


def portfolio_backtest(events: list[dict], filt, *, entry_dip, tp, sl, hold,
                       slots=5, fee_bps=5.0, years=5.0) -> dict:
    """等权 slots 槽，按事件时间顺序占槽，槽满则跳过。量化 胜率/CAGR/夏普/回撤。"""
    sel = [e for e in events if filt(e)]
    # 槽位：记录每个槽空闲的日期
    slot_free = [pd.Timestamp.min] * slots
    trade_rets, trade_dates = [], []
    for e in sel:
        free_idx = next((i for i, d in enumerate(slot_free) if e["date"] >= d), None)
        if free_idx is None:
            continue
        r = sim_trade(e, entry_dip=entry_dip, tp=tp, sl=sl, hold=hold)
        if r is None:
            continue
        net = r["ret"] - fee_bps / 1e4 * 2
        trade_rets.append(net)
        trade_dates.append(e["date"])
        slot_free[free_idx] = e["date"] + pd.Timedelta(days=int(r["days"]) + 1)
    if not trade_rets:
        return {"n": 0}
    rets = np.array(trade_rets)
    # 资金曲线：每槽等权 1/slots，串行复利近似
    eq = 1.0
    curve = []
    for r in rets:
        eq *= (1 + r / slots)
        curve.append(eq)
    curve = np.array(curve)
    cagr = eq ** (1 / years) - 1
    dd = (curve / np.maximum.accumulate(curve) - 1).min()
    sharpe = rets.mean() / rets.std() * np.sqrt(len(rets) / years) if rets.std() > 0 else 0
    return {"n": len(rets), "win%": round((rets > 0).mean() * 100, 1),
            "均%": round(rets.mean() * 100, 2), "累计%": round((eq - 1) * 100, 0),
            "CAGR%": round(cagr * 100, 1), "夏普": round(sharpe, 2),
            "最大回撤%": round(dd * 100, 1), "年均次数": round(len(rets) / years, 0)}


def main() -> None:
    print("① 数据(缓存) …")
    uni = build_universe()
    start = (date.today() - timedelta(days=365 * 5 + 10)).isoformat()
    end = date.today().isoformat()
    data, _ = fetch_gainer_data_yahoo(uni, start, end)
    secmap = json.loads(SECTOR_CACHE.read_text(encoding="utf-8")) if SECTOR_CACHE.exists() else {}
    years = 5.0
    print(f"   {len(data)} 只")
    print("② 建事件+路径 …")
    events = build_events(data, secmap)
    print(f"   事件 {len(events)} 笔")

    out = {"events": len(events)}

    # —— A 入场时机 × 出场（固定过滤=科技+高开+乖离 这类强续涨）——
    strong = lambda e: (e["sec"] == "Technology" and e["gap"] >= 0.05 and e["ext20"] >= 0.20)
    sel = [e for e in events if strong(e)]
    print(f"\n===== A 入场×出场实验（过滤=科技+高开≥5%+乖离≥20%，{len(sel)}笔）=====")
    A = []
    configs = [
        ("追收盘·持有20", dict(entry_dip=0, tp=None, sl=None, hold=20)),
        ("追收盘·TP15/SL8", dict(entry_dip=0, tp=0.15, sl=0.08, hold=20)),
        ("追收盘·TP20/SL10", dict(entry_dip=0, tp=0.20, sl=0.10, hold=20)),
        ("回踩3%·持有20", dict(entry_dip=0.03, tp=None, sl=None, hold=20)),
        ("回踩5%·持有20", dict(entry_dip=0.05, tp=None, sl=None, hold=20)),
        ("回踩5%·TP15/SL8", dict(entry_dip=0.05, tp=0.15, sl=0.08, hold=20)),
        ("回踩8%·TP20/SL10", dict(entry_dip=0.08, tp=0.20, sl=0.10, hold=20)),
    ]
    for name, kw in configs:
        trades = [r for e in sel if (r := sim_trade(e, **kw))]
        a = agg(trades); a["方案"] = name; a["触发率%"] = round(len(trades) / max(len(sel), 1) * 100, 0)
        A.append(a)
    adf = pd.DataFrame(A)[["方案", "n", "触发率%", "win%", "均%", "中%", "盈亏比"]]
    pd.set_option("display.width", 220)
    print(adf.to_string(index=False))
    out["entry_exit"] = A

    # —— B 过滤网格搜索（固定 入场=追收盘 持有10，比较边际）——
    print("\n===== B 过滤网格搜索（hold10, 追收盘；min样本60）=====")
    sec_opts = [("任意", None), ("科技", "Technology"), ("非弱板块", "exfin")]
    gap_opts = [("any", -9), ("gap≥0", 0.0), ("gap≥5%", 0.05)]
    ext_opts = [("any", -9), ("ext≥0", 0.0), ("ext≥20%", 0.20), ("ext≥40%", 0.40)]
    rsi_opts = [("any", -9), ("rsi≥60", 60), ("rsi≥75", 75)]
    rows = []
    weak = {"Healthcare", "Communication Services", "Consumer Cyclical", "Consumer Defensive"}
    for (sn, sv), (gn, gv), (en, ev_), (rn, rv) in product(sec_opts, gap_opts, ext_opts, rsi_opts):
        def filt(e, sv=sv, gv=gv, ev_=ev_, rv=rv):
            if sv == "Technology" and e["sec"] != "Technology":
                return False
            if sv == "exfin" and e["sec"] in weak:
                return False
            return e["gap"] >= gv and e["ext20"] >= ev_ and e["rsi"] >= rv
        tr = [r for e in events if filt(e) and (r := sim_trade(e, entry_dip=0, tp=None, sl=None, hold=10))]
        if len(tr) < 60:
            continue
        a = agg(tr)
        a["过滤"] = f"{sn}|{gn}|{en}|{rn}"
        a["score"] = round(a["win%"] * a["均%"], 1)
        rows.append(a)
    bdf = pd.DataFrame(rows).sort_values("score", ascending=False).head(12)
    print(bdf[["过滤", "n", "win%", "均%", "中%", "score"]].to_string(index=False))
    out["grid_top"] = bdf.to_dict("records")

    # —— C 组合回测（最优规则）——
    print("\n===== C 组合回测（5槽·等权·含费）=====")
    rules = {
        "基准·任意+追收盘TP15/SL8": (lambda e: True, dict(entry_dip=0, tp=0.15, sl=0.08, hold=20)),
        "科技+高开+乖离·追收盘持有20": (strong, dict(entry_dip=0, tp=None, sl=None, hold=20)),
        "科技+高开+乖离·TP20/SL10": (strong, dict(entry_dip=0, tp=0.20, sl=0.10, hold=20)),
        "科技+高开+乖离·回踩5%TP15/SL8": (strong, dict(entry_dip=0.05, tp=0.15, sl=0.08, hold=20)),
        "非弱板块+gap≥5%+ext≥20·TP20/SL10": (
            lambda e: e["sec"] not in weak and e["gap"] >= 0.05 and e["ext20"] >= 0.20,
            dict(entry_dip=0, tp=0.20, sl=0.10, hold=20)),
    }
    C = []
    for name, (filt, kw) in rules.items():
        r = portfolio_backtest(events, filt, years=years, **kw)
        r["规则"] = name
        C.append(r)
    cdf = pd.DataFrame(C)[["规则", "n", "年均次数", "win%", "均%", "CAGR%", "夏普", "最大回撤%", "累计%"]]
    print(cdf.to_string(index=False))
    out["portfolio"] = C

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n→ 落地 {OUT_JSON}")


if __name__ == "__main__":
    main()
