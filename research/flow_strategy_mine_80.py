"""挖掘 80% 胜率条件（全历史事件面板）。"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.capital_flow import build_flow_history, _match_down_patterns, _match_up_patterns
from quant.providers import DataConfig, get_provider, reset_provider_cache
from research.gainer_daily_backtest import GAINER_MOMENTUM


def build_event_panel(years: float = 3.0) -> pd.DataFrame:
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=int(years * 365) + 150)).isoformat()
    spy = yahoo.fetch_history("SPY", start, end)["Close"].astype(float)
    spy_ma50 = spy.rolling(50, min_periods=25).mean()
    spy_1d = spy.pct_change() * 100
    batch = yahoo.fetch_batch(GAINER_MOMENTUM, start, end)
    rows: list[dict] = []
    for tk, df in batch.items():
        hist = build_flow_history(df, spy)
        if hist.empty:
            continue
        for _, row in hist.iterrows():
            d = row["日期"]
            try:
                bull = float(spy.loc[d]) > float(spy_ma50.loc[d])
                s1d = float(spy_1d.loc[d])
            except Exception:  # noqa: BLE001
                bull, s1d = True, 0.0
            r = row.to_dict()
            if float(r.get("dvol_m", 0)) < 50 or float(r.get("现价", 0)) < 5:
                continue
            up = {h["规律ID"] for h in _match_up_patterns(r, spy_bull=bull)}
            down = {h["规律ID"] for h in _match_down_patterns(r, spy_bull=bull)}
            prev = float(r.get("前日涨幅%", 0) or 0)
            if prev > 40:
                down.add("D_OFFERING")
            fwd = float(r.get("fwd_1d", 0))
            rows.append({
                "代码": tk,
                "日期": pd.Timestamp(d),
                "fwd_1d": fwd,
                "long_win": fwd > 0,
                "short_win": fwd < 0,
                "涨幅%": float(r.get("涨幅%", 0)),
                "前日涨%": prev,
                "量比": float(r.get("量比", 0)),
                "收盘强度": float(r.get("close_strength", 0.5)),
                "5日涨%": float(r.get("涨幅5d%", 0)),
                "成交额M": float(r.get("dvol_m", 0)),
                "spy_bull": bull,
                "spy_1d%": s1d,
                "up": up,
                "down": down,
            })
    return pd.DataFrame(rows)


def mine_80(panel: pd.DataFrame, min_n: int = 12) -> pd.DataFrame:
    hits: list[dict] = []
    # SHORT combos
    sh_base = panel[panel["down"].map(lambda x: "D_S2" in x or "D_OFFERING" in x)]
    for prev_min in [40, 45, 50, 55, 60, 65, 70]:
        for gmax in [10, 5, 0, -3, -5, -8]:
            m = sh_base["前日涨%"] >= prev_min
            if gmax != 10:
                m = m & (sh_base["涨幅%"] <= gmax)
            s = sh_base[m]
            if len(s) >= min_n:
                wr = s["short_win"].mean()
                if wr >= 0.75:
                    hits.append({
                        "方向": "short", "规则": f"prev>={prev_min} gain<={gmax}",
                        "n": len(s), "胜率": wr, "均收益%": -s["fwd_1d"].mean() * 100,
                    })
    # LONG U_S2 combos
    lo = panel[panel["up"].map(lambda x: "U_S2" in x) & panel["spy_bull"]]
    for g0, g1 in [(7, 9), (7, 10), (8, 10), (8, 11), (9, 11), (9, 12), (10, 13)]:
        for v0, v1 in [(1.5, 2.0), (1.6, 2.0), (1.7, 2.2), (1.5, 1.75)]:
            for cs_min in [0.65, 0.70, 0.75, 0.80]:
                for g5_max in [15, 20, 25]:
                    m = (
                        (lo["涨幅%"] >= g0) & (lo["涨幅%"] <= g1)
                        & (lo["量比"] >= v0) & (lo["量比"] <= v1)
                        & (lo["收盘强度"] >= cs_min)
                        & (lo["5日涨%"] <= g5_max)
                    )
                    s = lo[m]
                    if len(s) >= min_n and s["long_win"].mean() >= 0.75:
                        hits.append({
                            "方向": "long",
                            "规则": f"U_S2 g{g0}-{g1} vr{v0}-{v1} cs>={cs_min} 5d<={g5_max}",
                            "n": len(s), "胜率": s["long_win"].mean(),
                            "均收益%": s["fwd_1d"].mean() * 100,
                        })
    # U_A2
    ua = panel[panel["up"].map(lambda x: "U_A2" in x) & panel["spy_bull"]]
    for g0, g1 in [(2, 4), (2.5, 5), (3, 5)]:
        m = (ua["涨幅%"] >= g0) & (ua["涨幅%"] <= g1)
        s = ua[m]
        if len(s) >= min_n and s["long_win"].mean() >= 0.75:
            hits.append({
                "方向": "long", "规则": f"U_A2 g{g0}-{g1}",
                "n": len(s), "胜率": s["long_win"].mean(),
                "均收益%": s["fwd_1d"].mean() * 100,
            })
  # D_B2 as short if we allow
    b2 = panel[panel["down"].map(lambda x: "D_B2" in x)]
    for gmin in [12, 15, 20]:
        s = b2[b2["涨幅%"] >= gmin]
        if len(s) >= min_n and s["short_win"].mean() >= 0.75:
            hits.append({
                "方向": "short", "规则": f"D_B2 gain>={gmin}",
                "n": len(s), "胜率": s["short_win"].mean(),
                "均收益%": -s["fwd_1d"].mean() * 100,
            })
    if not hits:
        return pd.DataFrame()
    out = pd.DataFrame(hits).sort_values(["胜率", "n"], ascending=False)
    return out[out["胜率"] >= 0.80]


if __name__ == "__main__":
    print("构建事件面板…")
    panel = build_event_panel(3.0)
    print(f"事件 {len(panel)} 条")
    df = mine_80(panel, min_n=10)
    print("\n=== 胜率>=80% 条件 ===")
    if df.empty:
        print("无满足条件（尝试放宽 min_n 或 75% 档）")
        df75 = mine_80(panel, min_n=8)
        df75 = df75[df75["胜率"] >= 0.75].head(20)
        print(df75.to_string(index=False))
    else:
        print(df.to_string(index=False))
    out = ROOT / "research" / "flow_strategy_80_mine.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n→ {out}")
