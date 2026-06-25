#!/usr/bin/env python3
"""近 5 年「日涨幅>10% + 成交额>1亿美元」事件研究。

问题：这类爆涨日之后，哪些会续涨、哪些会回落？与收盘强度/跳空/乖离/量能/板块的关系？
方法：
  1. universe = SP500 ∪ Nasdaq100 ∪ 动量池 (~700)
  2. 拉 5 年日线（磁盘缓存），重建事件：chg≥10% 且 收盘价×量≥1e8
  3. 事件日特征：收盘强度clv、跳空gap、日内冲幅、量比、前20日乖离、RSI、60日位置
  4. 前瞻收益 t+1/3/5/10/20（以事件日收盘为入场）
  5. 分组统计续涨率/均值；拉行业做板块特性
输出：分组表 + 板块表 + 可操作规则，落地 JSON。
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gainer_daily_backtest import GAINER_MOMENTUM, fetch_gainer_data_yahoo
from quant.screener import fetch_nasdaq100_tickers, fetch_sp500_tickers

GAIN_MIN = 0.10        # 日涨幅 ≥ 10%
DVOL_MIN = 1e8         # 成交额 ≥ 1 亿美元
HORIZONS = [1, 3, 5, 10, 20]
SECTOR_CACHE = ROOT / "research" / "sector_map.json"
OUT_JSON = ROOT / "research" / "gainer10_event_study.json"


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
    rs = g / l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def extract_events(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for t, df in data.items():
        if df is None or len(df) < 60:
            continue
        d = df.copy()
        o, h, l, c, v = (d[k].astype(float) for k in ["Open", "High", "Low", "Close", "Volume"])
        chg = c.pct_change()
        dvol = c * v
        rng = (h - l).replace(0, np.nan)
        clv = ((c - l) - (h - c)) / rng
        gap = o / c.shift(1) - 1
        intraday = c / o - 1
        vma20 = v.rolling(20).mean()
        vol_x = v / vma20
        ext20 = c / c.shift(20) - 1
        rsi = _rsi(c)
        hi60 = h.rolling(60).max(); lo60 = l.rolling(60).min()
        pos60 = (c - lo60) / (hi60 - lo60)
        c_arr = c.values
        n = len(c)
        ev = (chg >= GAIN_MIN) & (dvol >= DVOL_MIN)
        idxs = np.where(ev.values)[0]
        for i in idxs:
            if i < 25 or i >= n - 1:
                continue
            fwd = {}
            for hh in HORIZONS:
                j = i + hh
                fwd[f"fwd{hh}"] = (c_arr[j] / c_arr[i] - 1) if j < n else np.nan
            rows.append({
                "代码": t, "日期": d.index[i].date().isoformat(),
                "现价": round(float(c_arr[i]), 2),
                "当日涨%": round(float(chg.iloc[i]) * 100, 1),
                "成交额M": round(float(dvol.iloc[i]) / 1e6, 0),
                "收盘强度": round(float(clv.iloc[i]), 2),
                "跳空%": round(float(gap.iloc[i]) * 100, 1),
                "日内冲%": round(float(intraday.iloc[i]) * 100, 1),
                "量比": round(float(vol_x.iloc[i]), 2),
                "前20乖离%": round(float(ext20.iloc[i]) * 100, 1),
                "RSI": round(float(rsi.iloc[i]), 0),
                "位置60": round(float(pos60.iloc[i]), 2),
                **{k: (round(val * 100, 2) if val == val else np.nan) for k, val in fwd.items()},
            })
    return pd.DataFrame(rows)


def load_sectors(tickers: list[str]) -> dict[str, str]:
    cache: dict[str, str] = {}
    if SECTOR_CACHE.exists():
        cache = json.loads(SECTOR_CACHE.read_text(encoding="utf-8"))
    missing = [t for t in tickers if t not in cache]
    if missing:
        import yfinance as yf
        for i, t in enumerate(missing):
            sec = "Unknown"
            try:
                info = yf.Ticker(t).get_info()
                sec = info.get("sector") or "Unknown"
            except Exception:  # noqa: BLE001
                sec = "Unknown"
            cache[t] = sec
            if (i + 1) % 25 == 0:
                print(f"   行业 {i+1}/{len(missing)} …")
                SECTOR_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        SECTOR_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


def grp(df: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    sub = df[mask]
    if len(sub) < 20:
        return {}
    row = {"分组": label, "样本": len(sub)}
    for hh in HORIZONS:
        col = f"fwd{hh}"
        s = sub[col].dropna()
        row[f"t+{hh}均%"] = round(s.mean(), 2)
        row[f"t+{hh}胜%"] = round((s > 0).mean() * 100, 0)
    return row


def main() -> None:
    print("① 构建 universe …")
    uni = build_universe()
    print(f"   {len(uni)} 只")
    start = (date.today() - timedelta(days=365 * 5 + 10)).isoformat()
    end = date.today().isoformat()
    print(f"② 拉 5 年日线(缓存) {start}→{end} …")
    data, _spy = fetch_gainer_data_yahoo(uni, start, end)
    print(f"   到手 {len(data)} 只")
    print("③ 提取事件 …")
    ev = extract_events(data)
    if ev.empty:
        print("无事件")
        return
    print(f"   事件 {len(ev)} 笔，覆盖 {ev['代码'].nunique()} 只")

    print("④ 拉行业 …")
    secmap = load_sectors(sorted(ev["代码"].unique().tolist()))
    ev["板块"] = ev["代码"].map(secmap).fillna("Unknown")

    base = {"事件数": len(ev), "标的数": int(ev["代码"].nunique())}
    for hh in HORIZONS:
        s = ev[f"fwd{hh}"].dropna()
        base[f"t+{hh}均%"] = round(s.mean(), 2)
        base[f"t+{hh}续涨率%"] = round((s > 0).mean() * 100, 0)

    # 分组
    groups = []
    groups.append(grp(ev, ev["收盘强度"] >= 0.3, "收盘强(clv≥0.3)"))
    groups.append(grp(ev, ev["收盘强度"].between(-0.3, 0.3), "收盘中(−0.3~0.3)"))
    groups.append(grp(ev, ev["收盘强度"] <= -0.3, "收盘弱(clv≤−0.3·冲高回落)"))
    groups.append(grp(ev, ev["跳空%"] >= 5, "高开跳空≥5%"))
    groups.append(grp(ev, ev["跳空%"] <= 0, "平/低开冲高"))
    groups.append(grp(ev, ev["前20乖离%"] >= 30, "已乖离≥30%(追高)"))
    groups.append(grp(ev, ev["前20乖离%"] <= 0, "乖离≤0(低位首爆)"))
    groups.append(grp(ev, ev["量比"] >= 3, "天量≥3x"))
    groups.append(grp(ev, ev["量比"] <= 1.5, "温量≤1.5x"))
    groups.append(grp(ev, ev["RSI"] >= 75, "RSI≥75超买"))
    groups.append(grp(ev, ev["位置60"] >= 0.9, "贴60日高(≥0.9)"))
    # 组合信号
    groups.append(grp(ev, (ev["跳空%"] >= 5) & (ev["前20乖离%"] >= 20),
                      "★续涨组合 高开≥5%+乖离≥20%"))
    groups.append(grp(ev, (ev["跳空%"] <= 0) & (ev["前20乖离%"] <= 5),
                      "★回落组合 平低开+低位首爆"))
    groups.append(grp(ev, (ev["收盘强度"] >= 0.3) & (ev["跳空%"] >= 5) & (ev["前20乖离%"] >= 20),
                      "★强续涨 强收+高开+乖离"))
    is_tech = ev["板块"] == "Technology"
    groups.append(grp(ev, is_tech & (ev["跳空%"] >= 5) & (ev["前20乖离%"] >= 20),
                      "★Tech续涨 科技+高开+乖离"))
    groups.append(grp(ev, (ev["板块"].isin(["Healthcare", "Communication Services", "Consumer Cyclical"]))
                      & (ev["跳空%"] <= 0), "★弱板块平低开(回落)"))
    groups = [g for g in groups if g]

    # 板块
    sec_rows = []
    for sec, sub in ev.groupby("板块"):
        if len(sub) < 30:
            continue
        r = {"板块": sec, "样本": len(sub), "标的": int(sub["代码"].nunique())}
        for hh in [1, 5, 20]:
            s = sub[f"fwd{hh}"].dropna()
            r[f"t+{hh}均%"] = round(s.mean(), 2)
            r[f"t+{hh}续涨%"] = round((s > 0).mean() * 100, 0)
        sec_rows.append(r)
    sec_df = pd.DataFrame(sec_rows).sort_values("t+5均%", ascending=False) if sec_rows else pd.DataFrame()

    # 输出
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    print("\n========== 基准（全部事件）==========")
    for k, v in base.items():
        print(f"  {k}: {v}")
    gdf = pd.DataFrame(groups)
    print("\n========== 分组：续涨率/均值（按收盘强度等切片）==========")
    print(gdf.to_string(index=False))
    if not sec_df.empty:
        print("\n========== 板块特性（样本≥30，按 t+5 均值降序）==========")
        print(sec_df.to_string(index=False))

    OUT_JSON.write_text(json.dumps({
        "params": {"gain_min": GAIN_MIN, "dvol_min": DVOL_MIN, "horizons": HORIZONS,
                   "start": start, "end": end, "universe": len(uni)},
        "base": base,
        "groups": groups,
        "sectors": sec_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ 落地 {OUT_JSON}")


if __name__ == "__main__":
    main()
