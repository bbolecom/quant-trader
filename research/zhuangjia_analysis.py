"""主力做庄规律分析

针对 SNDK / MU / MSTR / TSLA，从日线 OHLCV 推断主力(庄家)行为：
  - 量价配合：上涨/下跌日的成交量比 (up/down volume ratio)
  - OBV 趋势：资金净流入/流出方向
  - A/D 线 (Accumulation/Distribution)：吸筹 vs 出货
  - 影线行为：上影线(出货/打压) vs 下影线(洗盘/护盘) —— 即"插针"
  - 异常放量日：放量上攻 vs 放量滞涨(出货嫌疑)
  - 缩量回调 / 地量见底：吸筹特征
  - 区间结构：阶段高低点 + 波动率(ATR%)

数据来源：research/charts/*.json（6mo 日线），MSTR 走 yfinance 实时补抓。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "charts"
TICKERS = ["SNDK", "MU", "MSTR", "TSLA"]


def load_local(ticker: str) -> pd.DataFrame | None:
    fp = CHARTS / f"{ticker}.json"
    if not fp.exists():
        return None
    data = json.loads(fp.read_text())
    df = pd.DataFrame(data["bars"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def fetch_yf(ticker: str, period="6mo") -> pd.DataFrame | None:
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        return df
    except Exception as e:  # noqa
        print(f"  [warn] yf fetch {ticker} failed: {e}")
        return None


def load(ticker: str) -> pd.DataFrame | None:
    df = load_local(ticker)
    if df is None:
        print(f"  [info] {ticker} 无本地数据，尝试 yfinance ...")
        df = fetch_yf(ticker)
    return df


def analyze(ticker: str, df: pd.DataFrame) -> dict:
    df = df.copy()
    o, h, l, c, v = df.open, df.high, df.low, df.close, df.volume
    rng = (h - l).replace(0, np.nan)
    chg = c.pct_change()

    # 量价：上涨日 vs 下跌日 平均量
    up_mask = c > c.shift(1)
    dn_mask = c < c.shift(1)
    up_vol = v[up_mask].mean()
    dn_vol = v[dn_mask].mean()
    updn_vol_ratio = up_vol / dn_vol if dn_vol else np.nan

    # OBV
    obv = (np.sign(c.diff().fillna(0)) * v).cumsum()
    obv_slope = np.polyfit(range(len(obv)), obv.values, 1)[0] / (v.mean() or 1)

    # A/D line (Chaikin)
    clv = ((c - l) - (h - c)) / rng
    ad = (clv.fillna(0) * v).cumsum()
    ad_slope = np.polyfit(range(len(ad)), ad.values, 1)[0] / (v.mean() or 1)

    # 影线：上/下影线占全幅比例
    upper_sh = (h - np.maximum(o, c)) / rng
    lower_sh = (np.minimum(o, c) - l) / rng
    upper_sh_mean = upper_sh.mean()
    lower_sh_mean = lower_sh.mean()

    # 异常放量日 (>2x 20日均量)
    vma = v.rolling(20).mean()
    spike = v > 2 * vma
    spike_days = df.index[spike]
    spike_up = ((c > c.shift(1)) & spike).sum()
    spike_dn = ((c < c.shift(1)) & spike).sum()
    # 放量滞涨：放量但当日振幅内收阴 / 长上影
    spike_distrib = (spike & ((c < o) | (upper_sh > 0.5))).sum()

    # 波动率
    atr = rng.rolling(14).mean()
    atr_pct = (atr / c).iloc[-1] * 100

    # 区间
    last = c.iloc[-1]
    hi = c.max(); lo = c.min()
    pos_in_range = (last - lo) / (hi - lo) * 100 if hi > lo else np.nan

    # 近20日 vs 前20日 量能变化
    vol_recent = v.iloc[-20:].mean()
    vol_prev = v.iloc[-40:-20].mean()
    vol_trend = (vol_recent / vol_prev - 1) * 100 if vol_prev else np.nan

    # 趋势
    ret_total = (last / c.iloc[0] - 1) * 100
    ret_20 = (last / c.iloc[-21] - 1) * 100 if len(c) > 21 else np.nan

    return {
        "ticker": ticker,
        "n_bars": len(df),
        "start": df.index[0].date(),
        "end": df.index[-1].date(),
        "last": round(last, 2),
        "ret_6mo_%": round(ret_total, 1),
        "ret_20d_%": round(ret_20, 1),
        "pos_in_range_%": round(pos_in_range, 1),
        "up/dn_vol_ratio": round(uped := updn_vol_ratio, 2),
        "OBV_slope_norm": round(obv_slope, 2),
        "AD_slope_norm": round(ad_slope, 2),
        "upper_shadow_avg": round(upper_sh_mean, 3),
        "lower_shadow_avg": round(lower_sh_mean, 3),
        "shadow_bias": round(lower_sh_mean - upper_sh_mean, 3),
        "spike_days": int(spike.sum()),
        "spike_up": int(spike_up),
        "spike_dn": int(spike_dn),
        "spike_distrib(放量滞涨/出货)": int(spike_distrib),
        "atr%_now": round(atr_pct, 2),
        "vol_trend_20v20_%": round(vol_trend, 1),
        "_spike_dates": [d.date().isoformat() for d in spike_days[-6:]],
    }


def verdict(r: dict) -> str:
    """根据指标输出主力行为判读。"""
    notes = []
    # 资金方向
    if r["OBV_slope_norm"] > 0 and r["AD_slope_norm"] > 0:
        notes.append("资金净流入(OBV/AD 同向上)，主力偏吸筹/做多")
    elif r["OBV_slope_norm"] < 0 and r["AD_slope_norm"] < 0:
        notes.append("资金净流出(OBV/AD 同向下)，主力偏派发/出货")
    else:
        notes.append("OBV与A/D背离，量价分歧，盘面有对倒/诱多诱空嫌疑")
    # 量价
    if r["up/dn_vol_ratio"] >= 1.15:
        notes.append("上涨放量>下跌缩量，承接强(健康拉升)")
    elif r["up/dn_vol_ratio"] <= 0.85:
        notes.append("下跌放量>上涨缩量，抛压重(出货或洗盘)")
    # 影线
    if r["shadow_bias"] >= 0.03:
        notes.append("下影线显著>上影线，盘中常被打到低位后拉回(护盘/洗盘吸筹)")
    elif r["shadow_bias"] <= -0.03:
        notes.append("上影线显著>下影线，冲高回落频繁(高位出货/压制)")
    # 出货
    if r["spike_distrib(放量滞涨/出货)"] >= 3:
        notes.append(f"放量滞涨日多({r['spike_distrib(放量滞涨/出货)']}次)，警惕高位派发")
    # 量能
    if r["vol_trend_20v20_%"] >= 30:
        notes.append("近月量能放大，主力活跃度上升")
    elif r["vol_trend_20v20_%"] <= -30:
        notes.append("近月量能萎缩，主力观望/吸筹后蓄势")
    # 位置
    if r["pos_in_range_%"] >= 80:
        notes.append("价处区间高位")
    elif r["pos_in_range_%"] <= 20:
        notes.append("价处区间低位")
    return "；".join(notes)


def main():
    rows = []
    for t in TICKERS:
        print(f"== {t} ==")
        df = load(t)
        if df is None or df.empty:
            print(f"  [skip] {t} 无数据")
            continue
        r = analyze(t, df)
        rows.append(r)
    if not rows:
        print("无可用数据")
        return
    res = pd.DataFrame(rows).set_index("ticker")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    print("\n================ 指标矩阵 ================")
    show_cols = [c for c in res.columns if not c.startswith("_")]
    print(res[show_cols].T.to_string())
    print("\n================ 主力行为判读 ================")
    for t, r in res.iterrows():
        print(f"\n[{t}] {r['start']}~{r['end']}  现价 {r['last']}  6月{r['ret_6mo_%']}%  近20日{r['ret_20d_%']}%")
        print("  放量日:", r["_spike_dates"])
        print("  判读:", verdict(r.to_dict() | {"ticker": t}))
    # 落盘
    out = ROOT / "zhuangjia_analysis_result.json"
    out.write_text(json.dumps(
        {t: {k: (v if not isinstance(v, (np.integer, np.floating)) else float(v))
             for k, v in r.items()} for t, r in res.iterrows()},
        ensure_ascii=False, indent=2, default=str))
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
