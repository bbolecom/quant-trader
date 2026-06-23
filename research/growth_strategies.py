"""激进增长策略 · 多方案回测（目标：年化翻倍级，含真实回撤）

设计思路：当前市场（流动性集中于大盘/动量、做空小盘已死）下，能博取
"一年翻几倍"的，本质都是 **杠杆 + 趋势/动量 + 风控** 的组合。本脚本把
几种可行路线做成统一回测，输出对比表与逐年表现，供选择。

方案：
  S1  SPY 买入持有                （基准）
  S2  QQQ 买入持有                （基准）
  S3  SOXL 买入持有              （3x 半导体，最猛但回撤巨大）
  S4  SOXL 趋势择时              （SOXL 在 MA 上方才持有，否则空仓）
  S5  TQQQ 趋势择时              （3x 纳指 + 趋势过滤）
  S6  杠杆ETF 动量轮动           （每月轮到最强的杠杆ETF）
  S7  动量轮动（个股 TopK）       （高 beta 池，按 3 月动量月度轮动）
  S8  动量轮动 + 大盘趋势过滤      （仅 SPY>MA200 时持仓，否则空仓）
  S9  绝对动量单票               （持有龙头池中过去 3 月最强的 1 只）
  S10 高 beta 彩票篮子 买入持有

用法：
    python research/growth_strategies.py            # 默认 2021-06 至今
    python research/growth_strategies.py --start 2020-01-01
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

from quant.providers import DataConfig, get_provider, reset_provider_cache

FEE = 5 / 10_000  # 单边 5bp

LEV_ETFS = ["SOXL", "TQQQ", "FNGU", "TECL", "UPRO", "TNA", "LABU", "SOXX", "USD"]
LEADERS = [
    "NVDA", "AVGO", "AMD", "MU", "MRVL", "ARM", "SMCI", "PLTR", "META", "MSFT",
    "TSLA", "COIN", "NFLX", "AAPL", "AMZN", "GOOGL", "CRWD", "NOW", "ANET", "DELL",
]
LOTTERY = ["QBTS", "RGTI", "QURE", "RXT", "WOLF", "RDW"]
BENCH = ["SPY", "QQQ"]


def _metrics(equity: pd.Series, rets: pd.Series) -> dict:
    if len(equity) < 2:
        return {}
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 0.1)
    cagr = (1 + total) ** (1 / years) - 1
    dd = float((equity / equity.cummax() - 1).min())
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    return {"总收益": total, "CAGR": cagr, "最大回撤": dd, "夏普": sharpe}


def _ret_to_equity(rets: pd.Series) -> pd.Series:
    return (1 + rets.fillna(0)).cumprod()


def buy_hold(close: pd.Series) -> pd.Series:
    return close.pct_change()


def trend_timing(close: pd.Series, ma: int = 100) -> pd.Series:
    """收盘在 MA 上方则次日满仓，否则空仓（含切换费用）。"""
    sig = (close > close.rolling(ma).mean()).astype(float).shift(1).fillna(0)
    raw = close.pct_change()
    switch = sig.diff().abs().fillna(0)
    return sig * raw - switch * FEE


def momentum_rotation(
    data: dict[str, pd.DataFrame],
    *,
    lookback: int = 63,
    hold: int = 21,
    top_k: int = 5,
    regime: pd.Series | None = None,
    regime_ma: int = 200,
) -> pd.Series:
    """每 hold 个交易日，轮动到过去 lookback 日动量最强的 top_k 只，等权。

    regime 给定时（如 SPY 收盘），仅当其在 regime_ma 上方才持仓，否则空仓。
    """
    closes = pd.DataFrame({t: df["Close"] for t, df in data.items()}).sort_index()
    closes = closes.dropna(how="all")
    rets = closes.pct_change()
    cal = closes.index
    reg_ok = None
    if regime is not None:
        reg = regime.reindex(cal).ffill()
        reg_ok = reg > reg.rolling(regime_ma).mean()

    port = pd.Series(0.0, index=cal)
    held: list[str] = []
    for i in range(lookback, len(cal) - 1):
        if (i - lookback) % hold == 0:
            if reg_ok is not None and not bool(reg_ok.iloc[i]):
                held = []
            else:
                mom = closes.iloc[i] / closes.iloc[i - lookback] - 1
                mom = mom.dropna()
                held = list(mom.sort_values(ascending=False).head(top_k).index)
        if held:
            nxt = rets.iloc[i + 1][held].mean()
            port.iloc[i + 1] = float(nxt) if np.isfinite(nxt) else 0.0
    return port


def absolute_momentum_single(
    data: dict[str, pd.DataFrame], *, lookback: int = 63, hold: int = 21
) -> pd.Series:
    return momentum_rotation(data, lookback=lookback, hold=hold, top_k=1)


def basket_buy_hold(data: dict[str, pd.DataFrame]) -> pd.Series:
    closes = pd.DataFrame({t: df["Close"] for t, df in data.items()}).sort_index()
    rets = closes.pct_change()
    return rets.mean(axis=1)


def run(start: str, end: str) -> None:
    reset_provider_cache()
    y = get_provider(DataConfig(provider="yahoo"))

    cache = ROOT / "research" / "gainer_universe_cache.json"
    uni = json.loads(cache.read_text()) if cache.exists() else []
    need = sorted(set(uni + LEV_ETFS + LEADERS + LOTTERY + BENCH))
    print(f"拉取 {len(need)} 只行情 {start} ~ {end} …")
    data = y.fetch_batch(need, start, end)
    print(f"有效 {len(data)} 只\n")

    def closeof(t: str) -> pd.Series | None:
        df = data.get(t)
        return df["Close"].astype(float) if df is not None and not df.empty else None

    spy = closeof("SPY")
    schemes: dict[str, pd.Series] = {}

    for t in ["SPY", "QQQ", "SOXL"]:
        c = closeof(t)
        if c is not None:
            schemes[f"{t} 买入持有"] = buy_hold(c)

    for t, ma in [("SOXL", 100), ("TQQQ", 100)]:
        c = closeof(t)
        if c is not None:
            schemes[f"{t} 趋势择时(MA{ma})"] = trend_timing(c, ma)

    lev_data = {t: data[t] for t in LEV_ETFS if t in data}
    if lev_data:
        schemes["杠杆ETF动量轮动(月)"] = momentum_rotation(lev_data, lookback=63, hold=21, top_k=1)

    leader_data = {t: data[t] for t in LEADERS if t in data}
    if leader_data:
        schemes["个股动量轮动Top5(月)"] = momentum_rotation(leader_data, lookback=63, hold=21, top_k=5)
        if spy is not None:
            schemes["动量Top5+大盘趋势过滤"] = momentum_rotation(
                leader_data, lookback=63, hold=21, top_k=5, regime=spy, regime_ma=200
            )
        schemes["绝对动量单票(最强1只)"] = absolute_momentum_single(leader_data)

    lot_data = {t: data[t] for t in LOTTERY if t in data}
    if lot_data:
        schemes["高beta彩票篮子持有"] = basket_buy_hold(lot_data)

    # 汇总
    rows = []
    equities: dict[str, pd.Series] = {}
    for name, r in schemes.items():
        r = r.dropna()
        if len(r) < 30:
            continue
        eq = _ret_to_equity(r)
        equities[name] = eq
        m = _metrics(eq, r)
        rows.append({"方案": name, **m})

    df = pd.DataFrame(rows).sort_values("CAGR", ascending=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("=" * 80)
    print(f"多方案对比（{start} ~ {end}）")
    print("=" * 80)
    disp = df.copy()
    for c in ["总收益", "CAGR", "最大回撤"]:
        disp[c] = disp[c].map(lambda x: f"{x*100:+.0f}%")
    disp["夏普"] = disp["夏普"].map(lambda x: f"{x:.2f}")
    print(disp.to_string(index=False))

    # 逐年
    print("\n" + "=" * 80)
    print("逐年收益率（%）")
    print("=" * 80)
    yearly = {}
    for name, eq in equities.items():
        yr = eq.resample("YE").last().pct_change()
        first_year = eq.index[0].year
        yr0 = eq[eq.index.year == first_year]
        if len(yr0) > 1:
            yr.iloc[0] = yr0.iloc[-1] / yr0.iloc[0] - 1
        yearly[name] = {d.year: v for d, v in yr.items()}
    ydf = pd.DataFrame(yearly).T
    ydf = ydf.reindex(df["方案"].values)
    ydf.columns = [str(int(c)) for c in ydf.columns]
    print((ydf * 100).round(0).fillna(0).astype(int).to_string())

    out = ROOT / "research" / "growth_strategies_results.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n结果已存 {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2021-06-01")
    p.add_argument("--end", default=date.today().isoformat())
    args = p.parse_args()
    run(args.start, args.end)


if __name__ == "__main__":
    main()
