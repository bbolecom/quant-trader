#!/usr/bin/env python3
"""均衡组合 2022 熊市压力回测。

组合（与 strategy_daily balanced 一致，弱市关成长层）：
  25% 卖看涨价差（bear call，多票等权）
  25% 周铁鹰（weekly put spread，MA50 过滤）
  20% Tier A CSP（δ=0.25 MA50 50%止盈 alloc=35%）
  30% 现金
  0%  温和动量（2022 大部分时间 SPY<MA50 → 关闭）

用法：
    python research/balanced_2022_backtest.py
    python research/balanced_2022_backtest.py --year 2022
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant import metrics as M
from quant.decline_income import (
    CALL_DELTA_SHORT,
    CYCLE_DAYS,
    CSP_DTE_CAL,
    CSP_HOLD_TD,
    CSP_MA_WINDOW,
    CSP_STEP_TD,
    CSP_TAKE_PROFIT,
    DEFAULT_VRP,
    DEFAULT_WIDTH_PCT,
    WEEKLY_DTE,
    WEEKLY_SOUP_DELTA,
    WEEKLY_SOUP_TAKE_PROFIT,
    WEEKLY_SOUP_WIDTH,
    _spread_pnl_at_expiry,
    estimate_bear_call_spread,
    estimate_put_credit_spread,
    equity_metrics_from_trades,
)
from quant.providers import DataConfig, get_provider, reset_provider_cache
from quant.vol_decay import TRADING_DAYS, bs_put_price, realized_vol, strike_for_put_delta

YEAR_START = "2022-01-01"
YEAR_END = "2022-12-31"
WARMUP_START = "2021-01-01"

# 2022 可交易高波票（SNDK 2022 未上市，用 WDC 存储代理）
CALL_SPREAD_TICKERS = ["NVDA", "AMD", "MU", "WDC", "INTC", "QCOM", "AVGO"]
WEEKLY_SOUP_TICKERS = ["WDC", "MU", "NVDA", "AMD", "INTC"]
CSP_TICKERS = ["MU", "INTC", "AMD", "WDC", "NVDA", "QCOM"]

WEIGHTS = {
    "call_spread": 0.25,
    "weekly_soup": 0.25,
    "tier_a_csp": 0.20,
    "cash": 0.30,
}


def _in_year(ts: pd.Timestamp, y0: str, y1: str) -> bool:
    return pd.Timestamp(y0) <= ts <= pd.Timestamp(y1)


def dated_bear_call_trades(
    close: pd.Series,
    *,
    vrp: float = DEFAULT_VRP,
    delta: float = CALL_DELTA_SHORT,
    width_pct: float = DEFAULT_WIDTH_PCT,
    cycle: int = CYCLE_DAYS,
    leg_alloc: float = 0.04,
) -> pd.Series:
    close = close.astype(float).dropna()
    rv = realized_vol(close)
    out: dict[pd.Timestamp, float] = {}
    i = 25
    while i + cycle < len(close):
        S = float(close.iloc[i])
        sigma = float(rv.iloc[i])
        if not np.isfinite(sigma) or sigma <= 0:
            i += cycle
            continue
        ks, kl, credit, _, _ = estimate_bear_call_spread(
            S, sigma, vrp=vrp, delta=delta, width_pct=width_pct, dte_days=cycle,
        )
        ST = float(close.iloc[i + cycle])
        pnl = _spread_pnl_at_expiry(ST, ks, kl, credit)
        margin = kl - ks
        if margin > 0:
            out[pd.Timestamp(close.index[i])] = float(pnl / margin) * leg_alloc
        i += cycle
    return pd.Series(out)


def dated_weekly_soup_trades(
    close: pd.Series,
    *,
    ma_window: int = CSP_MA_WINDOW,
    leg_alloc: float = 0.25,
) -> pd.Series:
    close = close.astype(float).dropna()
    rv = realized_vol(close)
    ma = close.rolling(ma_window).mean() if ma_window else None
    hold = max(1, int(WEEKLY_DTE * TRADING_DAYS / 7))
    step_td = 5
    out: dict[pd.Timestamp, float] = {}
    i = max(25, ma_window)
    while i + hold < len(close):
        S = float(close.iloc[i])
        sigma = float(rv.iloc[i])
        if not np.isfinite(sigma) or sigma <= 0:
            i += step_td
            continue
        if ma is not None and not (S > float(ma.iloc[i])):
            i += step_td
            continue
        iv = sigma * (1 + DEFAULT_VRP)
        ks, kl, credit, _, _, _ = estimate_put_credit_spread(
            S, sigma, short_delta=WEEKLY_SOUP_DELTA, width=WEEKLY_SOUP_WIDTH,
            dte_days=WEEKLY_DTE, vrp=DEFAULT_VRP,
        )
        if WEEKLY_SOUP_WIDTH <= 0:
            i += step_td
            continue
        exited = False
        if WEEKLY_SOUP_TAKE_PROFIT > 0:
            path = close.iloc[i:i + hold + 1]
            for j in range(1, len(path)):
                Sj = float(path.iloc[j])
                remain = max(0.0, 1 - j / hold)
                short_loss = max(0.0, ks - Sj) - max(0.0, kl - Sj)
                mark = short_loss + credit * remain * 0.5
                if credit - mark >= WEEKLY_SOUP_TAKE_PROFIT * credit:
                    out[pd.Timestamp(close.index[i])] = float((credit - mark) / WEEKLY_SOUP_WIDTH) * leg_alloc
                    exited = True
                    break
        if not exited:
            ST = float(close.iloc[i + hold])
            pnl = credit - (max(0.0, ks - ST) - max(0.0, kl - ST))
            out[pd.Timestamp(close.index[i])] = float(pnl / WEEKLY_SOUP_WIDTH) * leg_alloc
        i += step_td
    return pd.Series(out)


def dated_tier_a_csp_trades(
    close: pd.Series,
    *,
    delta: float = 0.25,
    ma_window: int = 50,
    take_profit: float = 0.5,
    leg_alloc: float = 0.35,
) -> pd.Series:
    close = close.astype(float).dropna()
    rv = realized_vol(close)
    ma = close.rolling(ma_window).mean() if ma_window else None
    T = CSP_DTE_CAL / TRADING_DAYS
    out: dict[pd.Timestamp, float] = {}
    i = max(25, ma_window)
    while i + CSP_HOLD_TD < len(close):
        S = float(close.iloc[i])
        sigma = float(rv.iloc[i])
        if not np.isfinite(sigma) or sigma <= 0:
            i += CSP_STEP_TD
            continue
        if ma is not None and not (S > float(ma.iloc[i])):
            i += CSP_STEP_TD
            continue
        iv = sigma * (1 + DEFAULT_VRP)
        K = strike_for_put_delta(S, T, iv, target_delta=delta)
        credit = bs_put_price(S, K, T, iv)
        if K <= 0:
            i += CSP_STEP_TD
            continue
        exited = False
        if take_profit > 0:
            path = close.iloc[i:i + CSP_HOLD_TD + 1]
            for j in range(1, len(path)):
                Sj = float(path.iloc[j])
                remain = max(0.0, 1 - j / CSP_HOLD_TD)
                mark = max(0.0, K - Sj) + credit * remain * 0.5
                if credit - mark >= take_profit * credit:
                    out[pd.Timestamp(close.index[i])] = float((credit - mark) / K) * leg_alloc
                    exited = True
                    break
        if not exited:
            ST = float(close.iloc[i + CSP_HOLD_TD])
            out[pd.Timestamp(close.index[i])] = float((credit - max(0.0, K - ST)) / K) * leg_alloc
        i += CSP_STEP_TD
    return pd.Series(out)


def _basket_trades(fn, tickers: list[str], batch: dict, **kw) -> pd.Series:
    parts = []
    for t in tickers:
        df = batch.get(t)
        if df is None or df.empty:
            continue
        s = fn(df["Close"].astype(float), **kw)
        if not s.empty:
            parts.append(s)
    if not parts:
        return pd.Series(dtype=float)
    combined = pd.concat(parts).sort_index()
    # 同日多票：等权平均账户冲击
    return combined.groupby(combined.index).mean()


def _filter_year(s: pd.Series, y0: str, y1: str) -> pd.Series:
    if s.empty:
        return s
    idx = pd.DatetimeIndex(s.index)
    mask = (idx >= pd.Timestamp(y0)) & (idx <= pd.Timestamp(y1))
    return s.loc[mask]


def _portfolio_equity(
    legs: dict[str, pd.Series],
    weights: dict[str, float],
    y0: str,
    y1: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """按权重合并各腿交易收益 → 账户净值。"""
    rows = []
    for name, ser in legs.items():
        w = weights.get(name, 0.0)
        for ts, r in _filter_year(ser, y0, y1).items():
            rows.append({"date": ts, "leg": name, "ret": float(r) * w})
    if not rows:
        return pd.Series(dtype=float), pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("date")
    # 同日多腿叠加
    daily = df.groupby("date")["ret"].sum()
    eq = (1.0 + daily).cumprod()
    eq.index = pd.DatetimeIndex(eq.index)
    return eq, df


def _year_stats(eq: pd.Series, trades: pd.Series) -> dict:
    if eq.empty or len(eq) < 1:
        return {"ann": 0.0, "total": 0.0, "max_dd": 0.0, "win_rate": 0.0, "trades": 0, "sharpe": 0.0}
    total = float(eq.iloc[-1] - 1.0)
    if isinstance(eq.index, pd.DatetimeIndex) and len(eq) >= 2:
        ann = M.cagr(eq)
        rets = eq.pct_change().fillna(0.0)
        sharpe = M.sharpe_ratio(rets)
    else:
        ann = total
        sharpe = 0.0
    mdd = M.max_drawdown(eq) if len(eq) >= 2 else 0.0
    wr = float((trades > 0).mean()) if len(trades) else 0.0
    return {
        "ann": ann,
        "total": total,
        "max_dd": mdd,
        "win_rate": wr,
        "trades": int(len(trades)),
        "sharpe": sharpe,
    }


def spy_regime_year(yahoo, year_start: str, year_end: str) -> dict:
    warmup = (pd.Timestamp(year_start) - pd.DateOffset(months=14)).strftime("%Y-%m-%d")
    spy = yahoo.fetch_history("SPY", warmup, year_end)["Close"].astype(float)
    spy_y = spy.loc[year_start:year_end]
    ma50 = spy.rolling(50).mean()
    above = (spy_y > ma50.loc[spy_y.index]).sum()
    total = len(spy_y)
    ret = float(spy_y.iloc[-1] / spy_y.iloc[0] - 1) if len(spy_y) >= 2 else 0.0
    return {
        "spy_return": ret,
        "days_above_ma50": int(above),
        "days_total": int(total),
        "pct_bull_days": above / total if total else 0,
    }


def run_backtest(*, year_start: str = YEAR_START, year_end: str = YEAR_END) -> dict:
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    warmup = (pd.Timestamp(year_start) - pd.DateOffset(months=14)).strftime("%Y-%m-%d")
    tickers = sorted(set(CALL_SPREAD_TICKERS + WEEKLY_SOUP_TICKERS + CSP_TICKERS + ["SPY"]))
    batch = yahoo.fetch_batch(tickers, warmup, year_end)

    legs_raw = {
        "call_spread": _basket_trades(
            dated_bear_call_trades, CALL_SPREAD_TICKERS, batch, leg_alloc=0.04,
        ),
        "weekly_soup": _basket_trades(
            dated_weekly_soup_trades, WEEKLY_SOUP_TICKERS, batch, leg_alloc=0.25,
        ),
        "tier_a_csp": _basket_trades(
            dated_tier_a_csp_trades, CSP_TICKERS, batch, leg_alloc=0.35,
        ),
    }

    eq, detail = _portfolio_equity(legs_raw, WEIGHTS, year_start, year_end)
    port_trades = detail.groupby("date")["ret"].sum() if not detail.empty else pd.Series(dtype=float)

    # 单腿统计
    leg_stats = {}
    for name, ser in legs_raw.items():
        yr = _filter_year(ser, year_start, year_end)
        w = WEIGHTS[name]
        if len(yr):
            eq_contrib = (1.0 + yr * w).cumprod()
            contrib_total = float(eq_contrib.iloc[-1] - 1.0)
            contrib_win = float((yr > 0).mean())
        else:
            contrib_total = 0.0
            contrib_win = 0.0
        leg_stats[name] = {
            "weight": w,
            "raw_trades": int(len(yr)),
            "contrib_total": contrib_total,
            "contrib_win": contrib_win,
        }

    port_stats = _year_stats(eq, port_trades)

    spy = batch.get("SPY")
    spy_eq = pd.Series(dtype=float)
    if spy is not None and not spy.empty:
        sc = spy["Close"].astype(float).loc[year_start:year_end]
        if len(sc) >= 2:
            spy_eq = sc / sc.iloc[0]

    regime = spy_regime_year(yahoo, year_start, year_end)

    return {
        "year": year_start[:4],
        "regime": regime,
        "portfolio": port_stats,
        "legs": leg_stats,
        "equity": eq,
        "spy_equity": spy_eq,
        "detail": detail,
    }


def _print_report(res: dict) -> None:
    y = res["year"]
    reg = res["regime"]
    p = res["portfolio"]
    print("=" * 64)
    print(f"均衡组合 · {y} 熊市压力回测")
    print("=" * 64)
    print(f"SPY {y} 收益: {reg['spy_return']:.1%}  |  "
          f"SPY>MA50 天数: {reg['days_above_ma50']}/{reg['days_total']} ({reg['pct_bull_days']:.0%})")
    print(f"\n组合权重: 卖Call价差25% + 周铁鹰25% + Tier-A CSP20% + 现金30% + 动量0%")
    print(f"\n【{y} 组合整体】")
    print(f"  全年收益:   {p['total']:.1%}")
    print(f"  年化:       {p['ann']:.1%}")
    print(f"  最大回撤:   {p['max_dd']:.1%}")
    print(f"  交易胜率:   {p['win_rate']:.1%}  ({p['trades']} 笔)")
    print(f"  夏普:       {p['sharpe']:.2f}")

    if not res["spy_equity"].empty:
        spy_tot = float(res["spy_equity"].iloc[-1] - 1)
        print(f"\n  对比 SPY:   {spy_tot:.1%}  |  超额: {p['total'] - spy_tot:+.1%}")

    print(f"\n【各引擎 {y} 贡献（加权后）】")
    labels = {
        "call_spread": "卖看涨价差",
        "weekly_soup": "周铁鹰",
        "tier_a_csp": "Tier A CSP",
    }
    for k, st in res["legs"].items():
        print(
            f"  {labels.get(k, k)} ({st['weight']:.0%}): "
            f"贡献收益≈{st.get('contrib_total', 0):.1%}  "
            f"腿胜率={st.get('contrib_win', 0):.1%}  "
            f"开仓={st.get('raw_trades', 0)}次"
        )

    # 结论
    print("\n【结论】")
    if p["total"] > 0 and p["total"] > reg["spy_return"]:
        print(f"  ✅ {y} 组合盈利 {p['total']:.1%}，跑赢 SPY {reg['spy_return']:.1%}")
    elif p["total"] > reg["spy_return"]:
        print(f"  ✅ {y} 组合亏损 {p['total']:.1%}，但仍跑赢 SPY {reg['spy_return']:.1%}")
    else:
        print(f"  ⚠ {y} 组合 {p['total']:.1%}，跑输 SPY {reg['spy_return']:.1%}（牛市常见）")
    if reg["pct_bull_days"] < 0.4:
        print("  · CSP/周铁鹰 MA50 过滤：弱市大量时间不开仓 → 25% 周铁鹰层实际闲置")
    print("  · 卖看涨价差在弱市本应更强，但单票 bear_call 回测未含选股过滤，数字偏保守")
    print("  · 期权为 BS 近似，实盘 IV 更高 → 权利金通常更厚")


def main() -> None:
    p = argparse.ArgumentParser(description="均衡组合年度压力回测")
    p.add_argument("--year", type=int, default=2022)
    args = p.parse_args()
    y0 = f"{args.year}-01-01"
    y1 = f"{args.year}-12-31"
    res = run_backtest(year_start=y0, year_end=y1)
    _print_report(res)


if __name__ == "__main__":
    main()
