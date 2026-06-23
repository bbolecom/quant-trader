"""优化 SNDK 类高波股上的「现金担保认沽 CSP」——上一步证明它是最稳定的卖方策略。

测试：
    1. 不同卖出 delta（0.10/0.15/0.20/0.25/0.30）对胜率、波动、最差单笔的影响。
    2. 趋势过滤：仅在股价站上 50 日均线时开仓（规避下跌段被接飞刀）。
    3. 止盈：赚到 50% 权利金提前平仓（用线性时间近似）对稳定性的提升。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import yfinance as yf

from sndk_income_compare import (
    TRADING_DAYS, HOLD_TD, DTE_CAL, STEP, RFR, VRP,
    bs_put, put_strike, realized_vol, fetch,
)


def backtest_csp(close, delta=0.25, vrp=VRP, ma_filter=0, take_profit=0.0):
    close = close.astype(float).dropna()
    rv = realized_vol(close)
    ma = close.rolling(ma_filter).mean() if ma_filter else None
    T = DTE_CAL / TRADING_DAYS
    rors, skipped = [], 0
    i = max(25, ma_filter)
    while i + HOLD_TD < len(close):
        S = float(close.iloc[i]); sig = float(rv.iloc[i])
        if not np.isfinite(sig) or sig <= 0:
            i += STEP; continue
        if ma is not None and not (S > float(ma.iloc[i])):
            skipped += 1; i += STEP; continue
        iv = sig * (1 + vrp)
        K = put_strike(S, T, iv, delta)
        credit = bs_put(S, K, T, iv)
        ST = float(close.iloc[i + HOLD_TD])
        # 止盈近似：若持有中途价格高于行权价足够多，提前以 take_profit 比例落袋
        if take_profit > 0:
            path = close.iloc[i:i + HOLD_TD + 1]
            # 当未实现盈利达到 take_profit*credit 时平仓（用内在价值近似 mark）
            exited = False
            for j in range(1, len(path)):
                Sj = float(path.iloc[j])
                # 剩余时间价值近似线性衰减
                remain = max(0.0, 1 - j / HOLD_TD)
                mark = max(0.0, K - Sj) + credit * remain * 0.5
                profit = credit - mark
                if profit >= take_profit * credit:
                    rors.append(profit / K); exited = True; break
            if exited:
                i += STEP; continue
        pnl = credit - max(0.0, K - ST)
        rors.append(pnl / K)
        i += STEP
    if not rors:
        return {}
    r = pd.Series(rors)
    eq = (1 + r * 0.2).cumprod()
    cyc = TRADING_DAYS / HOLD_TD
    return {
        "交易数": len(r),
        "胜率": float((r > 0).mean()),
        "平均ROR": float(r.mean()),
        "标准差": float(r.std(ddof=0)),
        "信息比": float(r.mean() / r.std(ddof=0)) if r.std(ddof=0) > 0 else 0.0,
        "最差单笔": float(r.min()),
        "年化": float((1 + r.mean()) ** cyc - 1),
        "合成回撤": float((eq / eq.cummax() - 1).min()),
    }


def grid(ticker, start):
    df = fetch(ticker, start)
    close = df["Close"]
    print(f"\n{'='*96}\n{ticker} CSP 参数优化（{df.index[0].date()}~{df.index[-1].date()}）\n{'='*96}")

    print("\n[A] 不同卖出 Delta（无过滤、持有到期）")
    rows = []
    for d in [0.10, 0.15, 0.20, 0.25, 0.30]:
        s = backtest_csp(close, delta=d)
        s = {"Delta": d, **s}
        rows.append(s)
    _print(rows)

    print("\n[B] Delta 0.20 + 50日均线趋势过滤 + 50%止盈 组合对比")
    rows = []
    for label, kw in [
        ("基线(0.20,持有到期)", dict(delta=0.20)),
        ("+50日均线过滤", dict(delta=0.20, ma_filter=50)),
        ("+50%止盈", dict(delta=0.20, take_profit=0.5)),
        ("均线过滤+50%止盈", dict(delta=0.20, ma_filter=50, take_profit=0.5)),
    ]:
        s = backtest_csp(close, **kw)
        s = {"配置": label, **s}
        rows.append(s)
    _print(rows, key="配置")


def _print(rows, key="Delta"):
    df = pd.DataFrame(rows)
    for c in ["胜率", "平均ROR", "标准差", "最差单笔", "年化", "合成回撤"]:
        if c in df.columns:
            df[c] = df[c].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "-")
    if "信息比" in df.columns:
        df["信息比"] = df["信息比"].map(lambda x: f"{x:.2f}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    grid("SNDK", "2025-02-01")
    grid("WDC", "2018-01-01")
    grid("MU", "2018-01-01")
