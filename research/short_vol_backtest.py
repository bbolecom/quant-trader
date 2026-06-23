"""穿越牛熊策略研究 1：长期做空波动率 ETF（UVXY / UVIX / VXX）。

核心逻辑：杠杆/做多波动率 ETF 持续 roll VIX 期货，长期处于 contango（远月贵于近月），
存在结构性衰减 → 做空并持有理论上吃这部分衰减。难点是极端行情下空头亏损可能爆仓。

本脚本用真实历史数据回测多种做空方式，重点考察「收益 vs 尾部风险」：
    A. 静态做空不再平衡（let it ride）—— 演示为何会爆仓
    B. 恒定比例每日再平衡（constant fraction）—— 控制单日最大损失
    C. 恒定比例 + 趋势过滤（ETF 站上 N 日高点时暂停做空，规避 vol spike）

把借券成本（borrow fee）与交易成本计入，输出年化、最大回撤、最差单日、Calmar。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252


def fetch(ticker: str, start: str = "2011-01-01", end: str = "2026-06-17") -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def perf(equity: pd.Series, rets: pd.Series) -> dict:
    n = len(rets)
    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = n / TRADING_DAYS
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0 if years > 0 else 0.0
    vol = rets.std(ddof=0) * np.sqrt(TRADING_DAYS)
    sharpe = (rets.mean() * TRADING_DAYS) / vol if vol > 0 else 0.0
    dd = (equity / equity.cummax() - 1.0).min()
    calmar = cagr / abs(dd) if dd < 0 else float("inf")
    return {
        "总收益": total,
        "年化(CAGR)": cagr,
        "年化波动": vol,
        "夏普": sharpe,
        "最大回撤": dd,
        "Calmar": calmar,
        "最差单日": rets.min(),
        "最好单日": rets.max(),
        "交易日数": n,
    }


def short_static(px: pd.Series, borrow_annual: float = 0.15, fee_bps: float = 5.0) -> tuple[pd.Series, pd.Series]:
    """A. 期初做空 1 倍名义、之后不再平衡。空头权益 = 初始保证金 + 累计空头盈亏。

    名义随价格涨而放大 → 价格翻倍时空头名义=2x，亏损吞噬本金，典型爆仓路径。
    """
    ret = px.pct_change().fillna(0.0)
    # 初始：权益 1.0，做空名义 1.0（1 倍）。空头盈亏 = -名义 * 价格变化比例 ……
    # 用价格直接模拟：空 1 股，期初股价 P0，权益含保证金 = P0。
    p0 = float(px.iloc[0])
    short_shares = 1.0
    equity = []
    eq = p0  # 初始权益 = 一份名义保证金
    borrow_daily = borrow_annual / TRADING_DAYS
    prev_p = p0
    for p in px:
        pnl = short_shares * (prev_p - p)          # 价格跌→盈利
        eq += pnl - short_shares * prev_p * borrow_daily
        equity.append(eq)
        prev_p = p
        if eq <= 0:                                 # 爆仓后归零
            equity[-1] = 0.0
    e = pd.Series(equity, index=px.index)
    e = e.clip(lower=0.0)
    r = e.pct_change().fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return e / e.iloc[0], r


def short_constant(px: pd.Series, w: float = 0.25, borrow_annual: float = 0.15,
                   fee_bps: float = 5.0) -> tuple[pd.Series, pd.Series]:
    """B. 每日把空头名义再平衡到 = w * 当前权益（恒定比例做空）。

    单日权益变化 ≈ -w * ETF日涨幅 - 借券成本 - 再平衡换手成本。
    亏损时自动缩小名义，避免无限放大 → 限制爆仓风险，但仍会吃大回撤。
    """
    ret = px.pct_change().fillna(0.0)
    borrow_daily = borrow_annual / TRADING_DAYS
    cost = fee_bps / 10_000.0
    # 恒定比例再平衡 → 每日换手 ≈ |组合权重变化|，近似用 w*|ret| 估算
    strat_ret = -w * ret - w * borrow_daily - cost * (w * ret.abs())
    equity = (1.0 + strat_ret).cumprod()
    return equity, strat_ret


def short_filtered(px: pd.Series, w: float = 0.25, hi_window: int = 10,
                   borrow_annual: float = 0.15, fee_bps: float = 5.0,
                   reentry: int = 5) -> tuple[pd.Series, pd.Series]:
    """C. 恒定比例做空 + 趋势过滤：ETF 收盘创 N 日新高（vol spike 启动）→ 次日起平空观望，
    待其回落（连续 reentry 日未创新高）再恢复做空。规避向上尖峰造成的爆炸性亏损。
    """
    ret = px.pct_change().fillna(0.0)
    roll_hi = px.rolling(hi_window).max()
    making_high = (px >= roll_hi - 1e-9)
    # 信号：是否持有空头（1=做空，0=观望），创新高后停，平静后恢复
    hold = np.ones(len(px))
    calm = 0
    active = True
    mh = making_high.to_numpy()
    for i in range(len(px)):
        if mh[i]:
            active = False
            calm = 0
        else:
            calm += 1
            if calm >= reentry:
                active = True
        hold[i] = 1.0 if active else 0.0
    hold = pd.Series(hold, index=px.index).shift(1).fillna(0.0)  # 次日才执行，无未来函数
    borrow_daily = borrow_annual / TRADING_DAYS
    cost = fee_bps / 10_000.0
    strat_ret = hold * (-w * ret - w * borrow_daily) - cost * (w * (hold.diff().abs().fillna(hold)))
    equity = (1.0 + strat_ret).cumprod()
    return equity, strat_ret


def run_for(ticker: str, start: str):
    print(f"\n{'='*78}\n做空标的：{ticker}（起始 {start}）\n{'='*78}")
    df = fetch(ticker, start=start)
    px = df["Close"]
    bh = px / px.iloc[0]
    print(f"标的本身买入持有：总收益 {bh.iloc[-1]-1:+.1%}，"
          f"最差单日 {px.pct_change().min():+.1%}，最好单日 {px.pct_change().max():+.1%}")

    variants = {
        "A 静态做空(不平衡)": short_static(px),
        "B 恒定25%做空(日平衡)": short_constant(px, w=0.25),
        "B' 恒定50%做空(日平衡)": short_constant(px, w=0.50),
        "C 恒定25%+趋势过滤": short_filtered(px, w=0.25),
        "C' 恒定50%+趋势过滤": short_filtered(px, w=0.50),
    }
    rows = []
    for name, (eq, r) in variants.items():
        s = perf(eq, r)
        s["策略"] = name
        rows.append(s)
    res = pd.DataFrame(rows).set_index("策略")
    cols = ["年化(CAGR)", "夏普", "最大回撤", "Calmar", "最差单日", "总收益"]
    fmt = res[cols].copy()
    for c in ["年化(CAGR)", "最大回撤", "最差单日", "总收益"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:+.1%}")
    for c in ["夏普", "Calmar"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:.2f}")
    print(fmt.to_string())
    return res


if __name__ == "__main__":
    run_for("UVXY", "2011-10-04")   # 最长历史，含 2018/2020 极端行情
    run_for("UVIX", "2022-03-30")   # 用户提到的标的
    run_for("VXX", "2018-01-25")
