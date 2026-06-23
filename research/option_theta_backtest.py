"""穿越牛熊策略研究 2：在高成交额个股上「卖期权吃时间损耗(theta)」。

⚠️ 重要前提：免费数据（yfinance）拿不到历史期权链/历史隐含波动率(IV)，系统也没有期权
定价模型。因此本脚本用 Black-Scholes + 「波动率风险溢价(VRP)」对卖方收益做**近似模拟**，
而非真实成交回放。结论用于判断「策略方向是否成立、哪些标的更适合」，不能当成精确收益。

模拟机制（每月一个周期，到期结算）：
    1. 每 21 个交易日开一次仓，卖出 1 个月期权，到期实现盈亏。
    2. 卖方报价 IV = 近 20 日已实现波动率 RV × (1+VRP)。历史上 IV 平均高于 RV（方差风险
       溢价），这正是卖方长期正期望的来源。VRP 越高、RV 越高（高成交额高波动个股）→ 收的
       权利金越厚。
    3. 三种卖方结构：备兑/现金担保认沽(CSP)、宽跨式(short strangle)，按 delta 选行权价。
    4. 盈亏 = 收到的权利金 − 到期内在价值；按所需保证金/名义计算周期收益率。

把交易成本计入；输出年化、夏普、最大回撤、胜率、最差单周期，并与买入持有对比。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252
CYCLE = 21               # 每个期权周期的交易日数（≈1 个月）
RFR = 0.04               # 无风险利率（年化）


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S: float, K: float, T: float, sigma: float, r: float, call: bool) -> float:
    """Black-Scholes 欧式期权价格（每股）。"""
    if T <= 0 or sigma <= 0 or S <= 0:
        intrinsic = max(0.0, (S - K) if call else (K - S))
        return intrinsic
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if call:
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def strike_for_delta(S: float, T: float, sigma: float, r: float, target_delta: float, call: bool) -> float:
    """给定目标 |delta| 反推行权价（近似）。target_delta 取 0.16~0.30 之间较常见。"""
    if T <= 0 or sigma <= 0:
        return S
    if call:
        d1 = _norm_cdf_inv(target_delta)
    else:
        d1 = _norm_cdf_inv(target_delta)
    # call: delta=N(d1)=target → d1=Φ⁻¹(target); put: |delta|=N(-d1)... 用对称近似
    # K = S * exp(-(d1*σ√T) + (r+0.5σ²)T) for call upper strike
    sqrtT = math.sqrt(T)
    if call:
        return S * math.exp(-(d1 * sigma * sqrtT) + (r + 0.5 * sigma ** 2) * T) * 0 + \
               S * math.exp((r + 0.5 * sigma ** 2) * T - d1 * sigma * sqrtT)
    else:
        return S * math.exp((r + 0.5 * sigma ** 2) * T + d1 * sigma * sqrtT)


def _norm_cdf_inv(p: float) -> float:
    """标准正态分位（Acklam 近似）。"""
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def realized_vol(px: pd.Series, window: int = 20) -> pd.Series:
    lr = np.log(px / px.shift(1))
    return lr.rolling(window).std(ddof=0) * np.sqrt(TRADING_DAYS)


def sim_short_put(px: pd.Series, vrp: float = 0.15, delta: float = 0.25,
                  fee_per_contract_bps: float = 2.0) -> dict:
    """现金担保认沽(CSP)：每月卖 1 个月、|delta|≈0.25 的 put，现金全额担保。
    周期收益率 = (权利金 − 到期内在) / 行权价(担保金)。
    """
    rv = realized_vol(px)
    idx = px.index
    cycle_rets = []
    wins = 0
    i = 20
    while i + CYCLE < len(px):
        S = float(px.iloc[i])
        sigma = float(rv.iloc[i])
        if not np.isfinite(sigma) or sigma <= 0:
            i += CYCLE
            continue
        iv = sigma * (1 + vrp)
        T = CYCLE / TRADING_DAYS
        K = strike_for_delta(S, T, iv, RFR, delta, call=False)
        prem = bs_price(S, K, T, iv, RFR, call=False)
        ST = float(px.iloc[i + CYCLE])
        intrinsic = max(0.0, K - ST)
        pnl = prem - intrinsic - (fee_per_contract_bps / 10_000.0) * S
        cycle_rets.append(pnl / K)           # 以担保金（行权价）为本金
        wins += 1 if pnl > 0 else 0
        i += CYCLE
    return _summarize(cycle_rets, wins)


def sim_short_strangle(px: pd.Series, vrp: float = 0.15, delta: float = 0.18,
                       fee_per_contract_bps: float = 2.0) -> dict:
    """宽跨式：同时卖 |delta|≈0.18 的 call 和 put。本金按行权价区间名义估算。"""
    rv = realized_vol(px)
    cycle_rets = []
    wins = 0
    i = 20
    while i + CYCLE < len(px):
        S = float(px.iloc[i])
        sigma = float(rv.iloc[i])
        if not np.isfinite(sigma) or sigma <= 0:
            i += CYCLE
            continue
        iv = sigma * (1 + vrp)
        T = CYCLE / TRADING_DAYS
        Kc = strike_for_delta(S, T, iv, RFR, delta, call=True)
        Kp = strike_for_delta(S, T, iv, RFR, delta, call=False)
        prem = bs_price(S, Kc, T, iv, RFR, call=True) + bs_price(S, Kp, T, iv, RFR, call=False)
        ST = float(px.iloc[i + CYCLE])
        intrinsic = max(0.0, ST - Kc) + max(0.0, Kp - ST)
        pnl = prem - intrinsic - (fee_per_contract_bps / 10_000.0) * S * 2
        cycle_rets.append(pnl / S)           # 以现价名义为本金（保证金近似）
        wins += 1 if pnl > 0 else 0
        i += CYCLE
    return _summarize(cycle_rets, wins)


def _summarize(cycle_rets: list[float], wins: int) -> dict:
    if not cycle_rets:
        return {}
    r = pd.Series(cycle_rets)
    eq = (1 + r).cumprod()
    n_cycles = len(r)
    years = n_cycles * CYCLE / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0
    cyc_per_year = TRADING_DAYS / CYCLE
    vol = r.std(ddof=0) * np.sqrt(cyc_per_year)
    sharpe = (r.mean() * cyc_per_year) / vol if vol > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    return {
        "年化": cagr,
        "夏普": sharpe,
        "最大回撤": dd,
        "胜率": wins / n_cycles,
        "最差单周期": r.min(),
        "平均周期收益": r.mean(),
        "周期数": n_cycles,
    }


def buy_hold(px: pd.Series) -> dict:
    r = px.pct_change().fillna(0)
    eq = px / px.iloc[0]
    years = len(r) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    return {"年化": cagr, "夏普": r.mean()*TRADING_DAYS/vol if vol>0 else 0,
            "最大回撤": (eq/eq.cummax()-1).min()}


def run():
    names = {
        "SPY": "标普500(低波基准)",
        "AAPL": "苹果(大盘蓝筹)",
        "NVDA": "英伟达(高波高量)",
        "TSLA": "特斯拉(高波高量)",
        "AMD": "AMD(高波高量)",
        "WDC": "西部数据(闪迪母体/高波)",
        "SNDK": "闪迪(2025上市·样本短)",
    }
    rows = []
    for t, label in names.items():
        try:
            df = yf.download(t, start="2016-01-01", end="2026-06-17", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            px = df["Close"].dropna()
            if len(px) < 60:
                print(f"{t}: 样本过短，跳过精细统计（{len(px)} 日）")
            rvmean = float(realized_vol(px).mean())
            csp = sim_short_put(px)
            strangle = sim_short_strangle(px)
            bh = buy_hold(px)
            rows.append({
                "标的": f"{t} {label}",
                "均RV": rvmean,
                "CSP年化": csp.get("年化"), "CSP夏普": csp.get("夏普"),
                "CSP回撤": csp.get("最大回撤"), "CSP胜率": csp.get("胜率"),
                "CSP最差周期": csp.get("最差单周期"),
                "宽跨年化": strangle.get("年化"), "宽跨夏普": strangle.get("夏普"),
                "宽跨回撤": strangle.get("最大回撤"), "宽跨最差周期": strangle.get("最差单周期"),
                "买入持有年化": bh.get("年化"),
            })
        except Exception as e:  # noqa: BLE001
            print(f"{t}: ERR {e}")
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    show = res.copy()
    for c in ["均RV","CSP年化","CSP回撤","CSP胜率","CSP最差周期","宽跨年化","宽跨回撤","宽跨最差周期","买入持有年化"]:
        show[c] = show[c].map(lambda x: f"{x:+.0%}" if pd.notna(x) else "-")
    for c in ["CSP夏普","宽跨夏普"]:
        show[c] = show[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    print(show.to_string(index=False))
    print("\n注：以上为 BS+VRP(15%) 近似模拟，IV=RV×1.15。真实卖方收益取决于实际 IV 水平、"
          "财报跳空、提前行权与点差，结果会有差异。")


if __name__ == "__main__":
    run()
