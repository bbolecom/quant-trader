"""为 SNDK（闪迪）这类「强势上涨 + 极高 IV」高波动股，横向比较多种期权卖方策略，
找出历史上「最稳定收益」的那个。

关键认知：SNDK 当前是强趋势上行 + 极高 IV。在这种票上稳定收租，应顺势卖 **下方** 的 put
（put credit spread / CSP），而不是逆势卖 call 价差（会被暴涨打穿）。

⚠️ 期权价格为 Black-Scholes + 波动率风险溢价(VRP) 近似（拿不到历史期权链）；
SNDK 上市仅约 1 年，样本短，故同时用 WDC（闪迪母体，存储半导体，长历史含 2022 熊市）做稳健性验证。

每个策略按「重叠开仓」（每 5 个交易日开一笔、持有 ~35 天到期）累计大量交易，
用以下「稳定性」指标比较：
    胜率、平均单笔ROR、ROR标准差、信息比(均值/标准差)、最差单笔、合成净值最大回撤、年化。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252
HOLD_TD = 25          # 持有交易日数（≈35 日历天）
DTE_CAL = 35          # 定价用到期日历天
STEP = 5              # 每 5 个交易日开一笔（重叠）
RFR = 0.04
VRP = 0.10            # 保守的波动率风险溢价


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _ncdf_inv(p):
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


def bs_put(S, K, T, sig, r=RFR):
    if T <= 0 or sig <= 0:
        return max(0.0, K - S)
    d1 = (math.log(S/K) + (r + 0.5*sig**2)*T) / (sig*math.sqrt(T))
    d2 = d1 - sig*math.sqrt(T)
    return K*math.exp(-r*T)*_ncdf(-d2) - S*_ncdf(-d1)


def bs_call(S, K, T, sig, r=RFR):
    if T <= 0 or sig <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S/K) + (r + 0.5*sig**2)*T) / (sig*math.sqrt(T))
    d2 = d1 - sig*math.sqrt(T)
    return S*_ncdf(d1) - K*math.exp(-r*T)*_ncdf(d2)


def put_strike(S, T, sig, delta, r=RFR):
    d1 = _ncdf_inv(delta)
    return S*math.exp((r + 0.5*sig**2)*T + d1*sig*math.sqrt(T))


def call_strike(S, T, sig, delta, r=RFR):
    d1 = _ncdf_inv(delta)
    return S*math.exp((r + 0.5*sig**2)*T - d1*sig*math.sqrt(T))


def realized_vol(close, window=20):
    lr = np.log(close / close.shift(1))
    return lr.rolling(window).std(ddof=0) * np.sqrt(TRADING_DAYS)


def _summ(rors: list[float], name: str) -> dict:
    if not rors:
        return {"策略": name, "交易数": 0}
    r = pd.Series(rors)
    eq = (1 + r * 0.2).cumprod()  # 每笔用 20% 资金（重叠 5 笔在途），合成净值看回撤
    dd = (eq / eq.cummax() - 1).min()
    info = r.mean() / r.std(ddof=0) if r.std(ddof=0) > 0 else 0.0
    # 年化：平均单笔 ROR × 每年开仓次数（顺序非重叠口径）
    cycles_per_year = TRADING_DAYS / HOLD_TD
    ann = (1 + r.mean()) ** cycles_per_year - 1
    return {
        "策略": name,
        "交易数": len(r),
        "胜率": float((r > 0).mean()),
        "平均ROR": float(r.mean()),
        "ROR标准差": float(r.std(ddof=0)),
        "信息比": float(info),
        "最差单笔": float(r.min()),
        "年化(近似)": float(ann),
        "合成回撤": float(dd),
    }


def backtest(close: pd.Series, vrp: float = VRP):
    close = close.astype(float).dropna()
    rv = realized_vol(close)
    T = DTE_CAL / TRADING_DAYS
    res = {k: [] for k in ["PCS", "CSP", "CC", "BCS", "IC"]}
    i = 25
    while i + HOLD_TD < len(close):
        S = float(close.iloc[i]); ST = float(close.iloc[i + HOLD_TD]); sig = float(rv.iloc[i])
        if not np.isfinite(sig) or sig <= 0:
            i += STEP; continue
        iv = sig * (1 + vrp)

        # --- Put Credit Spread（顺势，卖0.25put 买0.10put）---
        ks = put_strike(S, T, iv, 0.25); kl = put_strike(S, T, iv, 0.10)
        credit = bs_put(S, ks, T, iv) - bs_put(S, kl, T, iv)
        loss = max(0.0, ks - ST) - max(0.0, kl - ST)
        width = ks - kl
        if width > 0:
            res["PCS"].append((credit - loss) / width)

        # --- CSP（卖0.25put 现金担保）---
        kp = put_strike(S, T, iv, 0.25); cp = bs_put(S, kp, T, iv)
        res["CSP"].append((cp - max(0.0, kp - ST)) / kp)

        # --- Covered Call（持股 卖0.30call）---
        kc = call_strike(S, T, iv, 0.30); cc = bs_call(S, kc, T, iv)
        stock = (kc - S) if ST > kc else (ST - S)
        res["CC"].append((stock + cc) / S)

        # --- Bear Call Spread（逆势，卖0.25call 买0.10call）---
        kcs = call_strike(S, T, iv, 0.25); kcl = call_strike(S, T, iv, 0.10)
        ccredit = bs_call(S, kcs, T, iv) - bs_call(S, kcl, T, iv)
        closs = max(0.0, ST - kcs) - max(0.0, ST - kcl)
        cwidth = kcl - kcs
        if cwidth > 0:
            res["BCS"].append((ccredit - closs) / cwidth)

        # --- Iron Condor（双卖价差）---
        if width > 0 and cwidth > 0:
            ic_credit = credit + ccredit
            ic_loss = loss + closs
            ic_margin = max(width, cwidth)
            res["IC"].append((ic_credit - ic_loss) / ic_margin)

        i += STEP

    names = {"PCS": "认沽信用价差(顺势)", "CSP": "现金担保认沽", "CC": "备兑Call(持股)",
             "BCS": "认购信用价差(逆势)", "IC": "铁鹰(双卖)"}
    rows = [_summ(res[k], names[k]) for k in res]
    return pd.DataFrame(rows)


def fetch(ticker, start, end="2026-06-17"):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def show(ticker, start, note=""):
    df = fetch(ticker, start)
    print(f"\n{'='*92}\n{ticker} {note}（{df.index[0].date()} ~ {df.index[-1].date()}，{len(df)} 日）\n{'='*92}")
    tbl = backtest(df["Close"])
    disp = tbl.copy()
    for c in ["胜率", "平均ROR", "ROR标准差", "最差单笔", "年化(近似)", "合成回撤"]:
        disp[c] = disp[c].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "-")
    disp["信息比"] = disp["信息比"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    print(disp.to_string(index=False))
    return tbl


if __name__ == "__main__":
    show("SNDK", "2025-02-01", "闪迪（样本短，仅供当前参考）")
    show("WDC", "2018-01-01", "西部数据=闪迪母体（长样本，主稳健性证据）")
    show("WDC", "2022-01-01", "WDC 2022熊市压力测试")
    show("MU", "2018-01-01", "美光（同类存储，交叉验证）")
