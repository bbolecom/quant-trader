"""SNDK（闪迪）日历价差(Calendar Spread)分析。

日历价差 = 卖近期期权 + 买远期期权（同行权价）。
赚的是「近期 theta 衰减快于远期」+「远期保留时间价值」。

关键风险（高 IV 股尤甚）：日历价差是 **net long vega**（净买入波动率）。
SNDK 现在 IV≈105%（极高），若近期到期时 IV 回落（IV crush），远期腿大幅贬值 → 亏损。
所以对超高 IV 股，日历价差「不是最稳」——最稳的场景是低/中 IV + 预期 IV 上升。

本脚本：
  1. 用 BS 定价当前 SNDK 日历价差的成本(debit)。
  2. 在「近期腿到期」时，扫描标的价格 × 远期腿剩余 IV，画盈亏，找盈利概率。
  3. 对比不同行权(ATM/OTM)、不同到期间隔，给出「最稳」配置。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RFR = 0.04


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(S, K, T, sig, r=RFR):
    if T <= 0 or sig <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def bs_put(S, K, T, sig, r=RFR):
    if T <= 0 or sig <= 0:
        return max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def calendar_debit(S, K, T_short, T_long, iv_short, iv_long, kind="call"):
    """开仓净支出 = 买远期 - 卖近期。"""
    f = bs_call if kind == "call" else bs_put
    long_leg = f(S, K, T_long, iv_long)
    short_leg = f(S, K, T_short, iv_short)
    return long_leg - short_leg, long_leg, short_leg


def calendar_value_at_short_expiry(ST, K, T_remain, iv_long_now, debit, kind="call"):
    """近期腿到期时的价值：远期腿(剩余 T_remain)市值 - 近期腿内在价值。"""
    f = bs_call if kind == "call" else bs_put
    long_val = f(ST, K, T_remain, iv_long_now)
    short_intrinsic = max(0.0, (ST - K) if kind == "call" else (K - ST))
    return long_val - short_intrinsic - debit  # 盈亏（已扣开仓成本）


def analyze(S=1991.55, rv=0.91, kind="put"):
    iv = rv * 1.05  # 估算隐含波动率
    print(f"{'='*78}\nSNDK 日历价差分析  现价 ${S:,.2f}  RV {rv*100:.0f}%  估算IV {iv*100:.0f}%\n{'='*78}")

    # 标准配置：卖30天 / 买60天，ATM
    T_short, T_long = 30 / 365, 60 / 365
    K = round(S / 5) * 5  # ATM 近似
    debit, ll, sl = calendar_debit(S, K, T_short, T_long, iv, iv, kind)
    print(f"\n【标准配置】ATM 日历价差（卖30天 + 买60天，K=${K:,.0f}，{kind}）")
    print(f"  买远期(60天) ${ll:.2f}  -  卖近期(30天) ${sl:.2f}  =  净支出 ${debit:.2f}/股 (${debit*100:,.0f}/张)")
    print(f"  最大亏损 = 净支出 ${debit*100:,.0f}（远期腿归零的极端情形）")

    # 近期到期时，扫描 ST × 远期腿IV（IV crush 情景）
    print(f"\n【近期腿到期时盈亏】行=标的价格，列=远期腿剩余IV情景")
    sts = [K * (1 + x) for x in [-0.25, -0.15, -0.08, 0, 0.08, 0.15, 0.25]]
    ivs = {"IV暴跌(50%)": iv * 0.5, "IV回落(75%)": iv * 0.75, "IV维持(100%)": iv, "IV上升(120%)": iv * 1.2}
    T_remain = T_long - T_short
    header = "  ST\\IV    " + "".join(f"{name:>14s}" for name in ivs)
    print(header)
    rows = {}
    for st in sts:
        line = f"  ${st:7,.0f} "
        for name, ivl in ivs.items():
            pnl = calendar_value_at_short_expiry(st, K, T_remain, ivl, debit, kind)
            rows.setdefault(name, []).append(pnl)
            line += f"{pnl*100:>+13,.0f} "
        print(line)

    # 盈利概率粗估：用对数正态分布在 T_short 上模拟 ST，IV 维持/回落两情景
    print(f"\n【盈利概率粗估】30天后标的分布(对数正态, σ={iv:.0%}) × IV情景")
    np.random.seed(0)
    n = 200_000
    drift = (RFR - 0.5 * iv ** 2) * T_short
    diff = iv * math.sqrt(T_short) * np.random.standard_normal(n)
    ST_sim = S * np.exp(drift + diff)
    for name, ivl in ivs.items():
        pnls = np.array([calendar_value_at_short_expiry(st, K, T_remain, ivl, debit, kind) for st in ST_sim[:20000]])
        winrate = (pnls > 0).mean()
        avg = pnls.mean()
        print(f"  {name:14s}: 盈利概率 {winrate:5.1%}   平均盈亏 ${avg*100:+,.0f}/张")

    # 不同到期间隔对比
    print(f"\n【配置对比】不同近/远到期间隔的成本与最大盈利点（IV维持情景）")
    for ts, tl in [(7, 30), (14, 45), (30, 60), (30, 90)]:
        d, _, _ = calendar_debit(S, K, ts/365, tl/365, iv, iv, kind)
        # 近期到期、ST=K 时的最大盈利
        max_pnl = calendar_value_at_short_expiry(K, K, (tl-ts)/365, iv, d, kind)
        roi = max_pnl / d if d > 0 else 0
        print(f"  卖{ts:2d}天/买{tl:2d}天: 成本 ${d*100:6,.0f}  ATM最大盈利 ${max_pnl*100:+6,.0f}  峰值ROI {roi:+.0%}")


if __name__ == "__main__":
    analyze(kind="put")
