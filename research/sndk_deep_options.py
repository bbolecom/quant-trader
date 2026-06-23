"""SNDK（闪迪）深度期权玩法分析：Deep ITM / Deep OTM / PMCC。

闪迪现价≈$1,991、IV≈96%（极高）、单价高 → 直接买100股要$199k。
深度期权的意义：用更少钱获得类似敞口，或卖租。

玩法：
  1. Deep ITM Call（delta≈0.85）当「股票替代」：少占钱、有杠杆，但高 IV 时含较多时间价值。
  2. PMCC 穷人备兑 = 买 Deep ITM 远期 Call + 卖近期 OTM Call 收租（资金效率高）。
  3. Deep OTM：买=彩票（高 IV 下很贵、大概率归零）；卖=收尾部租（担保/价差）。

关键提醒：SNDK IV 极高 →「买」期权（Deep ITM/OTM）都偏贵且怕 IV crush；
「卖」方向更顺这只票的高 IV。
"""

from __future__ import annotations

import math

TRADING_DAYS = 252
RFR = 0.04


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


def bs_call(S, K, T, sig, r=RFR):
    if T <= 0 or sig <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def call_delta(S, K, T, sig, r=RFR):
    if T <= 0 or sig <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * math.sqrt(T))
    return _ncdf(d1)


def strike_for_call_delta(S, T, sig, delta, r=RFR):
    d1 = _ncdf_inv(delta)
    return S * math.exp((r + 0.5 * sig ** 2) * T - d1 * sig * math.sqrt(T))


def main(S=1991.55, rv=0.91):
    iv = rv * 1.05
    print(f"{'='*80}\nSNDK 深度期权分析  现价 ${S:,.2f}  RV {rv*100:.0f}%  估算IV {iv*100:.0f}%\n{'='*80}")
    print(f"\n直接买 100 股成本 = ${S*100:,.0f}（基准）")

    # 1. Deep ITM Call 当股票替代
    print(f"\n{'─'*80}\n【玩法1】Deep ITM Call 当股票替代（delta≈0.85，90天）")
    T = 90 / 365
    for d in [0.90, 0.85, 0.80]:
        K = strike_for_call_delta(S, T, iv, d)
        price = bs_call(S, K, T, iv)
        intrinsic = max(0.0, S - K)
        tv = price - intrinsic
        print(f"  Delta {d}: 买Call K=${K:,.0f}  权利金 ${price:.0f}/股 (${price*100:,.0f}/张)  "
              f"内在${intrinsic:.0f} 时间价值${tv:.0f}  占买股 {price/S*100:.0f}%")
    print("  ⚠ 高IV下时间价值很厚（白付的成本），且到期前 IV 回落会侵蚀价值。")

    # 2. PMCC 穷人备兑
    print(f"\n{'─'*80}\n【玩法2】PMCC 穷人备兑 = 买远期Deep ITM Call + 卖近期OTM Call 收租")
    T_long = 180 / 365
    T_short = 30 / 365
    K_long = strike_for_call_delta(S, T_long, iv, 0.80)
    long_price = bs_call(S, K_long, T_long, iv)
    K_short = strike_for_call_delta(S, T_short, iv, 0.30)
    short_price = bs_call(S, K_short, T_short, iv)
    net = long_price - short_price
    print(f"  买  180天 Call K=${K_long:,.0f} (Δ0.80)  付 ${long_price*100:,.0f}/张")
    print(f"  卖   30天 Call K=${K_short:,.0f} (Δ0.30)  收 ${short_price*100:,.0f}/张")
    print(f"  净成本 ${net*100:,.0f}/张（vs 买100股 ${S*100:,.0f}，省 {(1-net/S)*100:.0f}%）")
    print(f"  每月卖近期 Call 收租 ≈ ${short_price*100:,.0f}（约成本的 {short_price/net*100:.1f}%/月）")
    print(f"  上方风险：股价涨破 ${K_short:,.0f} → 近期腿亏损被远期腿盈利覆盖（价差封顶）")

    # 3. Deep OTM
    print(f"\n{'─'*80}\n【玩法3】Deep OTM 期权")
    T = 30 / 365
    print("  (a) 买 Deep OTM Call（彩票）:")
    for d in [0.10, 0.05]:
        K = strike_for_call_delta(S, T, iv, d)
        price = bs_call(S, K, T, iv)
        print(f"      Δ{d}: K=${K:,.0f}(距现价+{(K/S-1)*100:.0f}%)  花 ${price*100:,.0f}/张  "
              f"约{d*100:.0f}%概率到价，大概率归零")
    print("  (b) 卖 Deep OTM Put（收尾部租）= 就是低Delta CSP，见之前CSP方案。")

    # 4. 对 $10k 账户可行性
    print(f"\n{'─'*80}\n【$10,000 账户可行性】")
    pmcc_net = net * 100
    print(f"  PMCC 净成本 ${pmcc_net:,.0f}/张  → 占 $10k 的 {pmcc_net/10000*100:.0f}%  "
          f"{'❌ 不够' if pmcc_net>10000 else '✅'}")
    print(f"  Deep ITM Call(Δ0.85,90天) ${bs_call(S, strike_for_call_delta(S,90/365,iv,0.85),90/365,iv)*100:,.0f}/张 "
          f"→ ❌ 远超 $10k")
    print(f"  Deep OTM Call(Δ0.10) 花 ${bs_call(S, strike_for_call_delta(S,30/365,iv,0.10),30/365,iv)*100:,.0f}/张 "
          f"→ 买得起但是彩票（大概率归零）")


if __name__ == "__main__":
    main()
