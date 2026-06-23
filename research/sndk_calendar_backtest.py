"""SNDK 双日历价差历史回测：卖14天/买21天，行权 S±offset(call/put 各一)，持有7天平仓。

判断"颇丰收益"是可持续(结构性 theta)还是运气(IV 上升带来的 vega)：
  每笔 PnL 拆成
    · theta 腿 = 假设 IV 不变时的盈亏（近月衰减快于远月 → 结构性、可重复）
    · vega 腿  = IV 变化带来的额外盈亏（IV 升=赚，IV 崩=亏 → 择时/运气）
  若收益主要来自 theta 且各时期稳定为正 → 可持续；若主要靠 vega 或集中在某段 → 运气。

偏移：用户用固定 ±$250。SNDK 价格历史跨度大，固定美元在低价期等于极深虚值，
故按"$250 ÷ 现价"换成百分比(≈±11.4%)等比回测，更有可比性（可用 --offset-dollar 关闭）。

用法：
    python research/sndk_calendar_backtest.py
    python research/sndk_calendar_backtest.py --ticker SNDK --offset 250 --hold 5
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.providers import DataConfig, get_provider, reset_provider_cache

RFR = 0.045


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(S, K, T, sig, typ):
    if T <= 0 or sig <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if typ == "c" else (K - S))
    d1 = (math.log(S / K) + (RFR + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if typ == "c":
        return S * _ncdf(d1) - K * math.exp(-RFR * T) * _ncdf(d2)
    return K * math.exp(-RFR * T) * _ncdf(-d2) - S * _ncdf(-d1)


def double_calendar_value(S, Kc, Kp, T_short, T_long, iv):
    """双日历(call@Kc + put@Kp)的市值 = (买远月 - 卖近月) 两侧之和。"""
    call_cal = bs(S, Kc, T_long, iv, "c") - bs(S, Kc, T_short, iv, "c")
    put_cal = bs(S, Kp, T_long, iv, "p") - bs(S, Kp, T_short, iv, "p")
    return call_cal + put_cal


def backtest_records(c: pd.Series, offset, offset_dollar, hold_td, short_d, long_d,
                     iv_mult, step, k_sigma=None, iv_crush=None,
                     start_date=None, end_date=None,
                     iv_pct_max=None, iv_window=252) -> pd.DataFrame:
    """对单只收盘序列跑双日历回测，返回逐笔记录 DataFrame。
    k_sigma 不为 None 时：按各自"周波动率×k"放置行权价(跨票公平对比)；否则用固定%/$偏移。
    iv_crush 不为 None 时：强制平仓 IV = 入场 IV × iv_crush(模拟 IV 崩塌)。
    start_date/end_date：仅统计入场日落在该区间的交易(压力期切片)。
    iv_pct_max 不为 None 时：仅在入场 IV 处于过去 iv_window 天的 ≤该分位 时开仓。"""
    c = c.astype(float).dropna()
    if len(c) < 60:
        return pd.DataFrame()
    rv = c.pct_change(fill_method=None).rolling(20).std() * math.sqrt(252)
    iv_pct = rv.rolling(iv_window, min_periods=60).apply(
        lambda w: (w.iloc[-1] >= w).mean(), raw=False) if iv_pct_max is not None else None
    S_now = float(c.iloc[-1])
    off_pct = offset / S_now if not offset_dollar else None
    Ts_e, Tl_e = short_d / 365, long_d / 365
    Ts_x, Tl_x = (short_d - 7) / 365, (long_d - 7) / 365
    sd = pd.Timestamp(start_date) if start_date else None
    ed = pd.Timestamp(end_date) if end_date else None
    recs = []
    for i in range(25, len(c) - hold_td, step):
        if sd is not None and c.index[i] < sd:
            continue
        if ed is not None and c.index[i] > ed:
            continue
        if iv_pct is not None:
            p = iv_pct.iloc[i]
            if not np.isfinite(p) or p > iv_pct_max:
                continue
        S0 = float(c.iloc[i]); iv0 = float(rv.iloc[i]) * iv_mult
        S7 = float(c.iloc[i + hold_td]); iv7 = float(rv.iloc[i + hold_td]) * iv_mult
        if iv_crush is not None:
            iv7 = iv0 * iv_crush
        if not (np.isfinite(iv0) and np.isfinite(iv7)) or iv0 <= 0 or iv7 <= 0:
            continue
        if k_sigma is not None:
            wsig = iv0 * math.sqrt(7 / 365)          # 入场时1周波动率
            Kc, Kp = S0 * (1 + k_sigma * wsig), S0 * (1 - k_sigma * wsig)
        elif offset_dollar:
            Kc, Kp = S0 + offset, S0 - offset
        else:
            Kc, Kp = S0 * (1 + off_pct), S0 * (1 - off_pct)
        entry = double_calendar_value(S0, Kc, Kp, Ts_e, Tl_e, iv0)
        # 丢弃退化交易：净借方过小(<现价0.3%)时 pnl/entry 会爆炸且不可成交
        if entry <= 0 or entry < 0.003 * S0:
            continue
        exit_const = double_calendar_value(S7, Kc, Kp, Ts_x, Tl_x, iv0)
        exit_real = double_calendar_value(S7, Kc, Kp, Ts_x, Tl_x, iv7)
        recs.append({
            "date": c.index[i + hold_td], "ret": (exit_real - entry) / entry,
            "pnl": (exit_real - entry) * 100, "theta": (exit_const - entry) * 100,
            "vega": (exit_real - exit_const) * 100, "entry": entry * 100,
            "div": iv7 - iv0, "signed_move": S7 / S0 - 1, "move": abs(S7 / S0 - 1),
        })
    return pd.DataFrame(recs).set_index("date") if recs else pd.DataFrame()


def _verdict(r: pd.DataFrame) -> str:
    share = abs(r['vega'].mean()) / (abs(r['theta'].mean()) + abs(r['vega'].mean()) + 1e-9)
    cc = r["ret"].corr(r["div"])
    if r['theta'].mean() > 0 and (r['theta'] > 0).mean() > 0.58 and share < 0.5:
        return "✅ 偏可持续(theta主导)"
    if share >= 0.5 or (cc is not None and cc > 0.45):
        return "⚠ 偏运气(靠IV/vega)"
    return "⚖ 混合(依赖IV环境)"


def run_single(ticker, c, offset, offset_dollar, hold_td, short_d, long_d, iv_mult, step):
    r = backtest_records(c, offset, offset_dollar, hold_td, short_d, long_d, iv_mult, step)
    S_now = float(c.iloc[-1]); off_pct = offset / S_now
    print("=" * 86)
    print(f"{ticker} 双日历回测｜卖{short_d}天/买{long_d}天，持有≈{hold_td}个交易日平仓")
    print(f"现价 ${S_now:,.0f}｜偏移 {'固定 $'+format(offset,',.0f') if offset_dollar else f'±{off_pct*100:.1f}%'}"
          f"｜IV≈RV×{iv_mult}｜历史 {c.index[0].date()} ~ {c.index[-1].date()}")
    print("=" * 86)
    if r.empty or len(r) < 20:
        print("样本太少，无法判断。")
        return
    share = abs(r['vega'].mean()) / (abs(r['theta'].mean()) + abs(r['vega'].mean()) + 1e-9)
    print(f"样本 {len(r)} 笔｜开仓均成本 ${r['entry'].mean():,.0f}/张")
    print(f"  总胜率 {(r['pnl']>0).mean()*100:.0f}%｜单笔均收益率 {r['ret'].mean()*100:+.1f}%(中位 {r['ret'].median()*100:+.1f}%)｜均盈亏 ${r['pnl'].mean():+,.0f}/张")
    print(f"  theta(可持续) ${r['theta'].mean():+,.0f}/张(胜率{(r['theta']>0).mean()*100:.0f}%)｜vega(IV/运气) ${r['vega'].mean():+,.0f}/张｜vega占比 {share*100:.0f}%")

    # 方向分桶：回答"持续下跌(大幅移动)是否不可持续"
    print("\n  按 7 天后价格移动分桶（平均盈亏/张）：")
    bins = [(-9, -0.08, "大跌<-8%"), (-0.08, -0.03, "跌 -8~-3%"), (-0.03, 0.03, "横盘 ±3%"),
            (0.03, 0.08, "涨 3~8%"), (0.08, 9, "大涨>8%")]
    for lo, hi, name in bins:
        sub = r[(r["signed_move"] > lo) & (r["signed_move"] <= hi)]
        if len(sub):
            print(f"    {name:<12} n={len(sub):<4} 均盈亏 ${sub['pnl'].mean():+6,.0f}  胜率 {(sub['pnl']>0).mean()*100:.0f}%")

    cm = r["ret"].corr(r["move"]); cc = r["ret"].corr(r["div"])
    print(f"\n  收益 vs |价格移动| 相关 {cm:+.2f}（负=越大动越亏）；收益 vs IV变化 相关 {cc:+.2f}")
    print(f"  判定：{_verdict(r)}")
    print("\n注：BS 近似(两腿同 IV，未含 skew/期限结构)；样本期短、集中于该票特定行情，代表性有限；")
    print("    财报跳空与买卖价差在实盘会显著吃掉薄利。")


def run_basket(tickers, offset, offset_dollar, hold_td, short_d, long_d, iv_mult, step, k_sigma):
    reset_provider_cache()
    y = get_provider(DataConfig(provider="yahoo"))
    end = date.today().isoformat(); start = (date.today() - timedelta(days=1100)).isoformat()
    print(f"批量回测 {len(tickers)} 只流动性票｜抓取 {start} ~ {end} …")
    batch = y.fetch_batch(tickers, start, end)
    rows = []
    for t in tickers:
        d = batch.get(t)
        if d is None or d.empty:
            continue
        r = backtest_records(d["Close"], offset, offset_dollar, hold_td, short_d, long_d, iv_mult, step, k_sigma)
        if r.empty or len(r) < 30:
            continue
        share = abs(r['vega'].mean()) / (abs(r['theta'].mean()) + abs(r['vega'].mean()) + 1e-9)
        retc = r['ret'].clip(-1.0, 3.0)  # 截尾，避免极端退化值污染均值
        rows.append({
            "票": t, "笔数": len(r), "胜率%": round((r['pnl'] > 0).mean() * 100, 0),
            "中位收益%": round(r['ret'].median() * 100, 1),
            "均收益%(截尾)": round(retc.mean() * 100, 1),
            "theta$": round(r['theta'].mean(), 0), "vega占比%": round(share * 100, 0),
            "大跌时$": round(r[r['signed_move'] <= -0.08]['pnl'].mean(), 0) if (r['signed_move'] <= -0.08).any() else np.nan,
            "判定": _verdict(r),
        })
    if not rows:
        print("无足够样本。")
        return
    df = pd.DataFrame(rows).sort_values("中位收益%", ascending=False)
    off_desc = f"±{k_sigma:.1f}周σ(各票自适应)" if k_sigma is not None else (f"±${offset:,.0f}" if offset_dollar else "±11.4%")
    print("=" * 100)
    print(f"双日历·多票对比｜卖{short_d}/买{long_d}天 持有{hold_td}日 偏移{off_desc} IV≈RV×{iv_mult}")
    print("=" * 100)
    print(df.to_string(index=False))
    pos = (df["中位收益%"] > 0).mean() * 100
    print(f"\n  {pos:.0f}% 的票中位收益为正；'大跌时$' 普遍为负 → 持续大跌(大幅移动)时此结构会亏。")
    print("  解读：theta$ 多为正=结构性时间价值在多数票上成立(非 SNDK 独有)；但靠'横盘/小动'，")
    print("        大跌或大涨都伤；vega 占比高的票更依赖 IV 环境(运气成分大)。")
    print("\n注：BS 近似、样本期内多为科技牛市，含幸存者偏差；实盘扣买卖价差/滑点后薄利会缩水。")


DEFAULT_BASKET = ["NVDA", "TSLA", "AAPL", "AMD", "META", "AMZN", "MSFT", "GOOGL",
                  "MU", "AVGO", "NFLX", "COIN", "PLTR", "MSTR", "SMCI", "QQQ", "SPY"]


def _stress_table(batch, tickers, hold_td, short_d, long_d, iv_mult, step, k_sigma,
                  iv_crush, start_date, end_date):
    rows = []
    for t in tickers:
        d = batch.get(t)
        if d is None or d.empty:
            continue
        r = backtest_records(d["Close"], 0, False, hold_td, short_d, long_d, iv_mult,
                             step, k_sigma, iv_crush, start_date, end_date)
        if r.empty or len(r) < 15:
            continue
        rows.append({
            "票": t, "笔数": len(r), "胜率%": round((r['pnl'] > 0).mean() * 100, 0),
            "中位收益%": round(r['ret'].median() * 100, 1),
            "均收益%(截尾)": round(r['ret'].clip(-1.0, 3.0).mean() * 100, 1),
            "均盈亏$": round(r['pnl'].mean(), 0),
            "最差单笔$": round(r['pnl'].min(), 0),
        })
    return pd.DataFrame(rows)


def run_ivfilter(tickers, hold_td, short_d, long_d, iv_mult, step, k_sigma, iv_pct_max):
    reset_provider_cache()
    y = get_provider(DataConfig(provider="yahoo"))
    batch = y.fetch_batch(tickers, "2021-06-01", date.today().isoformat())
    rows = []
    for t in tickers:
        d = batch.get(t)
        if d is None or d.empty:
            continue
        cl = d["Close"]
        base = backtest_records(cl, 0, False, hold_td, short_d, long_d, iv_mult, step, k_sigma)
        filt = backtest_records(cl, 0, False, hold_td, short_d, long_d, iv_mult, step,
                               k_sigma, iv_pct_max=iv_pct_max)
        # 同样的"低IV才开"条件下，再叠加 IV 砍半，看尾部是否被驯服
        filt_crush = backtest_records(cl, 0, False, hold_td, short_d, long_d, iv_mult, step,
                                      k_sigma, iv_crush=0.5, iv_pct_max=iv_pct_max)
        base_crush = backtest_records(cl, 0, False, hold_td, short_d, long_d, iv_mult, step,
                                      k_sigma, iv_crush=0.5)
        if base.empty or filt.empty or len(filt) < 10:
            continue
        rows.append({
            "票": t,
            "全开_笔": len(base), "全开$": round(base['pnl'].mean(), 0),
            "低IV_笔": len(filt), "低IV$": round(filt['pnl'].mean(), 0),
            "低IV胜%": round((filt['pnl'] > 0).mean() * 100, 0),
            "全开+崩$": round(base_crush['pnl'].mean(), 0) if not base_crush.empty else np.nan,
            "低IV+崩$": round(filt_crush['pnl'].mean(), 0) if not filt_crush.empty else np.nan,
        })
    if not rows:
        print("无足够样本。")
        return
    df = pd.DataFrame(rows).sort_values("低IV$", ascending=False)
    print("=" * 104)
    print(f"IV 百分位过滤器｜只在入场 IV ≤ 过去1年第 {iv_pct_max*100:.0f} 分位时开仓")
    print(f"  双日历 卖{short_d}/买{long_d}天 持有{hold_td}日 行权价±{k_sigma:.1f}周σ IV≈RV×{iv_mult}")
    print("=" * 104)
    print(df.to_string(index=False))
    print("\n  含义：'$'=单笔平均盈亏/张；'+崩'=平仓时 IV 砍半的同场景。")
    print(f"  低IV 开仓中位均盈亏 ${df['低IV$'].median():+,.0f}/张 vs 全开 ${df['全开$'].median():+,.0f}/张")
    print(f"  IV 崩塌下：全开中位 ${df['全开+崩$'].median():+,.0f}  →  低IV过滤后 ${df['低IV+崩$'].median():+,.0f}/张")
    impr = df['低IV+崩$'].median() - df['全开+崩$'].median()
    print("\n【解读】")
    print(f"  • IV 崩塌的尾部被显著缓解：低分位入场时 IV 本就低，'再砍半'空间有限，损失从巨亏收敛(改善≈${impr:+,.0f}/张)。")
    print("  • 代价：开仓次数大幅减少(只在 IV 低位才出手)，年化交易频率下降，但每笔更安全。")
    print("  • 这正是把策略从'赌 IV 不崩'改成'只在 IV 已经低、没多少可崩时才收租'——风险收益结构更健康。")
    print("\n注：BS 近似、IV 用 RV 代理且 RV 分位≠真实 IV 分位；实盘请用期权链真实 IV Rank/IV Percentile。")


def run_stress(tickers, hold_td, short_d, long_d, iv_mult, step, k_sigma):
    reset_provider_cache()
    y = get_provider(DataConfig(provider="yahoo"))
    # 抓 2021 中到现在，覆盖 2022 全年熊市 + 当前
    batch = y.fetch_batch(tickers, "2021-06-01", date.today().isoformat())

    print("=" * 100)
    print("压力测试 A｜2022 全年真实熊市（SPY −19%，纳指 −33%，单边下跌+高 IV）")
    print(f"  双日历 卖{short_d}/买{long_d}天 持有{hold_td}日 行权价±{k_sigma:.1f}周σ IV≈RV×{iv_mult}")
    print("=" * 100)
    bear = _stress_table(batch, tickers, hold_td, short_d, long_d, iv_mult, step,
                         k_sigma, None, "2022-01-01", "2022-12-31")
    if bear.empty:
        print("  (无足够 2022 样本)")
    else:
        bear = bear.sort_values("均盈亏$", ascending=False)
        print(bear.to_string(index=False))
        print(f"\n  2022 熊市里 {(bear['均盈亏$']>0).mean()*100:.0f}% 的票均盈亏为正；"
              f"组合中位均盈亏 ${bear['均盈亏$'].median():+,.0f}/张，最差单笔可达 ${bear['最差单笔$'].min():+,.0f}。")

    print("\n" + "=" * 100)
    print("压力测试 B｜IV 崩塌（平仓时 IV = 入场 IV × 0.5，模拟暴涨见顶后 IV crush）")
    print("=" * 100)
    crush = _stress_table(batch, tickers, hold_td, short_d, long_d, iv_mult, step,
                          k_sigma, 0.5, None, None)
    if crush.empty:
        print("  (无足够样本)")
    else:
        crush = crush.sort_values("均盈亏$", ascending=False)
        print(crush.to_string(index=False))
        print(f"\n  IV 砍半后 {(crush['均盈亏$']>0).mean()*100:.0f}% 的票仍为正；"
              f"组合中位均盈亏 ${crush['均盈亏$'].median():+,.0f}/张。")

    print("\n【解读】")
    print("  • 日历是 long vega 结构：熊市初期 IV 上冲其实利好它(vega 救场)，能抵消一部分单边下跌的损失；")
    print("    真正杀它的是 IV 崩塌——见顶/财报后 IV 砍半，long vega 直接反噬。")
    print("  • 所以最危险的不是'持续下跌'本身，而是'见顶后 IV 大幅回落'。SNDK 暴涨期一旦 IV 见顶，")
    print("    这套结构会从顺风变逆风——而你的实盘样本恰好全在 IV 上行/高位区，没经历过这一段。")
    print("\n注：BS 近似、IV 用 RV 代理；真实 skew/期限结构会让 IV crush 的伤害更不均匀。")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SNDK")
    p.add_argument("--basket", action="store_true", help="批量回测一篮子流动性票")
    p.add_argument("--stress", action="store_true", help="压力测试：2022熊市 + IV砍半两种场景")
    p.add_argument("--iv-filter", action="store_true", help="IV百分位过滤器：低IV才开仓 + 抗崩塌对比")
    p.add_argument("--iv-pct-max", type=float, default=0.4, help="入场IV分位上限(0~1)，默认0.4")
    p.add_argument("--tickers", default="", help="自定义票列表(逗号分隔)，配合 --basket")
    p.add_argument("--offset", type=float, default=250.0)
    p.add_argument("--offset-dollar", action="store_true", help="用固定美元偏移(默认按等比%)")
    p.add_argument("--hold", type=int, default=5, help="持有交易日(≈7自然日)")
    p.add_argument("--short", type=int, default=14)
    p.add_argument("--long", type=int, default=21)
    p.add_argument("--iv-mult", type=float, default=1.1)
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--k-sigma", type=float, default=1.0,
                   help="篮子模式行权价偏移=入场周波动率×k(跨票公平，≈SNDK的±$250)；设0用固定%")
    args = p.parse_args()
    if args.iv_filter:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or DEFAULT_BASKET
        k = args.k_sigma if args.k_sigma and args.k_sigma > 0 else 1.0
        run_ivfilter(tickers, args.hold, args.short, args.long, args.iv_mult, args.step, k, args.iv_pct_max)
    elif args.stress:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or DEFAULT_BASKET
        k = args.k_sigma if args.k_sigma and args.k_sigma > 0 else 1.0
        run_stress(tickers, args.hold, args.short, args.long, args.iv_mult, args.step, k)
    elif args.basket:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or DEFAULT_BASKET
        k = args.k_sigma if args.k_sigma and args.k_sigma > 0 else None
        run_basket(tickers, args.offset, args.offset_dollar, args.hold,
                   args.short, args.long, args.iv_mult, args.step, k)
    else:
        reset_provider_cache()
        y = get_provider(DataConfig(provider="yahoo"))
        end = date.today().isoformat(); start = (date.today() - timedelta(days=1100)).isoformat()
        c = y.fetch_history(args.ticker, start, end)["Close"]
        run_single(args.ticker, c, args.offset, args.offset_dollar, args.hold,
                   args.short, args.long, args.iv_mult, args.step)


if __name__ == "__main__":
    main()
