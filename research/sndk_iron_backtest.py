"""SNDK 铁鹰回测：匹配舰队真实链参数（OTM% + 价差宽% + 短 DTE）。

⚠️ 期权用 BS+VRP 近似（无历史期权链）；价格路径用 OHLC 判断「击穿」。
重点输出：胜率、Call/Put 击穿率、50% 止盈率、最大亏笔、1σ 移动 vs 行权距离。

用法:
    python research/sndk_iron_backtest.py
    python research/sndk_iron_backtest.py --ticker WDC --start 2018-01-01
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.decline_income import bs_call_price, bs_put_price, realized_vol
from quant.vol_decay import DEFAULT_VRP, RFR, TRADING_DAYS

# 舰队账户1/2 参数（sndk_iron_config.json）
FLEET_PROFILES = [
    {"label": "账户1", "call_otm": 0.15, "put_otm": 0.12, "width_pct": 0.02},
    {"label": "账户2", "call_otm": 0.20, "put_otm": 0.15, "width_pct": 0.02},
]

DTE_TD = 5          # 持有交易日（≈ 4–7 日历天周期权）
STEP_TD = 5         # 每 5 日开一笔（重叠）
TAKE_PROFIT = 0.50  # 50% 权利金止盈
VRP = DEFAULT_VRP
STRIKE_GRID = 10.0  # SNDK 行权价 $10 档


@dataclass
class IronTrade:
    entry_date: str
    spot: float
    expiry_spot: float
    cs: float
    cl: float
    ps: float
    pl: float
    credit: float
    margin: float
    pnl: float
    pnl_pct_margin: float
    outcome: str
    call_breach: bool
    put_breach: bool
    touch_call: bool
    touch_put: bool
    tp_hit: bool
    rv_pct: float
    call_otm_pct: float
    put_otm_pct: float
    one_sigma_pct: float


def _round_strike(x: float, grid: float = STRIKE_GRID) -> float:
    if grid <= 0:
        return round(x, 2)
    return round(x / grid) * grid


def _iron_strikes(S: float, *, call_otm: float, put_otm: float, width_pct: float) -> tuple[float, float, float, float]:
    cs = _round_strike(S * (1 + call_otm))
    cl = _round_strike(cs * (1 + width_pct))
    if cl <= cs:
        cl = cs + max(STRIKE_GRID, S * width_pct)
    ps = _round_strike(S * (1 - put_otm))
    pl = _round_strike(ps * (1 - width_pct))
    if pl >= ps:
        pl = max(0.0, ps - max(STRIKE_GRID, S * width_pct))
    return cs, cl, ps, pl


def _iron_credit(S: float, cs: float, cl: float, ps: float, pl: float, iv: float, T: float) -> float:
    call_cr = bs_call_price(S, cs, T, iv) - bs_call_price(S, cl, T, iv)
    put_cr = bs_put_price(S, ps, T, iv) - bs_put_price(S, pl, T, iv)
    return max(0.0, call_cr + put_cr)


def _iron_pnl(ST: float, credit: float, cs: float, cl: float, ps: float, pl: float) -> float:
    call_loss = max(0.0, ST - cs) - max(0.0, ST - cl)
    put_loss = max(0.0, ps - ST) - max(0.0, pl - ST)
    return credit - call_loss - put_loss


def _mark_iron(Sj: float, credit: float, cs: float, cl: float, ps: float, pl: float, T_remain: float, iv: float) -> float:
    """持仓 Mark-to-market（BS，同 IV）。"""
    if T_remain <= 1 / 365:
        return _iron_pnl(Sj, credit, cs, cl, ps, pl)
    call_loss = bs_call_price(Sj, cs, T_remain, iv) - bs_call_price(Sj, cl, T_remain, iv)
    put_loss = bs_put_price(Sj, ps, T_remain, iv) - bs_put_price(Sj, pl, T_remain, iv)
    return credit - call_loss - put_loss


def backtest_iron(
    df: pd.DataFrame,
    *,
    call_otm: float = 0.15,
    put_otm: float = 0.12,
    width_pct: float = 0.02,
    dte_td: int = DTE_TD,
    step_td: int = STEP_TD,
    take_profit: float = TAKE_PROFIT,
    use_ma50: bool = False,
) -> list[IronTrade]:
    close = df["Close"].astype(float)
    high = df["High"].astype(float) if "High" in df.columns else close
    low = df["Low"].astype(float) if "Low" in df.columns else close
    rv = realized_vol(close)
    ma50 = close.rolling(50).mean() if use_ma50 else None
    T0 = dte_td / TRADING_DAYS
    trades: list[IronTrade] = []

    i = max(55, 50 if use_ma50 else 25)
    while i + dte_td < len(close):
        S = float(close.iloc[i])
        sigma = float(rv.iloc[i])
        if not np.isfinite(sigma) or sigma <= 0 or S <= 0:
            i += step_td
            continue
        if use_ma50 and ma50 is not None and not (S > float(ma50.iloc[i])):
            i += step_td
            continue

        iv = sigma * (1 + VRP)
        cs, cl, ps, pl = _iron_strikes(S, call_otm=call_otm, put_otm=put_otm, width_pct=width_pct)
        credit = _iron_credit(S, cs, cl, ps, pl, iv, T0)
        call_w = cl - cs
        put_w = ps - pl
        margin = max(call_w, put_w)
        if credit <= 0 or margin <= 0:
            i += step_td
            continue

        touch_call = touch_put = tp_hit = False
        pnl = 0.0
        outcome = "hold_expiry"

        path_idx = range(i + 1, i + dte_td + 1)
        for j in path_idx:
            hi = float(high.iloc[j])
            lo = float(low.iloc[j])
            if hi >= cs:
                touch_call = True
            if lo <= ps:
                touch_put = True

            if take_profit > 0:
                days_in = j - i
                T_rem = max((dte_td - days_in) / TRADING_DAYS, 1 / 365)
                Sj = float(close.iloc[j])
                mark = _mark_iron(Sj, credit, cs, cl, ps, pl, T_rem, iv)
                if mark >= take_profit * credit:
                    pnl = mark
                    tp_hit = True
                    outcome = "50%止盈"
                    break

        if not tp_hit:
            ST = float(close.iloc[i + dte_td])
            pnl = _iron_pnl(ST, credit, cs, cl, ps, pl)
            call_br = ST > cs
            put_br = ST < ps
            if call_br and put_br:
                outcome = "双侧击穿"
            elif call_br:
                outcome = "Call击穿"
            elif put_br:
                outcome = "Put击穿"
            elif pnl >= credit * 0.95:
                outcome = "全收权利金"
            elif pnl > 0:
                outcome = "小幅盈利"
            else:
                outcome = "亏损"
        else:
            call_br = float(close.iloc[min(i + dte_td, len(close) - 1)]) > cs
            put_br = float(close.iloc[min(i + dte_td, len(close) - 1)]) < ps

        one_sig = iv * math.sqrt(T0) * 100
        trades.append(IronTrade(
            entry_date=str(close.index[i].date()),
            spot=S,
            expiry_spot=float(close.iloc[i + dte_td]),
            cs=cs, cl=cl, ps=ps, pl=pl,
            credit=credit,
            margin=margin,
            pnl=pnl,
            pnl_pct_margin=pnl / margin if margin else 0.0,
            outcome=outcome,
            call_breach=call_br if not tp_hit else False,
            put_breach=put_br if not tp_hit else False,
            touch_call=touch_call,
            touch_put=touch_put,
            tp_hit=tp_hit,
            rv_pct=sigma * 100,
            call_otm_pct=(cs / S - 1) * 100,
            put_otm_pct=(1 - ps / S) * 100,
            one_sigma_pct=one_sig,
        ))
        i += step_td
    return trades


def summarize(trades: list[IronTrade], label: str) -> dict:
    if not trades:
        return {"label": label, "n": 0}
    pnls = np.array([t.pnl for t in trades])
    rors = np.array([t.pnl_pct_margin for t in trades])
    n = len(trades)
    return {
        "label": label,
        "n": n,
        "胜率": float((pnls > 0).mean()),
        "50%止盈率": float(np.mean([t.tp_hit for t in trades])),
        "到期Call击穿": float(np.mean([t.call_breach for t in trades])),
        "到期Put击穿": float(np.mean([t.put_breach for t in trades])),
        "到期任一侧击穿": float(np.mean([t.call_breach or t.put_breach for t in trades])),
        "持仓触Call(高)": float(np.mean([t.touch_call for t in trades])),
        "持仓触Put(低)": float(np.mean([t.touch_put for t in trades])),
        "持仓触任一侧": float(np.mean([t.touch_call or t.touch_put for t in trades])),
        "均盈亏$/股": float(pnls.mean()),
        "均ROI%": float(rors.mean() * 100),
        "最差$/股": float(pnls.min()),
        "最大亏ROI%": float(rors.min() * 100),
        "均Call OTM%": float(np.mean([t.call_otm_pct for t in trades])),
        "均Put OTM%": float(np.mean([t.put_otm_pct for t in trades])),
        "均1σ移动%": float(np.mean([t.one_sigma_pct for t in trades])),
    }


def _fmt_pct(x: float) -> str:
    return f"{x:.1%}"


def print_report(ticker: str, df: pd.DataFrame, note: str = "") -> None:
    print(f"\n{'=' * 88}")
    print(f"{ticker} 铁鹰回测 {note}")
    print(f"样本 {df.index[0].date()} ~ {df.index[-1].date()} · {len(df)} 日 · "
          f"DTE={DTE_TD}TD · 50%止盈 · BS+VRP")
    print(f"{'=' * 88}")

    rows = []
    for prof in FLEET_PROFILES:
        trades = backtest_iron(
            df,
            call_otm=prof["call_otm"],
            put_otm=prof["put_otm"],
            width_pct=prof["width_pct"],
        )
        s = summarize(trades, prof["label"])
        rows.append(s)
        if trades:
            worst = min(trades, key=lambda t: t.pnl)
            print(f"\n【{prof['label']}】 call_otm={prof['call_otm']:.0%} put_otm={prof['put_otm']:.0%} "
                  f"width={prof['width_pct']:.0%}")
            print(f"  交易数 {s['n']} · 胜率 {_fmt_pct(s['胜率'])} · 50%止盈率 {_fmt_pct(s['50%止盈率'])}")
            print(f"  到期击穿 Call {_fmt_pct(s['到期Call击穿'])} · Put {_fmt_pct(s['到期Put击穿'])} · "
                  f"任一侧 {_fmt_pct(s['到期任一侧击穿'])}")
            print(f"  持仓触碰(高/低) Call {_fmt_pct(s['持仓触Call(高)'])} · Put {_fmt_pct(s['持仓触Put(低)'])} · "
                  f"任一侧 {_fmt_pct(s['持仓触任一侧'])}")
            print(f"  均盈亏 ${s['均盈亏$/股'] * 100:,.0f}/张 · 均ROI {s['均ROI%']:+.1f}% · "
                  f"最差 ${s['最差$/股'] * 100:,.0f}/张 ({s['最大亏ROI%']:+.1f}%)")
            print(f"  行权距离 Call +{s['均Call OTM%']:.1f}% · Put -{s['均Put OTM%']:.1f}% · "
                  f"同期1σ≈{s['均1σ移动%']:.1f}%")
            print(f"  最差笔 {worst.entry_date} S=${worst.spot:,.0f}→${worst.expiry_spot:,.0f} "
                  f"C{worst.cs:g}/{worst.cl:g} P{worst.ps:g}/{worst.pl:g} {worst.outcome} "
                  f"${worst.pnl * 100:+,.0f}")

    # 仅 Put 侧铁鹰 vs 完整铁鹰对比（验证 Call 腿风险）
    put_only = backtest_iron(df, call_otm=0.99, put_otm=0.12, width_pct=0.02)  # call 极远≈单边 put spread
    full = backtest_iron(df, call_otm=0.15, put_otm=0.12, width_pct=0.02)
    if put_only and full:
        sp = summarize(put_only, "仅Put价差")
        sf = summarize(full, "完整铁鹰")
        print(f"\n【Call 腿增量风险】仅Put价差 击穿率 {_fmt_pct(sp['到期任一侧击穿'])} → "
              f"加Call后 {_fmt_pct(sf['到期任一侧击穿'])} · "
              f"触碰率 {_fmt_pct(sp['持仓触任一侧'])} → {_fmt_pct(sf['持仓触任一侧'])}")

    # 当前价情景
    S = float(df["Close"].iloc[-1])
    sigma = float(realized_vol(df["Close"]).iloc[-1])
    iv = sigma * (1 + VRP)
    T = DTE_TD / TRADING_DAYS
    print(f"\n【当前价 ${S:,.0f} · RV {sigma*100:.0f}% · 1σ({DTE_TD}日) ±{iv*math.sqrt(T)*100:.1f}%】")
    for prof in FLEET_PROFILES:
        cs, cl, ps, pl = _iron_strikes(S, **{k: prof[k] for k in ("call_otm", "put_otm", "width_pct")})
        cr = _iron_credit(S, cs, cl, ps, pl, iv, T)
        mrg = max(cl - cs, ps - pl)
        up = (cs / S - 1) * 100
        dn = (1 - ps / S) * 100
        sig = iv * math.sqrt(T) * 100
        print(f"  {prof['label']}: C{cs:g}/{cl:g} P{ps:g}/{pl:g} · 估收${cr*100:,.0f} · "
              f"区间 -{dn:.0f}%/+{up:.0f}% vs 1σ {sig:.1f}% · "
              f"{'⚠ Call距<2σ' if up < 2 * sig else 'Call距≥2σ'} · "
              f"{'⚠ Put距<2σ' if dn < 2 * sig else 'Put距≥2σ'}")


def fetch(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def main() -> None:
    ap = argparse.ArgumentParser(description="SNDK 铁鹰击穿率回测")
    ap.add_argument("--ticker", default="SNDK")
    ap.add_argument("--start", default="2025-02-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    df = fetch(args.ticker, args.start, args.end)
    if df.empty:
        print(f"{args.ticker} 无数据")
        return
    note = "（闪迪上市短样本）" if args.ticker.upper() == "SNDK" else "（长样本稳健性）"
    print_report(args.ticker.upper(), df, note)

    if args.ticker.upper() == "SNDK":
        print("\n" + "─" * 88)
        wdc = fetch("WDC", "2018-01-01", args.end)
        print_report("WDC", wdc, "（闪迪母体·长历史交叉验证）")
        wdc22 = fetch("WDC", "2022-01-01", "2022-12-31")
        print_report("WDC", wdc22, "（2022 熊市压力）")


if __name__ == "__main__":
    main()
