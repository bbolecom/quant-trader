#!/usr/bin/env python3
"""每日选股历史回测：按年列出选股理由、买进/卖出时机，并统计胜率与年化。

复刻 daily_pick.py 逻辑（SPY/MA50 开关 + 卖Call + 弱市铁鹰 + CSP 舰队 + 轨迹高置信）。

用法：
    python research/daily_pick_backtest.py --year 2025
    python research/daily_pick_backtest.py --year 2025 --quick
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

from quant.meme_route import (
    parse_meme_route,
    pnl_put_spread_hold,
    pnl_stock_short_1d,
    route_action,
    short_fade_direction,
    short_fade_module_label,
)
from quant.decline_income import (
    CSP_DTE_CAL,
    CSP_HOLD_TD,
    CSP_MA_WINDOW,
    CSP_STEP_TD,
    CYCLE_DAYS,
    DEFAULT_VRP,
    WEEKLY_DTE,
    WEEKLY_SOUP_DELTA,
    WEEKLY_SOUP_WIDTH,
    _spread_pnl_at_expiry,
    estimate_bear_call_spread,
    estimate_put_credit_spread,
)
from quant.providers import DataConfig, get_provider, reset_provider_cache
from quant.vol_decay import TRADING_DAYS, bs_put_price, realized_vol, strike_for_put_delta

OUT_DIR = ROOT / "research"


def _csp_trades_detail(
    close: pd.Series,
    *,
    ticker: str,
    module: str,
    account: str,
    delta: float = 0.25,
    take_profit: float = 0.5,
    max_margin_pct: float = 0.75,
    use_ma: bool = True,
    ma_window: int = 50,
    account_size: float = 10_000.0,
    year_start: str,
    year_end: str,
    csp_step: int = CSP_STEP_TD,
) -> list[dict]:
    close = close.astype(float).dropna()
    rv = realized_vol(close)
    ma = close.rolling(ma_window).mean() if use_ma and ma_window > 0 else None
    T = CSP_DTE_CAL / TRADING_DAYS
    cap = account_size * max_margin_pct
    rows: list[dict] = []
    i = max(25, ma_window if use_ma else 0)
    y0, y1 = pd.Timestamp(year_start), pd.Timestamp(year_end)

    while i + CSP_HOLD_TD < len(close):
        ts = pd.Timestamp(close.index[i])
        S = float(close.iloc[i])
        sigma = float(rv.iloc[i])
        above = True if ma is None else S > float(ma.iloc[i])
        can = np.isfinite(sigma) and sigma > 0 and above

        if y0 <= ts <= y1:
            if not can:
                reason = "股价跌破 MA50，暂不开仓" if use_ma and not above else "波动/定价不足"
                rows.append(_watch_row(ts, module, account, ticker, "卖Put", reason))
            else:
                iv = sigma * (1 + DEFAULT_VRP)
                K = strike_for_put_delta(S, T, iv, target_delta=delta)
                margin = K * 100
                if margin > cap or K <= 0:
                    rows.append(_watch_row(
                        ts, module, account, ticker, "卖Put",
                        f"保证金 ${margin:,.0f} 超账户上限",
                    ))
                else:
                    credit = bs_put_price(S, K, T, iv)
                    alloc = min(margin / account_size, max_margin_pct)
                    exited = False
                    exit_j = CSP_HOLD_TD
                    exit_reason = "持有至到期"
                    pnl_frac = 0.0
                    if take_profit > 0:
                        path = close.iloc[i:i + CSP_HOLD_TD + 1]
                        for j in range(1, len(path)):
                            Sj = float(path.iloc[j])
                            remain = max(0.0, 1 - j / CSP_HOLD_TD)
                            mark = max(0.0, K - Sj) + credit * remain * 0.5
                            if credit - mark >= take_profit * credit:
                                pnl_frac = (credit - mark) / K * alloc
                                exit_j = j
                                exit_reason = "赚到50%权利金止盈"
                                exited = True
                                break
                    if not exited:
                        ST = float(close.iloc[i + CSP_HOLD_TD])
                        pnl_frac = (credit - max(0.0, K - ST)) / K * alloc
                    exit_ts = pd.Timestamp(close.index[i + exit_j])
                    entry_reason = (
                        f"卖 Put K=${K:.2f} · δ={delta} · "
                        f"{'MA'+str(ma_window)+'上方' if use_ma else '无MA过滤'} · "
                        f"权利金≈${credit * 100:.0f}/张"
                    )
                    rows.append({
                        "选股日期": ts.strftime("%Y-%m-%d"),
                        "模块": module,
                        "账户": account,
                        "代码": ticker,
                        "状态": "可开仓",
                        "方向": "卖Put",
                        "选股理由": entry_reason,
                        "买进时机": f"{ts.strftime('%Y-%m-%d')} 收盘前 · 卖出 Put 开仓",
                        "卖出时机": f"{exit_ts.strftime('%Y-%m-%d')} 收盘前 · {exit_reason}",
                        "持有天数": exit_j,
                        "收益%": round(pnl_frac * 100, 3),
                        "胜负": "胜" if pnl_frac > 0 else "负",
                        "大盘": "",
                    })
        i += csp_step
    return rows


def _watch_row(ts: pd.Timestamp, module: str, account: str, ticker: str,
               direction: str, reason: str) -> dict:
    return {
        "选股日期": ts.strftime("%Y-%m-%d"),
        "模块": module,
        "账户": account,
        "代码": ticker,
        "状态": "观望",
        "方向": direction,
        "选股理由": reason,
        "买进时机": "",
        "卖出时机": "",
        "持有天数": "",
        "收益%": "",
        "胜负": "",
        "大盘": "",
    }


def _bear_call_trades_detail(
    close: pd.Series,
    *,
    ticker: str,
    year_start: str,
    year_end: str,
    cycle: int = 5,
) -> list[dict]:
    close = close.astype(float).dropna()
    rv = realized_vol(close)
    y0, y1 = pd.Timestamp(year_start), pd.Timestamp(year_end)
    rows: list[dict] = []
    i = 25
    while i + cycle < len(close):
        ts = pd.Timestamp(close.index[i])
        if y0 <= ts <= y1:
            S = float(close.iloc[i])
            sigma = float(rv.iloc[i])
            if np.isfinite(sigma) and sigma > 0:
                ks, kl, credit, _, _ = estimate_bear_call_spread(
                    S, sigma, dte_days=cycle,
                )
                ST = float(close.iloc[i + cycle])
                pnl = _spread_pnl_at_expiry(ST, ks, kl, credit)
                margin = kl - ks
                pnl_pct = float(pnl / margin * 100) if margin > 0 else 0.0
                exit_ts = pd.Timestamp(close.index[i + cycle])
                rows.append({
                    "选股日期": ts.strftime("%Y-%m-%d"),
                    "模块": "卖Call价差",
                    "账户": "收入引擎",
                    "代码": ticker,
                    "状态": "可开仓",
                    "方向": "卖Call价差",
                    "选股理由": (
                        f"高振幅/涨幅候选 · 卖C${ks:.0f}/买C${kl:.0f} · "
                        f"收${credit * 100:.0f}/张 · 赌{cycle}日不再暴涨"
                    ),
                    "买进时机": f"{ts.strftime('%Y-%m-%d')} 收盘前 · 卖出 Call 价差开仓",
                    "卖出时机": f"{exit_ts.strftime('%Y-%m-%d')} 收盘前 · 到期或平仓",
                    "持有天数": cycle,
                    "收益%": round(pnl_pct, 3),
                    "胜负": "胜" if pnl_pct > 0 else "负",
                    "大盘": "",
                })
        i += cycle
    return rows


def _iron_trades_detail(
    close: pd.Series,
    *,
    ticker: str,
    year_start: str,
    year_end: str,
    use_ma: bool = False,
) -> list[dict]:
    close = close.astype(float).dropna()
    rv = realized_vol(close)
    ma = close.rolling(CSP_MA_WINDOW).mean() if use_ma else None
    hold = max(1, int(WEEKLY_DTE * TRADING_DAYS / 7))
    step = 5
    y0, y1 = pd.Timestamp(year_start), pd.Timestamp(year_end)
    rows: list[dict] = []
    i = max(25, CSP_MA_WINDOW if use_ma else 25)
    while i + hold < len(close):
        ts = pd.Timestamp(close.index[i])
        if not (y0 <= ts <= y1):
            i += 1
            continue
        S = float(close.iloc[i])
        sigma = float(rv.iloc[i])
        if not np.isfinite(sigma) or sigma <= 0:
            i += step
            continue
        if use_ma and ma is not None and not (S > float(ma.iloc[i])):
            i += step
            continue
        ks, kl, credit, margin, max_loss, _ = estimate_put_credit_spread(
            S, sigma, short_delta=WEEKLY_SOUP_DELTA, width=WEEKLY_SOUP_WIDTH,
            dte_days=WEEKLY_DTE, vrp=DEFAULT_VRP,
        )
        if margin <= 0:
            i += step
            continue
        ST = float(close.iloc[i + hold])
        pnl = credit - (max(0.0, ks - ST) - max(0.0, kl - ST))
        pnl_pct = float(pnl / WEEKLY_SOUP_WIDTH * 100)
        exit_ts = pd.Timestamp(close.index[i + hold])
        rows.append({
            "选股日期": ts.strftime("%Y-%m-%d"),
            "模块": "弱市·ETF铁鹰",
            "账户": "收租锚点",
            "代码": ticker,
            "状态": "可开仓",
            "方向": "铁鹰",
            "选股理由": (
                f"{ticker} 偏斜铁鹰 · RV {sigma * 100:.0f}% · "
                f"卖P${ks:.0f}/买P${kl:.0f} · 收${credit * 100:.0f}/张"
            ),
            "买进时机": f"{ts.strftime('%Y-%m-%d')} 收盘前 · 开铁鹰组合",
            "卖出时机": f"{exit_ts.strftime('%Y-%m-%d')} 收盘前 · 到期平仓",
            "持有天数": hold,
            "收益%": round(pnl_pct, 3),
            "胜负": "胜" if pnl_pct > 0 else "负",
            "大盘": "",
        })
        i += step
    return rows


def _trajectory_trades(
    panel: pd.DataFrame,
    fwd_ret: dict[str, pd.Series],
    spy_close: pd.Series,
    *,
    year_start: str,
    year_end: str,
    bull_only: bool = True,
    trajectory_mode: str = "highwin",
    trajectory_top_n: int = 2,
) -> list[dict]:
    from research.gainer_daily_backtest import (
        edge_setup_filters,
        filters_for_mode,
        market_regime_ok,
        pick_from_panel,
        precompute_setup_edge,
    )

    filt = filters_for_mode(trajectory_mode, top_n=trajectory_top_n)
    mod = "高频·动量" if trajectory_mode == "highfreq" else "轨迹·高置信"
    acct = "每日Top5" if trajectory_mode == "highfreq" else "涨幅榜Top"
    panel = precompute_setup_edge(panel, fwd_ret, edge_setup_filters())
    cal = panel["日期"].drop_duplicates().sort_values()
    cal = cal[(cal >= pd.Timestamp(year_start)) & (cal <= pd.Timestamp(year_end))]
    spy_ma50 = spy_close.rolling(50).mean()
    panel_by_date = {d: g for d, g in panel.groupby("日期")}
    rows: list[dict] = []

    for i in range(len(cal) - 1):
        as_of = cal.iloc[i]
        nxt = cal.iloc[i + 1]
        bull = float(spy_close.loc[as_of]) > float(spy_ma50.loc[as_of]) if as_of in spy_ma50.index else True
        regime_label = "🟢 牛市" if bull else "🔴 弱市"
        day = panel_by_date.get(as_of)
        if day is None or day.empty:
            continue
        if bull_only and not bull:
            rows.append({
                "选股日期": pd.Timestamp(as_of).strftime("%Y-%m-%d"),
                "模块": mod,
                "账户": "全市场",
                "代码": "—",
                "状态": "观望",
                "方向": "—",
                "选股理由": "弱市（SPY<MA50）关闭轨迹做多",
                "买进时机": "",
                "卖出时机": "",
                "持有天数": "",
                "收益%": "",
                "胜负": "",
                "大盘": regime_label,
            })
            continue
        if not market_regime_ok(day, filt):
            rows.append({
                "选股日期": pd.Timestamp(as_of).strftime("%Y-%m-%d"),
                "模块": mod,
                "账户": "全市场",
                "代码": "—",
                "状态": "观望",
                "方向": "—",
                "选股理由": "大盘/因子条件未满足（正常空仓日）",
                "买进时机": "",
                "卖出时机": "",
                "持有天数": "",
                "收益%": "",
                "胜负": "",
                "大盘": regime_label,
            })
            continue
        top = pick_from_panel(day, filt)
        if top.empty:
            rows.append({
                "选股日期": pd.Timestamp(as_of).strftime("%Y-%m-%d"),
                "模块": mod,
                "账户": "全市场",
                "代码": "—",
                "状态": "观望",
                "方向": "—",
                "选股理由": "无标的满足温和涨+量比+形态胜率模板",
                "买进时机": "",
                "卖出时机": "",
                "持有天数": "",
                "收益%": "",
                "胜负": "",
                "大盘": regime_label,
            })
            continue
        for _, row in top.iterrows():
            t = str(row["代码"])
            fr = fwd_ret.get(t)
            r = float(fr.get(as_of, np.nan)) if fr is not None else np.nan
            if not np.isfinite(r):
                continue
            fee = 0.001
            ret_pct = (r - 2 * fee) * 100
            reason = (
                f"1日涨{row['涨幅%']:+.1f}% · 量比{row['量比']:.1f} · "
                f"{'站上' if row.get('站上MA20', True) else '跌破'}MA20 · "
                f"相对SPY {row['相对SPY20d%']:+.1f}%"
            )
            rows.append({
                "选股日期": pd.Timestamp(as_of).strftime("%Y-%m-%d"),
                "模块": mod,
                "账户": acct,
                "代码": t,
                "状态": "可开仓",
                "方向": "做多",
                "选股理由": reason,
                "买进时机": f"{pd.Timestamp(as_of).strftime('%Y-%m-%d')} 收盘买入",
                "卖出时机": f"{pd.Timestamp(nxt).strftime('%Y-%m-%d')} 收盘卖出（持1日）",
                "持有天数": 1,
                "收益%": round(ret_pct, 3),
                "胜负": "胜" if ret_pct > 0 else "负",
                "大盘": regime_label,
            })
    return rows


FEE = 5 / 10_000


def _daily_bear_call_picks(
    panel: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    *,
    year_start: str,
    year_end: str,
    top_n: int = 5,
    cycle: int = 5,
    cfg: dict | None = None,
    spy_close: pd.Series | None = None,
) -> list[dict]:
    """每日涨幅/振幅榜 TopN → meme 路由 → 卖 Call / 超涨Put价差 / 观望。"""
    mrc = parse_meme_route(cfg or {})
    modules = (cfg or {}).get("modules") or {}
    use_route = mrc.enabled and bool(modules.get("short_fade", True))
    short_cap = mrc.short_fade.top_n if use_route else 0
    sf = mrc.short_fade
    spy_ma50 = spy_close.rolling(50).mean() if spy_close is not None else None

    cal = panel["日期"].drop_duplicates().sort_values()
    cal = cal[(cal >= pd.Timestamp(year_start)) & (cal <= pd.Timestamp(year_end))]
    panel_by_date = {d: g for d, g in panel.groupby("日期")}
    rows: list[dict] = []

    for as_of in cal:
        day = panel_by_date.get(as_of)
        if day is None or day.empty:
            continue
        day = day.copy()
        day["涨幅%"] = pd.to_numeric(day["涨幅%"], errors="coerce")
        if "振幅%" in day.columns:
            day["振幅%"] = pd.to_numeric(day["振幅%"], errors="coerce").fillna(day["涨幅%"].abs())
        else:
            day["振幅%"] = day["涨幅%"].abs() * 1.2
        day = day[day["涨幅%"].fillna(0) >= 2.0]
        if day.empty:
            rows.append({
                "选股日期": pd.Timestamp(as_of).strftime("%Y-%m-%d"),
                "模块": "卖Call价差", "账户": "收入引擎", "代码": "—",
                "状态": "观望", "方向": "卖Call价差",
                "选股理由": "今日涨幅/振幅榜无合适标的",
                "买进时机": "", "卖出时机": "", "持有天数": "",
                "收益%": "", "胜负": "", "大盘": "",
            })
            continue
        day = day.assign(_score=day["涨幅%"] + day["振幅%"] * 0.5)
        day = day.sort_values("_score", ascending=False)
        ds = pd.Timestamp(as_of).strftime("%Y-%m-%d")
        spy1 = pd.to_numeric(day.iloc[0].get("SPY1d涨%"), errors="coerce") if use_route else np.nan
        spy_bear = None
        if use_route and spy_close is not None and as_of in spy_close.index and spy_ma50 is not None and as_of in spy_ma50.index:
            spy_bear = float(spy_close.loc[as_of]) < float(spy_ma50.loc[as_of])
        bear_n = 0
        short_n = 0
        any_open = False

        for _, r in day.iterrows():
            if use_route:
                action = route_action(
                    r, mrc,
                    spy_1d_pct=float(spy1) if np.isfinite(spy1) else None,
                    spy_bear=spy_bear,
                )
                if action == "skip":
                    continue
                if action == "short_fade":
                    if short_n >= short_cap:
                        continue
                    t = str(r["代码"])
                    fr = data.get(t)
                    if fr is None or fr.empty:
                        continue
                    close = fr["Close"].astype(float)
                    if as_of not in close.index:
                        continue
                    idx = close.index.get_loc(as_of)
                    hold = max(1, sf.hold_days)
                    if idx + hold >= len(close):
                        continue
                    S0 = float(close.iloc[idx])
                    S1 = float(close.iloc[idx + hold])
                    rv = realized_vol(close)
                    sigma = float(rv.iloc[idx]) if idx < len(rv) else np.nan
                    if not np.isfinite(sigma) or sigma <= 0:
                        continue
                    exit_ts = pd.Timestamp(close.index[idx + hold])
                    if sf.structure == "put_spread":
                        _, kl, ks, debit, ret_pct = pnl_put_spread_hold(
                            S0, S1, sigma,
                            delta=sf.put_delta, width_pct=sf.put_width_pct,
                            dte_days=sf.dte_days, hold_days=hold,
                        )
                        reason = (
                            f"meme路由·弱市 · 涨幅{float(r['涨幅%']):+.1f}% · "
                            f"买P${kl:.0f}/卖P${ks:.0f} · 付${debit * 100:.0f}/张"
                        )
                        buy_note = f"{ds} 收盘前 · 买入 Put 价差"
                        sell_note = f"{exit_ts.strftime('%Y-%m-%d')} 收盘前 · 平仓"
                    else:
                        r1 = S1 / S0 - 1
                        ret_pct = pnl_stock_short_1d(r1)
                        kl = ks = debit = 0.0
                        reason = (
                            f"meme路由 · 涨幅{float(r['涨幅%']):+.1f}% · "
                            f"收盘强度{float(r.get('收盘强度', 0.5)):.2f} · 次日平空"
                        )
                        buy_note = f"{ds} 收盘前 · 融券/反向开仓"
                        sell_note = f"{exit_ts.strftime('%Y-%m-%d')} 收盘前 · 平空"
                    short_n += 1
                    any_open = True
                    rows.append({
                        "选股日期": ds,
                        "模块": short_fade_module_label(sf.structure),
                        "账户": "meme路由",
                        "代码": t,
                        "状态": "可开仓",
                        "方向": short_fade_direction(sf.structure),
                        "选股理由": reason,
                        "买进时机": buy_note,
                        "卖出时机": sell_note,
                        "持有天数": hold,
                        "收益%": round(ret_pct, 3),
                        "胜负": "胜" if ret_pct > 0 else "负",
                        "大盘": "🔴 弱市" if spy_bear else "",
                    })
                    continue
            if bear_n >= top_n:
                continue
            t = str(r["代码"])
            df = data.get(t)
            if df is None or df.empty:
                continue
            close = df["Close"].astype(float)
            if as_of not in close.index:
                continue
            idx = close.index.get_loc(as_of)
            if idx + cycle >= len(close):
                continue
            S = float(close.iloc[idx])
            rv = realized_vol(close)
            sigma = float(rv.iloc[idx]) if idx < len(rv) else np.nan
            if not np.isfinite(sigma) or sigma <= 0:
                continue
            ks, kl, credit, _, _ = estimate_bear_call_spread(S, sigma, dte_days=cycle)
            ST = float(close.iloc[idx + cycle])
            pnl = _spread_pnl_at_expiry(ST, ks, kl, credit)
            margin = kl - ks
            pnl_pct = float(pnl / margin * 100) if margin > 0 else 0.0
            exit_ts = pd.Timestamp(close.index[idx + cycle])
            bear_n += 1
            any_open = True
            rows.append({
                "选股日期": ds,
                "模块": "卖Call价差",
                "账户": "收入引擎",
                "代码": t,
                "状态": "可开仓",
                "方向": "卖Call价差",
                "选股理由": (
                    f"涨幅{float(r['涨幅%']):+.1f}% · 振幅{float(r['振幅%']):.1f}% · "
                    f"卖C${ks:.0f}/买C${kl:.0f} · 收${credit * 100:.0f}/张 · 赌{cycle}日不再暴涨"
                ),
                "买进时机": f"{ds} 收盘前 · 卖出 Call 价差开仓",
                "卖出时机": f"{exit_ts.strftime('%Y-%m-%d')} 收盘前 · 到期或平仓",
                "持有天数": cycle,
                "收益%": round(pnl_pct, 3),
                "胜负": "胜" if pnl_pct > 0 else "负",
                "大盘": "",
            })
        if not any_open:
            rows.append({
                "选股日期": ds,
                "模块": "卖Call价差", "账户": "收入引擎", "代码": "—",
                "状态": "观望", "方向": "卖Call价差",
                "选股理由": "候选被meme路由过滤或数据不足",
                "买进时机": "", "卖出时机": "", "持有天数": "",
                "收益%": "", "胜负": "", "大盘": "",
            })
    return rows


def _stats(trades: pd.DataFrame, account_size: float = 10_000.0) -> dict:
    closed = trades[trades["状态"] == "可开仓"].copy()
    closed = closed[pd.to_numeric(closed["收益%"], errors="coerce").notna()]
    if closed.empty:
        return {"笔数": 0, "胜率": 0.0, "年化": 0.0, "累计收益%": 0.0}

    rets = pd.to_numeric(closed["收益%"], errors="coerce") / 100.0
    win_rate = float((rets > 0).mean())

    # 按日合并：同日多笔等权（贴近 daily_pick 多模块并行）
    closed = closed.copy()
    closed["_ret"] = rets
    daily = closed.groupby("选股日期")["_ret"].mean()
    equity = account_size
    for r in daily:
        equity *= 1.0 + float(r) * 0.05  # 单日组合约 5% 资金风险敞口
    total = equity / account_size - 1.0
    n_days = max((pd.Timestamp(daily.index.max()) - pd.Timestamp(daily.index.min())).days, 30)
    years = n_days / 365.25
    ann = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else total

    by_mod = {}
    for mod, grp in closed.groupby("模块"):
        gr = pd.to_numeric(grp["收益%"], errors="coerce") / 100.0
        by_mod[mod] = {
            "笔数": len(gr),
            "胜率": float((gr > 0).mean()) if len(gr) else 0.0,
            "均收益%": float(gr.mean() * 100) if len(gr) else 0.0,
        }

    watch = trades[trades["状态"].isin(["观望", "无数据"])]
    return {
        "笔数": len(closed),
        "胜率": win_rate,
        "年化": ann,
        "累计收益%": total * 100,
        "期末权益": equity,
        "期初权益": account_size,
        "观望条目": len(watch),
        "分模块": by_mod,
    }


def run_backtest(
    *,
    year: int | None = 2025,
    start: str | None = None,
    end: str | None = None,
    quick: bool = False,
    profile: str = "high_freq",
) -> dict:
    from research.gainer_daily_backtest import (
        GAINER_MOMENTUM,
        LIQUID100,
        build_factor_panels,
        fetch_gainer_data_yahoo,
    )
    from quant.daily_screen_fleet import fleet_accounts, load_fleet_config
    from daily_pick import load_config, resolve_profile

    cfg = load_config(ROOT / "daily_pick_config.json")
    if profile != cfg.get("frequency_profile"):
        cfg = {**cfg, "frequency_profile": profile}
    prof = resolve_profile(cfg)

    year_start = start or f"{year}-01-01"
    year_end = end or (f"{year}-12-31" if year else date.today().isoformat())
    label_year = year or f"{year_start[:4]}-{year_end[:4]}"
    warmup = (pd.Timestamp(year_start) - pd.DateOffset(months=14)).strftime("%Y-%m-%d")

    fleet_cfg = load_fleet_config()
    account_size = float(fleet_cfg.get("account_size", 10_000))

    pool = LIQUID100 if quick else GAINER_MOMENTUM
    fleet_tickers = [str(a.get("ticker", "")).upper() for a in fleet_accounts(fleet_cfg)]
    icfg = cfg.get("bear_iron_etf") or {}
    iron_tickers = icfg.get("tickers") or ["SPY", "QQQ"]
    if isinstance(iron_tickers, str):
        iron_tickers = [t.strip() for t in iron_tickers.replace(",", " ").split()]
    all_tickers = sorted(set(pool + fleet_tickers + iron_tickers))

    print(f"配置：{prof['name']} · 动量={prof['trajectory_mode']} Top{prof['trajectory_top_n']}")
    print(f"拉取 {len(all_tickers)} 只标的 ({warmup} ~ {year_end}) …")
    data, spy_df = fetch_gainer_data_yahoo(all_tickers, warmup, year_end)
    spy_close = spy_df["Close"].astype(float)
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]

    panel = build_factor_panels(data, spy_close)
    fwd_ret = {t: df["Close"].astype(float).pct_change(1).shift(-1) for t, df in data.items()}

    spy_ma50 = spy_close.rolling(50).mean()
    bull_days = int((spy_close.loc[year_start:year_end] > spy_ma50.loc[year_start:year_end]).sum())
    total_days = len(spy_close.loc[year_start:year_end])

    all_rows: list[dict] = []
    modules = cfg.get("modules") or {}

    print(f"回测 卖Call+meme路由（每日Top{prof['bear_call_top_n']}）…")
    all_rows.extend(_daily_bear_call_picks(
        panel, data, year_start=year_start, year_end=year_end,
        top_n=int(prof["bear_call_top_n"]),
        cfg=cfg,
        spy_close=spy_close,
    ))

    iron_mod = "并行·ETF铁鹰" if prof.get("etf_iron_always") else "弱市·ETF铁鹰"
    if modules.get("bear_iron_etf", True):
        print(f"回测 {iron_mod} …")
        for tk in iron_tickers:
            df = data.get(str(tk).upper())
            if df is None or df.empty:
                continue
            for tr in _iron_trades_detail(
                df["Close"].astype(float), ticker=str(tk).upper(),
                year_start=year_start, year_end=year_end, use_ma=False,
            ):
                ts = pd.Timestamp(tr["选股日期"])
                if not prof.get("etf_iron_always"):
                    if ts in spy_close.index and ts in spy_ma50.index:
                        if float(spy_close.loc[ts]) >= float(spy_ma50.loc[ts]):
                            continue
                    tr["大盘"] = "🔴 弱市"
                tr["模块"] = iron_mod
                all_rows.append(tr)
    else:
        print("  （ETF铁鹰已关闭，跳过）")

    # ③ CSP 舰队
    print("回测 5×CSP 舰队 …")
    for acct in fleet_accounts(fleet_cfg):
        sym = str(acct.get("ticker", "")).upper()
        df = data.get(sym)
        p = acct.get("csp_params") or {}
        if df is None or df.empty:
            all_rows.append({
                "选股日期": year_start, "模块": "5×舰队·CSP",
                "账户": acct.get("label", ""), "代码": sym,
                "状态": "无数据", "方向": "—",
                "选股理由": f"{sym} 2025 历史数据不足",
                "买进时机": "", "卖出时机": "", "持有天数": "",
                "收益%": "", "胜负": "", "大盘": "",
            })
            continue
        all_rows.extend(_csp_trades_detail(
            df["Close"].astype(float),
            ticker=sym,
            module="5×舰队·CSP",
            account=acct.get("label", acct["id"]),
            delta=float(p.get("delta", 0.25)),
            take_profit=float(p.get("take_profit", 0.5)),
            max_margin_pct=float(p.get("alloc_pct", 0.75)),
            use_ma=int(p.get("ma_window", 50)) > 0,
            ma_window=int(p.get("ma_window", 50)),
            account_size=account_size,
            year_start=year_start,
            year_end=year_end,
            csp_step=int(prof.get("csp_step_days", CSP_STEP_TD)),
        ))

    modules = cfg.get("modules") or {}
    if modules.get("trajectory_highwin", True) and prof.get("trajectory_enabled", True):
        print(f"回测 动量（{prof['trajectory_mode']} Top{prof['trajectory_top_n']}）…")
        all_rows.extend(_trajectory_trades(
            panel, fwd_ret, spy_close,
            year_start=year_start, year_end=year_end,
            bull_only=bool(prof.get("trajectory_bull_only", True)),
            trajectory_mode=str(prof.get("trajectory_mode", "highwin")),
            trajectory_top_n=int(prof.get("trajectory_top_n", 2)),
        ))
    else:
        print("  （动量腿已关闭，跳过）")

    df_out = pd.DataFrame(all_rows)
    if not df_out.empty:
        df_out = df_out.sort_values(["选股日期", "模块", "代码"]).reset_index(drop=True)

    stats = _stats(df_out, account_size=account_size)
    stats["年份"] = label_year
    stats["配置"] = prof["name"]
    stats["SPY牛市天数"] = f"{bull_days}/{total_days} ({bull_days / max(total_days, 1):.0%})"

    prefix = f"daily_pick_{label_year}_{prof['name']}".replace(" ", "")
    detail_csv = OUT_DIR / f"{prefix}_detail.csv"
    trades_csv = OUT_DIR / f"{prefix}_trades.csv"
    summary_json = OUT_DIR / f"{prefix}_summary.json"

    df_out.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    closed = df_out[df_out["状态"] == "可开仓"].copy()
    closed.to_csv(trades_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "detail": df_out,
        "stats": stats,
        "paths": {"detail": detail_csv, "trades": trades_csv, "summary": summary_json},
    }


def _print_report(res: dict) -> None:
    st = res["stats"]
    df = res["detail"]
    print(f"\n{'=' * 64}")
    print(f"每日选股回测 · {st['年份']} · 配置 {st.get('配置', '—')}")
    print(f"{'=' * 64}")
    print(f"SPY 牛市天数：{st.get('SPY牛市天数', '—')}")
    print(f"\n【整体】")
    print(f"  可交易笔数：{st['笔数']}")
    print(f"  笔胜率：     {st['胜率']:.1%}")
    print(f"  期初账户：   ${st.get('期初权益', 10000):,.0f}")
    print(f"  期末权益：   ${st.get('期末权益', 0):,.0f}")
    print(f"  累计收益：   {st['累计收益%']:+.2f}%")
    print(f"  年化：       {st['年化']:.1%}")
    print(f"  有信号交易日：{st.get('有信号交易日', 0)}")
    print(f"  观望条目：   {st.get('观望条目', 0)}")
    print(f"\n【分模块】")
    for mod, mst in (st.get("分模块") or {}).items():
        print(f"  {mod}: {mst['笔数']}笔  胜率{mst['胜率']:.1%}  均收益{mst['均收益%']:+.2f}%")
    print(f"\n明细 → {res['paths']['detail']}")
    print(f"成交 → {res['paths']['trades']}")
    if not df.empty:
        print(f"\n【2025 可开仓样本 Top10】")
        show = df[df["状态"] == "可开仓"].head(10)
        for _, r in show.iterrows():
            print(f"  {r['选股日期']} [{r['模块']}] {r['代码']} {r['方向']} "
                  f"买:{r['买进时机'][:20]}… 卖:{str(r['卖出时机'])[:20]}… "
                  f"收益{r['收益%']}%")


def main() -> None:
    p = argparse.ArgumentParser(description="每日选股年度回测")
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--start", default=None, help="起始日 YYYY-MM-DD")
    p.add_argument("--end", default=None, help="结束日 YYYY-MM-DD")
    p.add_argument("--quick", action="store_true", help="LIQUID100 子集加速")
    p.add_argument("--profile", choices=["standard", "high_freq"], default="high_freq")
    args = p.parse_args()
    year = args.year if args.year else (None if args.start else 2025)
    res = run_backtest(
        year=year, start=args.start, end=args.end,
        quick=args.quick, profile=args.profile,
    )
    _print_report(res)


if __name__ == "__main__":
    main()
