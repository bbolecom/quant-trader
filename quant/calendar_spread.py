"""双日历价差择时：IV Rank 低位才开 + 财报回避 + 横盘过滤。

结构：卖近月 call/put + 买远月 call/put（行权价 ± k×周波动率）。
回测结论：theta 有结构性优势，但 long vega → IV 高位/财报后 IV crush 是最大风险。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import decline_income as di
from .data import fetch_history
from .vol_decay import DEFAULT_VRP, RFR, TRADING_DAYS

DEFAULT_CALENDAR_TICKERS = [
    "NVDA", "PLTR", "AMD", "META", "MSFT", "GOOGL", "QQQ", "TSLA", "MU", "AVGO", "SNDK",
]


@dataclass
class CalendarSpreadPlan:
    ticker: str
    close: float
    rv_pct: float
    iv_pct: float
    iv_rank: float
    er: float
    can_open: bool
    call_strike: float
    put_strike: float
    debit_per_share: float
    debit_per_contract: float
    debit_pct_account: float
    profit_zone_pct: float
    theta_est_contract: float
    short_d: int
    long_d: int
    hold_trading_days: int
    max_contracts: int
    earnings_days: int | None = None
    flags: list[str] = field(default_factory=list)
    playbook: list[str] = field(default_factory=list)


def efficiency_ratio(close: pd.Series, n: int = 30) -> float:
    c = close.iloc[-(n + 1):].astype(float)
    if len(c) < n + 1:
        return 1.0
    net = abs(float(c.iloc[-1] - c.iloc[0]))
    path = float(c.diff().abs().sum())
    return net / path if path > 0 else 1.0


def iv_rank_from_close(close: pd.Series, window: int = 252) -> tuple[float, float]:
    rv = close.pct_change(fill_method=None).rolling(20).std() * math.sqrt(TRADING_DAYS)
    if rv.dropna().empty:
        return 0.0, 0.5
    iv = float(rv.iloc[-1])
    hist = rv.dropna().iloc[-window:]
    rank = float((hist.iloc[-1] >= hist).mean()) if len(hist) >= 60 else 0.5
    return iv, rank


def double_calendar_debit(S: float, Kc: float, Kp: float, T_short: float, T_long: float, iv: float) -> float:
    call_cal = di.bs_call_price(S, Kc, T_long, iv, RFR) - di.bs_call_price(S, Kc, T_short, iv, RFR)
    put_cal = di.bs_put_price(S, Kp, T_long, iv, RFR) - di.bs_put_price(S, Kp, T_short, iv, RFR)
    return call_cal + put_cal


def days_to_next_earnings(ticker: str, today: date | None = None) -> int | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=8)
        if ed is None or ed.empty:
            return None
        today = today or date.today()
        idx = pd.to_datetime(ed.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        future = [d.date() for d in idx if d.date() >= today]
        if not future:
            return None
        return (min(future) - today).days
    except Exception:  # noqa: BLE001
        return None


def _build_playbook(plan: CalendarSpreadPlan) -> list[str]:
    steps = [
        f"1. **{plan.ticker}** 现价 ${plan.close:,.2f}｜RV {plan.rv_pct:.0f}%｜IV Rank {plan.iv_rank:.0%}｜效率比 {plan.er:.2f}",
        f"2. 双日历：卖 {plan.short_d} 天 / 买 {plan.long_d} 天，持有约 {plan.hold_trading_days} 个交易日（≈7 自然日）后平。",
        f"3. Call 行权 ${plan.call_strike:,.0f} + Put 行权 ${plan.put_strike:,.0f}（±约 {plan.profit_zone_pct:.0f}% 周波动）。",
        f"4. 预估净付 ${plan.debit_per_contract:,.0f}/张（占账户 {plan.debit_pct_account:.0f}%），7 日 theta 约 +${plan.theta_est_contract:,.0f}/张（IV 不变近似）。",
        f"5. 建议 {plan.max_contracts} 张；赚到约 50% 时间价值可提前平；任一腿被逼近则减仓/平仓。",
        "6. **纪律**：IV Rank 高位不开；财报前 1 周不开；IV 崩塌(long vega) 是最大尾部风险。",
    ]
    if plan.earnings_days is not None:
        steps.append(f"7. 下次财报约 {plan.earnings_days} 天后（≤7 天则今日不应开）。")
    if plan.flags:
        steps.append("⚠ " + "；".join(plan.flags))
    return steps


def calendar_spread_plan(
    ticker: str,
    df: pd.DataFrame,
    *,
    account_size: float = 10_000.0,
    short_d: int = 14,
    long_d: int = 21,
    hold_trading_days: int = 5,
    k_sigma: float = 1.0,
    iv_mult: float = 1.1,
    iv_pct_max: float = 0.40,
    iv_window: int = 252,
    max_er: float = 0.45,
    earnings_buffer_days: int = 7,
    max_debit_pct: float = 0.50,
    check_earnings: bool = True,
) -> CalendarSpreadPlan | None:
    c = df["Close"].astype(float).dropna()
    if len(c) < 65:
        return None
    S = float(c.iloc[-1])
    iv, iv_rank = iv_rank_from_close(c, iv_window)
    iv_p = iv * (1 + DEFAULT_VRP) * iv_mult
    er = efficiency_ratio(c, 30)
    wsig = iv_p * math.sqrt(7 / 365)
    Kc = S * (1 + k_sigma * wsig)
    Kp = S * (1 - k_sigma * wsig)
    Ts_e, Tl_e = short_d / 365, long_d / 365
    Ts_x, Tl_x = max((short_d - 7) / 365, 1 / 365), max((long_d - 7) / 365, 8 / 365)
    debit = double_calendar_debit(S, Kc, Kp, Ts_e, Tl_e, iv_p)
    if debit <= 0 or debit < 0.003 * S:
        return None
    exit_theta = double_calendar_debit(S, Kc, Kp, Ts_x, Tl_x, iv_p)
    debit_c = debit * 100
    theta_c = (exit_theta - debit) * 100
    debit_pct = debit_c / account_size * 100
    earnings_days = days_to_next_earnings(ticker) if check_earnings else None

    flags: list[str] = []
    can = True
    if iv_rank > iv_pct_max:
        can = False
        flags.append(f"IV Rank {iv_rank:.0%} > {iv_pct_max:.0%}（IV 偏高，易遭 IV crush）")
    if earnings_days is not None and earnings_days <= earnings_buffer_days:
        can = False
        flags.append(f"距财报 {earnings_days} 天 ≤ {earnings_buffer_days}（财报周回避）")
    if er > max_er:
        can = False
        flags.append(f"效率比 {er:.2f} > {max_er}（单边趋势强，不适合钉价）")
    if debit_pct / 100 > max_debit_pct:
        can = False
        flags.append(f"开仓成本 {debit_pct:.0f}% > 账户 {max_debit_pct:.0%}（小账户玩不动）")

    max_c = max(1, int(account_size * max_debit_pct / debit_c)) if debit_c > 0 else 0
    if can and max_c < 1:
        can = False
        flags.append("账户不足以开 1 张")

    plan = CalendarSpreadPlan(
        ticker=ticker, close=S, rv_pct=iv * 100, iv_pct=iv_p * 100,
        iv_rank=iv_rank, er=er, can_open=can,
        call_strike=Kc, put_strike=Kp,
        debit_per_share=debit, debit_per_contract=debit_c,
        debit_pct_account=debit_pct, profit_zone_pct=wsig * 100,
        theta_est_contract=theta_c,
        short_d=short_d, long_d=long_d, hold_trading_days=hold_trading_days,
        max_contracts=max_c if can else 0,
        earnings_days=earnings_days, flags=flags,
    )
    plan.playbook = _build_playbook(plan)
    return plan


def scan_calendar_plans(
    tickers: list[str],
    start: str,
    end: str,
    **kwargs,
) -> tuple[list[CalendarSpreadPlan], list[str]]:
    plans: list[CalendarSpreadPlan] = []
    errors: list[str] = []
    for tk in tickers:
        try:
            df = fetch_history(tk, start=start, end=end)
            p = calendar_spread_plan(tk, df, **kwargs)
            if p:
                plans.append(p)
            else:
                errors.append(f"{tk}：数据不足或成本退化")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{tk}：{e}")
    plans.sort(key=lambda p: (p.can_open, p.theta_est_contract / max(p.debit_per_contract, 1)), reverse=True)
    return plans, errors
