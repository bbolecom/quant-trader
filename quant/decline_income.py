"""缓跌/高波个股期权收租策略（闪迪 SNDK、WDC 等）。

核心思路：预期「慢慢下跌或横盘」时，**不要裸卖 Put**（会被接飞刀），优先：
    1. 熊市认购价差（Bear Call Spread / Call Credit Spread）— 卖近 OTM call + 买更远 call
    2. 已持股 → 备兑卖 Call（Covered Call）持续收租
    3. 已持股且怕加速下跌 → 领口（Collar）

权利金与回测均为 BS + VRP 近似，供研究与扫描；实盘以券商报价为准。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .data import fetch_history, fetch_history_batch
from .vol_decay import (
    DEFAULT_VRP, RFR, TRADING_DAYS, _norm_cdf, _norm_cdf_inv,
    bs_put_price, realized_vol, strike_for_put_delta,
)

CYCLE_DAYS = 35
DEFAULT_WIDTH_PCT = 0.05
CALL_DELTA_SHORT = 0.25

# --- 「稳定收租」CSP 引擎参数（经 SNDK/WDC/MU 回测优选）---
CSP_HOLD_TD = 25       # 持有交易日数（≈35 日历天到期）
CSP_DTE_CAL = 35       # 定价用到期日历天
CSP_STEP_TD = 5        # 重叠回测每 5 个交易日开一笔
CSP_DELTA = 0.20       # 卖出 put 的 delta（收益/稳定甜点；更保守用 0.15）
CSP_MA_WINDOW = 50     # 趋势过滤：股价站上 50 日均线才开仓
CSP_TAKE_PROFIT = 0.5  # 赚到 50% 权利金即止盈（关键稳定器）

# 闪迪相关 + 同类高波存储/半导体
DECLINE_INCOME_UNIVERSE = [
    "SNDK", "WDC", "MU", "STX", "NVDA", "AMD", "SMCI", "PLTR", "COIN", "MARA",
    "TSLA", "INTC", "QCOM", "AVGO",
]


@dataclass
class DeclineFilters:
    min_dollar_vol_m: float = 200.0
    min_rv_pct: float = 35.0
    max_rv_pct: float = 120.0
    min_ret_60d_pct: float = -40.0   # 60日跌幅不超过 40%（排除崩盘股）
    max_ret_60d_pct: float = 15.0    # 60日涨幅不超过 15%（排除强趋势上行的）
    allow_strong_uptrend: bool = False  # True 时仍给出方案但标高风险
    vrp: float = DEFAULT_VRP
    call_delta: float = CALL_DELTA_SHORT
    spread_width_pct: float = DEFAULT_WIDTH_PCT
    dte_days: int = CYCLE_DAYS


@dataclass
class DeclineIncomePlan:
    ticker: str
    close: float
    rv20_pct: float
    ret_20d_pct: float
    ret_60d_pct: float
    trend_label: str
    primary_strategy: str
    short_call_strike: float
    long_call_strike: float
    net_credit: float
    max_loss: float
    monthly_yield_pct: float
    bt_win_rate: float | None
    bt_annual: float | None
    bt_worst_cycle: float | None
    score: float
    flags: list[str] = field(default_factory=list)
    playbook: list[str] = field(default_factory=list)


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = RFR) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def strike_for_call_delta(S: float, T: float, sigma: float, target_delta: float = CALL_DELTA_SHORT, r: float = RFR) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return S * 1.05
    d1 = _norm_cdf_inv(target_delta)
    sqrtT = math.sqrt(T)
    return S * math.exp((r + 0.5 * sigma ** 2) * T - d1 * sigma * sqrtT)


def estimate_bear_call_spread(
    spot: float,
    rv_annual: float,
    *,
    vrp: float = DEFAULT_VRP,
    delta: float = CALL_DELTA_SHORT,
    width_pct: float = DEFAULT_WIDTH_PCT,
    dte_days: int = CYCLE_DAYS,
) -> tuple[float, float, float, float, float]:
    """返回 (卖出行权K短, 买入行权K长, 净权利金/股, 最大亏损/股, 月化收益率%)。"""
    if spot <= 0 or rv_annual <= 0:
        return spot * 1.05, spot * 1.10, 0.0, 0.0, 0.0
    iv = rv_annual * (1 + vrp)
    T = dte_days / TRADING_DAYS
    k_short = strike_for_call_delta(spot, T, iv, target_delta=delta)
    k_long = k_short * (1 + width_pct)
    credit = bs_call_price(spot, k_short, T, iv) - bs_call_price(spot, k_long, T, iv)
    width = k_long - k_short
    max_loss = max(0.0, width - credit)
    margin = width  # 价差宽度作保证金近似
    monthly_y = (credit / margin) * (30.0 / dte_days) * 100.0 if margin > 0 else 0.0
    return round(k_short, 2), round(k_long, 2), round(credit, 2), round(max_loss, 2), round(monthly_y, 2)


def estimate_covered_call(
    spot: float,
    rv_annual: float,
    *,
    vrp: float = DEFAULT_VRP,
    delta: float = 0.30,
    dte_days: int = CYCLE_DAYS,
) -> tuple[float, float, float]:
    """返回 (卖Call行权, 权利金/股, 月化收益率% 相对股价)。"""
    if spot <= 0 or rv_annual <= 0:
        return spot * 1.08, 0.0, 0.0
    iv = rv_annual * (1 + vrp)
    T = dte_days / TRADING_DAYS
    k = strike_for_call_delta(spot, T, iv, target_delta=delta)
    prem = bs_call_price(spot, k, T, iv)
    monthly_y = (prem / spot) * (30.0 / dte_days) * 100.0
    return round(k, 2), round(prem, 2), round(monthly_y, 2)


def classify_decline_trend(close: pd.Series) -> tuple[str, float, float]:
    """判断缓跌/横盘/仍强趋势。返回 (标签, 20日%, 60日%)。"""
    close = close.astype(float)
    if len(close) < 65:
        return "数据不足", 0.0, 0.0
    r20 = float(close.iloc[-1] / close.iloc[-21] - 1) * 100
    r60 = float(close.iloc[-1] / close.iloc[-61] - 1) * 100
    if r60 > 10:
        label = "仍偏强（慎卖 call 价差）"
    elif r60 < -25:
        label = "跌速偏快（缩仓位/加保护）"
    elif r60 <= -3 and r20 <= 5:
        label = "缓跌/磨顶（适合 Call 价差收租）"
    else:
        label = "横盘震荡"
    return label, r20, r60


def _spread_pnl_at_expiry(st: float, k_short: float, k_long: float, credit: float) -> float:
    """单股熊市认购价差到期盈亏（已收 credit）。"""
    loss = min(max(0.0, st - k_short), k_long - k_short)
    return credit - loss


def backtest_bear_call_spread(
    close: pd.Series,
    *,
    vrp: float = DEFAULT_VRP,
    delta: float = CALL_DELTA_SHORT,
    width_pct: float = DEFAULT_WIDTH_PCT,
    cycle: int = CYCLE_DAYS,
    fee_bps: float = 2.0,
) -> dict:
    """滚动开熊市认购价差，到期结算。返回胜率、年化、最差单周期等。"""
    close = close.astype(float).dropna()
    rv = realized_vol(close)
    rets: list[float] = []
    i = 25
    while i + cycle < len(close):
        S = float(close.iloc[i])
        sigma = float(rv.iloc[i])
        if not np.isfinite(sigma) or sigma <= 0:
            i += cycle
            continue
        ks, kl, credit, max_l, _ = estimate_bear_call_spread(
            S, sigma, vrp=vrp, delta=delta, width_pct=width_pct, dte_days=cycle,
        )
        ST = float(close.iloc[i + cycle])
        pnl = _spread_pnl_at_expiry(ST, ks, kl, credit)
        pnl -= (fee_bps / 10_000.0) * S * 2
        margin = kl - ks
        rets.append(pnl / margin if margin > 0 else 0.0)
        i += cycle
    return _summarize_cycles(rets, cycle)


def backtest_covered_call(
    close: pd.Series,
    *,
    vrp: float = DEFAULT_VRP,
    delta: float = 0.30,
    cycle: int = CYCLE_DAYS,
    fee_bps: float = 2.0,
) -> dict:
    """持股 + 每月卖 call：收益 = 股票涨跌 + 权利金 - call 被行权机会成本。"""
    close = close.astype(float).dropna()
    rv = realized_vol(close)
    rets: list[float] = []
    i = 25
    while i + cycle < len(close):
        S0 = float(close.iloc[i])
        S1 = float(close.iloc[i + cycle])
        sigma = float(rv.iloc[i])
        if not np.isfinite(sigma) or sigma <= 0:
            i += cycle
            continue
        iv = sigma * (1 + vrp)
        T = cycle / TRADING_DAYS
        k, prem, _ = estimate_covered_call(S0, sigma, vrp=vrp, delta=delta, dte_days=cycle)
        prem -= (fee_bps / 10_000.0) * S0
        # 到期：若 ST > K 被 call 走，收益 capped at K；否则保留股票
        if S1 > k:
            stock_pnl = k - S0
        else:
            stock_pnl = S1 - S0
        total = stock_pnl + prem
        rets.append(total / S0)
        i += cycle
    return _summarize_cycles(rets, cycle)


def trades_to_equity(
    trade_returns: list[float] | pd.Series,
    *,
    alloc_pct: float = 1.0,
    initial: float = 1.0,
    dates: pd.DatetimeIndex | pd.Series | list | None = None,
) -> tuple[pd.Series, pd.Series]:
    """将交易级收益复利合成为账户净值曲线（非「合成回撤」近似）。

    每笔交易按 alloc_pct 占用当前权益；alloc_pct=1 表示全仓 sequential margin。
    返回 (equity, trade_returns_series)。
    """
    r = pd.Series(trade_returns, dtype=float).dropna()
    if r.empty:
        idx = pd.DatetimeIndex([]) if dates is None else pd.DatetimeIndex(dates)
        return pd.Series([initial], index=idx[:1] if len(idx) else pd.DatetimeIndex([pd.Timestamp("1970-01-01")])), r
    equity_vals = [initial]
    for ret in r:
        equity_vals.append(equity_vals[-1] * (1.0 + float(ret) * alloc_pct))
    eq = pd.Series(equity_vals[1:], dtype=float)
    if dates is not None and len(dates) == len(eq):
        eq.index = pd.DatetimeIndex(dates)
    else:
        eq.index = pd.RangeIndex(len(eq))
    return eq, r


def equity_metrics_from_trades(
    trade_returns: list[float] | pd.Series,
    *,
    alloc_pct: float = 1.0,
    initial: float = 1.0,
    dates: pd.DatetimeIndex | pd.Series | list | None = None,
    cycles_per_year: float | None = None,
) -> dict:
    """基于 trades_to_equity 净值曲线计算年化、最大回撤、胜率等。"""
    from . import metrics as M

    eq, r = trades_to_equity(trade_returns, alloc_pct=alloc_pct, initial=initial, dates=dates)
    if len(eq) < 1:
        return {}
    win = float((r > 0).mean()) if len(r) else 0.0
    mdd = M.max_drawdown(eq) if len(eq) >= 2 else 0.0
    if isinstance(eq.index, pd.DatetimeIndex) and len(eq) >= 2:
        ann = M.cagr(eq)
        rets = eq.pct_change().fillna(0.0)
        sharpe = M.sharpe_ratio(rets)
    elif cycles_per_year and len(r) > 0:
        years = max(len(r) / cycles_per_year, 0.1)
        ann = float(eq.iloc[-1] ** (1 / years) - 1)
        vol = (eq.pct_change().fillna(0.0).std(ddof=0)) * np.sqrt(cycles_per_year)
        sharpe = float((eq.iloc[-1] ** (1 / years) - 1) / vol) if vol > 0 else 0.0
    else:
        years = max(len(r) / 12.0, 0.1) if len(r) else max(len(eq) / 12.0, 0.1)
        ann = float(eq.iloc[-1] ** (1 / years) - 1) if len(eq) >= 1 else 0.0
        rets = eq.pct_change().fillna(0.0)
        sharpe = M.sharpe_ratio(rets) if len(rets) > 1 else 0.0
    return {
        "累计收益率": float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) >= 1 else 0.0,
        "年化收益率": ann,
        "最大回撤": mdd,
        "夏普比率": sharpe,
        "胜率": win,
        "交易次数": float(len(r)),
        "净值曲线": eq,
        "交易收益": r,
    }


def _summarize_cycles(cycle_rets: list[float], cycle: int) -> dict:
    if not cycle_rets:
        return {}
    r = pd.Series(cycle_rets)
    cyc_yr = TRADING_DAYS / cycle
    stats = equity_metrics_from_trades(r, alloc_pct=1.0, cycles_per_year=cyc_yr)
    vol = r.std(ddof=0) * np.sqrt(cyc_yr)
    sharpe = (r.mean() * cyc_yr) / vol if vol > 0 else 0.0
    return {
        "周期数": len(r),
        "胜率": stats.get("胜率", float((r > 0).mean())),
        "年化": stats.get("年化收益率", 0.0),
        "夏普": sharpe,
        "最大回撤": stats.get("最大回撤", 0.0),
        "最差单周期": float(r.min()),
        "平均周期收益": float(r.mean()),
        "净值曲线": stats.get("净值曲线"),
    }


def build_playbook(plan: DeclineIncomePlan, owns_shares: bool = False) -> list[str]:
    steps: list[str] = []
    t = plan.ticker
    if plan.primary_strategy == "熊市认购价差":
        steps.append(
            f"1. 开 **熊市认购价差**（无需持股）：卖 {CYCLE_DAYS} 天 "
            f"Call ${plan.short_call_strike:,.0f}，同时买 Call ${plan.long_call_strike:,.0f}；"
            f"净收约 ${plan.net_credit:.2f}/股 ×100。"
        )
        steps.append(
            f"   · 最大亏损约 ${plan.max_loss:.2f}/股（价差宽度−权利金）；"
            f"股价到期低于 ${plan.short_call_strike:,.0f} 则全赚权利金。"
        )
        steps.append("   · 单票占用保证金 ≤ 总资金 4%；同时最多 3 只；财报前 1 周不开仓。")
        steps.append("   · 赚到权利金 50% 可提前平仓；若股价突破卖 Call 行权价，考虑 roll 到更高行权。")
    elif plan.primary_strategy == "备兑卖Call":
        steps.append(
            f"1. 已持 {t} → 卖 {CYCLE_DAYS} 天 Call ${plan.short_call_strike:,.0f}，"
            f"收约 ${plan.net_credit:.2f}/股。"
        )
        steps.append("   · 股价缓跌时权利金缓冲跌幅；大涨会被行权卖飞。")
    steps.append("2. 禁止：裸卖 Put（缓跌股易接飞刀）、裸卖 Call、宽跨式。")
    steps.append("3. 若 60 日跌幅 > 25% 或单日 -8%：停开新仓，等波动回落。")
    if not owns_shares:
        steps.append("4. 未持股首选 **熊市认购价差**；若被 Put 接货后再转备兑 Wheel。")
    return steps


def analyze_ticker(
    ticker: str,
    df: pd.DataFrame,
    filters: DeclineFilters | None = None,
    owns_shares: bool = False,
) -> DeclineIncomePlan | None:
    filters = filters or DeclineFilters()
    if df is None or df.empty or len(df) < 80:
        return None
    close = df["Close"].astype(float)
    vol_s = df["Volume"].astype(float)
    px = float(close.iloc[-1])
    rv = float(realized_vol(close).iloc[-1])
    if not np.isfinite(rv) or rv <= 0:
        return None
    rv_pct = rv * 100
    dollar_vol = float((close * vol_s).tail(20).mean()) / 1e6
    trend, r20, r60 = classify_decline_trend(close)

    flags: list[str] = []
    if dollar_vol < filters.min_dollar_vol_m:
        return None
    if rv_pct < filters.min_rv_pct or rv_pct > filters.max_rv_pct:
        return None
    if r60 < filters.min_ret_60d_pct or r60 > filters.max_ret_60d_pct:
        if not filters.allow_strong_uptrend:
            return None
        flags.append("⚠ 60日仍强趋势，裸卖 call 风险极高，建议等转弱或仅用极窄仓")
    if rv_pct > 80:
        flags.append("极高波动，半仓")
    if r60 < -20:
        flags.append("跌速较快，缩窄价差")
    if len(df) < 150:
        flags.append("历史短，回测仅供参考")

    ks, kl, credit, max_l, monthly_y = estimate_bear_call_spread(
        px, rv, vrp=filters.vrp, delta=filters.call_delta,
        width_pct=filters.spread_width_pct, dte_days=filters.dte_days,
    )
    cc_k, cc_prem, cc_y = estimate_covered_call(px, rv, vrp=filters.vrp, dte_days=filters.dte_days)

    bt_spread = backtest_bear_call_spread(close, vrp=filters.vrp, delta=filters.call_delta,
                                          width_pct=filters.spread_width_pct, cycle=filters.dte_days)
    bt_cc = backtest_covered_call(close, vrp=filters.vrp, cycle=filters.dte_days)

    if owns_shares and trend.startswith("缓跌"):
        primary = "备兑卖Call"
        short_k, net_c, monthly_y = cc_k, cc_prem, cc_y
        long_k = 0.0
        max_loss = px  # 股票本身风险
        bt = bt_cc
    else:
        primary = "熊市认购价差"
        short_k, long_k, net_c, max_l = ks, kl, credit, max_l
        bt = bt_spread

    # 得分：月化收益 × 回测胜率 × 流动性；缓跌标签加分
    if "偏强" in trend:
        flags.append("当前偏强趋势，Call 价差易被击穿")
        trend_bonus = 0.5
    else:
        trend_bonus = 1.2 if "缓跌" in trend else 1.0
    win = bt.get("胜率", 0.7)
    score = monthly_y * win * min(1.0, dollar_vol / 1000.0) * trend_bonus

    plan = DeclineIncomePlan(
        ticker=ticker, close=px, rv20_pct=round(rv_pct, 1),
        ret_20d_pct=round(r20, 1), ret_60d_pct=round(r60, 1),
        trend_label=trend, primary_strategy=primary,
        short_call_strike=short_k, long_call_strike=long_k,
        net_credit=net_c, max_loss=max_loss if owns_shares else max_l,
        monthly_yield_pct=monthly_y,
        bt_win_rate=bt.get("胜率"), bt_annual=bt.get("年化"), bt_worst_cycle=bt.get("最差单周期"),
        score=round(score, 2), flags=flags,
    )
    plan.playbook = build_playbook(plan, owns_shares=owns_shares)
    return plan


def scan_decline_income(
    tickers: list[str],
    start: str,
    end: str,
    filters: DeclineFilters | None = None,
    owns_shares: bool = False,
) -> pd.DataFrame:
    filters = filters or DeclineFilters()
    syms = [t.strip().upper() for t in tickers if t and str(t).strip()]
    if not syms:
        return pd.DataFrame()
    batch = fetch_history_batch(syms, start=start, end=end)
    rows: list[dict] = []
    for t, df in batch.items():
        p = analyze_ticker(t, df, filters, owns_shares=owns_shares)
        if p is None:
            continue
        rows.append({
            "代码": p.ticker,
            "最新价": p.close,
            "趋势": p.trend_label,
            "60日%": p.ret_60d_pct,
            "RV20%": p.rv20_pct,
            "推荐策略": p.primary_strategy,
            "卖Call": p.short_call_strike,
            "买Call": p.long_call_strike if p.long_call_strike else "-",
            "净权利金": p.net_credit,
            "最大亏损": p.max_loss,
            "月化收益%": p.monthly_yield_pct,
            "回测胜率": p.bt_win_rate,
            "回测年化": p.bt_annual,
            "最差周期": p.bt_worst_cycle,
            "综合分": p.score,
            "提示": "；".join(p.flags) if p.flags else "",
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("综合分", ascending=False).reset_index(drop=True)


def analyze_single_for_ui(
    ticker: str, start: str, end: str, owns_shares: bool = False,
    filters: DeclineFilters | None = None,
) -> DeclineIncomePlan | None:
    f = replace(filters or DeclineFilters(), allow_strong_uptrend=True)
    df = fetch_history(ticker, start=start, end=end)
    return analyze_ticker(ticker, df, filters=f, owns_shares=owns_shares)


# ======================================================================
# 「稳定收租」核心：现金担保认沽 CSP（顺势卖下方 Put）
# 经 SNDK/WDC/MU 横向回测：在强势上涨 + 高 IV 股上，CSP 的胜率最高、
# 月度收益波动最小、回撤最浅；逆势卖 call 价差 / 铁鹰是灾难（回撤 -90%+）。
# 加「50 日均线过滤 + 50% 止盈」后，最差单笔从 -50% 压到 -14%，
# 合成回撤从 -29% 压到 -3%，信息比翻数倍。
# ======================================================================

def _summarize_csp(rors: list[float], *, alloc_pct: float = 0.10) -> dict:
    """CSP 重叠回测汇总：用 alloc_pct 占用权益复利（默认 10%/笔，非合成回撤 artifact）。"""
    if not rors:
        return {}
    r = pd.Series(rors)
    cyc_per_year = TRADING_DAYS / CSP_HOLD_TD
    stats = equity_metrics_from_trades(r, alloc_pct=alloc_pct, cycles_per_year=cyc_per_year)
    std = float(r.std(ddof=0))
    eq = stats.get("净值曲线", pd.Series(dtype=float))
    synth_legacy = float((eq / eq.cummax() - 1).min()) if len(eq) > 1 else 0.0
    return {
        "交易数": int(len(r)),
        "胜率": stats.get("胜率", float((r > 0).mean())),
        "平均ROR": float(r.mean()),
        "标准差": std,
        "信息比": float(r.mean() / std) if std > 0 else 0.0,
        "最差单笔": float(r.min()),
        "年化": stats.get("年化收益率", float((1 + r.mean()) ** cyc_per_year - 1)),
        "最大回撤": stats.get("最大回撤", 0.0),
        "夏普": stats.get("夏普比率", 0.0),
        "合成回撤": synth_legacy,
        "净值曲线": eq,
    }


def backtest_csp_income(
    close: pd.Series,
    *,
    delta: float = CSP_DELTA,
    vrp: float = DEFAULT_VRP,
    ma_window: int = CSP_MA_WINDOW,
    take_profit: float = CSP_TAKE_PROFIT,
) -> dict:
    """重叠回测现金担保认沽（顺势卖 put + 均线过滤 + 止盈）。ROR 以行权价（担保金）为分母。"""
    close = close.astype(float).dropna()
    rv = realized_vol(close)
    ma = close.rolling(ma_window).mean() if ma_window else None
    T = CSP_DTE_CAL / TRADING_DAYS
    rors: list[float] = []
    i = max(25, ma_window)
    while i + CSP_HOLD_TD < len(close):
        S = float(close.iloc[i]); sigma = float(rv.iloc[i])
        if not np.isfinite(sigma) or sigma <= 0:
            i += CSP_STEP_TD; continue
        if ma is not None and not (S > float(ma.iloc[i])):
            i += CSP_STEP_TD; continue
        iv = sigma * (1 + vrp)
        K = strike_for_put_delta(S, T, iv, target_delta=delta)
        credit = bs_put_price(S, K, T, iv)
        if K <= 0:
            i += CSP_STEP_TD; continue
        exited = False
        if take_profit > 0:
            path = close.iloc[i:i + CSP_HOLD_TD + 1]
            for j in range(1, len(path)):
                Sj = float(path.iloc[j])
                remain = max(0.0, 1 - j / CSP_HOLD_TD)
                mark = max(0.0, K - Sj) + credit * remain * 0.5   # 剩余时间价值近似
                if credit - mark >= take_profit * credit:
                    rors.append((credit - mark) / K); exited = True; break
        if not exited:
            ST = float(close.iloc[i + CSP_HOLD_TD])
            rors.append((credit - max(0.0, K - ST)) / K)
        i += CSP_STEP_TD
    return _summarize_csp(rors, alloc_pct=0.10)


def compare_income_strategies(close: pd.Series, *, vrp: float = DEFAULT_VRP) -> pd.DataFrame:
    """横向比较 5 种期权卖方策略，用「稳定性」指标排序（信息比降序）。"""
    close = close.astype(float).dropna()
    rv = realized_vol(close)
    T = CSP_DTE_CAL / TRADING_DAYS
    buckets: dict[str, list[float]] = {k: [] for k in ["CSP", "PCS", "CC", "BCS", "IC"]}
    i = 25
    while i + CSP_HOLD_TD < len(close):
        S = float(close.iloc[i]); ST = float(close.iloc[i + CSP_HOLD_TD]); sigma = float(rv.iloc[i])
        if not np.isfinite(sigma) or sigma <= 0:
            i += CSP_STEP_TD; continue
        iv = sigma * (1 + vrp)
        # CSP（顺势卖 0.20 put，现金担保）
        kp = strike_for_put_delta(S, T, iv, 0.20); cp = bs_put_price(S, kp, T, iv)
        if kp > 0:
            buckets["CSP"].append((cp - max(0.0, kp - ST)) / kp)
        # PCS 认沽信用价差（卖0.25put 买0.10put）
        ks = strike_for_put_delta(S, T, iv, 0.25); kl = strike_for_put_delta(S, T, iv, 0.10)
        credit = bs_put_price(S, ks, T, iv) - bs_put_price(S, kl, T, iv)
        wp = ks - kl
        if wp > 0:
            buckets["PCS"].append((credit - (max(0.0, ks - ST) - max(0.0, kl - ST))) / wp)
        # CC 备兑（持股卖 0.30 call）
        kc = strike_for_call_delta(S, T, iv, 0.30); cc = bs_call_price(S, kc, T, iv)
        stock = (kc - S) if ST > kc else (ST - S)
        buckets["CC"].append((stock + cc) / S)
        # BCS 熊市认购价差（逆势 卖0.25call 买0.10call）
        kcs = strike_for_call_delta(S, T, iv, 0.25); kcl = strike_for_call_delta(S, T, iv, 0.10)
        ccredit = bs_call_price(S, kcs, T, iv) - bs_call_price(S, kcl, T, iv)
        cw = kcl - kcs
        if cw > 0:
            buckets["BCS"].append((ccredit - (max(0.0, ST - kcs) - max(0.0, ST - kcl))) / cw)
        # IC 铁鹰
        if wp > 0 and cw > 0:
            buckets["IC"].append(
                ((credit + ccredit) - (max(0.0, ks - ST) - max(0.0, kl - ST)) - (max(0.0, ST - kcs) - max(0.0, ST - kcl)))
                / max(wp, cw)
            )
        i += CSP_STEP_TD
    names = {
        "CSP": "现金担保认沽(顺势·首选)", "PCS": "认沽信用价差(顺势·高效)",
        "CC": "备兑Call(需持股)", "BCS": "认购信用价差(逆势·危险)", "IC": "铁鹰双卖(危险)",
    }
    rows = [{"策略": names[k], **_summarize_csp(v)} for k, v in buckets.items()]
    df = pd.DataFrame(rows)
    if "信息比" in df.columns:
        df = df.sort_values("信息比", ascending=False).reset_index(drop=True)
    return df


@dataclass
class CspIncomePlan:
    ticker: str
    close: float
    rv_pct: float
    ma50: float
    above_ma: bool
    can_open: bool
    put_strike: float
    premium: float
    capital_per_contract: float   # 担保金 = 行权价 ×100
    monthly_yield_pct: float
    take_profit_price: float      # 止盈目标权利金
    breakeven: float
    bt_win_rate: float | None
    bt_info_ratio: float | None
    bt_worst: float | None
    bt_max_dd: float | None
    bt_annual: float | None
    flags: list[str] = field(default_factory=list)
    playbook: list[str] = field(default_factory=list)


def csp_income_plan(
    ticker: str,
    df: pd.DataFrame,
    *,
    delta: float = CSP_DELTA,
    vrp: float = DEFAULT_VRP,
    ma_window: int = CSP_MA_WINDOW,
    dte_days: int = CSP_DTE_CAL,
) -> CspIncomePlan | None:
    """生成以 CSP 为核心的「稳定收租」当前每单方案 + 回测稳定性指标。"""
    if df is None or df.empty or len(df) < ma_window + 30:
        return None
    close = df["Close"].astype(float).dropna()
    S = float(close.iloc[-1])
    sigma = float(realized_vol(close).iloc[-1])
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    ma50 = float(close.rolling(ma_window).mean().iloc[-1])
    above = S > ma50
    iv = sigma * (1 + vrp)
    T = dte_days / TRADING_DAYS
    K = strike_for_put_delta(S, T, iv, target_delta=delta)
    prem = bs_put_price(S, K, T, iv)
    capital = K * 100
    monthly_y = (prem / K) * (30.0 / dte_days) * 100.0 if K > 0 else 0.0
    bt = backtest_csp_income(close, delta=delta, vrp=vrp, ma_window=ma_window)

    flags: list[str] = []
    if not above:
        flags.append("⚠ 股价跌破 50 日均线 → 暂停开新仓（趋势过滤未通过）")
    if sigma * 100 > 80:
        flags.append("极高波动：单票担保金占比减半")
    if len(close) < 200:
        flags.append("历史样本短，回测仅供参考（建议参考 WDC/MU 长样本）")

    plan = CspIncomePlan(
        ticker=ticker, close=round(S, 2), rv_pct=round(sigma * 100, 1),
        ma50=round(ma50, 2), above_ma=above, can_open=above,
        put_strike=round(K, 2), premium=round(prem, 2),
        capital_per_contract=round(capital, 0),
        monthly_yield_pct=round(monthly_y, 2),
        take_profit_price=round(prem * (1 - CSP_TAKE_PROFIT), 2),
        breakeven=round(K - prem, 2),
        bt_win_rate=bt.get("胜率"), bt_info_ratio=bt.get("信息比"),
        bt_worst=bt.get("最差单笔"), bt_max_dd=bt.get("最大回撤", bt.get("合成回撤")),
        bt_annual=bt.get("年化"), flags=flags,
    )
    plan.playbook = _csp_playbook(plan, delta=delta, dte_days=dte_days)
    return plan


def _csp_playbook(plan: CspIncomePlan, *, delta: float, dte_days: int) -> list[str]:
    t = plan.ticker
    steps = [
        f"1. 趋势确认：仅当 {t} 收盘价 **站上 50 日均线**（现 ${plan.ma50:,.2f}）才开仓；"
        f"当前{'✅ 满足' if plan.above_ma else '❌ 不满足，先观望'}。",
        f"2. 卖出 **{dte_days} 天、Delta≈{delta:.2f} 的现金担保认沽**：行权价 ${plan.put_strike:,.2f}，"
        f"收权利金约 ${plan.premium:.2f}/股 ×100 = ${plan.premium*100:,.0f}/张。",
        f"3. 备好担保金 ${plan.capital_per_contract:,.0f}/张；单票占用 ≤ 总资金 10%（高波股建议 ≤5%）。",
        f"4. **止盈（核心）**：权利金跌到 ${plan.take_profit_price:.2f}（赚 50%）立即买回平仓，"
        f"再开新一轮 —— 这是把最差单笔从 −50% 压到 −14% 的关键。",
        f"5. 盈亏平衡价 ${plan.breakeven:,.2f}；若被指派接货，转 **备兑卖 Call（Wheel）** 继续收租。",
        "6. 财报前 1 周不开新仓；跌破 50 日均线立即停止开新仓（已开的按止盈/到期处理）。",
        "7. 禁止：裸卖 Call、逆势卖 Call 价差、铁鹰（回测在这类强势股上回撤 −90%+）。",
    ]
    return steps


# ======================================================================
# 周 PUT 信用价差「喝汤」：小资金参与「大量 PUT 归零」的卖方游戏
# ======================================================================

WEEKLY_DTE = 7
WEEKLY_SOUP_DELTA = 0.10
WEEKLY_SOUP_WIDTH = 25.0
WEEKLY_SOUP_TAKE_PROFIT = 0.5


def backtest_weekly_put_spread(
    close: pd.Series,
    *,
    short_delta: float = WEEKLY_SOUP_DELTA,
    width: float = WEEKLY_SOUP_WIDTH,
    dte_days: int = WEEKLY_DTE,
    vrp: float = DEFAULT_VRP,
    ma_window: int = CSP_MA_WINDOW,
    take_profit: float = WEEKLY_SOUP_TAKE_PROFIT,
    step_td: int = 5,
) -> dict:
    """周 PUT 信用价差回测（MA50 过滤 + 止盈）。"""
    close = close.astype(float).dropna()
    rv = realized_vol(close)
    ma = close.rolling(ma_window).mean() if ma_window else None
    T = dte_days / TRADING_DAYS
    hold = max(1, int(dte_days * TRADING_DAYS / 7))  # 7 日历天 ≈ 5 交易日
    rors: list[float] = []
    i = max(25, ma_window)
    while i + hold < len(close):
        S = float(close.iloc[i])
        sigma = float(rv.iloc[i])
        if not np.isfinite(sigma) or sigma <= 0:
            i += step_td
            continue
        if ma is not None and not (S > float(ma.iloc[i])):
            i += step_td
            continue
        iv = sigma * (1 + vrp)
        ks, kl, credit, margin, _, _ = estimate_put_credit_spread(
            S, sigma, short_delta=short_delta, width=width, dte_days=dte_days, vrp=vrp,
        )
        if margin <= 0 or width <= 0:
            i += step_td
            continue
        exited = False
        if take_profit > 0:
            path = close.iloc[i:i + hold + 1]
            for j in range(1, len(path)):
                Sj = float(path.iloc[j])
                remain = max(0.0, 1 - j / hold)
                short_loss = max(0.0, ks - Sj) - max(0.0, kl - Sj)
                mark = short_loss + credit * remain * 0.5
                if credit - mark >= take_profit * credit:
                    rors.append((credit - mark) / width)
                    exited = True
                    break
        if not exited:
            ST = float(close.iloc[i + hold])
            pnl = credit - (max(0.0, ks - ST) - max(0.0, kl - ST))
            rors.append(pnl / width)
        i += step_td
    cyc_yr = TRADING_DAYS / hold
    stats = equity_metrics_from_trades(rors, alloc_pct=0.25, cycles_per_year=cyc_yr) if rors else {}
    if not rors:
        return {}
    r = pd.Series(rors)
    return {
        "交易数": len(r),
        "胜率": stats.get("胜率", float((r > 0).mean())),
        "年化": stats.get("年化收益率", 0.0),
        "最大回撤": stats.get("最大回撤", 0.0),
        "夏普": stats.get("夏普比率", 0.0),
        "最差单笔": float(r.min()),
        "净值曲线": stats.get("净值曲线"),
    }


def put_expire_otm_prob(S: float, K: float, T: float, iv: float, r: float = RFR) -> float:
    """Put 到期归零概率 ≈ P(ST > K) = N(d2)。"""
    if T <= 0 or iv <= 0 or S <= 0 or K <= 0:
        return 1.0 if S > K else 0.0
    d2 = (math.log(S / K) + (r - 0.5 * iv ** 2) * T) / (iv * math.sqrt(T))
    return float(_norm_cdf(d2))


def estimate_put_credit_spread(
    spot: float,
    rv_annual: float,
    *,
    short_delta: float = WEEKLY_SOUP_DELTA,
    width: float = WEEKLY_SOUP_WIDTH,
    dte_days: int = WEEKLY_DTE,
    vrp: float = DEFAULT_VRP,
) -> tuple[float, float, float, float, float, float]:
    """返回 (卖Put K, 买Put K, 净权利金/股, 保证金/张, 最大亏损/张, 归零概率)。"""
    if spot <= 0 or rv_annual <= 0 or width <= 0:
        return spot * 0.9, spot * 0.85, 0.0, 0.0, 0.0, 0.0
    iv = rv_annual * (1 + vrp)
    T = dte_days / TRADING_DAYS
    ks = strike_for_put_delta(spot, T, iv, target_delta=short_delta)
    kl = max(0.01, ks - width)
    credit = bs_put_price(spot, ks, T, iv) - bs_put_price(spot, kl, T, iv)
    margin = width * 100
    max_loss = max(0.0, (width - credit) * 100)
    zp = put_expire_otm_prob(spot, ks, T, iv)
    return round(ks, 2), round(kl, 2), round(credit, 2), round(margin, 0), round(max_loss, 0), round(zp, 4)


@dataclass
class WeeklySoupPlan:
    ticker: str
    close: float
    rv_pct: float
    iv_pct: float
    ma50: float
    above_ma: bool
    can_open: bool
    dte_days: int
    short_delta: float
    width: float
    short_strike: float
    long_strike: float
    credit_per_share: float
    credit_per_contract: float
    margin_per_contract: float
    max_loss_per_contract: float
    take_profit_price: float
    zero_prob: float
    weekly_roi_pct: float
    otm_pct: float
    one_std_move_pct: float
    account_size: float
    max_contracts: int
    weekly_profit_if_zero: float
    weekly_loss_if_max: float
    # 偏斜铁鹰（双开）：在卖 Put 价差上叠一条卖得很远的 Call 价差（低 Delta）
    iron_condor: bool = False
    call_short_strike: float = 0.0
    call_long_strike: float = 0.0
    call_delta: float = 0.0
    call_credit_per_contract: float = 0.0
    call_otm_pct: float = 0.0
    total_credit_per_contract: float = 0.0
    combined_roi_pct: float = 0.0
    range_prob: float = 0.0
    trend_strong: bool = False
    flags: list[str] = field(default_factory=list)
    playbook: list[str] = field(default_factory=list)


def weekly_put_soup_plan(
    ticker: str,
    df: pd.DataFrame,
    *,
    account_size: float = 10_000.0,
    short_delta: float = WEEKLY_SOUP_DELTA,
    width: float = WEEKLY_SOUP_WIDTH,
    dte_days: int = WEEKLY_DTE,
    vrp: float = DEFAULT_VRP,
    ma_window: int = CSP_MA_WINDOW,
    take_profit: float = WEEKLY_SOUP_TAKE_PROFIT,
    max_margin_pct: float = 0.25,
    add_call: bool = False,
    call_delta: float = 0.05,
    call_width: float | None = None,
) -> WeeklySoupPlan | None:
    """生成本周 PUT 信用价差「喝汤」方案。

    add_call=True 时叠一条**卖得很远的 Call 价差**（低 Delta，默认 0.05）做成
    偏斜铁鹰：同保证金多收一份看涨腿权利金，赚"既不大跌也不爆涨"。
    """
    if df is None or df.empty or len(df) < ma_window + 30:
        return None
    close = df["Close"].astype(float).dropna()
    S = float(close.iloc[-1])
    sigma = float(realized_vol(close).iloc[-1])
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    ma50 = float(close.rolling(ma_window).mean().iloc[-1])
    above = S > ma50
    iv = sigma * (1 + vrp)
    T = dte_days / TRADING_DAYS
    r60 = float(close.iloc[-1] / close.iloc[-61] - 1) if len(close) > 61 else 0.0
    trend_strong = r60 > 0.10  # 近 60 日涨超 10% = 强上行趋势
    ks, kl, credit, margin, max_loss, zp = estimate_put_credit_spread(
        S, sigma, short_delta=short_delta, width=width, dte_days=dte_days, vrp=vrp,
    )
    if margin <= 0:
        return None

    # 偏斜铁鹰：远端卖 Call 价差
    ic = add_call
    ksc = klc = call_credit = call_otm = total_credit = comb_roi = range_p = 0.0
    eff_call_delta = call_delta
    if ic:
        cwidth = call_width if call_width else width
        if trend_strong:
            eff_call_delta = min(call_delta, 0.05)  # 强趋势把 Call 压得更远
        ksc = strike_for_call_delta(S, T, iv, target_delta=eff_call_delta)
        klc = ksc + cwidth
        call_credit = (bs_call_price(S, ksc, T, iv) - bs_call_price(S, klc, T, iv)) * 100
        call_otm = (ksc / S - 1) * 100
        zp_call = put_expire_otm_prob(S, ksc, T, iv)  # P(ST>Ksc)
        range_p = max(0.0, zp - zp_call)              # P(Ksp<ST<Ksc)
        total_credit = credit * 100 + call_credit
        ic_margin = max(width, cwidth) * 100          # 铁鹰只按单边收保证金
        comb_roi = total_credit / ic_margin * 100 if ic_margin else 0.0

    max_contracts = max(0, int(account_size * max_margin_pct // margin))
    flags: list[str] = []
    if not above:
        flags.append("⚠ 股价跌破 50 日均线 → 本周不喝汤，先观望")
    if sigma * 100 > 80:
        flags.append("极高波动：建议只开 1 张，Delta≤0.10")
    if max_contracts < 1:
        flags.append(f"⚠ 账户 ${account_size:,.0f} 按 {max_margin_pct:.0%} 上限不够 1 张")
    if len(close) < 200:
        flags.append("历史短，归零概率为模型估算，实盘以券商 IV 为准")
    if ic and trend_strong:
        flags.append(f"强上行趋势 → Call 腿已压到 Delta {eff_call_delta:.02f}（约 +{call_otm:.0f}%），必要时向上 roll")

    n_use = max(1, max_contracts) if max_contracts else 1
    plan = WeeklySoupPlan(
        ticker=ticker, close=round(S, 2), rv_pct=round(sigma * 100, 1),
        iv_pct=round(iv * 100, 1), ma50=round(ma50, 2), above_ma=above,
        can_open=above and max_contracts >= 1,
        dte_days=dte_days, short_delta=short_delta, width=width,
        short_strike=ks, long_strike=kl, credit_per_share=round(credit, 2),
        credit_per_contract=round(credit * 100, 0),
        margin_per_contract=margin, max_loss_per_contract=max_loss,
        take_profit_price=round(credit * (1 - take_profit), 2),
        zero_prob=zp, weekly_roi_pct=round(credit * 100 / margin * 100, 1) if margin else 0.0,
        otm_pct=round((1 - ks / S) * 100, 1),
        one_std_move_pct=round(iv * math.sqrt(T) * 100, 1),
        account_size=account_size, max_contracts=max_contracts,
        weekly_profit_if_zero=round((credit * 100 + call_credit) * n_use, 0),
        weekly_loss_if_max=round(max_loss * n_use, 0),
        iron_condor=ic,
        call_short_strike=round(ksc, 2), call_long_strike=round(klc, 2),
        call_delta=round(eff_call_delta, 3) if ic else 0.0,
        call_credit_per_contract=round(call_credit, 0),
        call_otm_pct=round(call_otm, 1),
        total_credit_per_contract=round(total_credit, 0),
        combined_roi_pct=round(comb_roi, 1),
        range_prob=round(range_p, 4),
        trend_strong=trend_strong,
        flags=flags,
    )
    plan.playbook = _weekly_soup_playbook(plan, take_profit=take_profit)
    return plan


def scan_weekly_soup_configs(
    spot: float,
    rv_annual: float,
    *,
    account_size: float = 10_000.0,
    dte_days: int = WEEKLY_DTE,
    vrp: float = DEFAULT_VRP,
    max_margin_pct: float = 0.25,
) -> pd.DataFrame:
    """扫描 Delta × 价差宽度组合。"""
    rows: list[dict] = []
    for d in [0.10, 0.15, 0.20]:
        for w in [25.0, 50.0, 100.0]:
            ks, kl, credit, margin, max_loss, zp = estimate_put_credit_spread(
                spot, rv_annual, short_delta=d, width=w, dte_days=dte_days, vrp=vrp,
            )
            n = max(0, int(account_size * max_margin_pct // margin)) if margin > 0 else 0
            rows.append({
                "Delta": d, "价差宽$": w,
                "卖Put": ks, "买Put": kl,
                "收租/张": round(credit * 100, 0),
                "保证金": margin, "最大亏": max_loss,
                "归零概率": zp, "周ROI%": round(credit * 100 / margin * 100, 1) if margin else 0,
                "可开张数": n,
            })
    return pd.DataFrame(rows).sort_values("归零概率", ascending=False).reset_index(drop=True)


def _weekly_soup_playbook(plan: WeeklySoupPlan, *, take_profit: float) -> list[str]:
    t = plan.ticker
    n = max(1, plan.max_contracts) if plan.max_contracts else 1
    steps = [
        f"1. **本周检查**：{t} 现价 ${plan.close:,.2f}，50日均线 ${plan.ma50:,.2f} → "
        f"{'✅ 可开' if plan.can_open else '❌ 暂停'}；财报周不开。",
        f"2. **组合单** Put Credit Spread（{plan.dte_days}天）："
        f" SELL Put ${plan.short_strike:,.0f} + BUY Put ${plan.long_strike:,.0f}（宽 ${plan.width:.0f}）。",
        f"3. 净收 ≥ ${plan.credit_per_share:.2f}/股 = **${plan.credit_per_contract:,.0f}/张**；"
        f"保证金 ${plan.margin_per_contract:,.0f}/张。",
    ]
    if plan.iron_condor and plan.call_short_strike > 0:
        steps.append(
            f"3b. **双开·远端 Call 价差**（Delta {plan.call_delta:.02f}，约 +{plan.call_otm_pct:.0f}%）："
            f" SELL Call ${plan.call_short_strike:,.0f} + BUY Call ${plan.call_long_strike:,.0f}"
            f"，多收 **${plan.call_credit_per_contract:,.0f}/张**。"
        )
        steps.append(
            f"3c. 合计收 **${plan.total_credit_per_contract:,.0f}/张**，保证金不变 → "
            f"周ROI **{plan.combined_roi_pct:.1f}%**；到期落在 "
            f"[${plan.short_strike:,.0f}, ${plan.call_short_strike:,.0f}] 概率 **{plan.range_prob:.0%}**。"
        )
    steps += [
        f"4. 账户 ${plan.account_size:,.0f} 建议最多 **{plan.max_contracts or 1} 张**（单笔保证金 ≤ 账户 25%）。",
        f"5. **止盈**：总权利金赚 {take_profit:.0%} 即整体平仓；顺的一周 +${(plan.total_credit_per_contract or plan.credit_per_contract) * n:,.0f}。",
        f"6. 归零概率 **{plan.zero_prob:.0%}**（7日需跌 {plan.otm_pct:.0f}%；1σ波动 ±{plan.one_std_move_pct:.0f}%）。",
        f"7. 最坏 -${plan.weekly_loss_if_max:,.0f}（{n}张）→ 只开 1 张、Put Delta 0.10。",
    ]
    if plan.iron_condor:
        steps.append("8. **强趋势看涨腿是逆势腿**：被逼近就把 Call 向上 roll，别硬扛；财报周整体不开。")
    return steps
